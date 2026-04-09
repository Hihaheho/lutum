use std::{io::ErrorKind, path::Path, sync::Arc};

use serde::{Deserialize, Serialize};
use sqlite_agent::{DatabaseListEntry, DbRegistry, SqliteDb};

pub const DEFAULT_REGISTRY_PATH: &str = "sqlite-agent-registry.json";

const REGISTRY_SNAPSHOT_VERSION: u32 = 1;
const MAIN_DB_ID: &str = "main";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct RegistrySnapshot {
    version: u32,
    databases: Vec<RegistrySnapshotEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct RegistrySnapshotEntry {
    db_id: String,
    path: String,
}

impl From<DatabaseListEntry> for RegistrySnapshotEntry {
    fn from(entry: DatabaseListEntry) -> Self {
        Self {
            db_id: entry.db_id,
            path: entry.path,
        }
    }
}

pub fn load_registry_snapshot(registry: &DbRegistry, path: Option<&Path>) -> Vec<String> {
    let Some(path) = path else {
        return vec![];
    };

    let json = match std::fs::read_to_string(path) {
        Ok(json) => json,
        Err(error) if error.kind() == ErrorKind::NotFound => return vec![],
        Err(error) => {
            let message = format!(
                "failed to read registry snapshot from {}: {error}",
                path.display()
            );
            tracing::warn!("{message}");
            return vec![message];
        }
    };

    let snapshot = match serde_json::from_str::<RegistrySnapshot>(&json) {
        Ok(snapshot) => snapshot,
        Err(error) => {
            let message = format!(
                "failed to parse registry snapshot from {}: {error}",
                path.display()
            );
            tracing::warn!("{message}");
            return vec![message];
        }
    };

    if snapshot.version != REGISTRY_SNAPSHOT_VERSION {
        let message = format!(
            "unsupported registry snapshot version {} in {}",
            snapshot.version,
            path.display(),
        );
        tracing::warn!("{message}");
        return vec![message];
    }

    let mut warnings = Vec::new();
    for entry in snapshot.databases {
        if entry.db_id == MAIN_DB_ID {
            push_warning(
                &mut warnings,
                format!(
                    "ignoring `{MAIN_DB_ID}` entry found in registry snapshot {}",
                    path.display(),
                ),
            );
            continue;
        }

        let db_path = Path::new(&entry.path);
        if !db_path.exists() {
            push_warning(
                &mut warnings,
                format!(
                    "skipping missing database `{}` from registry snapshot {}: {} does not exist",
                    entry.db_id,
                    path.display(),
                    db_path.display(),
                ),
            );
            continue;
        }

        match SqliteDb::open(db_path) {
            Ok(db) => {
                if !registry.register(entry.db_id.clone(), Arc::new(db), entry.path.clone()) {
                    push_warning(
                        &mut warnings,
                        format!(
                            "skipping duplicate db_id `{}` from registry snapshot {}",
                            entry.db_id,
                            path.display(),
                        ),
                    );
                }
            }
            Err(error) => {
                push_warning(
                    &mut warnings,
                    format!(
                        "failed to reopen database `{}` from registry snapshot {}: {error}",
                        entry.db_id,
                        path.display(),
                    ),
                );
            }
        }
    }

    warnings
}

pub fn save_registry_snapshot(registry: &DbRegistry, path: Option<&Path>) -> Result<(), String> {
    let Some(path) = path else {
        return Ok(());
    };

    let snapshot = RegistrySnapshot {
        version: REGISTRY_SNAPSHOT_VERSION,
        databases: registry
            .list()
            .databases
            .into_iter()
            .filter(|entry| entry.db_id != MAIN_DB_ID)
            .map(RegistrySnapshotEntry::from)
            .collect(),
    };

    let json = serde_json::to_string_pretty(&snapshot).map_err(|error| {
        format!(
            "failed to serialize registry snapshot for {}: {error}",
            path.display(),
        )
    })?;

    std::fs::write(path, json).map_err(|error| {
        format!(
            "failed to write registry snapshot to {}: {error}",
            path.display(),
        )
    })
}

fn push_warning(warnings: &mut Vec<String>, message: String) {
    tracing::warn!("{message}");
    warnings.push(message);
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::{Path, PathBuf},
        sync::Arc,
        time::{SystemTime, UNIX_EPOCH},
    };

    use super::*;

    fn unique_temp_dir(label: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        path.push(format!(
            "sqlite-agent-tui-{label}-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn create_db(path: &Path) {
        SqliteDb::open(path).unwrap();
    }

    fn registry_with_main(main_path: &Path) -> DbRegistry {
        create_db(main_path);
        let registry = DbRegistry::new();
        let main_db = Arc::new(SqliteDb::open(main_path).unwrap());
        assert!(registry.register(MAIN_DB_ID, main_db, main_path.to_string_lossy().to_string()));
        registry
    }

    #[test]
    fn snapshot_round_trip_restores_non_main_databases() {
        let temp_dir = unique_temp_dir("round-trip");
        let main_path = temp_dir.join("main.sqlite");
        let analytics_path = temp_dir.join("analytics.sqlite");
        let reporting_path = temp_dir.join("reporting.sqlite");
        let snapshot_path = temp_dir.join("registry.json");

        let registry = registry_with_main(&main_path);
        registry.create("analytics", &analytics_path).unwrap();
        registry.create("reporting", &reporting_path).unwrap();

        save_registry_snapshot(&registry, Some(&snapshot_path)).unwrap();

        let json = fs::read_to_string(&snapshot_path).unwrap();
        let snapshot: RegistrySnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(snapshot.version, REGISTRY_SNAPSHOT_VERSION);
        assert_eq!(
            snapshot.databases,
            vec![
                RegistrySnapshotEntry {
                    db_id: "analytics".to_string(),
                    path: analytics_path.to_string_lossy().to_string(),
                },
                RegistrySnapshotEntry {
                    db_id: "reporting".to_string(),
                    path: reporting_path.to_string_lossy().to_string(),
                },
            ]
        );

        let restored = registry_with_main(&main_path);
        let warnings = load_registry_snapshot(&restored, Some(&snapshot_path));
        assert!(warnings.is_empty());

        let databases = restored.list().databases;
        assert_eq!(databases.len(), 3);
        assert_eq!(databases[0].db_id, "analytics");
        assert_eq!(databases[1].db_id, MAIN_DB_ID);
        assert_eq!(databases[2].db_id, "reporting");

        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn missing_snapshot_is_ignored() {
        let temp_dir = unique_temp_dir("missing");
        let main_path = temp_dir.join("main.sqlite");
        let snapshot_path = temp_dir.join("missing.json");

        let registry = registry_with_main(&main_path);
        let warnings = load_registry_snapshot(&registry, Some(&snapshot_path));

        assert!(warnings.is_empty());
        assert_eq!(registry.list().databases.len(), 1);

        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn malformed_snapshot_warns_and_continues() {
        let temp_dir = unique_temp_dir("malformed");
        let main_path = temp_dir.join("main.sqlite");
        let snapshot_path = temp_dir.join("registry.json");

        fs::write(&snapshot_path, "{not valid json").unwrap();

        let registry = registry_with_main(&main_path);
        let warnings = load_registry_snapshot(&registry, Some(&snapshot_path));

        assert_eq!(warnings.len(), 1);
        assert_eq!(registry.list().databases.len(), 1);

        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn unreadable_database_path_warns_and_continues() {
        let temp_dir = unique_temp_dir("missing-db");
        let main_path = temp_dir.join("main.sqlite");
        let snapshot_path = temp_dir.join("registry.json");
        let missing_path = temp_dir.join("missing-dir").join("analytics.sqlite");

        let snapshot = RegistrySnapshot {
            version: REGISTRY_SNAPSHOT_VERSION,
            databases: vec![RegistrySnapshotEntry {
                db_id: "analytics".to_string(),
                path: missing_path.to_string_lossy().to_string(),
            }],
        };
        fs::write(&snapshot_path, serde_json::to_string(&snapshot).unwrap()).unwrap();

        let registry = registry_with_main(&main_path);
        let warnings = load_registry_snapshot(&registry, Some(&snapshot_path));

        assert_eq!(warnings.len(), 1);
        assert_eq!(registry.list().databases.len(), 1);

        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn missing_database_file_in_existing_directory_warns_and_is_not_recreated() {
        let temp_dir = unique_temp_dir("missing-file");
        let main_path = temp_dir.join("main.sqlite");
        let snapshot_path = temp_dir.join("registry.json");
        let db_dir = temp_dir.join("dbs");
        let missing_path = db_dir.join("analytics.sqlite");

        fs::create_dir_all(&db_dir).unwrap();

        let snapshot = RegistrySnapshot {
            version: REGISTRY_SNAPSHOT_VERSION,
            databases: vec![RegistrySnapshotEntry {
                db_id: "analytics".to_string(),
                path: missing_path.to_string_lossy().to_string(),
            }],
        };
        fs::write(&snapshot_path, serde_json::to_string(&snapshot).unwrap()).unwrap();

        let registry = registry_with_main(&main_path);
        let warnings = load_registry_snapshot(&registry, Some(&snapshot_path));

        assert_eq!(warnings.len(), 1);
        assert_eq!(registry.list().databases.len(), 1);
        assert!(!missing_path.exists());

        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn duplicate_db_ids_warn_and_keep_first_registration() {
        let temp_dir = unique_temp_dir("duplicate-id");
        let main_path = temp_dir.join("main.sqlite");
        let analytics_a_path = temp_dir.join("analytics-a.sqlite");
        let analytics_b_path = temp_dir.join("analytics-b.sqlite");
        let snapshot_path = temp_dir.join("registry.json");

        create_db(&analytics_a_path);
        create_db(&analytics_b_path);

        let snapshot = RegistrySnapshot {
            version: REGISTRY_SNAPSHOT_VERSION,
            databases: vec![
                RegistrySnapshotEntry {
                    db_id: "analytics".to_string(),
                    path: analytics_a_path.to_string_lossy().to_string(),
                },
                RegistrySnapshotEntry {
                    db_id: "analytics".to_string(),
                    path: analytics_b_path.to_string_lossy().to_string(),
                },
            ],
        };
        fs::write(&snapshot_path, serde_json::to_string(&snapshot).unwrap()).unwrap();

        let registry = registry_with_main(&main_path);
        let warnings = load_registry_snapshot(&registry, Some(&snapshot_path));

        assert_eq!(warnings.len(), 1);
        let databases = registry.list().databases;
        assert_eq!(databases.len(), 2);
        assert_eq!(databases[0].db_id, "analytics");
        assert_eq!(databases[0].path, analytics_a_path.to_string_lossy());

        let _ = fs::remove_dir_all(temp_dir);
    }
}
