# AGENTS.md

## Cursor Cloud specific instructions

This is a **pure Rust library workspace** (no web frontend, no database, no Docker). There are no services to start.

### Development commands

All standard commands are documented in the `README.md` under **Development**:

```bash
cargo check --workspace --all-targets
cargo test --workspace
cargo clippy --workspace --all-targets
```

### Known issues

- The trybuild test `hook_def_singleton_rejects_last_argument` (`cargo test -p lutum --test hook_def_macros`) may fail on first run because it auto-generates a `.stderr` snapshot file. This is a pre-existing repo issue (the expected stderr file is in `crates/lutum/wip/` instead of `crates/lutum/tests/ui/`). All other workspace tests pass.

### Rust toolchain

The workspace uses `edition = "2024"`, which requires **Rust 1.85+**. The update script installs `stable` via rustup. The currently installed version is 1.94.1.

### Examples

Examples require external LLM API credentials (`TOKEN`, `MODEL`, `ENDPOINT` env vars) and are not part of the standard test/check flow.
