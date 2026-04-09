use serde::Deserialize;

/// A single evaluation test case.
#[derive(Debug, Clone, Deserialize)]
pub struct TestCase {
    /// Human-readable name for this test.
    pub name: String,
    /// The user prompt to send to the agent.
    pub prompt: String,
    /// Expected tool name to be called (e.g. "select", "delete").
    pub expect_tool: Option<String>,
    /// If set, the result set must have exactly this many rows.
    pub expect_rows: Option<usize>,
    /// If true, automatically approve any write dry-run (useful for batch eval).
    #[serde(default)]
    pub auto_approve: bool,
    /// Override max_rows for this case (uses AgentConfig default otherwise).
    pub max_rows: Option<u64>,
    /// Whether to allow write ops for this case (default: read-only).
    #[serde(default)]
    pub allow_writes: bool,
}

#[derive(Debug, Deserialize)]
pub struct TestSuite {
    #[serde(rename = "case")]
    pub cases: Vec<TestCase>,
}

impl TestSuite {
    pub fn from_toml(src: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(src)
    }
}
