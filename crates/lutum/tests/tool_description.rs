/// Tests for runtime tool description overrides (use case 1: builder API,
/// use case 2: #[ToolSet]-generated description hooks).

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::executor::block_on;
use lutum::{
    AdapterStructuredCompletionRequest, AdapterStructuredTurn, AdapterTextTurn, AgentError,
    AssistantTurnView, CommittedTurn, CompletionAdapter, CompletionEventStream, CompletionRequest,
    ErasedStructuredCompletionEventStream, ErasedStructuredTurnEventStream, ErasedTextTurnEvent,
    ErasedTextTurnEventStream, FinishReason, InputMessageRole, Lutum, ModelInput, ModelInputItem,
    OperationKind, RequestExtensions, SharedPoolBudgetManager, SharedPoolBudgetOptions, TurnAdapter,
    Usage, UsageRecoveryAdapter,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

// ── shared tool definitions ───────────────────────────────────────────────────

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct WeatherResult {
    forecast: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct SearchResult {
    hits: Vec<String>,
}

/// Get weather
#[lutum::tool_input(name = "weather", output = WeatherResult)]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct WeatherArgs {
    city: String,
}

/// Search the web
#[lutum::tool_input(name = "search", output = SearchResult)]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct SearchArgs {
    query: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
enum Tools {
    Weather(WeatherArgs),
    Search(SearchArgs),
}

// ── spy adapter ───────────────────────────────────────────────────────────────

/// Captures the `AdapterTurnConfig` from each `text_turn` call so tests can
/// assert on the tool descriptions that were actually sent to the adapter.
#[derive(Clone, Default)]
struct SpyAdapter {
    captured: Arc<Mutex<Vec<AdapterTextTurn>>>,
}

impl SpyAdapter {
    fn new() -> Self {
        Self::default()
    }

    fn captured(&self) -> Vec<AdapterTextTurn> {
        self.captured.lock().unwrap().clone()
    }
}

#[async_trait]
impl TurnAdapter for SpyAdapter {
    async fn text_turn(
        &self,
        _input: ModelInput,
        turn: AdapterTextTurn,
    ) -> Result<ErasedTextTurnEventStream, AgentError> {
        self.captured.lock().unwrap().push(turn);
        // Return a minimal valid stream: Started + Completed (committed_turn required)
        let committed: CommittedTurn = Arc::new(AssistantTurnView::from_items(&[]));
        let events: Vec<Result<ErasedTextTurnEvent, AgentError>> = vec![
            Ok(ErasedTextTurnEvent::Started {
                request_id: Some("spy-req".into()),
                model: "spy-model".into(),
            }),
            Ok(ErasedTextTurnEvent::Completed {
                request_id: Some("spy-req".into()),
                finish_reason: FinishReason::Stop,
                usage: Usage::zero(),
                committed_turn: committed,
            }),
        ];
        Ok(Box::pin(futures::stream::iter(events)))
    }

    async fn structured_turn(
        &self,
        _input: ModelInput,
        _turn: AdapterStructuredTurn,
    ) -> Result<ErasedStructuredTurnEventStream, AgentError> {
        unimplemented!("not needed for description tests")
    }
}

#[async_trait]
impl UsageRecoveryAdapter for SpyAdapter {
    async fn recover_usage(
        &self,
        _kind: OperationKind,
        _request_id: &str,
    ) -> Result<Option<Usage>, AgentError> {
        Ok(None)
    }
}

#[async_trait]
impl CompletionAdapter for SpyAdapter {
    async fn completion(
        &self,
        _request: CompletionRequest,
        _extensions: &RequestExtensions,
    ) -> Result<CompletionEventStream, AgentError> {
        unimplemented!()
    }

    async fn structured_completion(
        &self,
        _request: AdapterStructuredCompletionRequest,
        _extensions: &RequestExtensions,
    ) -> Result<ErasedStructuredCompletionEventStream, AgentError> {
        unimplemented!()
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn make_ctx(spy: Arc<SpyAdapter>) -> Lutum {
    Lutum::new(spy, SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()))
}

fn user_input() -> ModelInput {
    ModelInput::from_items(vec![ModelInputItem::text(InputMessageRole::User, "hi")])
}

// ── use case 1: builder `.describe()` ────────────────────────────────────────

#[test]
fn describe_overrides_single_tool_description() {
    let spy = Arc::new(SpyAdapter::new());
    let ctx = make_ctx(Arc::clone(&spy));

    // Use `.start()` so the turn fires (populating the spy) without needing a
    // meaningful stream to collect — we only care about what was passed in.
    block_on(
        ctx.text_turn(user_input())
            .tools::<Tools>()
            .describe(ToolsSelector::Weather, "Weather (calls left: 2)")
            .start(),
    )
    .unwrap();

    let captured = spy.captured();
    assert_eq!(captured.len(), 1);
    let tool_map: std::collections::HashMap<&str, &str> = captured[0]
        .config
        .tools
        .iter()
        .map(|t| (t.name.as_str(), t.description.as_str()))
        .collect();

    assert_eq!(
        tool_map["weather"],
        "Weather (calls left: 2)",
        "description override should replace the static description"
    );
    assert_eq!(
        tool_map["search"],
        "Search the web",
        "unoverridden tool should keep its static description"
    );
}

#[test]
fn describe_many_applies_bulk_overrides() {
    let spy = Arc::new(SpyAdapter::new());
    let ctx = make_ctx(Arc::clone(&spy));

    let overrides = vec![
        (ToolsSelector::Weather, "Weather override".to_string()),
        (ToolsSelector::Search, "Search override".to_string()),
    ];

    block_on(
        ctx.text_turn(user_input())
            .tools::<Tools>()
            .describe_many(overrides)
            .start(),
    )
    .unwrap();

    let captured = spy.captured();
    assert_eq!(captured.len(), 1);
    let tool_map: std::collections::HashMap<&str, &str> = captured[0]
        .config
        .tools
        .iter()
        .map(|t| (t.name.as_str(), t.description.as_str()))
        .collect();

    assert_eq!(tool_map["weather"], "Weather override");
    assert_eq!(tool_map["search"], "Search override");
}

#[test]
fn describe_last_write_wins_for_same_selector() {
    let spy = Arc::new(SpyAdapter::new());
    let ctx = make_ctx(Arc::clone(&spy));

    block_on(
        ctx.text_turn(user_input())
            .tools::<Tools>()
            .describe(ToolsSelector::Weather, "first")
            .describe(ToolsSelector::Weather, "second")
            .start(),
    )
    .unwrap();

    let captured = spy.captured();
    let weather = captured[0]
        .config
        .tools
        .iter()
        .find(|t| t.name == "weather")
        .unwrap();

    assert_eq!(weather.description, "second", "last describe() wins");
}

// ── use case 2: #[ToolSet]-generated description hooks ───────────────────────

#[test]
fn description_hook_fires_when_registered() {
    let hooks = ToolsHooks::new().with_weather_description(|_def| async {
        Some("Dynamic weather description".to_string())
    });

    let overrides = block_on(hooks.description_overrides());
    assert_eq!(overrides.len(), 1, "only the registered hook should fire");
    let (sel, desc) = &overrides[0];
    assert_eq!(*sel, ToolsSelector::Weather);
    assert_eq!(desc, "Dynamic weather description");
}

#[test]
fn description_hook_returns_none_when_not_registered() {
    // No hooks registered — description_overrides should return empty vec
    let hooks = ToolsHooks::new();
    let overrides = block_on(hooks.description_overrides());
    assert!(
        overrides.is_empty(),
        "no hooks registered means no overrides, got: {overrides:?}"
    );
}

#[test]
fn description_hooks_for_all_tools_collected_together() {
    let hooks = ToolsHooks::new()
        .with_weather_description(|_def| async { Some("W override".to_string()) })
        .with_search_description(|_def| async { Some("S override".to_string()) });

    let overrides = block_on(hooks.description_overrides());
    assert_eq!(overrides.len(), 2);

    let map: std::collections::HashMap<ToolsSelector, &str> =
        overrides.iter().map(|(s, d)| (*s, d.as_str())).collect();
    assert_eq!(map[&ToolsSelector::Weather], "W override");
    assert_eq!(map[&ToolsSelector::Search], "S override");
}

#[test]
fn description_hook_receives_static_tool_def() {
    // The hook receives the ToolDef for the correct tool
    let received_name: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let received_name_clone = Arc::clone(&received_name);

    let hooks = ToolsHooks::new().with_weather_description(move |def| {
        let name = def.name.to_string();
        let slot = Arc::clone(&received_name_clone);
        async move {
            *slot.lock().unwrap() = Some(name);
            None::<String>
        }
    });

    block_on(hooks.description_overrides());

    assert_eq!(
        received_name.lock().unwrap().as_deref(),
        Some("weather"),
        "description hook should receive the ToolDef for that tool"
    );
}

#[test]
fn description_hooks_integrate_with_builder_describe_many() {
    // Combine use case 2 → use case 1: collect overrides from hooks then pass to
    // the builder via describe_many().
    let hooks = ToolsHooks::new()
        .with_weather_description(|_def| async { Some("Hooked weather".to_string()) });

    let overrides = block_on(hooks.description_overrides());

    let spy = Arc::new(SpyAdapter::new());
    let ctx = make_ctx(Arc::clone(&spy));

    block_on(
        ctx.text_turn(user_input())
            .tools::<Tools>()
            .describe_many(overrides)
            .start(),
    )
    .unwrap();

    let captured = spy.captured();
    let weather = captured[0]
        .config
        .tools
        .iter()
        .find(|t| t.name == "weather")
        .unwrap();
    assert_eq!(weather.description, "Hooked weather");
}
