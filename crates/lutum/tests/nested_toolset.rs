/// Integration tests for `#[toolset]`-annotated nested toolset variants in `#[derive(Toolset)]`.
use futures::executor::block_on;
use lutum::{RawJson, ToolCallError, ToolCallWrapper, ToolDecision, ToolHookOutcome, ToolMetadata, Toolset};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

// ── Inner toolset ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct WeatherResult {
    forecast: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct StockResult {
    price: u64,
}

/// Inner weather tool
#[lutum::tool_input(name = "weather", output = WeatherResult)]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct WeatherArgs {
    city: String,
}

/// Inner stock price tool
#[lutum::tool_input(name = "stock", output = StockResult)]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct StockArgs {
    ticker: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
enum InnerTools {
    Weather(WeatherArgs),
    Stock(StockArgs),
}

// ── Second inner toolset ──────────────────────────────────────────────────────

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct SearchResult {
    hits: u32,
}

/// Web search tool
#[lutum::tool_input(name = "search", output = SearchResult)]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct SearchArgs {
    query: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
enum SearchTools {
    Search(SearchArgs),
}

// ── Outer toolset (mixes nested + regular) ────────────────────────────────────

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct CalcResult {
    result: i64,
}

/// Calculator tool (direct, not nested)
#[lutum::tool_input(name = "calc", output = CalcResult)]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct CalcArgs {
    expression: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
enum OuterTools {
    /// Nested: all InnerTools tools (weather, stock)
    #[toolset]
    Inner(InnerTools),
    /// Direct regular tool
    Calc(CalcArgs),
    /// Nested: SearchTools
    #[toolset]
    Search(SearchTools),
}

// ── impl_hook for inner tools ─────────────────────────────────────────────────

#[lutum::impl_hook(WeatherHook)]
async fn cached_weather(
    _meta: &lutum::ToolMetadata,
    input: WeatherArgs,
) -> ToolDecision<WeatherArgs, WeatherResult> {
    if input.city == "Tokyo" {
        ToolDecision::Complete(WeatherResult {
            forecast: "cached: 22C".to_string(),
        })
    } else {
        ToolDecision::RunNormally(input)
    }
}

#[lutum::impl_hook(WeatherDescriptionHook)]
async fn inner_weather_desc(_def: &lutum::ToolDef) -> Option<String> {
    Some("Overridden weather description".to_string())
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn make_metadata(id: &str, name: &str, json: &str) -> ToolMetadata {
    ToolMetadata::new(id, name, RawJson::parse(json).unwrap())
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[test]
fn definitions_includes_all_tools() {
    let defs = OuterTools::definitions();
    let names: Vec<&str> = defs.iter().map(|d| d.name).collect();
    // All tool names should be present (order: inner first = weather, stock; then calc; then search)
    assert!(names.contains(&"weather"), "missing weather: {names:?}");
    assert!(names.contains(&"stock"), "missing stock: {names:?}");
    assert!(names.contains(&"calc"), "missing calc: {names:?}");
    assert!(names.contains(&"search"), "missing search: {names:?}");
    assert_eq!(names.len(), 4, "expected exactly 4 tools, got {names:?}");
}

#[test]
fn parse_regular_tool() {
    let meta = make_metadata("id1", "calc", r#"{"expression":"1+1"}"#);
    let call = OuterTools::parse_tool_call(meta).unwrap();
    assert!(
        matches!(call, OuterToolsCall::Calc(_)),
        "expected Calc variant, got: {call:?}"
    );
}

#[test]
fn parse_nested_inner_tool() {
    let meta = make_metadata("id2", "weather", r#"{"city":"Tokyo"}"#);
    let call = OuterTools::parse_tool_call(meta).unwrap();
    assert!(
        matches!(call, OuterToolsCall::Inner(_)),
        "expected Inner variant, got: {call:?}"
    );
}

#[test]
fn parse_nested_search_tool() {
    let meta = make_metadata("id3", "search", r#"{"query":"hello"}"#);
    let call = OuterTools::parse_tool_call(meta).unwrap();
    assert!(
        matches!(call, OuterToolsCall::Search(_)),
        "expected Search variant, got: {call:?}"
    );
}

#[test]
fn parse_unknown_tool_returns_error() {
    let meta = make_metadata("id4", "unknown_tool", r#"{}"#);
    let err = OuterTools::parse_tool_call(meta).unwrap_err();
    assert!(
        matches!(err, ToolCallError::UnknownTool { .. }),
        "expected UnknownTool error, got: {err:?}"
    );
}

#[test]
fn selector_all_contains_all_tools() {
    let all = OuterToolsSelector::all();
    let names: Vec<&str> = all.iter().map(|s| s.name()).collect();
    assert!(names.contains(&"weather"), "missing weather: {names:?}");
    assert!(names.contains(&"stock"), "missing stock: {names:?}");
    assert!(names.contains(&"calc"), "missing calc: {names:?}");
    assert!(names.contains(&"search"), "missing search: {names:?}");
    assert_eq!(names.len(), 4, "expected 4 selectors, got {names:?}");
}

#[test]
fn selector_try_from_name_works_for_all_tools() {
    assert!(
        matches!(
            OuterToolsSelector::try_from_name("weather"),
            Some(OuterToolsSelector::Inner(_))
        ),
        "weather should resolve to Inner variant"
    );
    assert!(
        matches!(
            OuterToolsSelector::try_from_name("calc"),
            Some(OuterToolsSelector::Calc)
        ),
        "calc should resolve to Calc variant"
    );
    assert!(
        matches!(
            OuterToolsSelector::try_from_name("search"),
            Some(OuterToolsSelector::Search(_))
        ),
        "search should resolve to Search variant"
    );
    assert!(
        OuterToolsSelector::try_from_name("unknown").is_none(),
        "unknown should resolve to None"
    );
}

#[test]
fn selector_name_roundtrips() {
    for sel in OuterToolsSelector::all() {
        let name = sel.name();
        let recovered = OuterToolsSelector::try_from_name(name)
            .unwrap_or_else(|| panic!("could not recover selector for name '{name}'"));
        assert_eq!(
            recovered.name(),
            name,
            "name should roundtrip via try_from_name"
        );
    }
}

#[test]
fn hooks_new_takes_nested_hooks_as_args() {
    // OuterToolsHooks::new should accept (InnerToolsHooks, SearchToolsHooks).
    let inner_hooks = InnerToolsHooks::new();
    let search_hooks = SearchToolsHooks::new();
    let _outer_hooks = OuterToolsHooks::new(inner_hooks, search_hooks);
    // No assertion needed — compile-time check is the goal.
}

#[test]
fn nested_field_accessible_for_registration() {
    let mut hooks = OuterToolsHooks::new(InnerToolsHooks::new(), SearchToolsHooks::new());
    // Register a hook on the nested InnerToolsHooks via field access.
    hooks.inner.register_weather_hook(CachedWeather);
    // This verifies hooks.inner is pub and has register_weather_hook.
}

#[test]
fn hook_dispatch_to_nested_complete() {
    let mut outer_hooks = OuterToolsHooks::new(InnerToolsHooks::new(), SearchToolsHooks::new());
    outer_hooks.inner.register_weather_hook(CachedWeather);

    let meta = make_metadata("id5", "weather", r#"{"city":"Tokyo"}"#);
    let call = OuterTools::parse_tool_call(meta).unwrap();

    let outcome = block_on(call.hook(&outer_hooks));
    assert!(
        matches!(outcome, ToolHookOutcome::Handled(OuterToolsHandled::Inner(_))),
        "expected Handled(Inner(...)), got: {outcome:?}"
    );
}

#[test]
fn hook_dispatch_to_nested_run_normally() {
    let mut outer_hooks = OuterToolsHooks::new(InnerToolsHooks::new(), SearchToolsHooks::new());
    outer_hooks.inner.register_weather_hook(CachedWeather);

    // Non-Tokyo city → RunNormally → Unhandled
    let meta = make_metadata("id6", "weather", r#"{"city":"NYC"}"#);
    let call = OuterTools::parse_tool_call(meta).unwrap();

    let outcome = block_on(call.hook(&outer_hooks));
    assert!(
        matches!(outcome, ToolHookOutcome::Unhandled(OuterToolsCall::Inner(_))),
        "expected Unhandled(Inner(...)), got: {outcome:?}"
    );
}

#[test]
fn hook_dispatch_regular_tool_unhandled() {
    let outer_hooks = OuterToolsHooks::new(InnerToolsHooks::new(), SearchToolsHooks::new());

    let meta = make_metadata("id7", "calc", r#"{"expression":"2+2"}"#);
    let call = OuterTools::parse_tool_call(meta).unwrap();

    let outcome = block_on(call.hook(&outer_hooks));
    assert!(
        matches!(outcome, ToolHookOutcome::Unhandled(OuterToolsCall::Calc(_))),
        "expected Unhandled(Calc(...)), got: {outcome:?}"
    );
}

#[test]
fn description_overrides_from_nested_hooks() {
    let mut outer_hooks = OuterToolsHooks::new(InnerToolsHooks::new(), SearchToolsHooks::new());
    outer_hooks
        .inner
        .register_weather_description_hook(InnerWeatherDesc);

    let overrides = block_on(outer_hooks.description_overrides());
    assert_eq!(overrides.len(), 1, "expected 1 override, got: {overrides:?}");
    let (sel, desc) = &overrides[0];
    assert!(
        matches!(sel, OuterToolsSelector::Inner(_)),
        "selector should be Inner variant, got: {sel:?}"
    );
    assert_eq!(desc, "Overridden weather description");
}

#[test]
fn selector_definition_accessible_for_nested() {
    let weather_sel = OuterToolsSelector::try_from_name("weather").unwrap();
    let def = weather_sel.definition();
    assert_eq!(def.name, "weather");
}

#[test]
fn selector_definition_accessible_for_regular() {
    let calc_sel = OuterToolsSelector::try_from_name("calc").unwrap();
    let def = calc_sel.definition();
    assert_eq!(def.name, "calc");
}

#[test]
fn call_metadata_accessible_for_nested() {
    let meta = make_metadata("my-id", "weather", r#"{"city":"London"}"#);
    let call = OuterTools::parse_tool_call(meta).unwrap();
    assert_eq!(call.metadata().id.as_str(), "my-id");
    assert_eq!(call.metadata().name.as_str(), "weather");
}

#[test]
fn call_selector_for_nested() {
    let meta = make_metadata("id8", "stock", r#"{"ticker":"AAPL"}"#);
    let call = OuterTools::parse_tool_call(meta).unwrap();
    let sel = call.selector();
    assert!(
        matches!(sel, OuterToolsSelector::Inner(_)),
        "selector should be Inner, got: {sel:?}"
    );
    assert_eq!(sel.name(), "stock");
}

#[test]
fn call_into_input_for_nested() {
    let meta = make_metadata("id9", "weather", r#"{"city":"Paris"}"#);
    let call = OuterTools::parse_tool_call(meta).unwrap();
    let input = call.into_input();
    assert!(
        matches!(input, OuterTools::Inner(_)),
        "into_input should return Inner variant, got: {input:?}"
    );
}

#[test]
fn handled_metadata_for_nested() {
    let mut outer_hooks = OuterToolsHooks::new(InnerToolsHooks::new(), SearchToolsHooks::new());
    outer_hooks.inner.register_weather_hook(CachedWeather);

    let meta_berlin = make_metadata("id10", "weather", r#"{"city":"Tokyo"}"#);
    let call_tokyo = OuterTools::parse_tool_call(meta_berlin).unwrap();
    let outcome = block_on(call_tokyo.hook(&outer_hooks));

    if let ToolHookOutcome::Handled(handled) = outcome {
        assert_eq!(handled.metadata().id.as_str(), "id10");
    } else {
        panic!("expected Handled outcome");
    }
}
