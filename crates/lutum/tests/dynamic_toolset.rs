use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::executor::block_on;
use lutum::{
    AdapterStructuredCompletionRequest, AdapterStructuredTurn, AdapterTextTurn, AdapterToolChoice,
    AgentError, AssistantTurnView, CommittedTurn, CompletionAdapter, CompletionEventStream,
    CompletionRequest, DynamicTool, ErasedStructuredCompletionEventStream,
    ErasedStructuredTurnEventStream, ErasedTextTurnEvent, ErasedTextTurnEventStream, FinishReason,
    IntoToolResult, Lutum, MockLlmAdapter, MockTextScenario, ModelInput, ModelInputItem,
    OperationKind, RawJson, RawTextTurnEvent, RecoverableToolCallIssueReason, RequestExtensions,
    Session, SharedPoolBudgetManager, SharedPoolBudgetOptions, TextStepOutcomeWithTools,
    ToolMetadata, Toolset, TurnAdapter, Usage, UsageRecoveryAdapter,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct SearchResult {
    answer: String,
}

/// Search indexed documents
#[lutum::tool_input(name = "search", output = SearchResult)]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct SearchInput {
    query: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
enum DynamicTools {
    Search(SearchInput),
    #[dynamic]
    Dynamic(lutum::DynamicTool),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
enum StaticTools {
    Lookup(SearchInput),
}

#[derive(Clone, Default)]
struct SpyAdapter {
    captured: Arc<Mutex<Vec<AdapterTextTurn>>>,
}

impl SpyAdapter {
    fn captured(&self) -> Vec<AdapterTextTurn> {
        self.captured.lock().unwrap().clone()
    }
}

#[cfg_attr(target_family = "wasm", async_trait(?Send))]
#[cfg_attr(not(target_family = "wasm"), async_trait)]
impl TurnAdapter for SpyAdapter {
    async fn text_turn(
        &self,
        _input: ModelInput,
        turn: AdapterTextTurn,
    ) -> Result<ErasedTextTurnEventStream, AgentError> {
        self.captured.lock().unwrap().push(turn);
        let committed: CommittedTurn = Arc::new(AssistantTurnView::from_items(&[]));
        Ok(Box::pin(futures::stream::iter(vec![
            Ok(ErasedTextTurnEvent::Started {
                request_id: Some("spy-req".into()),
                model: "spy-model".into(),
            }),
            Ok(ErasedTextTurnEvent::TextDelta { delta: "ok".into() }),
            Ok(ErasedTextTurnEvent::Completed {
                request_id: Some("spy-req".into()),
                finish_reason: FinishReason::Stop,
                usage: Usage::zero(),
                committed_turn: committed,
            }),
        ])))
    }

    async fn structured_turn(
        &self,
        _input: ModelInput,
        _turn: AdapterStructuredTurn,
    ) -> Result<ErasedStructuredTurnEventStream, AgentError> {
        unimplemented!("not needed for dynamic toolset tests")
    }
}

#[cfg_attr(target_family = "wasm", async_trait(?Send))]
#[cfg_attr(not(target_family = "wasm"), async_trait)]
impl UsageRecoveryAdapter for SpyAdapter {
    async fn recover_usage(
        &self,
        _kind: OperationKind,
        _request_id: &str,
    ) -> Result<Option<Usage>, AgentError> {
        Ok(None)
    }
}

#[cfg_attr(target_family = "wasm", async_trait(?Send))]
#[cfg_attr(not(target_family = "wasm"), async_trait)]
impl CompletionAdapter for SpyAdapter {
    async fn completion(
        &self,
        _request: CompletionRequest,
        _extensions: &RequestExtensions,
    ) -> Result<CompletionEventStream, AgentError> {
        unimplemented!("not needed for dynamic toolset tests")
    }

    async fn structured_completion(
        &self,
        _request: AdapterStructuredCompletionRequest,
        _extensions: &RequestExtensions,
    ) -> Result<ErasedStructuredCompletionEventStream, AgentError> {
        unimplemented!("not needed for dynamic toolset tests")
    }
}

fn ctx(adapter: impl TurnAdapter + UsageRecoveryAdapter + CompletionAdapter + 'static) -> Lutum {
    Lutum::new(
        Arc::new(adapter),
        SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
    )
}

fn input() -> ModelInput {
    ModelInput::new().user("use tools")
}

fn weather_tool() -> DynamicTool {
    DynamicTool::new(
        "weather",
        "Get current weather",
        serde_json::json!({
            "type": "object",
            "properties": {
                "city": { "type": "string" }
            },
            "required": ["city"]
        }),
    )
}

fn weather_tool_call_events() -> Vec<Result<RawTextTurnEvent, lutum::MockError>> {
    vec![
        Ok(RawTextTurnEvent::Started {
            request_id: Some("req-dyn".into()),
            model: "mock".into(),
        }),
        Ok(RawTextTurnEvent::ToolCallChunk {
            id: "call-weather".into(),
            name: "weather".into(),
            arguments_json_delta: "{\"city\":\"Tokyo\"}".into(),
        }),
        Ok(RawTextTurnEvent::Completed {
            request_id: Some("req-dyn".into()),
            finish_reason: FinishReason::ToolCall,
            usage: Usage::zero(),
        }),
    ]
}

#[test]
fn dynamic_toolset_shape_and_call_helpers() {
    assert!(DynamicTools::has_dynamic_slot());
    assert!(!StaticTools::has_dynamic_slot());
    assert_eq!(DynamicTools::definitions().len(), 1);
    assert_eq!(DynamicToolsSelector::all(), &[DynamicToolsSelector::Search]);
    assert_eq!(
        DynamicToolsSelector::try_from_name("weather"),
        None,
        "dynamic tools must not appear in the selector enum"
    );

    let dynamic_call = DynamicTools::parse_tool_call(ToolMetadata::new(
        "call-weather",
        "weather",
        RawJson::parse("{\"city\":\"Tokyo\"}").unwrap(),
    ))
    .unwrap();
    assert_eq!(dynamic_call.selector(), None);
    assert!(dynamic_call.as_dynamic().is_some());
    let (metadata, input) = dynamic_call.clone().into_parts();
    assert_eq!(metadata.name.as_str(), "weather");
    assert!(input.is_none());
    assert_eq!(dynamic_call.into_dynamic().unwrap().name(), "weather");

    let static_call = DynamicTools::parse_tool_call(ToolMetadata::new(
        "call-search",
        "search",
        RawJson::parse("{\"query\":\"rust\"}").unwrap(),
    ))
    .unwrap();
    assert_eq!(static_call.selector(), Some(DynamicToolsSelector::Search));
    assert!(matches!(
        static_call.into_input(),
        Some(DynamicTools::Search(SearchInput { ref query })) if query == "rust"
    ));
}

#[test]
fn dynamic_tool_call_dispatches_when_registered() {
    let adapter = MockLlmAdapter::new()
        .with_text_scenario(MockTextScenario::events(weather_tool_call_events()));
    let outcome = block_on(async {
        ctx(adapter)
            .text_turn(input())
            .tools::<DynamicTools>()
            .with_dynamic_tools([weather_tool()])
            .collect()
            .await
            .unwrap()
    });

    match outcome {
        TextStepOutcomeWithTools::NeedsTools(round) => {
            assert_eq!(round.tool_calls.len(), 1);
            assert!(round.recoverable_tool_call_issues().is_empty());
            assert!(matches!(
                &round.tool_calls[0],
                DynamicToolsCall::Dynamic(call)
                    if call.name() == "weather"
                        && call.arguments().get() == "{\"city\":\"Tokyo\"}"
            ));
        }
        other => panic!("expected tool round, got {other:?}"),
    }
}

#[test]
fn dynamic_tool_call_is_not_available_when_not_registered() {
    let adapter = MockLlmAdapter::new()
        .with_text_scenario(MockTextScenario::events(weather_tool_call_events()));
    let outcome = block_on(async {
        ctx(adapter)
            .text_turn(input())
            .tools::<DynamicTools>()
            .collect()
            .await
            .unwrap()
    });

    match outcome {
        TextStepOutcomeWithTools::NeedsTools(round) => {
            assert!(round.tool_calls.is_empty());
            assert_eq!(round.recoverable_tool_call_issues().len(), 1);
            assert_eq!(
                round.recoverable_tool_call_issues()[0].reason,
                RecoverableToolCallIssueReason::NotAvailable
            );
        }
        other => panic!("expected tool round, got {other:?}"),
    }
}

#[test]
fn mixed_static_and_dynamic_calls_commit_in_order() {
    let adapter = MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![
        Ok(RawTextTurnEvent::Started {
            request_id: Some("req-mixed".into()),
            model: "mock".into(),
        }),
        Ok(RawTextTurnEvent::ToolCallChunk {
            id: "call-search".into(),
            name: "search".into(),
            arguments_json_delta: "{\"query\":\"rust\"}".into(),
        }),
        Ok(RawTextTurnEvent::ToolCallChunk {
            id: "call-weather".into(),
            name: "weather".into(),
            arguments_json_delta: "{\"city\":\"Tokyo\"}".into(),
        }),
        Ok(RawTextTurnEvent::Completed {
            request_id: Some("req-mixed".into()),
            finish_reason: FinishReason::ToolCall,
            usage: Usage::zero(),
        }),
    ]));
    let ctx = ctx(adapter);
    let mut session = Session::new();
    session.push_user("search and get weather");

    let outcome = block_on(async {
        session
            .text_turn()
            .tools::<DynamicTools>()
            .with_dynamic_tools([weather_tool()])
            .collect(&ctx)
            .await
            .unwrap()
    });

    let round = match outcome {
        TextStepOutcomeWithTools::NeedsTools(round) => round,
        other => panic!("expected tool round, got {other:?}"),
    };
    let handled = round
        .tool_calls
        .iter()
        .cloned()
        .map(|call| match call {
            DynamicToolsCall::Search(call) => {
                DynamicToolsHandled::from(call.handled(SearchResult {
                    answer: "found".into(),
                }))
            }
            DynamicToolsCall::Dynamic(call) => DynamicToolsHandled::from(
                call.handled(RawJson::parse("{\"forecast\":\"sunny\"}").unwrap()),
            ),
        })
        .collect::<Vec<_>>();
    round.commit(&mut session, handled).unwrap();

    let items = session.input().items();
    assert_eq!(items.len(), 4);
    assert!(matches!(items[1], ModelInputItem::Turn(_)));
    assert!(matches!(
        &items[2],
        ModelInputItem::ToolResult(result)
            if result.name.as_str() == "search" && result.id.as_str() == "call-search"
    ));
    assert!(matches!(
        &items[3],
        ModelInputItem::ToolResult(result)
            if result.name.as_str() == "weather"
                && result.id.as_str() == "call-weather"
                && result.result.get() == "{\"forecast\":\"sunny\"}"
    ));
}

#[test]
fn handled_dynamic_tool_builds_matching_tool_result() {
    let call = DynamicTools::parse_tool_call(ToolMetadata::new(
        "call-weather",
        "weather",
        RawJson::parse("{\"city\":\"Tokyo\"}").unwrap(),
    ))
    .unwrap()
    .into_dynamic()
    .unwrap();
    let result = call
        .handled(RawJson::parse("{\"forecast\":\"sunny\"}").unwrap())
        .into_tool_result()
        .unwrap();

    assert_eq!(result.id.as_str(), "call-weather");
    assert_eq!(result.name.as_str(), "weather");
    assert_eq!(result.arguments.get(), "{\"city\":\"Tokyo\"}");
    assert_eq!(result.result.get(), "{\"forecast\":\"sunny\"}");
}

#[test]
fn dynamic_schema_reaches_adapter_verbatim() {
    let spy = SpyAdapter::default();
    let spy_handle = spy.clone();
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "city": { "type": "string" }
        }
    });

    block_on(async {
        ctx(spy)
            .text_turn(ModelInput::new().user("hello"))
            .tools::<DynamicTools>()
            .with_dynamic_tools([DynamicTool::new("weather", "Weather", schema.clone())])
            .collect()
            .await
            .unwrap();
    });

    let captured = spy_handle.captured();
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].config.tool_choice, AdapterToolChoice::Auto);
    let tools = &captured[0].config.tools;
    assert!(tools.iter().any(|tool| tool.name == "search"));
    let weather = tools.iter().find(|tool| tool.name == "weather").unwrap();
    assert_eq!(weather.description, "Weather");
    assert_eq!(weather.input_schema, schema);
}

#[test]
fn duplicate_dynamic_tool_names_are_rejected() {
    let err = block_on(async {
        ctx(SpyAdapter::default())
            .text_turn(input())
            .tools::<DynamicTools>()
            .with_dynamic_tools([weather_tool(), weather_tool()])
            .collect()
            .await
            .unwrap_err()
    });

    assert!(matches!(
        err,
        lutum::CollectError::Execution {
            source: AgentError::InvalidToolConstraints { ref tool },
            ..
        } if tool == "weather"
    ));
}

#[test]
fn dynamic_tool_name_colliding_with_static_tool_is_rejected() {
    let err = block_on(async {
        ctx(SpyAdapter::default())
            .text_turn(input())
            .tools::<DynamicTools>()
            .with_dynamic_tools([DynamicTool::new(
                "search",
                "conflict",
                serde_json::json!({}),
            )])
            .collect()
            .await
            .unwrap_err()
    });

    assert!(matches!(
        err,
        lutum::CollectError::Execution {
            source: AgentError::InvalidToolConstraints { ref tool },
            ..
        } if tool == "search"
    ));
}
