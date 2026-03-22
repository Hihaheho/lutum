use agents::{
    AdapterStructuredTurn, AdapterTextTurn, AgentError, AssistantInputItem, AssistantTurnItem,
    AssistantTurnView, BudgetLease, BudgetManager, CompletionAdapter, CompletionEventStream,
    CompletionRequest, Context, ContextError, ErasedStructuredTurnEventStream,
    ErasedTextTurnEventStream, InputMessageRole, MessageContent, ModelInput, ModelInputItem,
    ModelName, ModelNameError, NonEmpty, OperationKind, RawJson, RequestBudget, RequestExtensions,
    SharedPoolBudgetManager, SharedPoolBudgetOptions, StructuredTurn, Temperature, TextTurn,
    TextTurnReducer, ToolMetadata, ToolPolicy, ToolUse, TurnAdapter, Usage, UsageEstimate,
    UsageRecoveryAdapter,
};
use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

struct NullAdapter;

#[async_trait]
impl TurnAdapter for NullAdapter {
    async fn text_turn(
        &self,
        _input: ModelInput,
        _turn: AdapterTextTurn,
    ) -> Result<ErasedTextTurnEventStream, AgentError> {
        Ok(Box::pin(futures::stream::empty()) as ErasedTextTurnEventStream)
    }

    async fn structured_turn(
        &self,
        _input: ModelInput,
        _turn: AdapterStructuredTurn,
    ) -> Result<ErasedStructuredTurnEventStream, AgentError> {
        Ok(Box::pin(futures::stream::empty()) as ErasedStructuredTurnEventStream)
    }
}

#[async_trait]
impl CompletionAdapter for NullAdapter {
    async fn completion(
        &self,
        _request: CompletionRequest,
        _extensions: &RequestExtensions,
    ) -> Result<CompletionEventStream, AgentError> {
        Ok(Box::pin(futures::stream::empty()) as CompletionEventStream)
    }
}

#[async_trait]
impl UsageRecoveryAdapter for NullAdapter {
    async fn recover_usage(
        &self,
        _kind: OperationKind,
        _request_id: &str,
    ) -> Result<Option<Usage>, AgentError> {
        Ok(None)
    }
}

struct NonCloneBudget;

impl BudgetManager for NonCloneBudget {
    fn remaining(&self, _extensions: &RequestExtensions) -> agents::Remaining {
        agents::Remaining::default()
    }

    fn reserve(
        &self,
        _extensions: &RequestExtensions,
        estimate: &UsageEstimate,
        request_budget: RequestBudget,
    ) -> Result<BudgetLease, AgentError> {
        Ok(BudgetLease::new(1, *estimate, request_budget))
    }

    fn record_used(&self, _lease: BudgetLease, _usage: Usage) -> Result<(), AgentError> {
        Ok(())
    }
}

#[agents::tool_input(name = "weather", output = WeatherResult)]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct WeatherArgs {
    city: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct WeatherResult {
    forecast: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct Summary {
    answer: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct ToolPlan {
    tools: Vec<ToolsSelector>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema, agents::Toolset)]
enum Tools {
    Weather(WeatherArgs),
}

#[test]
fn typed_public_api_compiles_and_constructs_requests() {
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let ctx = Context::new(Arc::new(NullAdapter), budget);
    let _ = ctx;

    let input = ModelInput::from_items(vec![
        ModelInputItem::Message {
            role: InputMessageRole::User,
            content: NonEmpty::one(MessageContent::Text("hi".into())),
        },
        ModelInputItem::Assistant(AssistantInputItem::Reasoning("thinking".into())),
        ModelInputItem::ToolUse(ToolUse::new(
            "call-1",
            "weather",
            RawJson::parse("{\"city\":\"Tokyo\"}").unwrap(),
            RawJson::from_serializable(&WeatherResult {
                forecast: "sunny".into(),
            })
            .unwrap(),
        )),
    ]);

    let mut _text = TextTurn::<Tools>::new(ModelName::new("gpt-4.1").unwrap());
    _text.config.tools = ToolPolicy::allow_only(vec![ToolsSelector::Weather]);
    _text.config.budget = RequestBudget::from_tokens(256);

    let mut _structured = StructuredTurn::<Tools, Summary>::new(ModelName::new("gpt-4.1").unwrap());
    _structured.config.generation = agents::GenerationParams {
        temperature: Some(Temperature::try_from(0.3).unwrap()),
        max_output_tokens: Some(512),
    };
    let _completion = CompletionRequest::builder()
        .model(ModelName::new("gpt-4.1-mini").unwrap())
        .prompt("hello")
        .budget(RequestBudget::from_tokens(128))
        .build();
    let _estimate = UsageEstimate::zero();
    let _extensions = RequestExtensions::new();
    let _input = input;
}

#[test]
fn context_accepts_non_clone_budget_and_adapter() {
    let ctx = Context::new(Arc::new(NullAdapter), NonCloneBudget);
    let _ = ctx;
}

#[test]
fn model_name_rejects_empty_strings() {
    assert_eq!(ModelName::new("   "), Err(ModelNameError::Empty));
}

#[test]
fn selector_plans_round_trip_and_drive_tool_policy() {
    let plan = ToolPlan {
        tools: vec![ToolsSelector::Weather],
    };
    let json = serde_json::to_string(&plan).unwrap();
    assert_eq!(json, "{\"tools\":[\"weather\"]}");

    let decoded: ToolPlan = serde_json::from_str(&json).unwrap();
    let policy = ToolPolicy::<Tools>::allow_only(decoded.tools);
    let selected = policy
        .selected()
        .unwrap()
        .iter()
        .map(|selector| selector.name())
        .collect::<Vec<_>>();

    assert_eq!(selected, vec!["weather"]);
}

#[test]
fn reducer_is_public_and_directly_usable() {
    let mut reducer = TextTurnReducer::<Tools>::new();
    reducer
        .apply(&agents::TextTurnEvent::Started {
            request_id: Some("req-1".into()),
            model: "gpt-4.1".into(),
        })
        .unwrap();
    reducer
        .apply(&agents::TextTurnEvent::TextDelta {
            delta: "hello".into(),
        })
        .unwrap();
    reducer
        .apply(&agents::TextTurnEvent::Completed {
            request_id: Some("req-1".into()),
            finish_reason: agents::FinishReason::Stop,
            usage: Usage {
                total_tokens: 1,
                ..Usage::zero()
            },
            committed_turn: Arc::new(AssistantTurnView::from_items(&[])),
        })
        .unwrap();
    let result = reducer.into_result().unwrap();
    assert_eq!(result.assistant_text(), "hello");
    assert!(matches!(
        result.assistant_turn.items()[0],
        AssistantTurnItem::Text(ref text) if text == "hello"
    ));
}

#[test]
fn reducer_ignores_duplicate_tool_call_ready() {
    let invocation = ToolsCall::Weather(WeatherArgsCall {
        metadata: ToolMetadata::new(
            "call-1",
            "weather",
            RawJson::parse("{\"city\":\"Tokyo\"}").unwrap(),
        ),
        input: WeatherArgs {
            city: "Tokyo".into(),
        },
    });

    let mut reducer = TextTurnReducer::<Tools>::new();
    reducer
        .apply(&agents::TextTurnEvent::ToolCallReady(invocation.clone()))
        .unwrap();
    reducer
        .apply(&agents::TextTurnEvent::ToolCallReady(invocation))
        .unwrap();
    reducer
        .apply(&agents::TextTurnEvent::Completed {
            request_id: None,
            finish_reason: agents::FinishReason::ToolCall,
            usage: Usage::zero(),
            committed_turn: Arc::new(AssistantTurnView::from_items(&[])),
        })
        .unwrap();

    let result = reducer.into_result().unwrap();
    assert_eq!(result.tool_calls.len(), 1);
}

#[test]
fn context_rejects_invalid_model_input_before_adapter_call() {
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let ctx = Context::new(Arc::new(NullAdapter), budget);
    let input = ModelInput::from_items(vec![
        ModelInputItem::text(InputMessageRole::User, "hello"),
        ModelInputItem::tool_use_parts(
            "dup",
            "weather",
            RawJson::parse("{\"city\":\"Tokyo\"}").unwrap(),
            RawJson::parse("\"sunny\"").unwrap(),
        ),
        ModelInputItem::tool_use_parts(
            "dup",
            "weather",
            RawJson::parse("{\"city\":\"Tokyo\"}").unwrap(),
            RawJson::parse("\"rainy\"").unwrap(),
        ),
    ]);

    let err = futures::executor::block_on(ctx.text_turn(
        RequestExtensions::new(),
        input,
        TextTurn::<Tools>::new(ModelName::new("gpt-4.1").unwrap()),
        UsageEstimate::zero(),
    ));

    assert!(matches!(
        err,
        Err(ContextError::InvalidModelInput(
            agents::ModelInputValidationError::DuplicateToolUseId { .. }
        ))
    ));
}
