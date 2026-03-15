use agents::{
    AssistantInputItem, AssistantTurnItem, BudgetLease, BudgetManager, CompletionEventStream,
    CompletionRequest, Context, ContextError, InputMessageRole, LlmAdapter, Marker, MessageContent,
    ModelInput, ModelInputItem, ModelName, ModelNameError, NonEmpty, RawJson, ReasoningEffort,
    ReasoningParams, RequestBudget, SharedPoolBudgetManager, SharedPoolBudgetOptions, StreamKind,
    StructuredOutput, StructuredTurn, StructuredTurnEventStream, Temperature, TextTurn,
    TextTurnEventStream, TextTurnReducer, ToolMetadata, ToolPolicy, ToolUse, Toolset, Usage,
    UsageEstimate,
};
use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug)]
struct AppMarker;

impl Marker for AppMarker {
    fn span_name(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("typed")
    }
}

struct NullAdapter;

#[derive(Debug)]
struct Never;

impl std::fmt::Display for Never {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("never")
    }
}

impl std::error::Error for Never {}

#[async_trait]
impl LlmAdapter for NullAdapter {
    type Error = Never;

    async fn responses_text<T>(
        &self,
        _input: ModelInput,
        _turn: TextTurn<T>,
    ) -> Result<TextTurnEventStream<T, Self::Error>, Self::Error>
    where
        T: Toolset,
    {
        Ok(Box::pin(futures::stream::empty()) as TextTurnEventStream<T, Self::Error>)
    }

    async fn responses_structured<T, O>(
        &self,
        _input: ModelInput,
        _turn: StructuredTurn<T, O>,
    ) -> Result<StructuredTurnEventStream<T, O, Self::Error>, Self::Error>
    where
        T: Toolset,
        O: StructuredOutput,
    {
        Ok(Box::pin(futures::stream::empty()) as StructuredTurnEventStream<T, O, Self::Error>)
    }

    async fn completion(
        &self,
        _request: CompletionRequest,
    ) -> Result<CompletionEventStream<Self::Error>, Self::Error> {
        Ok(Box::pin(futures::stream::empty()) as CompletionEventStream<Self::Error>)
    }

    async fn recover_usage(
        &self,
        _kind: StreamKind,
        _request_id: &str,
    ) -> Result<Option<Usage>, Self::Error> {
        Ok(None)
    }
}

struct NonCloneBudget;

#[derive(Debug)]
struct BudgetNever;

impl std::fmt::Display for BudgetNever {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("never")
    }
}

impl std::error::Error for BudgetNever {}

impl BudgetManager<AppMarker> for NonCloneBudget {
    type Error = BudgetNever;

    fn remaining(&self, _marker: &AppMarker) -> agents::Remaining {
        agents::Remaining::default()
    }

    fn reserve(
        &self,
        marker: &AppMarker,
        estimate: &UsageEstimate,
        request_budget: RequestBudget,
    ) -> Result<BudgetLease<AppMarker>, Self::Error> {
        Ok(BudgetLease::new(
            1,
            marker.clone(),
            *estimate,
            request_budget,
        ))
    }

    fn record_used(
        &self,
        _lease: BudgetLease<AppMarker>,
        _usage: Usage,
    ) -> Result<(), Self::Error> {
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
    let ctx: Context<AppMarker, _, _> = Context::new(budget, NullAdapter);
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
    _structured.config.reasoning = ReasoningParams {
        effort: Some(ReasoningEffort::Low),
        summary: None,
    };
    let _completion = CompletionRequest::builder()
        .model(ModelName::new("gpt-4.1-mini").unwrap())
        .prompt("hello")
        .budget(RequestBudget::from_tokens(128))
        .build();
    let _estimate = UsageEstimate::zero();
    let _marker = AppMarker;
    let _input = input;
}

#[test]
fn context_accepts_non_clone_budget_and_adapter() {
    let ctx: Context<AppMarker, _, _> = Context::new(NonCloneBudget, NullAdapter);
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
        })
        .unwrap();

    let result = reducer.into_result().unwrap();
    assert_eq!(result.tool_calls.len(), 1);
}

#[test]
fn context_rejects_invalid_model_input_before_adapter_call() {
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let ctx: Context<AppMarker, _, _> = Context::new(budget, NullAdapter);
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

    let err = futures::executor::block_on(ctx.responses_text(
        AppMarker,
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
