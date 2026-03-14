use agents::{
    AssistantInputItem, AssistantTurnItem, BudgetLease, BudgetManager, CompletionEventStream,
    CompletionRequest, Context, ContextError, InputMessageRole, LlmAdapter, Marker, MessageContent,
    ModelInput, ModelInputItem, NonEmpty, RawJson, ReasoningConfig, RequestBudget,
    SharedPoolBudgetManager, SharedPoolBudgetOptions, StreamKind, StructuredOutput,
    StructuredTurnEventStream, StructuredTurnRequest, Temperature, TextTurnEventStream,
    TextTurnReducer, TextTurnRequest, ThinkingBudget, ToolCallError, ToolDef, ToolMode, ToolUse,
    Toolset, Usage, UsageEstimate,
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
        _turn: TextTurnRequest<T>,
    ) -> Result<TextTurnEventStream<T, Self::Error>, Self::Error>
    where
        T: Toolset,
    {
        Ok(Box::pin(futures::stream::empty()) as TextTurnEventStream<T, Self::Error>)
    }

    async fn responses_structured<T, O>(
        &self,
        _input: ModelInput,
        _turn: StructuredTurnRequest<T, O>,
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

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct WeatherArgs {
    city: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
enum Calls {
    Weather(WeatherArgs),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
enum Results {
    Weather { forecast: String },
}

#[derive(Clone, Copy, Debug, Default)]
struct Tools;

impl Toolset for Tools {
    type Call = Calls;
    type Result = Results;

    fn definitions() -> &'static [ToolDef<Self::Call, Self::Result>] {
        fn weather_args_schema() -> schemars::Schema {
            schemars::schema_for!(WeatherArgs)
        }

        static DEFS: [ToolDef<Calls, Results>; 1] =
            [ToolDef::new("weather", "Get weather", weather_args_schema)];
        &DEFS
    }

    fn parse_call(name: &str, arguments_json: &str) -> Result<Self::Call, ToolCallError> {
        match name {
            "weather" => serde_json::from_str(arguments_json)
                .map(Calls::Weather)
                .map_err(|source| ToolCallError::Deserialize {
                    name: name.to_string(),
                    source,
                }),
            _ => Err(ToolCallError::UnknownTool {
                name: name.to_string(),
            }),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct Summary {
    answer: String,
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
            RawJson::from_serializable(&Results::Weather {
                forecast: "sunny".into(),
            })
            .unwrap(),
        )),
    ]);

    let _text = TextTurnRequest::<Tools>::builder()
        .model("gpt-4.1")
        .tool_mode(ToolMode::AutoOnly(NonEmpty::one(
            Tools::definitions()[0].tool_ref(),
        )))
        .budget(RequestBudget::from_tokens(256))
        .build();
    let _structured = StructuredTurnRequest::<Tools, Summary>::builder()
        .model("gpt-4.1")
        .options(
            agents::ResponsesOptions::builder()
                .temperature(Temperature::try_from(0.3).unwrap())
                .max_output_tokens(512)
                .reasoning(
                    ReasoningConfig::builder()
                        .effort(ThinkingBudget::Low)
                        .build(),
                )
                .build(),
        )
        .build();
    let _completion = CompletionRequest::builder()
        .model("gpt-4.1-mini")
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
    let arguments = RawJson::parse("{\"city\":\"Tokyo\"}").unwrap();
    let invocation = agents::TypedToolInvocation {
        id: "call-1".into(),
        name: "weather".into(),
        call: Calls::Weather(WeatherArgs {
            city: "Tokyo".into(),
        }),
        arguments,
    };

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
    assert_eq!(result.typed_tool_calls.len(), 1);
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
        TextTurnRequest::<Tools>::new("gpt-4.1"),
        UsageEstimate::zero(),
    ));

    assert!(matches!(
        err,
        Err(ContextError::InvalidModelInput(
            agents::ModelInputValidationError::DuplicateToolUseId { .. }
        ))
    ));
}
