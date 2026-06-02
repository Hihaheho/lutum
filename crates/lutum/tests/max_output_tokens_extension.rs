use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::stream;
use lutum::{
    AdapterStructuredCompletionRequest, AdapterStructuredTurn, AdapterTextTurn, AgentError,
    AssistantTurn, AssistantTurnView, BudgetManager, CompletionAdapter, CompletionEvent,
    CompletionEventStream, ErasedStructuredCompletionEvent, ErasedStructuredCompletionEventStream,
    ErasedStructuredTurnEvent, ErasedStructuredTurnEventStream, ErasedTextTurnEvent,
    ErasedTextTurnEventStream, FinishReason, GenerationParams, InputMessageRole, Lutum,
    MaxOutputTokens, ModelInput, ModelInputItem, RawJson, RequestExtensions, Session,
    SessionDefaults, SharedPoolBudgetManager, SharedPoolBudgetOptions, TokenCount, TokenCounter,
    TurnAdapter, Usage, UsageEstimate,
};

fn input() -> ModelInput {
    ModelInput::from_items(vec![ModelInputItem::text(InputMessageRole::User, "hello")])
}

fn budget() -> SharedPoolBudgetManager {
    SharedPoolBudgetManager::new(SharedPoolBudgetOptions {
        capacity_tokens: 100,
        capacity_cost_micros_usd: 1_000,
        stop_threshold_tokens: 0,
        stop_threshold_cost_micros_usd: 0,
    })
}

#[derive(Default)]
struct RecordingTurnAdapter {
    text_max_output_tokens: Mutex<Vec<Option<u32>>>,
    structured_max_output_tokens: Mutex<Vec<Option<u32>>>,
}

impl RecordingTurnAdapter {
    fn text_max_output_tokens(&self) -> Vec<Option<u32>> {
        self.text_max_output_tokens.lock().unwrap().clone()
    }

    fn structured_max_output_tokens(&self) -> Vec<Option<u32>> {
        self.structured_max_output_tokens.lock().unwrap().clone()
    }
}

#[cfg_attr(target_family = "wasm", async_trait(?Send))]
#[cfg_attr(not(target_family = "wasm"), async_trait)]
impl TurnAdapter for RecordingTurnAdapter {
    async fn text_turn(
        &self,
        _input: ModelInput,
        turn: AdapterTextTurn,
    ) -> Result<ErasedTextTurnEventStream, AgentError> {
        self.text_max_output_tokens
            .lock()
            .unwrap()
            .push(turn.config.generation.max_output_tokens);
        let assistant_turn = AssistantTurn::text("ok");
        Ok(Box::pin(stream::iter([
            Ok(ErasedTextTurnEvent::Started {
                request_id: None,
                model: "mock".to_string(),
            }),
            Ok(ErasedTextTurnEvent::TextDelta {
                delta: "ok".to_string(),
            }),
            Ok(ErasedTextTurnEvent::Completed {
                request_id: None,
                finish_reason: FinishReason::Stop,
                usage: Usage::zero(),
                committed_turn: Arc::new(AssistantTurnView::from_items(assistant_turn.items())),
            }),
        ])))
    }

    async fn structured_turn(
        &self,
        _input: ModelInput,
        turn: AdapterStructuredTurn,
    ) -> Result<ErasedStructuredTurnEventStream, AgentError> {
        self.structured_max_output_tokens
            .lock()
            .unwrap()
            .push(turn.config.generation.max_output_tokens);
        let assistant_turn = AssistantTurn::text(r#"{"ok":true}"#);
        Ok(Box::pin(stream::iter([
            Ok(ErasedStructuredTurnEvent::Started {
                request_id: None,
                model: "mock".to_string(),
            }),
            Ok(ErasedStructuredTurnEvent::StructuredOutputChunk {
                json_delta: r#"{"ok":true}"#.to_string(),
            }),
            Ok(ErasedStructuredTurnEvent::StructuredOutputReady(
                RawJson::parse(r#"{"ok":true}"#).unwrap(),
            )),
            Ok(ErasedStructuredTurnEvent::Completed {
                request_id: None,
                finish_reason: FinishReason::Stop,
                usage: Usage::zero(),
                committed_turn: Arc::new(AssistantTurnView::from_items(assistant_turn.items())),
            }),
        ])))
    }
}

#[derive(Default)]
struct RecordingCompletionAdapter {
    completion_max_output_tokens: Mutex<Vec<Option<u32>>>,
    structured_completion_max_output_tokens: Mutex<Vec<Option<u32>>>,
}

impl RecordingCompletionAdapter {
    fn completion_max_output_tokens(&self) -> Vec<Option<u32>> {
        self.completion_max_output_tokens.lock().unwrap().clone()
    }

    fn structured_completion_max_output_tokens(&self) -> Vec<Option<u32>> {
        self.structured_completion_max_output_tokens
            .lock()
            .unwrap()
            .clone()
    }
}

#[cfg_attr(target_family = "wasm", async_trait(?Send))]
#[cfg_attr(not(target_family = "wasm"), async_trait)]
impl CompletionAdapter for RecordingCompletionAdapter {
    async fn completion(
        &self,
        request: lutum::CompletionRequest,
        _extensions: &RequestExtensions,
    ) -> Result<CompletionEventStream, AgentError> {
        self.completion_max_output_tokens
            .lock()
            .unwrap()
            .push(request.options.max_output_tokens);
        Ok(Box::pin(stream::iter([
            Ok(CompletionEvent::Started {
                request_id: None,
                model: "mock".to_string(),
            }),
            Ok(CompletionEvent::TextDelta("ok".to_string())),
            Ok(CompletionEvent::Completed {
                request_id: None,
                finish_reason: FinishReason::Stop,
                usage: Usage::zero(),
            }),
        ])))
    }

    async fn structured_completion(
        &self,
        request: AdapterStructuredCompletionRequest,
        _extensions: &RequestExtensions,
    ) -> Result<ErasedStructuredCompletionEventStream, AgentError> {
        self.structured_completion_max_output_tokens
            .lock()
            .unwrap()
            .push(request.generation.max_output_tokens);
        Ok(Box::pin(stream::iter([
            Ok(ErasedStructuredCompletionEvent::Started {
                request_id: None,
                model: "mock".to_string(),
            }),
            Ok(ErasedStructuredCompletionEvent::StructuredOutputChunk {
                json_delta: r#"{"ok":true}"#.to_string(),
            }),
            Ok(ErasedStructuredCompletionEvent::StructuredOutputReady(
                RawJson::parse(r#"{"ok":true}"#).unwrap(),
            )),
            Ok(ErasedStructuredCompletionEvent::Completed {
                request_id: None,
                finish_reason: FinishReason::Stop,
                usage: Usage::zero(),
            }),
        ])))
    }
}

#[derive(Default)]
struct RecordingTokenCounter {
    text_max_output_tokens: Mutex<Vec<Option<u32>>>,
}

impl RecordingTokenCounter {
    fn text_max_output_tokens(&self) -> Vec<Option<u32>> {
        self.text_max_output_tokens.lock().unwrap().clone()
    }
}

#[cfg_attr(target_family = "wasm", async_trait(?Send))]
#[cfg_attr(not(target_family = "wasm"), async_trait)]
impl TokenCounter for RecordingTokenCounter {
    async fn count_text_turn(
        &self,
        _input: &ModelInput,
        turn: &AdapterTextTurn,
    ) -> Result<Option<TokenCount>, AgentError> {
        self.text_max_output_tokens
            .lock()
            .unwrap()
            .push(turn.config.generation.max_output_tokens);
        Ok(Some(TokenCount::new(7)))
    }
}

#[tokio::test]
async fn max_output_tokens_extension_applies_to_text_turn_budget_estimate() {
    let budget = budget();
    let counter = Arc::new(RecordingTokenCounter::default());
    let adapter = Arc::new(RecordingTurnAdapter::default());
    let ctx = Lutum::new(adapter, budget.clone())
        .with_token_counter(Arc::clone(&counter))
        .with_extension(UsageEstimate {
            input_tokens: 5,
            output_tokens: 11,
            total_tokens: 16,
            cost_micros_usd: 0,
        });

    let pending = ctx
        .text_turn(input())
        .ext(MaxOutputTokens::new(15))
        .start()
        .await
        .unwrap();

    assert_eq!(counter.text_max_output_tokens(), vec![Some(15)]);
    assert_eq!(budget.remaining(&RequestExtensions::new()).tokens, 78);
    drop(pending);
}

#[tokio::test]
async fn count_tokens_uses_lutum_default_max_output_tokens_extension() {
    let counter = Arc::new(RecordingTokenCounter::default());
    let adapter = Arc::new(RecordingTurnAdapter::default());
    let ctx = Lutum::new(adapter, budget())
        .with_token_counter(Arc::clone(&counter))
        .with_extension(MaxOutputTokens::new(17));

    let count = ctx.text_turn(input()).count_tokens().await.unwrap();

    assert_eq!(count, Some(TokenCount::new(7)));
    assert_eq!(counter.text_max_output_tokens(), vec![Some(17)]);
}

#[tokio::test]
async fn builder_max_output_tokens_wins_over_extension() {
    let adapter = Arc::new(RecordingTurnAdapter::default());
    let ctx = Lutum::new(Arc::clone(&adapter), budget());

    let result = ctx
        .text_turn(input())
        .ext(MaxOutputTokens::new(15))
        .max_output_tokens(3)
        .collect()
        .await
        .unwrap();

    assert_eq!(result.assistant_text(), "ok");
    assert_eq!(adapter.text_max_output_tokens(), vec![Some(3)]);
}

#[tokio::test]
async fn structured_turn_uses_max_output_tokens_extension() {
    let adapter = Arc::new(RecordingTurnAdapter::default());
    let ctx = Lutum::new(Arc::clone(&adapter), budget());

    let result = ctx
        .structured_turn::<serde_json::Value>(input())
        .ext(MaxOutputTokens::new(18))
        .collect()
        .await
        .unwrap();

    assert_eq!(
        result.semantic,
        lutum::StructuredTurnOutcome::Structured(serde_json::json!({"ok": true}))
    );
    assert_eq!(adapter.structured_max_output_tokens(), vec![Some(18)]);
}

#[tokio::test]
async fn session_turn_precedence_is_request_then_session_then_lutum_default() {
    let adapter = Arc::new(RecordingTurnAdapter::default());
    let ctx = Lutum::new(Arc::clone(&adapter), budget()).with_extension(MaxOutputTokens::new(6));
    let defaults = SessionDefaults {
        generation: GenerationParams {
            max_output_tokens: Some(8),
            ..GenerationParams::default()
        },
        ..SessionDefaults::default()
    };

    let mut request_session = Session::new().with_defaults(defaults.clone());
    request_session.push_user("hello");
    request_session
        .text_turn()
        .ext(MaxOutputTokens::new(12))
        .collect(&ctx)
        .await
        .unwrap();

    let mut default_session = Session::new().with_defaults(defaults);
    default_session.push_user("hello");
    default_session.text_turn().collect(&ctx).await.unwrap();

    let mut lutum_default_session = Session::new();
    lutum_default_session.push_user("hello");
    lutum_default_session
        .text_turn()
        .collect(&ctx)
        .await
        .unwrap();

    assert_eq!(
        adapter.text_max_output_tokens(),
        vec![Some(12), Some(8), Some(6)]
    );
}

#[tokio::test]
async fn completion_requests_use_max_output_tokens_extension() {
    let turns = Arc::new(RecordingTurnAdapter::default());
    let completions = Arc::new(RecordingCompletionAdapter::default());
    let turn_adapter: Arc<dyn TurnAdapter> = turns;
    let completion_adapter: Arc<dyn CompletionAdapter> = completions.clone();
    let ctx = Lutum::from_parts(turn_adapter, completion_adapter, budget());

    let completion = ctx
        .completion("hello")
        .ext(MaxOutputTokens::new(21))
        .collect()
        .await
        .unwrap();
    let structured = ctx
        .structured_completion::<serde_json::Value>("hello")
        .ext(MaxOutputTokens::new(22))
        .collect()
        .await
        .unwrap();

    assert_eq!(completion.text, "ok");
    assert_eq!(
        structured.semantic,
        lutum::StructuredTurnOutcome::Structured(serde_json::json!({"ok": true}))
    );
    assert_eq!(completions.completion_max_output_tokens(), vec![Some(21)]);
    assert_eq!(
        completions.structured_completion_max_output_tokens(),
        vec![Some(22)]
    );
}

#[test]
fn max_output_tokens_extension_newtype_round_trips() {
    let max_output_tokens = MaxOutputTokens::from(42);

    assert_eq!(max_output_tokens.get(), 42);
}
