use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

use async_trait::async_trait;
use futures::stream;
use lutum::{
    AdapterStructuredTurn, AdapterTextTurn, AgentError, AssistantTurn, AssistantTurnView,
    BudgetManager, ErasedStructuredTurnEventStream, ErasedTextTurnEvent, ErasedTextTurnEventStream,
    FinishReason, InputMessageRole, Lutum, ModelInput, ModelInputItem, RequestExtensions, Session,
    SharedPoolBudgetManager, SharedPoolBudgetOptions, TokenCount, TokenCounter, TurnAdapter, Usage,
    UsageEstimate,
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

#[derive(Clone)]
struct RecordingAdapter {
    calls: Arc<AtomicUsize>,
}

#[cfg_attr(target_family = "wasm", async_trait(?Send))]
#[cfg_attr(not(target_family = "wasm"), async_trait)]
impl TurnAdapter for RecordingAdapter {
    async fn text_turn(
        &self,
        _input: ModelInput,
        _turn: AdapterTextTurn,
    ) -> Result<ErasedTextTurnEventStream, AgentError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
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
        _turn: AdapterStructuredTurn,
    ) -> Result<ErasedStructuredTurnEventStream, AgentError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(Box::pin(stream::empty()))
    }
}

#[derive(Clone)]
enum CounterMode {
    Count(u64),
    Unsupported,
    Error,
}

#[derive(Clone)]
struct TestCounter {
    mode: CounterMode,
    seen_tools: Arc<Mutex<Vec<usize>>>,
}

#[cfg_attr(target_family = "wasm", async_trait(?Send))]
#[cfg_attr(not(target_family = "wasm"), async_trait)]
impl TokenCounter for TestCounter {
    async fn count_text_turn(
        &self,
        _input: &ModelInput,
        turn: &AdapterTextTurn,
    ) -> Result<Option<TokenCount>, AgentError> {
        self.seen_tools
            .lock()
            .unwrap()
            .push(turn.config.tools.len());
        match self.mode {
            CounterMode::Count(tokens) => Ok(Some(TokenCount::new(tokens))),
            CounterMode::Unsupported => Ok(None),
            CounterMode::Error => Err(AgentError::other(std::io::Error::other("count failed"))),
        }
    }
}

#[tokio::test]
async fn token_counter_overrides_token_estimate_before_reservation() {
    let budget = budget();
    let counter = Arc::new(TestCounter {
        mode: CounterMode::Count(7),
        seen_tools: Arc::new(Mutex::new(Vec::new())),
    });
    let adapter = Arc::new(RecordingAdapter {
        calls: Arc::new(AtomicUsize::new(0)),
    });
    let ctx = Lutum::new(adapter, budget.clone())
        .with_token_counter(counter.clone())
        .with_extension(UsageEstimate {
            input_tokens: 5,
            output_tokens: 11,
            total_tokens: 16,
            cost_micros_usd: 750,
        });

    let pending = ctx
        .text_turn(input())
        .max_output_tokens(3)
        .start()
        .await
        .unwrap();

    assert_eq!(budget.remaining(&RequestExtensions::new()).tokens, 82);
    assert_eq!(
        budget.remaining(&RequestExtensions::new()).cost_micros_usd,
        250
    );
    assert_eq!(*counter.seen_tools.lock().unwrap(), vec![0]);
    drop(pending);
}

#[tokio::test]
async fn unsupported_token_counter_preserves_usage_estimate_fallback() {
    let budget = budget();
    let counter = Arc::new(TestCounter {
        mode: CounterMode::Unsupported,
        seen_tools: Arc::new(Mutex::new(Vec::new())),
    });
    let adapter = Arc::new(RecordingAdapter {
        calls: Arc::new(AtomicUsize::new(0)),
    });
    let ctx = Lutum::new(adapter, budget.clone())
        .with_token_counter(counter)
        .with_extension(UsageEstimate {
            total_tokens: 42,
            ..UsageEstimate::zero()
        });

    let pending = ctx.text_turn(input()).start().await.unwrap();

    assert_eq!(budget.remaining(&RequestExtensions::new()).tokens, 58);
    drop(pending);
}

#[tokio::test]
async fn text_turn_count_tokens_counts_without_generation_or_budget_reservation() {
    let budget = budget();
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = Arc::new(TestCounter {
        mode: CounterMode::Count(7),
        seen_tools: Arc::new(Mutex::new(Vec::new())),
    });
    let adapter = Arc::new(RecordingAdapter {
        calls: Arc::clone(&calls),
    });
    let ctx = Lutum::new(adapter, budget.clone()).with_token_counter(counter);

    let turn = ctx.text_turn(input()).max_output_tokens(3);
    let count = turn.count_tokens().await.unwrap();

    assert_eq!(count, Some(TokenCount::new(7)));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(budget.remaining(&RequestExtensions::new()).tokens, 100);

    let result = turn.collect().await.map_err(|err| err.to_string()).unwrap();
    assert_eq!(result.assistant_text(), "ok");
}

#[tokio::test]
async fn session_count_tokens_does_not_mutate_session() {
    let counter = Arc::new(TestCounter {
        mode: CounterMode::Count(7),
        seen_tools: Arc::new(Mutex::new(Vec::new())),
    });
    let adapter = Arc::new(RecordingAdapter {
        calls: Arc::new(AtomicUsize::new(0)),
    });
    let ctx = Lutum::new(adapter, budget()).with_token_counter(counter);
    let mut session = Session::new();
    session.push_user("persistent");
    session.push_ephemeral_user("ephemeral");
    let input_items_before = session.input().items().len();

    let turn = session.text_turn(&ctx).max_output_tokens(3);
    let count = turn.count_tokens().await.unwrap();

    assert_eq!(count, Some(TokenCount::new(7)));
    drop(turn);
    assert_eq!(session.input().items().len(), input_items_before);
}

#[tokio::test]
async fn session_turn_builder_can_collect_after_count_tokens() {
    let counter = Arc::new(TestCounter {
        mode: CounterMode::Count(7),
        seen_tools: Arc::new(Mutex::new(Vec::new())),
    });
    let adapter = Arc::new(RecordingAdapter {
        calls: Arc::new(AtomicUsize::new(0)),
    });
    let ctx = Lutum::new(adapter, budget()).with_token_counter(counter);
    let mut session = Session::new();
    session.push_user("persistent");

    let turn = session.text_turn(&ctx).max_output_tokens(3);
    let count = turn.count_tokens().await.unwrap();
    let result = turn.collect().await.map_err(|err| err.to_string()).unwrap();

    assert_eq!(count, Some(TokenCount::new(7)));
    assert_eq!(result.assistant_text(), "ok");
}

#[tokio::test]
async fn token_count_error_prevents_generation_request() {
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = Arc::new(TestCounter {
        mode: CounterMode::Error,
        seen_tools: Arc::new(Mutex::new(Vec::new())),
    });
    let adapter = Arc::new(RecordingAdapter {
        calls: Arc::clone(&calls),
    });
    let ctx = Lutum::new(adapter, budget()).with_token_counter(counter);

    let err = ctx.text_turn(input()).collect().await.unwrap_err();

    assert!(err.to_string().contains("count failed"));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}
