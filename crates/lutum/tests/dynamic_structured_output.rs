use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::executor::block_on;
use lutum::{
    AdapterStructuredCompletionRequest, AdapterStructuredOutputSpec, AdapterStructuredTurn,
    AdapterTextTurn, AgentError, AssistantTurnItem, AssistantTurnView, CommittedTurn,
    CompletionAdapter, CompletionEventStream, CompletionRequest, ErasedStructuredCompletionEvent,
    ErasedStructuredCompletionEventStream, ErasedStructuredTurnEvent,
    ErasedStructuredTurnEventStream, ErasedTextTurnEventStream, FinishReason, Lutum, ModelInput,
    OperationKind, RawJson, RequestExtensions, SharedPoolBudgetManager, SharedPoolBudgetOptions,
    StructuredTurnOutcome, TurnAdapter, Usage, UsageRecoveryAdapter,
};

#[derive(Clone, Default)]
struct SpyAdapter {
    structured_turns: Arc<Mutex<Vec<AdapterStructuredTurn>>>,
    structured_completions: Arc<Mutex<Vec<AdapterStructuredCompletionRequest>>>,
}

impl SpyAdapter {
    fn captured_structured_turns(&self) -> Vec<AdapterStructuredTurn> {
        self.structured_turns.lock().unwrap().clone()
    }

    fn captured_structured_completions(&self) -> Vec<AdapterStructuredCompletionRequest> {
        self.structured_completions.lock().unwrap().clone()
    }
}

#[cfg_attr(target_family = "wasm", async_trait(?Send))]
#[cfg_attr(not(target_family = "wasm"), async_trait)]
impl TurnAdapter for SpyAdapter {
    async fn text_turn(
        &self,
        _input: ModelInput,
        _turn: AdapterTextTurn,
    ) -> Result<ErasedTextTurnEventStream, AgentError> {
        unimplemented!("not needed for dynamic structured output tests")
    }

    async fn structured_turn(
        &self,
        _input: ModelInput,
        turn: AdapterStructuredTurn,
    ) -> Result<ErasedStructuredTurnEventStream, AgentError> {
        self.structured_turns.lock().unwrap().push(turn);
        let json = "{\"email\":\"user@example.com\"}";
        let committed: CommittedTurn =
            Arc::new(AssistantTurnView::from_items(&[AssistantTurnItem::Text(
                json.into(),
            )]));
        Ok(Box::pin(futures::stream::iter(vec![
            Ok(ErasedStructuredTurnEvent::Started {
                request_id: Some("spy-turn".into()),
                model: "spy-model".into(),
            }),
            Ok(ErasedStructuredTurnEvent::StructuredOutputChunk {
                json_delta: json.into(),
            }),
            Ok(ErasedStructuredTurnEvent::StructuredOutputReady(
                RawJson::parse(json).unwrap(),
            )),
            Ok(ErasedStructuredTurnEvent::Completed {
                request_id: Some("spy-turn".into()),
                finish_reason: FinishReason::Stop,
                usage: Usage::zero(),
                committed_turn: committed,
            }),
        ])))
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
        unimplemented!("not needed for dynamic structured output tests")
    }

    async fn structured_completion(
        &self,
        request: AdapterStructuredCompletionRequest,
        _extensions: &RequestExtensions,
    ) -> Result<ErasedStructuredCompletionEventStream, AgentError> {
        self.structured_completions.lock().unwrap().push(request);
        let json = "{\"email\":\"user@example.com\"}";
        Ok(Box::pin(futures::stream::iter(vec![
            Ok(ErasedStructuredCompletionEvent::Started {
                request_id: Some("spy-completion".into()),
                model: "spy-model".into(),
            }),
            Ok(ErasedStructuredCompletionEvent::StructuredOutputChunk {
                json_delta: json.into(),
            }),
            Ok(ErasedStructuredCompletionEvent::StructuredOutputReady(
                RawJson::parse(json).unwrap(),
            )),
            Ok(ErasedStructuredCompletionEvent::Completed {
                request_id: Some("spy-completion".into()),
                finish_reason: FinishReason::Stop,
                usage: Usage::zero(),
            }),
        ])))
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

fn ctx(adapter: &Arc<SpyAdapter>) -> Lutum {
    let turns: Arc<dyn TurnAdapter> = adapter.clone();
    let completions: Arc<dyn CompletionAdapter> = adapter.clone();
    Lutum::from_parts(
        turns,
        completions,
        SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
    )
}

fn schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "email": { "type": "string" }
        },
        "required": ["email"],
        "additionalProperties": false
    })
}

fn assert_output_spec(spec: &AdapterStructuredOutputSpec, expected_schema: &serde_json::Value) {
    assert_eq!(spec.schema_name, "runtime_contact");
    assert_eq!(&spec.schema, expected_schema);
}

#[test]
fn structured_turn_accepts_dynamic_output_schema() {
    let adapter = Arc::new(SpyAdapter::default());
    let ctx = ctx(&adapter);
    let expected_schema = schema();

    let result = block_on(async {
        ctx.structured_turn::<serde_json::Value>(ModelInput::new().user("extract email"))
            .output_schema("runtime_contact", expected_schema.clone())
            .collect()
            .await
            .unwrap()
    });

    assert_eq!(
        result.semantic,
        StructuredTurnOutcome::Structured(serde_json::json!({"email": "user@example.com"}))
    );
    let captured = adapter.captured_structured_turns();
    assert_eq!(captured.len(), 1);
    assert_output_spec(&captured[0].output, &expected_schema);
}

#[test]
fn structured_completion_accepts_dynamic_output_schema() {
    let adapter = Arc::new(SpyAdapter::default());
    let ctx = ctx(&adapter);
    let expected_schema = schema();

    let result = block_on(async {
        ctx.structured_completion::<serde_json::Value>("extract email")
            .output_schema("runtime_contact", expected_schema.clone())
            .collect()
            .await
            .unwrap()
    });

    assert_eq!(
        result.semantic,
        StructuredTurnOutcome::Structured(serde_json::json!({"email": "user@example.com"}))
    );
    let captured = adapter.captured_structured_completions();
    assert_eq!(captured.len(), 1);
    assert_output_spec(&captured[0].output, &expected_schema);
}
