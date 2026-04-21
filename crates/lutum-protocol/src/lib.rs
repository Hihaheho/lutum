/// Re-export of [`send_wrapper::SendWrapper`] for WASM targets.
///
/// On `wasm32-unknown-unknown`, `reqwest` futures are `!Send` because they
/// hold `js_sys::JsFuture` (which wraps `Rc`).  Adapter crates that use
/// `#[async_trait]` (which requires `+ Send`) wrap their `!Send` async
/// bodies in this type.  WASM is single-threaded so the assertion is safe.
#[cfg(target_family = "wasm")]
pub use send_wrapper::SendWrapper;

pub mod budget;
pub mod conversation;
pub mod error;
pub mod extensions;
pub mod hooks;
pub mod llm;
pub mod reducer;
pub mod structured;
pub mod telemetry;
pub mod toolset;
pub mod transcript;

pub use budget::{
    BudgetLease, BudgetManager, Remaining, RequestBudget, SharedPoolBudgetError,
    SharedPoolBudgetManager, SharedPoolBudgetOptions, Usage, UsageEstimate,
};
pub use conversation::{
    AssistantInputItem, AssistantTurn, AssistantTurnInputError, AssistantTurnItem,
    EmptyNonEmptyError, InputMessageRole, MessageContent, ModelInput, ModelInputItem,
    ModelInputValidationError, NonEmpty, REJECTED_TOOL_RESULT_PREFIX, RawJson, ToolCallId,
    ToolMetadata, ToolName, ToolResult, UncommittedAssistantTurn,
};
pub use error::{
    AgentError, BoxError, NoToolsContractViolation, RequestFailure, RequestFailureKind,
};
pub use extensions::RequestExtensions;
pub use hooks::{HookReentrancyError, Stateful};
pub use llm::{
    AdapterStructuredCompletionRequest, AdapterStructuredOutputSpec, AdapterStructuredTurn,
    AdapterTextTurn, AdapterToolChoice, AdapterToolDefinition, AdapterTurnConfig, BackoffPolicy,
    CompletionAdapter, CompletionEvent, CompletionEventStream, CompletionOptions,
    CompletionRequest, ErasedStructuredCompletionEvent, ErasedStructuredCompletionEventStream,
    ErasedStructuredTurnEvent, ErasedStructuredTurnEventStream, ErasedTextTurnEvent,
    ErasedTextTurnEventStream, FinishReason, GenerationParams, ModelName, ModelNameError,
    OperationKind, RetryPolicy, StructuredCompletionEvent, StructuredCompletionEventStream,
    StructuredCompletionRequest, StructuredOutputSpec, StructuredTurn, StructuredTurnEvent,
    StructuredTurnEventStream, StructuredTurnEventStreamWithTools, StructuredTurnEventWithTools,
    Temperature, TemperatureError, TextTurn, TextTurnEvent, TextTurnEventStream,
    TextTurnEventStreamWithTools, TextTurnEventWithTools, TurnAdapter, TurnConfig,
    UsageRecoveryAdapter,
};
pub use reducer::{
    CompletionReducer, CompletionReductionError, CompletionTurnResult, CompletionTurnState,
    StagedStructuredTurnResult, StagedStructuredTurnResultWithTools, StagedTextTurnResult,
    StagedTextTurnResultWithTools, StructuredCompletionReducer, StructuredCompletionReductionError,
    StructuredCompletionResult, StructuredCompletionState, StructuredTurnOutcome,
    StructuredTurnReducer, StructuredTurnReducerWithTools, StructuredTurnReductionError,
    StructuredTurnResult, StructuredTurnResultWithTools, StructuredTurnState,
    StructuredTurnStateWithTools, TextTurnReducer, TextTurnReducerWithTools,
    TextTurnReductionError, TextTurnResult, TextTurnResultWithTools, TextTurnState,
    TextTurnStateWithTools, assistant_json, find_tool_call_arguments,
};
pub use structured::StructuredOutput;
pub use telemetry::{
    CollectErrorKind, ParseErrorStage, RAW_FIELD_API, RAW_FIELD_COLLECT_KIND, RAW_FIELD_ERROR,
    RAW_FIELD_ERROR_DEBUG, RAW_FIELD_EVENT_NAME, RAW_FIELD_IS_BODY, RAW_FIELD_IS_CONNECT,
    RAW_FIELD_IS_DECODE, RAW_FIELD_IS_REQUEST, RAW_FIELD_IS_TIMEOUT, RAW_FIELD_KIND,
    RAW_FIELD_OPERATION, RAW_FIELD_PARTIAL_SUMMARY, RAW_FIELD_PAYLOAD, RAW_FIELD_PROVIDER,
    RAW_FIELD_REQUEST_ERROR_KIND, RAW_FIELD_REQUEST_ID, RAW_FIELD_SEQUENCE, RAW_FIELD_SOURCE_CHAIN,
    RAW_FIELD_STAGE, RAW_FIELD_STATUS, RAW_KIND_COLLECT_ERROR, RAW_KIND_PARSE_ERROR,
    RAW_KIND_REQUEST, RAW_KIND_REQUEST_ERROR, RAW_KIND_STREAM_EVENT, RAW_TELEMETRY_TARGET,
    RawTelemetryConfig, RawTelemetryEmitter, RequestErrorDebugInfo, RequestErrorKind,
    emit_collect_error, emit_collect_error_enabled, operation_kind_name,
    raw_collect_errors_enabled,
};
pub use toolset::{
    ContinueSuggestionReason, HandledTool, HookableToolset, IntoToolResult, NoToolSelector,
    NoTools, RecoverableToolCallIssue, RecoverableToolCallIssueReason, RejectedToolCall,
    RejectedToolSource, ToolAvailability, ToolCallError, ToolCallWrapper, ToolConstraints,
    ToolDecision, ToolDef, ToolExecutionError, ToolHookOutcome, ToolHooks, ToolInput,
    ToolRequirement, ToolResultError, ToolSelector, Toolset,
};
pub use transcript::{
    AssistantTurnView, CommittedTurn, EphemeralTurnView, ItemView, ToolCallItemView,
    ToolResultItemView, TurnItemIter, TurnRole, TurnView,
};
