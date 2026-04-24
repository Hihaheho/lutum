extern crate self as lutum;

pub mod agent_loop;

pub mod budget {
    pub use lutum_protocol::budget::*;
}

mod builders;

mod context;

pub mod conversation {
    pub use lutum_protocol::conversation::*;
}

pub mod error {
    pub use lutum_protocol::error::*;
}

pub mod extensions {
    pub use lutum_protocol::extensions::*;
}

pub mod llm {
    pub use lutum_protocol::llm::*;
}

pub mod hooks;

pub mod mock;

pub mod reducer {
    pub use lutum_protocol::reducer::*;
}

pub mod session;

pub mod structured {
    pub use lutum_protocol::structured::*;
}

pub mod telemetry {
    pub use lutum_protocol::telemetry::*;
}

pub mod toolset {
    pub use lutum_protocol::toolset::*;
}

pub use agent_loop::{AgentLoop, AgentLoopError, AgentLoopOutput};
pub use builders::{
    Completion, StructuredCompletion, StructuredTurn, StructuredTurnWithTools, TextTurn,
    TextTurnWithTools,
};
pub use context::{
    CollectError, EventHandler, HandlerContext, HandlerDirective, Lutum, LutumError,
    PendingCompletion, PendingStructuredCompletion, PendingStructuredTurn,
    PendingStructuredTurnWithTools, PendingTextTurn, PendingTextTurnWithTools,
    StructuredTurnPartial, StructuredTurnPartialWithTools,
};
pub use hooks::{
    Aggregate, AggregateInto, Chain, DynAggregate, DynAggregateInto, DynChain, DynFinalize,
    DynFinalizeInto, Finalize, FinalizeInto, FirstSuccess, HookFuture, HookObject,
    HookReentrancyError, LutumHooks, LutumHooksSet, MaybeSend, MaybeSync, ResolveUsageEstimate,
    ShortCircuit, Stateful, boxed_hook_future,
};
pub use session::{CommitTurn, ToolRoundPlan, UncommittedToolRound};

pub use lutum_macros::{Toolset, hooks, impl_hook, impl_hooks, nested_hooks, tool_fn, tool_input};
pub use lutum_protocol::{
    AdapterStructuredCompletionRequest, AdapterStructuredOutputSpec, AdapterStructuredTurn,
    AdapterTextTurn, AdapterToolChoice, AdapterToolDefinition, AdapterTurnConfig, AgentError,
    AssistantInputItem, AssistantTurn, AssistantTurnInputError, AssistantTurnItem,
    AssistantTurnView, BackoffPolicy, BudgetLease, BudgetManager, CollectErrorKind, CommittedTurn,
    CompletionAdapter, CompletionEvent, CompletionEventStream, CompletionOptions,
    CompletionReducer, CompletionReductionError, CompletionRequest, CompletionTurnResult,
    CompletionTurnState, ContinueSuggestionReason, EmptyNonEmptyError, EphemeralTurnView,
    ErasedStructuredCompletionEvent, ErasedStructuredCompletionEventStream,
    ErasedStructuredTurnEvent, ErasedStructuredTurnEventStream, ErasedTextTurnEvent,
    ErasedTextTurnEventStream, FinishReason, GenerationParams, HandledTool, HookableToolset,
    InputMessageRole, IntoToolResult, ItemView, MessageContent, ModelInput, ModelInputItem,
    ModelInputValidationError, ModelName, ModelNameError, NoToolSelector, NoTools, NonEmpty,
    OperationKind, ParseErrorStage, RAW_FIELD_API, RAW_FIELD_COLLECT_KIND, RAW_FIELD_ERROR,
    RAW_FIELD_ERROR_DEBUG, RAW_FIELD_EVENT_NAME, RAW_FIELD_IS_BODY, RAW_FIELD_IS_CONNECT,
    RAW_FIELD_IS_DECODE, RAW_FIELD_IS_REQUEST, RAW_FIELD_IS_TIMEOUT, RAW_FIELD_KIND,
    RAW_FIELD_OPERATION, RAW_FIELD_PARTIAL_SUMMARY, RAW_FIELD_PAYLOAD, RAW_FIELD_PROVIDER,
    RAW_FIELD_REQUEST_ERROR_KIND, RAW_FIELD_REQUEST_ID, RAW_FIELD_SEQUENCE, RAW_FIELD_SOURCE_CHAIN,
    RAW_FIELD_STAGE, RAW_FIELD_STATUS, RAW_KIND_COLLECT_ERROR, RAW_KIND_PARSE_ERROR,
    RAW_KIND_REQUEST, RAW_KIND_REQUEST_ERROR, RAW_KIND_STREAM_EVENT, RAW_TELEMETRY_TARGET,
    REJECTED_TOOL_RESULT_PREFIX, RawJson, RawTelemetryConfig, RawTelemetryEmitter,
    RecoverableToolCallIssue, RecoverableToolCallIssueReason, RejectedToolCall, RejectedToolSource,
    Remaining, RequestBudget, RequestErrorDebugInfo, RequestErrorKind, RequestExtensions,
    RequestFailure, RequestFailureKind, RetryPolicy, SharedPoolBudgetError,
    SharedPoolBudgetManager, SharedPoolBudgetOptions, StagedStructuredTurnResult,
    StagedStructuredTurnResultWithTools, StagedTextTurnResult, StagedTextTurnResultWithTools,
    StructuredCompletionEvent, StructuredCompletionEventStream, StructuredCompletionReducer,
    StructuredCompletionReductionError, StructuredCompletionRequest, StructuredCompletionResult,
    StructuredCompletionState, StructuredOutput, StructuredOutputSpec, StructuredTurnEvent,
    StructuredTurnEventStream, StructuredTurnEventStreamWithTools, StructuredTurnEventWithTools,
    StructuredTurnOutcome, StructuredTurnReducer, StructuredTurnReducerWithTools,
    StructuredTurnReductionError, StructuredTurnResult, StructuredTurnResultWithTools,
    StructuredTurnState, StructuredTurnStateWithTools, Temperature, TemperatureError,
    TextTurnEvent, TextTurnEventStream, TextTurnEventStreamWithTools, TextTurnEventWithTools,
    TextTurnReducer, TextTurnReducerWithTools, TextTurnReductionError, TextTurnResult,
    TextTurnResultWithTools, TextTurnState, TextTurnStateWithTools, ToolAvailability,
    ToolCallError, ToolCallId, ToolCallItemView, ToolCallWrapper, ToolConstraints, ToolDecision,
    ToolDef, ToolExecutionError, ToolHookOutcome, ToolHooks, ToolInput, ToolMetadata, ToolName,
    ToolRequirement, ToolResult, ToolResultError, ToolResultItemView, ToolSelector, Toolset,
    TurnAdapter, TurnConfig, TurnItemIter, TurnRole, TurnView, UncommittedAssistantTurn, Usage,
    UsageEstimate, UsageRecoveryAdapter, assistant_json, emit_collect_error_enabled,
    find_tool_call_arguments, raw_collect_errors_enabled,
};
pub use mock::{
    MockCompletionScenario, MockError, MockLlmAdapter, MockStructuredCompletionScenario,
    MockStructuredScenario, MockTextScenario, RawCompletionEvent, RawStructuredCompletionEvent,
    RawStructuredTurnEvent, RawTextTurnEvent,
};
pub use session::{
    Session, SessionDefaults, StructuredStepOutcomeWithTools, TextStepOutcomeWithTools,
    ToolRoundArityError, ToolRoundCommitError,
};
