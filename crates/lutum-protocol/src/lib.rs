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
pub use error::{AgentError, BoxError, NoToolsContractViolation};
pub use extensions::RequestExtensions;
pub use hooks::{HookReentrancyError, Stateful};
pub use llm::{
    AdapterStructuredCompletionRequest, AdapterStructuredOutputSpec, AdapterStructuredTurn,
    AdapterTextTurn, AdapterToolChoice, AdapterToolDefinition, AdapterTurnConfig,
    CompletionAdapter, CompletionEvent, CompletionEventStream, CompletionOptions,
    CompletionRequest, ErasedStructuredCompletionEvent, ErasedStructuredCompletionEventStream,
    ErasedStructuredTurnEvent, ErasedStructuredTurnEventStream, ErasedTextTurnEvent,
    ErasedTextTurnEventStream, FinishReason, GenerationParams, ModelName, ModelNameError,
    OperationKind, StructuredCompletionEvent, StructuredCompletionEventStream,
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
pub use toolset::{
    HandledTool, HookableToolset, IntoToolResult, NoToolSelector, NoTools, RejectedToolCall,
    RejectedToolSource, ToolAvailability, ToolCallError, ToolCallWrapper, ToolConstraints,
    ToolDecision, ToolDef, ToolExecutionError, ToolHookOutcome, ToolHooks, ToolInput,
    ToolRequirement, ToolResultError, ToolSelector, Toolset,
};
pub use transcript::{
    AssistantTurnView, CommittedTurn, ItemView, ToolCallItemView, ToolResultItemView, TurnItemIter,
    TurnRole, TurnView,
};
