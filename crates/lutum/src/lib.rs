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
    Aggregate, AggregateInto, Chain, Finalize, FinalizeInto, FirstSuccess, HookReentrancyError,
    LutumHooks, ResolveUsageEstimate, ShortCircuit, Stateful,
};
pub use session::{CommitTurn, ToolRoundPlan, UncommittedToolRound};

pub use lutum_macros::{Toolset, hooks, impl_hook, nested_hooks, tool_fn, tool_input};
pub use lutum_protocol::{
    AdapterStructuredCompletionRequest, AdapterStructuredOutputSpec, AdapterStructuredTurn,
    AdapterTextTurn, AdapterToolChoice, AdapterToolDefinition, AdapterTurnConfig, AgentError,
    AssistantInputItem, AssistantTurn, AssistantTurnInputError, AssistantTurnItem,
    AssistantTurnView, BudgetLease, BudgetManager, CommittedTurn, CompletionAdapter,
    CompletionEvent, CompletionEventStream, CompletionOptions, CompletionReducer,
    CompletionReductionError, CompletionRequest, CompletionTurnResult, CompletionTurnState,
    EmptyNonEmptyError, EphemeralTurnView, ErasedStructuredCompletionEvent,
    ErasedStructuredCompletionEventStream,
    ErasedStructuredTurnEvent, ErasedStructuredTurnEventStream, ErasedTextTurnEvent,
    ErasedTextTurnEventStream, FinishReason, GenerationParams, HandledTool, HookableToolset,
    InputMessageRole, IntoToolResult, ItemView, MessageContent, ModelInput, ModelInputItem,
    ModelInputValidationError, ModelName, ModelNameError, NoToolSelector, NoTools, NonEmpty,
    OperationKind, REJECTED_TOOL_RESULT_PREFIX, RawJson, RejectedToolCall, RejectedToolSource,
    Remaining, RequestBudget, RequestExtensions, SharedPoolBudgetError, SharedPoolBudgetManager,
    SharedPoolBudgetOptions, StagedStructuredTurnResult, StagedStructuredTurnResultWithTools,
    StagedTextTurnResult, StagedTextTurnResultWithTools, StructuredCompletionEvent,
    StructuredCompletionEventStream, StructuredCompletionReducer,
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
    UsageEstimate, UsageRecoveryAdapter, assistant_json, find_tool_call_arguments,
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
