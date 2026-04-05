extern crate self as lutum;

pub mod budget {
    pub use lutum_protocol::budget::*;
}

mod builders;

pub mod context;

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

#[cfg(feature = "claude")]
pub use lutum_claude as claude;
#[cfg(feature = "openai")]
pub use lutum_openai as openai;

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

pub use builders::{
    Completion, StructuredCompletion, StructuredTurn, StructuredTurnWithTools, TextTurn,
    TextTurnWithTools,
};
pub use context::{
    CollectError, Context, ContextError, EventHandler, HandlerContext, HandlerDirective,
    PendingCompletion, PendingStructuredCompletion, PendingStructuredTurn,
    PendingStructuredTurnWithTools, PendingTextTurn, PendingTextTurnWithTools,
    StructuredTurnPartial, StructuredTurnPartialWithTools,
};
pub use hooks::{
    HookReentrancyError, HookRegistry, ResolveUsageEstimate, ResolveUsageEstimateHook,
    ResolveUsageEstimateRegistryExt, Stateful,
};

#[cfg(feature = "claude")]
pub use lutum_claude::{
    ClaudeAdapter, ClaudeCommittedTurn, ClaudeError, ClaudeTurnItem, ResolveBudgetTokens,
    ResolveBudgetTokensHook, ResolveBudgetTokensRegistryExt, SelectClaudeModel,
    SelectClaudeModelHook, SelectClaudeModelRegistryExt,
};
pub use lutum_macros::{Toolset, def_hook, hook, tool_fn, tool_input};
#[cfg(feature = "openai")]
pub use lutum_openai::{
    OpenAiAdapter, OpenAiError, OpenAiReasoningEffort, ResolveReasoningEffort,
    ResolveReasoningEffortHook, ResolveReasoningEffortRegistryExt, SelectOpenaiModel,
    SelectOpenaiModelHook, SelectOpenaiModelRegistryExt,
};
pub use lutum_protocol::{
    AdapterStructuredCompletionRequest, AdapterStructuredOutputSpec, AdapterStructuredTurn,
    AdapterTextTurn, AdapterToolChoice, AdapterToolDefinition, AdapterTurnConfig, AgentError,
    AssistantInputItem, AssistantTurn, AssistantTurnInputError, AssistantTurnItem,
    AssistantTurnView, BudgetLease, BudgetManager, CommittedTurn, CompletionAdapter,
    CompletionEvent, CompletionEventStream, CompletionOptions, CompletionReducer,
    CompletionReductionError, CompletionRequest, CompletionTurnResult, CompletionTurnState,
    EmptyNonEmptyError, ErasedStructuredCompletionEvent, ErasedStructuredCompletionEventStream,
    ErasedStructuredTurnEvent, ErasedStructuredTurnEventStream, ErasedTextTurnEvent,
    ErasedTextTurnEventStream, FinishReason, GenerationParams, InputMessageRole, ItemView,
    MessageContent, ModelInput, ModelInputItem, ModelInputValidationError, ModelName,
    ModelNameError, NoToolSelector, NoTools, NonEmpty, OperationKind, RawJson, Remaining,
    RequestBudget, RequestExtensions, SharedPoolBudgetError, SharedPoolBudgetManager,
    SharedPoolBudgetOptions, StructuredCompletionEvent, StructuredCompletionEventStream,
    StructuredCompletionReducer, StructuredCompletionReductionError, StructuredCompletionRequest,
    StructuredCompletionResult, StructuredCompletionState, StructuredOutput, StructuredOutputSpec,
    StructuredTurnEvent, StructuredTurnEventStream, StructuredTurnEventStreamWithTools,
    StructuredTurnEventWithTools, StructuredTurnOutcome, StructuredTurnReducer,
    StructuredTurnReducerWithTools, StructuredTurnReductionError, StructuredTurnResult,
    StructuredTurnResultWithTools, StructuredTurnState, StructuredTurnStateWithTools, Temperature,
    TemperatureError, TextTurnEvent, TextTurnEventStream, TextTurnEventStreamWithTools,
    TextTurnEventWithTools, TextTurnReducer, TextTurnReducerWithTools, TextTurnReductionError,
    TextTurnResult, TextTurnResultWithTools, TextTurnState, TextTurnStateWithTools, ToolCallError,
    ToolCallId, ToolCallItemView, ToolCallWrapper, ToolDef, ToolExecutionError, ToolInput,
    ToolMetadata, ToolName, ToolPolicy, ToolResultItemView, ToolSelector, ToolUse, ToolUseError,
    Toolset, TurnAdapter, TurnConfig, TurnItemIter, TurnRole, TurnView, Usage, UsageEstimate,
    UsageRecoveryAdapter, assistant_json, find_tool_call_arguments,
};
pub use mock::{
    MockCompletionScenario, MockError, MockLlmAdapter, MockStructuredCompletionScenario,
    MockStructuredScenario, MockTextScenario, RawCompletionEvent, RawStructuredCompletionEvent,
    RawStructuredTurnEvent, RawTextTurnEvent,
};
pub use session::{
    Session, SessionDefaults, StructuredStepOutcomeWithTools, TextStepOutcomeWithTools, ToolRound,
    ToolRoundArityError,
};
