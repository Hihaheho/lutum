extern crate self as lutum;

pub mod budget {
    pub use lutum_protocol::budget::*;
}

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

pub use context::{
    CollectError, Context, ContextError, EventHandler, HandlerContext, HandlerDirective,
    PendingCompletion, PendingStructuredCompletion, PendingStructuredTurn, PendingTextTurn,
    StructuredTurnPartial,
};

#[cfg(feature = "claude")]
pub use lutum_claude::{
    BudgetTokensResolver, ClaudeAdapter, ClaudeCommittedTurn, ClaudeError, ClaudeTurnItem,
};
pub use lutum_macros::{Toolset, tool_fn, tool_input};
#[cfg(feature = "openai")]
pub use lutum_openai::{
    OpenAiAdapter, OpenAiError, OpenAiReasoningEffort, ReasoningEffortResolver,
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
    ModelNameError, ModelSelection, ModelSelector, NoToolSelector, NoTools, NonEmpty,
    OperationKind, RawJson, Remaining, RequestBudget, RequestExtensions, SharedPoolBudgetError,
    SharedPoolBudgetManager, SharedPoolBudgetOptions, StructuredCompletionEvent,
    StructuredCompletionEventStream, StructuredCompletionReducer,
    StructuredCompletionReductionError, StructuredCompletionRequest, StructuredCompletionResult,
    StructuredCompletionState, StructuredOutput, StructuredOutputSpec, StructuredTurn,
    StructuredTurnEvent, StructuredTurnEventStream, StructuredTurnOutcome, StructuredTurnReducer,
    StructuredTurnReductionError, StructuredTurnResult, StructuredTurnState, Temperature,
    TemperatureError, TextTurn, TextTurnEvent, TextTurnEventStream, TextTurnReducer,
    TextTurnReductionError, TextTurnResult, TextTurnState, ToolCallError, ToolCallId,
    ToolCallItemView, ToolCallWrapper, ToolDef, ToolExecutionError, ToolInput, ToolMetadata,
    ToolName, ToolPolicy, ToolResultItemView, ToolSelector, ToolUse, ToolUseError, Toolset,
    TurnAdapter, TurnConfig, TurnItemIter, TurnRole, TurnView, Usage, UsageEstimate,
    UsageRecoveryAdapter, assistant_json, find_tool_call_arguments,
};
pub use mock::{
    MockCompletionScenario, MockError, MockLlmAdapter, MockStructuredCompletionScenario,
    MockStructuredScenario, MockTextScenario, RawCompletionEvent, RawStructuredCompletionEvent,
    RawStructuredTurnEvent, RawTextTurnEvent,
};
pub use session::{
    Session, SessionDefaults, SessionPendingStructured, SessionPendingText, StructuredStepOutcome,
    TextStepOutcome, ToolRound, ToolRoundArityError,
};
