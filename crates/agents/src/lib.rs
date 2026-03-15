extern crate self as agents;

pub mod budget {
    pub use agents_protocol::budget::*;
}

pub mod context;

pub mod conversation {
    pub use agents_protocol::conversation::*;
}

pub mod llm {
    pub use agents_protocol::llm::*;
}

pub mod marker {
    pub use agents_protocol::marker::*;
}

pub mod mock;

#[cfg(feature = "openai")]
pub use agents_openai as openai;

pub mod reducer {
    pub use agents_protocol::reducer::*;
}

pub mod session;

pub mod structured {
    pub use agents_protocol::structured::*;
}

pub mod toolset {
    pub use agents_protocol::toolset::*;
}

pub use agents_macros::{Toolset, tool_fn, tool_input};
#[cfg(feature = "openai")]
pub use agents_openai::{OpenAiAdapter, OpenAiError};
pub use agents_protocol::{
    AssistantInputItem, AssistantTurn, AssistantTurnInputError, AssistantTurnItem, BudgetLease,
    BudgetManager, CompletionEvent, CompletionEventStream, CompletionOptions, CompletionReducer,
    CompletionReductionError, CompletionRequest, CompletionTurnResult, CompletionTurnState,
    EmptyNonEmptyError, FinishReason, GenerationParams, InputMessageRole, LlmAdapter, Marker,
    MessageContent, ModelInput, ModelInputItem, ModelInputValidationError, ModelName,
    ModelNameError, NoToolSelector, NoTools, NonEmpty, RawJson, ReasoningEffort, ReasoningParams,
    ReasoningSummary, Remaining, RequestBudget, SharedPoolBudgetError, SharedPoolBudgetManager,
    SharedPoolBudgetOptions, StreamKind, StructuredOutput, StructuredOutputSpec, StructuredTurn,
    StructuredTurnEvent, StructuredTurnEventStream, StructuredTurnOutcome, StructuredTurnReducer,
    StructuredTurnReductionError, StructuredTurnResult, StructuredTurnState, Temperature,
    TemperatureError, TextTurn, TextTurnEvent, TextTurnEventStream, TextTurnReducer,
    TextTurnReductionError, TextTurnResult, TextTurnState, ToolCallError, ToolCallId,
    ToolCallWrapper, ToolDef, ToolExecutionError, ToolInput, ToolMetadata, ToolName, ToolPolicy,
    ToolSelector, ToolUse, ToolUseError, Toolset, TurnConfig, Usage, UsageEstimate, assistant_json,
    find_tool_call_arguments,
};
pub use context::{
    CollectError, Context, ContextError, EventHandler, HandlerContext, HandlerDirective,
    PendingCompletion, PendingStructuredTurn, PendingTextTurn,
};
pub use mock::{
    MockCompletionScenario, MockError, MockLlmAdapter, MockStructuredScenario, MockTextScenario,
    RawCompletionEvent, RawStructuredTurnEvent, RawTextTurnEvent,
};
pub use session::{
    Session, SessionDefaults, SessionPendingStructured, SessionPendingText, StructuredStepOutcome,
    TextStepOutcome, ToolRound,
};
