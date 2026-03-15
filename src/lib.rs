extern crate self as agents;

pub mod budget;
pub mod context;
pub mod conversation;
pub mod llm;
pub mod marker;
pub mod mock;
pub mod providers;
pub mod reducer;
pub mod structured;
pub mod toolset;

pub use agents_macros::{Toolset, tool_fn, tool_input};
pub use budget::{
    BudgetLease, BudgetManager, Remaining, RequestBudget, SharedPoolBudgetError,
    SharedPoolBudgetManager, SharedPoolBudgetOptions, Usage, UsageEstimate,
};
pub use context::{
    CollectError, Context, ContextError, EventHandler, HandlerContext, HandlerDirective,
    PendingCompletion, PendingStructuredTurn, PendingTextTurn,
};
pub use conversation::{
    AssistantInputItem, AssistantTurn, AssistantTurnInputError, AssistantTurnItem,
    EmptyNonEmptyError, InputMessageRole, MessageContent, ModelInput, ModelInputItem,
    ModelInputValidationError, NonEmpty, RawJson, ToolCallId, ToolMetadata, ToolName, ToolUse,
};
pub use llm::{
    CompletionEvent, CompletionEventStream, CompletionOptions, CompletionRequest, FinishReason,
    LlmAdapter, ReasoningConfig, ReasoningEffort, ReasoningSummary, ResponsesOptions, StreamKind,
    StructuredTurnEvent, StructuredTurnEventStream, StructuredTurnRequest, Temperature,
    TemperatureError, TextTurnEvent, TextTurnEventStream, TextTurnRequest, ThinkingBudget,
};
pub use marker::Marker;
pub use mock::{
    MockCompletionScenario, MockError, MockLlmAdapter, MockStructuredScenario, MockTextScenario,
    RawCompletionEvent, RawStructuredTurnEvent, RawTextTurnEvent,
};
pub use providers::openai::{OpenAiAdapter, OpenAiError};
pub use reducer::{
    CompletionReducer, CompletionReductionError, CompletionTurnResult, CompletionTurnState,
    StructuredTurnOutcome, StructuredTurnReducer, StructuredTurnReductionError,
    StructuredTurnResult, StructuredTurnState, TextTurnReducer, TextTurnReductionError,
    TextTurnResult, TextTurnState, assistant_json, find_tool_call_arguments,
};
pub use structured::StructuredOutput;
pub use toolset::{
    NoTools, SupportsTool, ToolCallError, ToolCallWrapper, ToolDef, ToolExecutionError, ToolInput,
    ToolMode, ToolRef, ToolSelection, ToolSubset, ToolSubsetMarker, ToolUseError, Toolset,
};

#[macro_export]
macro_rules! tools {
    ($($tool:ty),+ $(,)?) => {
        $crate::ToolSubsetMarker::<($($tool,)+)>::new()
    };
}
