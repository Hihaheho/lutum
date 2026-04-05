pub mod adapter;
pub mod completion;
pub mod error;
pub mod responses;
pub mod sse;

pub use adapter::{
    FallbackSerializer, OpenAiAdapter, ResolveReasoningEffort, ResolveReasoningEffortHook,
    ResolveReasoningEffortRegistryExt, SelectOpenaiModel, SelectOpenaiModelHook,
    SelectOpenaiModelRegistryExt, SseEventRecoveryHook, SseHints,
};
pub use completion::CompletionRequest;
pub use error::OpenAiError;
pub use responses::{OpenAiCommittedTurn, OpenAiReasoningEffort, OpenAiTurnItem, ResponsesRequest};
