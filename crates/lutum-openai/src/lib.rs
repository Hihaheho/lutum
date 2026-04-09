pub mod adapter;
pub mod completion;
pub mod error;
pub mod responses;
pub mod sse;

pub use adapter::{
    FallbackSerializer, OpenAiAdapter, OpenAiHooks, ResolveReasoningEffort, SelectOpenaiModel,
    SseEventRecoveryHook, SseHints,
};
pub use completion::CompletionRequest;
pub use error::OpenAiError;
pub use responses::{OpenAiCommittedTurn, OpenAiReasoningEffort, OpenAiTurnItem, ResponsesRequest};
