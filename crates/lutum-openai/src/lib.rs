pub mod adapter;
pub mod chat;
pub mod completion;
pub mod error;
pub mod responses;
pub mod sse;

pub use adapter::{
    ChatMessageJsonSerializer, FallbackSerializer, OpenAiAdapter, OpenAiHooks, OpenAiHooksSet,
    ResolveReasoningEffort, SelectOpenaiModel, SseEventRecoveryHook, SseHints,
};
pub use chat::ChatCompletionRequest;
pub use completion::CompletionRequest;
pub use error::OpenAiError;
pub use responses::{OpenAiCommittedTurn, OpenAiReasoningEffort, OpenAiTurnItem, ResponsesRequest};
