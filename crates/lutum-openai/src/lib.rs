pub mod adapter;
pub mod chat;
pub mod completion;
pub mod error;
pub mod responses;
pub mod sse;
pub mod transport;

pub use adapter::{
    ChatMessageJsonSerializer, FallbackSerializer, HeadersCustomizer, OpenAiAdapter, OpenAiHooks,
    OpenAiHooksSet, ResolveReasoningEffort, SelectOpenaiModel, SseEventRecoveryHook, SseHints,
};
pub use chat::ChatCompletionRequest;
pub use completion::CompletionRequest;
pub use error::OpenAiError;
pub use responses::{OpenAiCommittedTurn, OpenAiReasoningEffort, OpenAiTurnItem, ResponsesRequest};
#[cfg(feature = "reqwest")]
pub use transport::ReqwestHttpClient;
pub use transport::{HttpByteStream, HttpClient, HttpError, HttpRequest, HttpResponse};
