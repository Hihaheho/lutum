use std::{error::Error as StdError, pin::Pin};

use bytes::Bytes;
use futures::Stream;
#[cfg(feature = "reqwest")]
use futures::StreamExt;
use http::{HeaderMap, Method, StatusCode};
use lutum_protocol::{BoxError, RequestErrorDebugInfo};
use thiserror::Error;

#[cfg(not(target_family = "wasm"))]
pub type HttpByteStream =
    Pin<Box<dyn Stream<Item = Result<Bytes, HttpError>> + Send + Sync + 'static>>;
#[cfg(target_family = "wasm")]
pub type HttpByteStream = Pin<Box<dyn Stream<Item = Result<Bytes, HttpError>> + 'static>>;

#[derive(Debug)]
pub struct HttpRequest {
    pub method: Method,
    pub url: String,
    pub headers: HeaderMap,
    pub body: Option<Vec<u8>>,
}

pub struct HttpResponse {
    pub status: StatusCode,
    pub headers: HeaderMap,
    pub body: HttpByteStream,
}

#[derive(Debug, Error)]
#[error("{message}")]
pub struct HttpError {
    message: String,
    status: Option<StatusCode>,
    debug_info: RequestErrorDebugInfo,
    #[source]
    source: Option<BoxError>,
}

impl HttpError {
    pub fn message(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            status: None,
            debug_info: RequestErrorDebugInfo::default(),
            source: None,
        }
    }

    pub fn transport(source: impl StdError + Send + Sync + 'static) -> Self {
        let mut source_chain = Vec::new();
        let mut current = StdError::source(&source);
        while let Some(next) = current {
            source_chain.push(next.to_string());
            current = next.source();
        }
        let debug_info = RequestErrorDebugInfo {
            error_debug: format!("{source:?}"),
            source_chain,
            is_request: true,
            ..RequestErrorDebugInfo::default()
        };
        Self {
            message: source.to_string(),
            status: None,
            debug_info,
            source: Some(Box::new(source)),
        }
    }

    pub fn with_status(mut self, status: StatusCode) -> Self {
        self.status = Some(status);
        self
    }

    pub fn with_debug_info(mut self, debug_info: RequestErrorDebugInfo) -> Self {
        self.debug_info = debug_info;
        self
    }

    pub fn status(&self) -> Option<StatusCode> {
        self.status
    }

    pub fn debug_info(&self) -> &RequestErrorDebugInfo {
        &self.debug_info
    }
}

#[cfg(feature = "reqwest")]
impl From<reqwest::Error> for HttpError {
    fn from(source: reqwest::Error) -> Self {
        let mut source_chain = Vec::new();
        let mut current = StdError::source(&source);
        while let Some(next) = current {
            source_chain.push(next.to_string());
            current = next.source();
        }
        let status = source.status();
        let debug_info = RequestErrorDebugInfo {
            error_debug: format!("{source:?}"),
            source_chain,
            is_timeout: source.is_timeout(),
            #[cfg(not(target_family = "wasm"))]
            is_connect: source.is_connect(),
            #[cfg(target_family = "wasm")]
            is_connect: false,
            is_request: source.is_request(),
            is_body: source.is_body(),
            is_decode: source.is_decode(),
        };
        Self {
            message: source.to_string(),
            status,
            debug_info,
            source: Some(Box::new(source)),
        }
    }
}

#[cfg_attr(target_family = "wasm", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_family = "wasm"), async_trait::async_trait)]
pub trait HttpClient: lutum_protocol::MaybeSend + lutum_protocol::MaybeSync {
    async fn send(&self, request: HttpRequest) -> Result<HttpResponse, HttpError>;
}

#[cfg(feature = "reqwest")]
#[derive(Clone, Default)]
pub struct ReqwestHttpClient {
    inner: reqwest::Client,
}

#[cfg(feature = "reqwest")]
impl ReqwestHttpClient {
    pub fn new() -> Self {
        Self {
            inner: reqwest::Client::new(),
        }
    }

    pub fn with_client(client: reqwest::Client) -> Self {
        Self { inner: client }
    }
}

#[cfg(feature = "reqwest")]
#[cfg_attr(target_family = "wasm", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_family = "wasm"), async_trait::async_trait)]
impl HttpClient for ReqwestHttpClient {
    async fn send(&self, request: HttpRequest) -> Result<HttpResponse, HttpError> {
        let mut builder = self
            .inner
            .request(request.method, request.url)
            .headers(request.headers);
        if let Some(body) = request.body {
            builder = builder.body(body);
        }
        let response = builder.send().await.map_err(HttpError::from)?;
        let status = response.status();
        let headers = response.headers().clone();
        let body = response
            .bytes_stream()
            .map(|chunk| chunk.map_err(HttpError::from));
        Ok(HttpResponse {
            status,
            headers,
            body: Box::pin(body),
        })
    }
}
