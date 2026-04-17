//! Built-in agentic tool-calling loop.
//!
//! Every tool-using agent needs the same outer loop:
//!
//! ```text
//! loop {
//!     outcome = session.text_turn().tools::<T>().collect_with(handler).await?;
//!     match outcome {
//!         Finished  => return usage,
//!         NeedsTools(round) => {
//!             results = dispatch all tool calls;
//!             round.commit(session, results);
//!         }
//!     }
//! }
//! ```
//!
//! [`AgentLoop`] encapsulates this pattern so application authors only need to
//! supply the dispatch closure that executes individual tool calls.

use std::{convert::Infallible, future::Future, marker::PhantomData};

use thiserror::Error;

use lutum_protocol::{ToolAvailability, ToolResult, Toolset, budget::Usage};

use crate::{
    HandlerContext, HandlerDirective, Session, TextStepOutcomeWithTools, TextTurnEventWithTools,
    TextTurnStateWithTools,
};

// ---------------------------------------------------------------------------
// Output / Error types
// ---------------------------------------------------------------------------

/// Output produced by a completed [`AgentLoop`].
#[derive(Clone, Debug, Default)]
pub struct AgentLoopOutput {
    /// Token usage accumulated across all rounds of the loop.
    pub usage: Usage,
    /// Number of rounds executed (including the final text-only round).
    pub rounds: usize,
}

/// Error returned by [`AgentLoop::run`].
#[derive(Debug, Error)]
pub enum AgentLoopError<E> {
    /// The model kept requesting tool calls beyond the configured round limit.
    #[error("reached {0}-round limit without a final answer")]
    RoundLimit(usize),
    /// The user-supplied dispatch closure returned an error.
    #[error("tool dispatch error: {0}")]
    Dispatch(E),
    /// The underlying turn collection failed.
    #[error("turn collection error: {0}")]
    Collect(String),
}

// ---------------------------------------------------------------------------
// AgentLoop builder
// ---------------------------------------------------------------------------

/// Builder for running a tool-calling agentic loop on a [`Session`].
///
/// Create via [`Session::agent_loop`].
///
/// # Example
///
/// ```rust,ignore
/// let output = session
///     .agent_loop::<MyTools>()
///     .max_rounds(20)
///     .on_text_delta(move |delta| { let _ = tx.send(delta); })
///     .run(|call| async move {
///         match call {
///             MyToolsCall::Foo(c) => {
///                 let result = do_foo(c.input()).await?;
///                 Ok(c.complete(result).unwrap())
///             }
///         }
///     })
///     .await?;
/// ```
pub struct AgentLoop<'s, T: Toolset> {
    session: &'s mut Session,
    max_rounds: usize,
    on_text_delta: Option<Box<dyn Fn(String) + Send + Sync>>,
    available: Option<ToolAvailability<T::Selector>>,
    _tools: PhantomData<T>,
}

impl<'s, T> AgentLoop<'s, T>
where
    T: Toolset,
{
    pub(crate) fn new(session: &'s mut Session) -> Self {
        Self {
            session,
            max_rounds: 20,
            on_text_delta: None,
            available: None,
            _tools: PhantomData,
        }
    }

    /// Maximum number of tool-call rounds before giving up with
    /// [`AgentLoopError::RoundLimit`]. Default: 20.
    pub fn max_rounds(mut self, n: usize) -> Self {
        self.max_rounds = n;
        self
    }

    /// Call `f` with each text delta produced by the model during the loop.
    ///
    /// Typically used to forward deltas to a UI or streaming channel:
    /// ```rust,ignore
    /// .on_text_delta(move |delta| { let _ = tx.send(delta); })
    /// ```
    pub fn on_text_delta<F>(mut self, f: F) -> Self
    where
        F: Fn(String) + Send + Sync + 'static,
    {
        self.on_text_delta = Some(Box::new(f));
        self
    }

    /// Restrict the available tools to exactly the listed selectors for every
    /// turn in this loop.
    pub fn available_tools(mut self, selectors: Vec<T::Selector>) -> Self {
        self.available = Some(ToolAvailability::Only(selectors));
        self
    }

    /// Expose the default-on toolset *plus* the listed selectors on every
    /// turn in this loop. Use this to temporarily re-enable variants marked
    /// `#[tool(off)]` / `#[toolset(off)]` (e.g. a loaded "skill").
    pub fn available_tools_default_plus(mut self, selectors: Vec<T::Selector>) -> Self {
        self.available = Some(ToolAvailability::DefaultPlus(selectors));
        self
    }

    /// Run the agentic loop.
    ///
    /// `dispatch` is called once per tool call. It receives the typed call enum
    /// value and must return a [`ToolResult`] (or an error that stops the loop).
    ///
    /// The loop terminates when:
    /// - the model produces a text-only response (success), or
    /// - the round limit is reached ([`AgentLoopError::RoundLimit`]), or
    /// - `dispatch` returns `Err` ([`AgentLoopError::Dispatch`]).
    pub async fn run<F, Fut, E>(self, dispatch: F) -> Result<AgentLoopOutput, AgentLoopError<E>>
    where
        F: Fn(T::ToolCall) -> Fut,
        Fut: Future<Output = Result<ToolResult, E>>,
        E: std::error::Error + 'static,
    {
        let AgentLoop {
            session,
            max_rounds,
            on_text_delta,
            available,
            ..
        } = self;

        // Wrap in Arc so it can be moved into the closure each round.
        let on_text_delta: Option<std::sync::Arc<dyn Fn(String) + Send + Sync>> =
            on_text_delta.map(std::sync::Arc::from);

        let mut output = AgentLoopOutput::default();

        for _round in 0..max_rounds {
            let cb = on_text_delta.clone();

            let mut turn = session.text_turn().tools::<T>();
            if let Some(ref availability) = available {
                turn = match availability {
                    ToolAvailability::All => turn,
                    ToolAvailability::Default => turn.available_tools_default_plus([]),
                    ToolAvailability::Only(selectors) => {
                        turn.available_tools(selectors.iter().copied())
                    }
                    ToolAvailability::DefaultPlus(selectors) => {
                        turn.available_tools_default_plus(selectors.iter().copied())
                    }
                };
            }

            let outcome = turn
                .collect_with(
                    move |event: &TextTurnEventWithTools<T>,
                          _cx: &HandlerContext<TextTurnStateWithTools<T>>|
                          -> Result<HandlerDirective, Infallible> {
                        if let TextTurnEventWithTools::TextDelta { delta } = event
                            && let Some(cb) = &cb
                        {
                            cb(delta.clone());
                        }
                        Ok(HandlerDirective::Continue)
                    },
                )
                .await
                .map_err(|e| AgentLoopError::Collect(e.to_string()))?;

            output.rounds += 1;

            match outcome {
                TextStepOutcomeWithTools::Finished(result) => {
                    accumulate_usage(&mut output.usage, result.usage);
                    return Ok(output);
                }
                TextStepOutcomeWithTools::NeedsTools(round) => {
                    accumulate_usage(&mut output.usage, round.usage);

                    let tool_calls = round.tool_calls.clone();
                    let mut tool_results = Vec::with_capacity(tool_calls.len());
                    for tool_call in tool_calls {
                        let result = dispatch(tool_call)
                            .await
                            .map_err(AgentLoopError::Dispatch)?;
                        tool_results.push(result);
                    }

                    round
                        .commit(session, tool_results)
                        .expect("tool result ordering should be valid");
                }
            }
        }

        Err(AgentLoopError::RoundLimit(max_rounds))
    }
}

fn accumulate_usage(acc: &mut Usage, u: Usage) {
    acc.input_tokens += u.input_tokens;
    acc.output_tokens += u.output_tokens;
    acc.total_tokens += u.total_tokens;
    acc.cost_micros_usd += u.cost_micros_usd;
    acc.cache_creation_tokens += u.cache_creation_tokens;
    acc.cache_read_tokens += u.cache_read_tokens;
}
