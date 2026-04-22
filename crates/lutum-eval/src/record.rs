use lutum_trace::{RawTraceSnapshot, TraceSnapshot};

use crate::{Score, Scored};

/// A scored evaluation result bundled with its trace.
///
/// `result` is `Err` when the eval or objective step fails. The trace is
/// always present regardless of outcome, making it usable for debugging both
/// successes and failures.
pub struct EvalRecord<R, E> {
    pub result: Result<Scored<R>, E>,
    pub trace: TraceSnapshot,
}

impl<R, E> EvalRecord<R, E> {
    pub fn passed(&self) -> bool {
        self.result.as_ref().is_ok_and(|s| s.score.value() > 0.0)
    }

    pub fn score(&self) -> Option<Score> {
        self.result.as_ref().ok().map(|s| s.score)
    }

    pub fn report(&self) -> Option<&R> {
        self.result.as_ref().ok().map(|s| &s.report)
    }

    pub fn error(&self) -> Option<&E> {
        self.result.as_ref().err()
    }

    pub fn into_raw(self, raw: RawTraceSnapshot) -> RawEvalRecord<R, E> {
        RawEvalRecord {
            result: self.result,
            trace: self.trace,
            raw,
        }
    }
}

/// A scored evaluation result bundled with its trace and raw protocol capture.
///
/// Useful for persisting full debug information when a trial fails.
pub struct RawEvalRecord<R, E> {
    pub result: Result<Scored<R>, E>,
    pub trace: TraceSnapshot,
    pub raw: RawTraceSnapshot,
}

impl<R, E> RawEvalRecord<R, E> {
    pub fn passed(&self) -> bool {
        self.result.as_ref().is_ok_and(|s| s.score.value() > 0.0)
    }

    pub fn score(&self) -> Option<Score> {
        self.result.as_ref().ok().map(|s| s.score)
    }

    pub fn report(&self) -> Option<&R> {
        self.result.as_ref().ok().map(|s| &s.report)
    }

    pub fn error(&self) -> Option<&E> {
        self.result.as_ref().err()
    }

    pub fn without_raw(self) -> EvalRecord<R, E> {
        EvalRecord {
            result: self.result,
            trace: self.trace,
        }
    }
}
