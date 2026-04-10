use lutum_eval::{CombineError, EvalRunError, ProbeRunError, ProbeScoreError, ScoreEvalError};
use lutum_trace::CaptureError;
use thiserror::Error;

#[derive(Debug, Error)]
enum EvalLeafError {
    #[error("eval leaf")]
    Eval,
}

#[derive(Debug, Error)]
enum ObjectiveLeafError {
    #[error("objective leaf")]
    Objective,
}

fn chain_messages(error: &(dyn std::error::Error + 'static)) -> Vec<String> {
    let mut messages = Vec::new();
    let mut current = error.source();

    while let Some(source) = current {
        messages.push(source.to_string());
        current = source.source();
    }

    messages
}

#[test]
fn score_eval_error_exposes_the_wrapped_source() {
    let error = ScoreEvalError::<EvalLeafError, ObjectiveLeafError>::Eval(EvalLeafError::Eval);

    assert_eq!(chain_messages(&error), vec!["eval leaf"]);
}

#[test]
fn eval_run_error_maps_to_score_eval_error() {
    let inner: ScoreEvalError<EvalLeafError, ObjectiveLeafError> =
        EvalRunError::Eval(EvalLeafError::Eval).into();
    assert!(matches!(inner, ScoreEvalError::Eval(EvalLeafError::Eval)));
}

#[test]
fn score_eval_error_capture_variant() {
    let error = ScoreEvalError::<EvalLeafError, ObjectiveLeafError>::Capture(
        CaptureError::CaptureLayerNotInstalled,
    );
    assert_eq!(
        chain_messages(&error),
        vec![
            "lutum-trace capture layer is not installed on the active tracing subscriber (use lutum_trace::subscriber::otel_capture_subscriber or equivalent)"
        ]
    );
}

#[test]
fn score_eval_error_exposes_the_objective_source() {
    let error = ScoreEvalError::<EvalLeafError, ObjectiveLeafError>::Objective(
        ObjectiveLeafError::Objective,
    );

    assert_eq!(chain_messages(&error), vec!["objective leaf"]);
}

#[test]
fn combine_error_exposes_the_failing_side_as_source() {
    let error = CombineError::<EvalLeafError, ObjectiveLeafError>::Left(EvalLeafError::Eval);

    assert_eq!(chain_messages(&error), vec!["eval leaf"]);
}

#[test]
fn probe_score_error_keeps_the_full_nested_chain() {
    let error = ProbeScoreError::<EvalLeafError, ObjectiveLeafError>::Probe(ProbeRunError::Probe(
        EvalLeafError::Eval,
    ));

    assert_eq!(
        chain_messages(&error),
        vec!["probe failed: eval leaf", "eval leaf"]
    );
}
