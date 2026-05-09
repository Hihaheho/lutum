use std::sync::Arc;

use lutum::{
    FinishReason, Lutum, MockLlmAdapter, MockTextScenario, RawTextTurnEvent, Session,
    SharedPoolBudgetManager, SharedPoolBudgetOptions, Usage,
};
use lutum_trace::{FieldValue, SpanNode};
use tracing::Instrument as _;

fn test_llm(request_id: &'static str, answer: &'static str) -> Lutum {
    let adapter = MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![
        Ok(RawTextTurnEvent::Started {
            request_id: Some(request_id.into()),
            model: "gpt-4.1-mini".into(),
        }),
        Ok(RawTextTurnEvent::TextDelta {
            delta: answer.into(),
        }),
        Ok(RawTextTurnEvent::Completed {
            request_id: Some(request_id.into()),
            finish_reason: FinishReason::Stop,
            usage: Usage {
                total_tokens: 1,
                ..Usage::zero()
            },
        }),
    ]));
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    Lutum::new(Arc::new(adapter), budget)
}

async fn run_session_turn(
    request_id: &'static str,
    prompt: &'static str,
    answer: &'static str,
) -> String {
    let llm = test_llm(request_id, answer);
    let mut session = Session::new();
    session.push_user(prompt);
    session
        .text_turn(&llm)
        .collect()
        .await
        .expect("mock text turn should collect")
        .assistant_text()
}

fn child_span_with_str_field<'a>(
    parent: &'a SpanNode,
    name: &str,
    key: &str,
    value: &str,
) -> &'a SpanNode {
    parent
        .children()
        .iter()
        .find(|child| {
            child.name == name && child.field(key) == Some(&FieldValue::Str(value.to_string()))
        })
        .unwrap_or_else(|| panic!("missing child span {name:?} with {key}={value:?}"))
}

fn single_llm_turn_with_request_id<'a>(agent: &'a SpanNode, request_id: &str) -> &'a SpanNode {
    let llm_turns = agent
        .children()
        .iter()
        .filter(|child| child.name == "llm_turn")
        .collect::<Vec<_>>();

    assert_eq!(llm_turns.len(), 1, "expected one direct llm_turn child");
    let llm_turn = llm_turns[0];
    assert_eq!(
        llm_turn.field("kind"),
        Some(&FieldValue::Str("text_turn".to_string()))
    );
    assert_eq!(
        llm_turn.field("request_id"),
        Some(&FieldValue::Str(request_id.to_string()))
    );
    llm_turn
}

#[tokio::test]
async fn parallel_sessions_trace_as_sibling_agents() {
    let collected = lutum_trace::test::collect(async {
        let workflow = tracing::info_span!(
            "workflow",
            lutum.capture = true,
            workflow = "parallel_sessions"
        );

        async {
            let planner = run_session_turn("req-planner", "plan", "planned").instrument(
                tracing::info_span!("planner_agent", lutum.capture = true, agent = "planner"),
            );
            let reviewer = run_session_turn("req-reviewer", "review", "reviewed").instrument(
                tracing::info_span!("reviewer_agent", lutum.capture = true, agent = "reviewer"),
            );

            tokio::join!(planner, reviewer)
        }
        .instrument(workflow)
        .await
    })
    .await;

    assert_eq!(
        collected.output,
        ("planned".to_string(), "reviewed".to_string())
    );

    let workflow = collected.trace.span("workflow").expect("workflow span");
    let planner = child_span_with_str_field(workflow, "planner_agent", "agent", "planner");
    let reviewer = child_span_with_str_field(workflow, "reviewer_agent", "agent", "reviewer");

    single_llm_turn_with_request_id(planner, "req-planner");
    single_llm_turn_with_request_id(reviewer, "req-reviewer");
}

#[tokio::test]
async fn sub_agent_session_traces_under_parent_agent() {
    let collected = lutum_trace::test::collect(async {
        let workflow =
            tracing::info_span!("workflow", lutum.capture = true, workflow = "sub_agent");

        async {
            let parent =
                tracing::info_span!("parent_agent", lutum.capture = true, agent = "parent");

            async {
                let parent_answer = run_session_turn("req-parent", "delegate", "delegating").await;

                let sub_agent =
                    tracing::info_span!("solver_agent", lutum.capture = true, agent = "sub");
                let sub_answer = run_session_turn("req-sub", "solve", "solved")
                    .instrument(sub_agent)
                    .await;

                (parent_answer, sub_answer)
            }
            .instrument(parent)
            .await
        }
        .instrument(workflow)
        .await
    })
    .await;

    assert_eq!(
        collected.output,
        ("delegating".to_string(), "solved".to_string())
    );

    let workflow = collected.trace.span("workflow").expect("workflow span");
    let parent = child_span_with_str_field(workflow, "parent_agent", "agent", "parent");
    let sub_agent = child_span_with_str_field(parent, "solver_agent", "agent", "sub");

    single_llm_turn_with_request_id(parent, "req-parent");
    single_llm_turn_with_request_id(sub_agent, "req-sub");
}
