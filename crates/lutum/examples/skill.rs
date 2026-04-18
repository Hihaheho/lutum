//! Skill pattern — progressive disclosure of large toolsets via load/unload.
//!
//! The "skill" pattern lets an agent work with a catalogue of potentially
//! large, context-expensive capability bundles (a "skill") without pouring
//! every instruction and every tool into every turn. Skills are declared with
//! `#[toolset(off)]` on a nested toolset variant: those tools are hidden from
//! the model until the agent explicitly loads them. While a skill is loaded,
//! its instructions are injected ephemerally and its tools are exposed via
//! [`TextTurnWithTools::available_tools_default_plus`].
//!
//! Design:
//!
//! 1. **Default-off variants.** `#[toolset(off)]` and `#[tool(off)]` mark
//!    variants as hidden by default. They do not appear in
//!    [`Toolset::default_selectors`] and so are invisible to the model under
//!    [`ToolAvailability::Default`] / [`ToolAvailability::DefaultPlus`].
//!
//! 2. **Explicit load/unload.** `load_skill` / `unload_skill` are **hand-
//!    written** meta-tools. The derive intentionally does NOT generate them
//!    — which skills exist, what loading means, and when it is allowed are
//!    application decisions.
//!
//! 3. **Ephemeral skill bodies.** While a skill is loaded, a developer
//!    message carrying the skill's full instructions is re-pushed before each
//!    turn via [`Session::push_ephemeral_developer`]. Ephemeral items are
//!    included in the next request and then stripped — they never bloat the
//!    persistent transcript.
//!
//! 4. **Persistent load markers.** After a successful `load_skill`, a normal
//!    developer message `[system] skill ... loaded` is pushed. It stays in
//!    the transcript so later turns can see the timeline of what was loaded
//!    when.
//!
//! 5. **Catalogue in tool description.** The `load_skill` tool's description
//!    is rewritten each turn via [`TextTurnWithTools::describe_tool`] to list
//!    available skills and their short summaries. This costs one description
//!    per turn but keeps the catalogue in sync with actual state.
//!
//! This example drives everything with [`MockLlmAdapter`] so it is
//! deterministic and requires no network.
use std::{collections::BTreeSet, sync::Arc};

use futures::executor::block_on;
use lutum::{
    FinishReason, MockLlmAdapter, MockTextScenario, RawTextTurnEvent, Session,
    SharedPoolBudgetManager, SharedPoolBudgetOptions, TextStepOutcomeWithTools, Usage,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

// ── Skill 1: code_review ──────────────────────────────────────────────────

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct Report {
    text: String,
}

/// Run static-analysis lints on the current repository.
#[lutum::tool_input(name = "run_lints", output = Report)]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct RunLintsArgs {}

/// Summarize the current git diff.
#[lutum::tool_input(name = "diff_summary", output = Report)]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct DiffSummaryArgs {}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
enum CodeReviewTools {
    RunLints(RunLintsArgs),
    DiffSummary(DiffSummaryArgs),
}

// ── Skill 2: web ──────────────────────────────────────────────────────────

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct Page {
    text: String,
}

/// Fetch a URL and return the body as plain text.
#[lutum::tool_input(name = "fetch_url", output = Page)]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct FetchUrlArgs {
    url: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
enum WebTools {
    FetchUrl(FetchUrlArgs),
}

// ── Meta tools: load_skill / unload_skill ─────────────────────────────────

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct SkillAck {
    ok: bool,
    message: String,
}

/// Load a skill by name. After loading, the skill's tools become available and
/// its instructions are included in every subsequent turn until `unload_skill`
/// is called.
#[lutum::tool_input(name = "load_skill", output = SkillAck)]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct LoadSkillArgs {
    /// Exactly one of the skill names listed in this tool's description.
    name: String,
}

/// Unload a previously loaded skill. Its tools and instructions stop being
/// offered.
#[lutum::tool_input(name = "unload_skill", output = SkillAck)]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct UnloadSkillArgs {
    name: String,
}

// ── Top-level toolset ─────────────────────────────────────────────────────
//
// `#[toolset(off)]` marks a nested skill as hidden by default. The meta-tools
// `load_skill` / `unload_skill` are regular (default-on) variants — nothing
// about loading is derived automatically.

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
enum AppTools {
    LoadSkill(LoadSkillArgs),
    UnloadSkill(UnloadSkillArgs),
    #[toolset(off)]
    CodeReview(CodeReviewTools),
    #[toolset(off)]
    Web(WebTools),
}

// ── Skill registry (static) ───────────────────────────────────────────────

struct SkillDef {
    name: &'static str,
    summary: &'static str,
    body: &'static str,
    selectors: fn() -> Vec<AppToolsSelector>,
}

const SKILLS: &[SkillDef] = &[
    SkillDef {
        name: "code_review",
        summary: "lint + diff summarisation tools, with a reviewer's rubric",
        body: "You are now in code-review mode. When producing the final answer, \
             structure it as: (1) blocking issues, (2) non-blocking suggestions, \
             (3) praise. Keep each bullet under 140 characters.",
        selectors: || {
            vec![
                AppToolsSelector::CodeReview(CodeReviewToolsSelector::RunLints),
                AppToolsSelector::CodeReview(CodeReviewToolsSelector::DiffSummary),
            ]
        },
    },
    SkillDef {
        name: "web",
        summary: "fetch arbitrary URLs and read their body as plain text",
        body: "You are now in web-research mode. Always cite the URL you fetched \
             next to any claim that depends on its content.",
        selectors: || vec![AppToolsSelector::Web(WebToolsSelector::FetchUrl)],
    },
];

fn find_skill(name: &str) -> Option<&'static SkillDef> {
    SKILLS.iter().find(|s| s.name == name)
}

fn render_load_catalogue(loaded: &BTreeSet<String>) -> String {
    let mut out = String::from(
        "Load a skill by name to unlock its tools and instructions. \
         Exactly one name per call. Available skills:\n",
    );
    for s in SKILLS {
        let status = if loaded.contains(s.name) {
            " (loaded)"
        } else {
            ""
        };
        out.push_str(&format!("- {}{}: {}\n", s.name, status, s.summary));
    }
    out
}

fn render_unload_catalogue(loaded: &BTreeSet<String>) -> String {
    if loaded.is_empty() {
        "Unload a skill by name. No skills are currently loaded.".to_string()
    } else {
        let mut out = String::from("Unload a skill by name. Currently loaded: ");
        out.push_str(&loaded.iter().cloned().collect::<Vec<_>>().join(", "));
        out.push('.');
        out
    }
}

// ── Mock LLM scripting ─────────────────────────────────────────────────────
//
// Five consecutive turns demonstrate the full skill lifecycle:
//   1. Model loads the `code_review` skill.
//   2. Model calls `run_lints` (now available because code_review is loaded).
//   3. Model calls `diff_summary`.
//   4. Model unloads `code_review`.
//   5. Model emits the final review report.

fn script_turn(id: &str, call_id: &str, name: &str, args_json: &str) -> MockTextScenario {
    MockTextScenario::events(vec![
        Ok(RawTextTurnEvent::Started {
            request_id: Some(id.into()),
            model: "mock-reviewer".into(),
        }),
        Ok(RawTextTurnEvent::ToolCallChunk {
            id: call_id.into(),
            name: name.into(),
            arguments_json_delta: args_json.into(),
        }),
        Ok(RawTextTurnEvent::Completed {
            request_id: Some(id.into()),
            finish_reason: FinishReason::ToolCall,
            usage: Usage::zero(),
        }),
    ])
}

fn script_text_turn(id: &str, text: &str) -> MockTextScenario {
    MockTextScenario::events(vec![
        Ok(RawTextTurnEvent::Started {
            request_id: Some(id.into()),
            model: "mock-reviewer".into(),
        }),
        Ok(RawTextTurnEvent::TextDelta { delta: text.into() }),
        Ok(RawTextTurnEvent::Completed {
            request_id: Some(id.into()),
            finish_reason: FinishReason::Stop,
            usage: Usage::zero(),
        }),
    ])
}

fn mock_adapter() -> MockLlmAdapter {
    MockLlmAdapter::new()
        .with_text_scenario(script_turn(
            "t1",
            "call-load",
            "load_skill",
            r#"{"name":"code_review"}"#,
        ))
        .with_text_scenario(script_turn("t2", "call-lint", "run_lints", "{}"))
        .with_text_scenario(script_turn("t3", "call-diff", "diff_summary", "{}"))
        .with_text_scenario(script_turn(
            "t4",
            "call-unload",
            "unload_skill",
            r#"{"name":"code_review"}"#,
        ))
        .with_text_scenario(script_text_turn(
            "t5",
            "Review:\n- blocking: none\n- suggestions: add test coverage for load errors\n- praise: clean skill registry design",
        ))
}

// ── Deterministic tool execution ──────────────────────────────────────────

fn run_lints() -> Report {
    Report {
        text: "lint report: 0 errors, 2 warnings (unused_variable x2).".into(),
    }
}

fn diff_summary() -> Report {
    Report {
        text: "diff: +42 / -11 across 3 files; renames one helper.".into(),
    }
}

fn fetch_url(url: &str) -> Page {
    Page {
        text: format!("<fetched body for {url}>"),
    }
}

// ── Loop ──────────────────────────────────────────────────────────────────

async fn run() -> anyhow::Result<()> {
    let ctx = lutum::Lutum::new(
        Arc::new(mock_adapter()),
        SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
    );
    let mut session = Session::new(ctx);
    session.push_system(
        "You are a code reviewer with access to skill loading. \
         Load only the skills you need. Unload skills you are done with.",
    );
    session.push_user("Please review the latest changes on this branch.");

    let mut loaded: BTreeSet<String> = BTreeSet::new();

    for step in 1..=8 {
        // 1. Inject the bodies of currently-loaded skills as ephemeral
        //    developer messages. They will be visible on this turn and then
        //    auto-stripped before the next commit.
        for name in &loaded {
            if let Some(skill) = find_skill(name) {
                session
                    .push_ephemeral_developer(format!("# skill: {}\n{}", skill.name, skill.body));
            }
        }

        // 2. Compute the set of skill tools to expose. `available_tools_default_plus`
        //    takes default-on tools (load_skill, unload_skill) plus this set.
        let extra_selectors: Vec<AppToolsSelector> = loaded
            .iter()
            .filter_map(|n| find_skill(n))
            .flat_map(|s| (s.selectors)())
            .collect();

        let outcome = session
            .text_turn()
            .tools::<AppTools>()
            .available_tools_default_plus(extra_selectors)
            .describe_tool(AppToolsSelector::LoadSkill, render_load_catalogue(&loaded))
            .describe_tool(
                AppToolsSelector::UnloadSkill,
                render_unload_catalogue(&loaded),
            )
            .collect()
            .await?;

        let round = match outcome {
            TextStepOutcomeWithTools::NeedsTools(round) => round,
            TextStepOutcomeWithTools::Finished(result) => {
                println!("\n=== final answer ===\n{}", result.assistant_text().trim());
                println!(
                    "\n=== transcript markers ({} turns persisted) ===",
                    session.list_turns().count()
                );
                return Ok(());
            }
        };

        // 3. Dispatch tool calls. Skills' `load`/`unload` queue persistent
        //    markers to push AFTER committing the tool round; this keeps the
        //    transcript in the order: assistant turn → tool results → marker.
        let mut results = Vec::with_capacity(round.tool_calls.len());
        let mut post_round_markers: Vec<String> = Vec::new();

        for call in round.tool_calls.iter().cloned() {
            match call {
                AppToolsCall::LoadSkill(c) => {
                    let name = c.input().name.clone();
                    let ack = if find_skill(&name).is_some() {
                        loaded.insert(name.clone());
                        post_round_markers.push(format!("[system] skill `{name}` loaded"));
                        SkillAck {
                            ok: true,
                            message: format!("skill `{name}` is now available"),
                        }
                    } else {
                        SkillAck {
                            ok: false,
                            message: format!("unknown skill `{name}`"),
                        }
                    };
                    println!("[step {step}] load_skill({name}) → ok={}", ack.ok);
                    results.push(c.complete(ack)?);
                }
                AppToolsCall::UnloadSkill(c) => {
                    let name = c.input().name.clone();
                    let removed = loaded.remove(&name);
                    if removed {
                        post_round_markers.push(format!("[system] skill `{name}` unloaded"));
                    }
                    println!("[step {step}] unload_skill({name}) → ok={removed}");
                    results.push(c.complete(SkillAck {
                        ok: removed,
                        message: if removed {
                            format!("skill `{name}` unloaded")
                        } else {
                            format!("skill `{name}` was not loaded")
                        },
                    })?);
                }
                AppToolsCall::CodeReview(CodeReviewToolsCall::RunLints(c)) => {
                    println!("[step {step}] run_lints()");
                    results.push(c.complete(run_lints())?);
                }
                AppToolsCall::CodeReview(CodeReviewToolsCall::DiffSummary(c)) => {
                    println!("[step {step}] diff_summary()");
                    results.push(c.complete(diff_summary())?);
                }
                AppToolsCall::Web(WebToolsCall::FetchUrl(c)) => {
                    let url = c.input().url.clone();
                    println!("[step {step}] fetch_url({url})");
                    results.push(c.complete(fetch_url(&url))?);
                }
            }
        }

        round.commit(&mut session, results)?;

        for marker in post_round_markers {
            session.push_developer(marker);
        }
    }

    anyhow::bail!("skill example hit the 8-step limit without a final answer");
}

fn main() -> anyhow::Result<()> {
    block_on(run())
}
