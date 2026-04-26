use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use lutum::{
    Lutum, ModelName, RawTelemetryConfig, RequestExtensions, Session, SharedPoolBudgetManager,
    SharedPoolBudgetOptions, StructuredTurnOutcome, TextStepOutcomeWithTools, Usage, UsageEstimate,
    UsageRecoveryAdapter,
};
use lutum_claude::{ClaudeAdapter, MessagesRequest};
use lutum_openai::{CompletionRequest, OpenAiAdapter, OpenAiReasoningEffort};
use lutum_trace::{EventRecord, FieldValue, RawTraceEntry, SpanNode, TraceSnapshot};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, Parser)]
#[command(about = "Run live smoke tests against configured lutum adapters")]
struct Args {
    #[arg(long, default_value = "data/adapter-smoke-config.eure")]
    config: PathBuf,
    #[arg(long)]
    only: Vec<String>,
    #[arg(long)]
    max_usd: Option<f64>,
    #[arg(long)]
    strict: bool,
    #[arg(long)]
    save_raw: Option<PathBuf>,
}

#[derive(Clone, Debug, eure::FromEure)]
#[eure(crate = ::eure::document)]
struct SmokeConfig {
    defaults: DefaultsConfig,
    endpoints: BTreeMap<String, EndpointConfig>,
}

#[derive(Clone, Debug, eure::FromEure)]
#[eure(crate = ::eure::document)]
struct DefaultsConfig {
    max_usd: f64,
    text_max_output_tokens: u32,
    structured_max_output_tokens: u32,
    claude_thinking_budget_tokens: u32,
    openai_reasoning_effort: String,
}

#[derive(Clone, Debug, eure::FromEure)]
#[eure(crate = ::eure::document)]
struct EndpointConfig {
    adapter: String,
    base_url: String,
    api_key_env: String,
    #[eure(default)]
    fallback_api_key: Option<String>,
    model: String,
    #[eure(default)]
    fallback_models: Vec<String>,
    cases: Vec<String>,
    #[eure(default)]
    expects_visible_thinking: bool,
    #[eure(default)]
    openrouter_usage_recovery: bool,
    #[eure(default)]
    recovery_base_url: Option<String>,
    #[eure(default)]
    input_price_per_million: f64,
    #[eure(default)]
    output_price_per_million: f64,
    #[eure(default)]
    text_max_output_tokens: Option<u32>,
    #[eure(default)]
    structured_max_output_tokens: Option<u32>,
    #[eure(default)]
    openai_reasoning_effort: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AdapterKind {
    ClaudeMessages,
    OpenAiChatCompletions,
    OpenAiCompletions,
    OpenAiResponses,
}

impl AdapterKind {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "claude-messages" => Ok(Self::ClaudeMessages),
            "openai-chat-completions" => Ok(Self::OpenAiChatCompletions),
            "openai-completions" => Ok(Self::OpenAiCompletions),
            "openai-responses" => Ok(Self::OpenAiResponses),
            _ => bail!("unknown adapter {value:?}"),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SmokeCase {
    Completion,
    Text,
    Structured,
    Tool,
    StructuredCompletion,
    ReasoningRequest,
    ThinkingRoundtrip,
}

impl SmokeCase {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "completion" => Ok(Self::Completion),
            "text" => Ok(Self::Text),
            "structured" => Ok(Self::Structured),
            "tool" => Ok(Self::Tool),
            "structured_completion" => Ok(Self::StructuredCompletion),
            "reasoning_request" => Ok(Self::ReasoningRequest),
            "thinking_roundtrip" => Ok(Self::ThinkingRoundtrip),
            _ => bail!("unknown smoke case {value:?}"),
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::Completion => "completion",
            Self::Text => "text",
            Self::Structured => "structured",
            Self::Tool => "tool",
            Self::StructuredCompletion => "structured_completion",
            Self::ReasoningRequest => "reasoning_request",
            Self::ThinkingRoundtrip => "thinking_roundtrip",
        }
    }
}

#[derive(Clone, Debug)]
struct CaseSpec {
    endpoint_id: String,
    case_id: String,
    kind: AdapterKind,
    case: SmokeCase,
    endpoint: EndpointConfig,
}

#[derive(Debug, Eq, PartialEq)]
enum CaseStatus {
    Passed,
    Skipped,
    Failed,
}

#[derive(Debug)]
struct CaseReport {
    case_id: String,
    status: CaseStatus,
    message: String,
    usage: Usage,
    raw: Vec<RawTraceEntry>,
    trace: TraceSnapshot,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct SmokeStructured {
    ok: bool,
    text: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct EchoWordResult {
    word: String,
}

#[lutum::tool_input(name = "echo_word", output = EchoWordResult)]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct EchoWordArgs {
    word: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
enum SmokeTools {
    EchoWord(EchoWordArgs),
}

#[derive(Clone, Copy)]
struct SmokeReasoningConfig(OpenAiReasoningEffort);

#[lutum::impl_hook(lutum_openai::ResolveReasoningEffort)]
async fn smoke_reasoning_effort(extensions: &RequestExtensions) -> Option<OpenAiReasoningEffort> {
    extensions
        .get::<SmokeReasoningConfig>()
        .map(|value| value.0)
}

#[derive(Clone, Debug)]
struct OpenRouterOpenAiFallback {
    models: Vec<String>,
}

impl lutum_openai::FallbackSerializer for OpenRouterOpenAiFallback {
    fn apply_to_responses(&self, request: &mut lutum_openai::ResponsesRequest) {
        request.models = Some(self.models.clone());
    }

    fn apply_to_completion(&self, request: &mut CompletionRequest) {
        request.models = Some(self.models.clone());
    }

    fn apply_to_chat(&self, request: &mut lutum_openai::ChatCompletionRequest) {
        request.models = Some(self.models.clone());
    }
}

#[derive(Clone, Debug)]
struct OpenRouterClaudeFallback {
    models: Vec<String>,
}

impl lutum_claude::FallbackSerializer for OpenRouterClaudeFallback {
    fn apply(&self, request: &mut MessagesRequest) {
        request.models = Some(self.models.clone());
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let config = load_config(&args.config)?;
    let cases = expand_cases(&config)?;
    let selected = filter_cases(cases, &args.only)?;
    let estimate = estimate_cases_usd(&selected, &config.defaults);
    let cap = args.max_usd.unwrap_or(config.defaults.max_usd);
    if estimate > cap {
        bail!(
            "estimated worst-case cost ${estimate:.4} exceeds max ${cap:.4}; pass --max-usd to raise the cap"
        );
    }

    println!(
        "running {} adapter smoke cases; estimated worst-case cost ${estimate:.4} (cap ${cap:.4})",
        selected.len()
    );

    let mut reports = Vec::new();
    for case in selected {
        let report = run_case(case, &config.defaults, args.strict).await;
        if let Some(path) = args.save_raw.as_deref() {
            write_raw(path, &report)
                .with_context(|| format!("failed to save raw trace to {}", path.display()))?;
        }
        print_report(&report);
        reports.push(report);
    }

    let failed = reports
        .iter()
        .filter(|report| matches!(report.status, CaseStatus::Failed))
        .count();
    let skipped = reports
        .iter()
        .filter(|report| matches!(report.status, CaseStatus::Skipped))
        .count();
    let usage = reports.iter().fold(Usage::zero(), |sum, report| {
        sum.saturating_add(report.usage)
    });

    println!(
        "summary: {} passed, {skipped} skipped, {failed} failed, usage={} tokens, cost=${:.6}",
        reports.len().saturating_sub(skipped + failed),
        usage.total_tokens,
        usage.cost_micros_usd as f64 / 1_000_000.0
    );

    if failed > 0 {
        bail!("{failed} smoke case(s) failed");
    }

    Ok(())
}

fn load_config(path: &Path) -> Result<SmokeConfig> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("failed to read smoke config {}", path.display()))?;
    eure::parse_content(&content, path.to_path_buf())
        .map_err(|err| anyhow!("failed to parse smoke config {}: {err}", path.display()))
}

fn expand_cases(config: &SmokeConfig) -> Result<Vec<CaseSpec>> {
    let mut specs = Vec::new();
    let mut ids = BTreeSet::new();
    for (endpoint_id, endpoint) in &config.endpoints {
        let kind = AdapterKind::parse(&endpoint.adapter)
            .with_context(|| format!("endpoint {endpoint_id} has invalid adapter"))?;
        for case_name in &endpoint.cases {
            let case = SmokeCase::parse(case_name)
                .with_context(|| format!("endpoint {endpoint_id} has invalid case"))?;
            validate_case(endpoint_id, kind, case)?;
            let case_id = format!("{endpoint_id}:{}", case.as_str());
            if !ids.insert(case_id.clone()) {
                bail!("duplicate smoke case id {case_id}");
            }
            specs.push(CaseSpec {
                endpoint_id: endpoint_id.clone(),
                case_id,
                kind,
                case,
                endpoint: endpoint.clone(),
            });
        }
    }
    Ok(specs)
}

fn validate_case(endpoint_id: &str, kind: AdapterKind, case: SmokeCase) -> Result<()> {
    match (kind, case) {
        (AdapterKind::OpenAiCompletions, SmokeCase::Completion) => Ok(()),
        (AdapterKind::OpenAiCompletions, _) => {
            bail!("endpoint {endpoint_id} uses /completions and can only run completion cases")
        }
        (_, SmokeCase::Completion) => {
            bail!("endpoint {endpoint_id} completion case requires openai-completions adapter")
        }
        (AdapterKind::ClaudeMessages, SmokeCase::StructuredCompletion) => {
            bail!("endpoint {endpoint_id} structured_completion requires openai-responses")
        }
        (AdapterKind::OpenAiChatCompletions, SmokeCase::StructuredCompletion) => {
            bail!("endpoint {endpoint_id} structured_completion requires openai-responses")
        }
        _ => Ok(()),
    }
}

fn filter_cases(cases: Vec<CaseSpec>, only: &[String]) -> Result<Vec<CaseSpec>> {
    if only.is_empty() {
        return Ok(cases);
    }
    let selected = cases
        .into_iter()
        .filter(|case| {
            only.iter().any(|needle| {
                case.case_id == *needle
                    || case.endpoint_id == *needle
                    || case.case.as_str() == needle.as_str()
                    || case.case_id.contains(needle)
            })
        })
        .collect::<Vec<_>>();
    if selected.is_empty() {
        bail!("--only did not match any configured smoke case");
    }
    Ok(selected)
}

fn estimate_cases_usd(cases: &[CaseSpec], defaults: &DefaultsConfig) -> f64 {
    cases
        .iter()
        .map(|case| estimate_case_usd(case, defaults))
        .sum()
}

fn estimate_case_usd(case: &CaseSpec, defaults: &DefaultsConfig) -> f64 {
    let (input_tokens, output_tokens) = worst_case_tokens(case, defaults);
    let input_cost = input_tokens as f64 * case.endpoint.input_price_per_million / 1_000_000.0;
    let output_cost = output_tokens as f64 * case.endpoint.output_price_per_million / 1_000_000.0;
    input_cost + output_cost
}

fn worst_case_tokens(case: &CaseSpec, defaults: &DefaultsConfig) -> (u64, u64) {
    match case.case {
        SmokeCase::Completion | SmokeCase::Text | SmokeCase::ReasoningRequest => {
            (96, text_max_output_tokens(&case.endpoint, defaults) as u64)
        }
        SmokeCase::Structured | SmokeCase::StructuredCompletion | SmokeCase::Tool => (
            192,
            structured_max_output_tokens(&case.endpoint, defaults) as u64,
        ),
        SmokeCase::ThinkingRoundtrip => {
            let output = if case.kind == AdapterKind::ClaudeMessages {
                defaults.claude_thinking_budget_tokens as u64 + 1024
            } else {
                structured_max_output_tokens(&case.endpoint, defaults) as u64
            };
            (
                256,
                output + text_max_output_tokens(&case.endpoint, defaults) as u64,
            )
        }
    }
}

async fn run_case(case: CaseSpec, defaults: &DefaultsConfig, strict: bool) -> CaseReport {
    let mut usage = Usage::zero();
    let credential = match resolve_api_key(&case.endpoint) {
        Ok(key) => key,
        Err(err) => {
            let status = missing_credential_status(strict);
            return CaseReport {
                case_id: case.case_id,
                status,
                message: err.to_string(),
                usage,
                raw: Vec::new(),
                trace: empty_trace(),
            };
        }
    };

    let run = async {
        let llm = build_lutum(&case, defaults, credential)?;
        let case_usage = match case.case {
            SmokeCase::Completion => run_completion(&llm, &case, defaults).await?,
            SmokeCase::Text => run_text(&llm, &case, defaults).await?,
            SmokeCase::Structured => run_structured(&llm, &case, defaults).await?,
            SmokeCase::Tool => run_tool(&llm, &case, defaults).await?,
            SmokeCase::StructuredCompletion => {
                run_structured_completion(&llm, &case, defaults).await?
            }
            SmokeCase::ReasoningRequest => run_reasoning_request(&llm, &case, defaults).await?,
            SmokeCase::ThinkingRoundtrip => run_thinking_roundtrip(&llm, &case, defaults).await?,
        };
        Ok::<Usage, anyhow::Error>(case_usage)
    };

    let collected = lutum_trace::test::collect_raw(run).await;
    let output = collected.output;
    let trace = collected.trace;
    let raw = collected.raw.entries;
    match output {
        Ok(case_usage) => {
            usage = usage.saturating_add(case_usage);
            if let Err(err) = verify_raw_expectations(&case, &raw) {
                return CaseReport {
                    case_id: case.case_id,
                    status: CaseStatus::Failed,
                    message: err.to_string(),
                    usage,
                    raw,
                    trace,
                };
            }
            CaseReport {
                case_id: case.case_id,
                status: CaseStatus::Passed,
                message: "ok".into(),
                usage,
                raw,
                trace,
            }
        }
        Err(err) => {
            let endpoint_unavailable = raw.iter().any(|entry| {
                matches!(
                    entry,
                    RawTraceEntry::RequestError {
                        is_connect: true,
                        ..
                    }
                )
            });
            let status = if !strict && endpoint_unavailable {
                CaseStatus::Skipped
            } else {
                CaseStatus::Failed
            };
            let message = if status == CaseStatus::Skipped {
                format!("endpoint unavailable: {err}")
            } else {
                err.to_string()
            };
            CaseReport {
                case_id: case.case_id,
                status,
                message,
                usage,
                raw,
                trace,
            }
        }
    }
}

fn resolve_api_key(endpoint: &EndpointConfig) -> Result<String> {
    resolve_api_key_with(endpoint, |name| env::var(name).ok())
}

fn resolve_api_key_with(
    endpoint: &EndpointConfig,
    get_env: impl FnOnce(&str) -> Option<String>,
) -> Result<String> {
    match get_env(&endpoint.api_key_env) {
        Some(value) if !value.trim().is_empty() => Ok(value),
        _ => endpoint
            .fallback_api_key
            .clone()
            .ok_or_else(|| anyhow!("missing credential env {}", endpoint.api_key_env)),
    }
}

const fn missing_credential_status(strict: bool) -> CaseStatus {
    if strict {
        CaseStatus::Failed
    } else {
        CaseStatus::Skipped
    }
}

fn build_lutum(case: &CaseSpec, defaults: &DefaultsConfig, api_key: String) -> Result<Lutum> {
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let usage_estimate = UsageEstimate {
        total_tokens: 4096,
        cost_micros_usd: (estimate_case_usd(case, defaults) * 1_000_000.0).ceil() as u64,
        ..UsageEstimate::zero()
    };
    match case.kind {
        AdapterKind::ClaudeMessages => {
            let mut adapter = ClaudeAdapter::new(api_key.clone())
                .with_base_url(case.endpoint.base_url.clone())
                .with_default_model(model_name(&case.endpoint.model)?)
                .without_prompt_caching();
            if case.case == SmokeCase::ThinkingRoundtrip {
                adapter =
                    adapter.with_default_thinking_budget(defaults.claude_thinking_budget_tokens);
            }
            if !case.endpoint.fallback_models.is_empty() {
                adapter.set_fallback_serializer(Box::new(OpenRouterClaudeFallback {
                    models: case.endpoint.fallback_models.clone(),
                }));
            }
            let adapter = Arc::new(adapter);
            let recovery = recovery_adapter(&case.endpoint, api_key, adapter.clone());
            Ok(Lutum::new(adapter.clone(), budget)
                .with_recovery(recovery)
                .with_extension(RawTelemetryConfig::all())
                .with_extension(usage_estimate))
        }
        AdapterKind::OpenAiChatCompletions
        | AdapterKind::OpenAiCompletions
        | AdapterKind::OpenAiResponses => {
            let mut adapter = OpenAiAdapter::new(api_key.clone())
                .with_base_url(case.endpoint.base_url.clone())
                .with_default_model(model_name(&case.endpoint.model)?)
                .with_resolve_reasoning_effort(SmokeReasoningEffort);
            if case.kind == AdapterKind::OpenAiChatCompletions {
                adapter = adapter.with_chat_completions();
            }
            if !case.endpoint.fallback_models.is_empty() {
                adapter.set_fallback_serializer(Box::new(OpenRouterOpenAiFallback {
                    models: case.endpoint.fallback_models.clone(),
                }));
            }
            let adapter = Arc::new(adapter);
            let recovery = recovery_adapter(&case.endpoint, api_key, adapter.clone());
            Ok(Lutum::from_parts(adapter.clone(), adapter.clone(), budget)
                .with_recovery(recovery)
                .with_extension(RawTelemetryConfig::all())
                .with_extension(SmokeReasoningConfig(parse_reasoning_effort(
                    case.endpoint
                        .openai_reasoning_effort
                        .as_deref()
                        .unwrap_or(&defaults.openai_reasoning_effort),
                )?))
                .with_extension(usage_estimate))
        }
    }
}

fn recovery_adapter<T>(
    endpoint: &EndpointConfig,
    api_key: String,
    fallback: Arc<T>,
) -> Arc<dyn UsageRecoveryAdapter>
where
    T: UsageRecoveryAdapter + 'static,
{
    if endpoint.openrouter_usage_recovery {
        let base_url = endpoint
            .recovery_base_url
            .clone()
            .unwrap_or_else(|| "https://openrouter.ai/api/v1".to_string());
        Arc::new(lutum_openrouter::OpenRouterGenerationClient::new(api_key).with_base_url(base_url))
    } else {
        fallback
    }
}

fn model_name(value: &str) -> Result<ModelName> {
    ModelName::new(value).map_err(|err| anyhow!("invalid model name {value:?}: {err}"))
}

fn parse_reasoning_effort(value: &str) -> Result<OpenAiReasoningEffort> {
    match value {
        "none" => Ok(OpenAiReasoningEffort::None),
        "minimal" => Ok(OpenAiReasoningEffort::Minimal),
        "low" => Ok(OpenAiReasoningEffort::Low),
        "medium" => Ok(OpenAiReasoningEffort::Medium),
        "high" => Ok(OpenAiReasoningEffort::High),
        "xhigh" => Ok(OpenAiReasoningEffort::Xhigh),
        _ => bail!("unknown OpenAI reasoning effort {value:?}"),
    }
}

fn text_max_output_tokens(endpoint: &EndpointConfig, defaults: &DefaultsConfig) -> u32 {
    endpoint
        .text_max_output_tokens
        .unwrap_or(defaults.text_max_output_tokens)
}

fn structured_max_output_tokens(endpoint: &EndpointConfig, defaults: &DefaultsConfig) -> u32 {
    endpoint
        .structured_max_output_tokens
        .unwrap_or(defaults.structured_max_output_tokens)
}

async fn run_completion(llm: &Lutum, case: &CaseSpec, defaults: &DefaultsConfig) -> Result<Usage> {
    let result = llm
        .completion("Return exactly OK.")
        .max_output_tokens(text_max_output_tokens(&case.endpoint, defaults))
        .collect()
        .await?;
    ensure_ok(&result.text, "completion")?;
    Ok(result.usage)
}

async fn run_text(llm: &Lutum, case: &CaseSpec, defaults: &DefaultsConfig) -> Result<Usage> {
    let mut session = Session::new(llm.clone());
    session.push_user("Return exactly OK.");
    let result = session
        .text_turn()
        .max_output_tokens(text_max_output_tokens(&case.endpoint, defaults))
        .collect()
        .await?;
    ensure_ok(&result.assistant_text(), "text")?;
    Ok(result.usage)
}

async fn run_structured(llm: &Lutum, case: &CaseSpec, defaults: &DefaultsConfig) -> Result<Usage> {
    let mut session = Session::new(llm.clone());
    session.push_user("Return JSON with ok true and text exactly OK.");
    let result = session
        .structured_turn::<SmokeStructured>()
        .max_output_tokens(structured_max_output_tokens(&case.endpoint, defaults))
        .collect()
        .await?;
    match result.semantic {
        StructuredTurnOutcome::Structured(value) if value.ok && normalize_ok(&value.text) => {
            Ok(result.usage)
        }
        other => bail!("structured output was not OK: {other:?}"),
    }
}

async fn run_structured_completion(
    llm: &Lutum,
    case: &CaseSpec,
    defaults: &DefaultsConfig,
) -> Result<Usage> {
    let result = llm
        .structured_completion::<SmokeStructured>("Return JSON with ok true and text exactly OK.")
        .max_output_tokens(structured_max_output_tokens(&case.endpoint, defaults))
        .collect()
        .await?;
    match result.semantic {
        StructuredTurnOutcome::Structured(value) if value.ok && normalize_ok(&value.text) => {
            Ok(result.usage)
        }
        other => bail!("structured completion output was not OK: {other:?}"),
    }
}

async fn run_tool(llm: &Lutum, case: &CaseSpec, defaults: &DefaultsConfig) -> Result<Usage> {
    let mut session = Session::new(llm.clone());
    session.push_user("Call echo_word with word OK, then answer exactly OK.");
    let mut usage = Usage::zero();
    let mut saw_tool = false;
    for _ in 0..3 {
        let mut turn = session
            .text_turn()
            .tools::<SmokeTools>()
            .available_tools(vec![SmokeToolsSelector::EchoWord])
            .max_output_tokens(structured_max_output_tokens(&case.endpoint, defaults));
        if !saw_tool {
            turn = turn.require_tool(SmokeToolsSelector::EchoWord);
        }
        let outcome = turn.collect().await?;
        match outcome {
            TextStepOutcomeWithTools::NeedsTools(round) => {
                usage = usage.saturating_add(round.usage);
                saw_tool = true;
                let results = round
                    .tool_calls
                    .iter()
                    .cloned()
                    .map(|call| match call {
                        SmokeToolsCall::EchoWord(call) => {
                            let word = call.input.word.clone();
                            call.complete(EchoWordResult { word })
                                .map_err(anyhow::Error::from)
                        }
                    })
                    .collect::<Result<Vec<_>>>()?;
                round.commit(&mut session, results)?;
            }
            TextStepOutcomeWithTools::Finished(result) => {
                usage = usage.saturating_add(result.usage);
                if !saw_tool {
                    bail!("tool case finished without a tool call");
                }
                ensure_ok(&result.assistant_text(), "tool final answer")?;
                return Ok(usage);
            }
        }
    }
    bail!("tool case did not finish after tool result")
}

async fn run_reasoning_request(
    llm: &Lutum,
    case: &CaseSpec,
    defaults: &DefaultsConfig,
) -> Result<Usage> {
    let usage = run_text(llm, case, defaults).await?;
    Ok(usage)
}

async fn run_thinking_roundtrip(
    llm: &Lutum,
    case: &CaseSpec,
    defaults: &DefaultsConfig,
) -> Result<Usage> {
    let mut session = Session::new(llm.clone());
    session.push_user("Think briefly, then answer exactly OK.");
    let max_tokens = if case.kind == AdapterKind::ClaudeMessages {
        defaults.claude_thinking_budget_tokens + 1024
    } else {
        structured_max_output_tokens(&case.endpoint, defaults)
    };
    let first = session
        .text_turn()
        .max_output_tokens(max_tokens)
        .collect()
        .await?;
    ensure_ok(&first.assistant_text(), "thinking first turn")?;

    session.push_user("Using the previous turn, answer exactly OK.");
    let second = session
        .text_turn()
        .max_output_tokens(text_max_output_tokens(&case.endpoint, defaults))
        .collect()
        .await?;
    ensure_ok(&second.assistant_text(), "thinking replay turn")?;
    Ok(first.usage.saturating_add(second.usage))
}

fn verify_raw_expectations(case: &CaseSpec, raw: &[RawTraceEntry]) -> Result<()> {
    if matches!(
        case.case,
        SmokeCase::ReasoningRequest | SmokeCase::ThinkingRoundtrip
    ) && case.kind == AdapterKind::OpenAiChatCompletions
        && !raw_request_contains(raw, &["reasoning_effort"])
    {
        bail!("chat completions reasoning request did not include reasoning_effort");
    }

    if matches!(
        case.case,
        SmokeCase::ReasoningRequest | SmokeCase::ThinkingRoundtrip
    ) && case.kind == AdapterKind::OpenAiResponses
        && !raw_request_contains(raw, &["reasoning"])
    {
        bail!("responses reasoning request did not include reasoning config");
    }

    if case.case == SmokeCase::ThinkingRoundtrip && case.kind == AdapterKind::ClaudeMessages {
        if !raw_request_contains(raw, &["thinking", "budget_tokens"]) {
            bail!("Claude thinking_roundtrip request did not include thinking budget");
        }
        if !raw_request_contains(raw, &["\"type\":\"thinking\""]) {
            bail!("Claude thinking_roundtrip replay did not preserve thinking blocks");
        }
    }

    if case.case == SmokeCase::ThinkingRoundtrip && case.endpoint.expects_visible_thinking {
        if !raw_stream_contains(raw, &["thinking", "reasoning", "reasoning_content"]) {
            bail!(
                "endpoint expected visible thinking, but no thinking/reasoning stream payload decoded"
            );
        }
        if case.kind == AdapterKind::OpenAiResponses
            && !raw_request_contains(raw, &["\"type\":\"reasoning\""])
        {
            bail!("responses replay request did not preserve reasoning items");
        }
    }

    if !case.endpoint.fallback_models.is_empty() && !raw_request_contains(raw, &["models"]) {
        bail!("fallback case did not serialize OpenRouter models");
    }

    Ok(())
}

fn raw_request_contains(raw: &[RawTraceEntry], needles: &[&str]) -> bool {
    raw.iter().any(|entry| {
        if let RawTraceEntry::Request { body, .. } = entry {
            needles
                .iter()
                .all(|needle| body.to_ascii_lowercase().contains(needle))
        } else {
            false
        }
    })
}

fn raw_stream_contains(raw: &[RawTraceEntry], needles: &[&str]) -> bool {
    raw.iter().any(|entry| {
        if let RawTraceEntry::StreamEvent {
            payload,
            event_name,
            ..
        } = entry
        {
            let payload = payload.to_ascii_lowercase();
            let event_name = event_name
                .as_deref()
                .unwrap_or_default()
                .to_ascii_lowercase();
            needles
                .iter()
                .any(|needle| payload.contains(needle) || event_name.contains(needle))
        } else {
            false
        }
    })
}

fn ensure_ok(text: &str, label: &str) -> Result<()> {
    if normalize_ok(text) {
        Ok(())
    } else {
        bail!("{label} expected OK, got {text:?}")
    }
}

fn normalize_ok(text: &str) -> bool {
    let trimmed = text.trim().trim_matches('"').trim();
    trimmed == "OK" || trimmed.starts_with("OK\n") || trimmed.starts_with("OK.")
}

fn print_report(report: &CaseReport) {
    let status = match report.status {
        CaseStatus::Passed => "PASS",
        CaseStatus::Skipped => "SKIP",
        CaseStatus::Failed => "FAIL",
    };
    println!(
        "{status} {}: {} ({} tokens, ${:.6})",
        report.case_id,
        report.message,
        report.usage.total_tokens,
        report.usage.cost_micros_usd as f64 / 1_000_000.0
    );
}

fn write_raw(path: &Path, report: &CaseReport) -> Result<()> {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("open {}", path.display()))?;
    let line = json!({
        "kind": "case_dump",
        "case_id": report.case_id,
        "status": format!("{:?}", report.status),
        "message": report.message,
        "usage": report.usage,
        "raw": report.raw.iter().map(raw_entry_json).collect::<Vec<_>>(),
        "trace": trace_snapshot_json(&report.trace),
    });
    serde_json::to_writer(&mut file, &line)?;
    file.write_all(b"\n")?;
    Ok(())
}

fn empty_trace() -> TraceSnapshot {
    TraceSnapshot {
        roots: Vec::new(),
        root_events: Vec::new(),
    }
}

fn trace_snapshot_json(snapshot: &TraceSnapshot) -> serde_json::Value {
    json!({
        "roots": snapshot.roots.iter().map(span_node_json).collect::<Vec<_>>(),
        "root_events": snapshot.root_events.iter().map(event_record_json).collect::<Vec<_>>(),
    })
}

fn span_node_json(span: &SpanNode) -> serde_json::Value {
    json!({
        "name": span.name,
        "target": span.target,
        "level": span.level,
        "fields": fields_json(&span.fields),
        "events": span.events.iter().map(event_record_json).collect::<Vec<_>>(),
        "children": span.children.iter().map(span_node_json).collect::<Vec<_>>(),
    })
}

fn event_record_json(event: &EventRecord) -> serde_json::Value {
    json!({
        "target": event.target,
        "level": event.level,
        "message": event.message,
        "fields": fields_json(&event.fields),
    })
}

fn fields_json(fields: &[(String, FieldValue)]) -> serde_json::Value {
    serde_json::Value::Object(
        fields
            .iter()
            .map(|(name, value)| (name.clone(), field_value_json(value)))
            .collect(),
    )
}

fn field_value_json(value: &FieldValue) -> serde_json::Value {
    match value {
        FieldValue::Bool(value) => json!(value),
        FieldValue::I64(value) => json!(value),
        FieldValue::U64(value) => json!(value),
        FieldValue::I128(value) => json!(value.to_string()),
        FieldValue::U128(value) => json!(value.to_string()),
        FieldValue::F64(value) => json!(value),
        FieldValue::Str(value) => json!(value),
    }
}

fn raw_entry_json(entry: &RawTraceEntry) -> serde_json::Value {
    match entry {
        RawTraceEntry::Request {
            provider,
            api,
            operation,
            request_id,
            body,
        } => json!({
            "kind": "request",
            "provider": provider,
            "api": api,
            "operation": operation,
            "request_id": request_id,
            "body": body,
        }),
        RawTraceEntry::StreamEvent {
            provider,
            api,
            operation,
            request_id,
            sequence,
            payload,
            event_name,
        } => json!({
            "kind": "stream_event",
            "provider": provider,
            "api": api,
            "operation": operation,
            "request_id": request_id,
            "sequence": sequence,
            "event_name": event_name,
            "payload": payload,
        }),
        RawTraceEntry::ParseError {
            provider,
            api,
            operation,
            request_id,
            stage,
            payload,
            error,
        } => json!({
            "kind": "parse_error",
            "provider": provider,
            "api": api,
            "operation": operation,
            "request_id": request_id,
            "stage": format!("{stage:?}"),
            "payload": payload,
            "error": error,
        }),
        RawTraceEntry::RequestError {
            provider,
            api,
            operation,
            request_id,
            kind,
            status,
            payload,
            error,
            error_debug,
            source_chain,
            is_timeout,
            is_connect,
            is_request,
            is_body,
            is_decode,
        } => json!({
            "kind": "request_error",
            "provider": provider,
            "api": api,
            "operation": operation,
            "request_id": request_id,
            "error_kind": format!("{kind:?}"),
            "status": status,
            "payload": payload,
            "error": error,
            "error_debug": error_debug,
            "source_chain": source_chain,
            "is_timeout": is_timeout,
            "is_connect": is_connect,
            "is_request": is_request,
            "is_body": is_body,
            "is_decode": is_decode,
        }),
        RawTraceEntry::CollectError {
            operation_kind,
            request_id,
            kind,
            partial_summary,
            error,
        } => json!({
            "kind": "collect_error",
            "operation_kind": format!("{operation_kind:?}"),
            "request_id": request_id,
            "collect_kind": format!("{kind:?}"),
            "partial_summary": partial_summary,
            "error": error,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
@ defaults
max_usd = 0.05
text_max_output_tokens = 8
structured_max_output_tokens = 64
claude_thinking_budget_tokens = 1024
openai_reasoning_effort = "minimal"

@ endpoints.openrouter_responses_qwen35_9b
adapter = "openai-responses"
base_url = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"
model = "qwen/qwen3.5-9b"
cases = ["text", "structured", "tool", "thinking_roundtrip"]
expects_visible_thinking = true
fallback_models = ["qwen/qwen3.5-9b", "openai/gpt-5.4-nano"]
input_price_per_million = 0.03
output_price_per_million = 0.06
"#;

    fn sample_config() -> SmokeConfig {
        eure::parse_content(SAMPLE, PathBuf::from("sample.eure")).unwrap()
    }

    #[test]
    fn parses_eure_config() {
        let config = sample_config();
        assert_eq!(config.defaults.text_max_output_tokens, 8);
        let endpoint = config
            .endpoints
            .get("openrouter_responses_qwen35_9b")
            .unwrap();
        assert_eq!(endpoint.adapter, "openai-responses");
        assert!(endpoint.expects_visible_thinking);
        assert_eq!(endpoint.fallback_models.len(), 2);
    }

    #[test]
    fn parses_default_eure_config() {
        let content = include_str!("../../../data/adapter-smoke-config.eure");
        let config: SmokeConfig =
            eure::parse_content(content, PathBuf::from("data/adapter-smoke-config.eure")).unwrap();
        let cases = expand_cases(&config).unwrap();
        assert!(
            cases.iter().any(|case| {
                case.case_id == "openai_responses_gpt54_nano:structured_completion"
            })
        );
        assert!(cases.iter().any(|case| {
            case.case_id == "openrouter_responses_fallback:text"
                && !case.endpoint.fallback_models.is_empty()
        }));
    }

    #[test]
    fn expands_matrix_case_ids() {
        let config = sample_config();
        let cases = expand_cases(&config).unwrap();
        let ids = cases
            .iter()
            .map(|case| case.case_id.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            ids,
            vec![
                "openrouter_responses_qwen35_9b:text",
                "openrouter_responses_qwen35_9b:structured",
                "openrouter_responses_qwen35_9b:tool",
                "openrouter_responses_qwen35_9b:thinking_roundtrip",
            ]
        );
    }

    #[test]
    fn filters_by_endpoint_case_or_substring() {
        let config = sample_config();
        let cases = expand_cases(&config).unwrap();
        assert_eq!(
            filter_cases(cases.clone(), &["structured".into()])
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            filter_cases(cases.clone(), &["openrouter_responses_qwen35_9b".into()])
                .unwrap()
                .len(),
            4
        );
        assert!(filter_cases(cases, &["missing".into()]).is_err());
    }

    #[test]
    fn estimates_configured_cost() {
        let config = sample_config();
        let cases = expand_cases(&config).unwrap();
        let estimate = estimate_cases_usd(&cases, &config.defaults);
        assert!(estimate > 0.0);
        assert!(estimate < config.defaults.max_usd);
    }

    #[test]
    fn validates_completions_case_shape() {
        assert!(validate_case("bad", AdapterKind::OpenAiCompletions, SmokeCase::Text).is_err());
        assert!(validate_case("ok", AdapterKind::OpenAiCompletions, SmokeCase::Completion).is_ok());
    }

    #[test]
    fn resolves_credentials_with_skip_strict_policy() {
        let mut endpoint = sample_config()
            .endpoints
            .remove("openrouter_responses_qwen35_9b")
            .unwrap();
        assert!(resolve_api_key_with(&endpoint, |_| None).is_err());
        assert_eq!(missing_credential_status(false), CaseStatus::Skipped);
        assert_eq!(missing_credential_status(true), CaseStatus::Failed);

        endpoint.fallback_api_key = Some("local".into());
        assert_eq!(resolve_api_key_with(&endpoint, |_| None).unwrap(), "local");
        assert_eq!(
            resolve_api_key_with(&endpoint, |_| Some("env-key".into())).unwrap(),
            "env-key"
        );
    }
}
