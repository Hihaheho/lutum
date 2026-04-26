use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    process,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use lutum::{
    CompletionOptions, Lutum, ModelName, RawTelemetryConfig, RequestExtensions, Session,
    SharedPoolBudgetManager, SharedPoolBudgetOptions, StructuredTurnOutcome,
    TextStepOutcomeWithTools, Usage, UsageEstimate, UsageRecoveryAdapter,
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
    #[arg(long)]
    save_summary: Option<PathBuf>,
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
    let raw_path = resolve_raw_dump_path(args.save_raw.as_deref())?;
    let summary_path = resolve_summary_path(args.save_summary.as_deref(), &raw_path);
    prepare_raw_dump(&raw_path)
        .with_context(|| format!("failed to initialize raw dump {}", raw_path.display()))?;

    println!(
        "running {} adapter smoke cases; estimated worst-case cost ${estimate:.4} (cap ${cap:.4})",
        selected.len()
    );
    println!("raw dump: {}", raw_path.display());
    println!("summary eure: {}", summary_path.display());

    let mut reports = Vec::new();
    for case in selected {
        let report = run_case(case, &config.defaults, args.strict).await;
        write_raw(&raw_path, &report)
            .with_context(|| format!("failed to save raw trace to {}", raw_path.display()))?;
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

    write_summary_eure(
        &summary_path,
        &args.config,
        &raw_path,
        &reports,
        usage,
        estimate,
        cap,
    )
    .with_context(|| format!("failed to save summary eure {}", summary_path.display()))?;

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
            usage = usage.saturating_add(enrich_usage(&case.endpoint, case_usage, &raw));
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
            usage = usage.saturating_add(enrich_usage(&case.endpoint, Usage::zero(), &raw));
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

fn enrich_usage(endpoint: &EndpointConfig, adapter_usage: Usage, raw: &[RawTraceEntry]) -> Usage {
    let raw_usage = usage_from_raw(raw);
    let mut usage = if adapter_usage.total_tokens == 0 && raw_usage.total_tokens > 0 {
        raw_usage
    } else {
        adapter_usage
    };

    if usage.cache_creation_tokens == 0 {
        usage.cache_creation_tokens = raw_usage.cache_creation_tokens;
    }
    if usage.cache_read_tokens == 0 {
        usage.cache_read_tokens = raw_usage.cache_read_tokens;
    }
    if usage.cost_micros_usd == 0 {
        usage.cost_micros_usd = if raw_usage.cost_micros_usd > 0 {
            raw_usage.cost_micros_usd
        } else {
            price_usage_micros(endpoint, usage)
        };
    }
    usage
}

fn usage_from_raw(raw: &[RawTraceEntry]) -> Usage {
    raw.iter().fold(Usage::zero(), |usage, entry| {
        let RawTraceEntry::StreamEvent {
            event_name,
            payload,
            ..
        } = entry
        else {
            return usage;
        };
        let Ok(value) = serde_json::from_str::<serde_json::Value>(payload) else {
            return usage;
        };

        let next = if event_name.as_deref() == Some("message_delta") {
            value.get("usage").map(usage_from_token_value)
        } else if value.get("object").and_then(|object| object.as_str())
            == Some("chat.completion.chunk")
        {
            value.get("usage").map(usage_from_chat_value)
        } else if value.get("type").and_then(|kind| kind.as_str()) == Some("response.completed") {
            value
                .get("response")
                .and_then(|response| response.get("usage"))
                .map(usage_from_token_value)
        } else {
            None
        };

        usage.saturating_add(next.unwrap_or_else(Usage::zero))
    })
}

fn usage_from_token_value(value: &serde_json::Value) -> Usage {
    if !value.is_object() {
        return Usage::zero();
    }
    let cache_creation_tokens = value
        .get("cache_creation_input_tokens")
        .and_then(|tokens| tokens.as_u64())
        .unwrap_or_default();
    let cache_read_tokens = value
        .get("cache_read_input_tokens")
        .or_else(|| {
            value
                .get("input_tokens_details")
                .and_then(|details| details.get("cached_tokens"))
        })
        .and_then(|tokens| tokens.as_u64())
        .unwrap_or_default();
    let input_tokens = value
        .get("input_tokens")
        .and_then(|tokens| tokens.as_u64())
        .unwrap_or_default()
        .saturating_add(cache_creation_tokens)
        .saturating_add(cache_read_tokens);
    let output_tokens = value
        .get("output_tokens")
        .and_then(|tokens| tokens.as_u64())
        .unwrap_or_default();
    let total_tokens = value
        .get("total_tokens")
        .and_then(|tokens| tokens.as_u64())
        .unwrap_or_else(|| input_tokens.saturating_add(output_tokens));
    Usage {
        input_tokens,
        output_tokens,
        total_tokens,
        cost_micros_usd: value
            .get("cost")
            .and_then(|cost| cost.as_f64())
            .map(cost_to_micros_usd)
            .unwrap_or_default(),
        cache_creation_tokens,
        cache_read_tokens,
    }
}

fn usage_from_chat_value(value: &serde_json::Value) -> Usage {
    if !value.is_object() {
        return Usage::zero();
    }
    let details = value.get("prompt_tokens_details");
    let cache_creation_tokens = details
        .and_then(|details| details.get("cache_write_tokens"))
        .and_then(|tokens| tokens.as_u64())
        .unwrap_or_default();
    let cache_read_tokens = details
        .and_then(|details| details.get("cached_tokens"))
        .and_then(|tokens| tokens.as_u64())
        .unwrap_or_default();
    Usage {
        input_tokens: value
            .get("prompt_tokens")
            .and_then(|tokens| tokens.as_u64())
            .unwrap_or_default(),
        output_tokens: value
            .get("completion_tokens")
            .and_then(|tokens| tokens.as_u64())
            .unwrap_or_default(),
        total_tokens: value
            .get("total_tokens")
            .and_then(|tokens| tokens.as_u64())
            .unwrap_or_default(),
        cost_micros_usd: value
            .get("cost")
            .and_then(|cost| cost.as_f64())
            .map(cost_to_micros_usd)
            .unwrap_or_default(),
        cache_creation_tokens,
        cache_read_tokens,
    }
}

fn price_usage_micros(endpoint: &EndpointConfig, usage: Usage) -> u64 {
    let cost = usage.input_tokens as f64 * endpoint.input_price_per_million / 1_000_000.0
        + usage.output_tokens as f64 * endpoint.output_price_per_million / 1_000_000.0;
    cost_to_micros_usd(cost)
}

fn cost_to_micros_usd(cost: f64) -> u64 {
    if cost.is_finite() && cost > 0.0 {
        (cost * 1_000_000.0).ceil() as u64
    } else {
        0
    }
}

async fn run_completion(llm: &Lutum, case: &CaseSpec, defaults: &DefaultsConfig) -> Result<Usage> {
    let result = llm
        .completion("Return exactly OK.")
        .completion_options(CompletionOptions {
            max_output_tokens: Some(text_max_output_tokens(&case.endpoint, defaults)),
            stop: vec![".".to_string(), "\n".to_string()],
            ..CompletionOptions::default()
        })
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
    for _ in 0..4 {
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
        if raw_stream_contains(raw, &["redacted_thinking"])
            && !raw_request_contains(raw, &["\"type\":\"redacted_thinking\""])
        {
            bail!("Claude thinking_roundtrip replay did not preserve redacted thinking blocks");
        }
        if raw_stream_contains(raw, &["signature_delta"])
            && !raw_request_contains(raw, &["\"type\":\"thinking\""])
        {
            bail!("Claude thinking_roundtrip replay did not preserve signed thinking blocks");
        }
    }

    if case.case == SmokeCase::ThinkingRoundtrip && case.endpoint.expects_visible_thinking {
        if !raw_stream_contains(raw, &["thinking", "reasoning", "reasoning_content"]) {
            bail!(
                "endpoint expected visible thinking, but no thinking/reasoning stream payload decoded"
            );
        }
        if case.kind == AdapterKind::OpenAiResponses
            && raw_stream_contains(raw, &["\"type\":\"reasoning\""])
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

fn resolve_raw_dump_path(override_path: Option<&Path>) -> Result<PathBuf> {
    match override_path {
        Some(path) => Ok(path.to_path_buf()),
        None => Ok(default_raw_dump_path(
            current_unix_seconds()?,
            process::id(),
        )),
    }
}

fn resolve_summary_path(override_path: Option<&Path>, raw_path: &Path) -> PathBuf {
    override_path
        .map(Path::to_path_buf)
        .unwrap_or_else(|| raw_path.with_extension("eure"))
}

fn current_unix_seconds() -> Result<u64> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|err| anyhow!("system clock is before unix epoch: {err}"))?
        .as_secs())
}

fn default_raw_dump_path(unix_seconds: u64, pid: u32) -> PathBuf {
    PathBuf::from(format!(
        "/tmp/lutum-adapter-smoke-{unix_seconds}-{pid}.jsonl"
    ))
}

fn prepare_raw_dump(path: &Path) -> Result<()> {
    OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(path)
        .with_context(|| format!("open {}", path.display()))?;
    Ok(())
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

fn write_summary_eure(
    path: &Path,
    config_path: &Path,
    raw_path: &Path,
    reports: &[CaseReport],
    usage: Usage,
    estimate_usd: f64,
    cap_usd: f64,
) -> Result<()> {
    let failed = reports
        .iter()
        .filter(|report| matches!(report.status, CaseStatus::Failed))
        .count();
    let skipped = reports
        .iter()
        .filter(|report| matches!(report.status, CaseStatus::Skipped))
        .count();
    let passed = reports.len().saturating_sub(skipped + failed);

    let mut content = String::new();
    content.push_str("@ summary\n");
    push_eure_string(&mut content, "config", &config_path.display().to_string());
    push_eure_string(&mut content, "raw_dump", &raw_path.display().to_string());
    content.push_str(&format!("total_cases = {}\n", reports.len()));
    content.push_str(&format!("passed = {passed}\n"));
    content.push_str(&format!("skipped = {skipped}\n"));
    content.push_str(&format!("failed = {failed}\n"));
    content.push_str(&format!("usage_total_tokens = {}\n", usage.total_tokens));
    content.push_str(&format!("usage_input_tokens = {}\n", usage.input_tokens));
    content.push_str(&format!("usage_output_tokens = {}\n", usage.output_tokens));
    content.push_str(&format!("cost_micros_usd = {}\n", usage.cost_micros_usd));
    content.push_str(&format!(
        "cost_usd = {:.6}\n",
        usage.cost_micros_usd as f64 / 1_000_000.0
    ));
    content.push_str(&format!("estimated_worst_case_usd = {estimate_usd:.6}\n"));
    content.push_str(&format!("cap_usd = {cap_usd:.6}\n\n"));

    for (index, report) in reports.iter().enumerate() {
        content.push_str(&format!("@ cases.case_{:03}\n", index + 1));
        push_eure_string(&mut content, "id", &report.case_id);
        push_eure_string(&mut content, "status", case_status_str(&report.status));
        push_eure_string(&mut content, "message", &report.message);
        content.push_str(&format!("total_tokens = {}\n", report.usage.total_tokens));
        content.push_str(&format!("input_tokens = {}\n", report.usage.input_tokens));
        content.push_str(&format!("output_tokens = {}\n", report.usage.output_tokens));
        content.push_str(&format!(
            "cache_creation_tokens = {}\n",
            report.usage.cache_creation_tokens
        ));
        content.push_str(&format!(
            "cache_read_tokens = {}\n",
            report.usage.cache_read_tokens
        ));
        content.push_str(&format!(
            "cost_micros_usd = {}\n",
            report.usage.cost_micros_usd
        ));
        content.push_str(&format!(
            "cost_usd = {:.6}\n\n",
            report.usage.cost_micros_usd as f64 / 1_000_000.0
        ));
    }

    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(path)
        .with_context(|| format!("open {}", path.display()))?;
    file.write_all(content.as_bytes())?;
    Ok(())
}

fn case_status_str(status: &CaseStatus) -> &'static str {
    match status {
        CaseStatus::Passed => "passed",
        CaseStatus::Skipped => "skipped",
        CaseStatus::Failed => "failed",
    }
}

fn push_eure_string(content: &mut String, key: &str, value: &str) {
    content.push_str(key);
    content.push_str(" = ");
    content.push_str(&serde_json::to_string(value).expect("string serialization cannot fail"));
    content.push('\n');
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

@ endpoints.openrouter_responses_gemma4_26b
adapter = "openai-responses"
base_url = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"
model = "google/gemma-4-26b-a4b-it"
cases = ["text", "structured", "tool", "thinking_roundtrip"]
expects_visible_thinking = true
fallback_models = ["google/gemma-4-26b-a4b-it", "openai/gpt-5.4-nano"]
input_price_per_million = 0.08
output_price_per_million = 0.35
"#;

    fn sample_config() -> SmokeConfig {
        eure::parse_content(SAMPLE, PathBuf::from("sample.eure")).unwrap()
    }

    #[derive(Debug, eure::FromEure)]
    #[eure(crate = ::eure::document)]
    #[allow(dead_code)]
    struct SummaryDoc {
        summary: SummarySection,
        cases: BTreeMap<String, SummaryCase>,
    }

    #[derive(Debug, eure::FromEure)]
    #[eure(crate = ::eure::document)]
    #[allow(dead_code)]
    struct SummarySection {
        config: String,
        raw_dump: String,
        total_cases: u64,
        passed: u64,
        skipped: u64,
        failed: u64,
        usage_total_tokens: u64,
        usage_input_tokens: u64,
        usage_output_tokens: u64,
        cost_micros_usd: u64,
        cost_usd: f64,
        estimated_worst_case_usd: f64,
        cap_usd: f64,
    }

    #[derive(Debug, eure::FromEure)]
    #[eure(crate = ::eure::document)]
    #[allow(dead_code)]
    struct SummaryCase {
        id: String,
        status: String,
        message: String,
        total_tokens: u64,
        input_tokens: u64,
        output_tokens: u64,
        cache_creation_tokens: u64,
        cache_read_tokens: u64,
        cost_micros_usd: u64,
        cost_usd: f64,
    }

    #[test]
    fn default_raw_dump_path_uses_tmp_timestamp_and_pid() {
        assert_eq!(
            default_raw_dump_path(1_777_189_191, 4242),
            PathBuf::from("/tmp/lutum-adapter-smoke-1777189191-4242.jsonl")
        );
    }

    #[test]
    fn raw_dump_path_override_is_exact() {
        let path = PathBuf::from("/tmp/custom-smoke.jsonl");
        assert_eq!(resolve_raw_dump_path(Some(&path)).unwrap(), path);
    }

    #[test]
    fn default_summary_path_replaces_raw_extension() {
        let raw_path = PathBuf::from("/tmp/custom-smoke.jsonl");
        assert_eq!(
            resolve_summary_path(None, &raw_path),
            PathBuf::from("/tmp/custom-smoke.eure")
        );

        let summary_path = PathBuf::from("/tmp/custom-summary.eure");
        assert_eq!(
            resolve_summary_path(Some(&summary_path), &raw_path),
            summary_path
        );
    }

    #[test]
    fn write_raw_writes_case_dump_with_raw_and_trace() {
        let path = PathBuf::from(format!(
            "/tmp/lutum-adapter-smoke-write-test-{}-{}.jsonl",
            current_unix_seconds().unwrap(),
            process::id()
        ));
        prepare_raw_dump(&path).unwrap();
        let report = CaseReport {
            case_id: "endpoint:text".into(),
            status: CaseStatus::Passed,
            message: "ok".into(),
            usage: Usage {
                total_tokens: 3,
                ..Usage::zero()
            },
            raw: vec![RawTraceEntry::Request {
                provider: "openai".into(),
                api: "responses".into(),
                operation: "text_turn".into(),
                request_id: Some("req_1".into()),
                body: "{\"model\":\"test\"}".into(),
            }],
            trace: empty_trace(),
        };

        write_raw(&path, &report).unwrap();
        let content = fs::read_to_string(&path).unwrap();
        let lines = content.lines().collect::<Vec<_>>();
        assert_eq!(lines.len(), 1);
        let value = serde_json::from_str::<serde_json::Value>(lines[0]).unwrap();
        assert_eq!(value["kind"], "case_dump");
        assert_eq!(value["case_id"], "endpoint:text");
        assert_eq!(value["raw"].as_array().unwrap().len(), 1);
        assert!(value["trace"]["roots"].is_array());
    }

    #[test]
    fn write_summary_eure_writes_parseable_summary_and_cases() {
        let path = PathBuf::from(format!(
            "/tmp/lutum-adapter-smoke-summary-test-{}-{}.eure",
            current_unix_seconds().unwrap(),
            process::id()
        ));
        let reports = vec![
            CaseReport {
                case_id: "endpoint:text".into(),
                status: CaseStatus::Passed,
                message: "ok".into(),
                usage: Usage {
                    input_tokens: 10,
                    output_tokens: 5,
                    total_tokens: 15,
                    cost_micros_usd: 3,
                    cache_creation_tokens: 1,
                    cache_read_tokens: 2,
                },
                raw: Vec::new(),
                trace: empty_trace(),
            },
            CaseReport {
                case_id: "endpoint:structured".into(),
                status: CaseStatus::Failed,
                message: "schema \"bad\"".into(),
                usage: Usage {
                    input_tokens: 20,
                    output_tokens: 7,
                    total_tokens: 27,
                    cost_micros_usd: 5,
                    cache_creation_tokens: 0,
                    cache_read_tokens: 0,
                },
                raw: Vec::new(),
                trace: empty_trace(),
            },
        ];
        let usage = reports.iter().fold(Usage::zero(), |sum, report| {
            sum.saturating_add(report.usage)
        });

        write_summary_eure(
            &path,
            Path::new("data/adapter-smoke-config.eure"),
            Path::new("/tmp/raw.jsonl"),
            &reports,
            usage,
            0.0123,
            0.05,
        )
        .unwrap();

        let content = fs::read_to_string(&path).unwrap();
        let doc: SummaryDoc = eure::parse_content(&content, path).unwrap();
        assert_eq!(doc.summary.total_cases, 2);
        assert_eq!(doc.summary.passed, 1);
        assert_eq!(doc.summary.failed, 1);
        assert_eq!(doc.summary.cost_micros_usd, 8);
        assert_eq!(doc.cases["case_001"].id, "endpoint:text");
        assert_eq!(doc.cases["case_002"].status, "failed");
        assert_eq!(doc.cases["case_002"].message, "schema \"bad\"");
    }

    #[test]
    fn parses_eure_config() {
        let config = sample_config();
        assert_eq!(config.defaults.text_max_output_tokens, 8);
        let endpoint = config
            .endpoints
            .get("openrouter_responses_gemma4_26b")
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
                "openrouter_responses_gemma4_26b:text",
                "openrouter_responses_gemma4_26b:structured",
                "openrouter_responses_gemma4_26b:tool",
                "openrouter_responses_gemma4_26b:thinking_roundtrip",
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
            filter_cases(cases.clone(), &["openrouter_responses_gemma4_26b".into()])
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
    fn usage_from_raw_extracts_openrouter_cost() {
        let raw = vec![RawTraceEntry::StreamEvent {
            provider: "openrouter".into(),
            api: "responses".into(),
            operation: "text_turn".into(),
            request_id: Some("req_1".into()),
            sequence: 1,
            event_name: None,
            payload: json!({
                "type": "response.completed",
                "response": {
                    "usage": {
                        "input_tokens": 3,
                        "output_tokens": 4,
                        "total_tokens": 7,
                        "cost": 0.000012,
                    }
                }
            })
            .to_string(),
        }];

        let usage = usage_from_raw(&raw);
        assert_eq!(usage.input_tokens, 3);
        assert_eq!(usage.output_tokens, 4);
        assert_eq!(usage.total_tokens, 7);
        assert_eq!(usage.cost_micros_usd, 12);
    }

    #[test]
    fn enrich_usage_uses_raw_cost_before_configured_pricing() {
        let endpoint = sample_config()
            .endpoints
            .remove("openrouter_responses_gemma4_26b")
            .unwrap();
        let raw = vec![RawTraceEntry::StreamEvent {
            provider: "openrouter".into(),
            api: "responses".into(),
            operation: "text_turn".into(),
            request_id: Some("req_1".into()),
            sequence: 1,
            event_name: None,
            payload: json!({
                "type": "response.completed",
                "response": {
                    "usage": {
                        "input_tokens": 3,
                        "output_tokens": 4,
                        "total_tokens": 7,
                        "cost": 0.000012,
                    }
                }
            })
            .to_string(),
        }];

        let usage = enrich_usage(
            &endpoint,
            Usage {
                input_tokens: 1_000_000,
                output_tokens: 2_000_000,
                total_tokens: 3_000_000,
                ..Usage::zero()
            },
            &raw,
        );
        assert_eq!(usage.total_tokens, 3_000_000);
        assert_eq!(usage.cost_micros_usd, 12);
    }

    #[test]
    fn enrich_usage_prices_tokens_when_provider_cost_is_missing() {
        let endpoint = sample_config()
            .endpoints
            .remove("openrouter_responses_gemma4_26b")
            .unwrap();
        let usage = enrich_usage(
            &endpoint,
            Usage {
                input_tokens: 1_000_000,
                output_tokens: 2_000_000,
                total_tokens: 3_000_000,
                ..Usage::zero()
            },
            &[],
        );
        assert_eq!(usage.cost_micros_usd, 780_000);
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
            .remove("openrouter_responses_gemma4_26b")
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
