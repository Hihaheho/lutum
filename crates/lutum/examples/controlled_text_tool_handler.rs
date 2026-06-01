use std::fmt;
use std::sync::Arc;

use futures::executor::block_on;
use lutum::{
    AssistantTurnItem, FinishReason, MockLlmAdapter, MockTextScenario, RawJson, RawTextTurnEvent,
    RecoveredTextToolCalls, Session, SharedPoolBudgetManager, SharedPoolBudgetOptions,
    TextStepOutcomeWithTools, TextToolCollectError, TextToolErrorDirective, TextToolHandlerContext,
    TextToolHandlerDirective, TextTurnEventWithTools, ToolCallFallbackError, Usage,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct WeatherResult {
    forecast: String,
}

#[lutum::tool_input(name = "weather", output = WeatherResult)]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct WeatherArgs {
    city: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
enum Tools {
    Weather(WeatherArgs),
}

#[derive(Debug, Deserialize)]
struct JsonToolCandidate {
    id: Option<String>,
    name: String,
    arguments: serde_json::Value,
}

#[derive(Debug)]
enum HandlerError {
    Json(serde_json::Error),
    Tool(ToolCallFallbackError),
}

impl fmt::Display for HandlerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HandlerError::Json(source) => write!(f, "json error: {source}"),
            HandlerError::Tool(source) => write!(f, "tool recovery error: {source}"),
        }
    }
}

impl std::error::Error for HandlerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            HandlerError::Json(source) => Some(source),
            HandlerError::Tool(source) => Some(source),
        }
    }
}

impl From<serde_json::Error> for HandlerError {
    fn from(source: serde_json::Error) -> Self {
        HandlerError::Json(source)
    }
}

impl From<ToolCallFallbackError> for HandlerError {
    fn from(source: ToolCallFallbackError) -> Self {
        HandlerError::Tool(source)
    }
}

struct JsonBlockHandler {
    buffered_text: String,
    recovered: Option<RecoveredTextToolCalls<Tools>>,
    return_early: bool,
}

impl JsonBlockHandler {
    fn return_early() -> Self {
        Self {
            buffered_text: String::new(),
            recovered: None,
            return_early: true,
        }
    }

    fn recover_on_length_error() -> Self {
        Self {
            buffered_text: String::new(),
            recovered: None,
            return_early: false,
        }
    }
}

#[cfg_attr(target_family = "wasm", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_family = "wasm"), async_trait::async_trait)]
impl lutum::TextToolEventHandler<Tools> for JsonBlockHandler {
    type Error = HandlerError;

    async fn on_event(
        &mut self,
        event: &TextTurnEventWithTools<Tools>,
        cx: &TextToolHandlerContext<Tools>,
    ) -> Result<TextToolHandlerDirective<Tools>, Self::Error> {
        if let TextTurnEventWithTools::TextDelta { delta } = event {
            self.buffered_text.push_str(delta);

            if self.recovered.is_none()
                && let Some(candidate) = parse_json_tool_codeblock(&self.buffered_text)?
            {
                let recovered = recover_json_tool_call(cx, candidate)?;
                if self.return_early {
                    return Ok(TextToolHandlerDirective::Return(
                        lutum::SyntheticTextToolTurn::needs_tools(recovered),
                    ));
                }
                self.recovered = Some(recovered);
            }
        }

        Ok(TextToolHandlerDirective::Continue)
    }

    async fn on_error(
        &mut self,
        error: TextToolCollectError<'_>,
        _cx: &TextToolHandlerContext<Tools>,
    ) -> Result<TextToolErrorDirective<Tools>, Self::Error> {
        if matches!(
            error,
            TextToolCollectError::Reduction(lutum::TextTurnReductionError::OutputLimitExceeded(_))
        ) && let Some(recovered) = self.recovered.take()
        {
            return Ok(TextToolErrorDirective::Return(
                lutum::SyntheticTextToolTurn::needs_tools(recovered),
            ));
        }

        Ok(TextToolErrorDirective::Propagate)
    }
}

fn parse_json_tool_codeblock(text: &str) -> Result<Option<JsonToolCandidate>, serde_json::Error> {
    let Some(start) = text.find("```json") else {
        return Ok(None);
    };
    let body = &text[start + "```json".len()..];
    let Some(end) = body.find("```") else {
        return Ok(None);
    };

    serde_json::from_str(body[..end].trim()).map(Some)
}

fn recover_json_tool_call(
    cx: &TextToolHandlerContext<Tools>,
    candidate: JsonToolCandidate,
) -> Result<RecoveredTextToolCalls<Tools>, HandlerError> {
    let id = candidate.id.unwrap_or_else(|| "json-call-1".to_string());
    let arguments = RawJson::from_serializable(&candidate.arguments)?;

    Ok(cx.recover_tool_calls_from_items(vec![
        AssistantTurnItem::Text("Recovered a JSON tool request from assistant text.".into()),
        AssistantTurnItem::ToolCall {
            id: id.into(),
            name: candidate.name.into(),
            arguments,
        },
    ])?)
}

fn execute_weather(input: &WeatherArgs) -> WeatherResult {
    WeatherResult {
        forecast: format!("forecast for {}: 24C and sunny", input.city),
    }
}

async fn run_early_return() -> Result<(), Box<dyn std::error::Error>> {
    let adapter = MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![
        Ok(RawTextTurnEvent::Started {
            request_id: Some("req-controlled-early".into()),
            model: "gpt-4.1-mini".into(),
        }),
        Ok(RawTextTurnEvent::TextDelta {
            delta: concat!(
                "I will call a tool using JSON.\n",
                "```json\n",
                r#"{"name":"weather","arguments":{"city":"Tokyo"}}"#,
                "\n```\n"
            )
            .into(),
        }),
        Ok(RawTextTurnEvent::Completed {
            request_id: Some("req-controlled-early".into()),
            finish_reason: FinishReason::Stop,
            usage: Usage {
                total_tokens: 99,
                ..Usage::zero()
            },
        }),
    ]));

    let ctx = lutum::Lutum::new(
        Arc::new(adapter),
        SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
    );
    let mut session = Session::new();
    session.push_user("Check the weather in Tokyo.");

    let outcome = session
        .text_turn(&ctx)
        .tools::<Tools>()
        .available_tools([ToolsSelector::Weather])
        .require_any_tool()
        .collect_controlled_with(JsonBlockHandler::return_early())
        .await?;

    let TextStepOutcomeWithTools::NeedsTools(round) = outcome else {
        panic!("expected a synthetic tool round");
    };

    println!(
        "early return: recovered {} tool call(s), usage total={}",
        round.tool_calls.len(),
        round.usage.total_tokens
    );

    let tool_results = round
        .tool_calls
        .iter()
        .cloned()
        .map(|call| match call {
            ToolsCall::Weather(call) => {
                println!("tool call: weather({})", call.input().city);
                let result = execute_weather(call.input());
                call.complete(result)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    round.commit(&mut session, tool_results)?;

    Ok(())
}

async fn run_length_recovery() -> Result<(), Box<dyn std::error::Error>> {
    let adapter = MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![
        Ok(RawTextTurnEvent::Started {
            request_id: Some("req-controlled-length".into()),
            model: "gpt-4.1-mini".into(),
        }),
        Ok(RawTextTurnEvent::TextDelta {
            delta: concat!(
                "Partial answer before the model hit its output cap.\n",
                "```json\n",
                r#"{"id":"json-weather-2","name":"weather","arguments":{"city":"Osaka"}}"#,
                "\n```\n"
            )
            .into(),
        }),
        Ok(RawTextTurnEvent::Completed {
            request_id: Some("req-controlled-length".into()),
            finish_reason: FinishReason::Length,
            usage: Usage {
                total_tokens: 42,
                ..Usage::zero()
            },
        }),
    ]));

    let ctx = lutum::Lutum::new(
        Arc::new(adapter),
        SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
    );
    let mut session = Session::new();
    session.push_user("Check the weather in Osaka.");

    let outcome = session
        .text_turn(&ctx)
        .tools::<Tools>()
        .available_tools([ToolsSelector::Weather])
        .require_any_tool()
        .collect_controlled_with(JsonBlockHandler::recover_on_length_error())
        .await?;

    let TextStepOutcomeWithTools::NeedsTools(round) = outcome else {
        panic!("expected output-limit recovery to synthesize a tool round");
    };

    println!(
        "length recovery: recovered {} tool call(s), usage total={}",
        round.tool_calls.len(),
        round.usage.total_tokens
    );

    Ok(())
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    run_early_return().await?;
    run_length_recovery().await?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    block_on(run())
}
