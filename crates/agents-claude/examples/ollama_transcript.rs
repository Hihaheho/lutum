//! Ollama transcript example - Claude / Messages backend
//!
//! Requires a running Ollama instance with the qwen3.5 model:
//!   ollama pull qwen3.5
//!   cargo run --example ollama_transcript -p agents-claude
//!
//! NOTE: seed is not supported by Ollama's Anthropic-compatible endpoint.

use std::io::Write;
use std::sync::Arc;

use agents_claude::ClaudeAdapter;
use agents_protocol::{
    budget::Usage,
    conversation::{ModelInput, ModelInputItem},
    extensions::RequestExtensions,
    llm::{
        AdapterTextTurn, AdapterToolChoice, AdapterTurnConfig, ErasedTextTurnEvent,
        GenerationParams, ModelName, TurnAdapter,
    },
};
use console::style;
use futures::StreamExt;

const MODEL: &str = "qwen3.5";
const BASE_URL: &str = "http://localhost:11434";

struct TurnRecord {
    role: &'static str,
    text: String,
    usage: Option<Usage>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Ollama ignores the API key, but the adapter requires a non-empty value.
    let adapter = ClaudeAdapter::new("ollama").with_base_url(BASE_URL);

    let model = ModelName::new(MODEL)?;
    let generation = GenerationParams {
        temperature: None,
        max_output_tokens: Some(2048),
        seed: None,
    };

    let questions = [
        "What is 2+2? Answer in one sentence.",
        "Multiply that result by 10. Answer in one sentence.",
        "Summarize our conversation in one sentence.",
    ];

    let mut input = ModelInput::new();
    let mut records = Vec::new();

    for question in questions {
        records.push(TurnRecord {
            role: "USER",
            text: question.to_string(),
            usage: None,
        });

        input = input.user(question);

        let turn = AdapterTextTurn {
            config: AdapterTurnConfig {
                model: model.clone(),
                generation: generation.clone(),
                tools: vec![],
                tool_choice: AdapterToolChoice::None,
            },
            extensions: Arc::new(RequestExtensions::default()),
        };

        let mut stream = adapter.text_turn(input.clone(), turn).await?;
        let mut assistant_text = String::new();
        let mut turn_usage = None;

        while let Some(event) = stream.next().await {
            match event? {
                ErasedTextTurnEvent::TextDelta { delta } => {
                    print!("{delta}");
                    let _ = std::io::stdout().flush();
                    assistant_text.push_str(&delta);
                }
                ErasedTextTurnEvent::Completed {
                    committed_turn,
                    usage,
                    ..
                } => {
                    println!();
                    turn_usage = Some(usage);
                    input.push(ModelInputItem::turn(committed_turn));
                }
                _ => {}
            }
        }

        records.push(TurnRecord {
            role: "ASSISTANT",
            text: assistant_text,
            usage: turn_usage,
        });
    }

    println!();
    println!("{}", style("-".repeat(60)).dim());
    println!(
        "{}  model={}  base_url={}",
        style("Claude / Messages API").bold(),
        style(MODEL).cyan(),
        style(BASE_URL).dim(),
    );
    println!("{}", style("-".repeat(60)).dim());

    for record in &records {
        if record.role == "USER" {
            println!(
                "\n{} {}",
                style("USER").bold().cyan(),
                style(&record.text).dim(),
            );
        } else {
            println!("\n{} {}", style("ASSISTANT").bold().green(), record.text);
            if let Some(usage) = &record.usage {
                println!(
                    "   {}  in={} out={}",
                    style("tokens").dim(),
                    style(usage.input_tokens).dim(),
                    style(usage.output_tokens).dim(),
                );
            }
        }
    }
    println!("\n{}", style("-".repeat(60)).dim());

    Ok(())
}
