use anyhow::Context as _;
use lutum::*;
use lutum_openai::OpenAiAdapter;
use std::{fs, sync::Arc};

const NAVIGATION_SYSTEM: &str = "You are a codebase guide. Read the AGENTS.md content. The user has a question. Identify which crate is most relevant. Reply with ONLY the crate name (e.g. `lutum-openai`), nothing else.";
const ANSWER_SYSTEM: &str = "Answer the question using only the provided crate detail. Be concise.";
const QUESTION: &str = "Which crate should I edit to fix a bug in the OpenAI SSE parser?";

async fn ask(llm: &Lutum, system: &str, user: impl Into<String>) -> anyhow::Result<String> {
    let mut session = Session::new(llm.clone());
    session.push_system(system);
    session.push_user(user);
    let result = session.text_turn().collect().await?;
    Ok(result.assistant_text())
}

fn read_agents_file() -> anyhow::Result<String> {
    let cwd = std::env::current_dir().context("failed to determine current directory")?;
    fs::read_to_string("AGENTS.md").with_context(|| {
        format!(
            "failed to read AGENTS.md from {}; run this example from the repository root",
            cwd.display()
        )
    })
}

fn parse_crate_name(raw: &str) -> anyhow::Result<String> {
    let first_line = raw
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .ok_or_else(|| anyhow::anyhow!("navigation turn returned an empty response"))?;

    let cleaned = first_line.trim_matches(|c: char| matches!(c, '`' | '"' | '\''));
    let candidate = cleaned
        .strip_prefix("crates/")
        .unwrap_or(cleaned)
        .split_whitespace()
        .next()
        .unwrap_or("")
        .trim_end_matches([',', '.', ':', ';', '`', '|'])
        .to_string();

    if candidate.is_empty() {
        anyhow::bail!("navigation turn did not return a crate name");
    }

    Ok(candidate)
}

fn extract_crate_detail(agents_content: &str, crate_name: &str) -> anyhow::Result<String> {
    let lines: Vec<&str> = agents_content.lines().collect();
    let row_index = lines
        .iter()
        .position(|line| table_row_crate_name(line) == Some(crate_name))
        .ok_or_else(|| anyhow::anyhow!("could not find crate `{crate_name}` in AGENTS.md"))?;

    let table_start = find_table_start(&lines, row_index);
    let table_end = find_table_end(&lines, row_index);

    let mut section = Vec::new();

    if table_start >= 1 {
        let intro = lines[table_start - 1].trim();
        if !intro.is_empty() {
            section.push(lines[table_start - 1]);
        }
    }

    for line in lines
        .iter()
        .take(table_start.saturating_add(1).min(table_end) + 1)
        .skip(table_start)
    {
        section.push(line);
    }
    section.push(lines[row_index]);

    if let Some(paragraph) = adjacent_paragraph(&lines, table_start, Direction::Backward)
        && paragraph
            .iter()
            .any(|line| line_mentions_crate(line, crate_name))
    {
        append_block(&mut section, paragraph);
    }

    if let Some(paragraph) = adjacent_paragraph(&lines, table_end, Direction::Forward)
        && paragraph
            .iter()
            .any(|line| line_mentions_crate(line, crate_name))
    {
        append_block(&mut section, paragraph);
    }

    Ok(section.join("\n"))
}

fn table_row_crate_name(line: &str) -> Option<&str> {
    let trimmed = line.trim();
    if !trimmed.starts_with('|') {
        return None;
    }

    let first_cell = trimmed
        .split('|')
        .map(str::trim)
        .find(|cell| !cell.is_empty())?;
    let crate_path = first_cell.trim_matches('`');

    crate_path.strip_prefix("crates/")
}

fn line_mentions_crate(line: &str, crate_name: &str) -> bool {
    let crate_path = format!("crates/{crate_name}");
    line.split(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '-' | '/' | '_')))
        .any(|token| token == crate_name || token == crate_path)
}

fn find_table_start(lines: &[&str], row_index: usize) -> usize {
    let mut idx = row_index;
    while idx > 0 && lines[idx - 1].trim_start().starts_with('|') {
        idx -= 1;
    }
    idx
}

fn find_table_end(lines: &[&str], row_index: usize) -> usize {
    let mut idx = row_index;
    while idx + 1 < lines.len() && lines[idx + 1].trim_start().starts_with('|') {
        idx += 1;
    }
    idx
}

fn append_block<'a>(section: &mut Vec<&'a str>, block: Vec<&'a str>) {
    if !section.is_empty() && !section.last().is_some_and(|line| line.trim().is_empty()) {
        section.push("");
    }
    section.extend(block);
}

fn adjacent_paragraph<'a>(
    lines: &'a [&'a str],
    anchor: usize,
    direction: Direction,
) -> Option<Vec<&'a str>> {
    match direction {
        Direction::Backward => {
            if anchor == 0 {
                return None;
            }

            let mut end = anchor;
            while end > 0 && lines[end - 1].trim().is_empty() {
                end -= 1;
            }
            if end == 0 || is_heading(lines[end - 1]) {
                return None;
            }

            let mut start = end - 1;
            while start > 0 && !lines[start - 1].trim().is_empty() && !is_heading(lines[start - 1])
            {
                start -= 1;
            }

            Some(lines[start..end].to_vec())
        }
        Direction::Forward => {
            let mut start = anchor + 1;
            while start < lines.len() && lines[start].trim().is_empty() {
                start += 1;
            }
            if start >= lines.len() || is_heading(lines[start]) {
                return None;
            }

            let mut end = start;
            while end + 1 < lines.len()
                && !lines[end + 1].trim().is_empty()
                && !is_heading(lines[end + 1])
            {
                end += 1;
            }

            Some(lines[start..=end].to_vec())
        }
    }
}

fn is_heading(line: &str) -> bool {
    line.trim_start().starts_with('#')
}

enum Direction {
    Backward,
    Forward,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let agents_content = read_agents_file()?;
    let endpoint = std::env::var("ENDPOINT").unwrap_or_else(|_| "http://localhost:11434/v1".into());
    let token = std::env::var("TOKEN").unwrap_or_else(|_| "local".into());
    let model_name = std::env::var("MODEL").unwrap_or_else(|_| "qwen3.5:2b".into());
    let model = ModelName::new(&model_name)?;
    let adapter = OpenAiAdapter::new(token)
        .with_base_url(endpoint)
        .with_default_model(model);
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let llm = Lutum::new(Arc::new(adapter), budget);

    let raw_crate = ask(
        &llm,
        NAVIGATION_SYSTEM,
        format!("AGENTS.md:\n{agents_content}\n\nQuestion: {QUESTION}"),
    )
    .await?;
    let crate_name = parse_crate_name(&raw_crate)?;
    let crate_detail = extract_crate_detail(&agents_content, &crate_name)?;

    println!("Navigated crate: {crate_name}");
    println!("Extracted section:\n{crate_detail}");

    let answer = ask(
        &llm,
        ANSWER_SYSTEM,
        format!("Crate detail:\n{crate_detail}\n\nQuestion: {QUESTION}"),
    )
    .await?;

    println!("Answer: {answer}");
    Ok(())
}
