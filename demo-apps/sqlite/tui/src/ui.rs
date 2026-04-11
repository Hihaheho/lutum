use lutum::{InputMessageRole, MessageContent, ModelInputItem, TurnItemIter, TurnRole};
use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Cell, Clear, Paragraph, Row, Table, Wrap},
};

use sqlite_agent::{QueryResult, SqlHistoryEntry, TransactionMode, WritePreview};

use crate::app::{AppState, TuiApp};

pub fn render(f: &mut Frame, app: &TuiApp) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // status bar
            Constraint::Min(0),    // main area
            Constraint::Length(3), // input
        ])
        .split(f.area());

    render_status_bar(f, chunks[0], app);
    render_main(f, chunks[1], app);
    render_input(f, chunks[2], app);

    if let AppState::Approval(ref preview) = app.state {
        render_approval_modal(f, f.area(), preview);
    }
    if let AppState::ModeRequest(ref reason) = app.state {
        render_mode_request_modal(f, f.area(), reason);
    }
}

fn render_status_bar(f: &mut Frame, area: Rect, app: &TuiApp) {
    let mode = app.current_mode();
    let mode_str = match mode {
        TransactionMode::ReadOnly => "● Read-Only",
        TransactionMode::Writable => "● Writable ",
    };
    let mode_color = match mode {
        TransactionMode::ReadOnly => Color::Yellow,
        TransactionMode::Writable => Color::Green,
    };
    let thinking = if matches!(app.state, AppState::Running) {
        " Thinking…"
    } else {
        ""
    };

    let stats = &app.token_stats;
    let tok_str = format!(
        " in:{} out:{} ↑cache:{} ↓cache:{}",
        stats.input_tokens,
        stats.output_tokens,
        stats.cache_creation_tokens,
        stats.cache_read_tokens,
    );

    let line = Line::from(vec![
        Span::styled(
            " sqlite-agent ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("│ "),
        Span::styled(mode_str, Style::default().fg(mode_color)),
        Span::styled(tok_str, Style::default().fg(Color::DarkGray)),
        Span::raw(thinking),
    ]);
    f.render_widget(Paragraph::new(line), area);
}

fn render_main(f: &mut Frame, area: Rect, app: &TuiApp) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
        .split(area);

    render_conversation(f, cols[0], app);
    render_sql_history(f, cols[1], app);
}

// ---------------------------------------------------------------------------
// Conversation — data-driven from session items
// ---------------------------------------------------------------------------

fn render_conversation(f: &mut Frame, area: Rect, app: &TuiApp) {
    let block = Block::default()
        .title(" Conversation ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray));

    let inner = block.inner(area);
    f.render_widget(block, area);

    let mut lines: Vec<Line> = vec![];

    for item in app.display_session.input().items() {
        match item {
            ModelInputItem::Message {
                role: InputMessageRole::System,
                ..
            } => {
                // skip system prompt
            }
            ModelInputItem::Message {
                role: InputMessageRole::User,
                content,
            } => {
                let text = content
                    .iter()
                    .map(|c| {
                        let MessageContent::Text(t) = c;
                        t.as_str()
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
                for (i, part) in wrap_text(&text, "You: ").into_iter().enumerate() {
                    lines.push(Line::from(Span::styled(
                        part,
                        if i == 0 {
                            Style::default()
                                .fg(Color::Cyan)
                                .add_modifier(Modifier::BOLD)
                        } else {
                            Style::default().fg(Color::Cyan)
                        },
                    )));
                }
                lines.push(Line::default());
            }
            ModelInputItem::Turn(committed_turn) => {
                // Walk items in this committed turn.
                let mut has_text = false;
                let mut text_buf = String::new();

                for item in TurnItemIter::new(committed_turn.as_ref()) {
                    if let Some(text) = item.as_text() {
                        if !text.is_empty() {
                            text_buf.push_str(text);
                            has_text = true;
                        }
                    } else if let Some(tc) = item.as_tool_call() {
                        // Flush accumulated text first.
                        if has_text {
                            emit_assistant_text(&text_buf, &mut lines);
                            text_buf.clear();
                            has_text = false;
                        }
                        // Show tool call.
                        let args_summary = summarize_args(tc.arguments.get());
                        lines.push(Line::from(vec![
                            Span::styled(
                                format!(" ⚙ {} ", tc.name),
                                Style::default()
                                    .fg(Color::Magenta)
                                    .add_modifier(Modifier::BOLD),
                            ),
                            Span::styled(args_summary, Style::default().fg(Color::DarkGray)),
                        ]));
                    }
                }

                // Flush any remaining text (final response).
                if has_text {
                    emit_assistant_text(&text_buf, &mut lines);
                    lines.push(Line::default());
                } else if committed_turn.role() == TurnRole::Assistant && !lines.is_empty() {
                    // Tool-call-only turn — add spacing.
                    lines.push(Line::default());
                }
            }
            ModelInputItem::ToolResult(tool_result) => {
                // Tool result following a committed tool-call turn.
                let summary = summarize_result(tool_result.result.get());
                lines.push(Line::from(vec![
                    Span::styled(" ✓ ", Style::default().fg(Color::Green)),
                    Span::styled(
                        format!("{}: ", tool_result.name),
                        Style::default().fg(Color::Green),
                    ),
                    Span::styled(summary, Style::default().fg(Color::DarkGray)),
                ]));
            }
            _ => {}
        }
    }

    // In-progress streaming response.
    if !app.streaming_text.is_empty() {
        emit_assistant_text(&app.streaming_text, &mut lines);
        // Blinking cursor indicator.
        lines.push(Line::from(Span::styled(
            "▍",
            Style::default().fg(Color::White),
        )));
    }

    // Turn errors.
    for err in &app.turn_errors {
        lines.push(Line::from(Span::styled(
            format!("Error: {err}"),
            Style::default().fg(Color::Red),
        )));
        lines.push(Line::default());
    }

    // Compute wrapped-line count for correct scroll clamping.
    let wrap_width = inner.width as usize;
    let wrapped_total: usize = lines
        .iter()
        .map(|l| {
            let w = l.width();
            if wrap_width == 0 || w == 0 {
                1
            } else {
                w.div_ceil(wrap_width)
            }
        })
        .sum();
    let height = inner.height as usize;
    let max_offset = wrapped_total.saturating_sub(height);

    if app.scroll_to_bottom.get() {
        app.scroll_to_bottom.set(false);
        app.scroll.set(max_offset);
    }
    let offset = app.scroll.get().min(max_offset);

    let para = Paragraph::new(Text::from(lines))
        .scroll((offset as u16, 0))
        .wrap(Wrap { trim: false });
    f.render_widget(para, inner);
}

/// Emit wrapped assistant text lines.
fn emit_assistant_text(text: &str, lines: &mut Vec<Line>) {
    let prefix = "Assistant: ";
    for (i, part) in wrap_text(text, prefix).into_iter().enumerate() {
        lines.push(Line::from(Span::styled(
            part,
            if i == 0 {
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::White)
            },
        )));
    }
}

/// Prepend `prefix` to the first line of `text` and return display lines.
fn wrap_text(text: &str, prefix: &str) -> Vec<String> {
    let mut result = Vec::new();
    let full = format!("{prefix}{text}");
    // Split on newlines for logical wrapping; ratatui handles terminal wrapping.
    for line in full.lines() {
        result.push(line.to_string());
    }
    if result.is_empty() {
        result.push(prefix.to_string());
    }
    result
}

/// Truncate `s` to at most `max_chars` Unicode scalar values, appending `…` if truncated.
fn truncate(s: &str, max_chars: usize) -> String {
    let mut chars = s.chars();
    let mut out = String::new();
    for _ in 0..max_chars {
        match chars.next() {
            Some(c) => out.push(c),
            None => return out,
        }
    }
    if chars.next().is_some() {
        out.push('…');
    }
    out
}

/// Extract a human-readable summary from tool argument JSON.
fn summarize_args(json: &str) -> String {
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(json) {
        if let Some(sql) = v.get("sql").and_then(|s| s.as_str()) {
            return truncate(&sql.replace('\n', " "), 80);
        }
        if let Some(db_id) = v.get("db_id").and_then(|s| s.as_str()) {
            return format!("db={db_id}");
        }
    }
    truncate(json, 60)
}

/// Extract a human-readable summary from tool result JSON.
fn summarize_result(json: &str) -> String {
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(json) {
        // error field
        if let Some(e) = v.get("error").and_then(|s| s.as_str()) {
            return format!("error: {e}");
        }
        // QueryResult: columns + rows
        if let (Some(cols), Some(rows)) = (v.get("columns"), v.get("rows"))
            && let (Some(nc), Some(nr)) = (cols.as_array(), rows.as_array())
        {
            if nc.len() == 1
                && nc[0].as_str() == Some("error")
                && let Some(first_row) = nr.first().and_then(|r| r.as_array())
                && let Some(msg) = first_row.first().and_then(|m| m.as_str())
            {
                return format!("error: {msg}");
            }
            return format!("{} rows, {} cols", nr.len(), nc.len());
        }
        // ModifyResult
        if let Some(rows) = v.get("rows_affected").and_then(|n| n.as_u64()) {
            if let Some(msg) = v.get("message").and_then(|s| s.as_str())
                && (msg.starts_with("rejected") || msg.starts_with("error"))
            {
                return msg.to_string();
            }
            return format!("{rows} row(s) affected");
        }
        // DdlResult / CreateDatabaseResult
        if let Some(msg) = v.get("message").and_then(|s| s.as_str()) {
            return msg.to_string();
        }
        // SchemaInfo: list table names
        if let Some(tables) = v.get("tables").and_then(|t| t.as_array()) {
            let names: Vec<_> = tables
                .iter()
                .filter_map(|t| t.get("name").and_then(|n| n.as_str()))
                .collect();
            return if names.is_empty() {
                "empty schema".to_string()
            } else {
                format!("tables: {}", names.join(", "))
            };
        }
        // ListDatabasesResult
        if let Some(dbs) = v.get("databases").and_then(|d| d.as_array()) {
            let ids: Vec<_> = dbs
                .iter()
                .filter_map(|d| d.get("db_id").and_then(|i| i.as_str()))
                .collect();
            return format!("databases: {}", ids.join(", "));
        }
    }
    truncate(json, 60)
}

// ---------------------------------------------------------------------------
// SQL history pane (right column)
// ---------------------------------------------------------------------------

fn render_sql_history(f: &mut Frame, area: Rect, app: &TuiApp) {
    let count = app.sql_history.len();
    let title = if count == 0 {
        " SQL History ".to_string()
    } else {
        format!(" SQL History ({count}) ")
    };
    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray));
    let inner = block.inner(area);
    f.render_widget(block, area);

    if app.sql_history.is_empty() {
        f.render_widget(
            Paragraph::new("No SQL executed yet.").style(Style::default().fg(Color::DarkGray)),
            inner,
        );
        return;
    }

    let lines = build_history_lines(&app.sql_history);

    // Scroll with auto-follow (same pattern as conversation pane).
    let wrap_width = inner.width as usize;
    let wrapped_total: usize = lines
        .iter()
        .map(|l| {
            let w = l.width();
            if wrap_width == 0 || w == 0 {
                1
            } else {
                w.div_ceil(wrap_width)
            }
        })
        .sum();
    let height = inner.height as usize;
    let max_offset = wrapped_total.saturating_sub(height);

    if app.result_scroll_to_bottom.get() {
        app.result_scroll_to_bottom.set(false);
        app.result_scroll.set(max_offset);
    }
    let offset = app.result_scroll.get().min(max_offset);

    let para = Paragraph::new(Text::from(lines))
        .scroll((offset as u16, 0))
        .wrap(Wrap { trim: false });
    f.render_widget(para, inner);
}

/// Build display lines for all SQL history entries.
fn build_history_lines(history: &[SqlHistoryEntry]) -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>> = Vec::new();

    for (i, entry) in history.iter().enumerate() {
        // Separator between entries.
        if i > 0 {
            lines.push(Line::from(Span::styled(
                "─".repeat(40),
                Style::default().fg(Color::DarkGray),
            )));
        }

        // SQL label — cyan for SELECT (rows_affected = None, has columns), yellow for writes/DDL.
        let is_select = entry.rows_affected.is_none() && !entry.result.columns.is_empty()
            || (entry.rows_affected.is_none()
                && entry.result.columns.is_empty()
                && entry.sql.trim_start().to_uppercase().starts_with("SELECT"));
        let label_color = if is_select {
            Color::Cyan
        } else {
            Color::Yellow
        };
        let sql_preview = truncate(&entry.sql.replace('\n', " "), 60);
        lines.push(Line::from(vec![
            Span::styled("▶ ", Style::default().fg(label_color)),
            Span::styled(
                sql_preview,
                Style::default()
                    .fg(label_color)
                    .add_modifier(Modifier::BOLD),
            ),
        ]));

        // Result body.
        if let Some(n) = entry.rows_affected {
            // Write operation.
            lines.push(Line::from(Span::styled(
                format!("  ✓ {n} row(s) affected"),
                Style::default().fg(Color::Green),
            )));
        } else if !entry.result.columns.is_empty() {
            // SELECT with results — render as text table.
            lines.extend(format_query_result(&entry.result));
        } else {
            // DDL or SELECT with no rows.
            let msg = if entry.sql.trim_start().to_uppercase().starts_with("SELECT") {
                "  (no rows)"
            } else {
                "  ✓ done"
            };
            lines.push(Line::from(Span::styled(
                msg,
                Style::default().fg(Color::DarkGray),
            )));
        }
    }

    lines
}

/// Format a QueryResult as text lines (pipe-separated columns).
fn format_query_result(qr: &QueryResult) -> Vec<Line<'static>> {
    let mut lines = Vec::new();

    // Compute column widths.
    let mut widths: Vec<usize> = qr.columns.iter().map(|c| c.len()).collect();
    for row in &qr.rows {
        for (i, cell) in row.iter().enumerate() {
            if i < widths.len() {
                widths[i] = widths[i].max(cell.len());
            }
        }
    }

    let fmt_row = |cells: &[String]| -> String {
        cells
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let w = widths.get(i).copied().unwrap_or(c.len());
                format!("{:width$}", c, width = w)
            })
            .collect::<Vec<_>>()
            .join(" │ ")
    };

    // Header.
    let header = fmt_row(&qr.columns);
    lines.push(Line::from(Span::styled(
        format!("  {header}"),
        Style::default().add_modifier(Modifier::BOLD),
    )));

    // Divider.
    let divider: String = widths
        .iter()
        .map(|&w| "─".repeat(w))
        .collect::<Vec<_>>()
        .join("─┼─");
    lines.push(Line::from(Span::styled(
        format!("  {divider}"),
        Style::default().fg(Color::DarkGray),
    )));

    // Rows (cap at 50 to avoid overwhelming the pane).
    let shown = qr.rows.len().min(50);
    for row in &qr.rows[..shown] {
        let text = fmt_row(row);
        lines.push(Line::from(Span::raw(format!("  {text}"))));
    }
    if qr.rows.len() > shown {
        lines.push(Line::from(Span::styled(
            format!("  … {} more rows", qr.rows.len() - shown),
            Style::default().fg(Color::DarkGray),
        )));
    }

    lines
}

// Keep render_query_table for use inside the approval modal preview.
fn render_query_table(f: &mut Frame, area: Rect, qr: &QueryResult) {
    if qr.columns.is_empty() {
        f.render_widget(
            Paragraph::new("(empty result)").style(Style::default().fg(Color::DarkGray)),
            area,
        );
        return;
    }

    let header_cells: Vec<Cell> = qr
        .columns
        .iter()
        .map(|c| Cell::from(c.as_str()).style(Style::default().add_modifier(Modifier::BOLD)))
        .collect();
    let header = Row::new(header_cells)
        .style(Style::default().bg(Color::DarkGray))
        .height(1);

    let rows: Vec<Row> = qr
        .rows
        .iter()
        .map(|row| {
            Row::new(
                row.iter()
                    .map(|v| Cell::from(v.as_str()))
                    .collect::<Vec<_>>(),
            )
            .height(1)
        })
        .collect();

    let col_count = qr.columns.len().max(1);
    let widths = vec![Constraint::Ratio(1, col_count as u32); col_count];

    let table = Table::new(rows, widths)
        .header(header)
        .block(Block::default())
        .row_highlight_style(Style::default().add_modifier(Modifier::REVERSED));

    f.render_widget(table, area);
}

// ---------------------------------------------------------------------------
// Input bar
// ---------------------------------------------------------------------------

fn render_input(f: &mut Frame, area: Rect, app: &TuiApp) {
    let hint = match app.state {
        AppState::Approval(_) => " [y] Accept  [n] Reject  [e] Edit SQL  [Esc] Cancel",
        AppState::ModeRequest(_) => " [y] Grant write access  [n] Deny",
        AppState::Running => " Thinking…",
        _ => {
            " [Tab] mode  [Enter] send  [↑↓] scroll chat  [Alt+↑↓] scroll SQL  [Ctrl+N] new  [q] quit"
        }
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(hint);
    let inner = block.inner(area);
    f.render_widget(block, area);

    // Only show the textarea when idle (not in modal/running state).
    if matches!(app.state, AppState::Idle | AppState::Done) {
        f.render_widget(&app.textarea, inner);
    }
}

// ---------------------------------------------------------------------------
// Approval modal
// ---------------------------------------------------------------------------

fn centered_modal(area: Rect, width: u16, height: u16) -> Rect {
    let w = area.width.min(width);
    let h = area.height.min(height);
    Rect::new(
        (area.width.saturating_sub(w)) / 2,
        (area.height.saturating_sub(h)) / 2,
        w,
        h,
    )
}

fn render_mode_request_modal(f: &mut Frame, area: Rect, reason: &str) {
    let modal_area = centered_modal(area, 70, 10);

    f.render_widget(Clear, modal_area);

    let block = Block::default()
        .title(" Write Access Requested ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Yellow));
    let inner = block.inner(modal_area);
    f.render_widget(block, modal_area);

    let text = format!(
        "The agent wants to switch to Writable mode.\n\nReason: {reason}\n\n[y] Grant   [n] Deny"
    );
    f.render_widget(
        Paragraph::new(text)
            .style(Style::default().fg(Color::White))
            .wrap(Wrap { trim: false }),
        inner,
    );
}

fn render_approval_modal(f: &mut Frame, area: Rect, preview: &WritePreview) {
    let modal_area = centered_modal(area, 70, 20);

    f.render_widget(Clear, modal_area);

    let block = Block::default()
        .title(" Approval Required ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Yellow));
    let inner = block.inner(modal_area);
    f.render_widget(block, modal_area);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2), // SQL + rows affected
            Constraint::Min(0),    // sample table
            Constraint::Length(1), // controls
        ])
        .split(inner);

    let header = Paragraph::new(vec![
        Line::from(vec![
            Span::styled("SQL: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(preview.sql.chars().take(60).collect::<String>()),
        ]),
        Line::from(vec![
            Span::styled(
                "Rows affected: ",
                Style::default().add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                preview.rows_affected.to_string(),
                Style::default().fg(if preview.rows_affected > 0 {
                    Color::Red
                } else {
                    Color::Green
                }),
            ),
        ]),
    ]);
    f.render_widget(header, chunks[0]);

    if !preview.sample.columns.is_empty() {
        render_query_table(f, chunks[1], &preview.sample);
    }

    let controls = Paragraph::new(Line::from(vec![
        Span::styled("[y] Accept", Style::default().fg(Color::Green)),
        Span::raw("   "),
        Span::styled("[n] Reject", Style::default().fg(Color::Red)),
        Span::raw("   "),
        Span::styled("[e] Edit SQL", Style::default().fg(Color::Yellow)),
        Span::raw("   "),
        Span::styled("[Esc] Cancel", Style::default().fg(Color::DarkGray)),
    ]));
    f.render_widget(controls, chunks[2]);
}
