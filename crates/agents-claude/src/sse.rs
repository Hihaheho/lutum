use crate::error::ClaudeError;

#[derive(Default)]
pub(crate) struct ClaudeSseParser {
    buffer: Vec<u8>,
    event: Option<String>,
    data_lines: Vec<String>,
}

#[derive(Debug)]
pub(crate) struct ClaudeSseFrame {
    pub(crate) event: Option<String>,
    pub(crate) data: String,
}

impl ClaudeSseParser {
    pub(crate) fn push(&mut self, chunk: &[u8]) -> Result<Vec<ClaudeSseFrame>, ClaudeError> {
        self.buffer.extend_from_slice(chunk);
        let mut frames = Vec::new();

        while let Some(pos) = self.buffer.iter().position(|byte| *byte == b'\n') {
            let mut line = self.buffer.drain(..=pos).collect::<Vec<_>>();
            if line.last() == Some(&b'\n') {
                line.pop();
            }
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            let line = String::from_utf8(line).map_err(|err| ClaudeError::Sse {
                message: format!("invalid UTF-8 in SSE stream: {err}"),
            })?;

            if line.is_empty() {
                self.finish_frame(&mut frames);
                continue;
            }
            if line.starts_with(':') {
                continue;
            }

            let (field, value) = line
                .split_once(':')
                .map_or((line.as_str(), ""), |(field, value)| {
                    (field, value.strip_prefix(' ').unwrap_or(value))
                });
            match field {
                "event" => self.event = Some(value.to_string()),
                "data" => self.data_lines.push(value.to_string()),
                _ => {}
            }
        }

        Ok(frames)
    }

    fn finish_frame(&mut self, frames: &mut Vec<ClaudeSseFrame>) {
        if self.event.is_none() && self.data_lines.is_empty() {
            return;
        }

        frames.push(ClaudeSseFrame {
            event: self.event.take(),
            data: self.data_lines.join("\n"),
        });
        self.data_lines.clear();
    }
}
