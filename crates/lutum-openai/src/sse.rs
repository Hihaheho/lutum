use crate::error::OpenAiError;

#[derive(Default)]
pub(crate) struct SseParser {
    buffer: Vec<u8>,
    data_lines: Vec<String>,
}

impl SseParser {
    pub(crate) fn push(&mut self, chunk: &[u8]) -> Result<Vec<String>, OpenAiError> {
        self.buffer.extend_from_slice(chunk);
        let mut frames = Vec::new();

        while let Some(pos) = self.buffer.iter().position(|byte| *byte == b'\n') {
            let mut line = self.buffer.drain(..=pos).collect::<Vec<_>>();
            if matches!(line.last(), Some(b'\n')) {
                line.pop();
            }
            if matches!(line.last(), Some(b'\r')) {
                line.pop();
            }
            if line.is_empty() {
                if !self.data_lines.is_empty() {
                    frames.push(self.data_lines.join("\n"));
                    self.data_lines.clear();
                }
                continue;
            }
            if let Some(rest) = line.strip_prefix(b"data:") {
                let rest = if rest.first() == Some(&b' ') {
                    &rest[1..]
                } else {
                    rest
                };
                let text = String::from_utf8(rest.to_vec()).map_err(|err| OpenAiError::Sse {
                    message: err.to_string(),
                })?;
                self.data_lines.push(text);
            }
        }

        Ok(frames)
    }
}
