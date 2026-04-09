//! Session persistence for Claude-adapter sessions.
//!
//! [`ModelInput`] and [`ModelInputItem`] are execution types and do not implement
//! `Serialize` / `Deserialize` by design.  This module provides a lossless bridge:
//!
//! - [`snapshot`] converts a `ModelInput` into a `Vec<ClaudeModelInputItem>` that can be
//!   serialized with `serde_json` (or any other Serde format).
//! - [`restore`] reconstructs a `ModelInput` from the snapshot, wrapping each committed
//!   turn back into `Arc<ClaudeCommittedTurn>` so exact same-adapter replay works.
//!
//! # Example
//!
//! ```
//! use lutum_protocol::conversation::{InputMessageRole, ModelInput, ModelInputItem};
//! use lutum_claude::persistence::{snapshot, restore};
//!
//! // Build a small model input (no committed turns for this example).
//! let mut input = ModelInput::new();
//! input.push(ModelInputItem::text(InputMessageRole::System, "You are helpful."));
//! input.push(ModelInputItem::text(InputMessageRole::User, "Hello!"));
//!
//! let items = snapshot(&input).unwrap();
//! let json = serde_json::to_string(&items).unwrap();
//!
//! let restored_items: Vec<lutum_claude::persistence::ClaudeModelInputItem> =
//!     serde_json::from_str(&json).unwrap();
//! let restored = restore(restored_items);
//!
//! assert_eq!(input.items().len(), restored.items().len());
//! ```

use std::sync::Arc;

use lutum_protocol::conversation::{
    AssistantInputItem, InputMessageRole, MessageContent, ModelInput, ModelInputItem, NonEmpty,
    RawJson, ToolCallId, ToolName, ToolUse,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::messages::ClaudeCommittedTurn;

// ---------------------------------------------------------------------------
// Serializable snapshot types
// ---------------------------------------------------------------------------

/// A serializable snapshot of a single [`ModelInputItem`] for sessions backed by the Claude
/// adapter.
///
/// All variants round-trip losslessly.  The `Turn` variant carries the full
/// [`ClaudeCommittedTurn`], enabling exact same-adapter replay after restore.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClaudeModelInputItem {
    Message {
        role: InputMessageRole,
        content: NonEmpty<MessageContent>,
    },
    Assistant(AssistantInputItem),
    ToolUse(PersistedToolUse),
    Turn(ClaudeCommittedTurn),
}

/// Serializable form of [`ToolUse`].
///
/// [`ToolUse`] carries its `arguments` and `result` fields as [`RawJson`] (`Box<RawValue>`
/// under the hood).  Serde's internally-tagged enum machinery routes deserialization through
/// a `Content` intermediary that is incompatible with `RawValue`.  This type sidesteps the
/// issue by storing both fields as `serde_json::Value`, which converts losslessly to/from
/// `RawJson`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PersistedToolUse {
    pub id: ToolCallId,
    pub name: ToolName,
    pub arguments: serde_json::Value,
    pub result: serde_json::Value,
}

impl PersistedToolUse {
    fn from_tool_use(tu: &ToolUse) -> Self {
        Self {
            id: tu.id.clone(),
            name: tu.name.clone(),
            arguments: serde_json::from_str(tu.arguments.get()).expect("RawJson is valid JSON"),
            result: serde_json::from_str(tu.result.get()).expect("RawJson is valid JSON"),
        }
    }

    fn into_tool_use(self) -> ToolUse {
        ToolUse {
            id: self.id,
            name: self.name,
            arguments: RawJson::from_serializable(&self.arguments)
                .expect("serde_json::Value is valid JSON"),
            result: RawJson::from_serializable(&self.result)
                .expect("serde_json::Value is valid JSON"),
        }
    }
}

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Error returned when [`snapshot`] encounters a committed turn that is not a
/// [`ClaudeCommittedTurn`].
///
/// This happens when the session contains turns produced by a different adapter (e.g. OpenAI).
/// Only sessions driven exclusively by [`ClaudeAdapter`][crate::ClaudeAdapter] can be
/// snapshotted with this module.
#[derive(Debug, Error)]
#[error("committed turn at item index {index} is not a ClaudeCommittedTurn (wrong adapter?)")]
pub struct SnapshotError {
    pub index: usize,
}

// ---------------------------------------------------------------------------
// snapshot / restore
// ---------------------------------------------------------------------------

/// Convert a [`ModelInput`] into a serializable snapshot.
///
/// Each [`ModelInputItem::Turn`] is downcast to [`ClaudeCommittedTurn`].  Returns
/// [`SnapshotError`] if any turn was produced by a different adapter.
pub fn snapshot(input: &ModelInput) -> Result<Vec<ClaudeModelInputItem>, SnapshotError> {
    input
        .items()
        .iter()
        .enumerate()
        .map(|(index, item)| match item {
            ModelInputItem::Message { role, content } => Ok(ClaudeModelInputItem::Message {
                role: *role,
                content: content.clone(),
            }),
            ModelInputItem::Assistant(item) => Ok(ClaudeModelInputItem::Assistant(item.clone())),
            ModelInputItem::ToolUse(tool_use) => Ok(ClaudeModelInputItem::ToolUse(
                PersistedToolUse::from_tool_use(tool_use),
            )),
            ModelInputItem::Turn(committed_turn) => committed_turn
                .as_ref()
                .as_any()
                .downcast_ref::<ClaudeCommittedTurn>()
                .map(|t| ClaudeModelInputItem::Turn(t.clone()))
                .ok_or(SnapshotError { index }),
        })
        .collect()
}

/// Reconstruct a [`ModelInput`] from a previously snapshotted item list.
///
/// `Turn` items are wrapped in [`Arc`] and stored as `CommittedTurn` values backed by the
/// concrete [`ClaudeCommittedTurn`] type, so the adapter can downcast them for exact replay.
pub fn restore(items: Vec<ClaudeModelInputItem>) -> ModelInput {
    ModelInput::from_items(
        items
            .into_iter()
            .map(|item| match item {
                ClaudeModelInputItem::Message { role, content } => {
                    ModelInputItem::Message { role, content }
                }
                ClaudeModelInputItem::Assistant(item) => ModelInputItem::Assistant(item),
                ClaudeModelInputItem::ToolUse(p) => ModelInputItem::ToolUse(p.into_tool_use()),
                ClaudeModelInputItem::Turn(turn) => ModelInputItem::Turn(Arc::new(turn)),
            })
            .collect(),
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use lutum_protocol::{
        budget::Usage,
        conversation::{RawJson, ToolCallId, ToolName},
        llm::FinishReason,
    };

    use super::*;
    use crate::messages::turn::ClaudeTurnItem;

    fn make_committed_turn() -> ClaudeCommittedTurn {
        ClaudeCommittedTurn {
            request_id: Some("msg_test".to_string()),
            model: "claude-haiku-4-5-20251001".to_string(),
            items: vec![
                ClaudeTurnItem::Text {
                    content: "Hello!".to_string(),
                },
                ClaudeTurnItem::ToolCall {
                    id: ToolCallId::from("call_1"),
                    name: ToolName::from("search"),
                    arguments: RawJson::parse(r#"{"q":"rust"}"#).unwrap(),
                },
            ],
            finish_reason: FinishReason::ToolCall,
            usage: Usage {
                input_tokens: 10,
                output_tokens: 5,
                total_tokens: 15,
                cost_micros_usd: 0,
            },
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 0,
        }
    }

    #[test]
    fn round_trip_message_items() {
        let mut input = ModelInput::new();
        input.push(ModelInputItem::text(InputMessageRole::System, "sys"));
        input.push(ModelInputItem::text(InputMessageRole::User, "hello"));

        let snap = snapshot(&input).unwrap();
        let json = serde_json::to_string(&snap).unwrap();
        let items: Vec<ClaudeModelInputItem> = serde_json::from_str(&json).unwrap();
        let restored = restore(items);

        assert_eq!(input.items().len(), restored.items().len());
    }

    #[test]
    fn round_trip_committed_turn() {
        let turn = make_committed_turn();
        let committed: Arc<dyn lutum_protocol::transcript::TurnView + Send + Sync> =
            Arc::new(turn.clone());

        let mut input = ModelInput::new();
        input.push(ModelInputItem::text(InputMessageRole::User, "hi"));
        input.push(ModelInputItem::Turn(committed));

        let snap = snapshot(&input).unwrap();
        let json = serde_json::to_string(&snap).unwrap();
        let items: Vec<ClaudeModelInputItem> = serde_json::from_str(&json).unwrap();
        let restored = restore(items);

        // Verify the restored Turn is a ClaudeCommittedTurn and matches the original.
        let ModelInputItem::Turn(ct) = &restored.items()[1] else {
            panic!("expected Turn item");
        };
        let recovered = ct
            .as_ref()
            .as_any()
            .downcast_ref::<ClaudeCommittedTurn>()
            .expect("must downcast to ClaudeCommittedTurn");
        assert_eq!(*recovered, turn);
    }

    #[test]
    fn snapshot_with_tool_use_round_trip() {
        let mut input = ModelInput::new();
        input.push(ModelInputItem::text(InputMessageRole::User, "search"));
        input.push(ModelInputItem::tool_use_parts(
            "call_1",
            "search",
            RawJson::parse(r#"{"q":"rust"}"#).unwrap(),
            RawJson::parse(r#""result""#).unwrap(),
        ));

        let snap = snapshot(&input).unwrap();
        let json = serde_json::to_string(&snap).unwrap();
        let items: Vec<ClaudeModelInputItem> = serde_json::from_str(&json).unwrap();
        let restored = restore(items);

        assert_eq!(input.items().len(), restored.items().len());
        let ModelInputItem::ToolUse(tu) = &restored.items()[1] else {
            panic!("expected ToolUse");
        };
        assert_eq!(tu.name.as_str(), "search");
    }
}
