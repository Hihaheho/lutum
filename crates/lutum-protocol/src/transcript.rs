use std::{any::Any, fmt, sync::Arc};

use crate::conversation::{AssistantTurnItem, RawJson, ToolCallId, ToolName};

/// Role or category of a committed turn in the session transcript.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TurnRole {
    System,
    Developer,
    User,
    Assistant,
}

/// Borrowed view of a tool-call item's key fields.
pub struct ToolCallItemView<'a> {
    pub id: &'a ToolCallId,
    pub name: &'a ToolName,
    pub arguments: &'a RawJson,
}

/// Borrowed view of a tool-result item's key fields.
pub struct ToolResultItemView<'a> {
    pub id: &'a ToolCallId,
    pub name: &'a ToolName,
    pub arguments: &'a RawJson,
    pub result: &'a RawJson,
}

/// Read-only view of a single item within a committed turn.
///
/// Implementations should return `Some` for exactly one accessor that matches
/// the item's kind, and `None` for all others.
pub trait ItemView {
    /// Returns the text content of this item, if it is a text item.
    fn as_text(&self) -> Option<&str>;

    /// Returns reasoning text, if this is a reasoning item.
    fn as_reasoning(&self) -> Option<&str>;

    /// Returns refusal text, if this is a refusal item.
    fn as_refusal(&self) -> Option<&str>;

    /// Returns tool-call details, if this is a tool-call item.
    fn as_tool_call(&self) -> Option<ToolCallItemView<'_>>;

    /// Returns tool-result details, if this is a tool-result item.
    fn as_tool_result(&self) -> Option<ToolResultItemView<'_>>;
}

/// Read-only view of a committed turn in the session transcript.
///
/// Object-safe: items are accessed by index so no generic iterators are needed.
pub trait TurnView: fmt::Debug + Send + Sync {
    /// Role or category of this turn.
    fn role(&self) -> TurnRole;

    /// Number of items in this turn.
    fn item_count(&self) -> usize;

    /// Returns a reference to the item at `index`, or `None` if out of bounds.
    fn item_at(&self, index: usize) -> Option<&dyn ItemView>;

    /// Returns this turn as `Any` for adapter-specific downcasting.
    fn as_any(&self) -> &dyn Any;

    /// Returns `true` if this turn is ephemeral (included in the next model request
    /// but not persisted to the session transcript). Defaults to `false`.
    fn ephemeral(&self) -> bool {
        false
    }
}

impl dyn TurnView {
    pub fn downcast_ref<T: TurnView + 'static>(&self) -> Option<&T> {
        self.as_any().downcast_ref::<T>()
    }
}

// ── blanket helpers ───────────────────────────────────────────────────────────

/// Iterator over items in a `TurnView`.
pub struct TurnItemIter<'a> {
    turn: &'a dyn TurnView,
    index: usize,
}

impl<'a> TurnItemIter<'a> {
    pub fn new(turn: &'a dyn TurnView) -> Self {
        Self { turn, index: 0 }
    }
}

impl<'a> Iterator for TurnItemIter<'a> {
    type Item = &'a dyn ItemView;

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.turn.item_at(self.index)?;
        self.index += 1;
        Some(item)
    }
}

// ── core-provided assistant-turn view ─────────────────────────────────────────

/// A committed assistant turn backed by the core `AssistantTurnItem` list.
///
/// Adapters that produce `AssistantTurn`-backed results can use this type
/// directly.  Provider-specific adapters may define their own concrete type
/// instead.
#[derive(Debug)]
pub struct AssistantTurnView {
    items: Vec<CoreAssistantItemView>,
}

impl AssistantTurnView {
    /// Construct a view from a slice of `AssistantTurnItem` values.
    pub fn from_items(items: &[AssistantTurnItem]) -> Self {
        Self {
            items: items.iter().map(CoreAssistantItemView::from_item).collect(),
        }
    }
}

impl TurnView for AssistantTurnView {
    fn role(&self) -> TurnRole {
        TurnRole::Assistant
    }

    fn item_count(&self) -> usize {
        self.items.len()
    }

    fn item_at(&self, index: usize) -> Option<&dyn ItemView> {
        self.items.get(index).map(|v| v as &dyn ItemView)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[derive(Debug)]
enum CoreAssistantItemKind {
    Text(String),
    Reasoning(String),
    Refusal(String),
    ToolCall {
        id: ToolCallId,
        name: ToolName,
        arguments: RawJson,
    },
}

#[derive(Debug)]
struct CoreAssistantItemView {
    kind: CoreAssistantItemKind,
}

impl CoreAssistantItemView {
    fn from_item(item: &AssistantTurnItem) -> Self {
        let kind = match item {
            AssistantTurnItem::Text(t) => CoreAssistantItemKind::Text(t.clone()),
            AssistantTurnItem::Reasoning(t) => CoreAssistantItemKind::Reasoning(t.clone()),
            AssistantTurnItem::Refusal(t) => CoreAssistantItemKind::Refusal(t.clone()),
            AssistantTurnItem::ToolCall {
                id,
                name,
                arguments,
            } => CoreAssistantItemKind::ToolCall {
                id: id.clone(),
                name: name.clone(),
                arguments: arguments.clone(),
            },
        };
        Self { kind }
    }
}

impl ItemView for CoreAssistantItemView {
    fn as_text(&self) -> Option<&str> {
        match &self.kind {
            CoreAssistantItemKind::Text(t) => Some(t),
            _ => None,
        }
    }

    fn as_reasoning(&self) -> Option<&str> {
        match &self.kind {
            CoreAssistantItemKind::Reasoning(t) => Some(t),
            _ => None,
        }
    }

    fn as_refusal(&self) -> Option<&str> {
        match &self.kind {
            CoreAssistantItemKind::Refusal(t) => Some(t),
            _ => None,
        }
    }

    fn as_tool_call(&self) -> Option<ToolCallItemView<'_>> {
        match &self.kind {
            CoreAssistantItemKind::ToolCall {
                id,
                name,
                arguments,
            } => Some(ToolCallItemView {
                id,
                name,
                arguments,
            }),
            _ => None,
        }
    }

    fn as_tool_result(&self) -> Option<ToolResultItemView<'_>> {
        None // assistant items don't carry tool results
    }
}

// ── Arc-erased committed turn ─────────────────────────────────────────────────

/// A committed turn stored in the session, erased behind `Arc<dyn TurnView>`.
///
/// Using `Arc` allows `Session` (which is `Clone`) to share committed turns
/// cheaply across branch sessions without copying.
pub type CommittedTurn = Arc<dyn TurnView + Send + Sync>;

// ── ephemeral turn wrapper ────────────────────────────────────────────────────

/// A wrapper that marks an inner [`CommittedTurn`] as ephemeral.
///
/// Ephemeral turns are included in the model input for a single request but are
/// not persisted back to the session after the turn completes.
///
/// Create one via [`Session::push_ephemeral_turn`], which wraps automatically.
#[derive(Debug)]
pub struct EphemeralTurnView {
    inner: CommittedTurn,
}

impl EphemeralTurnView {
    pub fn new(inner: CommittedTurn) -> Self {
        Self { inner }
    }
}

impl TurnView for EphemeralTurnView {
    fn role(&self) -> TurnRole {
        self.inner.role()
    }

    fn item_count(&self) -> usize {
        self.inner.item_count()
    }

    fn item_at(&self, index: usize) -> Option<&dyn ItemView> {
        self.inner.item_at(index)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn ephemeral(&self) -> bool {
        true
    }
}
