use crate::store::{InnerStore, SpanKey};

#[derive(Debug, Clone, PartialEq)]
pub enum FieldValue {
    Bool(bool),
    I64(i64),
    U64(u64),
    I128(i128),
    U128(u128),
    F64(f64),
    Str(String),
}

#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub struct TraceSpanId(pub u64);

#[derive(Debug, Clone, PartialEq)]
pub enum TraceEvent {
    SpanOpened {
        span_id: TraceSpanId,
        parent_span_id: Option<TraceSpanId>,
        name: String,
        target: String,
        level: String,
        fields: Vec<(String, FieldValue)>,
    },
    SpanRecorded {
        span_id: TraceSpanId,
        fields: Vec<(String, FieldValue)>,
    },
    Event {
        parent_span_id: Option<TraceSpanId>,
        record: EventRecord,
    },
    SpanClosed {
        span_id: TraceSpanId,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct TraceSnapshot {
    pub roots: Vec<SpanNode>,
    pub root_events: Vec<EventRecord>,
}

impl TraceSnapshot {
    pub fn span(&self, name: &str) -> Option<&SpanNode> {
        self.roots.iter().find_map(|root| {
            if root.name == name {
                Some(root)
            } else {
                root.find(name)
            }
        })
    }

    pub fn find_all(&self, name: &str) -> Vec<&SpanNode> {
        let mut matches = Vec::new();
        for root in &self.roots {
            collect_nodes(root, name, &mut matches);
        }
        matches
    }

    pub fn events(&self) -> &[EventRecord] {
        &self.root_events
    }

    pub fn has_event_message(&self, msg: &str) -> bool {
        self.root_events
            .iter()
            .any(|event| event.message.as_deref() == Some(msg))
            || self.roots.iter().any(|root| root.has_event_message(msg))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpanNode {
    pub name: String,
    pub target: String,
    pub level: String,
    pub fields: Vec<(String, FieldValue)>,
    pub events: Vec<EventRecord>,
    pub children: Vec<SpanNode>,
}

impl SpanNode {
    pub fn child(&self, name: &str) -> Option<&SpanNode> {
        self.children.iter().find(|child| child.name == name)
    }

    pub fn find(&self, name: &str) -> Option<&SpanNode> {
        self.children.iter().find_map(|child| {
            if child.name == name {
                Some(child)
            } else {
                child.find(name)
            }
        })
    }

    pub fn field(&self, key: &str) -> Option<&FieldValue> {
        self.fields
            .iter()
            .find_map(|(name, value)| (name == key).then_some(value))
    }

    pub fn event(&self, msg: &str) -> Option<&EventRecord> {
        self.events
            .iter()
            .find(|event| event.message.as_deref() == Some(msg))
    }

    pub fn events(&self) -> &[EventRecord] {
        &self.events
    }

    pub fn children(&self) -> &[SpanNode] {
        &self.children
    }

    fn has_event_message(&self, msg: &str) -> bool {
        self.events
            .iter()
            .any(|event| event.message.as_deref() == Some(msg))
            || self
                .children
                .iter()
                .any(|child| child.has_event_message(msg))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EventRecord {
    pub target: String,
    pub level: String,
    pub message: Option<String>,
    pub fields: Vec<(String, FieldValue)>,
}

impl EventRecord {
    pub fn message(&self) -> Option<&str> {
        self.message.as_deref()
    }

    pub fn field(&self, key: &str) -> Option<&FieldValue> {
        self.fields
            .iter()
            .find_map(|(name, value)| (name == key).then_some(value))
    }
}

pub(crate) fn build_snapshot(store: &InnerStore) -> TraceSnapshot {
    TraceSnapshot {
        roots: store
            .roots
            .iter()
            .filter_map(|&key| build_node(store, key))
            .collect(),
        root_events: store.root_events.clone(),
    }
}

fn collect_nodes<'a>(node: &'a SpanNode, name: &str, matches: &mut Vec<&'a SpanNode>) {
    if node.name == name {
        matches.push(node);
    }

    for child in &node.children {
        collect_nodes(child, name, matches);
    }
}

fn build_node(store: &InnerStore, key: SpanKey) -> Option<SpanNode> {
    let data = store.spans.get(&key)?;

    Some(SpanNode {
        name: data.name.to_string(),
        target: data.target.to_string(),
        level: data.level.clone(),
        fields: data.fields.clone(),
        events: data.events.clone(),
        children: data
            .children
            .iter()
            .filter_map(|&child| build_node(store, child))
            .collect(),
    })
}
