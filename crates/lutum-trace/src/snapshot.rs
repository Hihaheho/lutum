use std::collections::HashMap;

use crate::store::CaptureRecord;

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

    pub fn has_event_message(&self, msg: &str) -> bool {
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

fn collect_nodes<'a>(node: &'a SpanNode, name: &str, matches: &mut Vec<&'a SpanNode>) {
    if node.name == name {
        matches.push(node);
    }

    for child in &node.children {
        collect_nodes(child, name, matches);
    }
}

struct SpanData {
    name: &'static str,
    target: &'static str,
    level: String,
    fields: Vec<(String, FieldValue)>,
    events: Vec<EventRecord>,
    children: Vec<usize>,
}

impl SpanData {
    fn upsert_field(&mut self, name: String, value: FieldValue) {
        if let Some((_, existing)) = self.fields.iter_mut().find(|(key, _)| key == &name) {
            *existing = value;
            return;
        }

        self.fields.push((name, value));
    }
}

pub(crate) fn build_snapshot(records: &[CaptureRecord]) -> TraceSnapshot {
    let mut span_data: Vec<SpanData> = Vec::new();
    let mut open_spans: HashMap<u64, usize> = HashMap::new();
    let mut roots: Vec<usize> = Vec::new();
    let mut root_events: Vec<EventRecord> = Vec::new();

    for record in records {
        match record {
            CaptureRecord::SpanOpened {
                id,
                parent_id,
                name,
                target,
                level,
                fields,
            } => {
                let key = span_data.len();
                span_data.push(SpanData {
                    name,
                    target,
                    level: (*level).to_string(),
                    fields: fields.clone(),
                    events: Vec::new(),
                    children: Vec::new(),
                });

                if let Some(parent_key) =
                    parent_id.and_then(|parent| open_spans.get(&parent).copied())
                {
                    span_data[parent_key].children.push(key);
                } else {
                    roots.push(key);
                }

                open_spans.insert(*id, key);
            }
            CaptureRecord::SpanRecorded { id, fields } => {
                let Some(&key) = open_spans.get(id) else {
                    continue;
                };

                let span = &mut span_data[key];
                for (name, value) in fields {
                    span.upsert_field(name.clone(), value.clone());
                }
            }
            CaptureRecord::Event {
                parent_span_id,
                record,
            } => {
                if let Some(key) =
                    parent_span_id.and_then(|parent| open_spans.get(&parent).copied())
                {
                    span_data[key].events.push(record.clone());
                } else {
                    root_events.push(record.clone());
                }
            }
            CaptureRecord::SpanClosed { id } => {
                open_spans.remove(id);
            }
        }
    }

    TraceSnapshot {
        roots: roots
            .into_iter()
            .map(|key| build_node(&span_data, key))
            .collect(),
        root_events,
    }
}

fn build_node(span_data: &[SpanData], key: usize) -> SpanNode {
    let data = &span_data[key];

    SpanNode {
        name: data.name.to_string(),
        target: data.target.to_string(),
        level: data.level.clone(),
        fields: data.fields.clone(),
        events: data.events.clone(),
        children: data
            .children
            .iter()
            .map(|&child| build_node(span_data, child))
            .collect(),
    }
}
