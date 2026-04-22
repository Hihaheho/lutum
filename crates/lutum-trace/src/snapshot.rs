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

    pub fn span_exists(&self, name: &str) -> bool {
        self.span(name).is_some()
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

    /// All events across the entire trace in DFS order: root-level events first,
    /// then each root span's events (including descendants) in DFS order.
    pub fn all_events(&self) -> impl Iterator<Item = &EventRecord> + '_ {
        let mut all: Vec<&EventRecord> = self.root_events.iter().collect();
        for root in &self.roots {
            collect_all_events(root, &mut all);
        }
        all.into_iter()
    }

    pub fn events_matching(
        &self,
        pred: impl Fn(&EventRecord) -> bool,
    ) -> Vec<&EventRecord> {
        self.all_events().filter(|e| pred(e)).collect()
    }

    /// Returns `true` if the first DFS occurrence of each name in `names`
    /// appears at a strictly greater DFS position than the previous name.
    /// Returns `false` if any name is absent or out of order.
    pub fn spans_ordered(&self, names: &[&str]) -> bool {
        if names.is_empty() {
            return true;
        }
        let mut positions: Vec<Option<usize>> = vec![None; names.len()];
        let mut counter = 0usize;
        for root in &self.roots {
            collect_dfs_positions(root, names, &mut positions, &mut counter);
        }
        let mut last = 0usize;
        let mut first = true;
        for pos in &positions {
            match pos {
                None => return false,
                Some(p) => {
                    if !first && *p <= last {
                        return false;
                    }
                    last = *p;
                    first = false;
                }
            }
        }
        true
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

    /// All descendants (and self) with the given name, in DFS order.
    pub fn find_all(&self, name: &str) -> Vec<&SpanNode> {
        let mut matches = Vec::new();
        collect_nodes(self, name, &mut matches);
        matches
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

    /// All events in this span and its descendants in DFS order.
    pub fn all_events(&self) -> impl Iterator<Item = &EventRecord> + '_ {
        let mut all = Vec::new();
        collect_all_events(self, &mut all);
        all.into_iter()
    }

    pub fn events_matching(
        &self,
        pred: impl Fn(&EventRecord) -> bool,
    ) -> Vec<&EventRecord> {
        self.all_events().filter(|e| pred(e)).collect()
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

fn collect_all_events<'a>(node: &'a SpanNode, out: &mut Vec<&'a EventRecord>) {
    out.extend(node.events.iter());
    for child in &node.children {
        collect_all_events(child, out);
    }
}

fn collect_dfs_positions(
    node: &SpanNode,
    names: &[&str],
    positions: &mut Vec<Option<usize>>,
    counter: &mut usize,
) {
    let pos = *counter;
    *counter += 1;
    for (i, &name) in names.iter().enumerate() {
        if positions[i].is_none() && node.name == name {
            positions[i] = Some(pos);
        }
    }
    for child in &node.children {
        collect_dfs_positions(child, names, positions, counter);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_snapshot() -> TraceSnapshot {
        // Tree:
        //   span_a
        //     event("in_a")
        //     span_b
        //       event("in_b")
        //       span_c
        //         event("in_c")
        //   span_d
        //     event("in_d")
        // root_events: event("root")
        TraceSnapshot {
            root_events: vec![EventRecord {
                target: "t".into(),
                level: "INFO".into(),
                message: Some("root".into()),
                fields: vec![],
            }],
            roots: vec![
                SpanNode {
                    name: "span_a".into(),
                    target: "t".into(),
                    level: "INFO".into(),
                    fields: vec![],
                    events: vec![EventRecord {
                        target: "t".into(),
                        level: "INFO".into(),
                        message: Some("in_a".into()),
                        fields: vec![("key".into(), FieldValue::Str("val".into()))],
                    }],
                    children: vec![SpanNode {
                        name: "span_b".into(),
                        target: "t".into(),
                        level: "INFO".into(),
                        fields: vec![],
                        events: vec![EventRecord {
                            target: "t".into(),
                            level: "INFO".into(),
                            message: Some("in_b".into()),
                            fields: vec![],
                        }],
                        children: vec![SpanNode {
                            name: "span_c".into(),
                            target: "t".into(),
                            level: "INFO".into(),
                            fields: vec![],
                            events: vec![EventRecord {
                                target: "t".into(),
                                level: "INFO".into(),
                                message: Some("in_c".into()),
                                fields: vec![],
                            }],
                            children: vec![],
                        }],
                    }],
                },
                SpanNode {
                    name: "span_d".into(),
                    target: "t".into(),
                    level: "INFO".into(),
                    fields: vec![],
                    events: vec![EventRecord {
                        target: "t".into(),
                        level: "INFO".into(),
                        message: Some("in_d".into()),
                        fields: vec![],
                    }],
                    children: vec![],
                },
            ],
        }
    }

    #[test]
    fn span_exists_found_and_not_found() {
        let snap = make_snapshot();
        assert!(snap.span_exists("span_a"));
        assert!(snap.span_exists("span_b"));
        assert!(snap.span_exists("span_c"));
        assert!(snap.span_exists("span_d"));
        assert!(!snap.span_exists("span_z"));
    }

    #[test]
    fn all_events_order() {
        let snap = make_snapshot();
        let msgs: Vec<_> = snap
            .all_events()
            .filter_map(|e| e.message.as_deref())
            .collect();
        // root_events first, then DFS span events
        assert_eq!(msgs, ["root", "in_a", "in_b", "in_c", "in_d"]);
    }

    #[test]
    fn events_matching_by_field() {
        let snap = make_snapshot();
        let matched = snap.events_matching(|e| e.field("key").is_some());
        assert_eq!(matched.len(), 1);
        assert_eq!(matched[0].message.as_deref(), Some("in_a"));
    }

    #[test]
    fn spans_ordered_in_order() {
        let snap = make_snapshot();
        // DFS: span_a(0), span_b(1), span_c(2), span_d(3)
        assert!(snap.spans_ordered(&["span_a", "span_b", "span_c"]));
        assert!(snap.spans_ordered(&["span_a", "span_d"]));
        assert!(snap.spans_ordered(&["span_b", "span_d"]));
        assert!(snap.spans_ordered(&["span_c", "span_d"]));
    }

    #[test]
    fn spans_ordered_out_of_order() {
        let snap = make_snapshot();
        assert!(!snap.spans_ordered(&["span_d", "span_a"]));
        assert!(!snap.spans_ordered(&["span_d", "span_b"]));
        assert!(!snap.spans_ordered(&["span_c", "span_b"]));
    }

    #[test]
    fn spans_ordered_missing_span() {
        let snap = make_snapshot();
        assert!(!snap.spans_ordered(&["span_a", "span_z"]));
    }

    #[test]
    fn spans_ordered_empty() {
        let snap = make_snapshot();
        assert!(snap.spans_ordered(&[]));
    }

    #[test]
    fn span_node_find_all() {
        let snap = make_snapshot();
        let a = snap.span("span_a").unwrap();
        // find_all on span_a returns span_a itself (name match) + span_b + span_c
        let found = a.find_all("span_b");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].name, "span_b");
    }

    #[test]
    fn span_node_all_events_order() {
        let snap = make_snapshot();
        let a = snap.span("span_a").unwrap();
        let msgs: Vec<_> = a
            .all_events()
            .filter_map(|e| e.message.as_deref())
            .collect();
        assert_eq!(msgs, ["in_a", "in_b", "in_c"]);
    }

    #[test]
    fn span_node_events_matching() {
        let snap = make_snapshot();
        let a = snap.span("span_a").unwrap();
        let matched = a.events_matching(|e| e.message.as_deref() == Some("in_b"));
        assert_eq!(matched.len(), 1);
    }
}
