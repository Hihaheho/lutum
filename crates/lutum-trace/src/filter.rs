pub const LUTUM_CAPTURE_FIELD: &str = "lutum.capture";

pub(crate) fn lutum_target(target: &str) -> bool {
    target == "lutum" || target.starts_with("lutum::")
}

pub(crate) struct CaptureInterestVisitor {
    pub lutum_capture: bool,
}

impl tracing::field::Visit for CaptureInterestVisitor {
    fn record_bool(&mut self, field: &tracing::field::Field, value: bool) {
        if field.name() == LUTUM_CAPTURE_FIELD {
            self.lutum_capture |= value;
        }
    }

    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        if field.name() == LUTUM_CAPTURE_FIELD && value == "true" {
            self.lutum_capture = true;
        }
    }

    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        let rendered = format!("{value:?}");
        if field.name() == LUTUM_CAPTURE_FIELD && (rendered == "true" || rendered == "\"true\"") {
            self.lutum_capture = true;
        }
    }

    fn record_i64(&mut self, _: &tracing::field::Field, _: i64) {}

    fn record_u64(&mut self, _: &tracing::field::Field, _: u64) {}

    fn record_i128(&mut self, _: &tracing::field::Field, _: i128) {}

    fn record_u128(&mut self, _: &tracing::field::Field, _: u128) {}

    fn record_f64(&mut self, _: &tracing::field::Field, _: f64) {}

    fn record_error(&mut self, _: &tracing::field::Field, _: &(dyn std::error::Error + 'static)) {}
}

pub(crate) fn should_mark_span(attrs: &tracing::span::Attributes<'_>) -> bool {
    if lutum_target(attrs.metadata().target()) {
        return true;
    }

    let mut visitor = CaptureInterestVisitor {
        lutum_capture: false,
    };
    attrs.record(&mut visitor);
    visitor.lutum_capture
}

pub(crate) fn event_interesting(event: &tracing::Event<'_>) -> bool {
    if lutum_target(event.metadata().target()) {
        return true;
    }

    let mut visitor = CaptureInterestVisitor {
        lutum_capture: false,
    };
    event.record(&mut visitor);
    visitor.lutum_capture
}
