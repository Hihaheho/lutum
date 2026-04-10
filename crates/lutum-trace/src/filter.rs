/// Field name users set to opt spans into capture when `target` is not under `lutum`.
pub const LUTUM_CAPTURE_FIELD: &str = "lutum.capture";

pub(crate) fn lutum_target(target: &str) -> bool {
    target == "lutum" || target.starts_with("lutum::")
}

#[derive(Default)]
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
        let s = format!("{value:?}");
        if field.name() == LUTUM_CAPTURE_FIELD && (s == "true" || s == "\"true\"") {
            self.lutum_capture = true;
        }
    }

    fn record_i64(&mut self, _field: &tracing::field::Field, _value: i64) {}

    fn record_u64(&mut self, _field: &tracing::field::Field, _value: u64) {}

    fn record_i128(&mut self, _field: &tracing::field::Field, _value: i128) {}

    fn record_u128(&mut self, _field: &tracing::field::Field, _value: u128) {}

    fn record_f64(&mut self, _field: &tracing::field::Field, _value: f64) {}

    fn record_error(
        &mut self,
        _field: &tracing::field::Field,
        _value: &(dyn std::error::Error + 'static),
    ) {
    }
}
