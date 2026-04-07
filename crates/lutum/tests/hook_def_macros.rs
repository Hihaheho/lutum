#[test]
fn hook_def_optional_last_compiles() {
    let cases = trybuild::TestCases::new();
    cases.pass("tests/ui/hook_def_optional_last.rs");
}

#[test]
fn hook_def_mismatched_field_name_compiles() {
    let cases = trybuild::TestCases::new();
    cases.pass("tests/ui/hook_def_mismatched_field_name.rs");
}

#[test]
fn hook_def_singleton_rejects_last_argument() {
    let cases = trybuild::TestCases::new();
    cases.compile_fail("tests/ui/hook_def_singleton_last.rs");
}
