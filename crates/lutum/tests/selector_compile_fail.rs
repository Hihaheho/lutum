#[test]
fn mixed_toolset_selectors_do_not_type_check() {
    let cases = trybuild::TestCases::new();
    cases.compile_fail("tests/ui/mixed_selectors.rs");
}
