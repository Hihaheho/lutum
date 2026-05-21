#[test]
fn dynamic_toolset_compile_failures() {
    let cases = trybuild::TestCases::new();
    cases.compile_fail("tests/ui/dynamic_without_slot.rs");
    cases.compile_fail("tests/ui/dynamic_wrong_payload.rs");
    cases.compile_fail("tests/ui/dynamic_multiple_slots.rs");
}
