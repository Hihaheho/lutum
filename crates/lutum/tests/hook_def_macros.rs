#[test]
fn hook_trait_basics_compile() {
    let cases = trybuild::TestCases::new();
    cases.pass("tests/ui/hook_def_optional_last.rs");
}

#[test]
fn hook_def_chain_compiles() {
    let cases = trybuild::TestCases::new();
    cases.pass("tests/ui/hook_def_chain.rs");
}

#[test]
fn hook_def_output_override_compiles() {
    let cases = trybuild::TestCases::new();
    cases.pass("tests/ui/hook_def_output.rs");
}

#[test]
fn hook_impl_path_chain_compiles() {
    let cases = trybuild::TestCases::new();
    cases.pass("tests/ui/hook_impl_path_chain.rs");
}

#[test]
fn hook_slot_definition_rejects_last_argument() {
    let cases = trybuild::TestCases::new();
    cases.compile_fail("tests/ui/hook_def_singleton_last.rs");
}

#[test]
fn hook_singleton_rejects_chain_argument() {
    let cases = trybuild::TestCases::new();
    cases.compile_fail("tests/ui/hook_def_singleton_chain.rs");
}

#[test]
fn hook_def_singleton_rejects_output_argument() {
    let cases = trybuild::TestCases::new();
    cases.compile_fail("tests/ui/hook_def_singleton_output.rs");
}

#[test]
fn hook_def_output_without_companion_fails() {
    let cases = trybuild::TestCases::new();
    cases.compile_fail("tests/ui/hook_def_output_without_companion.rs");
}

#[test]
fn hook_def_rejects_aggregate_and_finalize_together() {
    let cases = trybuild::TestCases::new();
    cases.compile_fail("tests/ui/hook_def_aggregate_finalize.rs");
}

#[test]
fn hook_def_output_requires_into_companion_trait() {
    let cases = trybuild::TestCases::new();
    cases.compile_fail("tests/ui/hook_def_output_wrong_trait.rs");
}

#[test]
fn hook_impl_singleton_rejects_last_argument() {
    let cases = trybuild::TestCases::new();
    cases.compile_fail("tests/ui/hook_impl_singleton_last.rs");
}

#[test]
fn hooks_rejects_struct_input() {
    let cases = trybuild::TestCases::new();
    cases.compile_fail("tests/ui/hooks_path_slot.rs");
}

#[test]
fn hooks_reject_bodyless_methods() {
    let cases = trybuild::TestCases::new();
    cases.compile_fail("tests/ui/hook_def_mismatched_field_name.rs");
}

#[test]
fn hooks_reject_generic_user_slots() {
    let cases = trybuild::TestCases::new();
    cases.compile_fail("tests/ui/hook_def_generics.rs");
}

#[test]
fn hook_cross_module_path_compiles() {
    let cases = trybuild::TestCases::new();
    cases.pass("tests/ui/hook_cross_module_paths.rs");
}

#[test]
fn hook_cross_module_output_path_compiles() {
    let cases = trybuild::TestCases::new();
    cases.pass("tests/ui/hook_cross_module_output_paths.rs");
}

#[test]
fn tool_hook_toolset_compiles() {
    let cases = trybuild::TestCases::new();
    cases.pass("tests/ui/tool_hook_toolset.rs");
}
