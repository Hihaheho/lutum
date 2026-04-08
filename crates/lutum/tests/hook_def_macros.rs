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
fn hook_def_chain_compiles() {
    let cases = trybuild::TestCases::new();
    cases.pass("tests/ui/hook_def_chain.rs");
}

#[test]
fn hook_impl_path_chain_compiles() {
    let cases = trybuild::TestCases::new();
    cases.pass("tests/ui/hook_impl_path_chain.rs");
}

#[test]
fn hook_def_singleton_rejects_last_argument() {
    let cases = trybuild::TestCases::new();
    cases.compile_fail("tests/ui/hook_def_singleton_last.rs");
}

#[test]
fn hook_def_singleton_rejects_chain_argument() {
    let cases = trybuild::TestCases::new();
    cases.compile_fail("tests/ui/hook_def_singleton_chain.rs");
}

#[test]
fn hook_impl_singleton_rejects_last_argument() {
    let cases = trybuild::TestCases::new();
    cases.compile_fail("tests/ui/hook_impl_singleton_last.rs");
}

#[test]
fn hooks_path_slot_compiles() {
    let cases = trybuild::TestCases::new();
    cases.pass("tests/ui/hooks_path_slot.rs");
}

#[test]
fn hook_cross_module_path_compiles() {
    let cases = trybuild::TestCases::new();
    cases.pass("tests/ui/hook_cross_module_paths.rs");
}
