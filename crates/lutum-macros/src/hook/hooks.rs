use heck::ToSnakeCase;
use quote::{format_ident, quote};
use syn::{Attribute, Fields, ItemStruct, Path, PathArguments, Type};

pub fn expand_hooks(item_struct: ItemStruct) -> proc_macro2::TokenStream {
    if !item_struct.generics.params.is_empty() || item_struct.generics.where_clause.is_some() {
        return syn::Error::new_spanned(
            &item_struct.generics,
            "#[hooks] does not support generics or where clauses",
        )
        .to_compile_error();
    }

    let struct_cfg_attrs = conditional_attrs(&item_struct.attrs);
    let struct_attrs = item_struct
        .attrs
        .iter()
        .filter(|attr| !attr.path().is_ident("derive"))
        .cloned()
        .collect::<Vec<_>>();
    let ident = item_struct.ident.clone();
    let vis = item_struct.vis.clone();
    let Fields::Named(fields) = item_struct.fields else {
        return syn::Error::new_spanned(
            item_struct,
            "#[hooks] supports only structs with named fields",
        )
        .to_compile_error();
    };

    let mut slot_paths: Vec<Path> = Vec::new();
    for field in fields.named {
        let ty = field.ty;
        let Type::Path(type_path) = ty else {
            return syn::Error::new_spanned(ty, "#[hooks] fields must use a hook slot type")
                .to_compile_error();
        };
        if type_path.qself.is_some()
            || type_path
                .path
                .segments
                .iter()
                .any(|segment| !matches!(segment.arguments, PathArguments::None))
        {
            return syn::Error::new_spanned(
                type_path,
                "#[hooks] fields must use a hook slot type path without generic arguments",
            )
            .to_compile_error();
        }
        slot_paths.push(type_path.path.clone());
    }
    let helper_macro_ident =
        format_ident!("__lutum_define_{}_hooks", ident.to_string().to_snake_case());
    let struct_snake = ident.to_string().to_snake_case();

    // For each slot, generate two specific arms in the helper macro:
    //   A) [<path> , rest...]  — this slot is followed by more slots
    //   B) [<path> (,)?]       — this slot is the last one
    //
    // We can't use a generic `$head:path` arm because:
    // - `$head:path!(...)` is invalid Rust syntax (path metavar can't be a macro name)
    // - `use $head as alias; alias!(...)` imports the trait alongside the macro;
    //   multiple `use` items with the same alias name in the same scope → E0252
    //
    // Why two arms instead of optional `$(, $rest:tt+)?`:
    //   After passing through @accumulate the remaining list arrives as raw :tt tokens.
    //   Those tokens can be matched against literal tokens in patterns, but only when
    //   NOT captured as :path/:ident metavariables first.  Two arms keep the pattern
    //   explicit and avoid metavar restriction issues.
    //
    // Alias names encode both struct and slot index to prevent E0252 when multiple
    // #[hooks] structs expand in the same module.
    let slot_arms: Vec<proc_macro2::TokenStream> = slot_paths
        .iter()
        .enumerate()
        .map(|(i, slot_path)| {
            let hooks_macro_path = hooks_macro_path_for_slot(slot_path);
            let alias_ident = format_ident!("__lutum_hook_head_{}_{}", struct_snake, i);
            let use_imports = slot_use_imports(slot_path);
            quote! {
                // Arm A: this slot is followed by more slots.
                (
                    [$($fields:tt)*]
                    [$($field_inits:tt)*]
                    [$($register_methods:tt)*]
                    [$($dispatch_methods:tt)*]
                    [$($default_impls:tt)*]
                    [#slot_path , $($rest:tt)*]
                ) => {
                    #use_imports
                    #[allow(unused_imports)]
                    use #hooks_macro_path as #alias_ident;
                    #alias_ident!(
                        @accumulate
                        #helper_macro_ident
                        [$($fields)*]
                        [$($field_inits)*]
                        [$($register_methods)*]
                        [$($dispatch_methods)*]
                        [$($default_impls)*]
                        [$($rest)*]
                    );
                };
                // Arm B: this slot is the last one (optional trailing comma).
                (
                    [$($fields:tt)*]
                    [$($field_inits:tt)*]
                    [$($register_methods:tt)*]
                    [$($dispatch_methods:tt)*]
                    [$($default_impls:tt)*]
                    [#slot_path $(,)?]
                ) => {
                    #use_imports
                    #[allow(unused_imports)]
                    use #hooks_macro_path as #alias_ident;
                    #alias_ident!(
                        @accumulate
                        #helper_macro_ident
                        [$($fields)*]
                        [$($field_inits)*]
                        [$($register_methods)*]
                        [$($dispatch_methods)*]
                        [$($default_impls)*]
                        []
                    );
                };
            }
        })
        .collect();

    quote! {
        #[allow(unused_macros)]
        macro_rules! #helper_macro_ident {
            (
                [$($fields:tt)*]
                [$($field_inits:tt)*]
                [$($register_methods:tt)*]
                [$($dispatch_methods:tt)*]
                [$($default_impls:tt)*]
                []
            ) => {
                #(#struct_attrs)*
                #[allow(dead_code)]
                #[derive(::std::clone::Clone)]
                #vis struct #ident {
                    $($fields)*
                }

                #(#struct_cfg_attrs)*
                impl ::std::default::Default for #ident {
                    fn default() -> Self {
                        Self {
                            $($field_inits)*
                        }
                    }
                }

                #(#struct_cfg_attrs)*
                #[allow(dead_code)]
                impl #ident {
                    pub fn new() -> Self {
                        Self::default()
                    }

                    $($register_methods)*
                    $($dispatch_methods)*
                    $($default_impls)*
                }
            };
            #(#slot_arms)*
        }

        #helper_macro_ident!(
            []
            []
            []
            []
            []
            [#(#slot_paths),*]
        );
    }
}

/// Returns the path to the exported accumulate alias for a slot.
/// `slots::SelectLabel` → `slots::__lutum_hooks_SelectLabel`
/// `SelectLabel`        → `__lutum_hooks_SelectLabel`
fn hooks_macro_path_for_slot(slot_path: &Path) -> Path {
    let mut p = slot_path.clone();
    let last = p
        .segments
        .last_mut()
        .expect("slot path must have at least one segment");
    last.ident = format_ident!("__lutum_hooks_{}", last.ident);
    p
}

/// For path-based slots (more than one segment), emits `use` imports for the
/// implementation-detail items the `@accumulate` arm references by unqualified name:
///   - `<module>::__LutumDyn<SlotIdent>`     (the dyn dispatch trait)
///   - `<module>::<SlotIdent>`               (the public hook trait)
///   - `<module>::__lutum_hook_default_impl_<snake_ident>` (the default-impl fn)
///
/// For single-segment (plain ident) slots everything is already in scope.
fn slot_use_imports(slot_path: &Path) -> proc_macro2::TokenStream {
    if slot_path.segments.len() <= 1 {
        return quote! {};
    }

    let last_ident = &slot_path.segments.last().unwrap().ident;
    let snake = last_ident.to_string().to_snake_case();

    // Build fully-qualified paths by replacing the last segment (avoids trailing `::` issues).
    let dyn_path = replace_last_segment(slot_path, format_ident!("__LutumDyn{}", last_ident));
    let default_impl_path = replace_last_segment(
        slot_path,
        format_ident!("__lutum_hook_default_impl_{}", snake),
    );

    quote! {
        #[allow(unused_imports)]
        use #dyn_path;
        #[allow(unused_imports)]
        use #slot_path;
        #[allow(unused_imports)]
        use #default_impl_path;
    }
}

/// Returns a copy of `path` with the last segment's ident replaced by `new_ident`.
fn replace_last_segment(path: &Path, new_ident: proc_macro2::Ident) -> Path {
    let mut p = path.clone();
    p.segments
        .last_mut()
        .expect("path must have at least one segment")
        .ident = new_ident;
    p
}

fn conditional_attrs(attrs: &[Attribute]) -> Vec<Attribute> {
    attrs
        .iter()
        .filter(|attr| {
            let path = attr.path();
            path.is_ident("cfg") || path.is_ident("cfg_attr")
        })
        .cloned()
        .collect()
}
