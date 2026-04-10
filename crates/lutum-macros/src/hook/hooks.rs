use heck::ToSnakeCase;
use quote::{format_ident, quote};
use syn::{Attribute, Fields, ItemStruct, Path, PathArguments, Type};

struct HookField {
    ident: syn::Ident,
    slot_path: Path,
}

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

    let mut hook_fields = Vec::new();
    for field in fields.named {
        let Some(field_ident) = field.ident else {
            return syn::Error::new_spanned(field, "#[hooks] fields must be named")
                .to_compile_error();
        };
        let Type::Path(type_path) = field.ty else {
            return syn::Error::new_spanned(
                field_ident,
                "#[hooks] fields must use a hook slot type",
            )
            .to_compile_error();
        };
        if type_path.qself.is_some() {
            return syn::Error::new_spanned(
                type_path,
                "#[hooks] fields must use a hook slot type path",
            )
            .to_compile_error();
        }
        if type_path
            .path
            .segments
            .iter()
            .take(type_path.path.segments.len().saturating_sub(1))
            .any(|segment| !matches!(segment.arguments, PathArguments::None))
        {
            return syn::Error::new_spanned(
                type_path,
                "#[hooks] only the last slot path segment may use generic arguments",
            )
            .to_compile_error();
        }

        hook_fields.push(HookField {
            ident: field_ident,
            slot_path: type_path.path,
        });
    }

    let helper_macro_ident =
        format_ident!("__lutum_define_{}_hooks", ident.to_string().to_snake_case());

    let slot_arms = hook_fields
        .iter()
        .enumerate()
        .map(|(index, hook_field)| {
            let field_ident = &hook_field.ident;
            let slot_path = &hook_field.slot_path;
            let hooks_macro_path = hooks_macro_path_for_slot(slot_path);
            let alias_ident = format_ident!(
                "__lutum_hook_head_{}_{}",
                ident.to_string().to_snake_case(),
                index
            );
            let use_imports = slot_use_imports(slot_path);

            let with_fn_ident = format_ident!("with_{}", field_ident);
            let register_fn_ident = format_ident!("register_{}", field_ident);
            let dispatch_fn_ident = field_ident.clone();
            let default_method_ident = format_ident!("__lutum_hook_default_{}", field_ident);
            let chain_field_ident = format_ident!("{}_chain", field_ident);
            let with_chain_fn_ident = format_ident!("with_{}_chain", field_ident);
            let set_chain_fn_ident = format_ident!("set_{}_chain", field_ident);
            let aggregate_field_ident = format_ident!("{}_aggregate", field_ident);
            let with_aggregate_fn_ident = format_ident!("with_{}_aggregate", field_ident);
            let set_aggregate_fn_ident = format_ident!("set_{}_aggregate", field_ident);
            let finalize_field_ident = format_ident!("{}_finalize", field_ident);
            let with_finalize_fn_ident = format_ident!("with_{}_finalize", field_ident);
            let set_finalize_fn_ident = format_ident!("set_{}_finalize", field_ident);

            quote! {
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
                        [#field_ident]
                        [#with_fn_ident]
                        [#register_fn_ident]
                        [#dispatch_fn_ident]
                        [#default_method_ident]
                        [#chain_field_ident]
                        [#with_chain_fn_ident]
                        [#set_chain_fn_ident]
                        [#aggregate_field_ident]
                        [#with_aggregate_fn_ident]
                        [#set_aggregate_fn_ident]
                        [#finalize_field_ident]
                        [#with_finalize_fn_ident]
                        [#set_finalize_fn_ident]
                        [$($rest)*]
                    );
                };
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
                        [#field_ident]
                        [#with_fn_ident]
                        [#register_fn_ident]
                        [#dispatch_fn_ident]
                        [#default_method_ident]
                        [#chain_field_ident]
                        [#with_chain_fn_ident]
                        [#set_chain_fn_ident]
                        [#aggregate_field_ident]
                        [#with_aggregate_fn_ident]
                        [#set_aggregate_fn_ident]
                        [#finalize_field_ident]
                        [#with_finalize_fn_ident]
                        [#set_finalize_fn_ident]
                        []
                    );
                };
            }
        })
        .collect::<Vec<_>>();

    let slot_paths = hook_fields
        .iter()
        .map(|field| &field.slot_path)
        .collect::<Vec<_>>();

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

fn hooks_macro_path_for_slot(slot_path: &Path) -> Path {
    if is_lutum_root_tool_hook(slot_path) {
        return syn::parse_quote!(::lutum::hooks::__lutum_hooks_ToolHook);
    }

    let mut p = strip_last_segment_arguments(slot_path);
    let last = p
        .segments
        .last_mut()
        .expect("slot path must have at least one segment");
    last.ident = format_ident!("__lutum_hooks_{}", last.ident);
    p
}

fn slot_use_imports(slot_path: &Path) -> proc_macro2::TokenStream {
    if is_lutum_root_tool_hook(slot_path) {
        return quote! {
            #[allow(unused_imports)]
            use ::lutum::hooks::__LutumDynToolHook;
            #[allow(unused_imports)]
            use ::lutum::hooks::__lutum_hook_default_impl_tool_hook;
            #[allow(unused_imports)]
            use ::lutum::ToolHook;
        };
    }

    let bare_slot_path = strip_last_segment_arguments(slot_path);
    if bare_slot_path.segments.len() <= 1 {
        return quote! {};
    }

    let last_ident = &bare_slot_path.segments.last().unwrap().ident;
    let snake = last_ident.to_string().to_snake_case();

    let dyn_path = replace_last_segment(&bare_slot_path, format_ident!("__LutumDyn{}", last_ident));
    let default_impl_path = replace_last_segment(
        &bare_slot_path,
        format_ident!("__lutum_hook_default_impl_{}", snake),
    );

    quote! {
        #[allow(unused_imports)]
        use #dyn_path;
        #[allow(unused_imports)]
        use #bare_slot_path;
        #[allow(unused_imports)]
        use #default_impl_path;
    }
}

fn strip_last_segment_arguments(path: &Path) -> Path {
    let mut p = path.clone();
    if let Some(last) = p.segments.last_mut() {
        last.arguments = PathArguments::None;
    }
    p
}

fn replace_last_segment(path: &Path, new_ident: proc_macro2::Ident) -> Path {
    let mut p = path.clone();
    p.segments
        .last_mut()
        .expect("path must have at least one segment")
        .ident = new_ident;
    p
}

fn is_lutum_root_tool_hook(slot_path: &Path) -> bool {
    let mut segments = slot_path.segments.iter();
    let Some(first) = segments.next() else {
        return false;
    };
    let Some(second) = segments.next() else {
        return false;
    };
    segments.next().is_none() && first.ident == "lutum" && second.ident == "ToolHook"
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
