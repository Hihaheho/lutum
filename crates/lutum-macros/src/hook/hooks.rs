use heck::ToSnakeCase;
use quote::{format_ident, quote};
use syn::{Attribute, Fields, ItemStruct, PathArguments, Type};

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

    let mut slot_idents = Vec::new();
    for field in fields.named {
        let ty = field.ty;
        let Type::Path(type_path) = ty else {
            return syn::Error::new_spanned(ty, "#[hooks] fields must use a hook slot type")
                .to_compile_error();
        };
        if type_path.qself.is_some()
            || type_path.path.segments.len() != 1
            || type_path
                .path
                .segments
                .iter()
                .any(|segment| !matches!(segment.arguments, PathArguments::None))
        {
            return syn::Error::new_spanned(
                type_path,
                "#[hooks] fields must use a plain hook slot type identifier",
            )
            .to_compile_error();
        }
        slot_idents.push(
            type_path
                .path
                .get_ident()
                .expect("validated single-segment hook slot path")
                .clone(),
        );
    }
    let helper_macro_ident =
        format_ident!("__lutum_define_{}_hooks", ident.to_string().to_snake_case());

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
            (
                [$($fields:tt)*]
                [$($field_inits:tt)*]
                [$($register_methods:tt)*]
                [$($dispatch_methods:tt)*]
                [$($default_impls:tt)*]
                [$head:ident $(, $tail:ident)* $(,)?]
            ) => {
                $head!(
                    @accumulate
                    #helper_macro_ident
                    [$($fields)*]
                    [$($field_inits)*]
                    [$($register_methods)*]
                    [$($dispatch_methods)*]
                    [$($default_impls)*]
                    [$($tail),*]
                );
            };
        }

        #helper_macro_ident!(
            []
            []
            []
            []
            []
            [#(#slot_idents),*]
        );
    }
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
