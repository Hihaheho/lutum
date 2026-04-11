use super::*;
use quote::quote;
use syn::{Attribute, ItemFn, ItemTrait, TraitItem, TraitItemFn};

pub fn expand_hooks(item_trait: ItemTrait) -> proc_macro2::TokenStream {
    if item_trait.unsafety.is_some() || item_trait.auto_token.is_some() {
        return syn::Error::new_spanned(item_trait, "#[hooks] supports only plain traits")
            .to_compile_error();
    }
    if !item_trait.generics.params.is_empty() || item_trait.generics.where_clause.is_some() {
        return syn::Error::new_spanned(
            &item_trait.generics,
            "#[hooks] does not support generics or where clauses",
        )
        .to_compile_error();
    }
    if !item_trait.supertraits.is_empty() {
        return syn::Error::new_spanned(
            &item_trait.supertraits,
            "#[hooks] does not support supertraits",
        )
        .to_compile_error();
    }

    let struct_cfg_attrs = conditional_attrs(&item_trait.attrs);
    let struct_attrs = item_trait.attrs.clone();
    let ident = item_trait.ident.clone();
    let vis = item_trait.vis.clone();

    let mut items: Vec<proc_macro2::TokenStream> = Vec::new();
    let mut fields: Vec<proc_macro2::TokenStream> = Vec::new();
    let mut field_inits: Vec<proc_macro2::TokenStream> = Vec::new();
    let mut register_methods: Vec<proc_macro2::TokenStream> = Vec::new();
    let mut dispatch_methods: Vec<proc_macro2::TokenStream> = Vec::new();
    let mut default_methods: Vec<proc_macro2::TokenStream> = Vec::new();

    for item in item_trait.items {
        let method = match item {
            TraitItem::Fn(method) => method,
            other => {
                return syn::Error::new_spanned(
                    other,
                    "#[hooks] supports only async methods annotated with #[hook(...)]",
                )
                .to_compile_error();
            }
        };

        let kind = match extract_hook_kind(&method) {
            Ok(kind) => kind,
            Err(err) => return err.to_compile_error(),
        };
        let item_fn = trait_method_to_item_fn(&method);
        let slot = match expand_slot(item_fn, kind, &vis) {
            Ok(slot) => slot,
            Err(err) => return err.to_compile_error(),
        };

        items.push(slot.items);
        fields.push(slot.field);
        field_inits.push(slot.field_init);
        register_methods.push(slot.register_methods);
        dispatch_methods.push(slot.dispatch_method);
        default_methods.push(slot.default_method);
    }

    quote! {
        #(#items)*

        #(#struct_attrs)*
        #[allow(dead_code)]
        #[derive(::std::clone::Clone)]
        #vis struct #ident {
            #(#fields)*
        }

        #(#struct_cfg_attrs)*
        impl ::std::default::Default for #ident {
            fn default() -> Self {
                Self {
                    #(#field_inits)*
                }
            }
        }

        #(#struct_cfg_attrs)*
        #[allow(dead_code)]
        impl #ident {
            pub fn new() -> Self {
                Self::default()
            }

            #(#register_methods)*
            #(#dispatch_methods)*
            #(#default_methods)*
        }
    }
}

fn extract_hook_kind(method: &TraitItemFn) -> syn::Result<HookKind> {
    let mut hook_attr = None;
    for attr in &method.attrs {
        if is_hook_attr(attr) {
            if hook_attr.is_some() {
                return Err(syn::Error::new_spanned(
                    attr,
                    "#[hooks] methods must have exactly one #[hook(...)] attribute",
                ));
            }
            hook_attr = Some(attr);
        }
    }

    let Some(hook_attr) = hook_attr else {
        return Err(syn::Error::new_spanned(
            method,
            "#[hooks] methods must have exactly one #[hook(...)] attribute",
        ));
    };
    if method.default.is_none() {
        return Err(syn::Error::new_spanned(
            method,
            "#[hooks] methods must provide a default body; bodyless required hooks are not supported yet",
        ));
    }

    let attrs: crate::HookDefAttrs = hook_attr.parse_args()?;
    crate::build_hook_kind(attrs, "hook")
}

fn trait_method_to_item_fn(method: &TraitItemFn) -> ItemFn {
    let attrs = method
        .attrs
        .iter()
        .filter(|attr| !is_hook_attr(attr))
        .cloned()
        .collect::<Vec<_>>();

    ItemFn {
        attrs,
        vis: syn::Visibility::Inherited,
        sig: method.sig.clone(),
        block: Box::new(
            method
                .default
                .clone()
                .expect("default body presence checked earlier"),
        ),
    }
}

fn is_hook_attr(attr: &Attribute) -> bool {
    attr.path()
        .segments
        .last()
        .is_some_and(|segment| segment.ident == "hook")
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
