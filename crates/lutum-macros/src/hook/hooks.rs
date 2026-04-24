use super::*;
use quote::{format_ident, quote};
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

    // Separate #[nested_hooks(...)] from the remaining attrs.
    let mut nested_hooks_entries: Vec<(syn::Ident, syn::Type)> = Vec::new();
    let mut remaining_attrs: Vec<Attribute> = Vec::new();
    for attr in &item_trait.attrs {
        if is_nested_hooks_attr(attr) {
            match attr.parse_args::<crate::NestedHooksAttr>() {
                Ok(parsed) => nested_hooks_entries.extend(parsed.entries),
                Err(e) => return e.to_compile_error(),
            }
        } else {
            remaining_attrs.push(attr.clone());
        }
    }

    let struct_cfg_attrs = conditional_attrs(&remaining_attrs);
    let struct_attrs = remaining_attrs;
    let ident = item_trait.ident.clone();
    let set_ident = format_ident!("{ident}Set");
    let register_trait_ident = format_ident!("__Lutum{set_ident}RegisterHooks");
    let vis = item_trait.vis.clone();

    let mut items: Vec<proc_macro2::TokenStream> = Vec::new();
    let mut trait_methods: Vec<proc_macro2::TokenStream> = Vec::new();
    let mut trait_ref_impl_methods: Vec<proc_macro2::TokenStream> = Vec::new();
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
        let slot = match expand_slot(item_fn, kind, &vis, &ident) {
            Ok(slot) => slot,
            Err(err) => return err.to_compile_error(),
        };

        items.push(slot.items);
        trait_methods.push(slot.trait_method);
        trait_ref_impl_methods.push(slot.trait_ref_impl_method);
        fields.push(slot.field);
        field_inits.push(slot.field_init);
        register_methods.push(slot.register_methods);
        dispatch_methods.push(slot.dispatch_method);
        default_methods.push(slot.default_method);
    }

    // Build nested hooks fields and inits.
    let nested_fields: Vec<proc_macro2::TokenStream> = nested_hooks_entries
        .iter()
        .map(|(fid, fty)| quote! { pub #fid: #fty, })
        .collect();
    let nested_field_inits: Vec<proc_macro2::TokenStream> = nested_hooks_entries
        .iter()
        .map(|(fid, fty)| {
            quote! { #fid: <#fty as ::std::default::Default>::default(), }
        })
        .collect();

    // new() signature: no args when no nested hooks (backwards-compat), positional args otherwise.
    let new_fn = if nested_hooks_entries.is_empty() {
        quote! {
            pub fn new() -> Self {
                Self::default()
            }
        }
    } else {
        let params: Vec<proc_macro2::TokenStream> = nested_hooks_entries
            .iter()
            .map(|(fid, fty)| quote! { #fid: #fty })
            .collect();
        let param_names: Vec<&syn::Ident> =
            nested_hooks_entries.iter().map(|(fid, _)| fid).collect();
        quote! {
            pub fn new(#(#params,)*) -> Self {
                Self {
                    #(#param_names,)*
                    ..::std::default::Default::default()
                }
            }
        }
    };

    quote! {
        #(#items)*

        #(#struct_attrs)*
        #[allow(dead_code)]
        #vis trait #ident: ::lutum::HookObject {
            #(#trait_methods)*
        }

        impl<__LutumHooksRef> #ident for &__LutumHooksRef
        where
            __LutumHooksRef: #ident + ?Sized,
        {
            #(#trait_ref_impl_methods)*
        }

        #[doc(hidden)]
        #[allow(dead_code)]
        #vis trait #register_trait_ident<'__lutum_hooks> {
            fn __lutum_register_hooks(self, hooks: &mut #set_ident<'__lutum_hooks>);
        }

        #(#struct_attrs)*
        #[allow(dead_code)]
        #[derive(::std::clone::Clone)]
        #vis struct #set_ident<'__lutum_hooks> {
            #(#nested_fields)*
            #(#fields)*
        }

        #(#struct_cfg_attrs)*
        impl<'__lutum_hooks> ::std::default::Default for #set_ident<'__lutum_hooks> {
            fn default() -> Self {
                Self {
                    #(#nested_field_inits)*
                    #(#field_inits)*
                }
            }
        }

        #(#struct_cfg_attrs)*
        #[allow(dead_code)]
        impl<'__lutum_hooks> #set_ident<'__lutum_hooks> {
            #new_fn

            pub fn with_hooks<H>(mut self, hooks: H) -> Self
            where
                H: #register_trait_ident<'__lutum_hooks>,
            {
                self.register_hooks(hooks);
                self
            }

            pub fn register_hooks<H>(&mut self, hooks: H) -> &mut Self
            where
                H: #register_trait_ident<'__lutum_hooks>,
            {
                hooks.__lutum_register_hooks(self);
                self
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

fn is_nested_hooks_attr(attr: &Attribute) -> bool {
    let path = attr.path();
    // Accept both `nested_hooks` and `lutum::nested_hooks`.
    if path.is_ident("nested_hooks") {
        return true;
    }
    path.segments
        .last()
        .is_some_and(|last| last.ident == "nested_hooks")
}
