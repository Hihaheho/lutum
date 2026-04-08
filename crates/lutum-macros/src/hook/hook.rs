use super::*;
use quote::{format_ident, quote, quote_spanned};
use syn::{GenericParam, ItemFn, Path, PathArguments, Type};

pub fn expand_hook_impl(item_fn: ItemFn, slot_path: Path) -> proc_macro2::TokenStream {
    let (_slot_ident, _slot_type_args) = match slot_path.segments.last() {
        Some(seg) => match &seg.arguments {
            PathArguments::None => (seg.ident.clone(), None),
            PathArguments::AngleBracketed(args) => (seg.ident.clone(), Some(args.clone())),
            _ => {
                return syn::Error::new_spanned(
                    slot_path,
                    "unsupported slot path arguments",
                )
                .to_compile_error();
            }
        },
        None => {
            return syn::Error::new_spanned(
                slot_path,
                "empty slot path",
            )
            .to_compile_error()
        }
    };

    let helper_macro_path = hook_named_impl_helper_macro_path(&slot_path);

    let HookSignature {
        explicit_args,
        output_ty,
        has_last,
        last_span,
        generics,
    } = match analyze_hook_signature(
        &item_fn,
        HookLastRequirement::Optional,
        "#[hook(SlotType)] received an invalid `last: Option<Return>` argument",
        HookLastRecognition::CompatibleOption,
        true,
    ) {
        Ok(signature) => signature,
        Err(err) => return err.to_compile_error(),
    };

    let fn_ident = item_fn.sig.ident.clone();
    let vis = item_fn.vis.clone();
    let struct_ident = format_ident!("{}", fn_ident.to_string().to_upper_camel_case());
    let impl_fn_ident = format_ident!("__lutum_hook_impl_{}", fn_ident);
    let hook_trait_path = slot_path.clone();
    let struct_attrs = item_fn
        .attrs
        .iter()
        .filter(|attr| {
            let path = attr.path();
            path.is_ident("cfg")
                || path.is_ident("cfg_attr")
                || path.is_ident("doc")
                || path.is_ident("deprecated")
        })
        .cloned()
        .collect::<Vec<_>>();

    let mut item_fn = item_fn;
    item_fn.vis = syn::Visibility::Inherited;
    item_fn.sig.ident = impl_fn_ident.clone();
    item_fn.attrs.push(syn::parse_quote!(#[allow(dead_code)]));

    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
    let struct_generics = strip_generic_bounds(&generics);
    let (struct_impl_generics, struct_ty_generics, struct_where_clause) =
        struct_generics.split_for_impl();
    let (const_marker_defs, generic_marker_types) = generic_marker_types(&fn_ident, &generics);
    let (struct_def, default_impl) = if generic_marker_types.is_empty() {
        (
            quote! {
                #[allow(dead_code)]
                #(#struct_attrs)*
                #vis struct #struct_ident;
            },
            quote! {
                impl #struct_impl_generics ::std::default::Default for #struct_ident #struct_ty_generics
                #struct_where_clause
                {
                    fn default() -> Self {
                        Self
                    }
                }
            },
        )
    } else {
        (
            quote! {
                #[allow(dead_code)]
                #(#struct_attrs)*
                #vis struct #struct_ident #struct_generics(
                    ::std::marker::PhantomData<fn(#(#generic_marker_types),*)>
                );
            },
            quote! {
                impl #struct_impl_generics ::std::default::Default for #struct_ident #struct_ty_generics
                #struct_where_clause
                {
                    fn default() -> Self {
                        Self(::std::marker::PhantomData)
                    }
                }
            },
        )
    };

    // Field idents: original param names with leading `_` stripped where unambiguous.
    let args_field_idents = normalized_hook_arg_field_idents(&explicit_args);

    // Base trait method params (without `last`).
    let trait_args_no_last: Vec<proc_macro2::TokenStream> = explicit_args
        .iter()
        .zip(args_field_idents.iter())
        .map(|((_, ty), fi)| {
            let param_ty = hook_param_type(ty);
            quote! { #fi: #param_ty }
        })
        .collect();

    // fn_call_args: expressions to forward to the original function.
    // Uses field_idents (from the trait method params), re-adding `&` for &str args.
    let mut fn_call_args: Vec<proc_macro2::TokenStream> = explicit_args
        .iter()
        .zip(args_field_idents.iter())
        .map(|((_, ty), fi)| match ty {
            Type::Reference(r) if is_str_type(&r.elem) => quote! { &#fi },
            _ => quote! { #fi },
        })
        .collect();
    if has_last {
        fn_call_args.push(quote! { last });
    }
    let call_generics: Vec<proc_macro2::TokenStream> = generics
        .params
        .iter()
        .filter_map(|param| match param {
            GenericParam::Type(type_param) => {
                let ident = &type_param.ident;
                Some(quote! { #ident })
            }
            GenericParam::Const(const_param) => {
                let ident = &const_param.ident;
                Some(quote! { #ident })
            }
            GenericParam::Lifetime(_) => None,
        })
        .collect();
    let impl_fn_call = if call_generics.is_empty() {
        quote! { #impl_fn_ident(#(#fn_call_args,)*).await }
    } else {
        quote! { #impl_fn_ident::<#(#call_generics,)*>(#(#fn_call_args,)*).await }
    };

    // Impl body: trait `call` method forwards to the original fn.
    // `trait_args` and `fn_call_args` vary based on whether `last` is present.
    let make_impl = |trait_args: Vec<proc_macro2::TokenStream>| {
        quote! {
            #(#const_marker_defs)*
            #struct_def
            #default_impl
            #[::async_trait::async_trait]
            impl #impl_generics #hook_trait_path for #struct_ident #ty_generics
            #where_clause
            {
                async fn call(
                    &self,
                    #(#trait_args,)*
                ) -> #output_ty {
                    #impl_fn_call
                }
            }
        }
    };

    let with_last_dispatch = if has_last {
        // User opted into `last` — trait must also have it. Error for singleton slots.
        let mut trait_args_with_last = trait_args_no_last.clone();
        trait_args_with_last.push(quote! { last: ::std::option::Option<#output_ty> });
        let ok_impl = make_impl(trait_args_with_last);
        let err_span = last_span.expect("last span must exist when last is present");
        let err_impl = quote_spanned! { err_span =>
            compile_error!(
                "#[hook(SlotType)] implementations for this slot must not declare `last: Option<Return>`"
            );
        };
        quote! {
            #helper_macro_path!(@named_impl_with_last { #ok_impl } { #err_impl });
        }
    } else {
        // User did not opt into `last`. Dispatch on whether the slot trait has `last`:
        // - always/fallback slots: trait has `last`, accept it but don't forward to user fn
        // - singleton slots: trait has no `last`, emit without it
        let mut trait_args_with_last = trait_args_no_last.clone();
        trait_args_with_last.push(quote! { last: ::std::option::Option<#output_ty> });
        let ok_with_last = make_impl(trait_args_with_last);
        let ok_no_last = make_impl(trait_args_no_last);
        quote! {
            #helper_macro_path!(@named_impl_with_last { #ok_with_last } { #ok_no_last });
        }
    };

    quote! {
        #item_fn

        #with_last_dispatch
    }
}

fn hook_named_impl_helper_macro_path(slot_path: &Path) -> Path {
    let mut helper_path = slot_path.clone();
    let last = helper_path
        .segments
        .last_mut()
        .expect("hook slot paths must have at least one segment");
    last.ident = hook_named_impl_helper_macro_ident(&last.ident);
    last.arguments = PathArguments::None;
    helper_path
}

fn is_str_type(ty: &Type) -> bool {
    matches!(ty, Type::Path(p) if p.path.is_ident("str"))
}

fn strip_generic_bounds(generics: &syn::Generics) -> syn::Generics {
    let mut generics = generics.clone();
    generics.where_clause = None;
    for param in &mut generics.params {
        match param {
            GenericParam::Type(type_param) => {
                type_param.colon_token = None;
                type_param.bounds.clear();
                type_param.eq_token = None;
                type_param.default = None;
            }
            GenericParam::Lifetime(lifetime_param) => {
                lifetime_param.colon_token = None;
                lifetime_param.bounds.clear();
            }
            GenericParam::Const(const_param) => {
                const_param.eq_token = None;
                const_param.default = None;
            }
        }
    }
    generics
}

fn generic_marker_types(
    fn_ident: &syn::Ident,
    generics: &syn::Generics,
) -> (Vec<proc_macro2::TokenStream>, Vec<proc_macro2::TokenStream>) {
    let mut helper_defs = Vec::new();
    let marker_types = generics
        .params
        .iter()
        .map(|param| match param {
            GenericParam::Type(type_param) => {
                let ident = &type_param.ident;
                quote! { #ident }
            }
            GenericParam::Lifetime(lifetime_param) => {
                let lifetime = &lifetime_param.lifetime;
                quote! { &#lifetime () }
            }
            GenericParam::Const(const_param) => {
                let const_ty = &const_param.ty;
                let const_ident = &const_param.ident;
                let marker_ident = format_ident!(
                    "__LutumHookConstParamMarker_{}_{}",
                    fn_ident,
                    const_ident
                );
                helper_defs.push(quote! {
                    #[allow(dead_code)]
                    struct #marker_ident<const __VALUE: #const_ty>;
                });
                quote! { #marker_ident<#const_ident> }
            }
        })
        .collect();
    (helper_defs, marker_types)
}
