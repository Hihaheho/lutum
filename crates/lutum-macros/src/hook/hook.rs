use super::*;
use heck::ToUpperCamelCase;
use quote::{format_ident, quote};
use syn::{GenericParam, ItemFn, Path, PathArguments, Type};

pub fn expand_hook_impl(item_fn: ItemFn, slot_path: Path) -> proc_macro2::TokenStream {
    let (_slot_ident, _slot_type_args) = match slot_path.segments.last() {
        Some(seg) => match &seg.arguments {
            PathArguments::None => (seg.ident.clone(), None),
            PathArguments::AngleBracketed(args) => (seg.ident.clone(), Some(args.clone())),
            _ => {
                return syn::Error::new_spanned(slot_path, "unsupported hook path arguments")
                    .to_compile_error();
            }
        },
        None => return syn::Error::new_spanned(slot_path, "empty hook path").to_compile_error(),
    };

    let HookSignature {
        explicit_args,
        output_ty,
        has_last,
        generics,
    } = match analyze_hook_signature(
        &item_fn,
        HookLastRequirement::Optional,
        "#[impl_hook] received an invalid `last: Option<Return>` argument",
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

    let args_field_idents = normalized_hook_arg_field_idents(&explicit_args);
    let mut trait_args: Vec<proc_macro2::TokenStream> = explicit_args
        .iter()
        .zip(args_field_idents.iter())
        .map(|((_, ty), fi)| {
            let param_ty = hook_param_type(ty);
            quote! { #fi: #param_ty }
        })
        .collect();
    if has_last {
        trait_args.push(quote! { last: ::std::option::Option<#output_ty> });
    }

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

    quote! {
        #item_fn

        #(#const_marker_defs)*
        #struct_def
        #default_impl

        #[allow(refining_impl_trait)]
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
                let helper_ident =
                    format_ident!("__LutumConstMarker_{}_{}", fn_ident, const_param.ident);
                let ty = &const_param.ty;
                helper_defs.push(quote! {
                    #[allow(dead_code)]
                    struct #helper_ident<const V: #ty>;
                });
                let ident = &const_param.ident;
                quote! { #helper_ident<#ident> }
            }
        })
        .collect::<Vec<_>>();
    (helper_defs, marker_types)
}

fn is_str_type(ty: &Type) -> bool {
    matches!(ty, Type::Path(p) if p.path.is_ident("str"))
}
