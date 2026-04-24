use super::*;
use heck::ToUpperCamelCase;
use quote::{format_ident, quote};
use syn::{
    FnArg, ImplItem, ImplItemFn, ItemImpl, Pat, PatIdent, Path, PathArguments, PathSegment, Type,
    punctuated::Punctuated, token::PathSep,
};

pub fn expand_hooks_impl(
    mut item_impl: ItemImpl,
    hooks_set_path: Path,
) -> proc_macro2::TokenStream {
    let Some((_, trait_path, _)) = item_impl.trait_.clone() else {
        return syn::Error::new_spanned(item_impl, "#[impl_hooks] requires `impl Trait for Type`")
            .to_compile_error();
    };

    let set_ident = match hooks_set_path.segments.last() {
        Some(seg) if matches!(seg.arguments, PathArguments::None) => seg.ident.clone(),
        Some(seg) => {
            return syn::Error::new_spanned(
                &seg.arguments,
                "#[impl_hooks(...)] hook set path must not include generic arguments",
            )
            .to_compile_error();
        }
        None => {
            return syn::Error::new_spanned(hooks_set_path, "empty hook set path")
                .to_compile_error();
        }
    };
    let register_trait_ident = format_ident!("__Lutum{set_ident}RegisterHooks");
    let register_trait_path = sibling_path(&hooks_set_path, register_trait_ident);
    let self_ty = item_impl.self_ty.as_ref().clone();

    let mut methods = Vec::new();
    let mut transformed_items = Vec::new();
    let original_items = ::std::mem::take(&mut item_impl.items);
    for item in original_items.into_iter() {
        match item {
            ImplItem::Fn(method) => {
                let method = match transform_impl_method(method, &hooks_set_path, &trait_path) {
                    Ok((method, registration)) => {
                        methods.push(registration);
                        method
                    }
                    Err(err) => return err.to_compile_error(),
                };
                transformed_items.push(ImplItem::Fn(method));
            }
            other => {
                return syn::Error::new_spanned(
                    other,
                    "#[impl_hooks] supports only hook method implementations",
                )
                .to_compile_error();
            }
        }
    }

    if methods.is_empty() {
        return syn::Error::new_spanned(
            item_impl,
            "#[impl_hooks] impl block must contain at least one hook method",
        )
        .to_compile_error();
    }

    item_impl.items = transformed_items;

    let adapter_items = methods
        .iter()
        .map(|m| m.adapter_item.clone())
        .collect::<Vec<_>>();
    let register_calls = methods
        .iter()
        .map(|m| m.register_call.clone())
        .collect::<Vec<_>>();

    let mut owned_generics = item_impl.generics.clone();
    owned_generics
        .params
        .insert(0, syn::parse_quote!('__lutum_hooks));
    owned_generics
        .make_where_clause()
        .predicates
        .push(syn::parse_quote!(#self_ty: #trait_path + '__lutum_hooks));
    let (owned_impl_generics, _, owned_where_clause) = owned_generics.split_for_impl();

    let mut ref_generics = item_impl.generics.clone();
    ref_generics
        .params
        .insert(0, syn::parse_quote!('__lutum_hooks));
    ref_generics
        .make_where_clause()
        .predicates
        .push(syn::parse_quote!(#self_ty: #trait_path + '__lutum_hooks));
    let (ref_impl_generics, _, ref_where_clause) = ref_generics.split_for_impl();

    quote! {
        #item_impl

        const _: () = {
            #(#adapter_items)*

            impl #owned_impl_generics #register_trait_path<'__lutum_hooks> for #self_ty
            #owned_where_clause
            {
                fn __lutum_register_hooks(self, hooks: &mut #hooks_set_path<'__lutum_hooks>) {
                    let __lutum_hook = ::std::sync::Arc::new(self);
                    #(#register_calls)*
                }
            }

            impl #ref_impl_generics #register_trait_path<'__lutum_hooks> for &'__lutum_hooks #self_ty
            #ref_where_clause
            {
                fn __lutum_register_hooks(self, hooks: &mut #hooks_set_path<'__lutum_hooks>) {
                    let __lutum_hook = ::std::sync::Arc::new(self);
                    #(#register_calls)*
                }
            }
        };
    }
}

struct MethodRegistration {
    adapter_item: proc_macro2::TokenStream,
    register_call: proc_macro2::TokenStream,
}

fn transform_impl_method(
    method: ImplItemFn,
    hooks_set_path: &Path,
    owner_trait_path: &Path,
) -> syn::Result<(ImplItemFn, MethodRegistration)> {
    let signature = analyze_impl_method_signature(&method)?;
    let HookSignature {
        explicit_args,
        output_ty,
        has_last,
        ..
    } = signature;

    let method_ident = method.sig.ident.clone();
    let adapter_ident = format_ident!(
        "__LutumImplHooks{}",
        method_ident.to_string().to_upper_camel_case()
    );
    let slot_trait_ident = format_ident!("{}", method_ident.to_string().to_upper_camel_case());
    let slot_trait_path = sibling_path(hooks_set_path, slot_trait_ident);
    let register_fn_ident = format_ident!("register_{}", method_ident);

    let args_field_idents = normalized_hook_arg_field_idents(&explicit_args);
    let mut trait_args = explicit_args
        .iter()
        .zip(args_field_idents.iter())
        .map(|((_, ty), fi)| {
            let param_ty = hook_param_type(ty);
            quote! { #fi: #param_ty }
        })
        .collect::<Vec<_>>();
    if has_last {
        trait_args.push(quote! { last: ::std::option::Option<#output_ty> });
    }

    let mut call_args = explicit_args
        .iter()
        .zip(args_field_idents.iter())
        .map(|((_, ty), fi)| match ty {
            Type::Reference(r) if is_str_type(&r.elem) => quote! { &#fi },
            _ => quote! { #fi },
        })
        .collect::<Vec<_>>();
    if has_last {
        call_args.push(quote! { last });
    }

    let adapter_item = quote! {
        struct #adapter_ident<__LutumHook>(::std::sync::Arc<__LutumHook>);

        impl<__LutumHook> #slot_trait_path for #adapter_ident<__LutumHook>
        where
            __LutumHook: #owner_trait_path + ::lutum::HookObject,
        {
            async fn call(
                &self,
                #(#trait_args,)*
            ) -> #output_ty {
                <__LutumHook as #owner_trait_path>::#method_ident(&*self.0, #(#call_args,)*).await
            }
        }
    };
    let register_call = quote! {
        hooks.#register_fn_ident(#adapter_ident(::std::sync::Arc::clone(&__lutum_hook)));
    };

    Ok((
        method,
        MethodRegistration {
            adapter_item,
            register_call,
        },
    ))
}

fn analyze_impl_method_signature(method: &ImplItemFn) -> syn::Result<HookSignature> {
    if method.sig.asyncness.is_none() {
        return Err(syn::Error::new_spanned(
            &method.sig,
            "#[impl_hooks] methods must be async fn",
        ));
    }
    if !method.sig.generics.params.is_empty() || method.sig.generics.where_clause.is_some() {
        return Err(syn::Error::new_spanned(
            &method.sig.generics,
            "#[impl_hooks] methods do not support generics or where clauses",
        ));
    }
    if method.sig.constness.is_some()
        || method.sig.unsafety.is_some()
        || method.sig.abi.is_some()
        || method.sig.variadic.is_some()
    {
        return Err(syn::Error::new_spanned(
            &method.sig,
            "#[impl_hooks] supports only plain async methods",
        ));
    }

    let mut inputs = method.sig.inputs.iter();
    match inputs.next() {
        Some(FnArg::Receiver(receiver))
            if receiver.reference.is_some()
                && receiver.mutability.is_none()
                && receiver.colon_token.is_none() => {}
        Some(other) => {
            return Err(syn::Error::new_spanned(
                other,
                "#[impl_hooks] hook methods must take `&self`",
            ));
        }
        None => {
            return Err(syn::Error::new_spanned(
                &method.sig,
                "#[impl_hooks] hook methods must take `&self`",
            ));
        }
    }

    let output_ty = output_type_or_unit(&method.sig.output);
    let inputs = inputs.collect::<Vec<_>>();
    let last_arg = inputs.last().copied();
    let has_last = last_arg
        .map(|arg| hook_last_matches(arg, &output_ty, HookLastRecognition::CompatibleOption))
        .transpose()?
        .unwrap_or(false);

    let mut explicit_args = Vec::new();
    let explicit_end = inputs.len() - usize::from(has_last);
    for arg in &inputs[..explicit_end] {
        let FnArg::Typed(pat_ty) = arg else {
            return Err(syn::Error::new_spanned(arg, "unsupported hook argument"));
        };
        let Pat::Ident(PatIdent { ident, .. }) = pat_ty.pat.as_ref() else {
            return Err(syn::Error::new_spanned(
                &pat_ty.pat,
                "expected identifier pattern",
            ));
        };
        explicit_args.push((ident.clone(), (*pat_ty.ty).clone()));
    }

    Ok(HookSignature {
        explicit_args,
        output_ty,
        has_last,
        generics: syn::Generics::default(),
    })
}

fn sibling_path(path: &Path, ident: syn::Ident) -> Path {
    let mut segments: Punctuated<PathSegment, PathSep> = path.segments.clone();
    segments.pop();
    segments.push(PathSegment::from(ident));
    Path {
        leading_colon: path.leading_colon,
        segments,
    }
}
