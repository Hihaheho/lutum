use super::*;
use quote::{format_ident, quote, quote_spanned};
use syn::{ItemFn, Path, PathArguments, Type};

pub fn expand_hook_impl(item_fn: ItemFn, slot_path: Path) -> proc_macro2::TokenStream {
    match slot_path.segments.last() {
        Some(segment) if matches!(segment.arguments, PathArguments::None) => {}
        _ => {
            return syn::Error::new_spanned(
                slot_path,
                "#[hook(...)] expects a plain hook slot path like `SlotType` or `path::to::SlotType`",
            )
            .to_compile_error();
        }
    };

    let helper_macro_path = hook_named_impl_helper_macro_path(&slot_path);

    let HookSignature {
        ctx_ident,
        ctx_ty,
        explicit_args,
        output_ty,
        has_last,
        last_span,
    } = match analyze_hook_signature(
        &item_fn,
        HookLastRequirement::Optional,
        "#[hook(SlotType)] received an invalid `last: Option<Return>` argument",
        HookLastRecognition::CompatibleOption,
    ) {
        Ok(signature) => signature,
        Err(err) => return err.to_compile_error(),
    };

    let fn_ident = item_fn.sig.ident.clone();
    let vis = item_fn.vis.clone();
    let struct_ident = format_ident!("{}", fn_ident.to_string().to_upper_camel_case());
    let hook_trait_path = slot_path.clone();

    // Field idents: original param names with leading `_` stripped where unambiguous.
    let args_field_idents = normalized_hook_arg_field_idents(&explicit_args);

    // Trait method params: individual field_ident: hook_param_type (no *Args struct).
    let mut hook_trait_args: Vec<proc_macro2::TokenStream> = explicit_args
        .iter()
        .zip(args_field_idents.iter())
        .map(|((_, ty), fi)| {
            let param_ty = hook_param_type(ty);
            quote! { #fi: #param_ty }
        })
        .collect();
    if has_last {
        hook_trait_args.push(quote! { last: ::std::option::Option<#output_ty> });
    }

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

    let ok_impl = quote! {
        #[allow(dead_code)]
        #vis struct #struct_ident;

        #[::async_trait::async_trait]
        impl #hook_trait_path for #struct_ident {
            async fn call(
                &self,
                #ctx_ident: #ctx_ty,
                #(#hook_trait_args,)*
            ) -> #output_ty {
                #fn_ident(#ctx_ident, #(#fn_call_args,)*).await
            }
        }
    };

    let with_last_dispatch = if has_last {
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
        quote! { #ok_impl }
    };

    quote! {
        #[allow(dead_code)]
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
    helper_path
}

fn is_str_type(ty: &Type) -> bool {
    matches!(ty, Type::Path(p) if p.path.is_ident("str"))
}
