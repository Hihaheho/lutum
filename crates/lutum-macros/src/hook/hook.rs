use super::*;
use quote::{format_ident, quote};
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
    let hook_trait_path = hook_trait_path_for_slot_path(&slot_path);
    let args_struct_path = args_struct_path_for_slot_path(&slot_path);
    let has_explicit_args = !explicit_args.is_empty();
    let has_ref_arg = explicit_args.iter().any(|(_, ty)| is_non_str_ref(ty));
    let args_struct_type: proc_macro2::TokenStream = if has_ref_arg {
        quote! { #args_struct_path<'_> }
    } else {
        quote! { #args_struct_path }
    };

    // Trait method signature: uses XxxArgs when there are explicit args.
    let mut hook_trait_args: Vec<proc_macro2::TokenStream> = if has_explicit_args {
        vec![quote! { args: #args_struct_type }]
    } else {
        vec![]
    };
    if has_last {
        hook_trait_args.push(quote! { last: ::std::option::Option<#output_ty> });
    }

    // When calling the original function, unpack args back to original types through
    // `into_parts()`. This keeps #[hook] impls decoupled from the public field names.
    let args_unpack_bindings = explicit_args
        .iter()
        .map(|(ident, _)| quote! { #ident })
        .collect::<Vec<_>>();
    let args_unpack = if has_explicit_args {
        quote! {
            let (#(#args_unpack_bindings,)*) = args.into_parts();
        }
    } else {
        quote! {}
    };
    let mut fn_call_args: Vec<proc_macro2::TokenStream> = explicit_args
        .iter()
        .map(|(ident, ty)| match ty {
            Type::Reference(r) if is_str_type(&r.elem) => quote! { &#ident },
            _ => quote! { #ident },
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
                #args_unpack
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
        #item_fn

        #with_last_dispatch
    }
}

fn args_struct_path_for_slot_path(slot_path: &Path) -> Path {
    let mut args_path = slot_path.clone();
    let last = args_path
        .segments
        .last_mut()
        .expect("hook slot paths must have at least one segment");
    last.ident = format_ident!("{}Args", last.ident);
    args_path
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

fn hook_trait_path_for_slot_path(slot_path: &Path) -> Path {
    let mut trait_path = slot_path.clone();
    let last = trait_path
        .segments
        .last_mut()
        .expect("hook slot paths must have at least one segment");
    last.ident = format_ident!("{}Hook", last.ident);
    trait_path
}

fn is_str_type(ty: &Type) -> bool {
    matches!(ty, Type::Path(p) if p.path.is_ident("str"))
}
