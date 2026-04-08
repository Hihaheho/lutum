use super::*;
use heck::ToUpperCamelCase;
use quote::{format_ident, quote};
use syn::{ItemFn, Type};

pub fn expand_local_hook(mut item_fn: ItemFn, kind: HookKind) -> proc_macro2::TokenStream {
    let HookSignature {
        ctx_ident,
        ctx_ty,
        explicit_args,
        output_ty,
        has_last: _,
        last_span: _,
    } = match analyze_hook_signature(
        &item_fn,
        kind.default_last_requirement(),
        "#[def_hook(singleton)] does not accept a `last: Option<Return>` argument",
        HookLastRecognition::LastNamedCompatibleOption,
    ) {
        Ok(signature) => signature,
        Err(err) => return err.to_compile_error(),
    };
    let trait_has_last = kind.trait_has_last();

    let fn_ident = item_fn.sig.ident.clone();
    let vis = item_fn.vis.clone();
    let hook_name = fn_ident.to_string();
    let slot_ident = format_ident!("{}", hook_name.to_upper_camel_case());
    let hook_trait_ident = slot_ident.clone();
    let stateful_hook_trait_ident = format_ident!("Stateful{slot_ident}");
    let dyn_hook_trait_ident = format_ident!("__LutumDyn{slot_ident}");
    let default_impl_fn_ident = format_ident!("__lutum_hook_default_impl_{}", fn_ident);
    let default_method_ident = format_ident!("__lutum_hook_default_{}", fn_ident);
    let with_fn_ident = format_ident!("with_{}", fn_ident);
    let register_fn_ident = format_ident!("register_{}", fn_ident);
    let field_ident = fn_ident.clone();
    let is_lutum_hook = is_lutum_ref(&ctx_ty);
    let ctx_inner_ty: Type = match &ctx_ty {
        Type::Reference(r) => *r.elem.clone(),
        other => other.clone(),
    };

    item_fn.vis = syn::parse_quote!(pub(crate));
    item_fn.sig.ident = default_impl_fn_ident.clone();
    item_fn.attrs.push(syn::parse_quote!(#[allow(dead_code)]));

    let args_field_idents = normalized_hook_arg_field_idents(&explicit_args);

    // dispatch_args: params in the user-facing dispatch method (original types).
    let dispatch_args = explicit_args
        .iter()
        .map(|(ident, ty)| quote! { #ident: #ty })
        .collect::<Vec<_>>();

    // hook_call_arg_names: original param names used inside the dispatch body.
    let hook_call_arg_names = explicit_args
        .iter()
        .map(|(ident, _)| quote! { #ident })
        .collect::<Vec<_>>();

    // cloned_hook_call_arg_names: for default_call (original types, cloned/ref as needed).
    let cloned_hook_call_arg_names = explicit_args
        .iter()
        .map(|(ident, ty)| {
            if matches!(ty, Type::Reference(_)) {
                quote! { #ident }
            } else {
                quote! { #ident.clone() }
            }
        })
        .collect::<Vec<_>>();

    let has_explicit_args = !explicit_args.is_empty();
    let has_ref_arg = explicit_args.iter().any(|(_, ty)| is_non_str_ref(ty));

    let def_span = item_fn.sig.ident.span();
    let arg_tokens = compute_hook_arg_tokens(&explicit_args, &args_field_idents, &output_ty, trait_has_last);
    let dispatch_vars = &arg_tokens.dispatch_vars;
    let cloned_field_idents = &arg_tokens.cloned;
    let args_pre_conversion = &arg_tokens.pre_conversion;
    let clone_where = &arg_tokens.clone_where;

    let slot = HookSlotIdents {
        hook_trait: hook_trait_ident.clone(),
        stateful: stateful_hook_trait_ident.clone(),
        dyn_trait: dyn_hook_trait_ident.clone(),
    };
    let flags = HookSlotFlags {
        trait_has_last,
        is_lutum_hook,
        has_explicit_args,
        has_ref_arg,
    };

    let hook_trait_defs = generate_hook_trait_defs(
        def_span,
        &vis,
        &ctx_ty,
        &ctx_ident,
        &output_ty,
        &slot,
        &arg_tokens.trait_args,
    );

    let fn_impl = generate_fn_blanket_impl(
        &slot,
        &flags,
        &ctx_ty,
        &ctx_inner_ty,
        &ctx_ident,
        &output_ty,
        &arg_tokens,
    );

    let blanket_impls = generate_blanket_impls(
        &slot,
        &ctx_ty,
        &ctx_ident,
        &output_ty,
        &arg_tokens,
        &hook_name,
    );

    let default_impl_call = quote! {
        #default_impl_fn_ident(
            #ctx_ident,
            #(#hook_call_arg_names,)*
        )
        .await
    };
    let default_call = quote! {
        Self::#default_method_ident(
            #ctx_ident,
            #(#cloned_hook_call_arg_names,)*
        )
        .await
    };
    let dyn_hook_dispatch_call = if trait_has_last {
        quote! { hook.call_dyn(#ctx_ident, #(#cloned_field_idents,)* last).await }
    } else {
        quote! { hook.call_dyn(#ctx_ident, #(#cloned_field_idents,)*).await }
    };

    let (field_ty, field_init, register_impl) = match &kind {
        HookKind::Always { .. } | HookKind::Fallback { .. } => (
            quote! {
                ::std::vec::Vec<::std::sync::Arc<dyn #dyn_hook_trait_ident>>
            },
            quote! { ::std::vec::Vec::new() },
            quote! {
                self.#field_ident.push(::std::sync::Arc::new(hook));
            },
        ),
        HookKind::Singleton => (
            quote! {
                ::std::option::Option<::std::sync::Arc<dyn #dyn_hook_trait_ident>>
            },
            quote! { ::std::option::Option::None },
            quote! {
                let hook = ::std::sync::Arc::new(hook)
                    as ::std::sync::Arc<dyn #dyn_hook_trait_ident>;
                if self.#field_ident.replace(hook).is_some() {
                    ::tracing::warn!(
                        slot = #hook_name,
                        "singleton hook registration overwritten; last registered hook wins"
                    );
                }
            },
        ),
    };
    let inner_dispatch = match &kind {
        HookKind::Always(HookOptions { chain: None, accumulate: None, .. }) => quote! {
            let mut last = ::std::option::Option::Some(#default_call);
            for hook in &self.#field_ident {
                last = ::std::option::Option::Some(#dyn_hook_dispatch_call);
            }
            last.expect("hook chain unexpectedly empty")
        },
        HookKind::Always(HookOptions { chain: Some(chain_path), accumulate: None, .. }) => quote! {
            let mut last = ::std::option::Option::Some(#default_call);
            if #chain_path(last.as_ref().unwrap()).is_break() {
                return last.unwrap();
            }
            for hook in &self.#field_ident {
                let next = #dyn_hook_dispatch_call;
                if #chain_path(&next).is_break() {
                    return next;
                }
                last = ::std::option::Option::Some(next);
            }
            last.unwrap()
        },
        HookKind::Always(HookOptions { chain: None, accumulate: Some(accumulate_fn), .. }) => quote! {
            let mut __outputs = ::std::vec::Vec::new();
            __outputs.push(#default_call);
            for hook in &self.#field_ident {
                __outputs.push(#dyn_hook_dispatch_call);
            }
            #accumulate_fn(__outputs)
        },
        HookKind::Always(HookOptions { chain: Some(chain_path), accumulate: Some(accumulate_fn), .. }) => quote! {
            let mut __outputs = ::std::vec::Vec::new();
            let __first = #default_call;
            if #chain_path(&__first).is_break() {
                __outputs.push(__first);
                return #accumulate_fn(__outputs);
            }
            __outputs.push(__first);
            for hook in &self.#field_ident {
                let __out = #dyn_hook_dispatch_call;
                if #chain_path(&__out).is_break() {
                    __outputs.push(__out);
                    return #accumulate_fn(__outputs);
                }
                __outputs.push(__out);
            }
            #accumulate_fn(__outputs)
        },
        HookKind::Fallback(HookOptions { chain: None, accumulate: None, .. }) => quote! {
            if self.#field_ident.is_empty() {
                #default_call
            } else {
                let mut last = ::std::option::Option::None;
                for hook in &self.#field_ident {
                    last = ::std::option::Option::Some(#dyn_hook_dispatch_call);
                }
                last.expect("hook chain unexpectedly empty")
            }
        },
        HookKind::Fallback(HookOptions { chain: Some(chain_path), accumulate: None, .. }) => quote! {
            if self.#field_ident.is_empty() {
                #default_call
            } else {
                let mut last = ::std::option::Option::None;
                for hook in &self.#field_ident {
                    let next = #dyn_hook_dispatch_call;
                    if #chain_path(&next).is_break() {
                        return next;
                    }
                    last = ::std::option::Option::Some(next);
                }
                #default_call
            }
        },
        HookKind::Fallback(HookOptions { chain: None, accumulate: Some(accumulate_fn), .. }) => quote! {
            if self.#field_ident.is_empty() {
                #default_call
            } else {
                let mut __outputs = ::std::vec::Vec::new();
                for hook in &self.#field_ident {
                    __outputs.push(#dyn_hook_dispatch_call);
                }
                #accumulate_fn(__outputs)
            }
        },
        HookKind::Fallback(HookOptions { chain: Some(chain_path), accumulate: Some(accumulate_fn), .. }) => quote! {
            if self.#field_ident.is_empty() {
                #default_call
            } else {
                let mut __outputs = ::std::vec::Vec::new();
                for hook in &self.#field_ident {
                    let __out = #dyn_hook_dispatch_call;
                    if #chain_path(&__out).is_break() {
                        __outputs.push(__out);
                        return #accumulate_fn(__outputs);
                    }
                    __outputs.push(__out);
                }
                #accumulate_fn(__outputs)
            }
        },
        HookKind::Singleton => {
            let singleton_args: Vec<proc_macro2::TokenStream> =
                dispatch_vars.iter().map(|v| quote! { #v }).collect();
            let some_call = quote! {
                hook.call_dyn(#ctx_ident, #(#singleton_args,)*).await
            };
            quote! {
                match &self.#field_ident {
                    Some(hook) => #some_call,
                    None => #default_call,
                }
            }
        }
    };

    // Wrap with finalize if specified. Chain dispatch uses early `return`s, so we wrap
    // in an async block to capture all exit paths through the same finalize call.
    let dispatch = match kind.opts().and_then(|o| o.finalize.as_ref()) {
        Some(finalize_fn) => quote! {
            let __result = async move { #inner_dispatch }.await;
            #finalize_fn(__result)
        },
        None => inner_dispatch,
    };

    let hooks_slot_ident = format_ident!("__lutum_hooks_{}", slot_ident);
    let macro_reexport = quote! {
        #[doc(hidden)]
        pub(crate) use #slot_ident as #hooks_slot_ident;
    };
    let named_impl_helper_macro_ident = hook_named_impl_helper_macro_ident(&slot_ident);
    let helper_macro_reexport = quote! {
        #[doc(hidden)]
        pub(crate) use #named_impl_helper_macro_ident;
    };

    let named_impl_with_last_arm = if trait_has_last {
        quote! {
            (@named_impl_with_last { $($ok:tt)* } { $($err:tt)* }) => {
                $($ok)*
            };
        }
    } else {
        quote! {
            (@named_impl_with_last { $($ok:tt)* } { $($err:tt)* }) => {
                $($err)*
            };
        }
    };

    quote! {
        #item_fn

        #[doc(hidden)]
        #[allow(unused_macros)]
        macro_rules! #slot_ident {
            (@field) => {
                #field_ident: #field_ty,
            };
            (@field_init) => {
                #field_ident: #field_init,
            };
            (@register_methods) => {
                #[allow(dead_code)]
                pub fn #with_fn_ident(
                    mut self,
                    hook: impl #hook_trait_ident + 'static,
                ) -> Self {
                    #register_impl
                    self
                }

                #[allow(dead_code)]
                pub fn #register_fn_ident(
                    &mut self,
                    hook: impl #hook_trait_ident + 'static,
                ) -> &mut Self {
                    #register_impl
                    self
                }
            };
            (@dispatch_method) => {
                #[allow(dead_code)]
                pub async fn #fn_ident(
                    &self,
                    #ctx_ident: #ctx_ty,
                    #(#dispatch_args,)*
                ) -> #output_ty
                #clone_where {
                    use ::tracing::Instrument as _;

                    let span = ::tracing::info_span!("lutum_hook", name = #hook_name);
                    async move {
                        #args_pre_conversion
                        #dispatch
                    }
                    .instrument(span)
                    .await
                }
            };
            (@default_impl) => {
                #[allow(dead_code)]
                async fn #default_method_ident(
                    #ctx_ident: #ctx_ty,
                    #(#dispatch_args,)*
                ) -> #output_ty {
                    #default_impl_call
                }
            };
            #named_impl_with_last_arm
            (
                @accumulate
                $callback:ident
                [$($fields:tt)*]
                [$($field_inits:tt)*]
                [$($register_methods:tt)*]
                [$($dispatch_methods:tt)*]
                [$($default_impls:tt)*]
                [$($remaining:tt)*]
            ) => {
                $callback!(
                    [$($fields)* #field_ident: #field_ty,]
                    [$($field_inits)* #field_ident: #field_init,]
                    [$($register_methods)*
                        #[allow(dead_code)]
                        pub fn #with_fn_ident(
                            mut self,
                            hook: impl #hook_trait_ident + 'static,
                        ) -> Self {
                            #register_impl
                            self
                        }

                        #[allow(dead_code)]
                        pub fn #register_fn_ident(
                            &mut self,
                            hook: impl #hook_trait_ident + 'static,
                        ) -> &mut Self {
                            #register_impl
                            self
                        }
                    ]
                    [$($dispatch_methods)*
                        #[allow(dead_code)]
                        pub async fn #fn_ident(
                            &self,
                            #ctx_ident: #ctx_ty,
                            #(#dispatch_args,)*
                        ) -> #output_ty
                        #clone_where {
                            use ::tracing::Instrument as _;

                            let span = ::tracing::info_span!("lutum_hook", name = #hook_name);
                            async move {
                                #args_pre_conversion
                                #dispatch
                            }
                            .instrument(span)
                            .await
                        }
                    ]
                    [$($default_impls)*
                        #[allow(dead_code)]
                        async fn #default_method_ident(
                            #ctx_ident: #ctx_ty,
                            #(#dispatch_args,)*
                        ) -> #output_ty {
                            #default_impl_call
                        }
                    ]
                    [$($remaining)*]
                );
            };
        }

        #[doc(hidden)]
        #[allow(unused_macros)]
        macro_rules! #named_impl_helper_macro_ident {
            #named_impl_with_last_arm
        }

        #macro_reexport
        #helper_macro_reexport

        #fn_impl

        #hook_trait_defs

        #blanket_impls
    }
}
