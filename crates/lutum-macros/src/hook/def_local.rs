use super::*;
use heck::ToUpperCamelCase;
use quote::{format_ident, quote, quote_spanned};
use syn::{ItemFn, Type};

pub fn expand_local_hook(mut item_fn: ItemFn, kind: HookKind) -> proc_macro2::TokenStream {
    let HookSignature {
        ctx_ident,
        ctx_ty,
        explicit_args,
        output_ty,
        has_last: default_has_last,
        last_span: _,
    } = match analyze_hook_signature(
        &item_fn,
        kind.default_last_requirement(),
        if kind.is_chain() {
            "chain-dispatch hook definitions do not accept a `last: Option<Return>` argument"
        } else {
            "#[def_hook(singleton)] does not accept a `last: Option<Return>` argument"
        },
        HookLastRecognition::LastNamedCompatibleOption,
    ) {
        Ok(signature) => signature,
        Err(err) => return err.to_compile_error(),
    };
    let trait_has_last = kind.trait_has_last();
    if !trait_has_last && default_has_last {
        return syn::Error::new_spanned(
            item_fn.sig.ident,
            "hook function must not have a 'last' parameter for chain dispatch",
        )
        .to_compile_error();
    }

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

    item_fn.vis = syn::Visibility::Inherited;
    item_fn.sig.ident = default_impl_fn_ident.clone();
    item_fn.attrs.push(syn::parse_quote!(#[allow(dead_code)]));

    // Field idents: original param names with leading `_` stripped where unambiguous.
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

    // clone_bounds: `T: Clone` bounds needed when dispatch clones owned args.
    let clone_bounds = explicit_args
        .iter()
        .filter(|(_, ty)| !matches!(ty, Type::Reference(_)))
        .map(|(_, ty)| quote! { #ty: ::std::clone::Clone })
        .collect::<Vec<_>>();

    let has_explicit_args = !explicit_args.is_empty();
    // has_ref_arg: true when any arg is &T for non-str T (requires HRTB for closures).
    let has_ref_arg = explicit_args.iter().any(|(_, ty)| is_non_str_ref(ty));

    // Hook trait method params: individual field_ident: hook_param_type (no args struct).
    let hook_trait_args_no_last: Vec<proc_macro2::TokenStream> = explicit_args
        .iter()
        .zip(args_field_idents.iter())
        .map(|((_, ty), fi)| {
            let param_ty = hook_param_type(ty);
            quote! { #fi: #param_ty }
        })
        .collect();

    let mut hook_trait_args = hook_trait_args_no_last.clone();
    let mut hook_trait_call_arg_names: Vec<proc_macro2::TokenStream> =
        args_field_idents.iter().map(|fi| quote! { #fi }).collect();
    if trait_has_last {
        hook_trait_args.push(quote! { last: ::std::option::Option<#output_ty> });
        hook_trait_call_arg_names.push(quote! { last });
    }

    // dispatch_vars: variable names used INSIDE the dispatch body for each arg.
    // For &str args: use __lutum_<fi> to avoid shadowing the original &str before default_call.
    // For everything else: same as field ident.
    let dispatch_vars: Vec<proc_macro2::Ident> = explicit_args
        .iter()
        .zip(args_field_idents.iter())
        .map(|((_, ty), fi)| {
            if is_str_ref(ty) {
                format_ident!("__lutum_{}", fi)
            } else {
                fi.clone()
            }
        })
        .collect();

    // cloned_field_idents: pass to each hook in multi-dispatch loops (uses dispatch_vars).
    // Only non-str refs stay as references; &str (→ String) and owned types need .clone().
    let cloned_field_idents: Vec<proc_macro2::TokenStream> = explicit_args
        .iter()
        .zip(dispatch_vars.iter())
        .map(|((_, ty), var)| {
            if is_non_str_ref(ty) {
                quote! { #var }  // still a &T reference, pass as-is
            } else {
                quote! { #var.clone() }  // String (from &str) or owned: clone per hook
            }
        })
        .collect();

    // pre_conversion: converts dispatch-method args to hook-trait types, placed AFTER default_call.
    // &str → String (stored in __lutum_<fi>); rebind if field ident differs from orig param name.
    let pre_conversion: Vec<proc_macro2::TokenStream> = explicit_args
        .iter()
        .zip(args_field_idents.iter())
        .zip(dispatch_vars.iter())
        .map(|(((orig_ident, ty), fi), var)| {
            if is_str_ref(ty) {
                quote! { let #var = #orig_ident.to_owned(); }
            } else if orig_ident != fi {
                quote! { let #fi = #orig_ident; }
            } else {
                quote! {}
            }
        })
        .collect();
    let args_pre_conversion = quote! { #(#pre_conversion)* };

    let hook_trait_method_sig = quote_spanned! { item_fn.sig.ident.span() =>
        async fn call(
            &self,
            #ctx_ident: #ctx_ty,
            #(#hook_trait_args,)*
        ) -> #output_ty;
    };
    let stateful_hook_trait_method_sig = quote_spanned! { item_fn.sig.ident.span() =>
        async fn call_mut(
            &mut self,
            #ctx_ident: #ctx_ty,
            #(#hook_trait_args,)*
        ) -> #output_ty;
    };
    let dyn_hook_trait_method_sig = quote_spanned! { item_fn.sig.ident.span() =>
        async fn call_dyn(
            &self,
            #ctx_ident: #ctx_ty,
            #(#hook_trait_args,)*
        ) -> #output_ty;
    };
    let hook_trait_def = quote_spanned! { item_fn.sig.ident.span() =>
        #[allow(dead_code)]
        #[::async_trait::async_trait]
        #vis trait #hook_trait_ident: Send + Sync {
            #hook_trait_method_sig
        }
    };
    let stateful_hook_trait_def = quote_spanned! { item_fn.sig.ident.span() =>
        #[allow(dead_code)]
        #[::async_trait::async_trait]
        #vis trait #stateful_hook_trait_ident: Send {
            fn on_reentrancy(err: ::lutum_protocol::hooks::HookReentrancyError) -> #output_ty {
                panic!("stateful hook reentered: {err}");
            }

            #stateful_hook_trait_method_sig
        }
    };
    let dyn_hook_trait_def = quote_spanned! { item_fn.sig.ident.span() =>
        #[allow(dead_code)]
        #[::async_trait::async_trait]
        trait #dyn_hook_trait_ident: Send + Sync {
            #dyn_hook_trait_method_sig
        }
    };

    let default_impl_call = if default_has_last {
        quote! {
            #default_impl_fn_ident(
                #ctx_ident,
                #(#hook_call_arg_names,)*
                ::std::option::Option::None,
            )
            .await
        }
    } else {
        quote! {
            #default_impl_fn_ident(
                #ctx_ident,
                #(#hook_call_arg_names,)*
            )
            .await
        }
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
    let dyn_hook_impl_call =
        quote! { <T as #hook_trait_ident>::call(self, #ctx_ident, #(#hook_trait_call_arg_names,)*).await };
    let stateful_hook_impl_call = quote! {
        <T as #stateful_hook_trait_ident>::call_mut(
            &mut *hook,
            #ctx_ident,
            #(#hook_trait_call_arg_names,)*
        )
        .await
    };
    let clone_where = if clone_bounds.is_empty() {
        quote! {}
    } else {
        quote! {
            where
                #(#clone_bounds,)*
        }
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
    let dispatch = match &kind {
        HookKind::Always {
            dispatch: HookDispatch::Fold,
        } => quote! {
            let mut last = ::std::option::Option::Some(
                #default_call,
            );
            for hook in &self.#field_ident {
                last = ::std::option::Option::Some(
                    #dyn_hook_dispatch_call,
                );
            }
            last.expect("hook chain unexpectedly empty")
        },
        HookKind::Always {
            dispatch: HookDispatch::Chain(chain_path),
        } => quote! {
            let mut result = #default_call;
            if #chain_path(&result).is_break() {
                return result;
            }
            for hook in &self.#field_ident {
                result = #dyn_hook_dispatch_call;
                if #chain_path(&result).is_break() {
                    return result;
                }
            }
            result
        },
        HookKind::Fallback {
            dispatch: HookDispatch::Fold,
        } => quote! {
            if self.#field_ident.is_empty() {
                #default_call
            } else {
                let mut last = ::std::option::Option::None;
                for hook in &self.#field_ident {
                    last = ::std::option::Option::Some(
                        #dyn_hook_dispatch_call,
                    );
                }
                last.expect("hook chain unexpectedly empty")
            }
        },
        HookKind::Fallback {
            dispatch: HookDispatch::Chain(chain_path),
        } => quote! {
            if self.#field_ident.is_empty() {
                #default_call
            } else {
                for hook in &self.#field_ident {
                    let result = #dyn_hook_dispatch_call;
                    if #chain_path(&result).is_break() {
                        return result;
                    }
                }
                #default_call
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

    // Closure blanket impl (only when no non-str ref args, to avoid HRTB complexity).
    let fn_impl = if has_explicit_args && !has_ref_arg {
        let field_ident_names: Vec<proc_macro2::TokenStream> =
            args_field_idents.iter().map(|fi| quote! { #fi }).collect();
        let fn_bound_types: Vec<proc_macro2::TokenStream> =
            explicit_args.iter().map(|(_, ty)| hook_param_type(ty)).collect();

        let fn_impl_call = if trait_has_last {
            quote! { (self)(#(#field_ident_names,)* last).await }
        } else {
            quote! { (self)(#(#field_ident_names,)*).await }
        };
        let fn_bound = if trait_has_last {
            quote! { F: Fn(#(#fn_bound_types,)* ::std::option::Option<#output_ty>) -> __Fut + Send + Sync }
        } else {
            quote! { F: Fn(#(#fn_bound_types,)*) -> __Fut + Send + Sync }
        };

        let ctx_fn_blanket = if is_lutum_hook {
            let ctx_fn_bound = if trait_has_last {
                quote! { F: Fn(#ctx_inner_ty, #(#fn_bound_types,)* ::std::option::Option<#output_ty>) -> __Fut + Send + Sync }
            } else {
                quote! { F: Fn(#ctx_inner_ty, #(#fn_bound_types,)*) -> __Fut + Send + Sync }
            };
            let ctx_fn_call = if trait_has_last {
                quote! { (self.1)(#ctx_ident.clone(), #(#field_ident_names,)* last).await }
            } else {
                quote! { (self.1)(#ctx_ident.clone(), #(#field_ident_names,)*).await }
            };
            quote! {
                #[allow(dead_code)]
                #[::async_trait::async_trait]
                impl<F, __Fut> #hook_trait_ident
                    for (::std::marker::PhantomData<fn() -> #ctx_inner_ty>, F)
                where
                    #ctx_fn_bound,
                    __Fut: ::std::future::Future<Output = #output_ty> + Send + 'static,
                {
                    async fn call(
                        &self,
                        #ctx_ident: #ctx_ty,
                        #(#hook_trait_args,)*
                    ) -> #output_ty {
                        #ctx_fn_call
                    }
                }
            }
        } else {
            quote! {}
        };

        quote! {
            #[allow(dead_code)]
            #[::async_trait::async_trait]
            impl<F, __Fut> #hook_trait_ident for F
            where
                #fn_bound,
                __Fut: ::std::future::Future<Output = #output_ty> + Send + 'static,
            {
                async fn call(
                    &self,
                    _: #ctx_ty,
                    #(#hook_trait_args,)*
                ) -> #output_ty {
                    #fn_impl_call
                }
            }

            #ctx_fn_blanket
        }
    } else {
        quote! {}
    };

    let macro_reexport = quote! {};
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
                [$($remaining:ident),*]
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
                    [$($remaining),*]
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

        #hook_trait_def

        #stateful_hook_trait_def

        #dyn_hook_trait_def

        #[::async_trait::async_trait]
        impl<T> #dyn_hook_trait_ident for T
        where
            T: #hook_trait_ident,
        {
            async fn call_dyn(
                &self,
                #ctx_ident: #ctx_ty,
                #(#hook_trait_args,)*
            ) -> #output_ty {
                #dyn_hook_impl_call
            }
        }

        #[allow(dead_code)]
        #[::async_trait::async_trait]
        impl<T> #hook_trait_ident for ::lutum_protocol::hooks::Stateful<T>
        where
            T: #stateful_hook_trait_ident + 'static,
        {
            async fn call(
                &self,
                #ctx_ident: #ctx_ty,
                #(#hook_trait_args,)*
            ) -> #output_ty {
                let Some(mut hook) = self.try_lock() else {
                    return <T as #stateful_hook_trait_ident>::on_reentrancy(
                        ::lutum_protocol::hooks::HookReentrancyError {
                            slot: #hook_name,
                            hook_type: ::std::any::type_name::<T>(),
                        },
                    );
                };

                #stateful_hook_impl_call
            }
        }
    }
}
