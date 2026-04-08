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
    let hook_trait_ident = format_ident!("{slot_ident}Hook");
    let stateful_hook_trait_ident = format_ident!("Stateful{slot_ident}Hook");
    let dyn_hook_trait_ident = format_ident!("__LutumDyn{slot_ident}Hook");
    let args_struct_ident = format_ident!("{slot_ident}Args");
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

    let dispatch_args = explicit_args
        .iter()
        .map(|(ident, ty)| quote! { #ident: #ty })
        .collect::<Vec<_>>();
    let hook_call_arg_names = explicit_args
        .iter()
        .map(|(ident, _)| quote! { #ident })
        .collect::<Vec<_>>();
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
    let clone_bounds = explicit_args
        .iter()
        .filter(|(_, ty)| !matches!(ty, Type::Reference(_)))
        .map(|(_, ty)| {
            quote! { #ty: ::std::clone::Clone }
        })
        .collect::<Vec<_>>();

    let has_explicit_args = !explicit_args.is_empty();
    let has_ref_arg = explicit_args.iter().any(|(_, ty)| is_non_str_ref(ty));
    let args_struct_lifetime = if has_ref_arg {
        quote! { <'__lutum_args> }
    } else {
        quote! {}
    };
    let args_struct_type = if has_ref_arg {
        quote! { #args_struct_ident<'_> }
    } else {
        quote! { #args_struct_ident }
    };
    let args_field_idents = normalized_hook_arg_field_idents(&explicit_args);
    let args_struct_fields =
        explicit_args
            .iter()
            .zip(args_field_idents.iter())
            .map(|((_, ty), field_ident)| {
                let field_ty = args_field_type(ty);
                quote! { pub #field_ident: #field_ty }
            });
    let args_struct_constructor_args = explicit_args
        .iter()
        .zip(args_field_idents.iter())
        .map(|((_, ty), field_ident)| {
            let field_ty = args_field_type(ty);
            quote! { #field_ident: #field_ty }
        })
        .collect::<Vec<_>>();
    let args_struct_field_names = args_field_idents
        .iter()
        .map(|field_ident| quote! { #field_ident })
        .collect::<Vec<_>>();
    let args_struct_into_parts_types = explicit_args
        .iter()
        .map(|(_, ty)| {
            let field_ty = args_field_type(ty);
            quote! { #field_ty }
        })
        .collect::<Vec<_>>();
    let args_construction_fields = explicit_args.iter().map(|(ident, ty)| {
        let conv = args_field_conversion(ty);
        quote! { #ident #conv }
    });
    let args_construction = if has_explicit_args {
        quote! { let args = #args_struct_ident::new(#(#args_construction_fields,)*); }
    } else {
        quote! {}
    };

    let hook_trait_args_no_last: Vec<proc_macro2::TokenStream> = if has_explicit_args {
        vec![quote! { args: #args_struct_type }]
    } else {
        vec![]
    };
    let mut hook_trait_args = hook_trait_args_no_last.clone();
    let mut hook_trait_call_arg_names: Vec<proc_macro2::TokenStream> = if has_explicit_args {
        vec![quote! { args }]
    } else {
        vec![]
    };
    if trait_has_last {
        hook_trait_args.push(quote! { last: ::std::option::Option<#output_ty> });
        hook_trait_call_arg_names.push(quote! { last });
    }

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
        if has_explicit_args {
            quote! { hook.call_dyn(#ctx_ident, args.clone(), last).await }
        } else {
            quote! { hook.call_dyn(#ctx_ident, last).await }
        }
    } else if has_explicit_args {
        quote! { hook.call_dyn(#ctx_ident, args.clone()).await }
    } else {
        quote! { hook.call_dyn(#ctx_ident).await }
    };
    let dyn_hook_impl_call = if trait_has_last {
        if has_explicit_args {
            quote! { <T as #hook_trait_ident>::call(self, #ctx_ident, args, last).await }
        } else {
            quote! { <T as #hook_trait_ident>::call(self, #ctx_ident, last).await }
        }
    } else if has_explicit_args {
        quote! { <T as #hook_trait_ident>::call(self, #ctx_ident, args).await }
    } else {
        quote! { <T as #hook_trait_ident>::call(self, #ctx_ident).await }
    };
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
            let some_call = if has_explicit_args {
                quote! {
                    hook.call_dyn(#ctx_ident, args).await
                }
            } else {
                quote! {
                    hook.call_dyn(#ctx_ident).await
                }
            };
            quote! {
                match &self.#field_ident {
                    Some(hook) => #some_call,
                    None => #default_call,
                }
            }
        }
    };

    let args_struct_and_fn_impl = if has_explicit_args {
        let struct_def = quote! {
            #[allow(dead_code)]
            #[derive(::std::clone::Clone)]
            #vis struct #args_struct_ident #args_struct_lifetime {
                #(#args_struct_fields,)*
            }

            impl #args_struct_lifetime #args_struct_ident #args_struct_lifetime {
                pub fn new(#(#args_struct_constructor_args),*) -> Self {
                    Self {
                        #(#args_struct_field_names,)*
                    }
                }

                pub fn into_parts(self) -> (#(#args_struct_into_parts_types,)*) {
                    (
                        #(self.#args_struct_field_names,)*
                    )
                }
            }
        };
        let fn_impl = if has_ref_arg {
            quote! {}
        } else {
            let fn_impl_call = if trait_has_last {
                quote! { (self)(args, last).await }
            } else {
                quote! { (self)(args).await }
            };
            let fn_bound = if trait_has_last {
                quote! { F: Fn(#args_struct_ident, ::std::option::Option<#output_ty>) -> __Fut + Send + Sync }
            } else {
                quote! { F: Fn(#args_struct_ident) -> __Fut + Send + Sync }
            };
            let ctx_fn_blanket = if is_lutum_hook {
                let ctx_fn_bound = if trait_has_last {
                    quote! { F: Fn(#ctx_inner_ty, #args_struct_ident, ::std::option::Option<#output_ty>) -> __Fut + Send + Sync }
                } else {
                    quote! { F: Fn(#ctx_inner_ty, #args_struct_ident) -> __Fut + Send + Sync }
                };
                let ctx_fn_call = if trait_has_last {
                    quote! { (self.1)(#ctx_ident.clone(), args, last).await }
                } else {
                    quote! { (self.1)(#ctx_ident.clone(), args).await }
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
        };
        quote! { #struct_def #fn_impl }
    } else {
        quote! {}
    };
    let macro_reexport = quote! {};
    let named_impl_helper_macro_ident = hook_named_impl_helper_macro_ident(&slot_ident);
    let helper_macro_reexport = quote! {
        #[doc(hidden)]
        pub(crate) use #named_impl_helper_macro_ident;
    };

    let slot_dispatch_metadata = kind.dispatch_metadata_tokens();
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
                        #args_construction
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
                                #args_construction
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

        #[allow(dead_code)]
        #vis struct #slot_ident;

        #[doc(hidden)]
        impl #slot_ident {
            #slot_dispatch_metadata
        }

        #args_struct_and_fn_impl

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
