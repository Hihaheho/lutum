use super::*;
use heck::ToUpperCamelCase;
use quote::{format_ident, quote, quote_spanned};
use syn::{ItemFn, Lifetime, Type};

fn hook_ext_arg_type(ty: &Type) -> Type {
    match ty {
        Type::Reference(reference) => {
            let mut reference = reference.clone();
            reference.lifetime = Some(Lifetime::new("'a", proc_macro2::Span::call_site()));
            Type::Reference(reference)
        }
        _ => ty.clone(),
    }
}

pub fn expand_global_hook(mut item_fn: ItemFn, kind: HookKind) -> proc_macro2::TokenStream {
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
            "#[def_global_hook(singleton)] does not accept a `last: Option<Return>` argument"
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
    let registry_ext_ident = format_ident!("{slot_ident}RegistryExt");
    let lutum_ext_ident = format_ident!("{slot_ident}LutumExt");
    let args_struct_ident = format_ident!("{slot_ident}Args");
    let default_fn_ident = format_ident!("__lutum_hook_default_{}", fn_ident);
    let register_fn_ident = format_ident!("register_{}", fn_ident);
    let is_lutum_hook = is_lutum_ref(&ctx_ty);
    // For the registry ext method, the first-arg type needs a lifetime annotation if it's a reference
    let ctx_ty_with_lifetime: Type = match &ctx_ty {
        Type::Reference(r) => {
            let mut r2 = r.clone();
            r2.lifetime = Some(Lifetime::new("'a", proc_macro2::Span::call_site()));
            Type::Reference(r2)
        }
        other => other.clone(),
    };
    // The owned inner type for Fn(CtxInner, XxxArgs) blanket (strips the leading `&`).
    let ctx_inner_ty: Type = match &ctx_ty {
        Type::Reference(r) => *r.elem.clone(),
        other => other.clone(),
    };

    item_fn.vis = syn::Visibility::Inherited;
    item_fn.sig.ident = default_fn_ident.clone();

    // Individual arg signatures (for the default fn call and registry method signature).
    let _hook_call_args = explicit_args
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
    let registry_args = explicit_args
        .iter()
        .map(|(ident, ty)| {
            let ty = hook_ext_arg_type(ty);
            quote! { #ident: #ty }
        })
        .collect::<Vec<_>>();
    let context_args = registry_args.clone();
    let clone_bounds = explicit_args
        .iter()
        .filter(|(_, ty)| !matches!(ty, Type::Reference(_)))
        .map(|(_, ty)| {
            let ty = hook_ext_arg_type(ty);
            quote! { #ty: ::std::clone::Clone }
        })
        .collect::<Vec<_>>();

    // XxxArgs struct fields and construction.
    let has_explicit_args = !explicit_args.is_empty();
    // Whether any arg is a non-str reference: those become &'__lutum_args T fields,
    // so the generated struct needs a lifetime parameter.
    let has_ref_arg = explicit_args.iter().any(|(_, ty)| is_non_str_ref(ty));
    let args_struct_lifetime = if has_ref_arg {
        quote! { <'__lutum_args> }
    } else {
        quote! {}
    };
    // Type to use in trait method signatures: `XxxArgs<'_>` when struct has lifetime.
    let args_struct_type = if has_ref_arg {
        quote! { #args_struct_ident<'_> }
    } else {
        quote! { #args_struct_ident }
    };
    let args_field_idents = normalized_hook_arg_field_idents(&explicit_args);
    // Named XxxArgs fields preserve the slot-definition argument names (after trimming a
    // leading `_` when it does not cause conflicts), while the generated constructor and
    // `into_parts()` keep #[hook] impls working even if their parameter names differ.
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
    // Construct XxxArgs from individual registry method parameters.
    // &str → .to_owned(), &T → copy ref (no conversion), owned T → .clone() so
    // the original is still available for default_call.
    let args_construction_fields = explicit_args.iter().map(|(ident, ty)| {
        let conv = args_field_conversion(ty);
        quote! { #ident #conv }
    });
    let args_construction = if has_explicit_args {
        quote! { let args = #args_struct_ident::new(#(#args_construction_fields,)*); }
    } else {
        quote! {}
    };

    // Trait method signatures: XxxHook and StatefulXxxHook take `args: XxxArgs` instead
    // of individual parameters.
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

    let default_call = if default_has_last {
        quote! {
            #default_fn_ident(
                #ctx_ident,
                #(#cloned_hook_call_arg_names,)*
                None,
            )
            .await
        }
    } else {
        quote! {
            #default_fn_ident(
                #ctx_ident,
                #(#cloned_hook_call_arg_names,)*
            )
            .await
        }
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
    let register_impl = match kind {
        HookKind::Always { .. } | HookKind::Fallback { .. } => quote! {
            let slot = self
                .slots_mut()
                .entry(::std::any::TypeId::of::<::std::boxed::Box<dyn #slot_ident>>())
                .or_insert_with(|| {
                    ::std::boxed::Box::new(
                        ::std::vec::Vec::<::std::sync::Arc<dyn #dyn_hook_trait_ident>>::new(),
                    ) as ::std::boxed::Box<dyn ::std::any::Any + Send + Sync>
                });
            slot.downcast_mut::<::std::vec::Vec<::std::sync::Arc<dyn #dyn_hook_trait_ident>>>()
                .expect("hook slot type mismatch")
                .push(::std::sync::Arc::new(hook));
        },
        HookKind::Singleton => quote! {
            let hook = ::std::sync::Arc::new(hook) as ::std::sync::Arc<dyn #dyn_hook_trait_ident>;
            match self
                .slots_mut()
                .entry(::std::any::TypeId::of::<::std::boxed::Box<dyn #slot_ident>>())
            {
                ::std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(
                        ::std::boxed::Box::new(hook)
                            as ::std::boxed::Box<dyn ::std::any::Any + Send + Sync>,
                    );
                }
                ::std::collections::hash_map::Entry::Occupied(mut entry) => {
                    let slot = entry
                        .get_mut()
                        .downcast_mut::<::std::sync::Arc<dyn #dyn_hook_trait_ident>>()
                        .expect("hook slot type mismatch");
                    ::tracing::warn!(
                        slot = #hook_name,
                        "singleton hook registration overwritten; last registered hook wins"
                    );
                    *slot = hook;
                }
            }
        },
    };
    let slot_lookup = match kind {
        HookKind::Always { .. } | HookKind::Fallback { .. } => quote! {
            let chain = self
                .slots()
                .get(&::std::any::TypeId::of::<::std::boxed::Box<dyn #slot_ident>>())
                .and_then(|slot| {
                    slot.downcast_ref::<
                        ::std::vec::Vec<::std::sync::Arc<dyn #dyn_hook_trait_ident>>,
                    >()
                });
        },
        HookKind::Singleton => quote! {
            let hook = self
                .slots()
                .get(&::std::any::TypeId::of::<::std::boxed::Box<dyn #slot_ident>>())
                .and_then(|slot| {
                    slot.downcast_ref::<
                        ::std::sync::Arc<dyn #dyn_hook_trait_ident>,
                    >()
                });
        },
    };
    let dispatch = match &kind {
        HookKind::Always {
            dispatch: HookDispatch::Fold,
        } => quote! {
            let mut last = ::std::option::Option::Some(
                #default_call,
            );
            if let Some(hooks) = chain {
                for hook in hooks {
                    last = ::std::option::Option::Some(
                        #dyn_hook_dispatch_call,
                    );
                }
            }
            last.expect("hook chain unexpectedly empty")
        },
        HookKind::Always {
            dispatch: HookDispatch::Chain(chain_path),
        } => quote! {
            let result = #default_call;
            if #chain_path(&result).is_break() {
                return result;
            }
            if let Some(hooks) = chain {
                let mut result = result;
                for hook in hooks {
                    result = #dyn_hook_dispatch_call;
                    if #chain_path(&result).is_break() {
                        return result;
                    }
                }
                result
            } else {
                result
            }
        },
        HookKind::Fallback {
            dispatch: HookDispatch::Fold,
        } => quote! {
            match chain {
                Some(hooks) if !hooks.is_empty() => {
                    let mut last = ::std::option::Option::None;
                    for hook in hooks {
                        last = ::std::option::Option::Some(
                            #dyn_hook_dispatch_call,
                        );
                    }
                    last.expect("hook chain unexpectedly empty")
                }
                _ => {
                    #default_call
                }
            }
        },
        HookKind::Fallback {
            dispatch: HookDispatch::Chain(chain_path),
        } => quote! {
            match chain {
                Some(hooks) if !hooks.is_empty() => {
                    for hook in hooks {
                        let result = #dyn_hook_dispatch_call;
                        if #chain_path(&result).is_break() {
                            return result;
                        }
                    }
                    #default_call
                }
                _ => {
                    #default_call
                }
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
                match hook {
                    Some(hook) => #some_call,
                    None => #default_call,
                }
            }
        }
    };

    let lutum_ext = if is_lutum_hook {
        quote! {
            #[allow(dead_code)]
            #vis trait #lutum_ext_ident {
                fn #fn_ident<'a>(
                    &'a self,
                    #(#context_args,)*
                ) -> impl ::std::future::Future<Output = #output_ty> + 'a
                #clone_where;
            }

            impl #lutum_ext_ident for ::lutum::Lutum {
                fn #fn_ident<'a>(
                    &'a self,
                    #(#context_args,)*
                ) -> impl ::std::future::Future<Output = #output_ty> + 'a
                #clone_where {
                    <::lutum_protocol::HookRegistry as #registry_ext_ident>::#fn_ident(
                        self.hooks(),
                        self,
                        #(#hook_call_arg_names,)*
                    )
                }
            }
        }
    } else {
        quote! {}
    };

    // Generate XxxArgs struct and (when no reference args) a Fn blanket impl.
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
        // Fn blanket impl only for structs without lifetime (fully owned args).
        // Structs with reference lifetimes require HRTB which is too complex to generate,
        // and those hooks are unlikely to be registered via closures anyway.
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
            // Second blanket: Fn(CtxInner, XxxArgs) -> Fut — lets callers receive an owned
            // clone of the context alongside args (useful for probe dispatch closures).
            // Uses a (PhantomData<fn() -> CtxInner>, F) newtype to avoid conflicting with
            // the F: Fn(XxxArgs) blanket above (different Self types, no overlap).
            // Only generated for Lutum-ctx hooks because ctx.clone() requires CtxInner: Clone,
            // and RequestExtensions does not implement Clone.
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
                    impl<F, __Fut> #hook_trait_ident for (::std::marker::PhantomData<fn() -> #ctx_inner_ty>, F)
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

    quote! {
        #item_fn

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

        #[allow(dead_code)]
        #vis trait #registry_ext_ident {
            fn #register_fn_ident<H>(self, hook: H) -> Self
            where
                H: #hook_trait_ident + 'static,
                Self: Sized;

            fn #fn_ident<'a>(
                &'a self,
                #ctx_ident: #ctx_ty_with_lifetime,
                #(#registry_args,)*
            ) -> impl ::std::future::Future<Output = #output_ty> + 'a
            #clone_where;
        }

        impl #registry_ext_ident for ::lutum_protocol::HookRegistry {
            fn #register_fn_ident<H>(mut self, hook: H) -> Self
            where
                H: #hook_trait_ident + 'static,
                Self: Sized,
            {
                #register_impl
                self
            }

            fn #fn_ident<'a>(
                &'a self,
                #ctx_ident: #ctx_ty_with_lifetime,
                #(#registry_args,)*
            ) -> impl ::std::future::Future<Output = #output_ty> + 'a
            #clone_where {
                async move {
                    use ::tracing::Instrument as _;

                    let span = ::tracing::info_span!("lutum_hook", name = #hook_name);
                    async move {
                        #slot_lookup
                        #args_construction
                        #dispatch
                    }
                    .instrument(span)
                    .await
                }
            }
        }

        #lutum_ext
    }
}
