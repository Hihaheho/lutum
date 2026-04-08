use super::*;
use heck::ToUpperCamelCase;
use quote::{format_ident, quote};
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
        explicit_args,
        output_ty,
        has_last: _,
        last_span: _,
    } = match analyze_hook_signature(
        &item_fn,
        kind.default_last_requirement(),
        "#[def_global_hook(singleton)] does not accept a `last: Option<Return>` argument",
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
    let registry_ext_ident = format_ident!("{slot_ident}RegistryExt");
    let lutum_ext_ident = format_ident!("{slot_ident}LutumExt");
    let default_fn_ident = format_ident!("__lutum_hook_default_{}", fn_ident);
    let register_fn_ident = format_ident!("register_{}", fn_ident);

    item_fn.vis = syn::Visibility::Inherited;
    item_fn.sig.ident = default_fn_ident.clone();

    let args_field_idents = normalized_hook_arg_field_idents(&explicit_args);

    // hook_call_arg_names: original param names for default_call and lutum_ext forwarding.
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

    // registry_args: parameters in the registry/lutum ext method (with 'a lifetime on refs).
    let registry_args = explicit_args
        .iter()
        .map(|(ident, ty)| {
            let ty = hook_ext_arg_type(ty);
            quote! { #ident: #ty }
        })
        .collect::<Vec<_>>();
    let context_args = registry_args.clone();

    let has_explicit_args = !explicit_args.is_empty();
    let has_ref_arg = explicit_args.iter().any(|(_, ty)| is_non_str_ref(ty));

    let arg_tokens = compute_hook_arg_tokens(
        &explicit_args,
        &args_field_idents,
        &output_ty,
        trait_has_last,
    );
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
        has_explicit_args,
        has_ref_arg,
    };

    let hook_trait_defs = generate_hook_trait_defs(
        item_fn.sig.ident.span(),
        &vis,
        &output_ty,
        &slot,
        &arg_tokens.trait_args,
    );

    let fn_impl = generate_fn_blanket_impl(&slot, &flags, &output_ty, &arg_tokens);

    let blanket_impls = generate_blanket_impls(&slot, &output_ty, &arg_tokens, &hook_name);

    let default_call = quote! {
        #default_fn_ident(
            #(#cloned_hook_call_arg_names,)*
        )
        .await
    };
    let dyn_hook_dispatch_call = if trait_has_last {
        quote! { hook.call_dyn(#(#cloned_field_idents,)* last).await }
    } else {
        quote! { hook.call_dyn(#(#cloned_field_idents,)*).await }
    };

    let register_impl = match &kind {
        HookKind::Always(..) | HookKind::Fallback(..) => quote! {
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
    let slot_lookup = match &kind {
        HookKind::Always(..) | HookKind::Fallback(..) => quote! {
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
    // Helper macro for chain check in global hook dispatch (uses default type, no companion field).
    let make_chain_check = |chain_default_ty: &syn::Path| {
        quote! {
            {
                use ::lutum::Chain as _;
                let __d: #chain_default_ty = ::std::default::Default::default();
                __d.call(&__next).is_break()
            }
        }
    };

    let inner_dispatch = match &kind {
        HookKind::Always(HookOptions {
            chain: None,
            accumulate: None,
            ..
        }) => quote! {
            let mut last = ::std::option::Option::Some(#default_call);
            if let Some(hooks) = chain {
                for hook in hooks {
                    last = ::std::option::Option::Some(#dyn_hook_dispatch_call);
                }
            }
            last.expect("hook chain unexpectedly empty")
        },
        HookKind::Always(HookOptions {
            chain: Some(chain_default_ty),
            accumulate: None,
            ..
        }) => {
            let chain_check = make_chain_check(chain_default_ty);
            quote! {
                let mut last = ::std::option::Option::Some(#default_call);
                {
                    let __next = last.as_ref().unwrap().clone();
                    if #chain_check { return __next; }
                }
                if let Some(hooks) = chain {
                    for hook in hooks {
                        let __next = #dyn_hook_dispatch_call;
                        if #chain_check { return __next; }
                        last = ::std::option::Option::Some(__next);
                    }
                }
                last.unwrap()
            }
        },
        HookKind::Always(HookOptions {
            chain: None,
            accumulate: Some(accumulate_fn),
            ..
        }) => quote! {
            let mut __outputs = ::std::vec::Vec::new();
            __outputs.push(#default_call);
            if let Some(hooks) = chain {
                for hook in hooks {
                    __outputs.push(#dyn_hook_dispatch_call);
                }
            }
            #accumulate_fn(__outputs)
        },
        HookKind::Always(HookOptions {
            chain: Some(chain_default_ty),
            accumulate: Some(accumulate_fn),
            ..
        }) => {
            let chain_check = make_chain_check(chain_default_ty);
            quote! {
                let mut __outputs = ::std::vec::Vec::new();
                let __next = #default_call;
                if #chain_check {
                    __outputs.push(__next);
                    return #accumulate_fn(__outputs);
                }
                __outputs.push(__next);
                if let Some(hooks) = chain {
                    for hook in hooks {
                        let __next = #dyn_hook_dispatch_call;
                        if #chain_check {
                            __outputs.push(__next);
                            return #accumulate_fn(__outputs);
                        }
                        __outputs.push(__next);
                    }
                }
                #accumulate_fn(__outputs)
            }
        },
        HookKind::Fallback(HookOptions {
            chain: None,
            accumulate: None,
            ..
        }) => quote! {
            match chain {
                Some(hooks) if !hooks.is_empty() => {
                    let mut last = ::std::option::Option::None;
                    for hook in hooks {
                        last = ::std::option::Option::Some(#dyn_hook_dispatch_call);
                    }
                    last.expect("hook chain unexpectedly empty")
                }
                _ => #default_call,
            }
        },
        HookKind::Fallback(HookOptions {
            chain: Some(chain_default_ty),
            accumulate: None,
            ..
        }) => {
            let chain_check = make_chain_check(chain_default_ty);
            quote! {
                match chain {
                    Some(hooks) if !hooks.is_empty() => {
                        let mut last = ::std::option::Option::None;
                        for hook in hooks {
                            let __next = #dyn_hook_dispatch_call;
                            if #chain_check { return __next; }
                            last = ::std::option::Option::Some(__next);
                        }
                        #default_call
                    }
                    _ => #default_call,
                }
            }
        },
        HookKind::Fallback(HookOptions {
            chain: None,
            accumulate: Some(accumulate_fn),
            ..
        }) => quote! {
            match chain {
                Some(hooks) if !hooks.is_empty() => {
                    let mut __outputs = ::std::vec::Vec::new();
                    for hook in hooks {
                        __outputs.push(#dyn_hook_dispatch_call);
                    }
                    #accumulate_fn(__outputs)
                }
                _ => #default_call,
            }
        },
        HookKind::Fallback(HookOptions {
            chain: Some(chain_default_ty),
            accumulate: Some(accumulate_fn),
            ..
        }) => {
            let chain_check = make_chain_check(chain_default_ty);
            quote! {
                match chain {
                    Some(hooks) if !hooks.is_empty() => {
                        let mut __outputs = ::std::vec::Vec::new();
                        for hook in hooks {
                            let __next = #dyn_hook_dispatch_call;
                            if #chain_check {
                                __outputs.push(__next);
                                return #accumulate_fn(__outputs);
                            }
                            __outputs.push(__next);
                        }
                        #accumulate_fn(__outputs)
                    }
                    _ => #default_call,
                }
            }
        },
        HookKind::Singleton => {
            let singleton_args: Vec<proc_macro2::TokenStream> = arg_tokens
                .dispatch_vars
                .iter()
                .map(|v| quote! { #v })
                .collect();
            let some_call = quote! {
                hook.call_dyn(#(#singleton_args,)*).await
            };
            quote! {
                match hook {
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

    let lutum_ext = quote! {
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
                    #(#hook_call_arg_names,)*
                )
            }
        }
    };

    let named_impl_helper_macro_ident = hook_named_impl_helper_macro_ident(&slot_ident);
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
        macro_rules! #named_impl_helper_macro_ident {
            #named_impl_with_last_arm
        }
        #[doc(hidden)]
        pub(crate) use #named_impl_helper_macro_ident;

        #fn_impl

        #hook_trait_defs

        #blanket_impls

        #[allow(dead_code)]
        #vis trait #registry_ext_ident {
            fn #register_fn_ident<H>(self, hook: H) -> Self
            where
                H: #hook_trait_ident + 'static,
                Self: Sized;

            fn #fn_ident<'a>(
                &'a self,
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
                #(#registry_args,)*
            ) -> impl ::std::future::Future<Output = #output_ty> + 'a
            #clone_where {
                async move {
                    use ::tracing::Instrument as _;

                    let span = ::tracing::info_span!("lutum_hook", name = #hook_name);
                    async move {
                        #slot_lookup
                        #args_pre_conversion
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
