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
        ctx_ident,
        ctx_ty,
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
    let is_lutum_hook = is_lutum_ref(&ctx_ty);
    // For the registry ext method, ref-type args need an explicit `'a` lifetime.
    let ctx_ty_with_lifetime: Type = match &ctx_ty {
        Type::Reference(r) => {
            let mut r2 = r.clone();
            r2.lifetime = Some(Lifetime::new("'a", proc_macro2::Span::call_site()));
            Type::Reference(r2)
        }
        other => other.clone(),
    };
    // The owned inner type for Fn(CtxInner, ...) blanket (strips the leading `&`).
    let ctx_inner_ty: Type = match &ctx_ty {
        Type::Reference(r) => *r.elem.clone(),
        other => other.clone(),
    };

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

    let arg_tokens = compute_hook_arg_tokens(&explicit_args, &args_field_idents, &output_ty, trait_has_last);
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
        item_fn.sig.ident.span(),
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

    let default_call = quote! {
        #default_fn_ident(
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
        HookKind::Always { chain: None } => quote! {
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
            chain: Some(chain_path),
        } => quote! {
            let mut last = ::std::option::Option::Some(#default_call);
            if #chain_path(last.as_ref().unwrap()).is_break() {
                return last.unwrap();
            }
            if let Some(hooks) = chain {
                for hook in hooks {
                    let next = #dyn_hook_dispatch_call;
                    if #chain_path(&next).is_break() {
                        return next;
                    }
                    last = ::std::option::Option::Some(next);
                }
            }
            last.unwrap()
        },
        HookKind::Fallback { chain: None } => quote! {
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
            chain: Some(chain_path),
        } => quote! {
            match chain {
                Some(hooks) if !hooks.is_empty() => {
                    let mut last = ::std::option::Option::None;
                    for hook in hooks {
                        let next = #dyn_hook_dispatch_call;
                        if #chain_path(&next).is_break() {
                            return next;
                        }
                        last = ::std::option::Option::Some(next);
                    }
                    #default_call
                }
                _ => {
                    #default_call
                }
            }
        },
        HookKind::Singleton => {
            let singleton_args: Vec<proc_macro2::TokenStream> =
                arg_tokens.dispatch_vars.iter().map(|v| quote! { #v }).collect();
            let some_call = quote! {
                hook.call_dyn(#ctx_ident, #(#singleton_args,)*).await
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
