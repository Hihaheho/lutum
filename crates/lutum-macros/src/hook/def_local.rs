use super::*;
use heck::ToUpperCamelCase;
use quote::{format_ident, quote};
use syn::{ItemFn, Visibility};

pub struct SlotExpansion {
    pub items: proc_macro2::TokenStream,
    pub field: proc_macro2::TokenStream,
    pub field_init: proc_macro2::TokenStream,
    pub register_methods: proc_macro2::TokenStream,
    pub dispatch_method: proc_macro2::TokenStream,
    pub default_method: proc_macro2::TokenStream,
}

pub fn expand_slot(
    mut item_fn: ItemFn,
    kind: HookKind,
    vis: &Visibility,
) -> syn::Result<SlotExpansion> {
    let HookSignature {
        explicit_args,
        output_ty: hook_output_ty,
        has_last: _,
        generics,
    } = analyze_hook_signature(
        &item_fn,
        HookLastRequirement::Forbidden,
        "#[hooks] slot definitions must not declare `last: Option<Return>`",
        HookLastRecognition::LastNamedCompatibleOption,
        false,
    )?;
    let trait_has_last = kind.trait_has_last();

    let fn_ident = item_fn.sig.ident.clone();
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
    let chain_field_ident = format_ident!("{}_chain", fn_ident);
    let with_chain_fn_ident = format_ident!("with_{}_chain", fn_ident);
    let set_chain_fn_ident = format_ident!("set_{}_chain", fn_ident);
    let aggregate_field_ident = format_ident!("{}_aggregate", fn_ident);
    let with_aggregate_fn_ident = format_ident!("with_{}_aggregate", fn_ident);
    let set_aggregate_fn_ident = format_ident!("set_{}_aggregate", fn_ident);
    let finalize_field_ident = format_ident!("{}_finalize", fn_ident);
    let with_finalize_fn_ident = format_ident!("with_{}_finalize", fn_ident);
    let set_finalize_fn_ident = format_ident!("set_{}_finalize", fn_ident);

    item_fn.vis = Visibility::Inherited;
    item_fn.sig.ident = default_impl_fn_ident.clone();
    item_fn.attrs.clear();
    item_fn.attrs.push(syn::parse_quote!(#[allow(dead_code)]));

    let args_field_idents = normalized_hook_arg_field_idents(&explicit_args);

    let dispatch_args = explicit_args
        .iter()
        .map(|(ident, ty)| quote! { #ident: #ty })
        .collect::<Vec<_>>();
    let hook_call_arg_names = explicit_args
        .iter()
        .map(|(ident, _)| quote! { #ident })
        .collect::<Vec<_>>();
    let has_explicit_args = !explicit_args.is_empty();
    let has_ref_arg = explicit_args.iter().any(|(_, ty)| is_non_str_ref(ty));
    let dispatch_output_ty = dispatch_output_type(&kind, &hook_output_ty);
    let has_output_override = kind.opts().and_then(|opts| opts.output.as_ref()).is_some();

    let def_span = item_fn.sig.ident.span();
    let arg_tokens = compute_hook_arg_tokens(
        &explicit_args,
        &args_field_idents,
        &hook_output_ty,
        trait_has_last,
    );
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
        has_explicit_args,
        has_ref_arg,
    };

    let hook_trait_defs = generate_hook_trait_defs(
        def_span,
        vis,
        &hook_output_ty,
        &slot,
        &arg_tokens.trait_args,
        &generics,
    );
    let fn_impl = generate_fn_blanket_impl(&slot, &flags, &hook_output_ty, &arg_tokens, &generics);
    let blanket_impls =
        generate_blanket_impls(&slot, &hook_output_ty, &arg_tokens, &generics, &hook_name);

    let default_impl_call = quote! {
        #default_impl_fn_ident(
            #(#hook_call_arg_names,)*
        )
        .await
    };
    // For default_call we call `default_method`, which takes the ORIGINAL parameter types.
    // References (&T, &str) can be passed from the original ident (they are Copy / coerce fine).
    // Owned types must use the normalized dispatch_var (which was `let fi = orig_ident;` after
    // pre_conversion), because the original ident may have been moved.
    let default_call_arg_tokens: Vec<proc_macro2::TokenStream> = explicit_args
        .iter()
        .zip(dispatch_vars.iter())
        .map(|((orig_ident, ty), var)| {
            if matches!(ty, syn::Type::Reference(_)) {
                quote! { #orig_ident }
            } else {
                quote! { #var.clone() }
            }
        })
        .collect();
    let default_call = quote! {
        Self::#default_method_ident(
            #(#default_call_arg_tokens,)*
        )
        .await
    };
    let dyn_hook_dispatch_call = if trait_has_last {
        quote! { hook.call_dyn(#(#cloned_field_idents,)* last).await }
    } else {
        quote! { hook.call_dyn(#(#cloned_field_idents,)*).await }
    };
    let aggregate_trait = if has_output_override {
        quote! { ::lutum::AggregateInto<#hook_output_ty, #dispatch_output_ty> }
    } else {
        quote! { ::lutum::Aggregate<#hook_output_ty> }
    };
    let aggregate_trait_import = if has_output_override {
        quote! { use ::lutum::AggregateInto as _; }
    } else {
        quote! { use ::lutum::Aggregate as _; }
    };
    let finalize_trait = if has_output_override {
        quote! { ::lutum::FinalizeInto<#hook_output_ty, #dispatch_output_ty> }
    } else {
        quote! { ::lutum::Finalize<#hook_output_ty> }
    };
    let finalize_trait_import = if has_output_override {
        quote! { use ::lutum::FinalizeInto as _; }
    } else {
        quote! { use ::lutum::Finalize as _; }
    };

    let chain_companion_tokens =
        kind.opts()
            .and_then(|o| o.chain.as_ref())
            .map(|chain_default_ty| {
                let chain_field_ty = quote! {
                    ::std::option::Option<
                        ::std::sync::Arc<dyn ::lutum::Chain<#hook_output_ty> + Send + Sync>
                    >
                };
                let chain_field_init = quote! { ::std::option::Option::None };
                let chain_check = quote! {
                    if {
                        use ::lutum::Chain as _;
                        let __cf = match &self.#chain_field_ident {
                            ::std::option::Option::Some(__h) => (__h).call(&__next).await,
                            ::std::option::Option::None => {
                                let __d: #chain_default_ty = ::std::default::Default::default();
                                __d.call(&__next).await
                            }
                        };
                        __cf.is_break()
                    }
                };
                (chain_field_ty, chain_field_init, chain_check)
            });
    let aggregate_companion_tokens =
        kind.opts()
            .and_then(|o| o.aggregate.as_ref())
            .map(|aggregate_default_ty| {
                let aggregate_field_ty = quote! {
                    ::std::option::Option<
                        ::std::sync::Arc<dyn #aggregate_trait + Send + Sync>
                    >
                };
                let aggregate_field_init = quote! { ::std::option::Option::None };
                let aggregate_call = quote! {
                    {
                        #aggregate_trait_import
                        match &self.#aggregate_field_ident {
                            ::std::option::Option::Some(__h) => __h.call(__outputs).await,
                            ::std::option::Option::None => {
                                let __a: #aggregate_default_ty =
                                    ::std::default::Default::default();
                                __a.call(__outputs).await
                            }
                        }
                    }
                };
                (aggregate_field_ty, aggregate_field_init, aggregate_call)
            });
    let finalize_companion_tokens =
        kind.opts()
            .and_then(|o| o.finalize.as_ref())
            .map(|finalize_default_ty| {
                let finalize_field_ty = quote! {
                    ::std::option::Option<
                        ::std::sync::Arc<dyn #finalize_trait + Send + Sync>
                    >
                };
                let finalize_field_init = quote! { ::std::option::Option::None };
                let finalize_call = quote! {
                    {
                        #finalize_trait_import
                        match &self.#finalize_field_ident {
                            ::std::option::Option::Some(__h) => __h.call(__result).await,
                            ::std::option::Option::None => {
                                let __f: #finalize_default_ty =
                                    ::std::default::Default::default();
                                __f.call(__result).await
                            }
                        }
                    }
                };
                (finalize_field_ty, finalize_field_init, finalize_call)
            });

    let (field_ty, field_init, register_impl) = match &kind {
        HookKind::Always(_) | HookKind::Fallback(_) => (
            quote! { ::std::vec::Vec<::std::sync::Arc<dyn #dyn_hook_trait_ident>> },
            quote! { ::std::vec::Vec::new() },
            quote! {
                self.#field_ident.push(::std::sync::Arc::new(hook));
            },
        ),
        HookKind::Singleton => (
            quote! { ::std::option::Option<::std::sync::Arc<dyn #dyn_hook_trait_ident>> },
            quote! { ::std::option::Option::None },
            quote! {
                let hook = ::std::sync::Arc::new(hook)
                    as ::std::sync::Arc<dyn #dyn_hook_trait_ident>;
                if self.#field_ident.replace(hook).is_some() {
                    ::tracing::warn!(
                        target: "lutum",
                        slot = #hook_name,
                        "singleton hook registration overwritten; last registered hook wins"
                    );
                }
            },
        ),
    };

    let chain_field = chain_companion_tokens
        .as_ref()
        .map(|(field_ty, _, _)| quote! { #chain_field_ident: #field_ty, });
    let chain_field_init = chain_companion_tokens
        .as_ref()
        .map(|(_, field_init, _)| quote! { #chain_field_ident: #field_init, });
    let chain_methods = chain_companion_tokens
        .as_ref()
        .map(|_| {
            quote! {
                #[allow(dead_code)]
                pub fn #with_chain_fn_ident(
                    mut self,
                    h: impl ::lutum::Chain<#hook_output_ty> + 'static,
                ) -> Self {
                    if self.#chain_field_ident.replace(::std::sync::Arc::new(h)).is_some() {
                        ::tracing::warn!(
                            target: "lutum",
                            slot = ::std::concat!(#hook_name, ".chain"),
                            "companion chain overwritten; last registered wins"
                        );
                    }
                    self
                }

                #[allow(dead_code)]
                pub fn #set_chain_fn_ident(
                    &mut self,
                    h: impl ::lutum::Chain<#hook_output_ty> + 'static,
                ) {
                    self.#chain_field_ident =
                        ::std::option::Option::Some(::std::sync::Arc::new(h));
                }
            }
        })
        .unwrap_or_default();
    let aggregate_field = aggregate_companion_tokens
        .as_ref()
        .map(|(field_ty, _, _)| quote! { #aggregate_field_ident: #field_ty, });
    let aggregate_field_init = aggregate_companion_tokens
        .as_ref()
        .map(|(_, field_init, _)| quote! { #aggregate_field_ident: #field_init, });
    let aggregate_methods = aggregate_companion_tokens
        .as_ref()
        .map(|_| {
            quote! {
                #[allow(dead_code)]
                pub fn #with_aggregate_fn_ident(
                    mut self,
                    h: impl #aggregate_trait + 'static,
                ) -> Self {
                    if self
                        .#aggregate_field_ident
                        .replace(::std::sync::Arc::new(h))
                        .is_some()
                    {
                        ::tracing::warn!(
                            target: "lutum",
                            slot = ::std::concat!(#hook_name, ".aggregate"),
                            "companion aggregate overwritten; last registered wins"
                        );
                    }
                    self
                }

                #[allow(dead_code)]
                pub fn #set_aggregate_fn_ident(
                    &mut self,
                    h: impl #aggregate_trait + 'static,
                ) {
                    self.#aggregate_field_ident =
                        ::std::option::Option::Some(::std::sync::Arc::new(h));
                }
            }
        })
        .unwrap_or_default();
    let finalize_field = finalize_companion_tokens
        .as_ref()
        .map(|(field_ty, _, _)| quote! { #finalize_field_ident: #field_ty, });
    let finalize_field_init = finalize_companion_tokens
        .as_ref()
        .map(|(_, field_init, _)| quote! { #finalize_field_ident: #field_init, });
    let finalize_methods = finalize_companion_tokens
        .as_ref()
        .map(|_| {
            quote! {
                #[allow(dead_code)]
                pub fn #with_finalize_fn_ident(
                    mut self,
                    h: impl #finalize_trait + 'static,
                ) -> Self {
                    if self
                        .#finalize_field_ident
                        .replace(::std::sync::Arc::new(h))
                        .is_some()
                    {
                        ::tracing::warn!(
                            target: "lutum",
                            slot = ::std::concat!(#hook_name, ".finalize"),
                            "companion finalize overwritten; last registered wins"
                        );
                    }
                    self
                }

                #[allow(dead_code)]
                pub fn #set_finalize_fn_ident(
                    &mut self,
                    h: impl #finalize_trait + 'static,
                ) {
                    self.#finalize_field_ident =
                        ::std::option::Option::Some(::std::sync::Arc::new(h));
                }
            }
        })
        .unwrap_or_default();

    let inner_dispatch = match (&kind, &chain_companion_tokens, &aggregate_companion_tokens) {
        // Pipeline dispatch — custom replaces the entire loop with an owned-input threading loop.
        (HookKind::Fallback(HookOptions { custom: Some(_), .. }), _, _) => {
            // Identify the single owned (non-reference) arg and collect the reference args.
            let mut owned_var: Option<&syn::Ident> = None;
            let mut ref_vars: Vec<&syn::Ident> = Vec::new();
            for ((_, ty), var) in explicit_args.iter().zip(dispatch_vars.iter()) {
                if matches!(ty, syn::Type::Reference(_)) {
                    ref_vars.push(var);
                } else {
                    owned_var = Some(var);
                }
            }
            let owned_var = owned_var
                .expect("custom pipeline hook must have exactly one owned (non-reference) argument");
            let ref_var_tokens: Vec<proc_macro2::TokenStream> =
                ref_vars.iter().map(|v| quote! { #v }).collect();
            quote! {
                if self.#field_ident.is_empty() {
                    #default_call
                } else {
                    let mut __current = #owned_var;
                    for __hook in &self.#field_ident {
                        match __hook.call_dyn(#(#ref_var_tokens,)* __current).await {
                            ::lutum::ToolDecision::RunNormally(__next) => __current = __next,
                            __terminal => return __terminal,
                        }
                    }
                    ::lutum::ToolDecision::RunNormally(__current)
                }
            }
        }
        (
            HookKind::Always(HookOptions {
                chain: None,
                aggregate: None,
                ..
            }),
            _,
            _,
        ) => quote! {
            let mut last = ::std::option::Option::Some(#default_call);
            for hook in &self.#field_ident {
                last = ::std::option::Option::Some(#dyn_hook_dispatch_call);
            }
            last.expect("hook chain unexpectedly empty")
        },
        (
            HookKind::Always(HookOptions {
                chain: Some(_),
                aggregate: None,
                ..
            }),
            Some((_, _, chain_check)),
            _,
        ) => quote! {
            let mut last = ::std::option::Option::Some(#default_call);
            {
                let __next = last.as_ref().unwrap().clone();
                #chain_check { return __next; }
            }
            for hook in &self.#field_ident {
                let __next = #dyn_hook_dispatch_call;
                #chain_check { return __next; }
                last = ::std::option::Option::Some(__next);
            }
            last.unwrap()
        },
        (
            HookKind::Always(HookOptions {
                chain: None,
                aggregate: Some(_),
                ..
            }),
            _,
            Some((_, _, aggregate_call)),
        ) => quote! {
            let mut __outputs = ::std::vec::Vec::new();
            __outputs.push(#default_call);
            for hook in &self.#field_ident {
                __outputs.push(#dyn_hook_dispatch_call);
            }
            #aggregate_call
        },
        (
            HookKind::Always(HookOptions {
                chain: Some(_),
                aggregate: Some(_),
                ..
            }),
            Some((_, _, chain_check)),
            Some((_, _, aggregate_call)),
        ) => quote! {
            let mut __outputs = ::std::vec::Vec::new();
            let __next = #default_call;
            #chain_check {
                __outputs.push(__next);
                return #aggregate_call;
            }
            __outputs.push(__next);
            for hook in &self.#field_ident {
                let __next = #dyn_hook_dispatch_call;
                #chain_check {
                    __outputs.push(__next);
                    return #aggregate_call;
                }
                __outputs.push(__next);
            }
            #aggregate_call
        },
        (
            HookKind::Fallback(HookOptions {
                chain: None,
                aggregate: None,
                ..
            }),
            _,
            _,
        ) => quote! {
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
        (
            HookKind::Fallback(HookOptions {
                chain: Some(_),
                aggregate: None,
                ..
            }),
            Some((_, _, chain_check)),
            _,
        ) => quote! {
            if self.#field_ident.is_empty() {
                #default_call
            } else {
                for hook in &self.#field_ident {
                    let __next = #dyn_hook_dispatch_call;
                    #chain_check { return __next; }
                }
                #default_call
            }
        },
        (
            HookKind::Fallback(HookOptions {
                chain: None,
                aggregate: Some(_),
                ..
            }),
            _,
            Some((_, _, aggregate_call)),
        ) => {
            let fallback_default = if has_output_override {
                quote! {
                    let mut __outputs = ::std::vec::Vec::new();
                    __outputs.push(#default_call);
                    #aggregate_call
                }
            } else {
                quote! { #default_call }
            };
            quote! {
                if self.#field_ident.is_empty() {
                    #fallback_default
                } else {
                    let mut __outputs = ::std::vec::Vec::new();
                    for hook in &self.#field_ident {
                        __outputs.push(#dyn_hook_dispatch_call);
                    }
                    #aggregate_call
                }
            }
        }
        (
            HookKind::Fallback(HookOptions {
                chain: Some(_),
                aggregate: Some(_),
                ..
            }),
            Some((_, _, chain_check)),
            Some((_, _, aggregate_call)),
        ) => quote! {
            if self.#field_ident.is_empty() {
                let mut __outputs = ::std::vec::Vec::new();
                __outputs.push(#default_call);
                #aggregate_call
            } else {
                let mut __outputs = ::std::vec::Vec::new();
                for hook in &self.#field_ident {
                    let __next = #dyn_hook_dispatch_call;
                    #chain_check {
                        __outputs.push(__next);
                        return #aggregate_call;
                    }
                    __outputs.push(__next);
                }
                #aggregate_call
            }
        },
        (HookKind::Singleton, _, _) => {
            let singleton_args: Vec<proc_macro2::TokenStream> =
                dispatch_vars.iter().map(|v| quote! { #v }).collect();
            let some_call = quote! { hook.call_dyn(#(#singleton_args,)*).await };
            quote! {
                match &self.#field_ident {
                    ::std::option::Option::Some(hook) => #some_call,
                    ::std::option::Option::None => #default_call,
                }
            }
        }
        _ => unreachable!(
            "chain/aggregate companion tokens are Some iff the corresponding hook options are set"
        ),
    };

    let dispatch = match &finalize_companion_tokens {
        Some((_, _, finalize_call)) => quote! {
            let __result = async move { #inner_dispatch }.await;
            #finalize_call
        },
        None => inner_dispatch,
    };

    let items = quote! {
        #item_fn

        #fn_impl

        #hook_trait_defs

        #blanket_impls
    };

    let field = quote! {
        #field_ident: #field_ty,
        #chain_field
        #aggregate_field
        #finalize_field
    };
    let field_init = quote! {
        #field_ident: #field_init,
        #chain_field_init
        #aggregate_field_init
        #finalize_field_init
    };
    let register_methods = quote! {
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

        #chain_methods
        #aggregate_methods
        #finalize_methods
    };
    let dispatch_method = quote! {
        #[allow(dead_code)]
        pub async fn #fn_ident(
            &self,
            #(#dispatch_args,)*
        ) -> #dispatch_output_ty
        #clone_where
        {
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
    let default_method = quote! {
        #[allow(dead_code)]
        async fn #default_method_ident(
            #(#dispatch_args,)*
        ) -> #hook_output_ty {
            #default_impl_call
        }
    };

    Ok(SlotExpansion {
        items,
        field,
        field_init,
        register_methods,
        dispatch_method,
        default_method,
    })
}
