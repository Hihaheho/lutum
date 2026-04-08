use super::*;
use quote::{format_ident, quote, quote_spanned};
use syn::{Ident, Type, Visibility};

/// The three generated trait identifiers for a hook slot.
pub struct HookSlotIdents {
    pub hook_trait: Ident,
    pub stateful: Ident,
    pub dyn_trait: Ident,
}

/// Boolean properties of a hook slot derived from its definition.
pub struct HookSlotFlags {
    pub trait_has_last: bool,
    pub is_lutum_hook: bool,
    pub has_explicit_args: bool,
    pub has_ref_arg: bool,
}

/// All token streams derived from the hook's explicit argument list.
/// Computed once by [`compute_hook_arg_tokens`] and consumed by the
/// trait-definition, blanket-impl, and dispatch-method generators.
pub struct HookArgTokens {
    /// `fi: ParamType` params for hook trait methods (without `last`).
    #[allow(dead_code)]
    pub trait_args_no_last: Vec<proc_macro2::TokenStream>,
    /// Same with `last: Option<Output>` appended when `trait_has_last`.
    pub trait_args: Vec<proc_macro2::TokenStream>,
    /// Arg names for call forwarding: field idents (+ `last` when `trait_has_last`).
    pub trait_call_arg_names: Vec<proc_macro2::TokenStream>,
    /// Variable name inside the dispatch body for each arg.
    /// `&str` args use `__lutum_<fi>` to avoid shadowing the original before `default_call`.
    pub dispatch_vars: Vec<Ident>,
    /// Expression to pass each arg into a hook in a dispatch loop.
    /// Non-str refs pass as-is; everything else (including `String` from `&str`) gets `.clone()`.
    pub cloned: Vec<proc_macro2::TokenStream>,
    /// Pre-conversion statements before dispatch: `let __lutum_fi = fi.to_owned();` etc.
    pub pre_conversion: proc_macro2::TokenStream,
    /// `fi` expressions for closure `Fn(T...)` call sites.
    pub field_ident_names: Vec<proc_macro2::TokenStream>,
    /// `hook_param_type(ty)` for each arg, used in `Fn(T...)` bounds.
    pub fn_bound_types: Vec<proc_macro2::TokenStream>,
    /// `where T: Clone` clause for owned args; empty `TokenStream` when all args are refs.
    pub clone_where: proc_macro2::TokenStream,
}

/// Derives all argument-related token streams from the hook's parameter list.
pub fn compute_hook_arg_tokens(
    explicit_args: &[(Ident, Type)],
    args_field_idents: &[Ident],
    output_ty: &Type,
    trait_has_last: bool,
) -> HookArgTokens {
    let trait_args_no_last: Vec<proc_macro2::TokenStream> = explicit_args
        .iter()
        .zip(args_field_idents.iter())
        .map(|((_, ty), fi)| {
            let param_ty = hook_param_type(ty);
            quote! { #fi: #param_ty }
        })
        .collect();

    let mut trait_args = trait_args_no_last.clone();
    let mut trait_call_arg_names: Vec<proc_macro2::TokenStream> =
        args_field_idents.iter().map(|fi| quote! { #fi }).collect();
    if trait_has_last {
        trait_args.push(quote! { last: ::std::option::Option<#output_ty> });
        trait_call_arg_names.push(quote! { last });
    }

    let dispatch_vars: Vec<Ident> = explicit_args
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

    let cloned: Vec<proc_macro2::TokenStream> = explicit_args
        .iter()
        .zip(dispatch_vars.iter())
        .map(|((_, ty), var)| {
            if is_non_str_ref(ty) {
                quote! { #var }
            } else {
                quote! { #var.clone() }
            }
        })
        .collect();

    let pre_conversion_parts: Vec<proc_macro2::TokenStream> = explicit_args
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
    let pre_conversion = quote! { #(#pre_conversion_parts)* };

    let field_ident_names: Vec<proc_macro2::TokenStream> =
        args_field_idents.iter().map(|fi| quote! { #fi }).collect();
    let fn_bound_types: Vec<proc_macro2::TokenStream> =
        explicit_args.iter().map(|(_, ty)| hook_param_type(ty)).collect();

    let clone_bounds: Vec<proc_macro2::TokenStream> = explicit_args
        .iter()
        .filter(|(_, ty)| !matches!(ty, Type::Reference(_)))
        .map(|(_, ty)| quote! { #ty: ::std::clone::Clone })
        .collect();
    let clone_where = if clone_bounds.is_empty() {
        quote! {}
    } else {
        quote! { where #(#clone_bounds,)* }
    };

    HookArgTokens {
        trait_args_no_last,
        trait_args,
        trait_call_arg_names,
        dispatch_vars,
        cloned,
        pre_conversion,
        field_ident_names,
        fn_bound_types,
        clone_where,
    }
}

/// Emits the three hook trait definitions: `HookTrait`, `StatefulHookTrait`, `__LutumDynHookTrait`.
pub fn generate_hook_trait_defs(
    def_span: proc_macro2::Span,
    vis: &Visibility,
    ctx_ty: &Type,
    ctx_ident: &Ident,
    output_ty: &Type,
    slot: &HookSlotIdents,
    trait_args: &[proc_macro2::TokenStream],
) -> proc_macro2::TokenStream {
    let hook_trait_ident = &slot.hook_trait;
    let stateful_hook_trait_ident = &slot.stateful;
    let dyn_hook_trait_ident = &slot.dyn_trait;
    let hook_trait_def = quote_spanned! { def_span =>
        #[allow(dead_code)]
        #[::async_trait::async_trait]
        #vis trait #hook_trait_ident: Send + Sync {
            async fn call(
                &self,
                #ctx_ident: #ctx_ty,
                #(#trait_args,)*
            ) -> #output_ty;
        }
    };
    let stateful_hook_trait_def = quote_spanned! { def_span =>
        #[allow(dead_code)]
        #[::async_trait::async_trait]
        #vis trait #stateful_hook_trait_ident: Send {
            fn on_reentrancy(err: ::lutum_protocol::hooks::HookReentrancyError) -> #output_ty {
                panic!("stateful hook reentered: {err}");
            }

            async fn call_mut(
                &mut self,
                #ctx_ident: #ctx_ty,
                #(#trait_args,)*
            ) -> #output_ty;
        }
    };
    let dyn_hook_trait_def = quote_spanned! { def_span =>
        #[allow(dead_code)]
        #[::async_trait::async_trait]
        pub(crate) trait #dyn_hook_trait_ident: Send + Sync {
            async fn call_dyn(
                &self,
                #ctx_ident: #ctx_ty,
                #(#trait_args,)*
            ) -> #output_ty;
        }
    };
    quote! { #hook_trait_def #stateful_hook_trait_def #dyn_hook_trait_def }
}

/// Emits the closure `Fn(...)` blanket impls for the hook trait.
/// Returns an empty `TokenStream` when the hook has no explicit args or has non-str ref args
/// (which require HRTB that can't be expressed in a simple `Fn` bound).
pub fn generate_fn_blanket_impl(
    slot: &HookSlotIdents,
    flags: &HookSlotFlags,
    ctx_ty: &Type,
    ctx_inner_ty: &Type,
    ctx_ident: &Ident,
    output_ty: &Type,
    arg_tokens: &HookArgTokens,
) -> proc_macro2::TokenStream {
    if !flags.has_explicit_args || flags.has_ref_arg {
        return quote! {};
    }

    let hook_trait_ident = &slot.hook_trait;
    let trait_has_last = flags.trait_has_last;
    let is_lutum_hook = flags.is_lutum_hook;

    let HookArgTokens {
        trait_args,
        field_ident_names,
        fn_bound_types,
        ..
    } = arg_tokens;

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
                    #(#trait_args,)*
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
                #(#trait_args,)*
            ) -> #output_ty {
                #fn_impl_call
            }
        }

        #ctx_fn_blanket
    }
}

/// Emits the dyn-wrapper and `Stateful<T>` blanket impls for the hook slot.
pub fn generate_blanket_impls(
    slot: &HookSlotIdents,
    ctx_ty: &Type,
    ctx_ident: &Ident,
    output_ty: &Type,
    arg_tokens: &HookArgTokens,
    hook_name: &str,
) -> proc_macro2::TokenStream {
    let hook_trait_ident = &slot.hook_trait;
    let stateful_hook_trait_ident = &slot.stateful;
    let dyn_hook_trait_ident = &slot.dyn_trait;
    let HookArgTokens {
        trait_args,
        trait_call_arg_names,
        ..
    } = arg_tokens;

    let dyn_hook_impl_call = quote! {
        <T as #hook_trait_ident>::call(self, #ctx_ident, #(#trait_call_arg_names,)*).await
    };
    let stateful_hook_impl_call = quote! {
        <T as #stateful_hook_trait_ident>::call_mut(
            &mut *hook,
            #ctx_ident,
            #(#trait_call_arg_names,)*
        )
        .await
    };

    quote! {
        #[::async_trait::async_trait]
        impl<T> #dyn_hook_trait_ident for T
        where
            T: #hook_trait_ident,
        {
            async fn call_dyn(
                &self,
                #ctx_ident: #ctx_ty,
                #(#trait_args,)*
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
                #(#trait_args,)*
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
