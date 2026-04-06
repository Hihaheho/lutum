use heck::{ToSnakeCase, ToUpperCamelCase};
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    Attribute, Data, DeriveInput, Expr, Fields, FieldsNamed, FieldsUnnamed, FnArg, GenericArgument,
    Ident, ItemEnum, ItemFn, ItemStruct, Lifetime, Lit, Meta, MetaNameValue, Pat, PatIdent,
    PathArguments, ReturnType, Token, Type, TypePath, parse::Parse, parse_macro_input,
    punctuated::Punctuated,
};

struct ToolInputArgs {
    output: Type,
    name: Option<String>,
}

impl Parse for ToolInputArgs {
    fn parse(input: syn::parse::ParseStream<'_>) -> syn::Result<Self> {
        let metas = Punctuated::<Meta, Token![,]>::parse_terminated(input)?
            .into_iter()
            .collect::<Vec<_>>();
        let mut output = None;
        let mut name = None;
        for meta in metas {
            match meta {
                Meta::NameValue(MetaNameValue { path, value, .. }) if path.is_ident("output") => {
                    let Expr::Path(expr_path) = value else {
                        return Err(syn::Error::new_spanned(value, "expected type path"));
                    };
                    output = Some(Type::Path(TypePath {
                        qself: None,
                        path: expr_path.path,
                    }));
                }
                Meta::NameValue(MetaNameValue { path, value, .. }) if path.is_ident("name") => {
                    let Expr::Lit(expr_lit) = value else {
                        return Err(syn::Error::new_spanned(value, "expected string literal"));
                    };
                    let Lit::Str(lit) = expr_lit.lit else {
                        return Err(syn::Error::new_spanned(expr_lit, "expected string literal"));
                    };
                    name = Some(lit.value());
                }
                other => {
                    return Err(syn::Error::new_spanned(
                        other,
                        "expected `output = Type` or `name = \"...\"`",
                    ));
                }
            }
        }
        Ok(Self {
            output: output.ok_or_else(|| {
                syn::Error::new(proc_macro2::Span::call_site(), "missing `output = Type`")
            })?,
            name,
        })
    }
}

struct ToolFnArgs {
    skip: Vec<Ident>,
}

impl Parse for ToolFnArgs {
    fn parse(input: syn::parse::ParseStream<'_>) -> syn::Result<Self> {
        if input.is_empty() {
            return Ok(Self { skip: Vec::new() });
        }
        let meta = input.parse::<Meta>()?;
        let mut skip = Vec::new();
        match meta {
            Meta::List(list) if list.path.is_ident("skip") => {
                let idents = list
                    .parse_args_with(Punctuated::<Ident, Token![,]>::parse_terminated)?
                    .into_iter()
                    .collect::<Vec<_>>();
                skip.extend(idents);
            }
            other => {
                return Err(syn::Error::new_spanned(other, "expected `skip(name, ...)`"));
            }
        }
        Ok(Self { skip })
    }
}

#[proc_macro_attribute]
pub fn tool_input(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as ToolInputArgs);
    let input = parse_macro_input!(item as syn::Item);
    match input {
        syn::Item::Struct(item) => expand_tool_input_struct(item, args).into(),
        syn::Item::Enum(item) => expand_tool_input_enum(item, args).into(),
        other => syn::Error::new_spanned(other, "#[tool_input] supports only structs and enums")
            .to_compile_error()
            .into(),
    }
}

#[proc_macro_attribute]
pub fn tool_fn(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as ToolFnArgs);
    let item_fn = parse_macro_input!(item as ItemFn);
    expand_tool_fn(item_fn, args).into()
}

#[proc_macro_attribute]
pub fn hook(attr: TokenStream, item: TokenStream) -> TokenStream {
    if attr.is_empty() {
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            "use #[def_hook(always)], #[def_hook(fallback)], or #[def_hook(singleton)] to declare a hook slot, or #[hook(SlotType)] to implement one",
        )
        .to_compile_error()
        .into();
    }
    let slot_ident = parse_macro_input!(attr as Ident);
    let item_fn = parse_macro_input!(item as ItemFn);
    expand_hook_impl(item_fn, slot_ident).into()
}

#[proc_macro_attribute]
pub fn def_hook(attr: TokenStream, item: TokenStream) -> TokenStream {
    if attr.is_empty() {
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            "use #[def_hook(always)], #[def_hook(fallback)], or #[def_hook(singleton)]",
        )
        .to_compile_error()
        .into();
    }

    let kind_ident = parse_macro_input!(attr as Ident);
    let item_fn = parse_macro_input!(item as ItemFn);

    match kind_ident.to_string().as_str() {
        "always" => expand_hook(item_fn, HookKind::Always).into(),
        "fallback" => expand_hook(item_fn, HookKind::Fallback).into(),
        "singleton" => expand_hook(item_fn, HookKind::Singleton).into(),
        _ => syn::Error::new_spanned(
            kind_ident,
            "#[def_hook(...)] expects `always`, `fallback`, or `singleton`",
        )
        .to_compile_error()
        .into(),
    }
}

#[proc_macro_derive(Toolset)]
pub fn derive_toolset(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    expand_toolset(input).into()
}

fn expand_tool_input_struct(item: ItemStruct, args: ToolInputArgs) -> proc_macro2::TokenStream {
    let ident = item.ident.clone();
    let vis = item.vis.clone();
    let call_ident = format_ident!("{ident}Call");
    let description = doc_string(&item.attrs);
    let name = args
        .name
        .unwrap_or_else(|| ident.to_string().to_snake_case());
    let output = args.output;

    quote! {
        #item

        impl ::lutum::ToolInput for #ident {
            type Output = #output;

            const NAME: &'static str = #name;
            const DESCRIPTION: &'static str = #description;
        }

        impl #ident {
            pub fn tool_use(
                metadata: ::lutum::ToolMetadata,
                output: #output,
            ) -> Result<::lutum::ToolUse, ::lutum::ToolUseError> {
                <Self as ::lutum::ToolInput>::tool_use(metadata, output)
            }
        }

        #[derive(Clone, Debug, Eq, PartialEq)]
        #vis struct #call_ident {
            pub metadata: ::lutum::ToolMetadata,
            pub input: #ident,
        }

        impl ::lutum::toolset::ToolCallWrapper for #call_ident {
            fn metadata(&self) -> &::lutum::ToolMetadata {
                &self.metadata
            }
        }

        impl #call_ident {
            pub fn input(&self) -> &#ident {
                &self.input
            }

            pub fn input_mut(&mut self) -> &mut #ident {
                &mut self.input
            }

            pub fn into_input(self) -> #ident {
                self.input
            }

            pub fn into_parts(self) -> (::lutum::ToolMetadata, #ident) {
                (self.metadata, self.input)
            }

            pub fn tool_use(
                self,
                output: #output,
            ) -> Result<::lutum::ToolUse, ::lutum::ToolUseError> {
                #ident::tool_use(self.metadata, output)
            }
        }

        impl From<#call_ident> for #ident {
            fn from(value: #call_ident) -> Self {
                value.into_input()
            }
        }
    }
}

fn expand_tool_input_enum(item: ItemEnum, args: ToolInputArgs) -> proc_macro2::TokenStream {
    let ident = item.ident.clone();
    let vis = item.vis.clone();
    let call_ident = format_ident!("{ident}Call");
    let description = doc_string(&item.attrs);
    let name = args
        .name
        .unwrap_or_else(|| ident.to_string().to_snake_case());
    let output = args.output;

    quote! {
        #item

        impl ::lutum::ToolInput for #ident {
            type Output = #output;

            const NAME: &'static str = #name;
            const DESCRIPTION: &'static str = #description;
        }

        impl #ident {
            pub fn tool_use(
                metadata: ::lutum::ToolMetadata,
                output: #output,
            ) -> Result<::lutum::ToolUse, ::lutum::ToolUseError> {
                <Self as ::lutum::ToolInput>::tool_use(metadata, output)
            }
        }

        #[derive(Clone, Debug, Eq, PartialEq)]
        #vis struct #call_ident {
            pub metadata: ::lutum::ToolMetadata,
            pub input: #ident,
        }

        impl ::lutum::toolset::ToolCallWrapper for #call_ident {
            fn metadata(&self) -> &::lutum::ToolMetadata {
                &self.metadata
            }
        }

        impl #call_ident {
            pub fn input(&self) -> &#ident {
                &self.input
            }

            pub fn input_mut(&mut self) -> &mut #ident {
                &mut self.input
            }

            pub fn into_input(self) -> #ident {
                self.input
            }

            pub fn into_parts(self) -> (::lutum::ToolMetadata, #ident) {
                (self.metadata, self.input)
            }

            pub fn tool_use(
                self,
                output: #output,
            ) -> Result<::lutum::ToolUse, ::lutum::ToolUseError> {
                #ident::tool_use(self.metadata, output)
            }
        }

        impl From<#call_ident> for #ident {
            fn from(value: #call_ident) -> Self {
                value.into_input()
            }
        }
    }
}

fn expand_tool_fn(item_fn: ItemFn, args: ToolFnArgs) -> proc_macro2::TokenStream {
    if item_fn.sig.receiver().is_some() {
        return syn::Error::new_spanned(&item_fn.sig, "#[tool_fn] does not support methods")
            .to_compile_error();
    }
    let fn_ident = item_fn.sig.ident.clone();
    let input_ident = format_ident!("{}", fn_ident.to_string().to_upper_camel_case());
    let call_ident = format_ident!("{}Call", input_ident);
    let vis = item_fn.vis.clone();
    let description = doc_string(&item_fn.attrs);
    let tool_name = fn_ident.to_string();
    let output_ty = match result_output_type(&item_fn.sig.output) {
        Ok(ty) => ty,
        Err(err) => return err.to_compile_error(),
    };
    let error_ty = match error_type(&item_fn.sig.output) {
        Ok(ty) => ty,
        Err(err) => return err.to_compile_error(),
    };

    let mut call_args = Vec::new();
    let mut field_defs = Vec::new();
    let mut fn_call_args = Vec::new();

    for arg in &item_fn.sig.inputs {
        let FnArg::Typed(pat_ty) = arg else {
            return syn::Error::new_spanned(arg, "unsupported argument").to_compile_error();
        };
        let Pat::Ident(PatIdent { ident, .. }) = pat_ty.pat.as_ref() else {
            return syn::Error::new_spanned(&pat_ty.pat, "expected identifier pattern")
                .to_compile_error();
        };
        if args.skip.iter().any(|skip| skip == ident) {
            call_args.push((ident.clone(), (*pat_ty.ty).clone()));
            fn_call_args.push(quote! { #ident });
        } else {
            let ty = (*pat_ty.ty).clone();
            field_defs.push(quote! { pub #ident: #ty });
            fn_call_args.push(quote! { self.input.#ident });
        }
    }

    let call_method_args = call_args.iter().map(|(ident, ty)| quote! { #ident: #ty });

    quote! {
        #item_fn

        #[derive(
            Clone,
            Debug,
            Eq,
            PartialEq,
            ::serde::Serialize,
            ::serde::Deserialize,
            ::schemars::JsonSchema
        )]
        #vis struct #input_ident {
            #(#field_defs,)*
        }

        impl ::lutum::ToolInput for #input_ident {
            type Output = #output_ty;

            const NAME: &'static str = #tool_name;
            const DESCRIPTION: &'static str = #description;
        }

        impl #input_ident {
            pub fn tool_use(
                metadata: ::lutum::ToolMetadata,
                output: #output_ty,
            ) -> Result<::lutum::ToolUse, ::lutum::ToolUseError> {
                <Self as ::lutum::ToolInput>::tool_use(metadata, output)
            }
        }

        #[derive(Clone, Debug, Eq, PartialEq)]
        #vis struct #call_ident {
            pub metadata: ::lutum::ToolMetadata,
            pub input: #input_ident,
        }

        impl ::lutum::toolset::ToolCallWrapper for #call_ident {
            fn metadata(&self) -> &::lutum::ToolMetadata {
                &self.metadata
            }
        }

        impl #call_ident {
            pub fn input(&self) -> &#input_ident {
                &self.input
            }

            pub fn input_mut(&mut self) -> &mut #input_ident {
                &mut self.input
            }

            pub fn into_input(self) -> #input_ident {
                self.input
            }

            pub fn into_parts(self) -> (::lutum::ToolMetadata, #input_ident) {
                (self.metadata, self.input)
            }

            pub fn tool_use(
                self,
                output: #output_ty,
            ) -> Result<::lutum::ToolUse, ::lutum::ToolUseError> {
                #input_ident::tool_use(self.metadata, output)
            }

            pub async fn call(
                self,
                #(#call_method_args),*
            ) -> Result<::lutum::ToolUse, ::lutum::ToolExecutionError<#error_ty>> {
                let output = #fn_ident(#(#fn_call_args),*)
                    .await
                    .map_err(::lutum::ToolExecutionError::Execute)?;
                #input_ident::tool_use(self.metadata, output)
                    .map_err(::lutum::ToolExecutionError::ToolUse)
            }
        }

        impl From<#call_ident> for #input_ident {
            fn from(value: #call_ident) -> Self {
                value.into_input()
            }
        }
    }
}

struct HookSignature {
    ctx_ident: Ident,
    ctx_ty: Type,
    explicit_args: Vec<(Ident, Type)>,
    output_ty: Type,
    has_last: bool,
}

#[derive(Clone, Copy)]
enum HookKind {
    Always,
    Fallback,
    Singleton,
}

impl HookKind {
    fn default_last_requirement(&self) -> HookLastRequirement {
        match self {
            Self::Always | Self::Fallback => HookLastRequirement::Optional,
            Self::Singleton => HookLastRequirement::Forbidden,
        }
    }

    fn trait_has_last(&self) -> bool {
        matches!(self, Self::Always | Self::Fallback)
    }
}

enum HookLastRequirement {
    Forbidden,
    Optional,
}

#[derive(Clone, Copy)]
enum HookLastRecognition {
    CompatibleOption,
    LastNamedCompatibleOption,
}

fn expand_hook(mut item_fn: ItemFn, kind: HookKind) -> proc_macro2::TokenStream {
    let HookSignature {
        ctx_ident,
        ctx_ty,
        explicit_args,
        output_ty,
        has_last: default_has_last,
    } = match analyze_hook_signature(
        &item_fn,
        kind.default_last_requirement(),
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
    let hook_trait_ident = format_ident!("{slot_ident}Hook");
    let stateful_hook_trait_ident = format_ident!("Stateful{slot_ident}Hook");
    let dyn_hook_trait_ident = format_ident!("__LutumDyn{slot_ident}Hook");
    let registry_ext_ident = format_ident!("{slot_ident}RegistryExt");
    let lutum_ext_ident = format_ident!("{slot_ident}LutumExt");
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

    item_fn.vis = syn::Visibility::Inherited;
    item_fn.sig.ident = default_fn_ident.clone();

    let hook_call_args = explicit_args
        .iter()
        .map(|(ident, ty)| quote! { #ident: #ty })
        .collect::<Vec<_>>();
    let hook_call_arg_names = explicit_args
        .iter()
        .map(|(ident, _)| quote! { #ident })
        .collect::<Vec<_>>();
    let mut hook_trait_args = hook_call_args.clone();
    let mut hook_trait_call_arg_names = hook_call_arg_names.clone();
    if trait_has_last {
        hook_trait_args.push(quote! { last: ::std::option::Option<#output_ty> });
        hook_trait_call_arg_names.push(quote! { last });
    }
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
    let _async_fn_arg_tys = explicit_args
        .iter()
        .map(|(_, ty)| quote! { #ty })
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
    let dyn_hook_dispatch_call = quote! {
        hook.call_dyn(
            #ctx_ident,
            #(#cloned_hook_call_arg_names,)*
            last,
        )
        .await
    };
    let dyn_hook_impl_call = if trait_has_last {
        quote! {
            <T as #hook_trait_ident>::call(
                self,
                #ctx_ident,
                #(#hook_call_arg_names,)*
                last,
            )
            .await
        }
    } else {
        quote! {
            let _ = last;
            <T as #hook_trait_ident>::call(
                self,
                #ctx_ident,
                #(#hook_call_arg_names,)*
            )
            .await
        }
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
        HookKind::Always | HookKind::Fallback => quote! {
            let slot = self
                .slots_mut()
                .entry(::std::any::TypeId::of::<#slot_ident>())
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
                .entry(::std::any::TypeId::of::<#slot_ident>())
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
        HookKind::Always | HookKind::Fallback => quote! {
            let chain = self
                .slots()
                .get(&::std::any::TypeId::of::<#slot_ident>())
                .and_then(|slot| {
                    slot.downcast_ref::<
                        ::std::vec::Vec<::std::sync::Arc<dyn #dyn_hook_trait_ident>>,
                    >()
                });
        },
        HookKind::Singleton => quote! {
            let hook = self
                .slots()
                .get(&::std::any::TypeId::of::<#slot_ident>())
                .and_then(|slot| {
                    slot.downcast_ref::<
                        ::std::sync::Arc<dyn #dyn_hook_trait_ident>,
                    >()
                });
        },
    };
    let dispatch = match kind {
        HookKind::Always => quote! {
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
        HookKind::Fallback => quote! {
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
        HookKind::Singleton => quote! {
            match hook {
                Some(hook) => hook
                    .call_dyn(
                        #ctx_ident,
                        #(#cloned_hook_call_arg_names,)*
                        ::std::option::Option::None,
                    )
                    .await,
                None => #default_call,
            }
        },
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

    quote! {
        #item_fn

        #[allow(dead_code)]
        #vis struct #slot_ident;

        #[allow(dead_code)]
        #[::async_trait::async_trait]
        #vis trait #hook_trait_ident: Send + Sync {
            async fn call(
                &self,
                #ctx_ident: #ctx_ty,
                #(#hook_trait_args,)*
            ) -> #output_ty;
        }

        #[allow(dead_code)]
        #[::async_trait::async_trait]
        #vis trait #stateful_hook_trait_ident: Send {
            fn on_reentrancy(err: ::lutum_protocol::hooks::HookReentrancyError) -> #output_ty {
                panic!("stateful hook reentered: {err}");
            }

            async fn call_mut(
                &mut self,
                #ctx_ident: #ctx_ty,
                #(#hook_trait_args,)*
            ) -> #output_ty;
        }

        #[allow(dead_code)]
        #[::async_trait::async_trait]
        trait #dyn_hook_trait_ident: Send + Sync {
            async fn call_dyn(
                &self,
                #ctx_ident: #ctx_ty,
                #(#hook_call_args,)*
                last: ::std::option::Option<#output_ty>,
            ) -> #output_ty;
        }

        #[::async_trait::async_trait]
        impl<T> #dyn_hook_trait_ident for T
        where
            T: #hook_trait_ident,
        {
            async fn call_dyn(
                &self,
                #ctx_ident: #ctx_ty,
                #(#hook_call_args,)*
                last: ::std::option::Option<#output_ty>,
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

fn expand_hook_impl(item_fn: ItemFn, slot_ident: Ident) -> proc_macro2::TokenStream {
    let HookSignature {
        ctx_ident,
        ctx_ty,
        explicit_args,
        output_ty,
        has_last,
    } = match analyze_hook_signature(
        &item_fn,
        HookLastRequirement::Optional,
        HookLastRecognition::CompatibleOption,
    ) {
        Ok(signature) => signature,
        Err(err) => return err.to_compile_error(),
    };

    let fn_ident = item_fn.sig.ident.clone();
    let vis = item_fn.vis.clone();
    let struct_ident = format_ident!("{}", fn_ident.to_string().to_upper_camel_case());
    let hook_trait_ident = format_ident!("{slot_ident}Hook");
    let hook_call_args = explicit_args
        .iter()
        .map(|(ident, ty)| quote! { #ident: #ty })
        .collect::<Vec<_>>();
    let hook_call_arg_names = explicit_args
        .iter()
        .map(|(ident, _)| quote! { #ident })
        .collect::<Vec<_>>();
    let mut hook_trait_args = hook_call_args.clone();
    let mut hook_trait_call_arg_names = hook_call_arg_names.clone();
    if has_last {
        hook_trait_args.push(quote! { last: ::std::option::Option<#output_ty> });
        hook_trait_call_arg_names.push(quote! { last });
    }

    quote! {
        #item_fn

        #[allow(dead_code)]
        #vis struct #struct_ident;

        #[::async_trait::async_trait]
        impl #hook_trait_ident for #struct_ident {
            async fn call(
                &self,
                #ctx_ident: #ctx_ty,
                #(#hook_trait_args,)*
            ) -> #output_ty {
                #fn_ident(#ctx_ident, #(#hook_trait_call_arg_names,)*).await
            }
        }
    }
}

fn analyze_hook_signature(
    item_fn: &ItemFn,
    last_requirement: HookLastRequirement,
    last_recognition: HookLastRecognition,
) -> syn::Result<HookSignature> {
    if item_fn.sig.receiver().is_some() {
        return Err(syn::Error::new_spanned(
            &item_fn.sig,
            "hook attributes do not support methods",
        ));
    }
    if item_fn.sig.asyncness.is_none() {
        return Err(syn::Error::new_spanned(
            &item_fn.sig,
            "hook attributes require an async fn",
        ));
    }
    if !item_fn.sig.generics.params.is_empty() || item_fn.sig.generics.where_clause.is_some() {
        return Err(syn::Error::new_spanned(
            &item_fn.sig.generics,
            "hook attributes do not support generics or where clauses",
        ));
    }
    if item_fn.sig.constness.is_some()
        || item_fn.sig.unsafety.is_some()
        || item_fn.sig.abi.is_some()
        || item_fn.sig.variadic.is_some()
    {
        return Err(syn::Error::new_spanned(
            &item_fn.sig,
            "hook attributes support only plain async functions",
        ));
    }

    let output_ty = output_type_or_unit(&item_fn.sig.output);
    let inputs = item_fn.sig.inputs.iter().collect::<Vec<_>>();
    if inputs.is_empty() {
        return Err(syn::Error::new_spanned(
            &item_fn.sig.inputs,
            "hook attributes require a shared-reference first argument (e.g. `llm: &Lutum`)",
        ));
    }

    let Some(FnArg::Typed(ctx_arg)) = inputs.first().copied() else {
        return Err(syn::Error::new_spanned(
            inputs.first().expect("hook must have inputs"),
            "first hook argument must be a typed reference (e.g. `llm: &Lutum` or `extensions: &RequestExtensions`)",
        ));
    };
    let Pat::Ident(PatIdent {
        ident: ctx_ident, ..
    }) = ctx_arg.pat.as_ref()
    else {
        return Err(syn::Error::new_spanned(
            &ctx_arg.pat,
            "expected an identifier for the first hook argument",
        ));
    };
    let Type::Reference(_) = ctx_arg.ty.as_ref() else {
        return Err(syn::Error::new_spanned(
            &ctx_arg.ty,
            "first hook argument must be a shared reference (e.g. `&Lutum` or `&RequestExtensions`)",
        ));
    };
    let ctx_ident = ctx_ident.clone();
    let ctx_ty = (*ctx_arg.ty).clone();

    let has_last = inputs
        .last()
        .copied()
        .map(|arg| hook_last_matches(arg, &output_ty, last_recognition))
        .transpose()?
        .unwrap_or(false);

    match last_requirement {
        HookLastRequirement::Forbidden if has_last => {
            return Err(syn::Error::new_spanned(
                inputs.last().expect("hook must have final input"),
                "#[def_hook(singleton)] does not accept a `last: Option<Return>` argument",
            ));
        }
        _ => {}
    }

    let mut explicit_args = Vec::new();
    let explicit_end = inputs.len() - usize::from(has_last);
    for arg in &inputs[1..explicit_end] {
        let FnArg::Typed(pat_ty) = arg else {
            return Err(syn::Error::new_spanned(arg, "unsupported hook argument"));
        };
        let Pat::Ident(PatIdent { ident, .. }) = pat_ty.pat.as_ref() else {
            return Err(syn::Error::new_spanned(
                &pat_ty.pat,
                "expected identifier pattern",
            ));
        };
        explicit_args.push((ident.clone(), (*pat_ty.ty).clone()));
    }

    Ok(HookSignature {
        ctx_ident,
        ctx_ty,
        explicit_args,
        output_ty,
        has_last,
    })
}

fn hook_last_matches(
    arg: &FnArg,
    output_ty: &Type,
    recognition: HookLastRecognition,
) -> syn::Result<bool> {
    let FnArg::Typed(pat_ty) = arg else {
        return Ok(false);
    };
    let Some(inner) = option_inner_type(pat_ty.ty.as_ref()) else {
        return Ok(false);
    };
    if !types_match(inner, output_ty) {
        return Ok(false);
    }

    let Pat::Ident(PatIdent { ident, .. }) = pat_ty.pat.as_ref() else {
        return match recognition {
            HookLastRecognition::CompatibleOption => Err(syn::Error::new_spanned(
                &pat_ty.pat,
                "expected `last` identifier",
            )),
            HookLastRecognition::LastNamedCompatibleOption => Ok(false),
        };
    };

    Ok(match recognition {
        HookLastRecognition::CompatibleOption => true,
        HookLastRecognition::LastNamedCompatibleOption => is_hook_last_ident(ident),
    })
}

fn expand_toolset(input: DeriveInput) -> proc_macro2::TokenStream {
    let enum_ident = input.ident.clone();
    let vis = input.vis.clone();
    let Data::Enum(data_enum) = input.data else {
        return syn::Error::new_spanned(input, "Toolset can only be derived for enums")
            .to_compile_error();
    };

    let call_enum_ident = format_ident!("{enum_ident}Call");
    let selector_enum_ident = format_ident!("{enum_ident}Selector");
    let variants = data_enum.variants.into_iter().collect::<Vec<_>>();

    let mut wrapper_variants = Vec::new();
    let mut selector_variants = Vec::new();
    let mut metadata_arms = Vec::new();
    let mut call_selector_arms = Vec::new();
    let mut call_into_input_arms = Vec::new();
    let mut call_into_parts_arms = Vec::new();
    let mut parse_arms = Vec::new();
    let mut defs = Vec::new();
    let mut selector_name_arms = Vec::new();
    let mut selector_definition_arms = Vec::new();
    let mut selector_try_from_arms = Vec::new();
    let mut selector_all = Vec::new();
    let mut selector_expected_names = Vec::new();

    for (index, variant) in variants.into_iter().enumerate() {
        let variant_ident = variant.ident;
        let input_ty = match variant.fields {
            Fields::Unnamed(FieldsUnnamed { unnamed, .. }) if unnamed.len() == 1 => {
                unnamed.first().unwrap().ty.clone()
            }
            Fields::Named(FieldsNamed { named, .. }) if named.len() == 1 => {
                named.first().unwrap().ty.clone()
            }
            other => {
                return syn::Error::new_spanned(
                    other,
                    "Toolset variants must contain exactly one ToolInput payload",
                )
                .to_compile_error();
            }
        };
        let wrapper_ident = wrapper_ident_for_type(&input_ty);
        let tool_name = quote! { <#input_ty as ::lutum::ToolInput>::NAME };

        wrapper_variants.push(quote! { #variant_ident(#wrapper_ident) });
        selector_variants.push(quote! { #variant_ident });
        metadata_arms.push(quote! { Self::#variant_ident(inner) => &inner.metadata });
        call_selector_arms
            .push(quote! { Self::#variant_ident(_) => #selector_enum_ident::#variant_ident });
        call_into_input_arms.push(
            quote! { Self::#variant_ident(inner) => #enum_ident::#variant_ident(inner.into_input()) },
        );
        call_into_parts_arms.push(quote! {
            Self::#variant_ident(inner) => {
                let (metadata, input) = inner.into_parts();
                (metadata, #enum_ident::#variant_ident(input))
            }
        });
        defs.push(quote! { ::lutum::ToolDef::for_input::<#input_ty>() });
        selector_name_arms.push(quote! { Self::#variant_ident => #tool_name });
        selector_definition_arms.push(
            quote! { Self::#variant_ident => &<#enum_ident as ::lutum::Toolset>::definitions()[#index] },
        );
        selector_try_from_arms.push(quote! { #tool_name => Some(Self::#variant_ident) });
        selector_all.push(quote! { Self::#variant_ident });
        selector_expected_names.push(tool_name.clone());
        parse_arms.push(quote! {
            #tool_name => {
                let name = metadata.name.as_str().to_string();
                let input = ::serde_json::from_str::<#input_ty>(metadata.arguments.get())
                    .map_err(|source| ::lutum::ToolCallError::Deserialize {
                        name,
                        source,
                    })?;
                Ok(#call_enum_ident::#variant_ident(#wrapper_ident {
                    metadata,
                    input,
                }))
            }
        });
    }

    quote! {
        #[derive(Clone, Debug, Eq, PartialEq)]
        #vis enum #call_enum_ident {
            #(#wrapper_variants,)*
        }

        #[derive(
            Clone,
            Copy,
            Debug,
            Eq,
            PartialEq,
            Hash
        )]
        #vis enum #selector_enum_ident {
            #(#selector_variants,)*
        }

        impl #selector_enum_ident {
            pub const ALL: &'static [Self] = &[#(#selector_all),*];

            pub const fn name(self) -> &'static str {
                match self {
                    #(#selector_name_arms,)*
                }
            }

            pub fn definitions() -> &'static [::lutum::ToolDef] {
                <#enum_ident as ::lutum::Toolset>::definitions()
            }

            pub fn definition(self) -> &'static ::lutum::ToolDef {
                <Self as ::lutum::toolset::ToolSelector<#enum_ident>>::definition(self)
            }

            pub fn try_from_name(name: &str) -> Option<Self> {
                match name {
                    #(#selector_try_from_arms,)*
                    _ => None,
                }
            }

            pub fn from_name(name: &str) -> Result<Self, ::lutum::ToolCallError> {
                Self::try_from_name(name).ok_or_else(|| ::lutum::ToolCallError::UnknownTool {
                    name: name.to_string(),
                })
            }
        }

        impl ::serde::Serialize for #selector_enum_ident {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: ::serde::Serializer,
            {
                serializer.serialize_str(self.name())
            }
        }

        impl<'de> ::serde::Deserialize<'de> for #selector_enum_ident {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: ::serde::Deserializer<'de>,
            {
                let name = <::std::string::String as ::serde::Deserialize>::deserialize(deserializer)?;
                Self::try_from_name(&name).ok_or_else(|| {
                    <D::Error as ::serde::de::Error>::unknown_variant(
                        &name,
                        &[#(#selector_expected_names),*],
                    )
                })
            }
        }

        impl ::schemars::JsonSchema for #selector_enum_ident {
            fn inline_schema() -> bool {
                true
            }

            fn schema_name() -> ::std::borrow::Cow<'static, str> {
                ::std::borrow::Cow::Borrowed(stringify!(#selector_enum_ident))
            }

            fn json_schema(
                _generator: &mut ::schemars::SchemaGenerator,
            ) -> ::schemars::Schema {
                ::schemars::json_schema!({
                    "type": "string",
                    "enum": [#(#selector_expected_names),*]
                })
            }
        }

        impl ::lutum::Toolset for #enum_ident {
            type ToolCall = #call_enum_ident;
            type Selector = #selector_enum_ident;

            fn definitions() -> &'static [::lutum::ToolDef] {
                static DEFS: ::std::sync::OnceLock<::std::vec::Vec<::lutum::ToolDef>> =
                    ::std::sync::OnceLock::new();
                DEFS.get_or_init(|| vec![#(#defs),*]).as_slice()
            }

            fn parse_tool_call(
                metadata: ::lutum::ToolMetadata,
            ) -> Result<Self::ToolCall, ::lutum::ToolCallError> {
                match metadata.name.as_str() {
                    #(#parse_arms,)*
                    _ => Err(::lutum::ToolCallError::UnknownTool {
                        name: metadata.name.as_str().to_string(),
                    }),
                }
            }
        }

        impl ::lutum::toolset::ToolCallWrapper for #call_enum_ident {
            fn metadata(&self) -> &::lutum::ToolMetadata {
                match self {
                    #(#metadata_arms,)*
                }
            }
        }

        impl #call_enum_ident {
            pub fn selector(&self) -> #selector_enum_ident {
                match self {
                    #(#call_selector_arms,)*
                }
            }

            pub fn into_input(self) -> #enum_ident {
                match self {
                    #(#call_into_input_arms,)*
                }
            }

            pub fn into_parts(self) -> (::lutum::ToolMetadata, #enum_ident) {
                match self {
                    #(#call_into_parts_arms,)*
                }
            }
        }

        impl From<#call_enum_ident> for #enum_ident {
            fn from(value: #call_enum_ident) -> Self {
                value.into_input()
            }
        }

        impl ::lutum::toolset::ToolSelector<#enum_ident> for #selector_enum_ident {
            fn name(self) -> &'static str {
                self.name()
            }

            fn definition(self) -> &'static ::lutum::ToolDef {
                match self {
                    #(#selector_definition_arms,)*
                }
            }

            fn all() -> &'static [Self] {
                Self::ALL
            }

            fn try_from_name(name: &str) -> Option<Self> {
                Self::try_from_name(name)
            }
        }
    }
}

fn doc_string(attrs: &[Attribute]) -> String {
    attrs
        .iter()
        .filter_map(|attr| match &attr.meta {
            Meta::NameValue(MetaNameValue { path, value, .. }) if path.is_ident("doc") => {
                let Expr::Lit(expr_lit) = value else {
                    return None;
                };
                let Lit::Str(lit) = &expr_lit.lit else {
                    return None;
                };
                Some(lit.value().trim().to_string())
            }
            _ => None,
        })
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

fn output_type_or_unit(output: &ReturnType) -> Type {
    match output {
        ReturnType::Default => syn::parse_quote!(()),
        ReturnType::Type(_, ty) => *ty.clone(),
    }
}

fn result_output_type(output: &ReturnType) -> syn::Result<Type> {
    let ReturnType::Type(_, ty) = output else {
        return Err(syn::Error::new_spanned(
            output,
            "tool functions must return Result<_, _>",
        ));
    };
    let Type::Path(type_path) = ty.as_ref() else {
        return Err(syn::Error::new_spanned(ty, "expected Result<_, _>"));
    };
    let Some(segment) = type_path.path.segments.last() else {
        return Err(syn::Error::new_spanned(type_path, "expected Result<_, _>"));
    };
    if segment.ident != "Result" {
        return Err(syn::Error::new_spanned(segment, "expected Result<_, _>"));
    }
    let PathArguments::AngleBracketed(args) = &segment.arguments else {
        return Err(syn::Error::new_spanned(segment, "expected Result<_, _>"));
    };
    let Some(GenericArgument::Type(ok_ty)) = args.args.first() else {
        return Err(syn::Error::new_spanned(args, "expected Result<Ok, Err>"));
    };
    Ok(ok_ty.clone())
}

fn error_type(output: &ReturnType) -> syn::Result<Type> {
    let ReturnType::Type(_, ty) = output else {
        return Err(syn::Error::new_spanned(
            output,
            "tool functions must return Result<_, _>",
        ));
    };
    let Type::Path(type_path) = ty.as_ref() else {
        return Err(syn::Error::new_spanned(ty, "expected Result<_, _>"));
    };
    let Some(segment) = type_path.path.segments.last() else {
        return Err(syn::Error::new_spanned(type_path, "expected Result<_, _>"));
    };
    let PathArguments::AngleBracketed(args) = &segment.arguments else {
        return Err(syn::Error::new_spanned(segment, "expected Result<_, _>"));
    };
    let Some(GenericArgument::Type(err_ty)) = args.args.iter().nth(1) else {
        return Err(syn::Error::new_spanned(args, "expected Result<Ok, Err>"));
    };
    Ok(err_ty.clone())
}

fn wrapper_ident_for_type(ty: &Type) -> Ident {
    if let Type::Path(path) = ty {
        let ident = &path.path.segments.last().expect("type path").ident;
        format_ident!("{ident}Call")
    } else {
        panic!("Toolset variant payloads must be path types");
    }
}

fn is_lutum_ref(ty: &Type) -> bool {
    let Type::Reference(reference) = ty else {
        return false;
    };
    if reference.mutability.is_some() {
        return false;
    }
    let Type::Path(type_path) = reference.elem.as_ref() else {
        return false;
    };
    type_path
        .path
        .segments
        .last()
        .is_some_and(|segment| segment.ident == "Lutum")
}

fn option_inner_type(ty: &Type) -> Option<&Type> {
    let Type::Path(type_path) = ty else {
        return None;
    };
    let segment = type_path.path.segments.last()?;
    if segment.ident != "Option" {
        return None;
    }
    let PathArguments::AngleBracketed(args) = &segment.arguments else {
        return None;
    };
    match args.args.first()? {
        GenericArgument::Type(inner) => Some(inner),
        _ => None,
    }
}

fn is_hook_last_ident(ident: &Ident) -> bool {
    let ident = ident.to_string();
    ident == "last" || ident.starts_with("_last")
}

fn types_match(lhs: &Type, rhs: &Type) -> bool {
    strip_type_wrappers(lhs) == strip_type_wrappers(rhs)
}

fn strip_type_wrappers(ty: &Type) -> &Type {
    match ty {
        Type::Group(group) => strip_type_wrappers(&group.elem),
        Type::Paren(paren) => strip_type_wrappers(&paren.elem),
        _ => ty,
    }
}

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
