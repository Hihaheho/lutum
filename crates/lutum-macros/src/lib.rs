mod args;
mod hook;
mod tool_fn;
mod tool_input;
mod toolset;
pub(crate) mod utils;

pub(crate) use args::*;
use hook::*;
use tool_fn::*;
use tool_input::*;
use toolset::*;

use proc_macro::TokenStream;
use syn::{DeriveInput, ItemFn, ItemStruct, Path, parse_macro_input};

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
            "use #[def_hook(always)], #[def_hook(fallback)], #[def_hook(singleton)], #[def_global_hook(always)], #[def_global_hook(fallback)], or #[def_global_hook(singleton)] to declare a hook slot, or #[hook(SlotType)] / #[hook(path::to::SlotType)] to implement one",
        )
        .to_compile_error()
        .into();
    }
    let slot_path = parse_macro_input!(attr as Path);
    let item_fn = parse_macro_input!(item as ItemFn);
    expand_hook_impl(item_fn, slot_path).into()
}

#[proc_macro_attribute]
pub fn def_hook(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attrs: HookDefAttrs = match syn::parse(attr) {
        Ok(a) => a,
        Err(e) => return e.to_compile_error().into(),
    };
    let item_fn = match syn::parse(item) {
        Ok(f) => f,
        Err(e) => return e.to_compile_error().into(),
    };
    let mode_str = attrs.mode.to_string();
    let kind = match mode_str.as_str() {
        "always" => HookKind::Always(HookOptions {
            chain: attrs.chain,
            aggregate: attrs.aggregate,
            finalize: attrs.finalize,
        }),
        "fallback" => HookKind::Fallback(HookOptions {
            chain: attrs.chain,
            aggregate: attrs.aggregate,
            finalize: attrs.finalize,
        }),
        "singleton" => {
            if attrs.chain.is_some() || attrs.aggregate.is_some() || attrs.finalize.is_some() {
                return syn::Error::new_spanned(
                    attrs.mode,
                    "#[def_hook(singleton)] does not support 'chain', 'aggregate', or 'finalize'",
                )
                .to_compile_error()
                .into();
            }
            HookKind::Singleton
        }
        _ => {
            return syn::Error::new_spanned(
                attrs.mode,
                "#[def_hook(...)] expects `always`, `fallback`, or `singleton` as first argument",
            )
            .to_compile_error()
            .into();
        }
    };
    expand_local_hook(item_fn, kind).into()
}

#[proc_macro_attribute]
pub fn def_global_hook(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attrs: HookDefAttrs = match syn::parse(attr) {
        Ok(a) => a,
        Err(e) => return e.to_compile_error().into(),
    };
    let item_fn = match syn::parse(item) {
        Ok(f) => f,
        Err(e) => return e.to_compile_error().into(),
    };
    let mode_str = attrs.mode.to_string();
    let kind = match mode_str.as_str() {
        "always" => HookKind::Always(HookOptions {
            chain: attrs.chain,
            aggregate: attrs.aggregate,
            finalize: attrs.finalize,
        }),
        "fallback" => HookKind::Fallback(HookOptions {
            chain: attrs.chain,
            aggregate: attrs.aggregate,
            finalize: attrs.finalize,
        }),
        "singleton" => {
            if attrs.chain.is_some() || attrs.aggregate.is_some() || attrs.finalize.is_some() {
                return syn::Error::new_spanned(
                    attrs.mode,
                    "#[def_global_hook(singleton)] does not support 'chain', 'aggregate', or 'finalize'",
                )
                .to_compile_error()
                .into();
            }
            HookKind::Singleton
        }
        _ => {
            return syn::Error::new_spanned(
                attrs.mode,
                "#[def_global_hook(...)] expects `always`, `fallback`, or `singleton` as first argument",
            )
            .to_compile_error()
            .into();
        }
    };
    expand_global_hook(item_fn, kind).into()
}

#[proc_macro_attribute]
pub fn hooks(attr: TokenStream, item: TokenStream) -> TokenStream {
    if !attr.is_empty() {
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            "#[hooks] does not accept arguments",
        )
        .to_compile_error()
        .into();
    }

    let item_struct = parse_macro_input!(item as ItemStruct);
    expand_hooks(item_struct).into()
}

#[proc_macro_derive(Toolset)]
pub fn derive_toolset(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    expand_toolset(input).into()
}
