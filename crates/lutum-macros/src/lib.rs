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
use syn::{DeriveInput, ItemFn, ItemImpl, ItemTrait, Path, parse_macro_input};

fn build_hook_kind(attrs: HookDefAttrs, macro_name: &str) -> syn::Result<HookKind> {
    let mode_str = attrs.mode.to_string();
    match mode_str.as_str() {
        "always" | "fallback" => {
            if let (Some(_), Some(finalize)) = (&attrs.aggregate, &attrs.finalize) {
                return Err(syn::Error::new(
                    finalize.span,
                    format!(
                        "#[{macro_name}(...)] does not support using 'aggregate' and 'finalize' together"
                    ),
                ));
            }

            if let Some(output) = &attrs.output
                && attrs.aggregate.is_none()
                && attrs.finalize.is_none()
            {
                return Err(syn::Error::new(
                    output.span,
                    format!("#[{macro_name}(...)] 'output' requires 'aggregate' or 'finalize'"),
                ));
            }

            if let Some(custom) = &attrs.custom {
                if mode_str == "always" {
                    return Err(syn::Error::new(
                        custom.span,
                        format!(
                            "#[{macro_name}(always)] does not support 'custom'; use 'fallback' mode"
                        ),
                    ));
                }
                if let Some(chain) = &attrs.chain {
                    return Err(syn::Error::new(
                        chain.span,
                        format!("#[{macro_name}(...)] 'custom' and 'chain' are mutually exclusive"),
                    ));
                }
                if let Some(aggregate) = &attrs.aggregate {
                    return Err(syn::Error::new(
                        aggregate.span,
                        format!(
                            "#[{macro_name}(...)] 'custom' and 'aggregate' are mutually exclusive"
                        ),
                    ));
                }
                if let Some(finalize) = &attrs.finalize {
                    return Err(syn::Error::new(
                        finalize.span,
                        format!(
                            "#[{macro_name}(...)] 'custom' and 'finalize' are mutually exclusive"
                        ),
                    ));
                }
            }

            let opts = HookOptions {
                chain: attrs.chain.map(|chain| chain.value),
                aggregate: attrs.aggregate.map(|aggregate| aggregate.value),
                finalize: attrs.finalize.map(|finalize| finalize.value),
                output: attrs.output.map(|output| syn::Type::Path(output.value)),
                custom: attrs.custom.map(|custom| custom.value),
            };
            Ok(if mode_str == "always" {
                HookKind::Always(opts)
            } else {
                HookKind::Fallback(opts)
            })
        }
        "singleton" => {
            if attrs.chain.is_some()
                || attrs.aggregate.is_some()
                || attrs.finalize.is_some()
                || attrs.output.is_some()
                || attrs.custom.is_some()
            {
                return Err(syn::Error::new_spanned(
                    attrs.mode,
                    format!(
                        "#[{macro_name}(singleton)] does not support 'chain', 'aggregate', 'finalize', 'output', or 'custom'"
                    ),
                ));
            }
            Ok(HookKind::Singleton)
        }
        _ => Err(syn::Error::new_spanned(
            attrs.mode,
            format!(
                "#[{macro_name}(...)] expects `always`, `fallback`, or `singleton` as first argument"
            ),
        )),
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
pub fn hooks(attr: TokenStream, item: TokenStream) -> TokenStream {
    if !attr.is_empty() {
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            "#[hooks] does not accept arguments",
        )
        .to_compile_error()
        .into();
    }

    let item_trait = parse_macro_input!(item as ItemTrait);
    expand_hooks(item_trait).into()
}

#[proc_macro_attribute]
pub fn impl_hook(attr: TokenStream, item: TokenStream) -> TokenStream {
    let slot_path = parse_macro_input!(attr as Path);
    let item_fn = parse_macro_input!(item as ItemFn);
    expand_hook_impl(item_fn, slot_path).into()
}

#[proc_macro_attribute]
pub fn impl_hooks(attr: TokenStream, item: TokenStream) -> TokenStream {
    let hooks_set_path = parse_macro_input!(attr as Path);
    let item_impl = parse_macro_input!(item as ItemImpl);
    expand_hooks_impl(item_impl, hooks_set_path).into()
}

#[proc_macro_derive(Toolset, attributes(toolset, tool))]
pub fn derive_toolset(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    expand_toolset(input).into()
}

/// Helper attribute consumed by `#[hooks]` to declare nested hooks fields.
///
/// Usage: `#[nested_hooks(field_name = HooksType, ...)]` on a `#[hooks]`-annotated trait.
/// Not intended to be used standalone — `#[hooks]` strips and processes it.
#[proc_macro_attribute]
pub fn nested_hooks(_attr: TokenStream, item: TokenStream) -> TokenStream {
    // Pass-through: actual processing is done by #[hooks].
    item
}
