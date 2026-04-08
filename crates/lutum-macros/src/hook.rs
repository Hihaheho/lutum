mod def_global;
mod def_local;
mod def_shared;
#[allow(clippy::module_inception)]
mod hook;
mod hooks;

use std::collections::HashSet;

pub use def_global::*;
pub use def_local::*;
pub use def_shared::*;
pub use hook::*;
pub use hooks::*;

use heck::{ToSnakeCase, ToUpperCamelCase};
use quote::{format_ident, quote};
use syn::{
    FnArg, GenericArgument, Ident, ItemFn, Pat, PatIdent, PathArguments, ReturnType, Type,
    spanned::Spanned,
};

pub struct HookSignature {
    ctx_ident: Ident,
    ctx_ty: Type,
    explicit_args: Vec<(Ident, Type)>,
    output_ty: Type,
    has_last: bool,
    last_span: Option<proc_macro2::Span>,
}

pub struct HookOptions {
    pub chain: Option<syn::Path>,
    pub accumulate: Option<syn::Path>,
    pub finalize: Option<syn::Path>,
}

pub enum HookKind {
    Always(HookOptions),
    Fallback(HookOptions),
    Singleton,
}

impl HookKind {
    fn default_last_requirement(&self) -> HookLastRequirement {
        HookLastRequirement::Forbidden
    }

    fn trait_has_last(&self) -> bool {
        match self {
            Self::Always(opts) | Self::Fallback(opts) => opts.accumulate.is_none(),
            Self::Singleton => false,
        }
    }

    fn opts(&self) -> Option<&HookOptions> {
        match self {
            Self::Always(opts) | Self::Fallback(opts) => Some(opts),
            Self::Singleton => None,
        }
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

fn analyze_hook_signature(
    item_fn: &ItemFn,
    last_requirement: HookLastRequirement,
    forbidden_last_message: &str,
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
            "hook attributes require a shared-reference dispatch context as the first argument (e.g. `llm: &Lutum`)",
        ));
    }

    let Some(FnArg::Typed(ctx_arg)) = inputs.first().copied() else {
        return Err(syn::Error::new_spanned(
            inputs.first().expect("hook must have inputs"),
            "first hook argument must be a typed shared-reference dispatch context (e.g. `llm: &Lutum` or `extensions: &RequestExtensions`)",
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
            "first hook argument must be a shared-reference dispatch context (e.g. `&Lutum` or `&RequestExtensions`)",
        ));
    };
    let ctx_ident = ctx_ident.clone();
    let ctx_ty = (*ctx_arg.ty).clone();

    let last_arg = inputs.last().copied();
    let has_last = last_arg
        .map(|arg| hook_last_matches(arg, &output_ty, last_recognition))
        .transpose()?
        .unwrap_or(false);
    let last_span = if has_last {
        last_arg.map(|arg| match arg {
            FnArg::Typed(pat_ty) => pat_ty.span(),
            FnArg::Receiver(receiver) => receiver.span(),
        })
    } else {
        None
    };

    match last_requirement {
        HookLastRequirement::Forbidden if has_last => {
            return Err(syn::Error::new(
                last_span.expect("last span must exist when last is present"),
                forbidden_last_message,
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
        last_span,
    })
}

fn output_type_or_unit(output: &ReturnType) -> Type {
    match output {
        ReturnType::Default => syn::parse_quote!(()),
        ReturnType::Type(_, ty) => *ty.clone(),
    }
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

pub fn types_match(lhs: &Type, rhs: &Type) -> bool {
    strip_type_wrappers(lhs) == strip_type_wrappers(rhs)
}

pub fn strip_type_wrappers(ty: &Type) -> &Type {
    match ty {
        Type::Group(group) => strip_type_wrappers(&group.elem),
        Type::Paren(paren) => strip_type_wrappers(&paren.elem),
        _ => ty,
    }
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

fn hook_named_impl_helper_macro_ident(slot_ident: &Ident) -> Ident {
    format_ident!(
        "__lutum_hook_named_impl_{}",
        slot_ident.to_string().to_snake_case()
    )
}

fn is_hook_last_ident(ident: &Ident) -> bool {
    let ident = ident.to_string();
    ident == "last" || ident.starts_with("_last")
}

/// Returns the type to use in hook trait method parameters and `Fn()` closure bounds.
/// - `&str` → `String`  (owned, no lifetime needed)
/// - `&T`   → `&T`      (keep reference, no explicit lifetime — anonymous in trait methods)
/// - `T`    → `T`
pub fn hook_param_type(ty: &Type) -> proc_macro2::TokenStream {
    match ty {
        Type::Reference(r) if is_str_type(&r.elem) => quote! { ::std::string::String },
        other => quote! { #other },
    }
}

/// Returns true if the type is `&T` for some non-`str` T.
pub fn is_non_str_ref(ty: &Type) -> bool {
    matches!(ty, Type::Reference(r) if !is_str_type(&r.elem))
}

fn is_str_ref(ty: &Type) -> bool {
    matches!(ty, Type::Reference(r) if is_str_type(&r.elem))
}

fn is_str_type(ty: &Type) -> bool {
    matches!(ty, Type::Path(p) if p.path.is_ident("str"))
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

fn normalized_hook_arg_field_idents(explicit_args: &[(Ident, Type)]) -> Vec<Ident> {
    let mut used = HashSet::new();
    explicit_args
        .iter()
        .map(|(ident, _)| {
            let original = ident.to_string();
            let trimmed = original.trim_start_matches('_');
            let candidate = if !trimmed.is_empty() && !used.contains(trimmed) {
                trimmed.to_string()
            } else {
                original
            };
            used.insert(candidate.clone());
            format_ident!("{}", candidate)
        })
        .collect()
}
