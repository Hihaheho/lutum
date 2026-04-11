use syn::{
    Expr, Ident, Lit, Meta, MetaNameValue, Token, Type, TypePath, parse::Parse,
    punctuated::Punctuated,
};

/// Parses `ident = Type, ident2 = Type2, ...` from `#[nested_hooks(...)]`.
pub struct NestedHooksAttr {
    pub entries: Vec<(Ident, Type)>,
}

impl Parse for NestedHooksAttr {
    fn parse(input: syn::parse::ParseStream<'_>) -> syn::Result<Self> {
        let mut entries = Vec::new();
        while !input.is_empty() {
            let ident: Ident = input.parse()?;
            input.parse::<Token![=]>()?;
            let ty: Type = input.parse()?;
            entries.push((ident, ty));
            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }
        Ok(Self { entries })
    }
}

#[derive(Clone)]
pub struct HookAttrValue<T> {
    pub value: T,
    pub span: proc_macro2::Span,
}

pub struct ToolInputArgs {
    pub output: Type,
    pub name: Option<String>,
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

pub struct ToolFnArgs {
    pub skip: Vec<Ident>,
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

#[derive(Clone)]
pub struct HookDefAttrs {
    pub mode: syn::Ident,
    pub chain: Option<HookAttrValue<syn::Path>>,
    pub aggregate: Option<HookAttrValue<syn::Path>>,
    pub finalize: Option<HookAttrValue<syn::Path>>,
    pub output: Option<HookAttrValue<syn::TypePath>>,
    pub custom: Option<HookAttrValue<syn::Path>>,
}

impl syn::parse::Parse for HookDefAttrs {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let mode: syn::Ident = input.parse()?;
        let mut chain = None;
        let mut aggregate = None;
        let mut finalize = None;
        let mut output = None;
        let mut custom = None;
        while input.peek(Token![,]) {
            input.parse::<Token![,]>()?;
            let key: syn::Ident = input.parse()?;
            let key_span = key.span();
            match key.to_string().as_str() {
                "chain" => {
                    input.parse::<Token![=]>()?;
                    chain = Some(HookAttrValue {
                        value: input.parse::<syn::Path>()?,
                        span: key_span,
                    });
                }
                "aggregate" => {
                    input.parse::<Token![=]>()?;
                    aggregate = Some(HookAttrValue {
                        value: input.parse::<syn::Path>()?,
                        span: key_span,
                    });
                }
                "finalize" => {
                    input.parse::<Token![=]>()?;
                    finalize = Some(HookAttrValue {
                        value: input.parse::<syn::Path>()?,
                        span: key_span,
                    });
                }
                "output" => {
                    input.parse::<Token![=]>()?;
                    output = Some(HookAttrValue {
                        value: input.parse::<syn::TypePath>()?,
                        span: key_span,
                    });
                }
                "custom" => {
                    input.parse::<Token![=]>()?;
                    custom = Some(HookAttrValue {
                        value: input.parse::<syn::Path>()?,
                        span: key_span,
                    });
                }
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!(
                            "unknown hook option '{other}'; expected 'chain', 'aggregate', 'finalize', 'output', or 'custom'"
                        ),
                    ));
                }
            }
        }
        Ok(Self {
            mode,
            chain,
            aggregate,
            finalize,
            output,
            custom,
        })
    }
}
