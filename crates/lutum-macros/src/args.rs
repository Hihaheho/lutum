use syn::{
    Expr, Ident, Lit, Meta, MetaNameValue, Token, Type, TypePath, parse::Parse,
    punctuated::Punctuated,
};

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
    pub chain: Option<syn::Path>,
    pub accumulate: Option<syn::Path>,
    pub finalize: Option<syn::Path>,
}

impl syn::parse::Parse for HookDefAttrs {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let mode: syn::Ident = input.parse()?;
        let mut chain = None;
        let mut accumulate = None;
        let mut finalize = None;
        while input.peek(Token![,]) {
            input.parse::<Token![,]>()?;
            let key: syn::Ident = input.parse()?;
            match key.to_string().as_str() {
                "chain" => {
                    input.parse::<Token![=]>()?;
                    chain = Some(input.parse::<syn::Path>()?);
                }
                "accumulate" => {
                    input.parse::<Token![=]>()?;
                    accumulate = Some(input.parse::<syn::Path>()?);
                }
                "finalize" => {
                    input.parse::<Token![=]>()?;
                    finalize = Some(input.parse::<syn::Path>()?);
                }
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!(
                            "unknown #[def_hook] option '{other}'; expected 'chain', 'accumulate', or 'finalize'"
                        ),
                    ));
                }
            }
        }
        Ok(Self {
            mode,
            chain,
            accumulate,
            finalize,
        })
    }
}
