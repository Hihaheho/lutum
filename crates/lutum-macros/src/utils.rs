use syn::{Attribute, Expr, Lit, Meta, MetaNameValue};

pub fn doc_string(attrs: &[Attribute]) -> String {
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
