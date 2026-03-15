use heck::{ToSnakeCase, ToUpperCamelCase};
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    Attribute, Data, DeriveInput, Expr, Fields, FieldsNamed, FieldsUnnamed, FnArg, GenericArgument,
    Ident, ItemEnum, ItemFn, ItemStruct, Lit, Meta, MetaNameValue, Pat, PatIdent, PathArguments,
    ReturnType, Token, Type, TypePath, parse::Parse, parse_macro_input, punctuated::Punctuated,
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

        impl ::agents::ToolInput for #ident {
            type Output = #output;

            const NAME: &'static str = #name;
            const DESCRIPTION: &'static str = #description;
        }

        impl #ident {
            pub fn tool_use(
                metadata: ::agents::ToolMetadata,
                output: #output,
            ) -> Result<::agents::ToolUse, ::agents::ToolUseError> {
                <Self as ::agents::ToolInput>::tool_use(metadata, output)
            }
        }

        #[derive(Clone, Debug, Eq, PartialEq)]
        #vis struct #call_ident {
            pub metadata: ::agents::ToolMetadata,
            pub input: #ident,
        }

        impl ::agents::toolset::ToolCallWrapper for #call_ident {
            fn metadata(&self) -> &::agents::ToolMetadata {
                &self.metadata
            }
        }

        impl #call_ident {
            pub fn tool_use(
                self,
                output: #output,
            ) -> Result<::agents::ToolUse, ::agents::ToolUseError> {
                #ident::tool_use(self.metadata, output)
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

        impl ::agents::ToolInput for #ident {
            type Output = #output;

            const NAME: &'static str = #name;
            const DESCRIPTION: &'static str = #description;
        }

        impl #ident {
            pub fn tool_use(
                metadata: ::agents::ToolMetadata,
                output: #output,
            ) -> Result<::agents::ToolUse, ::agents::ToolUseError> {
                <Self as ::agents::ToolInput>::tool_use(metadata, output)
            }
        }

        #[derive(Clone, Debug, Eq, PartialEq)]
        #vis struct #call_ident {
            pub metadata: ::agents::ToolMetadata,
            pub input: #ident,
        }

        impl ::agents::toolset::ToolCallWrapper for #call_ident {
            fn metadata(&self) -> &::agents::ToolMetadata {
                &self.metadata
            }
        }

        impl #call_ident {
            pub fn tool_use(
                self,
                output: #output,
            ) -> Result<::agents::ToolUse, ::agents::ToolUseError> {
                #ident::tool_use(self.metadata, output)
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

        impl ::agents::ToolInput for #input_ident {
            type Output = #output_ty;

            const NAME: &'static str = #tool_name;
            const DESCRIPTION: &'static str = #description;
        }

        impl #input_ident {
            pub fn tool_use(
                metadata: ::agents::ToolMetadata,
                output: #output_ty,
            ) -> Result<::agents::ToolUse, ::agents::ToolUseError> {
                <Self as ::agents::ToolInput>::tool_use(metadata, output)
            }
        }

        #[derive(Clone, Debug, Eq, PartialEq)]
        #vis struct #call_ident {
            pub metadata: ::agents::ToolMetadata,
            pub input: #input_ident,
        }

        impl ::agents::toolset::ToolCallWrapper for #call_ident {
            fn metadata(&self) -> &::agents::ToolMetadata {
                &self.metadata
            }
        }

        impl #call_ident {
            pub fn tool_use(
                self,
                output: #output_ty,
            ) -> Result<::agents::ToolUse, ::agents::ToolUseError> {
                #input_ident::tool_use(self.metadata, output)
            }

            pub async fn call(
                self,
                #(#call_method_args),*
            ) -> Result<::agents::ToolUse, ::agents::ToolExecutionError<#error_ty>> {
                let output = #fn_ident(#(#fn_call_args),*)
                    .await
                    .map_err(::agents::ToolExecutionError::Execute)?;
                #input_ident::tool_use(self.metadata, output)
                    .map_err(::agents::ToolExecutionError::ToolUse)
            }
        }
    }
}

fn expand_toolset(input: DeriveInput) -> proc_macro2::TokenStream {
    let enum_ident = input.ident.clone();
    let vis = input.vis.clone();
    let Data::Enum(data_enum) = input.data else {
        return syn::Error::new_spanned(input, "Toolset can only be derived for enums")
            .to_compile_error();
    };

    let call_enum_ident = format_ident!("{enum_ident}Call");
    let variants = data_enum.variants.into_iter().collect::<Vec<_>>();

    let mut wrapper_variants = Vec::new();
    let mut metadata_arms = Vec::new();
    let mut parse_arms = Vec::new();
    let mut defs = Vec::new();
    let mut supports = Vec::new();

    for variant in variants {
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

        wrapper_variants.push(quote! { #variant_ident(#wrapper_ident) });
        metadata_arms.push(quote! { Self::#variant_ident(inner) => &inner.metadata });
        defs.push(quote! { ::agents::ToolDef::for_input::<#input_ty>() });
        supports.push(quote! { impl ::agents::SupportsTool<#input_ty> for #enum_ident {} });
        parse_arms.push(quote! {
            <#input_ty as ::agents::ToolInput>::NAME => {
                let input = ::serde_json::from_str::<#input_ty>(arguments_json)
                    .map_err(|source| ::agents::ToolCallError::Deserialize {
                        name: name.to_string(),
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

        impl ::agents::Toolset for #enum_ident {
            type ToolCall = #call_enum_ident;

            fn definitions() -> &'static [::agents::ToolDef] {
                static DEFS: ::std::sync::OnceLock<::std::vec::Vec<::agents::ToolDef>> =
                    ::std::sync::OnceLock::new();
                DEFS.get_or_init(|| vec![#(#defs),*]).as_slice()
            }

            fn parse_tool_call(
                metadata: ::agents::ToolMetadata,
                name: &str,
                arguments_json: &str,
            ) -> Result<Self::ToolCall, ::agents::ToolCallError> {
                match name {
                    #(#parse_arms,)*
                    _ => Err(::agents::ToolCallError::UnknownTool {
                        name: name.to_string(),
                    }),
                }
            }
        }

        impl ::agents::toolset::ToolCallWrapper for #call_enum_ident {
            fn metadata(&self) -> &::agents::ToolMetadata {
                match self {
                    #(#metadata_arms,)*
                }
            }
        }

        #(#supports)*
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
