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
