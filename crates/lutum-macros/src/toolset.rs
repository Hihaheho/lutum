use quote::{format_ident, quote};
use syn::{Data, DeriveInput, Fields, FieldsNamed, FieldsUnnamed, Ident, Type};

pub fn expand_toolset(input: DeriveInput) -> proc_macro2::TokenStream {
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

fn wrapper_ident_for_type(ty: &Type) -> Ident {
    if let Type::Path(path) = ty {
        let ident = &path.path.segments.last().expect("type path").ident;
        format_ident!("{ident}Call")
    } else {
        panic!("Toolset variant payloads must be path types");
    }
}
