use heck::ToSnakeCase;
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
    let handled_enum_ident = format_ident!("{enum_ident}Handled");
    let selector_enum_ident = format_ident!("{enum_ident}Selector");
    let hooks_struct_ident = format_ident!("{enum_ident}Hooks");
    let variants = data_enum.variants.into_iter().collect::<Vec<_>>();

    let mut wrapper_variants = Vec::new();
    let mut handled_variants = Vec::new();
    let mut selector_variants = Vec::new();
    let mut metadata_arms = Vec::new();
    let mut handled_metadata_arms = Vec::new();
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
    let mut handled_into_tool_result_arms = Vec::new();
    let mut handled_from_impls = Vec::new();
    let mut hooks_inner_fields = Vec::new();
    let mut hooks_field_inits = Vec::new();
    let mut hooks_with_methods = Vec::new();
    let mut hooks_register_methods = Vec::new();
    let mut hooks_dispatch_methods = Vec::new();
    let mut call_hook_arms = Vec::new();

    for (index, variant) in variants.into_iter().enumerate() {
        let variant_ident = variant.ident;
        let method_ident = format_ident!("{}", variant_ident.to_string().to_snake_case());
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
        let handled_variant_ty = quote! {
            ::lutum::HandledTool<#input_ty, <#input_ty as ::lutum::ToolInput>::Output>
        };
        let tool_name = quote! { <#input_ty as ::lutum::ToolInput>::NAME };
        let output_ty = quote! { <#input_ty as ::lutum::ToolInput>::Output };

        wrapper_variants.push(quote! { #variant_ident(#wrapper_ident) });
        handled_variants.push(quote! { #variant_ident(#handled_variant_ty) });
        selector_variants.push(quote! { #variant_ident });
        metadata_arms.push(quote! { Self::#variant_ident(inner) => &inner.metadata });
        handled_metadata_arms.push(quote! { Self::#variant_ident(inner) => inner.metadata() });
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
        handled_into_tool_result_arms.push(quote! {
            Self::#variant_ident(inner) => inner.into_tool_result()
        });
        handled_from_impls.push(quote! {
            impl From<#handled_variant_ty> for #handled_enum_ident {
                fn from(value: #handled_variant_ty) -> Self {
                    Self::#variant_ident(value)
                }
            }
        });
        let with_fn_ident = format_ident!("with_{}", method_ident);
        let register_fn_ident = format_ident!("register_{}", method_ident);
        let hook_fn_ty = quote! {
            ::std::option::Option<
                ::std::sync::Arc<
                    dyn Fn(
                        &::lutum::ToolMetadata,
                        &#input_ty,
                    ) -> ::std::pin::Pin<
                        ::std::boxed::Box<
                            dyn ::std::future::Future<
                                Output = ::std::option::Option<#output_ty>
                            > + ::std::marker::Send + 'static
                        >
                    >
                    + ::std::marker::Send
                    + ::std::marker::Sync
                >
            >
        };
        hooks_inner_fields.push(quote! {
            #method_ident: #hook_fn_ty
        });
        hooks_field_inits.push(quote! {
            #method_ident: ::std::option::Option::None
        });
        let registration_body = quote! {
            let wrapped = ::std::sync::Arc::new(
                move |metadata: &::lutum::ToolMetadata, input: &#input_ty| -> ::std::pin::Pin<::std::boxed::Box<dyn ::std::future::Future<Output = ::std::option::Option<#output_ty>> + ::std::marker::Send + 'static>> {
                    ::std::boxed::Box::pin(hook(metadata, input))
                }
            );
            if self.#method_ident.replace(wrapped).is_some() {
                ::tracing::warn!(
                    slot = "tool_hook",
                    "singleton hook registration overwritten; last registered hook wins"
                );
            }
        };
        hooks_with_methods.push(quote! {
            #[allow(dead_code)]
            pub fn #with_fn_ident<__F, __Fut>(
                mut self,
                hook: __F,
            ) -> Self
            where
                __F: Fn(&::lutum::ToolMetadata, &#input_ty) -> __Fut + ::std::marker::Send + ::std::marker::Sync + 'static,
                __Fut: ::std::future::Future<Output = ::std::option::Option<#output_ty>> + ::std::marker::Send + 'static,
            {
                #registration_body
                self
            }
        });
        hooks_register_methods.push(quote! {
            #[allow(dead_code)]
            pub fn #register_fn_ident<__F, __Fut>(
                &mut self,
                hook: __F,
            ) -> &mut Self
            where
                __F: Fn(&::lutum::ToolMetadata, &#input_ty) -> __Fut + ::std::marker::Send + ::std::marker::Sync + 'static,
                __Fut: ::std::future::Future<Output = ::std::option::Option<#output_ty>> + ::std::marker::Send + 'static,
            {
                #registration_body
                self
            }
        });
        hooks_dispatch_methods.push(quote! {
            #[allow(dead_code)]
            pub async fn #method_ident(
                &self,
                metadata: &::lutum::ToolMetadata,
                input: &#input_ty,
            ) -> ::std::option::Option<#output_ty> {
                use ::tracing::Instrument as _;
                let span = ::tracing::info_span!("lutum_hook", name = "tool_hook");
                async move {
                    match &self.#method_ident {
                        ::std::option::Option::Some(hook) => hook(metadata, input).await,
                        ::std::option::Option::None => ::std::option::Option::None,
                    }
                }
                .instrument(span)
                .await
            }
        });
        call_hook_arms.push(quote! {
            Self::#variant_ident(call) => {
                match hooks.#method_ident(&call.metadata, &call.input).await {
                    ::std::option::Option::Some(output) => {
                        ::lutum::ToolHookOutcome::Handled(
                            #handled_enum_ident::#variant_ident(call.handled(output))
                        )
                    }
                    ::std::option::Option::None => {
                        ::lutum::ToolHookOutcome::Unhandled(Self::#variant_ident(call))
                    }
                }
            }
        });
    }

    quote! {
        #[derive(Clone, Debug, Eq, PartialEq)]
        #vis enum #call_enum_ident {
            #(#wrapper_variants,)*
        }

        #[derive(Clone, Debug)]
        #vis enum #handled_enum_ident {
            #(#handled_variants,)*
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

        #[derive(::std::clone::Clone)]
        #vis struct #hooks_struct_ident {
            #(#hooks_inner_fields,)*
        }

        impl ::std::default::Default for #hooks_struct_ident {
            fn default() -> Self {
                Self {
                    #(#hooks_field_inits,)*
                }
            }
        }

        #[allow(dead_code)]
        impl #hooks_struct_ident {
            pub fn new() -> Self {
                Self::default()
            }

            #(#hooks_with_methods)*
            #(#hooks_register_methods)*
            #(#hooks_dispatch_methods)*
        }

        impl #handled_enum_ident {
            pub fn metadata(&self) -> &::lutum::ToolMetadata {
                match self {
                    #(#handled_metadata_arms,)*
                }
            }
        }

        impl ::lutum::IntoToolResult for #handled_enum_ident {
            fn into_tool_result(self) -> Result<::lutum::ToolResult, ::lutum::ToolResultError> {
                match self {
                    #(#handled_into_tool_result_arms,)*
                }
            }
        }

        #(#handled_from_impls)*

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

            pub async fn hook(
                self,
                hooks: &#hooks_struct_ident,
            ) -> ::lutum::ToolHookOutcome<Self, #handled_enum_ident> {
                match self {
                    #(#call_hook_arms,)*
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
