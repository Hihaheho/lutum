use heck::ToSnakeCase;
use quote::{format_ident, quote};
use syn::{ItemEnum, ItemStruct};

use crate::{ToolInputArgs, utils::doc_string};

pub fn expand_tool_input_struct(item: ItemStruct, args: ToolInputArgs) -> proc_macro2::TokenStream {
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
            pub fn tool_result(
                metadata: ::lutum::ToolMetadata,
                output: #output,
            ) -> Result<::lutum::ToolResult, ::lutum::ToolResultError> {
                <Self as ::lutum::ToolInput>::tool_result(metadata, output)
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

            pub fn complete(
                self,
                output: #output,
            ) -> Result<::lutum::ToolResult, ::lutum::ToolResultError> {
                #ident::tool_result(self.metadata, output)
            }

            pub fn handled(
                self,
                output: #output,
            ) -> ::lutum::HandledTool<#ident, #output> {
                ::lutum::HandledTool::new(self.metadata, self.input, output)
            }
        }

        impl From<#call_ident> for #ident {
            fn from(value: #call_ident) -> Self {
                value.into_input()
            }
        }
    }
}

pub fn expand_tool_input_enum(item: ItemEnum, args: ToolInputArgs) -> proc_macro2::TokenStream {
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
            pub fn tool_result(
                metadata: ::lutum::ToolMetadata,
                output: #output,
            ) -> Result<::lutum::ToolResult, ::lutum::ToolResultError> {
                <Self as ::lutum::ToolInput>::tool_result(metadata, output)
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

            pub fn complete(
                self,
                output: #output,
            ) -> Result<::lutum::ToolResult, ::lutum::ToolResultError> {
                #ident::tool_result(self.metadata, output)
            }

            pub fn handled(
                self,
                output: #output,
            ) -> ::lutum::HandledTool<#ident, #output> {
                ::lutum::HandledTool::new(self.metadata, self.input, output)
            }
        }

        impl From<#call_ident> for #ident {
            fn from(value: #call_ident) -> Self {
                value.into_input()
            }
        }
    }
}
