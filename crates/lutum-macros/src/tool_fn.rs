use heck::ToUpperCamelCase;
use quote::{format_ident, quote};
use syn::{FnArg, GenericArgument, ItemFn, Pat, PatIdent, PathArguments, ReturnType, Type};

use crate::{ToolFnArgs, utils::doc_string};

pub fn expand_tool_fn(item_fn: ItemFn, args: ToolFnArgs) -> proc_macro2::TokenStream {
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
            pub fn tool_result(
                metadata: ::lutum::ToolMetadata,
                output: #output_ty,
            ) -> Result<::lutum::ToolResult, ::lutum::ToolResultError> {
                <Self as ::lutum::ToolInput>::tool_result(metadata, output)
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

            pub fn complete(
                self,
                output: #output_ty,
            ) -> Result<::lutum::ToolResult, ::lutum::ToolResultError> {
                #input_ident::tool_result(self.metadata, output)
            }

            pub fn handled(
                self,
                output: #output_ty,
            ) -> ::lutum::HandledTool<#input_ident, #output_ty> {
                ::lutum::HandledTool::new(self.metadata, self.input, output)
            }

            pub async fn call(
                self,
                #(#call_method_args),*
            ) -> Result<::lutum::ToolResult, ::lutum::ToolExecutionError<#error_ty>> {
                let output = #fn_ident(#(#fn_call_args),*)
                    .await
                    .map_err(::lutum::ToolExecutionError::Execute)?;
                #input_ident::tool_result(self.metadata, output)
                    .map_err(::lutum::ToolExecutionError::ToolResult)
            }
        }

        impl From<#call_ident> for #input_ident {
            fn from(value: #call_ident) -> Self {
                value.into_input()
            }
        }
    }
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
