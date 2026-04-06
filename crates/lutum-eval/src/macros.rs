/// Register a hook slot so this probe receives its calls, inline in `register_hooks`.
///
/// Generates a dispatch closure that routes every hook invocation through the probe's
/// internal event loop. The probe (`Self`) must implement the corresponding
/// `Stateful*Hook` trait.
///
/// Internally wraps the closure in a `(PhantomData<fn() -> Lutum>, F)` tuple to select
/// the context-aware `Fn(Lutum, XxxArgs)` blanket impl rather than the context-free
/// `Fn(XxxArgs)` blanket impl.
///
/// # Example
/// ```ignore
/// use lutum_eval::register_probe_hook;
///
/// impl Probe for MyProbe {
///     fn register_hooks(&self, cx: &mut ProbeContext<'_, Self>) {
///         register_probe_hook!(cx, ValidateResponse);
///         register_probe_hook!(cx, RewriteLabel);
///     }
/// }
/// ```
#[macro_export]
macro_rules! register_probe_hook {
    ($cx:expr, $Slot:ident) => {
        $crate::paste::paste! {
            {
                let __dispatcher = $cx.dispatcher();
                $cx.update_hooks(|__h| __h.[< register_ $Slot:snake >](
                    (
                        ::std::marker::PhantomData::<fn() -> ::lutum::Lutum>,
                        move |__ctx: ::lutum::Lutum, __args: [< $Slot Args >]| {
                            let __dispatcher = __dispatcher.clone();
                            async move {
                                __dispatcher
                                    .dispatch(move |__probe| {
                                        ::std::boxed::Box::pin(async move {
                                            <Self as [< Stateful $Slot Hook >]>::call_mut(
                                                __probe, &__ctx, __args,
                                            )
                                            .await
                                        })
                                    })
                                    .await
                                    .expect("probe dispatcher alive")
                            }
                        },
                    )
                ));
            }
        }
    };
}
