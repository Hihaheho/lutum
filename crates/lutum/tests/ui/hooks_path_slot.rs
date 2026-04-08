mod slots {
    #[lutum::def_hook(singleton)]
    pub async fn select_label(_ctx: &lutum::Lutum, default: String) -> String {
        default
    }
}

#[lutum::hooks]
struct MyHooks {
    label: slots::SelectLabel,
}

fn main() {
    let _ = MyHooks::new();
}
