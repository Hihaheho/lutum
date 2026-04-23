use anyhow::{Context, Result, anyhow};
use eure::value::Text;
use eure::FromEure;
use eure_mark::{PageRenderer, parse_article_file};
use maud::{DOCTYPE, Markup, PreEscaped, html};
use std::fs;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Navigation document structs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, FromEure)]
#[eure(crate = ::eure::document)]
struct NavDocument {
    #[eure(default)]
    title: Option<String>,
    #[eure(default)]
    groups: Vec<NavGroup>,
}

#[derive(Debug, Clone, FromEure)]
#[eure(crate = ::eure::document)]
struct NavGroup {
    title: String,
    #[eure(default)]
    description: Option<String>,
    #[eure(default)]
    pages: Vec<NavPage>,
}

#[derive(Debug, Clone, FromEure)]
#[eure(crate = ::eure::document)]
struct NavPage {
    path: String,
    label: String,
}

// ---------------------------------------------------------------------------
// ADR document structs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, FromEure)]
#[eure(crate = ::eure::document)]
struct AdrDocument {
    id: String,
    title: String,
    status: AdrStatus,
    #[eure(rename = "decision-date")]
    decision_date: String,
    #[eure(default)]
    tags: Vec<String>,
    context: Text,
    decision: Text,
    consequences: Text,
    #[eure(rename = "alternatives-considered", default)]
    alternatives_considered: Vec<Text>,
    #[eure(rename = "related-adrs", default)]
    related_adrs: Vec<String>,
    #[eure(rename = "related-links", default)]
    related_links: Vec<String>,
}

#[derive(Debug, Clone, FromEure)]
#[eure(crate = ::eure::document)]
enum AdrStatus {
    #[eure(rename = "proposed")]
    Proposed,
    #[eure(rename = "accepted")]
    Accepted,
    #[eure(rename = "rejected")]
    Rejected { reason: Text },
    #[eure(rename = "deprecated")]
    Deprecated { reason: Text },
    #[eure(rename = "superseded")]
    Superseded {
        #[eure(rename = "superseded_by")]
        superseded_by: String,
    },
}

impl AdrStatus {
    fn label(&self) -> &'static str {
        match self {
            Self::Proposed => "proposed",
            Self::Accepted => "accepted",
            Self::Rejected { .. } => "rejected",
            Self::Deprecated { .. } => "deprecated",
            Self::Superseded { .. } => "superseded",
        }
    }
}

// ---------------------------------------------------------------------------
// Parsed guide page
// ---------------------------------------------------------------------------

struct GuidePage {
    public_path: String,
    title: String,
    description: String,
    html_body: String,
}

// ---------------------------------------------------------------------------
// Main build entry point
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let docs_dir = Path::new("../docs");
    let dist_dir = Path::new("dist");

    fs::create_dir_all(dist_dir.join("docs/adrs"))?;
    fs::create_dir_all(dist_dir.join("styles"))?;

    // Parse navigation
    let nav = parse_nav(docs_dir)?;

    // Initialize eure-mark renderer (once — loads syntax highlight data)
    let renderer = PageRenderer::new().context("initialise PageRenderer")?;
    let css = format!("{}\n{}", renderer.css(), SHELL_CSS);
    fs::write(dist_dir.join("styles/docs.css"), &css)?;

    // Render guide pages
    let guide_pages = collect_guide_pages(docs_dir, &renderer)?;

    // Parse ADR pages
    let adrs = collect_adrs(docs_dir)?;

    // Write root redirect
    fs::write(dist_dir.join("index.html"), redirect_html("/docs/"))?;

    // Write guide pages
    for page in &guide_pages {
        let markup = render_shell(&nav, &page.public_path, &page.title, &page.description, PreEscaped(page.html_body.clone()));
        write_public_path(dist_dir, &page.public_path, markup.into_string())?;
    }

    // Write ADR index
    let adr_index = render_adr_index(&nav, &adrs);
    write_public_path(dist_dir, "/docs/adrs/", adr_index.into_string())?;

    // Write individual ADR pages
    for adr in &adrs {
        let body = render_adr_body(&renderer, adr)?;
        let public_path = format!("/docs/adrs/{}", adr.id);
        let markup = render_shell(
            &nav,
            &public_path,
            &adr.title,
            &format!("ADR {} — {}", adr.id, adr.title),
            body,
        );
        write_public_path(dist_dir, &public_path, markup.into_string())?;
    }

    // Write llms.txt
    let llms_txt = generate_llms_txt(&nav, &guide_pages, &adrs);
    fs::write(dist_dir.join("llms.txt"), llms_txt)?;
    println!("  wrote dist/llms.txt");

    println!("Built → dist/");
    Ok(())
}

// ---------------------------------------------------------------------------
// llms.txt generation (https://llmstxt.org)
// ---------------------------------------------------------------------------

fn generate_llms_txt(nav: &NavDocument, guide_pages: &[GuidePage], adrs: &[AdrDocument]) -> String {
    const BASE_URL: &str = "https://lutum.dev";
    let mut out = String::new();

    out.push_str("# Lutum\n\n");
    out.push_str("> A composable, streaming LLM toolkit for Rust. Lutum provides typed turns, structured output, tool use, session transcript management, and an eval framework — without owning the agent loop.\n\n");
    out.push_str("Lutum is an open-source Rust crate. This documentation covers the public API, key concepts, and usage patterns.\n\n");

    // Group guide pages by nav group
    let mut emitted_nav = false;
    for group in &nav.groups {
        if group.pages.is_empty() {
            continue;
        }
        if !emitted_nav {
            out.push_str("## Docs\n\n");
            emitted_nav = true;
        }
        out.push_str(&format!("### {}\n\n", group.title));
        for nav_page in &group.pages {
            // Find the matching guide page for its description
            let description = guide_pages
                .iter()
                .find(|p| p.public_path == nav_page.path || p.public_path.trim_end_matches('/') == nav_page.path.trim_end_matches('/'))
                .map(|p| p.description.as_str())
                .unwrap_or("");
            let url = format!("{}{}", BASE_URL, nav_page.path);
            if description.is_empty() {
                out.push_str(&format!("- [{}]({})\n", nav_page.label, url));
            } else {
                out.push_str(&format!("- [{}]({}): {}\n", nav_page.label, url, description));
            }
        }
        out.push('\n');
    }

    // ADR section
    if !adrs.is_empty() {
        out.push_str("## ADRs\n\n");
        for adr in adrs {
            let url = format!("{}/docs/adrs/{}", BASE_URL, adr.id);
            out.push_str(&format!("- [{}]({}): {} ({})\n", adr.title, url, adr.id, adr.status.label()));
        }
        out.push('\n');
    }

    // Optional section
    out.push_str("## Optional\n\n");
    out.push_str("- [GitHub Repository](https://github.com/Hihaheho/lutum): Source code, issue tracker, and contribution guide\n");
    out.push_str("- [crates.io](https://crates.io/crates/lutum): Published crate versions and download stats\n");

    out
}

// ---------------------------------------------------------------------------
// Parsing helpers
// ---------------------------------------------------------------------------

fn parse_nav(docs_dir: &Path) -> Result<NavDocument> {
    let path = docs_dir.join("_nav.eure");
    let content = fs::read_to_string(&path)
        .with_context(|| format!("reading {}", path.display()))?;
    eure::parse_content(&content, path).map_err(|e| anyhow!("{e}"))
}

fn collect_guide_pages(docs_dir: &Path, renderer: &PageRenderer) -> Result<Vec<GuidePage>> {
    let mut pages = Vec::new();
    let mut entries: Vec<_> = fs::read_dir(docs_dir)?
        .filter_map(|e| e.ok())
        .collect();
    entries.sort_by_key(|e| e.path());

    for entry in entries {
        let path = entry.path();
        if path.extension().map_or(false, |e| e == "eure") {
            let stem = path.file_stem().unwrap().to_str().unwrap().to_string();
            if stem == "_nav" {
                continue;
            }
            let article = parse_article_file(&path)
                .with_context(|| format!("parsing {}", path.display()))?;
            let rendered = renderer
                .render_article(&article)
                .with_context(|| format!("rendering {}", path.display()))?;
            let public_path = if stem == "index" {
                "/docs/".to_string()
            } else {
                format!("/docs/{stem}")
            };
            pages.push(GuidePage {
                public_path,
                title: rendered.title,
                description: rendered.description,
                html_body: rendered.html,
            });
        }
    }
    Ok(pages)
}

fn collect_adrs(docs_dir: &Path) -> Result<Vec<AdrDocument>> {
    let adrs_dir = docs_dir.join("adrs");
    if !adrs_dir.exists() {
        return Ok(Vec::new());
    }
    let mut entries: Vec<_> = fs::read_dir(&adrs_dir)?
        .filter_map(|e| e.ok())
        .collect();
    entries.sort_by_key(|e| e.path());

    let mut adrs = Vec::new();
    for entry in entries {
        let path = entry.path();
        if path.extension().map_or(false, |e| e == "eure") {
            let content = fs::read_to_string(&path)
                .with_context(|| format!("reading {}", path.display()))?;
            let adr: AdrDocument = eure::parse_content(&content, path.clone())
                .map_err(|e| anyhow!("{e}"))?;
            adrs.push(adr);
        }
    }
    Ok(adrs)
}

// ---------------------------------------------------------------------------
// Output helpers
// ---------------------------------------------------------------------------

/// Map a public URL path to a dist-relative file path and write the content.
fn write_public_path(dist_dir: &Path, public_path: &str, content: String) -> Result<()> {
    let rel = public_path.trim_start_matches('/');
    let file_path: PathBuf = if public_path.ends_with('/') {
        dist_dir.join(rel).join("index.html")
    } else {
        dist_dir.join(format!("{rel}.html"))
    };
    if let Some(parent) = file_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&file_path, content)
        .with_context(|| format!("writing {}", file_path.display()))?;
    println!("  wrote {}", file_path.display());
    Ok(())
}

fn redirect_html(target: &str) -> String {
    html! {
        (DOCTYPE)
        html {
            head {
                meta charset="UTF-8" {}
                meta http-equiv="refresh" content=(format!("0; url={target}")) {}
                title { "Lutum Documentation" }
            }
            body {
                p { a href=(target) { "Go to documentation" } }
            }
        }
    }
    .into_string()
}

// ---------------------------------------------------------------------------
// Site shell template
// ---------------------------------------------------------------------------

fn render_shell(
    nav: &NavDocument,
    current_path: &str,
    title: &str,
    description: &str,
    body: Markup,
) -> Markup {
    let site_title = if title == "Lutum Documentation" {
        "Lutum Docs".to_string()
    } else {
        format!("{title} — Lutum Docs")
    };

    html! {
        (DOCTYPE)
        html lang="en" {
            head {
                meta charset="UTF-8" {}
                meta name="viewport" content="width=device-width, initial-scale=1.0" {}
                title { (site_title) }
                meta name="description" content=(description) {}
                link rel="stylesheet" href="/styles/docs.css" {}
            }
            body {
                header class="site-header" {
                    a class="brand" href="/docs/" { "Lutum" }
                    nav class="header-nav" {
                        a href="https://docs.rs/lutum" target="_blank" rel="noopener" { "API Docs" }
                        a href="https://github.com/Hihaheho/lutum" target="_blank" rel="noopener" { "GitHub" }
                    }
                }
                div class="layout" {
                    aside class="sidebar" {
                        (render_sidebar(nav, current_path))
                    }
                    main class="content" {
                        (body)
                    }
                }
                footer class="site-footer" {
                    p { "Lutum documentation · " a href="https://github.com/Hihaheho/lutum/blob/main/LICENSE" { "MIT/Apache-2.0" } }
                }
            }
        }
    }
}

fn render_sidebar(nav: &NavDocument, current_path: &str) -> Markup {
    html! {
        nav class="sidebar-nav" {
            @if let Some(title) = &nav.title {
                div class="nav-title" { (title) }
            }
            @for group in &nav.groups {
                div class="nav-group" {
                    div class="nav-group-title" title=[group.description.as_deref()] { (group.title) }
                    @for page in &group.pages {
                        @let is_active = is_active_path(&page.path, current_path);
                        a
                            href=(page.path)
                            class=(if is_active { "nav-link active" } else { "nav-link" })
                        {
                            (page.label)
                        }
                    }
                }
            }
            // ADR section always present if there are ADRs
            div class="nav-group" {
                div class="nav-group-title" { "ADRs" }
                a
                    href="/docs/adrs/"
                    class=(if current_path.starts_with("/docs/adrs") { "nav-link active" } else { "nav-link" })
                {
                    "Decision Records"
                }
            }
        }
    }
}

fn is_active_path(nav_path: &str, current_path: &str) -> bool {
    let nav = nav_path.trim_end_matches('/');
    let cur = current_path.trim_end_matches('/');
    nav == cur
}

// ---------------------------------------------------------------------------
// ADR rendering
// ---------------------------------------------------------------------------

fn render_adr_body(renderer: &PageRenderer, adr: &AdrDocument) -> Result<Markup> {
    let context_html = renderer
        .render_text_fragment(&adr.context)
        .map_err(|e| anyhow!("{e:?}"))?;
    let decision_html = renderer
        .render_text_fragment(&adr.decision)
        .map_err(|e| anyhow!("{e:?}"))?;
    let consequences_html = renderer
        .render_text_fragment(&adr.consequences)
        .map_err(|e| anyhow!("{e:?}"))?;

    let mut alt_htmls = Vec::new();
    for alt in &adr.alternatives_considered {
        let h = renderer
            .render_text_fragment(alt)
            .map_err(|e| anyhow!("{e:?}"))?;
        alt_htmls.push(h);
    }

    Ok(html! {
        article class="adr-page" {
            header class="adr-header" {
                div class="adr-meta" {
                    span class="adr-id" { (adr.id) }
                    span class=(format!("adr-status adr-status-{}", adr.status.label())) {
                        (adr.status.label())
                        @match &adr.status {
                            AdrStatus::Rejected { reason } => {
                                span class="adr-status-reason" { " — " (reason.content.trim()) }
                            }
                            AdrStatus::Deprecated { reason } => {
                                span class="adr-status-reason" { " — " (reason.content.trim()) }
                            }
                            AdrStatus::Superseded { superseded_by } => {
                                span class="adr-status-reason" { " → " (superseded_by) }
                            }
                            _ => {}
                        }
                    }
                    span class="adr-date" { (adr.decision_date.trim_matches('`')) }
                }
                h1 { (adr.title) }
                @if !adr.tags.is_empty() {
                    div class="adr-tags" {
                        @for tag in &adr.tags {
                            span class="tag" { (tag) }
                        }
                    }
                }
            }

            section class="adr-section" {
                h2 { "Context" }
                (PreEscaped(context_html))
            }

            section class="adr-section" {
                h2 { "Decision" }
                (PreEscaped(decision_html))
            }

            section class="adr-section" {
                h2 { "Consequences" }
                (PreEscaped(consequences_html))
            }

            @if !alt_htmls.is_empty() {
                section class="adr-section" {
                    h2 { "Alternatives Considered" }
                    @for html_content in &alt_htmls {
                        (PreEscaped(html_content))
                    }
                }
            }

            @if !adr.related_adrs.is_empty() {
                section class="adr-section" {
                    h2 { "Related ADRs" }
                    ul {
                        @for id in &adr.related_adrs {
                            li { a href=(format!("/docs/adrs/{id}")) { (id) } }
                        }
                    }
                }
            }

            @if !adr.related_links.is_empty() {
                section class="adr-section" {
                    h2 { "Related Links" }
                    ul {
                        @for link in &adr.related_links {
                            li { a href=(link) target="_blank" rel="noopener" { (link) } }
                        }
                    }
                }
            }
        }
    })
}

fn render_adr_index(nav: &NavDocument, adrs: &[AdrDocument]) -> Markup {
    let body = html! {
        div class="adr-index" {
            h1 { "Architecture Decision Records" }
            p { "These records document significant design decisions made in the lutum project." }
            @if adrs.is_empty() {
                p { "No ADRs yet." }
            } @else {
                div class="adr-list" {
                    @for adr in adrs {
                        a class="adr-card" href=(format!("/docs/adrs/{}", adr.id)) {
                            div class="adr-card-header" {
                                span class="adr-id" { (adr.id) }
                                span class=(format!("adr-status adr-status-{}", adr.status.label())) {
                                    (adr.status.label())
                                }
                            }
                            div class="adr-card-title" { (adr.title) }
                            @if !adr.tags.is_empty() {
                                div class="adr-tags" {
                                    @for tag in &adr.tags {
                                        span class="tag" { (tag) }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    };

    render_shell(
        nav,
        "/docs/adrs/",
        "Architecture Decision Records",
        "Design decisions documented as Architecture Decision Records for the lutum project.",
        body,
    )
}

// ---------------------------------------------------------------------------
// Site shell CSS
// ---------------------------------------------------------------------------

const SHELL_CSS: &str = r#"
/* ---- Catppuccin Mocha palette ---- */
:root {
  --base:    #1e1e2e;
  --mantle:  #181825;
  --crust:   #11111b;
  --surface0: #313244;
  --surface1: #45475a;
  --surface2: #585b70;
  --text:    #cdd6f4;
  --subtext0: #a6adc8;
  --subtext1: #bac2de;
  --blue:    #89b4fa;
  --lavender:#b4befe;
  --sapphire:#74c7ec;
  --green:   #a6e3a1;
  --yellow:  #f9e2af;
  --peach:   #fab387;
  --red:     #f38ba8;
  --mauve:   #cba6f7;
}

/* ---- Reset / base ---- */
*, *::before, *::after { box-sizing: border-box; }

html {
  font-size: 16px;
  line-height: 1.6;
  scroll-behavior: smooth;
}

body {
  margin: 0;
  background: var(--base);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  display: flex;
  flex-direction: column;
  min-height: 100vh;
}

a { color: var(--blue); text-decoration: none; }
a:hover { text-decoration: underline; }

/* ---- Site header ---- */
.site-header {
  background: var(--mantle);
  border-bottom: 1px solid var(--surface0);
  padding: 0 1.5rem;
  height: 56px;
  display: flex;
  align-items: center;
  gap: 2rem;
  position: sticky;
  top: 0;
  z-index: 100;
}

.brand {
  font-weight: 700;
  font-size: 1.125rem;
  color: var(--lavender);
  letter-spacing: 0.01em;
}
.brand:hover { text-decoration: none; color: var(--mauve); }

.header-nav {
  margin-left: auto;
  display: flex;
  gap: 1.25rem;
}
.header-nav a { color: var(--subtext1); font-size: 0.875rem; }
.header-nav a:hover { color: var(--text); }

/* ---- Page layout ---- */
.layout {
  display: flex;
  flex: 1;
}

/* ---- Sidebar ---- */
.sidebar {
  width: 240px;
  flex-shrink: 0;
  background: var(--mantle);
  border-right: 1px solid var(--surface0);
  padding: 1.25rem 0;
  position: sticky;
  top: 56px;
  height: calc(100vh - 56px);
  overflow-y: auto;
}

.sidebar-nav { padding: 0 0.5rem; }

.nav-group { margin-bottom: 1.25rem; }

.nav-group-title {
  font-size: 0.7rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--subtext0);
  padding: 0 0.75rem;
  margin-bottom: 0.25rem;
}

.nav-link {
  display: block;
  padding: 0.35rem 0.75rem;
  border-radius: 6px;
  color: var(--subtext1);
  font-size: 0.875rem;
  transition: background 0.1s, color 0.1s;
}
.nav-link:hover { background: var(--surface0); color: var(--text); text-decoration: none; }
.nav-link.active { background: var(--surface0); color: var(--blue); font-weight: 500; }

/* ---- Content area ---- */
.content {
  flex: 1;
  min-width: 0;
  padding: 2.5rem clamp(1rem, 4vw, 3rem);
  max-width: 860px;
}

/* ---- Prose typography ---- */
.content h1 { font-size: 2rem; font-weight: 700; margin-top: 0; color: var(--lavender); }
.content h2 { font-size: 1.375rem; font-weight: 600; margin-top: 2.5rem; color: var(--text); border-bottom: 1px solid var(--surface0); padding-bottom: 0.25rem; }
.content h3 { font-size: 1.125rem; font-weight: 600; margin-top: 1.75rem; color: var(--subtext1); }
.content h4, .content h5, .content h6 { font-size: 1rem; margin-top: 1.25rem; color: var(--subtext0); }

.content p { margin: 0.75rem 0; }
.content ul, .content ol { padding-left: 1.5rem; }
.content li { margin: 0.25rem 0; }

.content table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.9rem; }
.content th, .content td { border: 1px solid var(--surface1); padding: 0.5rem 0.75rem; text-align: left; }
.content th { background: var(--surface0); color: var(--subtext1); font-weight: 600; }
.content tr:hover td { background: var(--mantle); }

.content code { background: var(--surface0); padding: 0.15em 0.4em; border-radius: 4px; font-size: 0.875em; font-family: "JetBrains Mono", "Fira Code", Consolas, monospace; color: var(--peach); }
.content pre { background: var(--mantle); border: 1px solid var(--surface0); border-radius: 8px; padding: 1rem 1.25rem; overflow-x: auto; margin: 1rem 0; }
.content pre code { background: none; padding: 0; color: inherit; font-size: 0.875rem; }

/* ---- ADR styles ---- */
.adr-header { margin-bottom: 2rem; }
.adr-meta { display: flex; gap: 1rem; align-items: center; margin-bottom: 0.75rem; font-size: 0.875rem; }
.adr-id { color: var(--subtext0); font-family: monospace; }
.adr-date { color: var(--subtext0); }

.adr-status { padding: 0.2rem 0.6rem; border-radius: 12px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; }
.adr-status-accepted { background: color-mix(in srgb, var(--green) 20%, transparent); color: var(--green); }
.adr-status-proposed { background: color-mix(in srgb, var(--yellow) 20%, transparent); color: var(--yellow); }
.adr-status-rejected { background: color-mix(in srgb, var(--red) 20%, transparent); color: var(--red); }
.adr-status-deprecated { background: color-mix(in srgb, var(--surface2) 40%, transparent); color: var(--subtext0); }
.adr-status-superseded { background: color-mix(in srgb, var(--peach) 20%, transparent); color: var(--peach); }

.adr-section { margin: 2rem 0; }
.adr-section h2 { margin-top: 0; }

.adr-tags, .tag { display: inline-flex; flex-wrap: wrap; gap: 0.4rem; }
.tag { background: var(--surface0); color: var(--subtext1); padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.75rem; }

.adr-list { display: grid; gap: 1rem; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); margin-top: 1.5rem; }
.adr-card { display: block; background: var(--mantle); border: 1px solid var(--surface0); border-radius: 8px; padding: 1rem 1.25rem; color: var(--text); transition: border-color 0.15s; }
.adr-card:hover { border-color: var(--blue); text-decoration: none; }
.adr-card-header { display: flex; gap: 0.75rem; align-items: center; margin-bottom: 0.5rem; }
.adr-card-title { font-weight: 500; font-size: 0.9375rem; }

/* ---- Footer ---- */
.site-footer {
  background: var(--mantle);
  border-top: 1px solid var(--surface0);
  padding: 1rem 1.5rem;
  font-size: 0.8125rem;
  color: var(--subtext0);
  text-align: center;
}
.site-footer a { color: var(--subtext0); }
.site-footer a:hover { color: var(--text); }

/* ---- Responsive ---- */
@media (max-width: 768px) {
  .layout { flex-direction: column; }
  .sidebar { width: 100%; height: auto; position: static; border-right: none; border-bottom: 1px solid var(--surface0); }
  .content { padding: 1.5rem 1rem; }
}
"#;
