# Algolia DocSearch — application and activation checklist

Status: **prepared, inactive**. The integration is committed but dormant: the
`sphinx_docsearch` extension only loads when the three `DOCSEARCH_*`
environment variables are set (see the guard in `docs/conf.py`). Until then,
builds use the built-in Sphinx search exactly as before.

This directory is listed in `exclude_patterns` in `conf.py`, so this note is
never published.

## 1. Apply for the DocSearch open-source program

Submit at <https://docsearch.algolia.com/apply/> with:

| Field | Value |
|---|---|
| Documentation site URL | `https://ross.readthedocs.io/en/stable/` |
| Repository | `https://github.com/petrobras/ross` |
| Email | a maintainer email (use the address that manages the RTD project) |

Open-source justification (paste/adapt):

> ROSS (Rotordynamic Open Source Software) is an MIT-like (Apache-2.0)
> licensed Python library for rotordynamic analysis, maintained by Petrobras
> and academic collaborators, published on PyPI as `ross-rotordynamics`. The
> documentation at ross.readthedocs.io is fully public, technical (theory,
> API reference, executable tutorials), and the site is non-commercial with
> no paywall. Repository: github.com/petrobras/ross (JOSS paper:
> doi:10.21105/joss.02120).

Eligibility notes: the docs are publicly available, the project is
open-source with an OSI-approved license, and the applicant must be a
maintainer — all satisfied.

## 2. Recommended crawler configuration

Ask for (or edit in the Algolia Crawler dashboard once access is granted):

- **Start URL / sitemap**: crawl only the stable version to avoid duplicate
  hits across versions:
  - `startUrls`: `["https://ross.readthedocs.io/en/stable/"]`
  - `sitemaps`: `["https://ross.readthedocs.io/sitemap.xml"]`
  - `discoveryPatterns` / `pathsToMatch`:
    `["https://ross.readthedocs.io/en/stable/**"]`
  - Exclude non-content pages: `genindex.html`, `py-modindex.html`,
    `search.html`, `_sources/**`.
- **Index name**: `ross` (this is the value for `DOCSEARCH_INDEX_NAME`).
- **Record selectors**: standard Sphinx selectors (sphinx-book-theme is a
  pydata-sphinx-theme derivative, so the article lives in `article.bd-article`):

  ```js
  recordProps: {
    lvl0: {
      selectors: ".bd-links__title, nav.bd-links li.current > a",
      defaultValue: "Documentation",
    },
    lvl1: "article.bd-article h1",
    lvl2: "article.bd-article h2",
    lvl3: "article.bd-article h3",
    lvl4: "article.bd-article h4",
    lvl5: "article.bd-article h5",
    content: "article.bd-article p, article.bd-article li, article.bd-article dd",
  }
  ```

- **Version facet** (optional, future-proofing if more versions are ever
  indexed): extract the RTD version from the URL into a `version` custom
  variable and set `docsearch_search_parameters = {"facetFilters": ["version:stable"]}`
  in `conf.py`.

## 3. When credentials arrive — activation steps

1. Read the Docs dashboard → project **ross** → **Admin → Environment
   Variables**, add (all three; the conf.py guard requires all of them):
   - `DOCSEARCH_APP_ID` — the Algolia application ID
   - `DOCSEARCH_API_KEY` — the **search-only** public API key (never the
     admin/write key; the search key is safe to expose in built HTML)
   - `DOCSEARCH_INDEX_NAME` — e.g. `ross`
2. Trigger a rebuild of the affected versions (env vars only apply to new
   builds). No code change is needed: `docs/conf.py` picks the variables up,
   loads `sphinx_docsearch`, and adds `_static/docsearch-ross.css` (ROSS
   design tokens for the modal, light + dark).
3. Disable the search UI injected by **Read the Docs Addons** so the two
   search interfaces don't fight over the `/` shortcut and the search button:
   RTD dashboard → **Settings → Addons → Search** → uncheck/disable.
   (DocSearch replaces in-page search; RTD's server-side search API remains
   available at `/search/` regardless.)
4. Verify on the built site:
   - the magnifier / `Ctrl-K` opens the Algolia modal, styled with ROSS
     tokens in both light and dark (toggle the theme button);
   - a build **without** the env vars (e.g. a local `make html`) still shows
     the classic Sphinx search — the guard keeps it as the fallback.

## 4. Related files

- `docs/conf.py` — env-var guard, activation comment block
- `docs/requirements.txt` — `sphinx-docsearch` pin
- `docs/_static/docsearch-ross.css` — `--docsearch-*` → ROSS token mapping
