// sphinx-docsearch ships DocSearch as a UMD bundle. require.js is on every
// page (the Plotly notebook outputs need it), so the bundle would register as
// an anonymous AMD module instead of defining window.docsearch, and
// docsearch_config.js would then fail. Hide `define` while docsearch.js runs;
// docsearch-amd-on.js restores it. Both are deferred and ordered by priority
// around the extension's scripts in docs/conf.py.
window.rossDefineBackup = window.define;
window.define = undefined;
