// A resolver for the absolute imports of an extension's frontend.
//
// An extension's modules reach the interface's own by the path the browser
// serves them at -- `/core/i18n.js`, `/features/screens.js` -- because that is
// what the page gives them (see api/extensions.py). Node reads an import that
// starts with `/` as a path from the filesystem root, so without this hook a
// battery over an extension's screen dies on its first import.
//
// A battery registers it before importing the extension, with
//
//     import { registerExtensionImports } from './extension_imports.mjs';
//     registerExtensionImports();
//
// and then imports by file URL (`new URL(...)`), which this hook leaves alone.
// `register` loads this same file again, in node's loader thread, and calls
// the `resolve` it exports.
import { register } from "node:module";
import { fileURLToPath, pathToFileURL } from "node:url";
import path from "node:path";

const FRONTEND = path.join(
    path.dirname(path.dirname(path.dirname(fileURLToPath(import.meta.url)))),
    "frontend",
);

// The folders the interface serves at the root of the page. A specifier that
// starts with one of them is the interface's own module; anything else that
// starts with `/` is a real path, and is left to node.
const SERVED = ["/core/", "/features/", "/components/", "/design/", "/vendor/"];

export function resolve(specifier, context, next) {
    if (SERVED.some(folder => specifier.startsWith(folder))) {
        return next(
            pathToFileURL(path.join(FRONTEND, specifier.slice(1))).href,
            context,
        );
    }
    return next(specifier, context);
}

export function registerExtensionImports() {
    register(import.meta.url);
}
