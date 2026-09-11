# -*- coding: utf-8 -*-
"""The frontend is ES modules, and the boundaries between them hold.

Until this point the frontend was a 3030-line file in a single scope: ~164
top-level names, all global, none declaring what it depended on. The practical
cost was not aesthetic -- it was that the JS tests had to **cut text** out of
the file between two markers and `eval` it, because there was no way to import
anything. That cut broke four guards of this refactor when a comment moved
somewhere else.

The boundaries were not chosen by taste: they were measured. The usage graph
over the 164 names pointed at four edges crossing layers in the wrong
direction, and all four were fixed (the `restoreState` that told the Hub to
redraw, the `renderList` that called the rotor builder, the `toggleAdvanced`
that lived far from the form, and the bootstrap that had landed in the middle
of the help).

The behaviour is in `tests/js/`, importing the real modules. Here are the
structural properties."""

import io
import os
import re

import pytest

import js_analysis
from frontend_source import FRONTEND, ROOT, code_lines, modules, relative

MODULES = [relative(path) for path in modules()]


def _text(module):
    with io.open(
        os.path.join(FRONTEND, module), encoding="utf-8", newline=""
    ) as handle:
        return handle.read()


def _index():
    with io.open(
        os.path.join(FRONTEND, "index.html"), encoding="utf-8", newline=""
    ) as handle:
        return handle.read()


IMPORTS_FROM = re.compile(r"""import\s+(?:\{[^}]*\}\s+from\s+)?['"]([^'"]+)['"]""")


def _imports(module):
    """The modules this one imports, as paths relative to the frontend folder."""
    folder = os.path.dirname(module)
    for target in IMPORTS_FROM.findall(_text(module)):
        if target.startswith("."):
            yield os.path.normpath(os.path.join(folder, target)).replace(os.sep, "/")


def test_the_frontend_is_no_longer_one_file():
    assert not os.path.exists(os.path.join(FRONTEND, "app.js")), "o app.js voltou"
    assert os.path.isdir(os.path.join(FRONTEND, "core"))
    assert os.path.isdir(os.path.join(FRONTEND, "components"))
    assert len(MODULES) >= 10


@pytest.mark.parametrize("module", MODULES)
def test_no_module_uses_a_name_it_did_not_declare(module):
    """An ES module runs in strict mode: a loose name only throws when the line runs.

    Loading the modules proves almost nothing -- nearly all of this code only runs
    on a click. This sweep reads each file and requires every identifier to be
    declared, imported or a browser global. It is what caught the missing imports
    during the extraction itself."""
    loose = js_analysis.free_names(_text(module))
    assert loose == [], "%s uses without declaring: %s" % (module, loose)


def test_the_page_loads_the_frontend_as_a_module():
    html = _index()
    assert '<script type="module" src="main.js">' in html
    assert 'src="app.js"' not in html


def test_the_file_protocol_guard_is_a_classic_script():
    """With `type="module"`, opening from disk fails on CORS before our code runs.

    The guard tells the reader to run `python app.py`. If it became a module, the
    browser would refuse the file and the message would never appear -- in exactly
    the one case it exists to appear in."""
    html = _index()
    assert "window.location.protocol === 'file:'" in html
    excerpt = html[: html.index("window.location.protocol === 'file:'")]
    opening = excerpt.rindex("<script")
    assert "type=" not in html[opening : opening + 40], "the guard became a module"

    for module in MODULES:
        assert "protocol === 'file:'" not in _text(module), (
            "the guard went back into the JS"
        )


def test_only_one_place_publishes_to_window():
    """The bridge to the inline handlers lives in a single block.

    Before there were 21 `window.x = function` scattered through the file, each one
    an invisible decision to make something global. The bridge has to be a list you
    read top to bottom -- and one that shrinks when the next slice replaces the
    inline handlers with event delegation."""
    scattered = []
    for module in MODULES:
        for number, line in enumerate(_text(module).split("\n"), 1):
            if re.match(r"\s*window\.[A-Za-z_$][\w$]*\s*=\s*(async\s+)?function", line):
                scattered.append("%s:%d" % (module, number))
    assert scattered == [], "the bridge is scattered across %s" % scattered

    blocks = [m for m in MODULES if "Object.assign(window, {" in _text(m)]
    assert blocks == ["main.js"], "the bridge is in %s" % blocks


def test_nothing_imports_the_entry_point():
    """`main.js` is a leaf of the graph: importing the entry point makes a cycle."""
    culprits = [m for m in MODULES if m != "main.js" and "main.js" in list(_imports(m))]
    assert culprits == [], "%s imports the entry point" % culprits


def test_the_lower_layers_have_no_cycle():
    """In `core/` and `components/`, a cycle denounces a wrong boundary.

    Three cycles showed up in the measurement before the first extraction, and all
    three were layer defects. Inside `features/` the rule is different -- see
    `test_features_may_reference_each_other_but_run_nothing_on_load`."""
    lower_layers = [m for m in MODULES if m.startswith(("core/", "components/"))]
    graph = {m: set(_imports(m)) & set(lower_layers) for m in lower_layers}

    def path_to(start):
        stack, seen = [(start, [start])], set()
        while stack:
            current, trail = stack.pop()
            for neighbour in sorted(graph.get(current, ())):
                if neighbour == start:
                    return trail + [neighbour]
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append((neighbour, trail + [neighbour]))
        return None

    cycles = [c for c in (path_to(m) for m in lower_layers) if c]
    assert cycles == [], "ciclo: %s" % cycles


@pytest.mark.parametrize("module", [m for m in MODULES if m != "main.js"])
def test_the_layers_only_look_downwards(module):
    """`core/` does not know `components/`, and neither knows the features.

    It is the rule that gives the folders meaning. Without it, `core/` is just a
    place where some files happen to live."""
    layer = module.split("/")[0]
    allowed = {
        "core": ("core",),
        "components": ("core", "components"),
        "features": ("core", "components", "features"),
    }[layer]
    for target in _imports(module):
        assert target.split("/")[0] in allowed, "%s imports %s" % (module, target)


@pytest.mark.parametrize("module", MODULES)
def test_no_module_assigns_to_something_it_imported(module):
    """An imported binding is read-only, and the error only shows up at run time.

    `core/schema.js` did `SCHEMA_LANGUAGE = schema.language`, with the name coming
    from `core/i18n.js`. The parser accepts it, the load accepts it, and the first
    language change throws a TypeError. Whoever needs to change another module's
    state asks for a function -- `setSchemaLanguage` -- instead of assigning."""
    raw = _text(module)
    source = js_analysis.code_only(raw)
    culprits = []
    for name in sorted(js_analysis.imported(raw)):
        pattern = r"(?<![\w$.])%s\s*(?:=[^=>]|\+\+|--|\+=|-=|\*=)" % re.escape(name)
        if re.search(pattern, source):
            culprits.append(name)
    assert culprits == [], "%s assigns to %s, which it imported" % (module, culprits)


@pytest.mark.parametrize("module", MODULES)
def test_no_module_shadows_a_name_it_imported(module):
    """A local with the name of an import is not a syntax error -- it is worse.

    When the six shared values moved into an object called `state`, three places
    already had a local with that name. In one of them the mass rename started
    reading the wrong object, and read it before its declaration on top of that
    (a guaranteed ReferenceError, in the language-change function). Nothing in the
    tooling says a word about it."""
    raw = _text(module)
    declared = js_analysis.declared(js_analysis.code_only(raw))
    shadowed = sorted(js_analysis.imported(raw) & declared)
    assert shadowed == [], "%s declares %s, which it also imports" % (module, shadowed)


# A live function nobody calls. There is exactly one exception, and it is
# written down with its reason: without that, the test would become a list of
# suppressions that only grows.
ACCEPTED_ORPHANS = {
    "changeLanguage": "it exists and works; the selector is missing from index.html (FE-11)",
    "persistenceIsOff": "a read accessor, today consumed only by the tests",
}


def test_no_function_is_left_unreachable():
    """A function nobody calls is either a lost wiring or dead code.

    That is what happened to `onReorder`: slice 3 created the hook, nobody
    subscribed, and dragging an element stopped updating the figure -- with no
    error, no warning, only the user noticing. The loose-name sweep does not see
    this: the name is declared and imported, it is just not used."""
    # Arrow functions too: `const getNum = (id, def) => ...` escaped the first
    # version of this test and stayed dead in the file for two slices.
    definition = re.compile(
        r"^(?:export\s+)?(?:async\s+)?(?:function\s+|const\s+(?=[A-Za-z_$][\w$]*\s*=\s*"
        r"(?:async\s*)?(?:function|\(|[A-Za-z_$][\w$]*\s*=>)))([A-Za-z_$][\w$]*)",
        re.M,
    )

    defined = {}
    for module in MODULES:
        for found in definition.finditer(_text(module)):
            defined[found.group(1)] = module

    every_name = "\n".join(_text(m) for m in MODULES) + "\n" + _index()
    orphans = []
    for name, module in sorted(defined.items()):
        if name in ACCEPTED_ORPHANS:
            continue
        uses = len(re.findall(r"(?<![\w$.])%s(?![\w$])" % re.escape(name), every_name))
        if uses <= 1:
            orphans.append("%s (%s)" % (name, module))
    assert orphans == [], "nobody calls: %s" % orphans


def test_every_hook_a_component_offers_is_registered():
    """A declared hook has to have a subscriber, and the subscriber lives in `main.js`.

    The components do not know the features: when one needs to announce that
    something changed, it exposes an `onX(fn)`. Declaring the hook and forgetting
    to connect it leaves the interface partly inert with no symptom at all.

    The prefix was `quando` until the translation. This guard was the only one the
    rename table did not reach, and the reason is worth recording: it does not look
    for a NAME, it looks for a CONVENTION. Replacing `quandoReordenar` with
    `onReorder` says nothing about the `quando[A-Z]` pattern written here -- only
    the control on the next line ("the sweep stopped working") kept it from passing
    while measuring zero."""
    hooks = set()
    for module in MODULES:
        if module.startswith("components/") or module.startswith("core/"):
            hooks.update(
                re.findall(r"^export function (on[A-Z][\w$]*)", _text(module), re.M)
            )
    assert hooks, "no hook found -- the sweep stopped working"

    main = _text("main.js")
    missing = [
        g for g in sorted(hooks) if not re.search(r"(?<![\w$.])%s\s*\(" % g, main)
    ]
    assert missing == [], "hook with no subscriber: %s" % missing


def test_features_may_reference_each_other_but_run_nothing_on_load():
    """Screens reference each other; that is the domain, not disorganisation.

    The Hub opens the modeling screen, and loading a file in the modeling screen
    goes back to the Hub. Forcing `features/` to be acyclic would require an
    indirect navigation layer that would hide the real call graph -- worse to read
    and worse to change.

    What makes the cycle safe is something else: no feature module **runs**
    anything when imported. If one called another in the body of the module, the
    cycle would stop being a shape and become a defect -- a shape that changes
    depending on who imports first. As long as there are only declarations, the
    order of evaluation does not matter.

    The only file that runs code on load is `main.js` -- and nobody imports it."""
    allowed = (
        "import ",
        "export ",
        "function ",
        "async function ",
        "const ",
        "let ",
        "var ",
        "class ",
        "//",
        "/*",
        " ",
        "*",
        "}",
        ")",
        "]",
        "`",
    )
    culprits = []
    for module in [m for m in MODULES if m.startswith("features/")]:
        depth = 0
        for number, line in enumerate(_text(module).split("\n"), 1):
            if depth == 0 and line.strip() and not line.startswith(allowed):
                culprits.append("%s:%d %s" % (module, number, line.strip()[:60]))
            depth += (
                line.count("{")
                + line.count("(")
                + line.count("[")
                - line.count("}")
                - line.count(")")
                - line.count("]")
            )
    assert culprits == [], "feature running code at load time: %s" % culprits


def test_the_entry_point_holds_no_logic():
    """`main.js` wires the layers and decides nothing.

    After the second delivery it is imports, the wiring between layers, the
    bootstrap and the bridge. Any function declared there is logic that lost its
    home."""
    declarations = re.findall(
        r"^(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)",
        _text("main.js"),
        re.M,
    )
    assert declarations == [], "main.js voltou a declarar %s" % declarations
    assert len(_text("main.js").split("\n")) < 120


def test_every_relative_import_points_at_a_file_that_exists():
    """An `import` to a renamed file gives no error until someone runs node.

    Ten of them appeared at once when the modules were renamed to English, and they
    started throwing `ERR_MODULE_NOT_FOUND`. Except that what saw them throw was
    `node` -- and the lesson of the previous slice is that node may not be
    installed. This guard reads the source: it accuses the same defect on a machine
    with no node.

    It holds for both sides of the frontend and for the batteries: all of them use
    relative paths, and none has anything resolving them before execution."""
    broken = []
    HERE = os.path.dirname(os.path.abspath(__file__))
    roots = [FRONTEND, os.path.join(HERE, "js")]
    for root_folder in roots:
        for folder, _, names in os.walk(root_folder):
            if "vendor" in folder or "node_modules" in folder:
                continue
            for name in sorted(n for n in names if n.endswith(".js")):
                path = os.path.join(folder, name)
                with io.open(path, encoding="utf-8", newline="") as handle:
                    js = handle.read()
                for target in re.findall(r"""from\s+['"](\.[^'"]+)['"]""", js):
                    destination = os.path.normpath(os.path.join(folder, target))
                    if not os.path.exists(destination):
                        broken.append("%s -> %s" % (os.path.basename(path), target))
    assert broken == [], "import to a file that does not exist: %s" % broken


def test_the_import_sweep_actually_resolves_something():
    """Control: if the regular expression stopped matching, everything would pass."""
    found_items = 0
    for folder, _, names in os.walk(FRONTEND):
        if "vendor" in folder:
            continue
        for name in sorted(n for n in names if n.endswith(".js")):
            with io.open(
                os.path.join(folder, name), encoding="utf-8", newline=""
            ) as handle:
                found_items += len(
                    re.findall(r"""from\s+['"](\.[^'"]+)['"]""", handle.read())
                )
    assert found_items > 30, "only %d relative imports: the sweep changed" % found_items


# --- A styled class nobody writes is dead style ------------------------------
#
# `.lang-select` had 17 lines of rule and no element: the language selector was
# born in the i18n slice with `class="ui-language"`, and the CSS was left
# pointing at the old name. Nothing breaks -- the selector simply shows up
# unstyled, with the browser's default look, and that passes for a choice.
#
# It is the missing half of a renaming: renaming the class in the JS and
# forgetting the CSS (or the other way round) produces no error at all, only a
# silent loss of style.

THIRD_PARTY_CLASSES = {
    "js-plotly-plot",  # what writes it is Plotly, inside the chart
}


def _css_classes():
    with io.open(
        os.path.join(FRONTEND, "style.css"), encoding="utf-8", newline=""
    ) as handle:
        return set(re.findall(r"\.([a-zA-Z][\w-]*)", handle.read()))


def _frontend_text():
    parts = []
    for folder, folders, names in os.walk(FRONTEND):
        folders[:] = [p for p in folders if p not in ("vendor", "lib")]
        for name in sorted(names):
            if name.endswith((".js", ".html")):
                with io.open(
                    os.path.join(folder, name), encoding="utf-8", newline=""
                ) as handle:
                    parts.append(handle.read())
    return "\n".join(parts)


def test_every_styled_class_is_written_by_something():
    classes = _css_classes()
    assert len(classes) > 50, "only %d classes in the css: the sweep changed" % len(
        classes
    )

    written = _frontend_text()
    assert "class=" in written, "the frontend sweep read nothing"

    orphans = sorted(
        name
        for name in classes - THIRD_PARTY_CLASSES
        if not re.search(r"\b%s\b" % re.escape(name), written)
    )
    assert orphans == [], "class with a style and nobody writing it: %s" % orphans


# --- o state escondido no window --------------------------------------------

HIDDEN = (
    "multiRotorEditTarget",
    "isAddingFromHub",
    "targetNodeForHub",
    "hiddenTargetNode",
    "nodeMap",
)


@pytest.mark.parametrize("name", HIDDEN)
def test_no_state_hangs_off_the_window_object(name):
    """State on `window` has neither owner nor scope.

    Any script on the page could read and write it, and nothing in the code said
    whose it was. Four of these were locals of a single screen; `multiRotorEditTarget`
    crosses modules and became the seventh field of `state`."""
    found_items = [
        "%s:%d" % (module, number)
        for module, number, line in code_lines()
        if "window." + name in line
    ]
    assert found_items == [], "%s ainda pendurado no window em %s" % (name, found_items)


def test_only_browser_apis_are_read_off_the_window():
    """Control: if someone hangs another piece of state on the window, it shows here."""
    allowed_names = {
        "addEventListener",
        "removeEventListener",
        "dispatchEvent",
        "location",
        "innerWidth",
        "innerHeight",
        "close",
        "open",
        "scrollY",
        "scrollX",
        "getComputedStyle",
        "ROSS_TOKEN",
        "Plotly",
        "Sortable",
        "matchMedia",
        "requestAnimationFrame",
    }
    found_items = []
    for module, number, line in code_lines():
        for name in re.findall(r"window\.([A-Za-z_$][\w$]*)", line):
            if name not in allowed_names:
                found_items.append("%s:%d %s" % (module, number, name))
    assert found_items == [], "new state on the window: %s" % found_items


def test_the_shared_value_has_a_home():
    with io.open(
        os.path.join(FRONTEND, "core", "state.js"), encoding="utf-8", newline=""
    ) as handle:
        state_js = handle.read()
    assert "multiRotorEditTarget:" in state_js


def test_the_badge_classes_exist_in_the_stylesheet():
    """A class with no rule would leave the badge with its neighbour's colour."""
    path = os.path.join(ROOT, "frontend", "style.css")
    with io.open(path, encoding="utf-8", newline="") as handle:
        css = handle.read()
    for klass in (".badge-6dof", ".badge-4dof", ".badge-torsional"):
        assert klass + " {" in css, "%s has no rule" % klass
