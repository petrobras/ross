# -*- coding: utf-8 -*-
"""Extensions: packages that add screens and routes to the interface.

An extension lives in `ross/interface/extensions/<name>/` and is listed in
`BUILT_IN` below. A separate package can add one too, by declaring in its own
`pyproject.toml`

    [project.entry-points."ross.interface.extensions"]
    my_screens = "my_package.extension:EXTENSION"

Either way the target is an `Extension` instance.

What an extension brings:

* `blueprints`: Flask blueprints. They sit behind the same session token and
  the same error handlers as the interface's own routes.
* `static_dir`: a folder served at `/ext/<name>/`, public like the
  interface's own frontend files.
* `script`: the ES module, inside `static_dir`, that the page loads. It imports
  the interface's modules by absolute path (`/core/i18n.js`, `/core/actions.js`,
  ...), so it shares their state: the same action table, the same texts, the
  same modals.
* `styles`: stylesheets inside `static_dir`.
* `home`: an HTML file inside `static_dir` with the buttons the home screen
  shows for this extension.
* `hub`: an HTML file inside `static_dir` with the buttons the header of the
  Rotor Hub ("My Rotors") shows for this extension.
* `analysis`: an HTML file inside `static_dir` with the buttons the top bar
  of the analysis screen ("ROSS - Analysis") shows for this extension.
* `screens`: an HTML file inside `static_dir` with the extension's screens.

The HTML is inserted into `index.html` by the server, before the page reaches
the browser, so the `data-i18n` and `data-action` attributes in it are handled
by the interface exactly as its own.
"""

import os
import re
from importlib import import_module
from importlib.metadata import entry_points

from flask import Blueprint, abort, current_app, send_from_directory

GROUP = "ross.interface.extensions"

# The extensions that ship inside the interface, as "module:attribute".
BUILT_IN = ("ross.interface.extensions.digital_twin.extension:EXTENSION",)

NAME = re.compile(r"[a-z][a-z0-9_-]{0,39}")

HOME_MARKER = "<!-- extensions:home -->"
HUB_MARKER = "<!-- extensions:hub -->"
ANALYSIS_MARKER = "<!-- extensions:analysis -->"
SCREENS_MARKER = "<!-- extensions:screens -->"

# Each HTML fragment of an extension, and the marker of index.html it replaces.
FRAGMENTS = {
    "home": HOME_MARKER,
    "hub": HUB_MARKER,
    "analysis": ANALYSIS_MARKER,
    "screens": SCREENS_MARKER,
}


class Extension:
    """What one extension adds to the interface.

    Parameters
    ----------
    name : str
        Short identifier, used in the URL `/ext/<name>/`. Lowercase letters,
        digits, `-` and `_`.
    static_dir : str
        Folder with the extension's frontend files.
    blueprints : tuple of flask.Blueprint, optional
        Routes the extension adds.
    script : str, optional
        ES module loaded by the page, relative to `static_dir`.
    styles : tuple of str, optional
        Stylesheets, relative to `static_dir`.
    home : str, optional
        HTML file with the home screen buttons, relative to `static_dir`.
    hub : str, optional
        HTML file with the Rotor Hub header buttons, relative to `static_dir`.
    analysis : str, optional
        HTML file with the analysis screen top bar buttons, relative to
        `static_dir`.
    screens : str, optional
        HTML file with the extension's screens, relative to `static_dir`.
    """

    def __init__(
        self,
        name,
        static_dir,
        blueprints=(),
        script=None,
        styles=(),
        home=None,
        screens=None,
        hub=None,
        analysis=None,
    ):
        if not isinstance(name, str) or not NAME.fullmatch(name):
            raise ValueError("Invalid extension name: %r" % (name,))
        if not os.path.isdir(static_dir):
            raise ValueError(
                "Extension %s: static_dir %s does not exist." % (name, static_dir)
            )
        self.name = name
        self.static_dir = os.path.abspath(static_dir)
        self.blueprints = tuple(blueprints)
        self.script = script
        self.styles = tuple(styles)
        self.home = home
        self.hub = hub
        self.analysis = analysis
        self.screens = screens

    def url(self, path):
        return "ext/%s/%s?v=%d" % (self.name, path, self._mtime(path))

    def read(self, path):
        if not path:
            return ""
        with open(self._file(path), encoding="utf-8") as handle:
            return handle.read()

    def _file(self, path):
        return os.path.join(self.static_dir, *path.split("/"))

    def _mtime(self, path):
        handle = self._file(path)
        return int(os.path.getmtime(handle)) if os.path.isfile(handle) else 0


def discover():
    """Load the built-in extensions and the installed ones, sorted by name."""
    found = [_load(target, import_module) for target in BUILT_IN]
    for entry in entry_points(group=GROUP):
        found.append(
            _checked(entry.load(), "Entry point %s of group %s" % (entry.name, GROUP))
        )
    return sorted(found, key=lambda extension: extension.name)


def _load(target, importer):
    module, attribute = target.split(":")
    return _checked(getattr(importer(module), attribute), target)


def _checked(extension, origin):
    if not isinstance(extension, Extension):
        raise TypeError("%s must point at an Extension." % origin)
    return extension


def register_extensions(app, extensions):
    """Register the extensions' routes and keep them for the page."""
    names = [extension.name for extension in extensions]
    if len(names) != len(set(names)):
        raise ValueError("Two extensions share a name: %s" % names)
    for extension in extensions:
        for blueprint in extension.blueprints:
            app.register_blueprint(blueprint)
    app.config["ROSS_EXTENSIONS"] = tuple(extensions)
    app.register_blueprint(extension_files)


def installed(app=None):
    """The extensions of the running application."""
    return (app or current_app).config.get("ROSS_EXTENSIONS", ())


def insert_into_page(html, extensions):
    """Put each extension's styles, script, buttons and screens in the page."""
    if not extensions:
        return html
    head = []
    fragments = {name: [] for name in FRAGMENTS}
    for extension in extensions:
        for style in extension.styles:
            head.append('<link rel="stylesheet" href="%s">' % extension.url(style))
        if extension.script:
            head.append(
                '<script type="module" src="%s"></script>'
                % extension.url(extension.script)
            )
        for name, found in fragments.items():
            found.append(extension.read(getattr(extension, name)))
    html = html.replace("</head>", "    %s\n</head>" % "\n    ".join(head), 1)
    for name, marker in FRAGMENTS.items():
        html = html.replace(marker, "\n".join(fragments[name]), 1)
    return html


extension_files = Blueprint("extensions", __name__)


@extension_files.route("/ext/<name>/<path:filename>")
def static_file(name, filename):
    """Serve a file from an extension's `static_dir`."""
    for extension in installed():
        if extension.name == name:
            response = send_from_directory(extension.static_dir, filename)
            response.headers["Cache-Control"] = "no-store"
            return response
    abort(404)
