# -*- coding: utf-8 -*-
"""Extensions: a package adds screens, files and routes to the interface.

An extension is found through an entry point and brings blueprints, a folder
of frontend files and two HTML fragments; the ones that ship with the
interface are listed in `BUILT_IN`. These tests build one in a temporary
folder and check what it gets: its routes behind the same token, its files
public, its HTML and its script in the page, and nothing at all when no
extension is installed."""

import pytest
from flask import Blueprint, jsonify

from ross.interface.api import create_app
from ross.interface.api.extensions import (
    HOME_MARKER,
    SCREENS_MARKER,
    Extension,
    discover,
)
from ross.interface.api.security import SESSION_TOKEN

HEADERS = {"X-ROSS-Token": SESSION_TOKEN}


def _blueprint():
    blueprint = Blueprint("sample_extension", __name__)

    @blueprint.route("/api/sample/ping")
    def ping():
        return jsonify({"status": "ok"})

    @blueprint.route("/api/sample/refuse")
    def refuse():
        raise ValueError("Refused by the sample extension.")

    return blueprint


@pytest.fixture
def sample(tmp_path):
    (tmp_path / "main.js").write_text("export const SAMPLE = 1;\n", encoding="utf-8")
    (tmp_path / "style.css").write_text(
        ".sample { color: inherit; }\n", encoding="utf-8"
    )
    (tmp_path / "home.html").write_text(
        '<button id="sample-home-button" data-action="show-screen" '
        'data-screen="screen-sample">Sample</button>\n',
        encoding="utf-8",
    )
    (tmp_path / "screens.html").write_text(
        '<div id="screen-sample" class="screen"></div>\n', encoding="utf-8"
    )
    return Extension(
        name="sample",
        static_dir=str(tmp_path),
        blueprints=(_blueprint(),),
        script="main.js",
        styles=("style.css",),
        home="home.html",
        screens="screens.html",
    )


def test_page_carries_the_extension(sample):
    page = create_app(extensions=[sample]).test_client().get("/").get_data(as_text=True)
    assert '<script type="module" src="ext/sample/main.js?v=' in page
    assert '<link rel="stylesheet" href="ext/sample/style.css?v=' in page
    assert 'id="screen-sample"' in page
    assert HOME_MARKER not in page and SCREENS_MARKER not in page
    home = page.index('id="screen-home"')
    button = page.index('id="sample-home-button"')
    screen = page.index('id="screen-sample"')
    assert home < button < page.index('id="screen-rotor-hub"')
    assert screen < page.index('id="help-modal-overlay"')


def test_page_without_extensions_keeps_only_the_markers():
    page = create_app(extensions=[]).test_client().get("/").get_data(as_text=True)
    assert "ext/" not in page
    assert HOME_MARKER in page and SCREENS_MARKER in page


def test_files_are_public_and_never_cached(sample):
    client = create_app(extensions=[sample]).test_client()
    response = client.get("/ext/sample/main.js")
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    assert b"SAMPLE" in response.data
    response.close()
    assert client.get("/ext/sample/missing.js").status_code == 404
    assert client.get("/ext/other/main.js").status_code == 404
    assert client.get("/ext/sample/../app.py").status_code == 404


def test_routes_sit_behind_the_token_and_the_error_handlers(sample):
    client = create_app(extensions=[sample]).test_client()
    assert client.get("/api/sample/ping").status_code == 403
    assert client.get("/api/sample/ping", headers=HEADERS).get_json() == {
        "status": "ok"
    }
    refused = client.get("/api/sample/refuse", headers=HEADERS)
    assert refused.status_code == 400
    assert refused.get_json()["message"] == "Refused by the sample extension."


def test_two_extensions_with_one_name_are_refused(sample, tmp_path):
    twin = Extension(name="sample", static_dir=str(tmp_path))
    with pytest.raises(ValueError, match="share a name"):
        create_app(extensions=[sample, twin])


@pytest.mark.parametrize("name", ["", "Sample", "../x", "a b", "x" * 41, None])
def test_bad_names_are_refused(name, tmp_path):
    with pytest.raises(ValueError, match="Invalid extension name"):
        Extension(name=name, static_dir=str(tmp_path))


def test_missing_folder_is_refused(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        Extension(name="sample", static_dir=str(tmp_path / "missing"))


def test_the_digital_twin_ships_as_a_built_in_extension(monkeypatch):
    monkeypatch.setattr("ross.interface.api.extensions.entry_points", lambda group: [])
    assert [extension.name for extension in discover()] == ["dt"]
    page = create_app().test_client().get("/").get_data(as_text=True)
    assert 'id="screen-dt"' in page


def test_entry_point_must_name_an_extension(monkeypatch):
    class Entry:
        name = "broken"

        def load(self):
            return object()

    monkeypatch.setattr("ross.interface.api.extensions.BUILT_IN", ())
    monkeypatch.setattr(
        "ross.interface.api.extensions.entry_points", lambda group: [Entry()]
    )
    with pytest.raises(TypeError, match="must point at an Extension"):
        discover()


def test_discovery_returns_the_installed_extensions_by_name(monkeypatch, tmp_path):
    def entry(extension):
        class Entry:
            name = extension.name

            def load(self):
                return extension

        return Entry()

    found = [
        Extension(name="zeta", static_dir=str(tmp_path)),
        Extension(name="alpha", static_dir=str(tmp_path)),
    ]
    monkeypatch.setattr("ross.interface.api.extensions.BUILT_IN", ())
    monkeypatch.setattr(
        "ross.interface.api.extensions.entry_points",
        lambda group: [entry(extension) for extension in found],
    )
    assert [extension.name for extension in discover()] == ["alpha", "zeta"]
