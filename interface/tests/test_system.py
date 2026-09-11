# -*- coding: utf-8 -*-
"""The page and the files it loads: no caching, and a URL per version.

Two problems live here, and both are invisible when they happen.

The page can **never** be kept by the browser: the token changes on every run,
and a page kept from yesterday can only earn a 403.

And every local file it references goes out stamped with its modification time.
Without that, changing `app.js` and reloading can serve the old file -- and the
user reports a defect that has already been fixed, or a fixed one that came
back. The stamp holds for every relative reference, and not for a list of two
files: a third one added by anyone would be left out in silence."""

import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from api import FRONTEND_DIR, create_app
from api.system import LOCAL_REFERENCE

APP = create_app()


@pytest.fixture
def client():
    APP.config["TESTING"] = True
    with APP.test_client() as client:
        yield client


def test_the_page_is_never_cached(client):
    """The token changes on every run; a kept page could only earn a 403."""
    assert client.get("/").headers["Cache-Control"] == "no-store"


def test_every_local_asset_is_versioned(client):
    """A changed file has to earn another URL.

    The page is not cached, but the JS and the CSS are served by Flask's static
    handler, and the cache header they carry depends on the installed Flask
    version. Changing `main.js` and the browser going on running the previous one
    is not a hypothesis: it happened, and the symptom was a fix that "did not
    work" -- the worst kind of failure, because it does not look like caching, it
    looks like wrong code.

    The guard holds for **every** local reference that exists on disk, and not for
    a list of two names: a third file added to the index has to be born stamped."""
    html = client.get("/").get_data(as_text=True)
    for _, path in LOCAL_REFERENCE.findall(html):
        handle = os.path.join(FRONTEND_DIR, *path.split("/"))
        assert not os.path.isfile(handle), "%s exists and went out with no stamp" % path

    for name in ("main.js", "style.css"):
        assert re.search(r'"%s\?v=\d+"' % re.escape(name), html), (
            "%s with no version" % name
        )


def test_a_changed_file_changes_its_url(client):
    """Control: a constant stamp would pass the test above just the same."""
    before = client.get("/").get_data(as_text=True)
    path = os.path.join(FRONTEND_DIR, "style.css")
    original = os.path.getmtime(path)
    try:
        os.utime(path, (original + 60, original + 60))
        after = client.get("/").get_data(as_text=True)
    finally:
        os.utime(path, (original, original))

    def version(html):
        return re.search(r"style\.css\?v=(\d+)", html).group(1)

    assert version(before) != version(after)


def test_only_relative_references_are_candidates():
    """What does not live in the frontend folder is not stamped.

    Tested on the pattern and not on the response: which files exist on disk
    changes between the packaged copy and the development one, and a guard that
    depends on that measures the environment instead of the code."""
    html = (
        '<link href="style.css">'
        '<script src="https://exemplo/x.js"></script>'
        '<script src="//exemplo/y.js"></script>'
        '<script src="/absoluto/z.js"></script>'
        '<a href="#topo">topo</a>'
    )
    assert [path for _, path in LOCAL_REFERENCE.findall(html)] == ["style.css"]


def test_a_reference_already_carrying_a_query_is_left_alone():
    """Carimbar duas vezes produziria `a.js?v=1?v=2`."""
    assert LOCAL_REFERENCE.findall('<script src="main.js?v=1"></script>') == []
