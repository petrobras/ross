# -*- coding: utf-8 -*-
"""In-process caches: bounded, locked, and keyed honestly.

Two defects used to live here.

**BE-06.** The element cache was pruned at the end of every assembly, dropping
everything that did not belong to the rotor just built. Switching between two
rotors in the Hub emptied the cache in both directions -- precisely the case it
was there to serve. And both dictionaries were globals without a lock, inside a
Flask that serves requests on threads.

**BE-07.** The analysis key was the whole payload, and the payload carried
`savedAnalyses` -- the already-rendered charts, complete Plotly figures. Saving
one chart changed the key of every following analysis: a cold cache without
anything about the rotor having changed.

On sharing ROSS objects between rotors, which is what the cache does:
`Rotor.__init__` copies the shaft elements before touching them (ROSS's own
code explains this is so it does not alter elements used by another rotor); for
disks, bearings and point masses it only calls `set_tag`, which respects a tag
that is already set -- and this interface always sets one. What is left is
`elm.n_l = elm.n` on bearings, which is idempotent. `tests/test_ross_premises.py` pins
both premises: if a ROSS release changes them, the suite says so.
"""

import hashlib
import json
import threading
from collections import OrderedDict

from .element_registry import categories

# The project keys that describe the rotor. Derived from the element registry,
# so a new category joins on its own. It is an allow list, not a deny list: a
# new field that belongs only to the interface (a label, some screen state)
# cannot invalidate a cache entry by accident.
STRUCTURAL_KEYS = tuple(categories()) + (
    "isMultiRotor",
    "driving_rotor",
    "driven_rotor",
    "multi_params",
)

_NESTED_PROJECTS = ("driving_rotor", "driven_rotor")

_MISSING = object()


def structural_project(project):
    """Return only what changes the rotor: no name, no uid, no saved charts."""
    if not isinstance(project, dict):
        return {}
    lean = {}
    for cache_key in STRUCTURAL_KEYS:
        if cache_key not in project:
            continue
        cached = project[cache_key]
        lean[cache_key] = (
            structural_project(cached) if cache_key in _NESTED_PROJECTS else cached
        )
    return lean


def fingerprint(value):
    """Return a stable fingerprint of a JSON structure."""
    text = json.dumps(value, sort_keys=True, default=str)
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def rotor_key(project, conversion_type=""):
    return fingerprint([structural_project(project), conversion_type or ""])


def spec_key(project, conversion_type, analysis_type, spec):
    """Key an analysis result: the rotor plus the computation's spec.

    The `spec` is exactly what the runner's `compute` receives -- no more, no
    less (see services/analysis/base.py). That makes it impossible for a
    computation to depend on something left out of the key.

    Until slice 2 the key was "the parameters minus a hand-kept list of plot
    keys", and the list was wrong: Campbell reads `plot_type` during the
    computation, to clip the speeds to 15 in mode-shape mode, while
    `plot_type` sat outside the key. Opening Mode Shape and going back to
    Default returned a 15-point diagram with the form still saying 50.
    """
    return fingerprint(
        [structural_project(project), conversion_type or "", analysis_type, spec]
    )


class BoundedCache:
    """A dictionary with a size bound, least-recently-used eviction and a lock."""

    def __init__(self, maxsize, name=""):
        self.maxsize = maxsize
        self.name = name
        self._items = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key, default=None):
        with self._lock:
            if key not in self._items:
                self.misses += 1
                return default
            self._items.move_to_end(key)
            self.hits += 1
            return self._items[key]

    def put(self, key, value):
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self.maxsize:
                self._items.popitem(last=False)
        return value

    def get_or_create(self, key, factory):
        """Return the stored value, or build one.

        The factory runs OUTSIDE the lock, deliberately: building a
        PlainJournal solves Reynolds' equation and takes seconds, and holding
        the lock there would serialise the whole rotor assembly. The worst case
        becomes the same element built twice -- waste, never corruption,
        because only the dictionary operations are protected.
        """
        stored = self.get(key, _MISSING)
        if stored is not _MISSING:
            return stored
        return self.put(key, factory())

    def clear(self):
        with self._lock:
            self._items.clear()
            self.hits = 0
            self.misses = 0

    def stats(self):
        with self._lock:
            return {
                "name": self.name,
                "size": len(self._items),
                "maxsize": self.maxsize,
                "hits": self.hits,
                "misses": self.misses,
            }

    def __len__(self):
        with self._lock:
            return len(self._items)

    def __contains__(self, key):
        with self._lock:
            return key in self._items


# A ROSS element takes little room; what costs is rebuilding it. 200 covers
# several Hub rotors at once, which is the case the old pruning destroyed.
ELEMENT_CACHE = BoundedCache(200, "elements")

# An analysis result carries matrices, so this bound stays small.
ANALYSIS_CACHE = BoundedCache(15, "analyses")
