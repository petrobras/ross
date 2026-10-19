"""Collect the facts a ROSS pull request is checked against.

Compare two revisions of the repository and report what AGENTS.md asks
contributors and reviewers to look at: the files that changed, scope signals
(new subpackages, new runtime dependencies), public API changes, architecture
flags, changes to existing tests and reference data, and writing signals.

The comparison is static: files are read with git and parsed with ``ast``.
Nothing from the change is imported or executed, so it is safe to run on a pull
request before reading its code. Flags are prompts to look at something, not
verdicts.

Usage::

    python tools/pr_facts.py                        # working tree vs merge base
    python tools/pr_facts.py --head HEAD            # last commit vs merge base
    python tools/pr_facts.py --head refs/review/pr-1427
    python tools/pr_facts.py --json                 # machine-readable output
"""

import argparse
import ast
import difflib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path, PurePosixPath

ELEMENT_METHODS = ("M", "K", "C", "G", "dof_mapping")
LARGE_CHANGE_LINES = 1000
LONG_DOCSTRING_WORDS = 400
MOVED_TEST_SIMILARITY = 0.6
TOLERANCE_KEYWORDS = ("rtol", "atol", "decimal", "rel", "abs")
POSITIONAL_TOLERANCES = {
    "assert_allclose": {2: "rtol", 3: "atol"},
    "assert_almost_equal": {2: "decimal"},
    "assert_array_almost_equal": {2: "decimal"},
    "isclose": {2: "rtol", 3: "atol"},
    "allclose": {2: "rtol", 3: "atol"},
}
FILLER_PHRASES = (
    "it is worth noting",
    "it's worth noting",
    "it is important to note",
    "it's important to note",
    "comprehensive",
    "seamless",
    "leverage",
    "delve",
    "cutting-edge",
    "state-of-the-art",
    "in order to",
)
EMOJI = re.compile("[\U0001f300-\U0001faff\u2600-\u27bf\u2b50\u2b55]")
TEST_CONFIG_FILES = ("pytest.ini", "setup.cfg", "codecov.yml", ".coveragerc", "tox.ini")
CONFIG_FILES = TEST_CONFIG_FILES + (
    "pyproject.toml",
    "requirements.txt",
    "MANIFEST.in",
    ".pre-commit-config.yaml",
    ".gitattributes",
    ".gitignore",
    "AGENTS.md",
)
HANDWRITTEN = ("core", "interface", "tests", "tools")
CONSTITUTION_FILES = (
    "ross/element.py",
    "ross/units.py",
    "AGENTS.md",
    "requirements.txt",
    "pyproject.toml",
    "tools/pr_facts.py",
)
CONSTITUTION_DIRS = (".github/", ".agents/", ".claude/")
REPEATED_TEXT_MIN_WORDS = 6
REPEATED_TEXT_MIN_COUNT = 3
CODE_SUFFIXES = (".py", ".js", ".mjs", ".ts", ".css", ".html", ".md")


class Repo:
    """Read files and diffs from a git repository.

    Parameters
    ----------
    path : str or Path
        Any directory inside the repository.
    """

    def __init__(self, path="."):
        self.root = Path(self.git("rev-parse", "--show-toplevel", cwd=path).strip())

    def git(self, *args, cwd=None):
        """Run a git command and return its standard output."""
        result = subprocess.run(
            ["git", *args], cwd=cwd or self.root, capture_output=True, check=True
        )
        return result.stdout.decode("utf-8", errors="replace")

    def ref_exists(self, ref):
        """Return True when ``ref`` names a commit."""
        try:
            self.git("rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")
        except subprocess.CalledProcessError:
            return False
        return True

    def read(self, ref, path):
        """Return the text of ``path`` at ``ref``; None reads the working tree."""
        if ref is None:
            file = self.root / path
            if not file.is_file():
                return None
            return file.read_text(encoding="utf-8", errors="replace")
        try:
            return self.git("show", f"{ref}:{path}")
        except subprocess.CalledProcessError:
            return None

    def untracked(self):
        """List untracked, non-ignored files of the working tree."""
        out = self.git("ls-files", "--others", "--exclude-standard", "-z")
        return [f for f in out.split("\0") if f and (self.root / f).is_file()]

    def files(self, ref):
        """List the files at ``ref``; None lists the working tree."""
        if ref is None:
            tracked = self.git("ls-files", "-z").split("\0")
            present = [f for f in tracked if f and (self.root / f).is_file()]
            return sorted(set(present + self.untracked()))
        out = self.git("ls-tree", "-r", "--name-only", "-z", ref)
        return [f for f in out.split("\0") if f]

    def changes(self, base, head):
        """Return the files changed between ``base`` and ``head``.

        Returns
        -------
        list of dict
            One entry per file with ``path``, ``old_path``, ``status`` (git
            letter) and the ``added`` and ``deleted`` line counts (None for
            binary files).
        """
        revs = [base] if head is None else [base, head]
        fields = self.git("diff", "--name-status", "-M", "-z", *revs).split("\0")
        statuses = {}
        i = 0
        while i < len(fields) - 1:
            code = fields[i]
            if code[:1] in ("R", "C"):
                statuses[fields[i + 2]] = (code[0], fields[i + 1])
                i += 3
            elif code:
                statuses[fields[i + 1]] = (code[0], fields[i + 1])
                i += 2
            else:
                i += 1
        counts = {}
        for line in self.git("diff", "--numstat", "-M", *revs).splitlines():
            added, deleted, name = line.split("\t", 2)
            counts[rename_target(name)] = (
                None if added == "-" else int(added),
                None if deleted == "-" else int(deleted),
            )
        if head is None:
            for path in self.untracked():
                statuses[path] = ("A", path)
                counts[path] = (len(self.read(None, path).splitlines()), 0)
        changes = []
        for path in sorted(statuses):
            status, old_path = statuses[path]
            added, deleted = counts.get(path, (0, 0))
            changes.append(
                {
                    "path": path,
                    "old_path": old_path,
                    "status": status,
                    "added": added,
                    "deleted": deleted,
                }
            )
        return changes

    def added_lines(self, base, head, path):
        """Return the lines added to ``path`` between ``base`` and ``head``."""
        if head is None and self.read(base, path) is None:
            return (self.read(None, path) or "").splitlines()
        revs = [base] if head is None else [base, head]
        diff = self.git("diff", "-U0", *revs, "--", path)
        return [
            line[1:]
            for line in diff.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        ]

    def whitespace_only(self, base, head, path):
        """Return True when ``path`` differs between the revisions only in whitespace."""
        revs = [base] if head is None else [base, head]
        result = subprocess.run(
            ["git", "diff", "-w", "--ignore-blank-lines", "--quiet", *revs, "--", path],
            cwd=self.root,
            capture_output=True,
        )
        return result.returncode == 0

    def generated(self, paths):
        """Return the paths marked linguist-vendored or linguist-generated."""
        if not paths:
            return set()
        out = self.git(
            "check-attr", "-z", "linguist-vendored", "linguist-generated", "--", *paths
        ).split("\0")
        return {out[i] for i in range(0, len(out) - 2, 3) if out[i + 2] == "set"}


def rename_target(name):
    """Turn numstat rename notation (``a/{b => c}/d``) into the new path."""
    match = re.match(r"(.*)\{(.*) => (.*)\}(.*)", name)
    if match:
        prefix, _, new, suffix = match.groups()
        return (prefix + new + suffix).replace("//", "/")
    if " => " in name:
        return name.split(" => ")[1]
    return name


def category(path, generated):
    """Classify a changed path for the diff map."""
    parts = PurePosixPath(path).parts
    if path.endswith(".ipynb"):
        return "notebooks"
    if path in generated or "vendor" in parts or "golden" in parts:
        return "vendored/generated"
    if path.startswith((".github/", ".agents/", ".claude/")) or path in CONFIG_FILES:
        return "config"
    if path.startswith("ross/interface/"):
        return "interface"
    if path.startswith("ross/") and "tests" in parts:
        return "tests"
    if path.startswith("ross/agent_skills/"):
        return "agent skill"
    if path.startswith("ross/") and path.endswith(".py"):
        return "core"
    if path.startswith("ross/"):
        return "package data"
    if path.startswith("docs/") or path.endswith((".md", ".rst")):
        return "docs"
    if path.startswith("tools/"):
        return "tools"
    return "other"


def is_test_path(path):
    """Return True for Python files inside a ``tests`` directory."""
    return path.endswith(".py") and "tests" in PurePosixPath(path).parts


def is_api_path(path):
    """Return True for modules whose public names form the ROSS API."""
    return (
        path.startswith("ross/")
        and path.endswith(".py")
        and not is_test_path(path)
        and not path.startswith(("ross/interface/", "ross/agent_skills/"))
    )


def module_name(path):
    """Return the dotted module name of a ``.py`` path."""
    parts = list(PurePosixPath(path).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def parse(source):
    """Parse Python source; return None when it is missing or does not parse."""
    if source is None:
        return None
    try:
        return ast.parse(source)
    except SyntaxError:
        return None


def call_name(node):
    """Return the last component of a call's function name."""
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def base_names(cls):
    """Return the last component of each base-class expression."""
    names = []
    for base in cls.bases:
        if isinstance(base, ast.Attribute):
            names.append(base.attr)
        elif isinstance(base, ast.Name):
            names.append(base.id)
    return names


def is_public(name):
    """Return True for names without a leading underscore."""
    return not name.startswith("_")


def parameters(func):
    """Return a function's parameters as ``[name, default source or None]``.

    ``self`` and ``cls`` are left out. Keyword-only parameters are prefixed
    with ``*`` and the ``*args``/``**kwargs`` collectors keep their stars.
    """
    args = func.args
    positional = args.posonlyargs + args.args
    defaults = [None] * (len(positional) - len(args.defaults)) + args.defaults
    params = [
        [a.arg, None if d is None else ast.unparse(d)]
        for a, d in zip(positional, defaults, strict=True)
    ]
    if params and params[0][0] in ("self", "cls"):
        params = params[1:]
    if args.vararg:
        params.append([f"*{args.vararg.arg}", None])
    for a, d in zip(args.kwonlyargs, args.kw_defaults, strict=True):
        params.append([f"*{a.arg}", None if d is None else ast.unparse(d)])
    if args.kwarg:
        params.append([f"**{args.kwarg.arg}", None])
    return params


def has_annotations(func):
    """Return True when a function annotates an argument or its return value."""
    args = func.args
    every = args.posonlyargs + args.args + args.kwonlyargs
    every += [a for a in (args.vararg, args.kwarg) if a is not None]
    return func.returns is not None or any(a.annotation is not None for a in every)


def declared_all(tree):
    """Return the names in a module-level ``__all__`` list, or None."""
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            try:
                return set(ast.literal_eval(node.value))
            except ValueError:
                return None
    return None


def docstring_words(node):
    """Return the number of words in a node's docstring, or None."""
    doc = ast.get_docstring(node)
    return None if doc is None else len(doc.split())


def function_record(kind, func):
    """Describe a public function or method."""
    return {
        "kind": kind,
        "params": parameters(func),
        "annotated": has_annotations(func),
        "docstring": docstring_words(func),
    }


def instance_attributes(method):
    """Return the public ``self.<name>`` attributes a method assigns."""
    names = set()
    for node in ast.walk(method):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and is_public(target.attr)
            ):
                names.add(target.attr)
    return names


def public_api(source, module):
    """Return the public API defined in a module.

    Parameters
    ----------
    source : str or None
        Module source.
    module : str
        Dotted module name used to qualify the entries.

    Returns
    -------
    dict
        Qualified name mapped to a record with ``kind`` (class, function,
        method or attribute), ``params``, ``annotated`` and ``docstring``
        (word count or None). Classes also have ``bases`` and ``defines`` (all
        method names, private ones included).
    """
    tree = parse(source)
    if tree is None:
        return {}
    exported = declared_all(tree)
    api = {}
    for node in tree.body:
        name = getattr(node, "name", None)
        if name is None or not is_public(name):
            continue
        if exported is not None and name not in exported:
            continue
        qualname = f"{module}.{name}"
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            api[qualname] = function_record("function", node)
        elif isinstance(node, ast.ClassDef):
            record = {
                "kind": "class",
                "params": [],
                "annotated": False,
                "docstring": docstring_words(node),
                "bases": base_names(node),
                "defines": [],
            }
            attributes = set()
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    record["defines"].append(item.name)
                    attributes |= instance_attributes(item)
                    if item.name == "__init__":
                        record["params"] = parameters(item)
                        record["annotated"] = has_annotations(item)
                    elif is_public(item.name):
                        api[f"{qualname}.{item.name}"] = function_record("method", item)
                elif isinstance(item, ast.Assign):
                    attributes |= {
                        t.id
                        for t in item.targets
                        if isinstance(t, ast.Name) and is_public(t.id)
                    }
            for attribute in attributes - set(record["defines"]):
                api[f"{qualname}.{attribute}"] = {
                    "kind": "attribute",
                    "params": [],
                    "annotated": False,
                    "docstring": None,
                }
            api[qualname] = record
    return api


def parameter_changes(before, after):
    """Describe how a parameter list changed, or return [] if it did not."""
    old, new = dict(before), dict(after)
    notes = []
    for name in old:
        if name not in new:
            notes.append(f"removed {name}")
        elif old[name] != new[name]:
            notes.append(f"default of {name}: {old[name]} -> {new[name]}")
    for name in new:
        if name not in old:
            default = "required" if new[name] is None else f"default {new[name]}"
            notes.append(f"added {name} ({default})")
    shared_old = [n for n, _ in before if n in new and not n.startswith("*")]
    shared_new = [n for n, _ in after if n in old and not n.startswith("*")]
    if shared_old != shared_new:
        notes.append("positional order changed")
    return notes


def compare_api(base_api, head_api):
    """Compare two public APIs.

    A class or function that disappears from one module and appears in
    another under the same name is reported as moved, and its members are
    compared with their new copies instead of being reported as removed.

    Returns
    -------
    dict
        ``added`` and ``removed`` names, ``moved`` (old name to new name) and
        ``changed`` (name, list of parameter changes).
    """
    removed = sorted(set(base_api) - set(head_api))
    added = sorted(set(head_api) - set(base_api))
    moved = {}
    for old in removed:
        if base_api[old]["kind"] not in ("class", "function"):
            continue
        leaf = old.rsplit(".", 1)[-1]
        for new in added:
            same_kind = head_api[new]["kind"] == base_api[old]["kind"]
            if same_kind and new.rsplit(".", 1)[-1] == leaf:
                moved[old] = new
                break
    pairs = {name: name for name in set(base_api) & set(head_api)}
    pairs.update(moved)
    for old_owner, new_owner in moved.items():
        for old in removed:
            if old.startswith(old_owner + "."):
                new = new_owner + old[len(old_owner) :]
                if new in head_api:
                    pairs[old] = new
    changed = []
    for old, new in sorted(pairs.items()):
        notes = parameter_changes(base_api[old]["params"], head_api[new]["params"])
        if notes:
            changed.append({"name": new, "changes": notes})
    return {
        "added": [n for n in added if n not in pairs.values()],
        "removed": [n for n in removed if n not in pairs],
        "moved": moved,
        "changed": changed,
    }


def class_index(repo, ref, files):
    """Index every class under ``ross/`` by name.

    Returns
    -------
    dict
        Class name mapped to ``{"bases": set, "members": set}``, where the
        members are the public methods and attributes the class defines.
    """
    index = {}
    for path in files:
        if not (path.startswith("ross/") and path.endswith(".py")):
            continue
        tree = parse(repo.read(ref, path))
        for node in ast.walk(tree) if tree else ():
            if not isinstance(node, ast.ClassDef):
                continue
            entry = index.setdefault(node.name, {"bases": set(), "members": set()})
            entry["bases"].update(base_names(node))
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    entry["members"].add(item.name)
                    entry["members"] |= instance_attributes(item)
                elif isinstance(item, ast.Assign):
                    entry["members"] |= {
                        t.id for t in item.targets if isinstance(t, ast.Name)
                    }
    return index


def ancestors(name, index):
    """Return the names of the classes ``name`` derives from, itself excluded."""
    seen, stack = set(), list(index.get(name, {}).get("bases", ()))
    while stack:
        current = stack.pop()
        if current not in seen:
            seen.add(current)
            stack.extend(index.get(current, {}).get("bases", ()))
    return seen


def inherits(name, target, index):
    """Return True when class ``name`` is or derives from ``target`` (by name)."""
    return name == target or target in ancestors(name, index)


def inherited_member(qualname, index):
    """Return True when a member removed from a class is inherited instead."""
    parts = qualname.split(".")
    if len(parts) < 3 or parts[-2] not in index:
        return False
    return any(
        parts[-1] in index.get(base, {}).get("members", ())
        for base in ancestors(parts[-2], index)
    )


def literal(node):
    """Return a hashable literal value, or the node's source text."""
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return ast.unparse(node)
    return value if isinstance(value, (int, float, str)) else repr(value)


def describe_tests(source, path):
    """Describe the test functions of a test module.

    Returns
    -------
    dict
        ``(path, name)`` mapped to a record with ``asserts`` (count),
        ``tolerances`` (Counter of ``(call, keyword, value)``), ``numbers``
        (Counter of the other numeric literals), ``skips`` (skip and xfail
        markers and calls) and ``body`` (source, for matching moved tests).
    """
    tree = parse(source)
    if tree is None:
        return {}
    records = {}
    for node in ast.walk(tree):
        is_function = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        if not is_function or not node.name.startswith("test"):
            continue
        asserts, skips = 0, 0
        tolerances, numbers, tolerance_nodes = Counter(), Counter(), set()
        for decorator in node.decorator_list:
            text = ast.unparse(decorator)
            skips += "skip" in text or "xfail" in text
        for sub in ast.walk(node):
            if isinstance(sub, ast.Assert):
                asserts += 1
            if not isinstance(sub, ast.Call):
                continue
            name = call_name(sub)
            asserts += name.startswith("assert") or name in ("raises", "warns")
            skips += name in ("skip", "xfail", "importorskip")
            if not (name.startswith("assert") or name in POSITIONAL_TOLERANCES):
                continue
            found = [
                (k.arg, k.value) for k in sub.keywords if k.arg in TOLERANCE_KEYWORDS
            ]
            for index, arg in POSITIONAL_TOLERANCES.get(name, {}).items():
                if len(sub.args) > index:
                    found.append((arg, sub.args[index]))
            for arg, value in found:
                tolerances[(name, arg, literal(value))] += 1
                tolerance_nodes.update(id(n) for n in ast.walk(value))
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.Constant)
                and isinstance(sub.value, (int, float, complex))
                and not isinstance(sub.value, bool)
                and id(sub) not in tolerance_nodes
            ):
                numbers[repr(sub.value)] += 1
        records[(path, node.name)] = {
            "asserts": asserts,
            "tolerances": tolerances,
            "numbers": numbers,
            "skips": skips,
            "body": ast.unparse(node.body),
        }
    return records


def describe_helpers(source, path):
    """Return the fixtures, helpers and module-level values of a test module.

    Returns
    -------
    dict
        ``(path, name)`` mapped to the source of each module-level function
        that is not a test (decorators included) and of each module-level
        assignment.
    """
    tree = parse(source)
    if tree is None:
        return {}
    helpers = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("test"):
                helpers[(path, node.name)] = ast.unparse(node)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    helpers[(path, target.id)] = ast.unparse(node)
    return helpers


def tolerance_direction(keyword, before, after):
    """Say whether a tolerance change loosens or tightens the check."""
    if not all(isinstance(v, (int, float)) for v in (before, after)):
        return "changed"
    if keyword == "decimal":
        return "loosened" if after < before else "tightened"
    return "loosened" if after > before else "tightened"


def tolerance_details(before, after):
    """Describe the differences between two tolerance Counters."""
    lost, gained = before - after, after - before
    details = []
    for call, keyword in sorted({(c, k) for c, k, _ in lost + gained}):
        old = sorted(
            (v for c, k, v in lost.elements() if (c, k) == (call, keyword)), key=repr
        )
        new = sorted(
            (v for c, k, v in gained.elements() if (c, k) == (call, keyword)), key=repr
        )
        for b, a in zip(old, new, strict=False):
            direction = tolerance_direction(keyword, b, a)
            details.append(f"{call} {keyword}: {b} -> {a} ({direction})")
        details += [f"{call} {keyword}={b} removed" for b in old[len(new) :]]
        details += [f"{call} {keyword}={a} added" for a in new[len(old) :]]
    return details


def match_moved_test(record, candidates):
    """Return ``(key, similarity)`` of the closest candidate, or ``(None, score)``."""
    best, score = None, MOVED_TEST_SIMILARITY
    matcher = difflib.SequenceMatcher(None, b=record["body"])
    for key, candidate in candidates.items():
        matcher.set_seq1(candidate["body"])
        if matcher.real_quick_ratio() < score or matcher.quick_ratio() < score:
            continue
        ratio = matcher.ratio()
        if ratio >= score:
            best, score = key, ratio
    return best, score


def compare_tests(base_records, head_records):
    """Compare test records and return the integrity findings.

    Tests that leave one file and reappear (by name or by a similar body) in
    another are reported as moved and compared with their new copy.

    Returns
    -------
    list of dict
        Findings with ``kind``, ``test``, ``detail`` and, for moved tests,
        ``new``. Edited tests with no other finding are reported as
        ``changed``.
    """
    findings = []
    added = {k: r for k, r in head_records.items() if k not in base_records}
    pairs = []
    for key, record in base_records.items():
        label = f"{key[0]}::{key[1]}"
        if key in head_records:
            pairs.append((key, key))
            continue
        same_name = [k for k in added if k[1] == key[1]]
        match, score = (
            (same_name[0], 1.0) if same_name else match_moved_test(record, added)
        )
        if match is None:
            findings.append(
                {"kind": "removed", "test": label, "detail": "no similar test found"}
            )
            continue
        new_label = f"{match[0]}::{match[1]}"
        findings.append(
            {
                "kind": "moved",
                "test": label,
                "new": new_label,
                "detail": f"now {new_label}, similarity {score:.2f}",
            }
        )
        pairs.append((key, match))
    for old_key, new_key in pairs:
        old, new = base_records[old_key], head_records[new_key]
        label = f"{new_key[0]}::{new_key[1]}"
        flagged = len(findings)
        if new["asserts"] < old["asserts"]:
            findings.append(
                {
                    "kind": "fewer assertions",
                    "test": label,
                    "detail": f"{old['asserts']} -> {new['asserts']}",
                }
            )
        if new["skips"] > old["skips"]:
            findings.append(
                {
                    "kind": "skip or xfail added",
                    "test": label,
                    "detail": f"{old['skips']} -> {new['skips']}",
                }
            )
        if old["tolerances"] != new["tolerances"]:
            details = tolerance_details(old["tolerances"], new["tolerances"])
            findings.append(
                {
                    "kind": "tolerance changed",
                    "test": label,
                    "detail": "; ".join(details),
                }
            )
        lost = old["numbers"] - new["numbers"]
        if lost:
            gained = new["numbers"] - old["numbers"]
            findings.append(
                {
                    "kind": "expected values changed",
                    "test": label,
                    "detail": (
                        f"{sum(lost.values())} numeric literals removed, "
                        f"{sum(gained.values())} added"
                    ),
                }
            )
        if len(findings) == flagged and old["body"] != new["body"]:
            findings.append({"kind": "changed", "test": label, "detail": "body edited"})
    return findings


def silent_excepts(source):
    """Count bare ``except:`` clauses and handlers that only pass or continue."""
    tree = parse(source)
    if tree is None:
        return 0
    return sum(
        node.type is None
        or all(isinstance(s, (ast.Pass, ast.Continue)) for s in node.body)
        for node in ast.walk(tree)
        if isinstance(node, ast.ExceptHandler)
    )


def requirement_names(text):
    """Return the requirement lines of a requirements file, keyed by name."""
    names = {}
    for line in (text or "").splitlines():
        line = line.split("#")[0].strip()
        match = re.match(r"([A-Za-z0-9_.\-]+)", line)
        if match:
            names[match.group(1).lower()] = line
    return names


def python_dirs(files):
    """Return the directories under ``ross/`` that contain Python files."""
    return {
        str(PurePosixPath(p).parent)
        for p in files
        if p.startswith("ross/") and p.endswith(".py")
    }


def default_base(repo, head):
    """Return the merge base of ``head`` with the first main branch found."""
    for candidate in ("upstream/main", "origin/main", "main"):
        if repo.ref_exists(candidate):
            return repo.git("merge-base", candidate, head or "HEAD").strip()
    raise SystemExit("No main branch found; pass --base.")


def diff_map(changes, generated):
    """Classify the changes and return the line totals and size flags."""
    totals = Counter()
    for change in changes:
        change["category"] = category(change["path"], generated)
        totals[change["category"]] += (change["added"] or 0) + (change["deleted"] or 0)
    handwritten = sum(totals[c] for c in HANDWRITTEN)
    flags = []
    if totals["core"] and totals["interface"]:
        flags.append(("size", "Mixes core-library and interface changes; split them."))
    if totals["vendored/generated"] and len(totals) > 1:
        flags.append(("size", "Vendored or generated files mixed with other changes."))
    if handwritten > LARGE_CHANGE_LINES:
        flags.append(
            (
                "size",
                f"{handwritten} hand-written lines changed; consider a stack of PRs.",
            )
        )
    return dict(totals), handwritten, flags


def scope_facts(repo, base, head):
    """Return new subpackages and dependency changes, with their flags."""
    base_dirs = python_dirs(repo.files(base))
    new_dirs = sorted(
        d
        for d in python_dirs(repo.files(head)) - base_dirs
        if not d.startswith("ross/interface") and "tests" not in PurePosixPath(d).parts
    )
    new_dirs = [d for d in new_dirs if not any(d.startswith(o + "/") for o in new_dirs)]
    before = requirement_names(repo.read(base, "requirements.txt"))
    after = requirement_names(repo.read(head, "requirements.txt"))
    scope = {
        "new_subpackages": new_dirs,
        "new_dependencies": sorted(set(after) - set(before)),
        "removed_dependencies": sorted(set(before) - set(after)),
        "changed_dependencies": sorted(
            n for n in set(before) & set(after) if before[n] != after[n]
        ),
    }
    flags = [
        ("scope", f"New subpackage {d}: explain in the Scope section why it belongs.")
        for d in scope["new_subpackages"]
    ]
    flags += [
        ("scope", f"New runtime dependency {d}: justify it in the Scope section.")
        for d in scope["new_dependencies"]
    ]
    return scope, flags


def api_facts(repo, base, head, changes):
    """Compare the public API and return it with the architecture flags."""
    base_api, head_api, flags = {}, {}, []
    for change in changes:
        new_path, old_path = change["path"], change["old_path"]
        if not (is_api_path(new_path) or is_api_path(old_path)):
            continue
        before = repo.read(base, old_path)
        after = None if change["status"] == "D" else repo.read(head, new_path)
        base_api.update(public_api(before, module_name(old_path)))
        head_api.update(public_api(after, module_name(new_path)))
        if silent_excepts(after) > silent_excepts(before):
            flags.append(("tests", f"{new_path} adds a bare or silent except clause."))
        if after is not None:
            for line in repo.added_lines(base, head, new_path):
                if "pytest" in line or "PYTEST_CURRENT_TEST" in line:
                    flags.append(
                        (
                            "tests",
                            f"Library code mentions pytest: {new_path}: {line.strip()}",
                        )
                    )
    api = compare_api(base_api, head_api)
    index = class_index(repo, head, repo.files(head))
    api["inherited"] = [n for n in api["removed"] if inherited_member(n, index)]
    api["removed"] = [n for n in api["removed"] if n not in api["inherited"]]
    tables = [
        text
        for text in (repo.read(head, "AGENTS.md"), repo.read(head, "CLAUDE.md"))
        if text and "run_modal" in text
    ]
    skill_md = repo.read(head, "ross/agent_skills/ross/SKILL.md") or ""
    for name in api["added"] + list(api["moved"].values()):
        record, parts = head_api[name], name.split(".")
        leaf = parts[-1]
        if record["kind"] == "method" and leaf.startswith("run_"):
            if not inherits(parts[-2], "Rotor", index):
                flags.append(
                    (
                        "architecture",
                        f"{name}: analyses are run_* methods on the rotor classes, not on {parts[-2]}.",
                    )
                )
            else:
                if tables and not any(leaf in table for table in tables):
                    flags.append(
                        (
                            "architecture",
                            f"{name}: add it to the analysis table in AGENTS.md.",
                        )
                    )
                if leaf not in skill_md:
                    flags.append(
                        (
                            "architecture",
                            f"{name}: add it to ross/agent_skills/ross/SKILL.md.",
                        )
                    )
        if record["kind"] == "function" and leaf.startswith("run_"):
            flags.append(
                (
                    "architecture",
                    f"{name}: public run_* function; analyses belong on the rotor classes.",
                )
            )
        if record["kind"] == "class" and "Element" in record["bases"]:
            missing = [m for m in ELEMENT_METHODS if m not in record["defines"]]
            if missing:
                flags.append(
                    (
                        "architecture",
                        f"{name} subclasses Element but does not define {', '.join(missing)}.",
                    )
                )
        if record["kind"] == "class" and leaf.endswith("Results"):
            if not inherits(leaf, "Results", index):
                flags.append(("architecture", f"{name} does not subclass Results."))
        if record["annotated"]:
            flags.append(
                (
                    "architecture",
                    f"{name}: type annotations; ROSS puts types in docstrings.",
                )
            )
    for old in api["removed"]:
        if base_api[old]["kind"] == "attribute":
            flags.append(
                (
                    "api",
                    f"Public attribute {old} removed or renamed: state it in API changes.",
                )
            )
        else:
            flags.append(("api", f"{old} removed or renamed: state it in API changes."))
    for old, new in api["moved"].items():
        flags.append(
            (
                "api",
                f"{old} moved to {new}: the old import path breaks; state it in API changes.",
            )
        )
    for change in api["changed"]:
        flags.append(("api", f"{change['name']}: {'; '.join(change['changes'])}."))
    return api, head_api, flags


def test_facts(repo, base, head, changes):
    """Compare the tests and return new tests, findings and flags."""
    base_tests, head_tests, flags = {}, {}, []
    base_helpers, head_helpers = {}, {}
    for change in changes:
        new_path, old_path = change["path"], change["old_path"]
        parts = PurePosixPath(new_path).parts
        if is_test_path(old_path) or is_test_path(new_path):
            before = repo.read(base, old_path)
            base_tests.update(describe_tests(before, old_path))
            base_helpers.update(describe_helpers(before, new_path))
            if change["status"] != "D":
                after = repo.read(head, new_path)
                head_tests.update(describe_tests(after, new_path))
                head_helpers.update(describe_helpers(after, new_path))
        if parts[-1] == "conftest.py" or new_path in TEST_CONFIG_FILES:
            flags.append(("tests", f"Test configuration changed: {new_path}"))
        elif new_path.startswith(".github/workflows/"):
            flags.append(("tests", f"CI workflow changed: {new_path}"))
        elif new_path == "pyproject.toml":
            touched = repo.added_lines(base, head, new_path)
            if any(re.search(r"pytest|coverage|ruff", line) for line in touched):
                flags.append(
                    ("tests", "Test or lint settings in pyproject.toml changed.")
                )
        in_test_data = (
            "tests" in parts or "golden" in parts
        ) and not new_path.endswith(CODE_SUFFIXES)
        if in_test_data and change["status"] != "A":
            flags.append(
                ("tests", f"Reference data changed: {new_path} ({change['status']})")
            )
    for key in sorted(set(base_helpers) & set(head_helpers)):
        if base_helpers[key] != head_helpers[key]:
            flags.append(
                (
                    "tests",
                    f"{key[0]}::{key[1]}: existing fixture, helper or module-level "
                    "value changed; the tests that use it may now check something else.",
                )
            )
    findings = compare_tests(base_tests, head_tests)
    for item in findings:
        info = item["kind"] in ("moved", "changed")
        group = "tests (info)" if info else "tests"
        flags.append((group, f"{item['test']}: {item['kind']} ({item['detail']})"))
    moved_to = {item["new"] for item in findings if item["kind"] == "moved"}
    new_tests = [
        f"{path}::{name}"
        for path, name in head_tests
        if (path, name) not in base_tests and f"{path}::{name}" not in moved_to
    ]
    return new_tests, findings, flags


def constitution_facts(repo, base, head, changes):
    """Flag changes to the files that define the project's rules."""
    flags = []
    for change in changes:
        path = change["path"]
        if path in CONSTITUTION_FILES or path.startswith(CONSTITUTION_DIRS):
            trivial = change["status"] == "M" and repo.whitespace_only(base, head, path)
            change["whitespace_only"] = trivial
            note = " (whitespace only)" if trivial else ""
            flags.append(("constitution", f"{path} changed{note}."))
    return flags


def stale_notebook_cells(before, after):
    """Return the code cells whose source changed while their stored output did not.

    Parameters
    ----------
    before, after : str
        Notebook JSON at the two revisions.

    Returns
    -------
    list of int
        Zero-based cell indices in ``after``; ``[-1]`` when cells were added or
        removed and code changed while no stored output did.
    """
    try:
        old = json.loads(before)["cells"]
        new = json.loads(after)["cells"]
    except (TypeError, ValueError, KeyError):
        return []

    def code(cells):
        return [(i, c) for i, c in enumerate(cells) if c.get("cell_type") == "code"]

    def source(cell):
        text = cell.get("source", "")
        return "".join(text) if isinstance(text, list) else text

    old_code, new_code = code(old), code(new)
    if len(old_code) != len(new_code):
        sources_changed = [source(c) for _, c in old_code] != [
            source(c) for _, c in new_code
        ]
        outputs_same = [c.get("outputs") for _, c in old_code if c.get("outputs")] == [
            c.get("outputs") for _, c in new_code if c.get("outputs")
        ]
        return [-1] if sources_changed and outputs_same else []
    return [
        i
        for (_, a), (i, b) in zip(old_code, new_code, strict=True)
        if source(a) != source(b)
        and a.get("outputs")
        and a.get("outputs") == b.get("outputs")
    ]


def prose_line(line):
    """Normalize an added line for repetition checks, or return None for code."""
    text = line.strip().strip("#*>-\"', ").replace("\\n", " ").strip()
    if re.search(r" = |==|^(def|class|return|import|from|assert|self\.)\b", text):
        return None
    text = re.sub(r"\s+", " ", text).lower()
    return text if len(text.split()) >= REPEATED_TEXT_MIN_WORDS else None


def writing_facts(repo, base, head, changes, api, head_api):
    """Return docstring and prose signals for the change, with flags."""
    writing = {
        "missing_docstrings": [],
        "long_docstrings": [],
        "filler": [],
        "emoji": [],
        "repeated": [],
        "stale_notebooks": [],
    }
    prose = Counter()
    for name in api["added"]:
        record = head_api[name]
        if record["kind"] == "attribute":
            continue
        if record["docstring"] is None:
            writing["missing_docstrings"].append(name)
        elif record["docstring"] > LONG_DOCSTRING_WORDS:
            writing["long_docstrings"].append(f"{name} ({record['docstring']} words)")
    for change in changes:
        path = change["path"]
        if change["status"] == "D" or change["category"] == "vendored/generated":
            continue
        if not path.endswith((".py", ".md", ".rst", ".ipynb")):
            continue
        if path.endswith(".ipynb") and change["status"] == "M":
            stale = stale_notebook_cells(
                repo.read(base, change["old_path"]), repo.read(head, path)
            )
            if stale:
                cells = "cells added or removed" if stale == [-1] else f"cells {stale}"
                writing["stale_notebooks"].append(f"{path} ({cells})")
        for line in repo.added_lines(base, head, path):
            normalized = prose_line(line)
            if normalized:
                prose[normalized] += 1
            lowered = line.lower()
            writing["filler"] += [
                f"{path}: {p}" for p in FILLER_PHRASES if p in lowered
            ]
            if EMOJI.search(line):
                writing["emoji"].append(f"{path}: {line.strip()[:80]}")
    writing["repeated"] = [
        f"{text!r} (x{count})"
        for text, count in prose.most_common()
        if count >= REPEATED_TEXT_MIN_COUNT
    ]
    flags = [
        ("writing", f"{name} has no docstring.")
        for name in writing["missing_docstrings"]
    ]
    flags += [
        ("writing (info)", f"Same text added several times: {item}")
        for item in writing["repeated"]
    ]
    flags += [
        (
            "writing",
            f"Notebook code changed but its stored outputs did not: {item}; "
            "re-execute it.",
        )
        for item in writing["stale_notebooks"]
    ]
    flags += [
        ("writing (info)", f"Long docstring: {item}; check that every part is needed.")
        for item in writing["long_docstrings"]
    ]
    flags += [("writing", f"Emoji added: {item}") for item in writing["emoji"]]
    return writing, flags


def collect(base=None, head=None, path="."):
    """Collect the facts for a change.

    Parameters
    ----------
    base : str, optional
        Base revision. Defaults to the merge base of ``head`` with
        ``upstream/main`` (or ``origin/main``, or ``main``).
    head : str, optional
        Head revision. None (default) uses the working tree, untracked files
        included.
    path : str, optional
        Any directory inside the repository.

    Returns
    -------
    dict
        The facts, ready for :func:`render` or for JSON output.
    """
    repo = Repo(path)
    base = base or default_base(repo, head)
    changes = repo.changes(base, head)
    totals, handwritten, size_flags = diff_map(
        changes, repo.generated([c["path"] for c in changes])
    )
    scope, scope_flags = scope_facts(repo, base, head)
    constitution_flags = constitution_facts(repo, base, head, changes)
    api, head_api, api_flags = api_facts(repo, base, head, changes)
    new_tests, test_findings, test_flags = test_facts(repo, base, head, changes)
    writing, writing_flags = writing_facts(repo, base, head, changes, api, head_api)
    flags = (
        size_flags
        + scope_flags
        + constitution_flags
        + api_flags
        + test_flags
        + writing_flags
    )
    return {
        "base": repo.git("rev-parse", "--short", base).strip(),
        "head": "working tree"
        if head is None
        else repo.git("rev-parse", "--short", head).strip(),
        "changes": changes,
        "totals": totals,
        "handwritten_lines": handwritten,
        "scope": scope,
        "api": api,
        "new_tests": new_tests,
        "test_findings": test_findings,
        "writing": writing,
        "flags": [{"group": g, "message": m} for g, m in flags],
    }


def render(facts):
    """Render the facts as Markdown."""
    lines = [f"# Change facts: {facts['base']} -> {facts['head']}", "", "## Flags", ""]
    lines += [f"- **{f['group']}**: {f['message']}" for f in facts["flags"]] or [
        "- none"
    ]
    lines += ["", "## Diff map", "", "| Category | Lines changed |", "|---|---|"]
    for name, count in sorted(facts["totals"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {name} | {count} |")
    lines += [
        "",
        f"Hand-written lines ({', '.join(HANDWRITTEN)}): {facts['handwritten_lines']}",
        "",
    ]
    for change in facts["changes"]:
        counts = (
            "binary"
            if change["added"] is None
            else f"+{change['added']} -{change['deleted']}"
        )
        origin = (
            f" (from {change['old_path']})"
            if change["old_path"] != change["path"]
            else ""
        )
        lines.append(
            f"- {change['status']} `{change['path']}`{origin} {counts} [{change['category']}]"
        )
    scope, api = facts["scope"], facts["api"]
    lines += ["", "## Scope signals", ""]
    for key in (
        "new_subpackages",
        "new_dependencies",
        "removed_dependencies",
        "changed_dependencies",
    ):
        lines.append(
            f"- {key.replace('_', ' ').capitalize()}: {', '.join(scope[key]) or 'none'}"
        )
    lines += ["", "## Public API", ""]
    added = set(api["added"])
    shown = []
    for name in api["added"]:
        if name.rsplit(".", 1)[0] in added:
            continue
        members = sum(1 for other in added if other.startswith(name + "."))
        shown.append(f"`{name}`" + (f" (+{members} members)" if members else ""))
    lines.append(f"- Added: {', '.join(shown) or 'none'}")
    lines.append(f"- Removed: {', '.join(f'`{n}`' for n in api['removed']) or 'none'}")
    inherited = ", ".join(f"`{n}`" for n in api["inherited"]) or "none"
    lines.append(f"- No longer defined but inherited: {inherited}")
    moved = [f"`{a}` -> `{b}`" for a, b in api["moved"].items()]
    lines.append(f"- Moved: {', '.join(moved) or 'none'}")
    lines.append(
        "- Changed parameters:" if api["changed"] else "- Changed parameters: none"
    )
    lines += [f"  - `{c['name']}`: {'; '.join(c['changes'])}" for c in api["changed"]]
    lines += ["", "## Tests", "", f"- New tests: {len(facts['new_tests'])}"]
    lines += [
        f"- {t['kind']}: `{t['test']}` ({t['detail']})" for t in facts["test_findings"]
    ]
    writing = facts["writing"]
    filler = Counter(writing["filler"])
    lines += [
        "",
        "## Writing signals",
        "",
        f"- Missing docstrings: {', '.join(writing['missing_docstrings']) or 'none'}",
        f"- Long docstrings: {', '.join(writing['long_docstrings']) or 'none'}",
        f"- Repeated text: {', '.join(writing['repeated']) or 'none'}",
        f"- Notebooks with stale outputs: {', '.join(writing['stale_notebooks']) or 'none'}",
        "- Filler phrases (judge in context): "
        + (", ".join(f"{k} (x{v})" for k, v in sorted(filler.items())) or "none"),
    ]
    return "\n".join(lines) + "\n"


def main(argv=None):
    """Run the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", help="base revision (default: merge base with main)")
    parser.add_argument("--head", help="head revision (default: the working tree)")
    parser.add_argument("--json", action="store_true", help="print JSON")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    facts = collect(args.base, args.head)
    if args.json:
        sys.stdout.write(json.dumps(facts, indent=1, default=str) + "\n")
    else:
        sys.stdout.write(render(facts))


if __name__ == "__main__":
    main()
