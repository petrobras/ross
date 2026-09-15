# -*- coding: utf-8 -*-
"""Approximate reading of JS: what is an identifier and what only looks like one.

It strips strings, comments and regex literals, and gathers the declared names.

Inside a template literal only what is in `${...}` is code, and each piece comes
out separated by a space -- joining `${a}${b}` would give the non-existent name
`ab`, which was this tool's first result and almost all of its noise."""

import re

BEFORE_A_REGEX = set("(,=:[!&|?{};+-*%~^\n")


def code_only(t):
    out, i, n, previous = [], 0, len(t), "\n"
    while i < n:
        c = t[i]
        if c == "/" and i + 1 < n and t[i + 1] == "/":
            j = t.find("\n", i)
            i = n if j < 0 else j
        elif c == "/" and i + 1 < n and t[i + 1] == "*":
            j = t.find("*/", i)
            i = n if j < 0 else j + 2
        elif c == "/" and previous in BEFORE_A_REGEX:
            j, klass = i + 1, False
            while j < n:
                if t[j] == "\\":
                    j += 2
                    continue
                if t[j] == "[":
                    klass = True
                elif t[j] == "]":
                    klass = False
                elif t[j] == "/" and not klass:
                    break
                elif t[j] == "\n":
                    break
                j += 1
            j += 1  # the closing slash
            while j < n and t[j].isalpha():
                j += 1  # as flags
            out.append(" ")
            i = j
        elif c in "\"'":
            j = i + 1
            while j < n and t[j] != c:
                j += 2 if t[j] == "\\" else 1
            out.append(" ")
            i = j + 1
        elif c == "`":
            j = i + 1
            while j < n:
                if t[j] == "\\":
                    j += 2
                    continue
                if t[j] == "$" and j + 1 < n and t[j + 1] == "{":
                    k2, level = j + 2, 1
                    while k2 < n and level:
                        if t[k2] == "{":
                            level += 1
                        elif t[k2] == "}":
                            level -= 1
                        k2 += 1
                    out.append(" " + code_only(t[j + 2 : k2 - 1]) + " ")
                    j = k2
                    continue
                if t[j] == "`":
                    break
                j += 1
            i = j + 1
        else:
            out.append(c)
            if not c.isspace():
                previous = c
            i += 1
    return "".join(out)


IDENT = re.compile(r"(?<![\w$.])([A-Za-z_$][\w$]*)")

# Object literal key: `{ name: ...` or `, name: ...` or a line starting with
# `name:`. They are not a use of the name, and are deleted **in place**.
#
# Deleting by name, which is how this tool started, is what made it blind to
# the defect it existed to find: `core/state.js` declares the key
# `projectData:` in the state object, and so the loose `projectData` inside
# `getActiveData` -- a real ReferenceError -- fell out of the count.
POSITIONAL_KEY = re.compile(r"([{,\n])(\s*)([A-Za-z_$][\w$]*)(\s*:)")


def without_keys(source):
    return POSITIONAL_KEY.sub(
        lambda m: m.group(1) + m.group(2) + " " * len(m.group(3)) + m.group(4), source
    )


GLOBALS = set(
    """window document globalThis console Math JSON Object Array String Number
Boolean Date Promise Map Set WeakMap RegExp Error TypeError Infinity NaN undefined null
true false parseInt parseFloat isNaN isFinite encodeURIComponent decodeURIComponent
setTimeout setInterval clearTimeout clearInterval fetch FileReader Blob URL Event
CustomEvent AbortController AbortSignal localStorage sessionStorage alert confirm prompt
navigator location performance requestAnimationFrame Intl TextDecoder DOMParser
Plotly Sortable structuredClone
this arguments new typeof instanceof void delete in of from as let const var function
return if else for while do switch case default break continue try catch finally throw
class extends super async await yield export import static get set""".split()
)


# --- names declared inside a module -----------------------------------------
#
# This is not a scope analyser, and does not want to be: it errs towards the
# false alarm, which someone reads, and not towards silence. A name it fails
# to recognise as declared becomes a failure someone investigates; a loose
# name it lets through would become a ReferenceError in the user's hands.
DECLARES = [
    re.compile(r"(?:^|[\s;{}(])(?:async\s+)?function\s*\*?\s*([A-Za-z_$][\w$]*)"),
    re.compile(r"(?:^|[\s;{}(])(?:const|let|var)\s+([A-Za-z_$][\w$]*)"),
    re.compile(r"(?:^|[\s;{}(])(?:const|let|var)\s*\{([^}]*)\}"),
    re.compile(r"(?:^|[\s;{}(])(?:const|let|var)\s*\[([^\]]*)\]"),
    re.compile(r"catch\s*\(\s*([A-Za-z_$][\w$]*)"),
    re.compile(r"(?:^|[\s;{}(,])([A-Za-z_$][\w$]*)\s*=>"),
    # parametro desestruturado de arrow: `([a, b]) =>` e `({a, b}) =>`
    re.compile(r"\(\s*\[([^\]]*)\]\s*\)\s*=>"),
    re.compile(r"\(\s*\{([^}]*)\}\s*\)\s*=>"),
]
PARAMETERS = re.compile(
    r"(?:function\s*\*?\s*[A-Za-z_$][\w$]*\s*|function\s*|=>\s*)?"
    r"\(([^()]*)\)\s*(?:=>|\{)"
)
IMPORTED = re.compile(r"import\s*\{([^}]*)\}\s*from")


def _clear(chunk):
    chunk = chunk.split(":")[-1].split("=")[0].strip().lstrip(".")
    return chunk if re.fullmatch(r"[A-Za-z_$][\w$]*", chunk) else None


def declared(source):
    names = set()
    for pattern in DECLARES:
        for found in pattern.finditer(source):
            for chunk in found.group(1).split(","):
                name = _clear(chunk)
                if name:
                    names.add(name)
    for found in PARAMETERS.finditer(source):
        for chunk in found.group(1).split(","):
            name = _clear(chunk)
            if name:
                names.add(name)
    return names


def imported(raw):
    return {
        n.strip()
        for found in IMPORTED.finditer(raw)
        for n in found.group(1).split(",")
        if n.strip()
    }


def free_names(raw):
    """The identifiers the module uses without declaring, importing or inheriting."""
    source = code_only(raw)
    used = {m.group(1) for m in IDENT.finditer(without_keys(source))}
    return sorted(used - declared(source) - imported(raw) - GLOBALS)
