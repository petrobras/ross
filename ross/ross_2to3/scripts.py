"""Rewrite Python scripts and notebooks from the ROSS 2 API to ROSS 3.

The source is parsed with :mod:`ast` and edited in place through the node
positions, so formatting and comments are preserved. Constructor calls of the
classes listed in :data:`ross.ross_2to3.renames.CLASS_RENAMES` have their
keyword arguments renamed and, where the convention changed (radius to
diameter, degrees to radians, ...), their literal values converted. Anything
that cannot be rewritten safely is reported with its line number.
"""

import ast
import json
import math
import re

from ross.ross_2to3.renames import (
    CLASS_RENAMES,
    DEG_TO_RAD,
    DEGC_TO_KELVIN,
    DOUBLE,
    HYBRID_SEAL_NESTED,
    INT_TO_BOOL,
    METHOD_RENAMES,
    MOVED_MODULES,
    REMOVED_MODULES,
    REMOVED_NAMES,
    REORDERED_CLASSES,
)
from ross.ross_2to3.report import CHANGED, CHECK, MANUAL, SKIPPED

SCRIPT_SUFFIXES = (".py", ".ipynb")

Q_IMPORT = "from ross.units import Q_"

MAGIC_MARKER = "#__ross_2to3_magic__"

ATOMS = (ast.Name, ast.Attribute, ast.Call, ast.Subscript, ast.Constant)


class Source:
    """Source text with ast-position lookups and deferred, non-overlapping edits."""

    def __init__(self, text):
        self.text = text
        self.lines = text.splitlines(keepends=True) or [""]
        self.line_starts = []
        position = 0
        for line in self.lines:
            self.line_starts.append(position)
            position += len(line)
        self.edits = []

    def offset(self, lineno, col_offset):
        """Return the character offset of an ast (line, utf-8 column) position."""
        line = self.lines[lineno - 1]
        column = len(line.encode("utf-8")[:col_offset].decode("utf-8", "ignore"))
        return self.line_starts[lineno - 1] + column

    def span(self, node):
        """Return the (start, end) character offsets of a node."""
        return (
            self.offset(node.lineno, node.col_offset),
            self.offset(node.end_lineno, node.end_col_offset),
        )

    def segment(self, node):
        """Return the source text of a node."""
        start, end = self.span(node)
        return self.text[start:end]

    def replace(self, start, end, new_text):
        """Schedule the replacement of ``text[start:end]``."""
        self.edits.append((start, end, new_text))

    def remove_item(self, start, end):
        """Schedule the removal of a call argument or dict item and its comma."""
        text = self.text
        line_start = text.rfind("\n", 0, start) + 1
        line_end = text.find("\n", end)
        line_end = len(text) if line_end == -1 else line_end
        before = text[line_start:start]
        after = text[end:line_end]
        if before.strip() == "" and re.fullmatch(r"[ \t]*,?[ \t]*", after):
            self.replace(line_start, min(line_end + 1, len(text)), "")
            return
        following = re.match(r"[ \t]*,[ \t]*", after)
        if following:
            self.replace(start, end + following.end(), "")
            return
        preceding = re.search(r",\s*$", text[:start])
        if preceding:
            self.replace(preceding.start(), end, "")
        else:
            self.replace(start, end, "")

    def merged_edits(self):
        """Return the edits sorted, with overlapping removals merged into one.

        A removal that leaves a dangling comma before a closing bracket
        (the last items of a call were dropped) is widened to eat that comma.
        """
        edits = sorted(self.edits, key=lambda edit: edit[:2])
        while True:
            merged = []
            for start, end, new_text in edits:
                if merged and start < merged[-1][1]:
                    previous_start, previous_end, previous_text = merged[-1]
                    if previous_text or new_text:
                        raise ValueError("overlapping edits")
                    merged[-1] = (previous_start, max(previous_end, end), "")
                    continue
                merged.append((start, end, new_text))
            widened = [self._eat_dangling_comma(edit) for edit in merged]
            if widened == edits:
                return widened
            edits = widened

    def _eat_dangling_comma(self, edit):
        start, end, new_text = edit
        if new_text or not self.text[end:].lstrip().startswith((")", "]", "}")):
            return edit
        preceding = re.search(r",\s*$", self.text[:start])
        if preceding is None:
            return edit
        return (preceding.start(), end, "")

    def apply(self):
        """Return the text with every scheduled edit applied."""
        pieces = []
        position = 0
        for start, end, new_text in self.merged_edits():
            pieces.append(self.text[position:start])
            pieces.append(new_text)
            position = end
        pieces.append(self.text[position:])
        return "".join(pieces)


def _is_number(node):
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        node = node.operand
    return isinstance(node, ast.Constant) and type(node.value) in (int, float)


def _is_number_sequence(node):
    return isinstance(node, (ast.List, ast.Tuple)) and all(
        _is_number(item) for item in node.elts
    )


def _number_value(node):
    sign = 1
    if isinstance(node, ast.UnaryOp):
        sign = -1 if isinstance(node.op, ast.USub) else 1
        node = node.operand
    return sign * node.value


def format_number(value, like):
    """Format a converted number in the style of the literal it replaces."""
    if isinstance(like, int) and float(value).is_integer():
        return str(int(value))
    return repr(float(value))


def _double_text(node):
    value = _number_value(node)
    return format_number(value * 2, value)


class Converter:
    """Convert one Python source to the ROSS 3 API."""

    def __init__(self, text, path, report, location_prefix="", q_accessor=None):
        self.source = Source(text)
        self.path = path
        self.report = report
        self.location_prefix = location_prefix
        self.tree = ast.parse(text)
        self.aliases = {}
        self.module_aliases = {}
        self.q_accessor = q_accessor
        self.dict_literals = {}
        self.converted_dicts = set()
        self.needs_q_import = False
        self.reported = set()
        self._scan_names()

    def _scan_names(self):
        accessor = None
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    self.aliases[alias.asname or alias.name] = alias.name
                    if alias.name == "Q_" and node.module in ("ross.units", "ross"):
                        accessor = alias.asname or "Q_"
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    self.module_aliases[alias.asname or alias.name] = alias.name
                    if alias.name == "ross" and accessor is None:
                        accessor = f"{alias.asname or 'ross'}.Q_"
            elif isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name):
                    if target.id == "Q_":
                        accessor = "Q_"
                    if isinstance(node.value, ast.Dict):
                        self.dict_literals[target.id] = node.value
        if self.q_accessor is None:
            self.q_accessor = accessor

    def location(self, node):
        """Return the report location of a node."""
        return f"{self.location_prefix}line {node.lineno}"

    def note(self, node, level, message):
        """Record a finding once per (line, message)."""
        key = (node.lineno, message)
        if key in self.reported:
            return
        self.reported.add(key)
        self.report.add(self.path, self.location(node), level, message)

    def resolve(self, func):
        """Return the ROSS name a call target refers to."""
        if isinstance(func, ast.Name):
            return self.aliases.get(func.id, func.id)
        if isinstance(func, ast.Attribute):
            return func.attr
        return None

    def is_quantity(self, node):
        """Tell whether a node is a ``Q_(...)`` call."""
        return isinstance(node, ast.Call) and self.resolve(node.func) == "Q_"

    def run(self):
        """Apply every rewrite and return the converted text."""
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call):
                self.convert_call(node)
            elif isinstance(node, ast.ImportFrom):
                self.convert_import_from(node)
            elif isinstance(node, ast.Import):
                self.convert_import(node)
            elif isinstance(node, ast.Name) and node.id in REMOVED_NAMES:
                self.note(
                    node, MANUAL, f"{node.id} was removed: {REMOVED_NAMES[node.id]}"
                )
            elif isinstance(node, ast.Attribute) and node.attr in REMOVED_NAMES:
                self.note(
                    node, MANUAL, f"{node.attr} was removed: {REMOVED_NAMES[node.attr]}"
                )
        if self.needs_q_import:
            self.add_q_import()
        return self.source.apply()

    def convert_call(self, node):
        """Rename and convert the keyword arguments of one call."""
        name = self.resolve(node.func)
        if name in CLASS_RENAMES:
            table = CLASS_RENAMES[name]
        elif isinstance(node.func, ast.Attribute) and name in METHOD_RENAMES:
            table = METHOD_RENAMES[name]
        else:
            return
        if name in REORDERED_CLASSES and len(node.args) > 1:
            self.note(
                node,
                CHECK,
                f"{name}: positional arguments were reordered in ROSS 3, "
                "pass them by keyword",
            )
        for keyword in node.keywords:
            if keyword.arg is None:
                self.note(
                    node,
                    MANUAL,
                    f"{name}(**{self.source.segment(keyword.value)}): "
                    "rename the keys of the unpacked dictionary by hand",
                )
                continue
            if keyword.arg in HYBRID_SEAL_NESTED and name == "HybridSeal":
                self.convert_nested(keyword, HYBRID_SEAL_NESTED[keyword.arg])
                continue
            change = table.get(keyword.arg)
            if change is None:
                continue
            start = self.source.offset(keyword.lineno, keyword.col_offset)
            self.apply_change(
                owner=name,
                old=keyword.arg,
                change=change,
                name_span=(start, start + len(keyword.arg)),
                value=keyword.value,
                item_span=(start, self.source.span(keyword.value)[1]),
                node=keyword,
            )

    def convert_nested(self, keyword, nested_class):
        """Convert the dict passed as ``hole_pattern_parameters`` / ``labyrinth_parameters``."""
        value = keyword.value
        if isinstance(value, ast.Name) and value.id in self.dict_literals:
            value = self.dict_literals[value.id]
        if not isinstance(value, ast.Dict):
            self.note(
                keyword,
                MANUAL,
                f"HybridSeal({keyword.arg}={self.source.segment(keyword.value)}): "
                f"not a dict literal, rename its keys as for {nested_class}",
            )
            return
        if id(value) in self.converted_dicts:
            return
        self.converted_dicts.add(id(value))
        table = CLASS_RENAMES[nested_class]
        for key, item in zip(value.keys, value.values, strict=True):
            if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                continue
            change = table.get(key.value)
            if change is None:
                continue
            key_start, key_end = self.source.span(key)
            self.apply_change(
                owner=f"HybridSeal.{keyword.arg}",
                old=key.value,
                change=change,
                name_span=(key_start, key_end),
                value=item,
                item_span=(key_start, self.source.span(item)[1]),
                node=key,
                quote=self.source.text[key_start],
            )

    def apply_change(
        self, owner, old, change, name_span, value, item_span, node, quote=None
    ):
        """Schedule the edits for one renamed, converted or removed parameter."""
        if change.new is None and change.values is None:
            self.source.remove_item(*item_span)
            self.note(node, CHANGED, f"{owner}: dropped {old} ({change.note})")
            return
        if change.new == "fxs_load, fys_load":
            self.split_load(owner, node, value, item_span)
            return
        if change.new is not None and change.new != old:
            new_name = change.new if quote is None else f"{quote}{change.new}{quote}"
            self.source.replace(*name_span, new_name)
            message = f"{owner}: {old} -> {change.new}"
            if change.note:
                message += f" ({change.note})"
            self.note(node, CHANGED, message)
        if change.value is not None:
            self.convert_value(owner, old, change, value, node)
        if change.values is not None:
            self.revalue(owner, old, change, value, node)

    def split_load(self, owner, node, value, item_span):
        """Rewrite ``load=[fx, fy]`` as ``fxs_load=fx, fys_load=fy``."""
        if isinstance(value, (ast.List, ast.Tuple)) and len(value.elts) == 2:
            fx, fy = (self.source.segment(item) for item in value.elts)
            level = CHANGED
        else:
            text = self.source.segment(value)
            fx, fy = f"{text}[0]", f"{text}[1]"
            level = CHECK
        self.source.replace(*item_span, f"fxs_load={fx}, fys_load={fy}")
        self.note(node, level, f"{owner}: load -> fxs_load, fys_load")

    def revalue(self, owner, old, change, value, node):
        """Map an enumerated string value onto the ROSS 3 vocabulary."""
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            if value.value in change.values:
                quote = self.source.segment(value)[0]
                new_value = change.values[value.value]
                self.source.replace(
                    *self.source.span(value), f"{quote}{new_value}{quote}"
                )
                self.note(
                    node, CHANGED, f'{owner}: {old}="{value.value}" -> "{new_value}"'
                )
            return
        pairs = ", ".join(f'"{k}" -> "{v}"' for k, v in change.values.items())
        self.note(node, CHECK, f"{owner}: {old} values were renamed ({pairs})")

    def convert_value(self, owner, old, change, value, node):
        """Convert a literal value whose convention changed."""
        new = change.new or old
        if change.value == DOUBLE:
            self.double(owner, old, new, value, node)
        elif change.value in (DEG_TO_RAD, DEGC_TO_KELVIN):
            unit = "deg" if change.value == DEG_TO_RAD else "degC"
            self.to_quantity(owner, old, new, value, node, unit)
        elif change.value == INT_TO_BOOL:
            self.to_bool(owner, old, new, value, node)

    def double(self, owner, old, new, value, node):
        target = value
        if self.is_quantity(value) and value.args:
            target = value.args[0]
        if _is_number(target):
            self.source.replace(*self.source.span(target), _double_text(target))
        elif _is_number_sequence(target):
            for item in target.elts:
                self.source.replace(*self.source.span(item), _double_text(item))
        else:
            text = self.source.segment(value)
            wrapped = f"2 * {text}" if isinstance(value, ATOMS) else f"2 * ({text})"
            self.source.replace(*self.source.span(value), wrapped)
            self.note(
                node,
                CHECK,
                f"{owner}: {new} = 2 * {old}; simplify if the variable can hold a diameter",
            )

    def to_quantity(self, owner, old, new, value, node, unit):
        if self.is_quantity(value):
            return
        if _is_number(value) or _is_number_sequence(value):
            text = self.source.segment(value)
            if self.q_accessor is None:
                self.needs_q_import = True
                accessor = "Q_"
            else:
                accessor = self.q_accessor
            self.source.replace(
                *self.source.span(value), f'{accessor}({text}, "{unit}")'
            )
            self.note(
                node, CHANGED, f'{owner}: {old}={text} -> {new}=Q_({text}, "{unit}")'
            )
            return
        was, expects = (
            ("degrees", "radians") if unit == "deg" else ("degrees Celsius", "kelvin")
        )
        self.note(
            node,
            MANUAL,
            f"{owner}: {old} took plain numbers in {was}; {new} expects {expects} "
            f"or a pint quantity — check {self.source.segment(value)}",
        )

    def to_bool(self, owner, old, new, value, node):
        if isinstance(value, ast.Constant) and value.value in (0, 1, True, False):
            self.source.replace(*self.source.span(value), str(bool(value.value)))
            return
        text = self.source.segment(value)
        self.source.replace(*self.source.span(value), f"bool({text})")
        self.note(node, CHECK, f"{owner}: {new}=bool({text})")

    def convert_import_from(self, node):
        module = node.module or ""
        if any(
            module == removed or module.startswith(removed + ".")
            for removed in REMOVED_MODULES
        ):
            self.note(
                node,
                MANUAL,
                f"{module} was removed: {REMOVED_MODULES[_removed_root(module)]}",
            )
            return
        if module in MOVED_MODULES:
            statement = self.source.segment(node)
            match = re.search(
                r"(from\s+)" + re.escape(module) + r"(\s+import)", statement
            )
            if match:
                start = self.source.span(node)[0]
                self.source.replace(
                    start + match.start(),
                    start + match.end(),
                    f"{match.group(1)}{MOVED_MODULES[module]}{match.group(2)}",
                )
                self.note(node, CHANGED, f"{module} -> {MOVED_MODULES[module]}")

    def convert_import(self, node):
        statement = self.source.segment(node)
        start = self.source.span(node)[0]
        for alias in node.names:
            module = alias.name
            if any(
                module == removed or module.startswith(removed + ".")
                for removed in REMOVED_MODULES
            ):
                self.note(
                    node,
                    MANUAL,
                    f"{module} was removed: {REMOVED_MODULES[_removed_root(module)]}",
                )
                continue
            if module in MOVED_MODULES:
                match = re.search(
                    r"(?<![\w.])" + re.escape(module) + r"(?![\w.])", statement
                )
                if match:
                    self.source.replace(
                        start + match.start(),
                        start + match.end(),
                        MOVED_MODULES[module],
                    )
                    message = f"{module} -> {MOVED_MODULES[module]}"
                    if alias.asname is None:
                        message += "; update attribute access through the old path"
                    self.note(node, CHANGED if alias.asname else CHECK, message)

    def add_q_import(self):
        """Insert ``from ross.units import Q_`` after the last top-level import."""
        last_import = None
        for statement in self.tree.body:
            if isinstance(statement, (ast.Import, ast.ImportFrom)):
                last_import = statement
        if last_import is not None:
            end = self.source.span(last_import)[1]
            self.source.replace(end, end, f"\n{Q_IMPORT}")
            location = last_import
        else:
            body = self.tree.body
            docstring = (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(getattr(body[0], "value", None), ast.Constant)
                and isinstance(body[0].value.value, str)
            )
            if docstring:
                end = self.source.span(body[0])[1]
                self.source.replace(end, end, f"\n{Q_IMPORT}")
                location = body[0]
            else:
                self.source.replace(0, 0, f"{Q_IMPORT}\n")
                location = None
        line = (
            f"{self.location_prefix}line {location.lineno}"
            if location
            else f"{self.location_prefix}line 1"
        )
        self.report.add(self.path, line, CHANGED, f"added `{Q_IMPORT}`")


def _removed_root(module):
    for removed in REMOVED_MODULES:
        if module == removed or module.startswith(removed + "."):
            return removed
    return module


def _mask_magics(text):
    lines = []
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped[:1] in ("%", "!"):
            indent = line[: len(line) - len(stripped)]
            line = f"{indent}{MAGIC_MARKER}{stripped}"
        lines.append(line)
    return "".join(lines)


def convert_source(
    text, path, report, location_prefix="", q_accessor=None, magics=False
):
    """Convert Python source text to the ROSS 3 API.

    Parameters
    ----------
    text : str
        Python source.
    path : str or pathlib.Path
        File name used in the report.
    report : ross.ross_2to3.report.Report
        Findings are appended here.
    location_prefix : str, optional
        Prefix for report locations (e.g. ``"cell 3, "`` for notebooks).
    q_accessor : str, optional
        Expression that gives ``Q_`` in this source when it is defined
        elsewhere (other notebook cells); detected from the imports otherwise.
    magics : bool, optional
        Tolerate IPython ``%magic`` / ``!shell`` lines.

    Returns
    -------
    str
        The converted source (unchanged when nothing applies or on a syntax error).

    Examples
    --------
    >>> from ross.ross_2to3.report import Report
    >>> print(convert_source("rs.LabyrinthSeal(n=0, shaft_radius=0.0725, frequency=w)", "s.py", Report()))
    rs.LabyrinthSeal(n=0, shaft_diameter=0.145, speed=w)
    """
    working = _mask_magics(text) if magics else text
    try:
        converter = Converter(working, path, report, location_prefix, q_accessor)
    except SyntaxError as exc:
        report.add(
            path,
            f"{location_prefix}line {exc.lineno}",
            SKIPPED,
            f"could not parse: {exc.msg}",
        )
        return text
    converted = converter.run()
    if magics:
        converted = converted.replace(MAGIC_MARKER, "")
    return converted


def notebook_q_accessor(cells):
    """Find how ``Q_`` is reachable across the code cells of a notebook."""
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        source = _cell_source(cell)
        try:
            converter = Converter(_mask_magics(source), "", None)
        except SyntaxError:
            continue
        if converter.q_accessor is not None:
            return converter.q_accessor
    return None


def _cell_source(cell):
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def convert_notebook(text, path, report):
    """Convert the code cells of a Jupyter notebook.

    Returns
    -------
    str
        The notebook JSON (unchanged text when no cell needed conversion).
    """
    notebook = json.loads(text)
    cells = notebook.get("cells", [])
    q_accessor = notebook_q_accessor(cells)
    changed = False
    for index, cell in enumerate(cells, start=1):
        if cell.get("cell_type") != "code":
            continue
        source = _cell_source(cell)
        converted = convert_source(
            source,
            path,
            report,
            location_prefix=f"cell {index}, ",
            q_accessor=q_accessor,
            magics=True,
        )
        if converted != source:
            cell["source"] = converted.splitlines(keepends=True)
            changed = True
    if not changed:
        return text
    return json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"
