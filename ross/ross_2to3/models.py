"""Convert rotor and element files saved by ROSS 2 to the ROSS 3 format.

The files written by ``Rotor.save()`` and ``Element.save()`` hold one
section per element named ``<ClassName>_<tag>``. ROSS 2 saved the solver
based bearings (``PlainJournal``, ``TiltingPad``, ``ThrustPad``, ...) with
their model parameters next to the solved coefficient table; ROSS 3 saves
them as plain ``BearingElement`` coefficient tables, and the seals keep their
class with the renamed parameters. This module applies the same policy so a
converted file loads without re-running any solver.
"""

import json
import math
from copy import deepcopy
from pathlib import Path

import toml

from ross.ross_2to3.renames import (
    CLASS_RENAMES,
    DEG_TO_RAD,
    DEGC_TO_KELVIN,
    DOUBLE,
    FREQUENCY_TABLE_CLASSES,
    HYBRID_SEAL_NESTED,
    INT_TO_BOOL,
    SEAL_CLASSES,
    SOLVER_BEARING_CLASSES,
    TABLE_KEYS,
)
from ross.ross_2to3.report import CHANGED, CHECK, ERROR, VERIFIED

MODEL_SUFFIXES = (".toml", ".json")

RESERVED_KEYS = ("ross_version", "parameters")

LABYRINTH_SCALAR_KEYS = ("pitch", "radial_clearance", "tooth_height", "tooth_width")


def _scale(value, factor):
    if isinstance(value, list):
        return [_scale(item, factor) for item in value]
    return value * factor


def _map_numbers(value, function):
    if isinstance(value, list):
        return [_map_numbers(item, function) for item in value]
    return function(value)


def convert_value(value, conversion):
    """Apply one of the ``VALUE_CONVERSIONS`` to a saved (SI) value.

    Parameters
    ----------
    value : number, list or bool
        Value read from the file.
    conversion : str
        Key of the conversion (``"double"``, ``"deg_to_rad"``,
        ``"degc_to_kelvin"`` or ``"int_to_bool"``).

    Returns
    -------
    The converted value.

    Examples
    --------
    >>> convert_value(0.0725, "double")
    0.145
    >>> convert_value(1, "int_to_bool")
    True
    """
    if conversion == DOUBLE:
        return _scale(value, 2)
    if conversion == DEG_TO_RAD:
        return _map_numbers(value, math.radians)
    if conversion == DEGC_TO_KELVIN:
        return _map_numbers(value, lambda v: v + 273.15)
    if conversion == INT_TO_BOOL:
        return bool(value)
    raise ValueError(f"Unknown value conversion {conversion!r}")


def is_model(data):
    """Tell whether a loaded TOML/JSON dictionary looks like a ROSS file."""
    if not isinstance(data, dict):
        return False
    if any(key in data for key in RESERVED_KEYS):
        return True
    return any(
        isinstance(value, dict) and key[:1].isupper() and "_" in key
        for key, value in data.items()
    )


def split_section_name(name):
    """Split ``<ClassName>_<tag>`` into its class name and tag."""
    class_name, _, tag = name.partition("_")
    return class_name, tag


def _rename_section(section, table, report, path, location):
    converted = {}
    for key, value in section.items():
        change = table.get(key)
        if change is None:
            converted[key] = value
            continue
        if change.new is None and change.values is None:
            report.add(path, location, CHANGED, f"dropped {key} ({change.note})")
            continue
        new_key = key if change.new is None else change.new
        new_value = value
        if change.value is not None:
            new_value = convert_value(value, change.value)
        if change.values is not None and value in change.values:
            new_value = change.values[value]
        converted[new_key] = new_value
        if new_key != key:
            message = f"{key} -> {new_key}"
            if change.value is not None:
                message += f" ({value!r} -> {new_value!r})"
        else:
            message = f"{key}: {value!r} -> {new_value!r}"
        report.add(path, location, CHANGED, message)
    return converted


def _collapse_constant_lists(section, keys, report, path, location):
    """Turn the per-node arrays ROSS 2 stored for scalar inputs back into scalars."""
    for key in keys:
        value = section.get(key)
        if not isinstance(value, list) or not value:
            continue
        if all(item == value[0] for item in value):
            section[key] = value[0]
        else:
            report.add(
                path,
                location,
                CHECK,
                f"{key} is a non-uniform array; ROSS 3 expects one value per seal",
            )
    return section


def _flatten_kwargs(section):
    kwargs = section.pop("kwargs", None)
    if isinstance(kwargs, dict):
        for key, value in kwargs.items():
            section.setdefault(key, value)
    return section


def _coefficient_table(section, keep_frequency=False):
    table = {key: value for key, value in section.items() if key in TABLE_KEYS}
    if not keep_frequency and "frequency" in table and "speed" not in table:
        table["speed"] = table.pop("frequency")
    return dict(sorted(table.items()))


def convert_section(name, section, report, path):
    """Convert one ``<ClassName>_<tag>`` section.

    Parameters
    ----------
    name : str
        Section name as saved by ROSS 2.
    section : dict
        Section content.
    report : ross.ross_2to3.report.Report
        Findings are appended here.
    path : str or pathlib.Path
        File name used in the report.

    Returns
    -------
    name : str
        Section name to use in the ROSS 3 file.
    section : dict
        Converted content.
    """
    class_name, tag = split_section_name(name)
    section = _flatten_kwargs(dict(section))

    if class_name in SOLVER_BEARING_CLASSES or class_name in FREQUENCY_TABLE_CLASSES:
        if class_name == "MagneticBearingElement":
            return name, section
        keep_frequency = class_name in FREQUENCY_TABLE_CLASSES
        table = _coefficient_table(section, keep_frequency=keep_frequency)
        dropped = sorted(set(section) - set(table) - {"frequency"})
        new_name = f"BearingElement_{tag}"
        report.add(
            path,
            name,
            CHANGED,
            f"saved as the coefficient table {new_name}, as ROSS 3 does; "
            f"model parameters dropped: {', '.join(dropped)}",
        )
        return new_name, table

    if class_name in ("BearingElement", "SealElement", "ST_BearingElement"):
        table = _coefficient_table(section)
        dropped = sorted(set(section) - set(table) - {"frequency"})
        if "speed" in table and "frequency" in section:
            report.add(path, name, CHANGED, "frequency -> speed")
        if dropped:
            report.add(
                path,
                name,
                CHANGED,
                "dropped the ROSS 2 model parameters kept next to the table: "
                + ", ".join(dropped),
            )
        return name, table

    if class_name in SEAL_CLASSES:
        if class_name == "LabyrinthSeal":
            section = _collapse_constant_lists(
                section, LABYRINTH_SCALAR_KEYS, report, path, name
            )
        converted = _rename_section(
            section, CLASS_RENAMES[class_name], report, path, name
        )
        for key, nested_class in HYBRID_SEAL_NESTED.items():
            if isinstance(converted.get(key), dict):
                converted[key] = _rename_section(
                    converted[key],
                    CLASS_RENAMES[nested_class],
                    report,
                    path,
                    f"{name}.{key}",
                )
        return name, converted

    return name, section


def convert_model_data(data, report, path="<data>", version="3"):
    """Convert the dictionary of a ROSS 2 rotor or element file.

    Parameters
    ----------
    data : dict
        Content of the file as returned by ``toml.load`` or ``json.load``.
    report : ross.ross_2to3.report.Report
        Findings are appended here.
    path : str or pathlib.Path, optional
        File name used in the report.
    version : str, optional
        Value written to ``ross_version``.

    Returns
    -------
    dict
        The converted content, in the original section order.

    Examples
    --------
    >>> from ross.ross_2to3.report import Report
    >>> data = {"BearingElement_b0": {"n": 0, "kxx": [1e6, 1e6], "cxx": [0, 0],
    ...         "frequency": [0.0, 100.0], "journal_radius": 0.1}}
    >>> converted = convert_model_data(data, Report())
    >>> sorted(converted["BearingElement_b0"])
    ['cxx', 'kxx', 'n', 'speed']
    """
    converted = {}
    for name, section in data.items():
        if name == "ross_version":
            converted[name] = version
            if section != version:
                report.add(path, name, CHANGED, f"{section!r} -> {version!r}")
        elif (
            name in RESERVED_KEYS
            or name.startswith("_")
            or not isinstance(section, dict)
        ):
            converted[name] = deepcopy(section)
        else:
            new_name, new_section = convert_section(
                name, deepcopy(section), report, path
            )
            converted[new_name] = new_section
    return converted


def _header_comments(text):
    lines = []
    for line in text.splitlines():
        if line.startswith("#"):
            lines.append(line)
        elif line.strip():
            break
    return lines


def render_model(data, suffix, header_comments=(), note=None):
    """Serialize converted data to TOML or JSON text.

    Parameters
    ----------
    data : dict
        Converted content.
    suffix : str
        ``".toml"`` or ``".json"``.
    header_comments : iterable of str, optional
        Comment lines kept from the original TOML file.
    note : str, optional
        Extra comment (TOML) or ``_note`` prefix (JSON) recording the conversion.
    """
    if suffix.lower() == ".json":
        if note is not None:
            existing = data.get("_note")
            data = {
                "_note": f"{note}\n{existing}" if existing else note,
                **{k: v for k, v in data.items() if k != "_note"},
            }
        return json.dumps(data, indent=2) + "\n"
    header = list(header_comments)
    if note is not None:
        header.append(f"# {note}")
    body = toml.dumps(data)
    return ("\n".join(header) + "\n\n" if header else "") + body


def load_model_text(text, suffix):
    """Parse TOML or JSON text into a dictionary."""
    if suffix.lower() == ".json":
        return json.loads(text)
    return toml.loads(text)


def convert_model_text(text, suffix, report, path="<data>", version="3"):
    """Convert the text of a ROSS 2 file and return the ROSS 3 text.

    Returns
    -------
    text : str or None
        Converted file content, or None when the file is not a ROSS model.
    """
    data = load_model_text(text, suffix)
    if not is_model(data):
        return None
    old_version = data.get("ross_version", "2.x")
    converted = convert_model_data(data, report, path=path, version=version)
    note = f"Converted from ROSS {old_version} to ROSS {version} by ross_2to3."
    return render_model(
        converted, suffix, header_comments=_header_comments(text), note=note
    )


def check_model_file(path, report, label=None):
    """Load a converted file with the installed ROSS to prove it is valid.

    Rotor files (those with a ``parameters`` section) go through
    ``Rotor.load``; element files load every section with its class.

    Parameters
    ----------
    path : str or pathlib.Path
        Converted file to load.
    report : ross.ross_2to3.report.Report
        Findings are appended here.
    label : str or pathlib.Path, optional
        Name reported instead of ``path`` (the original file when the
        converted one is a temporary copy).

    Returns
    -------
    bool
        True when the file loads.
    """
    import ross
    from ross.utils import load_data

    path = Path(path)
    label = path if label is None else label
    try:
        data = load_data(path)
        if "parameters" in data:
            ross.Rotor.load(path)
        else:
            for name, section in data.items():
                if name in RESERVED_KEYS or name.startswith("_"):
                    continue
                class_name, _ = split_section_name(name)
                getattr(ross, class_name).read_toml_data(section)
    except Exception as exc:
        report.add(label, "", ERROR, f"converted file does not load: {exc!r}")
        return False
    report.add(
        label, "", VERIFIED, f"converted file loads with ROSS {ross.__version__}"
    )
    return True
