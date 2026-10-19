# -*- coding: utf-8 -*-
"""Build the ROSS rotor from the project the interface describes.

Nothing here belongs to Flask: this is the translation between the form and the
library. It left app.py in Phase 2 so that the transport layer (api/) only
transports, and so that this logic can be exercised without starting a server.

Elements already built live in a bounded, locked cache (domain/cache.py).
Sharing them between rotors is safe for three reasons written in that module's
header -- and pinned by tests, since they are three reasons a future ROSS
release could remove.
"""

import ast
import hashlib
import json
from typing import NamedTuple

import numpy as np
import ross as rs
from ross.units import Q_

from .cache import ELEMENT_CACHE
from .element_registry import (
    VIEW_FIELDS,
    resolve_constructor,
    ross_class_name,
    ross_constructor,
    takes_units,
)
from .legacy import migrate_element
from .material_names import material_key, ross_material_name, validate_materials
from .node_resolver import effective_nodes, listed_nodes, validate_node_topology
from .units import INT_PARAMETERS, UNITS_MAPPING
from ross.interface.services.expressions import safe_math_eval


def extract_kwargs(d, mat_dict, element_type, ignore_keys=["element_type", "n"]):
    kwargs = {}

    unit_map = UNITS_MAPPING.get(element_type, {})

    for k, v in d.items():
        if k in ignore_keys or k in VIEW_FIELDS or k.endswith("_unit"):
            continue
        if v is None:
            continue

        if k == "material":
            if str(v).strip().lower() in ("", "default (steel)"):
                kwargs[k] = rs.materials.steel
                continue

            kwargs[k] = (
                mat_dict.get(material_key(v), list(mat_dict.values())[0])
                if mat_dict
                else rs.materials.steel
            )
            continue

        if isinstance(v, str):
            v_strip = v.strip()

            # "None" is what a null became in projects imported before the
            # importer learned to leave nulls out; either way it means unset.
            if v_strip == "" or v_strip == "None":
                continue

            if k == "initial_position":
                try:
                    kwargs[k] = tuple(
                        float(x.strip())
                        for x in v_strip.replace("(", "").replace(")", "").split(",")
                    )
                except Exception:
                    kwargs[k] = (0.1, -0.1)
                continue

            if v_strip.startswith("[") or v_strip.startswith("{"):
                try:
                    val_parsed = ast.literal_eval(v_strip)
                    if isinstance(val_parsed, list):
                        # dtype=float also takes a nested list, which is how
                        # a 2-D coefficient table (speed x frequency) is typed.
                        val_parsed = np.array(val_parsed, dtype=float)
                        unit = d.get(f"{k}_unit", unit_map.get(k))
                        if unit:
                            kwargs[k] = Q_(val_parsed, unit).to_base_units()
                        else:
                            kwargs[k] = val_parsed
                    else:
                        kwargs[k] = val_parsed
                    continue
                except Exception:
                    pass

            if v_strip.lower() == "true":
                kwargs[k] = True
                continue
            if v_strip.lower() == "false":
                kwargs[k] = False
                continue

            try:
                val_num = float(v_strip)
            except ValueError:
                try:
                    val_num = safe_math_eval(v_strip)
                except ValueError:
                    kwargs[k] = v_strip
                    continue

            if k in INT_PARAMETERS:
                val_num = int(val_num)

            unit = d.get(f"{k}_unit", unit_map.get(k))
            if unit:
                kwargs[k] = Q_(val_num, unit).to_base_units()
            else:
                kwargs[k] = val_num
            continue
        else:
            kwargs[k] = v

    return kwargs


class Assembled(NamedTuple):
    """A built rotor, and where each of its elements came from.

    `placed` maps each category of the project to the ROSS elements built from
    it, **in the order of the project's list** -- the order the screen numbers
    its rows. The rotor itself cannot say this: `Rotor` sorts its elements by
    node, and the element cache (domain/cache.py) shares one element among
    every rotor that describes it the same way, so neither position nor
    identity inside the rotor leads back to a row. It is `domain/rotor_scene.py`
    that needs it, so a click on a part in the 3D view opens the right row.

    For a MultiRotor, `placed` is None, `halves` holds the two halves, each an
    `Assembled` of its own, and `coupling` what joins them: the coupled nodes,
    the side the driven line is drawn on and the orientation angle, as read
    from the project.

    `geometry` keeps the dimensions an element was built from when ROSS does
    not: a disk made by `DiskElement.from_geometry` holds only m, Ip and Id.
    It maps a category to a list parallel to `placed`'s, with the dimensions
    in SI -- or None for an element built from its mass and inertias.
    """

    rotor: object
    placed: dict | None
    halves: tuple | None = None
    coupling: dict | None = None
    geometry: dict | None = None


# The dimensions both `from_geometry` constructors take -- DiskElement's and
# GearElement's -- named as the form names them.
GEOMETRY_DIMENSIONS = ("width", "i_d", "o_d")

# How the messages call the element and its two diameters. A gear's o_d is its
# pitch diameter and its i_d the bore.
_GEOMETRY_WORDS = {
    "DiskElement.from_geometry": ("disk", "outer diameter", "inner one"),
    "GearElement.from_geometry": ("gear", "pitch diameter", "bore"),
}


def geometry_dimensions(row, constructor):
    """The width and diameters of a "Geometry" row, in metres.

    Checked here and not left to ROSS: `from_geometry` computes the mass as
    rho * pi * w * (o_d^2 - i_d^2) / 4 and takes whatever comes out, so an
    inner diameter larger than the outer one builds an element of negative
    mass without a word.
    """
    word, outer, inner = _GEOMETRY_WORDS[constructor]
    others = [k for k in row if k not in GEOMETRY_DIMENSIONS]
    kwargs = extract_kwargs(row, {}, constructor, others)
    missing = [k for k in GEOMETRY_DIMENSIONS if k not in kwargs]
    if missing:
        raise ValueError(
            "A %s from its geometry needs the width and both diameters "
            "(missing: %s)." % (word, ", ".join(missing))
        )
    width, i_d, o_d = (
        float(getattr(kwargs[k], "m", kwargs[k])) for k in GEOMETRY_DIMENSIONS
    )
    if width <= 0 or i_d < 0 or o_d <= i_d:
        raise ValueError(
            "Invalid %s geometry: the width must be positive and the %s "
            "larger than the %s." % (word, outer, inner)
        )
    return {"width": width, "i_d": i_d, "o_d": o_d}


def _in_si_when_unchecked(constructor, kwargs):
    """Plain SI numbers for a call ROSS does not convert (element_registry).

    `extract_kwargs` has already put every quantity in base units, so the
    number is its magnitude.
    """
    if takes_units(constructor):
        return kwargs
    return {
        k: v.to_base_units().magnitude if isinstance(v, Q_) else v
        for k, v in kwargs.items()
    }


# A probe is checked here, where its row can be named, because ROSS checks
# little of it and only later: `rs.Probe` refuses a radial probe with no angle,
# but a probe on a node the rotor does not have is taken, and the response
# plots then read nothing there.
PROBE_NO_NODE = (
    "Probe #%d is on node %d, which this rotor does not have: its nodes go from "
    "%d to %d."
)
PROBE_DIRECTION = "Probe #%d: the direction is 'radial' or 'axial', not '%s'."
PROBE_NEEDS_ANGLE = "Probe #%d is radial and needs the angle it reads at."
PROBE_ANGLE_UNREADABLE = "Probe #%d: '%s' is not an angle."


def build_probes(rows, rotor):
    """The `rs.Probe` of each row of the project's probes, in the list's order.

    Built to be checked and described (domain/rotor_scene.py), not to be built
    into anything: ROSS's `Rotor` takes no probes. The angle goes in radians,
    and an axial probe has none, as `rs.Probe` itself sets it.
    """
    nodes = [int(n) for n in rotor.nodes]
    resolved = effective_nodes(rows)
    probes = []
    for i, row in enumerate(rows):
        node = resolved[i]
        if node not in nodes:
            raise ValueError(PROBE_NO_NODE % (i + 1, node, min(nodes), max(nodes)))
        direction = str(row.get("direction", "") or "radial").strip().lower()
        if direction not in ("radial", "axial"):
            raise ValueError(PROBE_DIRECTION % (i + 1, row.get("direction")))
        angle = None
        if direction == "radial":
            raw = str(row.get("angle", "")).strip()
            if raw == "":
                raise ValueError(PROBE_NEEDS_ANGLE % (i + 1))
            # Its own unit when the row names one, else the form's degrees.
            typed = {"angle": raw}
            if str(row.get("angle_unit", "") or "").strip():
                typed["angle_unit"] = str(row["angle_unit"]).strip()
            value = extract_kwargs(typed, {}, "Probe", []).get("angle")
            if isinstance(value, Q_):
                angle = float(value.to("rad").m)
            elif isinstance(value, (int, float)):
                angle = float(value)
            else:
                raise ValueError(PROBE_ANGLE_UNREADABLE % (i + 1, raw))
        # The tag is the probe's name, read as typed: through `extract_kwargs`
        # a probe called "1" would come back as the number 1.0.
        tag = str(row.get("tag", "") or "").strip() or None
        probes.append(rs.Probe(node, angle, direction=direction, tag=tag))
    return probes


def build_rotor_from_ui(data):
    """The ROSS rotor the project describes."""
    return assemble_rotor(data).rotor


def assemble_rotor(data):
    """The rotor, together with which element came from which row."""
    # Before anything is built: an element naming a material this rotor does
    # not have used to be answered with the first material of the list, in
    # silence (domain/material_names.py). A MultiRotor is checked half by half,
    # through this same call.
    validate_materials(data)

    if data.get("isMultiRotor"):
        driving_half = assemble_rotor(data["driving_rotor"])
        driven_half = assemble_rotor(data["driven_rotor"])
        driving = driving_half.rotor
        driven = driven_half.rotor
        params = data.get("multi_params", {})

        c_nodes_str = str(params.get("coupled_nodes", "0, 0")).split(",")
        if len(c_nodes_str) < 2:
            c_nodes_str = ["0", "0"]
        coupled_nodes = (int(c_nodes_str[0].strip()), int(c_nodes_str[1].strip()))

        multi_kwargs = {
            "coupled_nodes": coupled_nodes,
            "position": params.get("position", "above"),
        }

        if params.get("gear_mesh_stiffness"):
            multi_kwargs["gear_mesh_stiffness"] = float(params["gear_mesh_stiffness"])
        if str(params.get("update_mesh_stiffness")).lower() == "true":
            multi_kwargs["update_mesh_stiffness"] = True

        if "square_varying_stiffness" in params:
            svs = params["square_varying_stiffness"]
            multi_kwargs["square_varying_stiffness"] = {
                "enable": str(svs.get("enable")).lower() == "true",
                "amplitude_ratio": float(svs.get("amplitude_ratio", 0.0)),
            }

        if "backlash" in params:
            bl = params["backlash"]
            multi_kwargs["backlash"] = {
                "enable": str(bl.get("enable")).lower() == "true",
                "initial_value": float(bl.get("initial_value", 0.0)),
                "error_amp": float(bl.get("error_amp", 0.0)),
                "smooth_operator": str(bl.get("smooth_operator")).lower() == "true",
                "sigma": float(bl.get("sigma", 10000.0)),
            }

        if params.get("orientation_angle"):
            multi_kwargs["orientation_angle"] = float(params["orientation_angle"])

        return Assembled(
            rs.MultiRotor(driving, driven, **multi_kwargs),
            None,
            (driving_half, driven_half),
            {
                "coupled_nodes": list(coupled_nodes),
                "position": multi_kwargs["position"],
                "orientation_angle": multi_kwargs.get("orientation_angle", 0.0),
            },
        )

    mat_ui_props = {
        material_key(m.get("name", "MaterialCustom")): m
        for m in data.get("materials", [])
    }

    created_materials = {}
    for mat in data.get("materials", []):
        # ROSS refuses a space in the name; see domain/material_names.py.
        name = ross_material_name(mat.get("name", "MaterialCustom"))
        kwargs = extract_kwargs(mat, {}, "Material", ["name", "element_type"])
        if "poisson" in kwargs:
            kwargs["Poisson"] = kwargs.pop("poisson")
        created_materials[material_key(name)] = rs.Material(name=name, **kwargs)

    def instantiate_with_cache(category, el_dict, n_val, builder_func, auto_tag):
        # A view field changes the picture, not the element: a new shape must
        # not cost a new ROSS element.
        hash_data = {
            k: v
            for k, v in el_dict.items()
            if str(v).strip() != "" and k not in VIEW_FIELDS
        }
        hash_data["__cat"] = category
        hash_data["__n_calc"] = n_val
        hash_data["__tag"] = el_dict.get("tag", auto_tag)

        if "material" in hash_data:
            m_name = material_key(hash_data["material"])
            if m_name in mat_ui_props:
                hash_data["__mat_props"] = {
                    k: v
                    for k, v in mat_ui_props[m_name].items()
                    if str(v).strip() != ""
                }

        dict_str = json.dumps(hash_data, sort_keys=True)
        h = hashlib.md5(dict_str.encode("utf-8")).hexdigest()
        return ELEMENT_CACHE.get_or_create(h, builder_func)

    # Shafts
    ross_shafts = []
    shafts_eff = effective_nodes(data.get("shafts", []))
    for i, shaft in enumerate(data.get("shafts", [])):
        n_val = shafts_eff[i]
        auto_tag = f"shaft_{i}"

        def build_shaft():
            kwargs = extract_kwargs(
                shaft, created_materials, "ShaftElement", ["element_type", "n"]
            )
            if "tag" not in kwargs:
                kwargs["tag"] = auto_tag
            odl_mag = (
                kwargs["odl"].m
                if hasattr(kwargs.get("odl"), "m")
                else kwargs.get("odl", 0)
            )
            if odl_mag <= 0.0:
                raise ValueError("Invalid outer diameter for Shaft!")
            return rs.ShaftElement(n=n_val, **kwargs)

        ross_shafts.append(
            instantiate_with_cache("shaft", shaft, n_val, build_shaft, auto_tag)
        )

    if not ross_shafts:
        raise ValueError("Add at least one Shaft!")

    # Disks: from m, Ip and Id (BASIC), or from their dimensions (Geometry,
    # DiskElement.from_geometry).
    ross_disks = []
    disk_geometry = []
    disks_eff = effective_nodes(data.get("disks", []))
    for i, d in enumerate(data.get("disks", [])):
        n_val = disks_eff[i]
        auto_tag = f"disk_{i}"
        constructor = ross_constructor("disks", d.get("element_type"))
        dimensions = geometry_dimensions(d, constructor) if "." in constructor else None

        def build_disk():
            kwargs = extract_kwargs(
                d, created_materials, constructor, ["n", "element_type"]
            )
            if "tag" not in kwargs:
                kwargs["tag"] = auto_tag
            if dimensions is not None:
                # No material chosen is the form's own "Default (Steel)".
                kwargs.setdefault("material", rs.materials.steel)
            make = resolve_constructor(constructor)
            return make(n=n_val, **_in_si_when_unchecked(constructor, kwargs))

        ross_disks.append(
            instantiate_with_cache("disk", d, n_val, build_disk, auto_tag)
        )
        disk_geometry.append(dimensions)

    # Gears
    ross_gears = []
    gears_eff = effective_nodes(data.get("gears", []))
    for i, g in enumerate(data.get("gears", [])):
        n_val = gears_eff[i]
        type_val = g.get("element_type", "BASIC")
        auto_tag = f"gear_{i}_{type_val}"

        constructor = ross_constructor("gears", type_val)
        if "." in constructor:
            geometry_dimensions(g, constructor)

        def build_gear():
            kwargs = extract_kwargs(
                g, created_materials, constructor, ["n", "element_type"]
            )
            if "tag" not in kwargs:
                kwargs["tag"] = auto_tag
            if (
                type_val == "BASIC"
                and "pitch_diameter" not in kwargs
                and "base_diameter" not in kwargs
            ):
                raise ValueError(
                    "For gears, provide either Pitch Diameter or Base Diameter."
                )
            if "." in constructor:
                # No material chosen is the form's own "Default (Steel)".
                kwargs.setdefault("material", rs.materials.steel)
            make = resolve_constructor(constructor) or rs.GearElement
            return make(n=n_val, **_in_si_when_unchecked(constructor, kwargs))

        ross_gears.append(
            instantiate_with_cache("gear", g, n_val, build_gear, auto_tag)
        )

    # Bearings
    ross_bearings = []
    bearings_eff = effective_nodes(data.get("bearings", []))
    for i, m in enumerate(data.get("bearings", [])):
        n_val = bearings_eff[i]
        type_val = m.get("element_type", "BASIC")
        auto_tag = f"bearing_{i}_{type_val}"

        def build_bearing():
            element_class = ross_class_name("bearings", type_val)
            kwargs = extract_kwargs(
                migrate_element(m, element_class),
                created_materials,
                element_class,
                ["n", "element_type"],
            )
            if "tag" not in kwargs:
                kwargs["tag"] = auto_tag
            ross_class = getattr(rs, element_class, rs.BearingElement)
            return ross_class(n=n_val, **kwargs)

        ross_bearings.append(
            instantiate_with_cache("bearing", m, n_val, build_bearing, auto_tag)
        )

    # Seals
    ross_seals = []
    seals_eff = effective_nodes(data.get("seals", []))
    for i, s in enumerate(data.get("seals", [])):
        n_val = seals_eff[i]
        type_val = s.get("element_type", "BASIC")
        auto_tag = f"seal_{i}_{type_val}"

        def build_seal():
            element_class = ross_class_name("seals", type_val)
            kwargs = extract_kwargs(
                migrate_element(s, element_class),
                created_materials,
                element_class,
                ["n", "element_type"],
            )
            if "tag" not in kwargs:
                kwargs["tag"] = auto_tag
            ross_class = getattr(rs, element_class, rs.SealElement)
            return ross_class(n=n_val, **kwargs)

        ross_seals.append(
            instantiate_with_cache("seal", s, n_val, build_seal, auto_tag)
        )

    # Couplings
    ross_couplings = []
    couplings_listed = listed_nodes(data.get("couplings", []))
    for i, c in enumerate(data.get("couplings", [])):
        n_val = couplings_listed[i]
        auto_tag = f"coupling_{i}"

        def build_coupling():
            kwargs = extract_kwargs(
                c, created_materials, "CouplingElement", ["n", "element_type"]
            )
            if "tag" not in kwargs:
                kwargs["tag"] = auto_tag
            return rs.CouplingElement(n=n_val, **kwargs)

        ross_couplings.append(
            instantiate_with_cache("coupling", c, n_val, build_coupling, auto_tag)
        )

    # Point Masses
    ross_pointmasses = []
    pointmass_eff = effective_nodes(data.get("pointmasses", []))
    for i, p in enumerate(data.get("pointmasses", [])):
        n_val = pointmass_eff[i]
        auto_tag = f"pointmass_{i}"

        def build_pointmass():
            kwargs = extract_kwargs(
                p, created_materials, "PointMass", ["n", "element_type"]
            )
            if "tag" not in kwargs:
                kwargs["tag"] = auto_tag
            return rs.PointMass(n=n_val, **kwargs)

        ross_pointmasses.append(
            instantiate_with_cache("pointmass", p, n_val, build_pointmass, auto_tag)
        )

    # Rotor Assembly

    all_elements = (
        ross_shafts
        + ross_couplings
        + ross_bearings
        + ross_seals
        + ross_disks
        + ross_gears
        + ross_pointmasses
    )

    validate_node_topology(all_elements)

    # Taken before the sorts below, which put the lists in node order.
    placed = {
        "shafts": list(ross_shafts),
        "couplings": list(ross_couplings),
        "disks": list(ross_disks),
        "gears": list(ross_gears),
        "bearings": list(ross_bearings),
        "seals": list(ross_seals),
        "pointmasses": list(ross_pointmasses),
    }

    ross_shafts.sort(key=lambda x: x.n)
    ross_disks.sort(key=lambda x: x.n)
    ross_gears.sort(key=lambda x: x.n)
    ross_bearings.sort(key=lambda x: x.n)
    ross_seals.sort(key=lambda x: x.n)
    ross_pointmasses.sort(key=lambda x: x.n)

    rotor = rs.Rotor(
        shaft_elements=ross_shafts + ross_couplings,
        disk_elements=ross_disks + ross_gears,
        bearing_elements=ross_bearings + ross_seals,
        point_mass_elements=ross_pointmasses,
    )
    placed["probes"] = build_probes(data.get("probes", []) or [], rotor)
    return Assembled(rotor, placed, geometry={"disks": disk_geometry})
