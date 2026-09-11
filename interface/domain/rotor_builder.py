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

import numpy as np
import ross as rs
from ross.units import Q_

from .cache import ELEMENT_CACHE
from .element_registry import ross_class_name
from .node_resolver import effective_nodes, validate_node_topology
from .units import INT_PARAMETERS, UNITS_MAPPING
from services.expressions import safe_math_eval


def extract_kwargs(d, mat_dict, element_type, ignore_keys=["element_type", "n"]):
    kwargs = {}

    unit_map = UNITS_MAPPING.get(element_type, {})

    for k, v in d.items():
        if k in ignore_keys or k.endswith("_unit"):
            continue
        if v is None:
            continue

        if k == "material":
            mat_name = str(v).strip().lower()

            if mat_name == "" or mat_name == "default (steel)":
                kwargs[k] = rs.materials.steel
                continue

            kwargs[k] = (
                mat_dict.get(mat_name, list(mat_dict.values())[0])
                if mat_dict
                else rs.materials.steel
            )
            continue

        if isinstance(v, str):
            v_strip = v.strip()

            if v_strip == "":
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
                        val_parsed = [float(x) for x in val_parsed]
                        unit = d.get(f"{k}_unit", unit_map.get(k))
                        if unit:
                            kwargs[k] = Q_(np.array(val_parsed), unit).to_base_units()
                        else:
                            kwargs[k] = np.array(val_parsed)
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


def build_rotor_from_ui(data):
    if data.get("isMultiRotor"):
        driving = build_rotor_from_ui(data["driving_rotor"])
        driven = build_rotor_from_ui(data["driven_rotor"])
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

        return rs.MultiRotor(driving, driven, **multi_kwargs)

    mat_ui_props = {
        str(m.get("name", "MaterialCustom")).strip().lower(): m
        for m in data.get("materials", [])
    }

    created_materials = {}
    for mat in data.get("materials", []):
        name = str(mat.get("name", "MaterialCustom")).strip()
        kwargs = extract_kwargs(mat, {}, "Material", ["name", "element_type"])
        if "poisson" in kwargs:
            kwargs["Poisson"] = kwargs.pop("poisson")
        created_materials[name.lower()] = rs.Material(name=name, **kwargs)

    def instantiate_with_cache(category, el_dict, n_val, builder_func, auto_tag):
        hash_data = {k: v for k, v in el_dict.items() if str(v).strip() != ""}
        hash_data["__cat"] = category
        hash_data["__n_calc"] = n_val
        hash_data["__tag"] = el_dict.get("tag", auto_tag)

        if "material" in hash_data:
            m_name = str(hash_data["material"]).strip().lower()
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

    # Disks
    ross_disks = []
    disks_eff = effective_nodes(data.get("disks", []))
    for i, d in enumerate(data.get("disks", [])):
        n_val = disks_eff[i]
        auto_tag = f"disk_{i}"

        def build_disk():
            kwargs = extract_kwargs(
                d, created_materials, "DiskElement", ["n", "element_type"]
            )
            if "tag" not in kwargs:
                kwargs["tag"] = auto_tag
            return rs.DiskElement(n=n_val, **kwargs)

        ross_disks.append(
            instantiate_with_cache("disk", d, n_val, build_disk, auto_tag)
        )

    # Gears
    ross_gears = []
    gears_eff = effective_nodes(data.get("gears", []))
    for i, g in enumerate(data.get("gears", [])):
        n_val = gears_eff[i]
        type_val = g.get("element_type", "BASIC")
        auto_tag = f"gear_{i}_{type_val}"

        def build_gear():
            element_class = ross_class_name("gears", type_val)
            kwargs = extract_kwargs(
                g, created_materials, element_class, ["n", "element_type"]
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
            ross_class = getattr(rs, element_class, rs.GearElement)
            return ross_class(n=n_val, **kwargs)

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
                m, created_materials, element_class, ["n", "element_type"]
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
                s, created_materials, element_class, ["n", "element_type"]
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
    for i, c in enumerate(data.get("couplings", [])):
        n_val_str = str(c.get("n", "")).strip()
        n_val = int(float(n_val_str)) if n_val_str else i
        auto_tag = f"coupling_{i}"

        def build_coupling():
            kwargs = extract_kwargs(
                c, created_materials, "CouplingElement", ignore_keys=["element_type"]
            )

            if "n" in kwargs and str(kwargs["n"]).strip() != "":
                kwargs["n"] = int(float(kwargs["n"]))
            else:
                kwargs["n"] = n_val

            if "tag" not in kwargs:
                kwargs["tag"] = auto_tag
            return rs.CouplingElement(**kwargs)

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

    ross_shafts.sort(key=lambda x: x.n)
    ross_disks.sort(key=lambda x: x.n)
    ross_gears.sort(key=lambda x: x.n)
    ross_bearings.sort(key=lambda x: x.n)
    ross_seals.sort(key=lambda x: x.n)
    ross_pointmasses.sort(key=lambda x: x.n)

    return rs.Rotor(
        shaft_elements=ross_shafts + ross_couplings,
        disk_elements=ross_disks + ross_gears,
        bearing_elements=ross_bearings + ross_seals,
        point_mass_elements=ross_pointmasses,
    )
