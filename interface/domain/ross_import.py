# -*- coding: utf-8 -*-
"""Read a native ROSS file and return the project in the screen's format.

ROSS writes rotors as TOML (and, in older releases, JSON), with quantities in
SI base units. The interface works in text and in whichever unit the user chose
on the form, so the conversion happens here.

The ROSS-class -> interface-tab map comes from element_registry: the same one
that builds the forms and the one that generates the Python script. Before
Phase 1 this function had a `class_map` of its own, inverted by hand.
"""

import json

import toml
from ross.units import Q_

from .element_registry import ui_type_for_ross_class
from .units import UNITS_MAPPING


def project_from_ross_file(content):
    """Return the projectData equivalent to the file, or raise ValueError."""
    # TOML first, JSON as the fallback: older ROSS releases wrote JSON. If
    # neither opens, the error says so -- before, the except was bare and the
    # second error reached the user as if it were the only one.
    try:
        ross_data = toml.loads(content)
    except Exception:
        try:
            ross_data = json.loads(content)
        except Exception as error:
            raise ValueError("Could not read the file as TOML nor as JSON: %s" % error)

    if not isinstance(ross_data, dict):
        raise ValueError(
            "The file does not describe a rotor: expected an object at the top."
        )

    project = {
        "materials": [],
        "shafts": [],
        "disks": [],
        "gears": [],
        "couplings": [],
        "seals": [],
        "bearings": [],
        "pointmasses": [],
    }
    seen_materials = set()

    for key, val in ross_data.items():
        if key == "parameters" or not isinstance(val, dict):
            continue

        class_name = key.split("_")[0]

        mapped = ui_type_for_ross_class(class_name)
        if mapped is not None:
            tab, type_val = mapped
            item = {"element_type": type_val}

            for k, v in val.items():
                if k == "material" and isinstance(v, dict):
                    mat_name = v.get("name", "CustomMaterial")
                    item["material"] = mat_name

                    if mat_name not in seen_materials:
                        seen_materials.add(mat_name)
                        mat_obj = {"element_type": "BASIC", "name": mat_name}
                        unit_map_mat = UNITS_MAPPING.get("Material", {})

                        for mk, mv in v.items():
                            if mk != "name":
                                if mk in unit_map_mat and mv is not None:
                                    try:
                                        tgt_unit = unit_map_mat[mk]
                                        base_u = Q_(1, tgt_unit).to_base_units().units
                                        if isinstance(mv, list):
                                            mv = [
                                                float(
                                                    Q_(float(x), base_u).to(tgt_unit).m
                                                )
                                                for x in mv
                                            ]
                                        else:
                                            mv = float(
                                                Q_(float(mv), base_u).to(tgt_unit).m
                                            )
                                    except Exception:
                                        pass

                                mat_obj[mk] = str(mv)
                        project["materials"].append(mat_obj)

                else:
                    unit_map = UNITS_MAPPING.get(class_name, {})

                    if k in unit_map and v is not None:
                        try:
                            tgt_unit = unit_map[k]
                            base_u = Q_(1, tgt_unit).to_base_units().units
                            if isinstance(v, list):
                                v = [
                                    float(Q_(float(x), base_u).to(tgt_unit).m)
                                    for x in v
                                ]
                            else:
                                v = float(Q_(float(v), base_u).to(tgt_unit).m)
                        except Exception:
                            pass

                    if isinstance(v, list):
                        item[k] = str(v)
                    else:
                        item[k] = str(v)

            project[tab].append(item)

    return project
