# -*- coding: utf-8 -*-
"""What the 3D view draws: the rotor ROSS built, as geometry in SI.

The 3D view is a *view*, like the element list and like ROSS's own 2D figure.
It has no model of its own: the project is still the only description of the
rotor, and ROSS is still the only thing that turns it into one. This module
reads the rotor ROSS just built -- the same one `/build_rotor` draws in 2D, from
the same cache -- and writes down what a drawing needs, so the two views can
never disagree about where anything is.

Why here and not in the browser:

* **positions come from ROSS.** A node is where `rotor.nodes_pos` says it is.
  The screen's own numbering (`getEffectiveNodes`) knows the order of the
  nodes, not their coordinates; and ROSS allows several shaft elements in one
  span (layers), which a running sum of lengths would stack end to end;
* **units come from ROSS.** Every form field carries its own unit (millimetres
  by default). Only the built elements hold the SI values, and converting in
  the browser would need a second copy of the unit table;
* **each part knows its row.** `index` is the element's position in its list
  on the screen, taken from `Assembled.placed` (domain/rotor_builder.py), so a
  click on a part opens the row that describes it.

Most elements have no shape in ROSS. A disk is a mass and two inertias; a
bearing is stiffness and damping at a node. Decided with Leonardo: a disk is
drawn as the **equivalent steel disk** -- the one with the same mass and polar
inertia, bored to the shaft it sits on -- so its size on screen means something
physical, and it is flagged as equivalent.

Every part carries the colour ROSS gives the element (`color`: a CSS colour,
'#525252' or 'Firebrick'), the one its 2D figure paints it with and the one
the form's colour field changes. A disk the project describes by its
dimensions (the "Geometry" form, `DiskElement.from_geometry`) is drawn with
those, and flagged as exact. Bearings, seals and point masses are drawn in
proportion to the shaft, as ROSS's 2D figure does, and carry no dimensions here.
"""

import math

import ross as rs

# The density the equivalent disks are made of: ROSS's own steel, read from it
# rather than copied.
STEEL_DENSITY = float(rs.materials.steel.rho)

FORMAT = 1


def _f(value):
    """A plain float for JSON, or None. ROSS hands back numpy scalars."""
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _color(element):
    """The colour ROSS paints the element with, as text, or None."""
    color = getattr(element, "color", None)
    return str(color) if color else None


def _node_z(rotor, placed):
    """Where each node is along the axis, link nodes included.

    `nodes_pos` has the structural nodes only. A link node -- the ground side of
    a bearing (`n_link`), where a support or a point mass hangs -- has no
    coordinate of its own in ROSS: it sits under the node of the element that
    links to it, and that is where ROSS's 2D figure draws it too.
    """
    z = {int(n): float(p) for n, p in zip(rotor.nodes, rotor.nodes_pos, strict=True)}
    links = {}
    for category in ("bearings", "seals"):
        for element in placed.get(category, []):
            link = getattr(element, "n_link", None)
            if link is not None:
                links.setdefault(int(link), int(element.n))
    # A chain of links (a support on a support) resolves in as many passes as
    # it is long; the bound keeps a cycle from looping.
    for _ in range(len(links) + 1):
        for link, parent in links.items():
            if link not in z and parent in z:
                z[link] = z[parent]
    return z


def _shaft_radius_at(shafts, node):
    """The largest outer radius of the shaft elements touching a node."""
    radii = [float(s.odl) / 2 for s in shafts if int(s.n) == node]
    radii += [float(s.odr) / 2 for s in shafts if int(s.n) + 1 == node]
    return max(radii) if radii else None


def equivalent_disk(m, Ip, bore_radius):
    """The steel disk with this mass and polar inertia, bored to the shaft.

    The inverse of `DiskElement.from_geometry`: for an annulus of outer radius
    ro and inner radius ri, Ip = m (ro^2 + ri^2) / 2 and m = rho pi (ro^2 - ri^2) w.
    When the inertia is too small for a disk with that bore -- ro would come
    out no larger than ri -- it is drawn solid, and says so.

    Returns None for a mass or an inertia that is not positive: there is no
    disk to draw, and the screen falls back to a symbol.
    """
    m, Ip = _f(m), _f(Ip)
    if not m or not Ip or m <= 0 or Ip <= 0:
        return None
    ri = bore_radius or 0.0
    squared = 2 * Ip / m - ri * ri
    bored = ri > 0 and squared > ri * ri
    if not bored:
        ri = 0.0
        squared = 2 * Ip / m
    ro = math.sqrt(squared)
    width = ring_width(m, ro, ri)
    return {
        "outer_radius": ro,
        "inner_radius": ri,
        "width": width,
        "bored": bored,
        "exact": False,
    }


def exact_disk(dimensions):
    """The disk as the project describes it: width and diameters, in metres."""
    ri = dimensions["i_d"] / 2
    return {
        "outer_radius": dimensions["o_d"] / 2,
        "inner_radius": ri,
        "width": dimensions["width"],
        "bored": ri > 0,
        "exact": True,
    }


def ring_width(m, outer_radius, inner_radius):
    """How wide a steel ring of these radii has to be to weigh m."""
    m, ro, ri = _f(m), _f(outer_radius), _f(inner_radius) or 0.0
    if not m or not ro or m <= 0 or ro <= ri:
        return None
    return m / (STEEL_DENSITY * math.pi * (ro * ro - ri * ri))


def _describe(rotor, placed, geometry=None):
    z = _node_z(rotor, placed)
    disk_geometry = (geometry or {}).get("disks") or [None] * len(placed["disks"])
    structural = {int(n) for n in rotor.nodes}
    shafts = placed["shafts"]

    def at(element):
        node = int(element.n)
        return {"n": node, "z": z.get(node), "link": node not in structural}

    scene = {
        "format": FORMAT,
        "length": _f(rotor.L),
        "nodes": [
            {"n": int(n), "z": float(p)}
            for n, p in zip(rotor.nodes, rotor.nodes_pos, strict=True)
        ],
        "shafts": [],
        "couplings": [],
        "disks": [],
        "gears": [],
        "bearings": [],
        "seals": [],
        "pointmasses": [],
        "probes": [],
    }

    for index, s in enumerate(shafts):
        n = int(s.n)
        scene["shafts"].append(
            {
                "index": index,
                "n": n,
                "z0": z.get(n),
                "z1": z.get(n + 1),
                "idl": _f(s.idl),
                "odl": _f(s.odl),
                "idr": _f(s.idr),
                "odr": _f(s.odr),
                "tag": s.tag,
                "color": _color(s),
            }
        )

    for index, c in enumerate(placed["couplings"]):
        n = int(c.n)
        scene["couplings"].append(
            {
                "index": index,
                "n": n,
                "z0": z.get(n),
                "z1": z.get(n + 1),
                "outer_diameter": _f(getattr(c, "o_d", None)),
                # ROSS lumps the coupling's two halves at its two nodes: m_l
                # at the left one, m_r at the right. The view draws a hub there.
                "m_l": _f(getattr(c, "m_l", None)),
                "m_r": _f(getattr(c, "m_r", None)),
                "tag": c.tag,
                "color": _color(c),
            }
        )

    for index, (d, dimensions) in enumerate(
        zip(placed["disks"], disk_geometry, strict=True)
    ):
        scene["disks"].append(
            {
                "index": index,
                **at(d),
                "m": _f(d.m),
                "Ip": _f(d.Ip),
                "Id": _f(d.Id),
                "tag": d.tag,
                "color": _color(d),
                "shape": (
                    exact_disk(dimensions)
                    if dimensions
                    else equivalent_disk(d.m, d.Ip, _shaft_radius_at(shafts, int(d.n)))
                ),
            }
        )

    for index, g in enumerate(placed["gears"]):
        bore = _f(getattr(g, "bore_diameter", None))
        pitch = _f(getattr(g, "pitch_diameter", None))
        addendum = _f(getattr(g, "addendum_radius", None))
        bore_radius = bore / 2 if bore else _shaft_radius_at(shafts, int(g.n))
        outer = addendum or (pitch / 2 if pitch else None)
        width = _f(getattr(g, "width", None))
        # The teeth and the diameters are the gear's own geometry. Its width
        # is optional in ROSS; when it is missing, the width that gives this
        # mass to a steel ring of that size stands in for it, and says so.
        scene["gears"].append(
            {
                "index": index,
                **at(g),
                "m": _f(g.m),
                "Ip": _f(g.Ip),
                "tag": g.tag,
                "color": _color(g),
                "teeth": int(g.n_teeth) if getattr(g, "n_teeth", None) else None,
                "pitch_radius": pitch / 2 if pitch else None,
                "outer_radius": outer,
                "bore_radius": bore_radius,
                "width": width if width else ring_width(g.m, outer, bore_radius),
                "width_is_equivalent": not width,
                # In radians, as ROSS keeps it: the 3D view draws a gear with
                # one as a helical gear.
                "helix_angle": _f(getattr(g, "helix_angle", 0)) or 0.0,
            }
        )

    for category in ("bearings", "seals"):
        for index, b in enumerate(placed[category]):
            link = getattr(b, "n_link", None)
            scene[category].append(
                {
                    "index": index,
                    **at(b),
                    "n_link": int(link) if link is not None else None,
                    "kind": type(b).__name__,
                    "tag": b.tag,
                    "color": _color(b),
                }
            )

    for index, p in enumerate(placed["pointmasses"]):
        scene["pointmasses"].append(
            {
                "index": index,
                **at(p),
                "m": _f(p.m),
                "tag": p.tag,
                "color": _color(p),
            }
        )

    # A probe is not in the rotor (domain/rotor_builder.py `build_probes`): it
    # is placed by its node, and drawn by its direction and angle -- radians
    # from x toward y, ROSS's convention, or None for an axial one.
    for index, probe in enumerate(placed.get("probes", [])):
        node = int(probe.node)
        scene["probes"].append(
            {
                "index": index,
                "n": node,
                "z": z.get(node),
                "direction": probe.direction,
                "angle": _f(probe.angle),
                "tag": probe.tag,
            }
        )

    return scene


def describe_scene(assembled):
    """The scene of an `Assembled` rotor (domain/rotor_builder.py).

    A MultiRotor is two shaft lines: each half is described on its own, in its
    own coordinates, with what the view needs to put them side by side -- the
    coupled nodes, and which side the driven line is drawn on.
    """
    if assembled.halves is None:
        return {
            "kind": "rotor",
            **_describe(assembled.rotor, assembled.placed, assembled.geometry),
        }

    driving, driven = assembled.halves
    multi = assembled.rotor
    coupling = assembled.coupling or {}
    return {
        "kind": "multirotor",
        "format": FORMAT,
        "driving": _describe(driving.rotor, driving.placed, driving.geometry),
        "driven": _describe(driven.rotor, driven.placed, driven.geometry),
        "coupled_nodes": [int(n) for n in coupling.get("coupled_nodes", [])],
        "position": coupling.get("position"),
        "orientation_angle": _f(coupling.get("orientation_angle")),
        # How far along the axis the driven line starts from the driving one,
        # as ROSS computed it to line the two coupled gears up. The distance
        # between the two axes is the view's to work out from the pitch radii:
        # ROSS's own `dy_pos` is measured on the 2D drawing, not on the gears.
        "driven_offset": _f(getattr(multi, "dz_pos", None)),
    }
