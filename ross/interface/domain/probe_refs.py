# -*- coding: utf-8 -*-
"""An analysis's probe that names a probe of the model, read from the model.

The probes of the modelling screen (domain/rotor_builder.py `build_probes`)
are the ones the analyses are meant to read at, and a row of an analysis's
probe table can name one instead of typing a node and an angle. It names it
by the probe's `id`, which the screen gives every probe and keeps through a
rename, an edit and a move -- so a probe renamed, turned or carried along by
a split shaft is still the one the analysis reads at, without the analysis
being touched.

The row is resolved here, on every run: the node, the angle, the direction
and the name come from the model as it is now. The name goes to `rs.Probe`'s
`tag`, which is what ROSS's charts put in their legend.

On a MultiRotor the two lines are numbered on their own on the screen, and
ROSS numbers the driven line after the driving one (`MultiRotor.__init__`
moves every driven element up by `d_node`). A probe of the driven line is
moved up by the same amount, read from the MultiRotor ROSS built.
"""

import copy

import ross as rs

NO_SUCH_PROBE = (
    "A probe this analysis reads at is not in the model any more: choose another "
    "one in its row, or type its node and angle."
)

# The analysis fields that are probe tables a row of which can name a probe:
# the probes of the response plots, and the outputs of the frequency response
# (services/analysis/freq_response.py reads a probe there).
PROBE_TABLES = ("probes", "outs")


def references(params):
    """Whether any probe table of these parameters names a probe of the model."""
    for key in PROBE_TABLES:
        for row in (params or {}).get(key) or []:
            if isinstance(row, dict) and str(row.get("probe", "") or "").strip():
                return True
    return False


def _shifted(probe, shift):
    if not shift:
        return probe
    return rs.Probe(
        probe.node + shift, probe.angle, direction=probe.direction, tag=probe.tag
    )


def model_probes(assembled, project):
    """`id -> rs.Probe` for every probe of the model, numbered as ROSS numbers
    the rotor the analysis runs on."""
    found = {}

    def collect(rows, probes, shift=0):
        for row, probe in zip(rows, probes, strict=True):
            key = str(row.get("id", "") or "").strip()
            if key:
                found[key] = _shifted(probe, shift)

    if assembled.halves is None:
        collect(project.get("probes") or [], assembled.placed["probes"])
        return found

    driving, driven = assembled.halves
    multi = assembled.rotor
    shift = int(multi.driven_nodes[0]) - int(multi.rotors["driven"].nodes[0])
    collect(project["driving_rotor"].get("probes") or [], driving.placed["probes"])
    collect(project["driven_rotor"].get("probes") or [], driven.placed["probes"], shift)
    return found


def resolved(params, probes):
    """`params` with each row that names a probe filled in from `probes`.

    A row that names none is left as it was typed. A row that names a probe
    the model no longer has is refused: reading at node 0 instead, as an empty
    row would, is a chart of something else with nothing to say so.
    """
    out = copy.deepcopy(params)
    for key in PROBE_TABLES:
        rows = out.get(key)
        if not isinstance(rows, list):
            continue
        for position, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            name = str(row.get("probe", "") or "").strip()
            if not name:
                continue
            probe = probes.get(name)
            if probe is None:
                raise ValueError(NO_SUCH_PROBE)
            rows[position] = {
                "probe": name,
                "node": int(probe.node),
                "angle": None if probe.angle is None else float(probe.angle),
                "direction": probe.direction,
                "tag": probe.tag,
            }
    return out
