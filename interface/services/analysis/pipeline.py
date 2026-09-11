# -*- coding: utf-8 -*-
"""One analysis, end to end: the unit of work, with no transport around it.

Everything between "here is what the user asked for" and "here is the chart"
used to live in the route. That was fine while there was exactly one caller.
There are about to be two -- the route, and a worker process that owns the ROSS
objects -- and two callers of the same steps is precisely the shape this
refactoring exists to prevent: a copy kept by hand, diverging in silence.

So the steps live here, once, and both callers ask for the same function. The
route keeps what is genuinely transport (reading the envelope, the token, the
status code); the worker keeps what is genuinely process (a pipe, a loop, a
child). Neither keeps a second copy of what an analysis *is*.

Note what does **not** move: the cache. `ANALYSIS_CACHE` is consulted here, so
whoever runs this function gets the caching -- which is what makes it correct to
run it inside the worker later, where the cache and the ROSS objects will live
together in one process instead of being rebuilt for every request.
"""

import ross as rs

from domain.cache import ANALYSIS_CACHE, spec_key
from domain.compatibility import reason as unsupported_reason
from domain.rotor_builder import build_rotor_from_ui
from services.analysis import get_runner


def refuse_if_unsupported(analysis_type, conversion_type, language):
    """Not every ROSS analysis accepts the converted rotor.

    The refusal happens **before** the rotor is assembled, and it names the
    reason. Without it, what reached the user was ROSS's raw error -- or worse,
    a chart badged "4 DoF" carrying full-model numbers, because the conversion
    swaps the matrix methods and an analysis that assembles its own does not
    notice. See domain/compatibility.py.
    """
    why = unsupported_reason(analysis_type, conversion_type, language)
    if why:
        raise ValueError(why)


def prepared_rotor(project, conversion_type):
    """Build the rotor and apply the requested degree-of-freedom conversion."""
    rotor = build_rotor_from_ui(project)
    if conversion_type == "4dof":
        return rs.utils.convert_6dof_to_4dof(rotor)
    if conversion_type == "torsional":
        return rs.utils.convert_6dof_to_torsional(rotor)
    return rotor


def result_for(runner, project, conversion_type, analysis_type, params, rotor):
    """The ROSS result, from the cache or freshly computed.

    The key is the fingerprint of the spec itself, and the runner's `compute`
    receives only the spec: with that, a computation cannot depend on anything
    left out of the key. See services/analysis/base.py.
    """
    spec = runner.spec(params, rotor)
    key = spec_key(project, conversion_type, analysis_type, spec)

    cached = ANALYSIS_CACHE.get(key)
    if cached is not None:
        return cached
    return ANALYSIS_CACHE.put(key, runner.compute(rotor, spec))


def figure_json(analysis_type, params, conversion_type, project, language="en"):
    """Run one analysis and return the chart, serialised.

    The string is what crosses whatever boundary the caller has: an HTTP
    response today, a pipe to a worker process tomorrow. Returning the figure
    object instead would make the boundary the caller's problem, and the two
    callers would solve it differently.
    """
    runner = get_runner(analysis_type)
    refuse_if_unsupported(analysis_type, conversion_type, language)
    rotor = prepared_rotor(project, conversion_type)
    result = result_for(runner, project, conversion_type, analysis_type, params, rotor)
    figure = runner.plot(result, params, rotor)

    layout = dict(margin=dict(l=60, r=60, t=50, b=100))
    if analysis_type == "campbell":
        layout["legend"] = dict(yanchor="top", y=-0.25)
    figure.update_layout(**layout)
    return figure.to_json()


def mode_shape_json(params, conversion_type, project, point):
    """The 3D mode shape for a point clicked on the Campbell diagram.

    The Campbell result comes from the cache: it is the same object that drew
    the diagram now on screen, and it already carries one modal result per
    speed. The click picks among them; it does not recompute.
    """
    runner = get_runner("campbell")
    rotor = prepared_rotor(project, conversion_type)
    result = result_for(runner, project, conversion_type, "campbell", params, rotor)
    figure = runner.mode_shape_figure(result, params, point)
    figure.update_layout(margin=dict(l=0, r=0, t=30, b=0))
    return figure.to_json()
