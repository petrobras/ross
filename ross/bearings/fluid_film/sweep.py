"""Film force at a prescribed journal position.

:func:`film_force_at` evaluates the hydrodynamic film of a journal bearing
with the journal held at a given position: no equilibrium (Newton) search and
no dynamic coefficients. For tilting pads the pad tilts are either solved
from the pad moment balance, held at the tilts of the case's equilibrium, or
given. The thermal and deformation loops of the case run as usual, with the
journal held.

It is the finite-difference reference of the stiffness coefficients
(central differences of the force about the equilibrium give the stiffness
the engine computes from its perturbation solves) and the building block of
a force map on a grid of journal positions.
"""

import numpy as np

__all__ = ["film_force_at"]

TILTING_PAD_TYPES = (
    "conventional_tilting_pad",
    "inlet_groove_tilting_pad",
    "spray_bar_tilting_pad",
)


def film_force_at(inputs, speed, xj, yj, tilts="solve", tolerance_scale=1.0):
    """Evaluate the film force with the journal held at ``(xj, yj)``.

    Parameters
    ----------
    inputs : dict
        Keyword arguments of :func:`ross.bearings.fluid_film.driver.run_case`
        (SI). ``frequency``, ``equilibrium_type``, ``xj`` and ``yj`` are
        replaced.
    speed : float
        Rotor speed, rad/s.
    xj, yj : float
        Journal centre position, m.
    tilts : {"solve", "frozen"} or array_like of float, optional
        Pad tilt angles of a tilting-pad bearing. ``"solve"`` (default)
        solves each pad's moment balance at the given position; ``"frozen"``
        holds the tilts of the equilibrium of ``inputs`` at ``speed`` (one
        extra equilibrium solve); an array holds the given tilts, rad, one
        per pad. Ignored for fixed-geometry bearings.
    tolerance_scale : float, optional
        Factor on every loop tolerance of the case (the private
        ``_tolerance_scale`` of ``run_case``). A finite difference of the
        force with re-solved tilts needs a tilt tolerance well below the tilt
        change of the step. Default is 1.0.

    Returns
    -------
    dict
        ``fx``, ``fy`` : float
            Force of the film (and of the groove pressure) on the journal, N.
        ``fx_pad``, ``fy_pad`` : numpy.ndarray
            Film force of each pad on the journal, N.
        ``moment_pad`` : numpy.ndarray
            Moment of each pad's film pressure about its pivot, N*m
            (see :func:`ross.bearings.fluid_film.hydrodynamics.film_moment`).
        ``tilt_angle`` : numpy.ndarray
            Pad tilts used, rad.
        ``h_min`` : numpy.ndarray
            Minimum film thickness of each pad, m.
        ``nodal_pressure`` : numpy.ndarray
            Film pressure, Pa, shape ``(total_pads, dim_xz)``.
        ``cavitated`` : numpy.ndarray of bool
            Nodes held at the cavitation pressure, same shape.
        ``converged`` : bool
            Every loop that ran converged (the Newton loop is not run).
        ``convergence`` : dict
            Loop records, as ``run_case(..., diagnostics=True)``.

    Examples
    --------
    >>> import json
    >>> from pathlib import Path
    >>> import ross
    >>> path = Path(ross.__file__).parent / "tests/data/fluid_film/fixed_isoviscous.json"
    >>> inputs = json.loads(path.read_text())["inputs"]
    >>> cb = inputs["radial_clearance"]
    >>> out = film_force_at(inputs, 94.25, 0.3 * cb, -0.4 * cb)
    >>> out["fy"] > 0.0, out["converged"]
    (True, True)
    """
    from ross.bearings.fluid_film.driver import run_case

    kw = dict(inputs)
    total_pads = len(np.atleast_1d(kw["pivot_angle"]))
    cb = float(kw["radial_clearance"])
    kw["frequency"] = [float(speed)]
    kw["_tolerance_scale"] = float(tolerance_scale)
    for name in ("_film_force_only", "_held_tilts", "_coefficient_blocks"):
        kw.pop(name, None)

    held = None
    if kw["bearing_type"] in TILTING_PAD_TYPES:
        if isinstance(tilts, str):
            if tilts == "frozen":
                equilibrium = run_case(**kw)
                held = np.asarray(equilibrium["tilt_angle"][0], dtype=float)
            elif tilts != "solve":
                raise ValueError(
                    f"tilts must be 'solve', 'frozen' or an array, got {tilts!r}"
                )
        else:
            held = np.asarray(tilts, dtype=float).ravel()
            if held.shape != (total_pads,) or not np.all(np.isfinite(held)):
                raise ValueError(
                    f"tilts must hold {total_pads} finite angles (rad), got {tilts!r}"
                )

    kw["equilibrium_type"] = "match_eccentricity"
    kw["xj"] = float(xj) / cb
    kw["yj"] = float(yj) / cb
    kw["_held_tilts"] = held
    kw["_film_force_only"] = True
    return run_case(**kw)
