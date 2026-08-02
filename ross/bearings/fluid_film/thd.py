"""Thermohydrodynamic (THD) driver and temperature post-processing.

This module contains the per-case inner driver that couples the hydrodynamic
(pressure / film) solution to the thermal (film + pad temperature) solution,
plus the temperature post-processing routines it calls:

* :func:`thermohydrodynamics` -- the nested-iteration driver: it runs
  :func:`hydrodynamics` and, when thermal effects are enabled, the thermal
  routines (:func:`thermal_adiabatic` / :func:`thermal_full`), the journal /
  outlet / mixing temperature updates, until the film-pad, journal-surface and
  pad-inlet temperatures all converge.
* :func:`temp_maximum` -- maximum Babbitt-surface temperature in the bearing.
* :func:`temp_journal_film_average` -- journal surface temperature as the area-weighted
  radially-averaged film temperature.
* :func:`temp_journal_zero_flux` -- journal surface temperature found by bisection so
  the bulk heat flux into the shaft is zero.
* :func:`t_outlet` -- area- and flow-weighted pad outlet temperatures.
* :func:`temp_mixing_carryover` -- pad inlet temperature from the conventional
  hot-oil-carryover mixing model (all bearings except spray-bar).
* :func:`temp_mixing_spray_bar` -- pad inlet temperature for spray-bar bearings.
* :func:`temp_inlet_residual` -- RMS of the pad inlet-temperature change.

Indexing / data-structure convention
-------------------------------------
As in :mod:`ross.bearings.fluid_film.mesh`, :mod:`ross.bearings.fluid_film.thermal` and
:mod:`ross.bearings.fluid_film.deform`, everything is 0-based with no padding:

* The connectivity index arrays on the mesh objects (``mesh.n_index``,
  ``mesh.e_index``, ``mesh.node_i``, ...) store node and element numbers used
  directly as indices; element/node loops run ``range(total_*)``.
* ``match_nodes_xz[node, j]`` gives the 3-D film node ``m``, used as
  ``temp_3d[pad, m]``; the cross-film column ``j`` is a slot number, with
  ``-1`` marking unused slots.
* Per-pad 2-D fields are shaped ``(total_pads, dim...)`` and indexed
  ``[pad_index, node]``.
* Per-pad scalars (``temp_inlet``, ``temp_outlet``, ``q_in``, ...) are plain
  length-``total_pads`` arrays indexed ``[pad_index]``.

Injected helpers (NOT defined here)
-----------------------------------
The driver and the journal-temperature routines need physics from sibling
modules plus two integrators from the orchestrator. They are taken as keyword
callables on :func:`thermohydrodynamics` (and as positional callables on
:func:`temp_journal_film_average` / :func:`temp_journal_zero_flux`); the driver
falls back to a lazy import of the sibling modules when one is not supplied:

``hydrodynamics(...)`` (``ross.bearings.fluid_film.hydrodynamics``)
    The hydrodynamic (Reynolds / pressure / journal-equilibrium) solver.
``thermal_adiabatic(...)`` / ``thermal_full(...)`` (``ross.bearings.fluid_film.thermal``)
    The two thermal models, selected by ``thermal_type``.
``integrate_xz(pad_index, mesh, f) -> inte_f``
    Surface integral of a nodal field ``f`` over the Reynolds (film) mesh.
``trapezoid(t, f, start, stop) -> inte``
    Trapezoidal integral of ``f`` against ``t`` over the samples
    ``t[start:stop]`` (lives in :mod:`ross.bearings.fluid_film.driver`). Accepted for
    backwards compatibility; the journal and outlet routines now integrate
    inside compiled kernels and no longer call it.
"""

import inspect

import numpy as np

from ross.bearings.fluid_film._numba_kernels import t_outlet_jit, temp_journal_film_average_jit
from ross.bearings.fluid_film.constants import (
    JTEMP_ERROR,
    MAX_ITERATION,
    TEMP_ERROR,
    TEMP_INLET_ERROR,
)


def _filter_kw(fn, kw):
    """Drop kwargs not in ``fn``'s signature (so a state-dict splat is safe)."""
    params = inspect.signature(fn).parameters
    return {k: v for k, v in kw.items() if k in params}


def _extras(extras, fn, *positional_names):
    """Filter ``extras`` to kwargs accepted by ``fn`` that are *not* already
    provided positionally. Lets the orchestrator dump the full state dict into
    ``*_inputs`` without colliding with the positional args of the per-helper
    callsite below.
    """
    if not extras:
        return {}
    keep = _filter_kw(fn, extras)
    for name in positional_names:
        keep.pop(name, None)
    return keep


def thermohydrodynamics(
    total_pads,
    operating,
    thermal_type,
    temp_j_type,
    relax_t,
    hd_inputs,
    hydrodynamics,
    thermal_adiabatic=None,
    thermal_full=None,
    integrate_xz=None,
    trapezoid=None,
    thermal_inputs=None,
    temp_max_inputs=None,
    temp_journal_inputs=None,
    t_outlet_inputs=None,
    mixing_inputs=None,
    hd_converged_key="hd_converged",
):
    """Run the coupled thermohydrodynamic solution for one configuration.

    The control structure is three nested fixed-point loops, each capped at
    ``MAX_ITERATION``:

    * outer -- pad **inlet** temperature mixing;
    * middle -- **journal-surface** temperature;
    * inner -- **film + pad** temperature / viscosity.

    The inner loop always begins by calling ``hydrodynamics`` for the current
    thermal state. When ``thermal_type is None`` (isoviscous) the driver runs
    hydrodynamics **once** and returns immediately, touching no thermal
    routine. Otherwise it runs ``thermal_adiabatic``
    (``thermal_type == "adiabatic"``) or ``thermal_full``
    (``thermal_type == "full"``), checks the film-pad RMS error, then computes
    the maximum temperature (:func:`temp_maximum`), updates the
    journal-surface temperature (:func:`temp_journal_film_average` /
    :func:`temp_journal_zero_flux`), the outlet temperatures
    (:func:`t_outlet`) and the mixing inlet temperatures
    (:func:`temp_mixing_carryover` / :func:`temp_mixing_spray_bar`),
    under-relaxing each.

    Divergence handling: each loop tolerates up to ten "growing-error"
    iterations, and if the journal equilibrium has not converged
    (``hd_converged == 0``) any temperature loop bails out early rather than
    chase a false equilibrium.

    Parameters
    ----------
    total_pads : int
        Number of pads.
    operating : OperatingPoint
        Speed and pressure conditions of the case; ``bearing_type`` selects
        the mixing model (``"spray_bar_tilting_pad"`` -> spray-bar) and
        ``operating_type`` is passed through to it.
    thermal_type : str or None
        Thermal model, one of
        :data:`~ross.bearings.fluid_film.constants.THERMAL_TYPES`: ``None``
        isoviscous (no thermal update), ``"adiabatic"``, or ``"full"``
        conducting-pad.
    temp_j_type : str
        Journal-temperature treatment, one of
        :data:`~ross.bearings.fluid_film.constants.TEMP_J_TYPES`:
        ``"averaged_film_temperature"`` uses
        :func:`temp_journal_film_average`,
        ``"no_heat_flux_into_journal"`` the zero-flux bisection
        (:func:`temp_journal_zero_flux`); for any other value ``temp_j``
        keeps its previous value.
    relax_t : float
        Temperature under-relaxation factor (``RelaxT``).
    hd_inputs : dict
        The initial solver state: it seeds the running ``state`` dict that is
        splatted into ``hydrodynamics`` (everything the hydrodynamic solver
        needs that is not recomputed here).
    hydrodynamics : callable
        The hydrodynamic (Reynolds / pressure / journal-equilibrium) solver.
        Called ``hydrodynamics(**state)`` and must return a dict; its returned
        dict is merged into the driver state each inner iteration and
        ultimately returned. It must provide the ``hd_converged_key`` flag and
        the film fields (``temp_3d``, ``velocity_x_n`` ...) consumed by the
        thermal step.
    thermal_adiabatic, thermal_full : callable, optional
        The two thermal models, selected by ``thermal_type``. Each is called
        with the keyword union of ``thermal_inputs`` and the iterated state,
        and must return the updated temperature / viscosity / flow-regime
        fields with the film-pad RMS residual last. Lazily imported from
        :mod:`ross.bearings.fluid_film.thermal` when ``thermal_type is not None`` and
        not supplied.
    integrate_xz : callable, optional
        Surface integrator over the Reynolds (film) mesh, called
        ``integrate_xz(pad_index, mesh, f)`` for a nodal field ``f``. Needed
        by :func:`temp_journal_zero_flux`.
    trapezoid : callable, optional
        Trapezoidal integrator ``trapezoid(t, f, start, stop)``. Resolved from
        :mod:`ross.bearings.fluid_film.driver` when ``thermal_type is not None`` and
        not supplied; the journal and outlet routines now integrate inside
        compiled kernels, so it is accepted but not called.
    thermal_inputs : dict, optional
        Name-binding namespace -- normally the whole solver state dict -- that
        the selected thermal model draws its remaining arguments from
        (geometry / mesh / lube data not in the iterated state).
    temp_max_inputs, temp_journal_inputs, t_outlet_inputs, mixing_inputs : dict, optional
        The same kind of namespace for :func:`temp_maximum`, the journal
        routine, :func:`t_outlet` and the mixing routine respectively: each is
        filtered down to the keywords that routine accepts and that are not
        already passed positionally.
    hd_converged_key : str, optional
        Key under which ``hydrodynamics`` reports journal-equilibrium
        convergence (``0`` not converged). Default ``"hd_converged"``.

    Returns
    -------
    dict
        The merged solver state after convergence (or after the iteration
        cap): every field returned by the last ``hydrodynamics`` call, plus
        the updated thermal quantities ``temp_3d``, ``temp_adiab`` /
        ``temp_full``, ``temp_j``, ``tpad_max``, ``tpad_max_pad``,
        ``temp_inlet``, ``temp_outlet``, ``temp_outlet_bulk``, the converged
        flags and the final ``rms_temp`` / ``d_tj`` / ``rms_temp_inlet``
        residuals.

    Notes
    -----
    The immutable mesh / geometry / lubricant data is bundled into the
    ``hd_inputs`` / ``thermal_inputs`` / ``*_inputs`` dicts, while the
    *iterated* state (temperatures, viscosity, pressure, coefficients) flows
    through the merged ``state`` dict, so the coupling is explicit.
    """
    # State that is read/written across the coupled loops. ``hydrodynamics``
    # mutates most of it; the thermal step reads the film fields and writes the
    # temperature/viscosity fields back.
    state = dict(hd_inputs)
    mesh = state["mesh"]
    pads = state["pads"]
    thermal_inputs = thermal_inputs or {}
    temp_max_inputs = temp_max_inputs or {}
    temp_journal_inputs = temp_journal_inputs or {}
    t_outlet_inputs = t_outlet_inputs or {}
    mixing_inputs = mixing_inputs or {}

    if thermal_type is not None:
        if thermal_adiabatic is None or thermal_full is None:
            from ross.bearings.fluid_film import thermal as _thermal

            if thermal_adiabatic is None:
                thermal_adiabatic = _thermal.thermal_adiabatic
            if thermal_full is None:
                thermal_full = _thermal.thermal_full
        if trapezoid is None:
            from ross.bearings.fluid_film.driver import trapezoid as _trap

            trapezoid = _trap

    rms_temp = 0.0
    d_tj = 0.0
    rms_temp_inlet = 0.0

    # ----- outer loop: pad inlet (mixing) temperature ---------------------
    diverge_tin = 0
    for mixtemp_index in range(1, MAX_ITERATION + 1):
        # ----- middle loop: journal surface temperature ------------------
        diverge_tj = 0
        for jtemp_index in range(1, MAX_ITERATION + 1):
            # ----- inner loop: film/pad temperature & viscosity ----------
            diverge_t = 0
            for vis_index in range(1, MAX_ITERATION + 1):
                # Hydrodynamic analysis under the current thermal condition.
                hd_out = hydrodynamics(**state)
                state.update(hd_out)

                # Isoviscous: skip every thermal branch.
                if thermal_type is None:
                    return state

                rms_temp_old = 0.0 if vis_index == 1 else rms_temp

                # Natural-shape (0-based) view of state for the thermal /
                # thd-helper boundary. See ``_state_for_thermal``.
                nat = _state_for_thermal(state)
                # View ``thermal_inputs`` too: it may contain the
                # mesh/geometry arrays (the orchestrator usually passes the
                # full state dict) needed by ``thermal_full`` arguments that
                # are not in ``_thermal_state``'s ``shared`` list.
                nat_inputs = _state_for_thermal(thermal_inputs)
                nat_temp_max = _state_for_thermal(temp_max_inputs)
                nat_temp_journal = _state_for_thermal(temp_journal_inputs)
                nat_t_outlet = _state_for_thermal(t_outlet_inputs)
                nat_mixing = _state_for_thermal(mixing_inputs)

                if thermal_type == "adiabatic":
                    *thermal_out, rms_temp = thermal_adiabatic(
                        **{**nat_inputs, **_thermal_state(state, 1)}
                    )
                    _scatter_thermal(state, thermal_out, 1)
                elif thermal_type == "full":
                    *thermal_out, rms_temp = thermal_full(
                        **{**nat_inputs, **_thermal_state(state, 2)}
                    )
                    _scatter_thermal(state, thermal_out, 2)

                hd_converged = state.get(hd_converged_key, 1)

                # Convergence / divergence on the film-pad temperature.
                if rms_temp < TEMP_ERROR:
                    break
                if rms_temp > 0.95 * rms_temp_old and vis_index > 1 and diverge_t <= 10:
                    if hd_converged == 0:
                        break
                    diverge_t += 1
                elif (
                    rms_temp > 0.95 * rms_temp_old and vis_index > 1 and diverge_t > 10
                ):
                    break
            # (label 100)

            # Refresh the natural-shape view after the inner loop's last
            # ``state.update(hd_out)`` (state arrays may have been reallocated).
            nat = _state_for_thermal(state)
            nat_temp_max = _state_for_thermal(temp_max_inputs)
            nat_temp_journal = _state_for_thermal(temp_journal_inputs)
            nat_t_outlet = _state_for_thermal(t_outlet_inputs)
            nat_mixing = _state_for_thermal(mixing_inputs)

            # Maximum bearing temperature.
            tpad_max_kw = _extras(
                nat_temp_max,
                temp_maximum,
                "total_pads",
                "mesh",
                "pads",
                "temp_supply",
                "temp_3d",
            )
            tpad_max, tpad_max_pad = temp_maximum(
                total_pads,
                mesh,
                pads,
                state["temp_supply"],
                nat["temp_3d"],
                **tpad_max_kw,
            )
            state["tpad_max"] = tpad_max
            state["tpad_max_pad"] = tpad_max_pad

            # Save the old journal temperature, update it, check convergence.
            temp_j_old = state["temp_j"]
            d_tj_old = 0.0 if jtemp_index == 1 else d_tj

            # ``integrate_xz`` operates on 0-based natural mesh fields; wrap it
            # so the natural-shape helpers in this module can use it without
            # rebuilding all the connectivity arrays per call.
            integrate_xz_nat = _wrap_integrate_xz_nat(integrate_xz)

            # ``temp_j`` is only recomputed for ``averaged_film_temperature``
            # and ``no_heat_flux_into_journal``. For the other modes
            # (``insulated_shaft_surface`` / user-specified) neither branch
            # fires, so ``temp_j`` keeps its previous value -- ``d_tj`` is then
            # 0 and the journal-temperature loop converges on the first pass.
            temp_j = temp_j_old
            if temp_j_type == "averaged_film_temperature":
                tj1_kw = _extras(
                    nat_temp_journal,
                    temp_journal_film_average,
                    "total_pads",
                    "mesh",
                    "pads",
                    "h_n",
                    "temp_3d",
                    "integrate_xz",
                    "trapezoid",
                )
                temp_j = temp_journal_film_average(
                    total_pads,
                    mesh,
                    pads,
                    nat["h_n"],
                    nat["temp_3d"],
                    **tj1_kw,
                )
            elif temp_j_type == "no_heat_flux_into_journal":
                tj2_kw = _extras(
                    nat_temp_journal,
                    temp_journal_zero_flux,
                    "total_pads",
                    "mesh",
                    "temp_supply",
                    "tpad_max",
                    "temp_3d",
                    "scale_dissip",
                    "integrate_xz",
                )
                temp_j = temp_journal_zero_flux(
                    total_pads,
                    mesh,
                    state["temp_supply"],
                    tpad_max,
                    nat["temp_3d"],
                    nat["scale_dissip"],
                    integrate_xz_nat,
                    **tj2_kw,
                )

            d_tj = abs(temp_j - temp_j_old)
            hd_converged = state.get(hd_converged_key, 1)

            converged_tj = False
            if abs(d_tj) < JTEMP_ERROR:
                converged_tj = True
            elif d_tj > 0.95 * d_tj_old and jtemp_index > 1 and diverge_tj <= 10:
                if hd_converged == 0:
                    converged_tj = True
                else:
                    diverge_tj += 1
            elif d_tj > 0.95 * d_tj_old and jtemp_index > 1 and diverge_tj > 10:
                converged_tj = True

            # Limit the journal temperature change to 10 degF per iteration.
            # Everything here is in kelvin, so that is 10/1.8 K; a bare 10.0
            # would cap at 10 K (= 18 degF) and under-relax large early
            # temperature swings.
            jtemp_step_limit = 10.0 / 1.8
            if abs(d_tj) > jtemp_step_limit:
                relax_tj = jtemp_step_limit / abs(d_tj)
            else:
                relax_tj = 1.0
            relax_tj = min(relax_tj, relax_t)
            state["temp_j"] = relax_tj * temp_j + (1.0 - relax_tj) * temp_j_old

            if converged_tj:
                break
        # (label 200)

        # Refresh natural-shape views after the journal-temperature loop.
        nat = _state_for_thermal(state)
        nat_t_outlet = _state_for_thermal(t_outlet_inputs)
        nat_mixing = _state_for_thermal(mixing_inputs)

        # Outlet temperatures of all pads (feeds the inlet mixing).
        to_kw = _extras(
            nat_t_outlet,
            t_outlet,
            "total_pads",
            "mesh",
            "pads",
            "h_n",
            "temp_3d",
            "velocity_x_n",
            "q_x",
            "trapezoid",
        )
        temp_outlet, temp_outlet_bulk = t_outlet(
            total_pads,
            mesh,
            pads,
            nat["h_n"],
            nat["temp_3d"],
            nat["velocity_x_n"],
            nat["q_x"],
            **to_kw,
        )
        # ``temp_outlet`` / ``temp_outlet_bulk`` are 0-based natural per-pad.
        _assign_padded(state, "temp_outlet", temp_outlet)
        _assign_padded(state, "temp_outlet_bulk", temp_outlet_bulk)

        # Save the old inlet temperatures before mixing (natural-shape copy).
        temp_inlet_old = np.array(nat["temp_inlet"], dtype=float, copy=True)

        # Update the pad inlet temperature with the appropriate mixing model.
        if operating.bearing_type in (
            "fixed_geometry",
            "conventional_tilting_pad",
            "inlet_groove_tilting_pad",
            "pressure_dam",
        ):
            tm1_kw = _extras(
                nat_mixing,
                temp_mixing_carryover,
                "total_pads",
                "operating_type",
                "q_in",
                "q_out",
                "q_carryover",
                "temp_sump",
                "temp_outlet",
                "temp_inlet",
            )
            temp_inlet = temp_mixing_carryover(
                total_pads,
                operating.operating_type,
                nat["q_in"],
                nat["q_carryover"],
                state["temp_sump"],
                temp_outlet,
                nat["temp_inlet"],
                **tm1_kw,
            )
        elif operating.bearing_type == "spray_bar_tilting_pad":
            tm2_kw = _extras(
                nat_mixing,
                temp_mixing_spray_bar,
                "total_pads",
                "operating_type",
                "q_in",
                "q_carryover",
                "q_supply",
                "temp_supply",
                "temp_outlet",
                "temp_inlet",
            )
            temp_inlet = temp_mixing_spray_bar(
                total_pads,
                operating,
                nat["q_in"],
                nat["q_carryover"],
                state["q_supply"],
                temp_outlet,
                nat["temp_inlet"],
                **tm2_kw,
            )
        else:
            temp_inlet = np.array(nat["temp_inlet"], dtype=float, copy=True)
        _assign_padded(state, "temp_inlet", temp_inlet)

        rms_temp_inlet_old = 0.0 if mixtemp_index == 1 else rms_temp_inlet
        rms_temp_inlet = temp_inlet_residual(total_pads, temp_inlet_old, temp_inlet)

        hd_converged = state.get(hd_converged_key, 1)

        if rms_temp_inlet < TEMP_INLET_ERROR:
            break
        if (
            rms_temp_inlet > 0.95 * rms_temp_inlet_old
            and mixtemp_index > 1
            and diverge_tin <= 10
        ):
            if hd_converged == 0:
                break
            diverge_tin += 1
        elif (
            rms_temp_inlet > 0.95 * rms_temp_inlet_old
            and mixtemp_index > 1
            and diverge_tin > 10
        ):
            break
        else:
            # Under-relax the inlet temperature for the next outer pass.
            for pad_index in range(total_pads):
                temp_inlet[pad_index] = (
                    relax_t * temp_inlet[pad_index]
                    + (1.0 - relax_t) * temp_inlet_old[pad_index]
                )
            _assign_padded(state, "temp_inlet", temp_inlet)
    # (label 300)

    state["rms_temp"] = rms_temp
    state["d_tj"] = d_tj
    state["rms_temp_inlet"] = rms_temp_inlet
    return state


# ---------------------------------------------------------------------------
# Thermal / thd-helper boundary (now 0-based natural throughout)
# ---------------------------------------------------------------------------
# The driver state reaching :func:`thermohydrodynamics` is the package's
# uniformly 0-based natural state, passed as ``g`` for every ``*_inputs``
# argument; the thermal module and the per-pad helpers in this file are all
# 0-based-native. The helpers below are therefore pass-throughs; they are kept
# so the driver body reads cleanly and the ``*_inputs`` aliasing stays explicit.
def _state_for_thermal(state):
    """Identity view of the running state (already 0-based natural).

    The driver state reaching :func:`thermohydrodynamics` is the package's
    uniformly 0-based natural state, as are the thermal module and the per-pad
    helpers in this file, so no slicing is needed -- this returns a shallow copy
    of the dict so callers can splat it safely.
    """
    return dict(state)


def _wrap_integrate_xz_nat(integrate_xz):
    """Return the surface integrator unchanged (already 0-based native).

    ``integrate_xz`` here is the 0-based ``_integrate_xz_coeff`` from
    ``driver``; the thd temperature helpers carry 0-based connectivity / nodal
    fields, so no re-padding or pad-index shifting is required.
    """
    return integrate_xz


def _assign_padded(state, key, value):
    """Store a 0-based natural ``value`` into ``state[key]`` directly.

    The driver state is 0-based natural, matching what the thermal / thd
    helpers return, so this is a plain dict assignment.
    """
    state[key] = value


def _thermal_state(state, thermal_type):
    """Project the running ``state`` onto the thermal model's keyword args.

    Helper for :func:`thermohydrodynamics`. The selected thermal model is
    invoked as ``thermal_*(**thermal_inputs, **_thermal_state(state, ...))``;
    this returns only the *iterated* fields (viscosity / temperature / flow
    regime / pressure) that flow through ``state``, so the static mesh/geometry
    data stays in ``thermal_inputs``. Keys absent from ``state`` are dropped so
    they can be supplied via ``thermal_inputs`` instead.

    The state arrays are 0-based natural, so they are passed through unchanged.
    """
    shared = [
        "vis_n_3d",
        "vis_n_average",
        "vis_eddy_3d",
        "vis_effect_3d",
        "flow_regime_track",
        "flow_regime_dam",
        "scale_turb_track",
        "scale_turb_dam",
        "nodal_pressure",
        "velocity_x_n",
        "velocity_y_n",
        "velocity_z_n",
        "dudy_n",
        "dwdy_n",
        "h_n",
        "temp_inlet",
        "temp_j",
        "temp_3d",
        "temp_adiab" if thermal_type == "adiabatic" else "temp_full",
        "film_onset",
    ]
    return {k: state[k] for k in shared if k in state}


def _scatter_thermal(state, thermal_out, thermal_type):
    """Write the thermal model's returned arrays back into ``state``.

    Helper for :func:`thermohydrodynamics`. ``thermal_adiabatic`` returns
    ``(temp_adiab, temp_3d, vis_n_3d, vis_n_average, flow_regime_track,
    flow_regime_dam, scale_turb_track, scale_turb_dam)`` and ``thermal_full``
    returns ``(temp_full, temp_3d, vis_n_3d, vis_n_average, flow_regime_dam,
    scale_turb_dam)`` (the trailing ``rms_temp`` is stripped by the caller).

    Returned values are natural-shape arrays (because they came back through
    :func:`_thermal_state`); the running state is 0-based natural too, so
    :func:`_assign_padded` writes them straight back into ``state``.
    """
    if thermal_type == "adiabatic":
        keys = [
            "temp_adiab",
            "temp_3d",
            "vis_n_3d",
            "vis_n_average",
            "flow_regime_track",
            "flow_regime_dam",
            "scale_turb_track",
            "scale_turb_dam",
        ]
    else:
        keys = [
            "temp_full",
            "temp_3d",
            "vis_n_3d",
            "vis_n_average",
            "flow_regime_dam",
            "scale_turb_dam",
        ]
    for key, value in zip(keys, thermal_out):
        _assign_padded(state, key, value)


def temp_maximum(
    total_pads,
    mesh,
    pads,
    temp_supply,
    temp_3d,
):
    """Maximum temperature on the Babbitt (pad) surface across all pads.

    For every Reynolds node it picks the radial layer on the pad surface -- the
    film/pad interface in the dam region or the pocket floor in the pocket
    region (see :meth:`~ross.bearings.fluid_film.state.PadGeometry.node_in_pocket`) --
    and keeps the largest 3-D temperature, never below the supply temperature.

    Parameters
    ----------
    total_pads : int
        Number of pads.
    mesh : ReynoldsMesh
        Film (x-z) mesh: node coordinates, the ``match_nodes_xz`` map
        ``[node, layer - 1]`` -> 3-D node id, and the per-pad track
        boundary-layer / core element counts that locate the pocket floor.
    pads : PadGeometry
        Per-pad geometry; classifies each node as pocket or dam.
    temp_supply : float
        Oil supply temperature (lower bound on the maximum), K.
    temp_3d : numpy.ndarray of float
        3-D film/pad temperature, shape ``(total_pads, dim_3d)``, K.

    Returns
    -------
    tpad_max : float
        Maximum pad-surface temperature.
    tpad_max_pad : int
        1-based pad index where the maximum occurs (``0`` if none exceeds
        ``temp_supply``).
    """
    tpad_max = temp_supply
    tpad_max_pad = 0
    for pad_index in range(total_pads):
        pad = pad_index
        for i in range(mesh.total_nodes):
            node = mesh.n_index[i]
            x = mesh.x[pad, node]
            z = mesh.z[pad, node]
            if pads.node_in_pocket(x, z, pad):
                layer = 1
            else:
                layer = mesh.total_e_y_trackbl[pad] + mesh.total_e_y_trackcore[pad] + 1
            m = mesh.match_nodes_xz[node, layer - 1]
            if temp_3d[pad, m] > tpad_max:
                tpad_max = temp_3d[pad, m]
                # ``tpad_max_pad`` is a reported 1-based pad number, not an
                # array index.
                tpad_max_pad = pad_index + 1
    return tpad_max, tpad_max_pad


def temp_journal_film_average(
    total_pads,
    mesh,
    pads,
    h_n,
    temp_3d,
):
    """Journal surface temperature as the area-weighted mean film temperature.

    For every Reynolds node it integrates the 3-D temperature radially across
    the local film (full thickness in the pocket, reduced thickness in the dam)
    to get a node-average temperature, then surface-integrates that field over
    each pad and divides the pad-weighted sum by the total pad area.

    Parameters
    ----------
    total_pads : int
        Number of pads.
    mesh : ReynoldsMesh
        Film (x-z) mesh: node coordinates, element sizes and connectivity,
        the 3-D radial coordinate ``y_3d``, and the ``match_nodes_xz`` map
        ``[node, layer - 1]`` -> 3-D node id. The per-pad track
        boundary-layer / core element counts set the dam-region lower
        integration limit.
    pads : PadGeometry
        Per-pad geometry; the pad thickness offsets the radial coordinate to
        a film-local ``y``.
    h_n : numpy.ndarray of float
        Nodal film thickness, shape ``(total_pads, dim_xz)``, m.
    temp_3d : numpy.ndarray of float
        3-D film temperature, shape ``(total_pads, dim_3d)``, K.

    Returns
    -------
    float
        The journal surface temperature.
    """
    dim_xz = mesh.x.shape[1]
    dim_yf = mesh.match_nodes_xz.shape[1]
    return temp_journal_film_average_jit(
        int(total_pads),
        int(mesh.total_nodes),
        np.ascontiguousarray(mesh.n_index, dtype=np.int64),
        np.ascontiguousarray(mesh.total_e_y_trackbl, dtype=np.int64),
        np.ascontiguousarray(mesh.total_e_y_trackcore, dtype=np.int64),
        np.ascontiguousarray(mesh.match_nodes_xz, dtype=np.int64),
        int(mesh.total_e_y_film),
        int(mesh.total_elements),
        np.ascontiguousarray(mesh.e_index, dtype=np.int64),
        np.ascontiguousarray(mesh.node_i, dtype=np.int64),
        np.ascontiguousarray(mesh.node_j, dtype=np.int64),
        np.ascontiguousarray(mesh.node_k, dtype=np.int64),
        np.ascontiguousarray(mesh.node_l, dtype=np.int64),
        float(pads.pad_thickness),
        np.ascontiguousarray(pads.pad_length, dtype=np.float64),
        np.ascontiguousarray(pads.axial_length, dtype=np.float64),
        np.ascontiguousarray(pads.length_track, dtype=np.float64),
        np.ascontiguousarray(pads.depth_track, dtype=np.float64),
        np.ascontiguousarray(pads.axial_length_dam, dtype=np.float64),
        np.ascontiguousarray(pads.axial_length_track, dtype=np.float64),
        np.ascontiguousarray(mesh.x, dtype=np.float64),
        np.ascontiguousarray(mesh.z, dtype=np.float64),
        np.ascontiguousarray(mesh.y_3d, dtype=np.float64),
        np.ascontiguousarray(h_n, dtype=np.float64),
        np.ascontiguousarray(temp_3d, dtype=np.float64),
        np.ascontiguousarray(mesh.e_length, dtype=np.float64),
        np.ascontiguousarray(mesh.e_width, dtype=np.float64),
        int(dim_xz),
        int(dim_yf),
    )


def temp_journal_zero_flux(
    total_pads,
    mesh,
    temp_supply,
    tpad_max,
    temp_3d,
    scale_dissip,
    integrate_xz,
):
    """Journal surface temperature giving zero net heat flux into the shaft.

    Bisects the journal temperature between the supply temperature and the
    maximum pad temperature; at each step it forms the conductive heat flux
    ``scale_dissip * (T_j - T_film) / dy`` at the film/journal interface for
    every Reynolds node, surface-integrates it over each pad, and moves the
    bracket so the summed flux goes to zero. The search stops when the bracket
    is narrower than ``0.1 / 1.8`` K.

    Parameters
    ----------
    total_pads : int
        Number of pads.
    mesh : ReynoldsMesh
        Film (x-z) mesh: node index list, the 3-D radial coordinate ``y_3d``
        and the ``match_nodes_xz`` map ``[node, layer - 1]`` -> 3-D node id
        used to reach the two layers below the journal surface.
    temp_supply : float
        Lower bisection bound (oil supply temperature), K.
    tpad_max : float
        Upper bisection bound (maximum pad temperature), K.
    temp_3d : numpy.ndarray of float
        3-D film temperature, shape ``(total_pads, dim_3d)``, K.
    scale_dissip : numpy.ndarray of float
        Per-node turbulent conductivity scaling, shape
        ``(total_pads, dim_xz)``.
    integrate_xz : callable
        Surface integrator over the Reynolds mesh, called
        ``integrate_xz(pad_index, mesh, f)`` for a nodal field ``f``.

    Returns
    -------
    float
        The journal surface temperature.
    """
    dim_xz = scale_dissip.shape[1]

    bound_lower = temp_supply
    bound_upper = tpad_max
    temp_j = 0.5 * (bound_lower + bound_upper)

    for _ in range(1, MAX_ITERATION + 1):
        temp_j = 0.5 * (bound_lower + bound_upper)
        sum_flux = 0.0

        for pad_index in range(total_pads):
            pad = pad_index
            dtdy = np.zeros(dim_xz, dtype=float)
            for i in range(mesh.total_nodes):
                node = mesh.n_index[i]
                t1 = temp_j
                m_film = mesh.match_nodes_xz[node, mesh.total_e_y_film - 1]
                m_next = mesh.match_nodes_xz[node, mesh.total_e_y_film + 1 - 1]
                t2 = temp_3d[pad, m_film]
                step_length = mesh.y_3d[pad, m_next] - mesh.y_3d[pad, m_film]
                dtdy[node] = scale_dissip[pad, node] * (t1 - t2) / step_length

            inte_flux = integrate_xz(
                pad_index,
                mesh,
                dtdy,
            )
            sum_flux += inte_flux

        # Bisection bracket tolerance: 0.1 degF, i.e. 0.1/1.8 K here. A bare
        # 0.1 would stop the bisection a step early (0.1 K ~= 0.18 degF),
        # coarser than the enclosing journal-temperature loop's own
        # JTEMP_ERROR, leaving temp_j under-resolved for the
        # "no heat flux into journal" mode.
        if abs(bound_upper - bound_lower) < 0.1 / 1.8:
            break

        if sum_flux < 0.0:
            bound_lower = temp_j
        else:
            bound_upper = temp_j

    return temp_j


def t_outlet(
    total_pads,
    mesh,
    pads,
    h_n,
    temp_3d,
    velocity_x_n,
    q_x,
):
    """Area- and flow-weighted pad outlet temperatures.

    For each pad it radially integrates two fields at every Reynolds node --
    the temperature (giving a node-average ``temp_n_average``) and the
    temperature times the circumferential velocity (giving a node bulk
    ``temp_n_average_bulk``) -- using the same pocket/dam radial limits as
    :func:`temp_journal_film_average`. It then sums over the trailing-edge
    element row to form the flow-area-weighted outlet temperature and the
    mass-flux-weighted bulk outlet temperature.

    Parameters
    ----------
    total_pads : int
        Number of pads.
    mesh : ReynoldsMesh
        Film (x-z) mesh: node coordinates, element widths and connectivity,
        the 3-D radial coordinate ``y_3d``, the ``match_nodes_xz`` map
        ``[node, layer - 1]`` -> 3-D node id, and the per-pad track
        boundary-layer / core element counts setting the radial limits.
    pads : PadGeometry
        Per-pad geometry.
    h_n : numpy.ndarray of float
        Nodal film thickness, shape ``(total_pads, dim_xz)``, m.
    temp_3d : numpy.ndarray of float
        3-D film temperature, shape ``(total_pads, dim_3d)``, K.
    velocity_x_n : numpy.ndarray of float
        Circumferential film velocity at the 3-D nodes, same shape, m/s.
    q_x : numpy.ndarray of float
        Circumferential flow at each ``x`` station, shape
        ``(total_pads, dim_x)``, m^3/s.

    Returns
    -------
    temp_outlet : numpy.ndarray of float
        Flow-area-weighted outlet temperature per pad (0-based natural, length
        ``total_pads``).
    temp_outlet_bulk : numpy.ndarray of float
        Mass-flux-weighted bulk outlet temperature per pad (same layout).
    """
    # Whole pad/node/element sweep runs in a single JIT call: the per-node
    # radial trapezoid integrations allocated three dim_yf scratch arrays per
    # node and crossed into trapezoid twice.
    return t_outlet_jit(
        int(total_pads),
        int(mesh.total_e_x_film),
        int(mesh.total_e_y_film),
        int(mesh.total_e_z_film),
        np.ascontiguousarray(mesh.total_e_y_trackbl, dtype=np.int64),
        np.ascontiguousarray(mesh.total_e_y_trackcore, dtype=np.int64),
        int(mesh.total_nodes),
        np.ascontiguousarray(mesh.n_index, dtype=np.int64),
        int(mesh.total_elements),
        np.ascontiguousarray(mesh.e_index, dtype=np.int64),
        np.ascontiguousarray(mesh.node_j, dtype=np.int64),
        np.ascontiguousarray(mesh.node_k, dtype=np.int64),
        np.ascontiguousarray(mesh.match_nodes_xz, dtype=np.int64),
        float(pads.pad_thickness),
        np.ascontiguousarray(mesh.y_3d, dtype=np.float64),
        np.ascontiguousarray(pads.length_track, dtype=np.float64),
        np.ascontiguousarray(pads.depth_track, dtype=np.float64),
        np.ascontiguousarray(pads.axial_length, dtype=np.float64),
        np.ascontiguousarray(pads.axial_length_dam, dtype=np.float64),
        np.ascontiguousarray(pads.axial_length_track, dtype=np.float64),
        np.ascontiguousarray(pads.pad_length, dtype=np.float64),
        np.ascontiguousarray(mesh.x, dtype=np.float64),
        np.ascontiguousarray(mesh.z, dtype=np.float64),
        np.ascontiguousarray(h_n, dtype=np.float64),
        np.ascontiguousarray(temp_3d, dtype=np.float64),
        np.ascontiguousarray(mesh.e_width, dtype=np.float64),
        np.ascontiguousarray(velocity_x_n, dtype=np.float64),
        np.ascontiguousarray(q_x, dtype=np.float64),
        int(mesh.x.shape[1]),
        int(mesh.match_nodes_xz.shape[1]),
    )


def temp_mixing_carryover(
    total_pads,
    operating_type,
    q_in,
    q_carryover,
    temp_sump,
    temp_outlet,
    temp_inlet,
):
    """Pad inlet temperature from the conventional hot-oil mixing model.

    Used for all bearings except spray-bar. The inlet
    temperature of pad ``p`` mixes the hot oil carried over from the upstream
    pad (``p - 1``, wrapping the first pad to the last) with fresh sump oil,
    weighted by flow. If the carryover exceeds the pad inlet flow the inlet is
    simply the upstream outlet. For axial-flow (``operating_type == "axial_flow"``) or
    ring-lubricated (``operating_type == "oil_ring_lubricated"``) bearings every inlet is fixed at
    the sump temperature.

    Parameters
    ----------
    total_pads : int
        Number of pads.
    operating_type : str
        Lubrication model, one of
        :data:`~ross.bearings.fluid_film.constants.OPERATING_TYPES`.
    q_in, q_carryover : array_like of float
        Per-pad inlet and carryover flow rates, m^3/s.
    temp_sump : float
        Sump (reservoir) temperature, K.
    temp_outlet : array_like of float
        Per-pad outlet temperature.
    temp_inlet : array_like of float
        Per-pad inlet temperature (overwritten copy returned).

    Returns
    -------
    numpy.ndarray of float
        Updated per-pad inlet temperature (same layout as ``temp_inlet``).
    """
    temp_inlet = np.array(temp_inlet, dtype=float, copy=True)
    for pad_index in range(total_pads):
        # Upstream pad (0-based), wrapping the first pad to the last.
        u = total_pads - 1 if pad_index == 0 else pad_index - 1
        p = pad_index
        if q_carryover[u] > q_in[p]:
            temp_inlet[p] = temp_outlet[u]
        else:
            temp_inlet[p] = (
                q_carryover[u] * temp_outlet[u] + (q_in[p] - q_carryover[u]) * temp_sump
            ) / q_in[p]

    if operating_type in ("axial_flow", "oil_ring_lubricated"):
        for pad_index in range(total_pads):
            temp_inlet[pad_index] = temp_sump

    return temp_inlet


def temp_mixing_spray_bar(
    total_pads,
    operating,
    q_in,
    q_carryover,
    q_supply,
    temp_outlet,
    temp_inlet,
):
    """Pad inlet temperature for spray-bar bearings.

    Each pad receives an equal share ``q_supply / total_pads`` of fresh spray
    oil at ``temp_supply`` mixed with hot oil carried over from the upstream
    pad (capped at the available carryover), and the inlet is floored at the
    supply temperature. For axial-flow bearings (``operating_type ==
    "axial_flow"``) every inlet is fixed at the supply temperature.

    Parameters
    ----------
    total_pads : int
        Number of pads.
    operating : OperatingPoint
        Speed and pressure conditions of the case; supplies the spray oil
        supply temperature (also the inlet floor) and the ``operating_type``
        that pins every inlet to it for axial flow.
    q_in, q_carryover : array_like of float
        Per-pad inlet and carryover flow rates, m^3/s.
    q_supply : float
        Total spray supply flow rate, m^3/s.
    temp_outlet : array_like of float
        Per-pad outlet temperature.
    temp_inlet : array_like of float
        Per-pad inlet temperature (overwritten copy returned).

    Returns
    -------
    numpy.ndarray of float
        Updated per-pad inlet temperature (same layout as ``temp_inlet``).
    """
    temp_inlet = np.array(temp_inlet, dtype=float, copy=True)
    q_share = q_supply / total_pads
    for idx in range(total_pads):
        # Upstream pad (0-based), wrapping the first pad to the last.
        u = total_pads - 1 if idx == 0 else idx - 1
        p = idx
        q1 = q_in[p] - q_share
        q_hot = min(q_carryover[u], q1)
        temp_inlet[p] = (
            q_share * operating.temp_supply + q_hot * temp_outlet[u]
        ) / q_in[p]
        temp_inlet[p] = max(temp_inlet[p], operating.temp_supply)

    if operating.operating_type == "axial_flow":
        for idx in range(total_pads):
            temp_inlet[idx] = operating.temp_supply

    return temp_inlet


def temp_inlet_residual(total_pads, temp_inlet_old, temp_inlet):
    """RMS of the pad inlet-temperature change between two outer iterations.

    Parameters
    ----------
    total_pads : int
        Number of pads.
    temp_inlet_old, temp_inlet : array_like of float
        Previous and current per-pad inlet temperatures, K, indexed
        ``[pad_index]``.

    Returns
    -------
    float
        ``sqrt(sum((temp_inlet - temp_inlet_old)**2) / total_pads)``.
    """
    total = 0.0
    for pad_index in range(total_pads):
        total += (temp_inlet[pad_index] - temp_inlet_old[pad_index]) ** 2
    return np.sqrt(total / total_pads)
