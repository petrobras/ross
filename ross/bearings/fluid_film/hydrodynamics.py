"""Isoviscous hydrodynamic analysis of the bearing film.

Solves the film for a *fixed* journal and pad position (:func:`fixed_brg`) and
drives the Newton search on the journal equilibrium position plus the
starvation / film-onset iteration (:func:`hydrodynamics`). It also computes
the pad forces, pivot moment, flow rates, velocities, the turbulence flow
regime and effective viscosity, and the velocity gradients that feed the shear
stress and heat dissipation.

Indexing / data-structure convention
-------------------------------------
As everywhere in the package, arrays are 0-based and allocated exactly as
large as needed:

* connectivity arrays (``node_i_reynolds``, ...) store node numbers used
  directly as indices into the coordinate and field arrays;
* loops run ``range(total_*)``, and ``film_onset`` / ``n_index_*`` / ``node_*``
  likewise store values used directly as indices;
* per-pad arrays are shaped ``(total_pads, dim*)`` and indexed
  ``[pad_index, node]``;
* ``match_nodes_xz`` stores 3-D node numbers with ``-1`` for unused slots.

So the arrays produced by :func:`~ross.bearings.fluid_film.mesh.mesh_reynolds`,
:func:`~ross.bearings.fluid_film.mesh.mesh_energy` and
:func:`~ross.bearings.fluid_film.mesh.mesh_3d` wire straight into these functions with
no reindexing.

Every routine returns its outputs as numpy arrays or a dict; the caller
assigns them back. No module global holds per-call state.

Injected dependencies
---------------------
These callables are passed in rather than imported, to avoid an import cycle
with the ``pressure`` / ``coefficients`` modules:

``press(...)`` (from :mod:`ross.bearings.fluid_film.pressure`)
    Solves the Reynolds pressure on one pad. Called by :func:`fixed_brg`,
    which imports it lazily if not supplied.
``jacobian(...)`` (from :mod:`ross.bearings.fluid_film.coefficients`)
    Builds the journal/pad stiffness Jacobian. Called by
    :func:`hydrodynamics` during the equilibrium Newton search.
``integrate_xz(pad_index, mesh, f) -> inte_f``
    Surface integral of the nodal field ``f`` over the Reynolds (film) mesh.
``trapezoid(t, f, start, stop) -> inte``
    Trapezoidal integral of ``f`` against ``t`` over the samples
    ``t[start:stop]``.

References
----------
.. [1] Reichardt, H. (1951). Vollstaendige Darstellung der turbulenten
       Geschwindigkeitsverteilung in glatten Leitungen. ZAMM, 31(7),
       208-219. (The eddy-viscosity wall law used for the turbulent
       effective viscosity.)
.. [2] Safar, Z., & Szeri, A. Z. (1974). Thermohydrodynamic lubrication in
       laminar and turbulent regimes. ASME Journal of Lubrication
       Technology, 96(1), 48-56.
"""

import numpy as np

from ross.bearings.fluid_film._numba_kernels import (
    dudy_dwdy_jit,
    effective_viscosity_jit,
    film_thickness_jit,
    film_thickness_rebuild_jit,
    flow_rates_jit,
    flow_regime_jit,
    update_shear_jit,
    velocity_jit,
)
from ross.bearings.fluid_film.constants import (
    CLOSED_FILM,
    FILM_FRACTION_ERROR,
    MAX_ITERATION,
    MAX_STEP_HALVINGS,
    NEWTON_ERROR,
    PI,
    SHEAR_ERROR,
    STARVED_NEWTON_SCALE,
    UNLOADED_STEP,
    TILT_ERROR,
    TILT_MOMENT_ERROR,
)


def hydrodynamics(
    state,
    fx_ext,
    fy_ext,
    equilibrium_type,
    starve_number,
    sump_index,
    elasto_index,
    mixtemp_index,
    jtemp_index,
    vis_index,
    relaxp,
    fixed_brg_fn,
    jacobian_fn,
    integrate_xz,
    press=None,
    filmonset_search=0,
):
    """Drive the journal-equilibrium and film-onset (starvation) iterations.

    This driver is almost pure orchestration over :func:`fixed_brg`,
    ``jacobian``, :func:`velocity`, :func:`flow_rates` and friends, each of
    which wants a large slice of solver state. Rather than re-list every field
    as a parameter, it takes the mutable solver-state mapping ``state``
    directly, plus the handful of scalars and callables the loop logic itself
    needs. It updates ``state`` in place with the converged fields and returns
    it together with the converged journal position and the convergence flag.

    The control flow is:

    * outer loop over the film-onset location (``1 .. total_e_x_film - 1``);
    * inner Newton search on ``(xj, yj)`` using the effective stiffness from
      ``jacobian`` + :func:`effective_stiffness` (skipped when
      ``equilibrium_type == "match_eccentricity"``, i.e. a prescribed journal
      position);
    * divergence handling (restore the closest position after 5 diverging
      steps, set ``hd_converged``);
    * post-convergence :func:`velocity` and :func:`flow_rates`;
    * starvation update (:func:`available_flow_even` / :func:`cool_oil` +
      :func:`sort_ascending` + :func:`available_flow_uneven`) and
      :func:`update_film_onset`, unless the operating type is flooded or the
      search key is off;
    * final :func:`cavitation_scale`.

    Parameters
    ----------
    state : dict
        Mutable solver-state mapping holding the state objects (``mesh``,
        ``pads``, ``lube``, ``operating``), every film / pressure / velocity
        field and the loose scalars (``total_pads``, ``operating_type``,
        ``xj``, ``yj``, ``cb``, ``speed_surface``, ``q_supply``,
        ``hotoil_lamda``, ``temp_3d``, ``temp_inlet``, ...). It is updated in
        place. See the module docstring for the field naming.
    fx_ext, fy_ext : float
        External load components, N.
    equilibrium_type : str
        ``"match_load"`` to solve for the equilibrium journal position with
        the Newton search, ``"match_eccentricity"`` for a prescribed journal
        position (no search).
    starve_number : int
        Number of mix-temperature iterations over which the film-onset search
        stays active.
    sump_index, elasto_index, mixtemp_index, jtemp_index, vis_index : int
        Outer-iteration counters that gate the film-onset search.
    relaxp : float
        Relaxation factor for the journal-position update. A warm-started
        case (see :class:`~ross.bearings.fluid_film.sweep.WarmStart`) sets
        ``state["_newton_relax"]``, which replaces it for the position update
        only (the shear-stress iteration keeps ``relaxp``).
    fixed_brg_fn : callable
        ``fixed_brg_fn(state, equilpost_index, relaxp, press=press)`` ->
        the :func:`fixed_brg` result dict; solves the film at the current
        fixed journal and pad position.
    jacobian_fn : callable
        ``jacobian_fn(state)`` ->
        :class:`~ross.bearings.fluid_film.state.CoefficientBlock`
        (``ross.bearings.fluid_film.coefficients.jacobian``, injected to avoid an
        import cycle). Its blocks are passed straight to
        :func:`effective_stiffness`.
    integrate_xz : callable
        ``integrate_xz(pad_index, mesh, f)`` -> the surface integral of the
        nodal field ``f`` over the Reynolds mesh.
    press : callable, optional
        Reynolds pressure solver, forwarded unchanged to ``fixed_brg_fn``.
    filmonset_search : int, optional
        Film-onset search switch (default 0 = off).

    Returns
    -------
    dict
        ``state`` (updated in place), with at least the keys ``xj``, ``yj``,
        ``fx_hydro``, ``fy_hydro``, ``fx_hydroi``, ``fy_hydroi``,
        ``hd_converged`` and the converged film / pressure / flow fields.

    Notes
    -----
    Listing all ~120 state fields explicitly would duplicate the bookkeeping
    :mod:`ross.bearings.fluid_film.driver` already does, which is why this routine
    takes the state mapping rather than a parameter per field.
    """
    s = state
    mesh = state["mesh"]
    pads = state["pads"]
    lube = state["lube"]
    operating = state["operating"]
    total_pads = s["total_pads"]
    operating_type = s["operating_type"]
    total_e_x_film = s["total_e_x_film"]
    monitor = s.get("_monitor")
    newton_tolerance = NEWTON_ERROR * s.get("_tolerance_scale", 1.0)
    # Relaxation of the journal-position update: ``relaxp``, or the value a
    # warm-started case sets (full Newton steps from a nearby equilibrium).
    step_relax = s.get("_newton_relax", relaxp)

    f = 0.0
    f_old = 0.0
    xj_old = s["xj"]
    yj_old = s["yj"]
    hd_converged = 0
    onset_moves = np.zeros(total_pads, dtype=np.int64)
    onset_travelling = 0
    onset_searched = False
    # The mass-conserving film keeps its film onset at the leading edge and
    # starves the pad through the film fraction of its inlet instead.
    film = s.get("_film")
    fraction_change = 0.0
    inlet_previous = None
    fraction_tolerance = FILM_FRACTION_ERROR * s.get("_tolerance_scale", 1.0)
    onset_passes = total_e_x_film - 1 if film is None else MAX_ITERATION
    # Inside the inlet film-fraction loop of a starved mass-conserving case
    # the force balance is solved to a tighter tolerance, so that the oil the
    # film draws, which sets the next inlet fraction, is converged well below
    # the fraction tolerance.
    fraction_loop = (
        film is not None
        and operating_type not in _FLOODED_TYPES
        and not (
            (
                sump_index > 1
                or elasto_index > 1
                or mixtemp_index > starve_number
                or jtemp_index > 1
                or vis_index > 1
            )
            and filmonset_search == 0
        )
    )
    if fraction_loop:
        newton_tolerance *= STARVED_NEWTON_SCALE

    for filmonset_index in range(1, onset_passes + 1):
        unconverge_number = 0
        hd_converged = 0
        newton_residual = np.nan
        newton_exit = "max_iterations"
        singular_step = False
        onset_travelling = 0
        halvings = 0
        f_accepted = np.inf

        # ``equilpost_index`` is a pure iteration counter (compared ``> 1`` and
        # passed through to ``fixed_brg``); kept 1-based for readability.
        for equilpost_index in range(1, MAX_ITERATION + 1):
            fb = fixed_brg_fn(s, equilpost_index, relaxp, press=press)
            s.update(fb)

            # Mass-conserving film: a Newton step that puts the journal
            # through a pad surface (closed film) is halved back towards the
            # last solved position instead of being solved.
            if (
                film is not None
                and equilpost_index > 1
                and halvings < MAX_STEP_HALVINGS
                and _film_closed(s["h_min"], total_pads, s["cb"])
            ):
                halvings += 1
                s["xj"] = 0.5 * (xj_old + s["xj"])
                s["yj"] = 0.5 * (yj_old + s["yj"])
                continue

            forces_out = forces(
                total_pads,
                mesh,
                s["leading_angle_rad"],
                s["nodal_pressure"],
                integrate_xz,
            )
            fx_hydro = forces_out["fx_hydro"]
            fy_hydro = forces_out["fy_hydro"]

            fx_groove, fy_groove = forces_groove(
                total_pads,
                pads,
                operating,
            )
            fx_hydro += fx_groove
            fy_hydro += fy_groove

            f_old = f
            fx_net = fx_hydro + fx_ext
            fy_net = fy_hydro + fy_ext
            f = 0.5 * (fx_net**2 + fy_net**2)

            # With no external load (held-eccentricity cases) the relative
            # force test is meaningless; skip it instead of dividing by zero.
            ext_mag = np.sqrt(fx_ext**2 + fy_ext**2)
            # Without an external load the relative residual is undefined and
            # the loop cannot meet its test; it is reported as infinite.
            newton_residual = (
                np.sqrt(fx_net**2 + fy_net**2) / ext_mag if ext_mag > 0.0 else np.inf
            )
            if (
                ext_mag > 0.0
                and np.sqrt(fx_net**2 + fy_net**2) / ext_mag < newton_tolerance
                and equilpost_index > 1
            ):
                hd_converged = 0 if unconverge_number > 0 else 1
                newton_exit = "converged"
                break
            elif (
                film is not None
                and equilibrium_type != "match_eccentricity"
                and f > f_accepted
                and halvings < MAX_STEP_HALVINGS
            ):
                # Mass-conserving film: a step that raises the force
                # residual is halved back towards the position it started
                # from (a backtracking line search), instead of counting
                # towards abandoning the pass.
                halvings += 1
                s["xj"] = 0.5 * (xj_old + s["xj"])
                s["yj"] = 0.5 * (yj_old + s["yj"])
                continue
            elif film is None and f > f_old and equilpost_index > 1:
                unconverge_number += 1
                if unconverge_number > 5:
                    newton_exit = "diverged"
                    break
                f_min = f_old
                xj_min = xj_old
                yj_min = yj_old
                if unconverge_number == 5:
                    s["xj"] = xj_min
                    s["yj"] = yj_min
                del f_min
            else:
                if equilibrium_type == "match_eccentricity":
                    newton_exit = "not_applicable"
                    break
                jac = jacobian_fn(s)
                kxx_e, kxy_e, kyx_e, kyy_e = effective_stiffness(
                    total_pads,
                    jac.xx,
                    jac.xy,
                    jac.yx,
                    jac.yy,
                    jac.deltax,
                    jac.deltay,
                    jac.xdelta,
                    jac.ydelta,
                    jac.deltadelta,
                    s["bearing_type"],
                    s["k_rotate"],
                )

                delta_xj, delta_yj = _newton_step(
                    kxx_e, kxy_e, kyx_e, kyy_e, fx_net, fy_net, s["cb"]
                )
                if delta_xj is None and film is not None and ext_mag > 0.0:
                    # The mass-conserving film carries no load at this
                    # position (a starved film cavitated everywhere): step
                    # along the load from here. The reset below would send
                    # the journal back to the same unloaded position.
                    delta_xj = UNLOADED_STEP * s["cb"] * fx_ext / ext_mag
                    delta_yj = UNLOADED_STEP * s["cb"] * fy_ext / ext_mag
                elif delta_xj is None:
                    # Singular Jacobian fallback.
                    singular_step = True
                    s["xj"] = 0.0
                    s["yj"] = 0.0
                    delta_xj = 0.0
                    delta_yj = -0.2 * s["cb"]
                halvings = 0
                f_accepted = f

                xj_old = s["xj"]
                yj_old = s["yj"]
                xj_new = xj_old + delta_xj
                yj_new = yj_old + delta_yj
                s["xj"] = (1.0 - step_relax) * xj_old + step_relax * xj_new
                s["yj"] = (1.0 - step_relax) * yj_old + step_relax * yj_new

        if monitor is not None:
            _record_newton(
                monitor,
                equilpost_index,
                newton_residual,
                newton_tolerance,
                newton_exit,
                singular_step,
            )

        s["fx_hydro"] = fx_hydro
        s["fy_hydro"] = fy_hydro
        s["fx_hydroi"] = forces_out["fx_hydroi"]
        s["fy_hydroi"] = forces_out["fy_hydroi"]
        s["hd_converged"] = hd_converged

        # Velocities for the convection / flow-rate calculations.
        vx, vy, vz = velocity(
            total_pads,
            mesh,
            pads,
            s["dpdx_n"],
            s["dpdz_n"],
            s["speed_surface"],
            s["h_n"],
            s["dhdx_n"],
            s["vis_effect_3d"],
            s["velocity_x_n"],
            s["velocity_y_n"],
            s["velocity_z_n"],
        )
        s["velocity_x_n"] = vx
        s["velocity_y_n"] = vy
        s["velocity_z_n"] = vz

        flow = flow_rates(
            total_pads,
            mesh,
            operating,
            s["film_onset"],
            pads,
            vx,
            vz,
            s["h_n"],
            s["x_hmin"],
            s["hotoil_lamda"],
            s["temp_3d"],
            s["temp_inlet"],
            lube,
            s["xj"],
            s["yj"],
            s.get("q_available", np.full(total_pads, np.inf)),
        )
        if film is not None:
            # The flows above integrate full-film velocity profiles; the
            # mass-conserving film takes them from its own boundary flows.
            flow.update(
                film_boundary_flows(
                    film, mesh, total_pads, s["film_onset"], s["hotoil_lamda"]
                )
            )
        s.update(flow)

        # Starvation: stop here when flooded / suppressed.
        if operating_type in _FLOODED_TYPES:
            break

        if operating_type in ("starved_condition_even", "oil_ring_lubricated"):
            s["q_available"] = available_flow_even(
                total_pads, s["q_supply"], s["q_carryover"]
            )
        elif operating_type == "starved_condition_uneven":
            qr_supply = cool_oil(
                total_pads, s["film_onset"], s["q_x"], s["q_carryover"]
            )
            order = sort_ascending(total_pads, qr_supply)
            s["q_available"] = available_flow_uneven(
                total_pads,
                order,
                s["film_onset"],
                s["q_supply"],
                qr_supply,
                s["q_carryover"],
            )

        if (
            sump_index > 1
            or elasto_index > 1
            or mixtemp_index > starve_number
            or jtemp_index > 1
            or vis_index > 1
        ) and filmonset_search == 0:
            break

        if film is not None:
            onset_searched = True
            outflow = np.asarray(s["q_out"], dtype=float)[:total_pads]
            fraction = inlet_film_fraction(
                film.film_fraction_in,
                s["q_available"],
                film.inlet_flow,
                film.inlet_capacity,
                previous=inlet_previous,
                outflow=outflow,
                carryover=float(s["hotoil_lamda"]),
            )
            inlet_previous = (
                film.film_fraction_in.copy(),
                film.inlet_flow.copy(),
                outflow.copy(),
            )
            fraction_change = float(np.max(np.abs(fraction - film.film_fraction_in)))
            film.film_fraction_in[:] = fraction
            if fraction_change < fraction_tolerance:
                break
            continue

        onset_old = np.array(s["film_onset"], dtype=np.int64)
        s["film_onset"] = update_film_onset(
            total_pads,
            total_e_x_film,
            s["q_x"],
            s["q_available"],
            s["film_onset"],
        )
        onset_searched = True
        move = np.asarray(s["film_onset"], dtype=np.int64) - onset_old
        onset_travelling = int(np.sum((move != 0) & (move * onset_moves >= 0)))
        onset_moves = move

    if film is not None:
        if monitor is not None and (
            onset_searched or "film_onset" not in monitor.records
        ):
            monitor.record(
                "film_onset",
                filmonset_index,
                fraction_change,
                fraction_tolerance,
                fraction_change < fraction_tolerance,
                searched=onset_searched,
            )
    elif monitor is not None and (
        onset_searched or "film_onset" not in monitor.records
    ):
        # The film-onset search steps one element per pass; it has found the
        # discrete balance once no pad keeps moving in the same direction.
        # The later calls of a thermal or deformation case skip the search
        # and keep the onset it found, so they keep its record too.
        monitor.record(
            "film_onset",
            filmonset_index,
            onset_travelling,
            0.0,
            onset_travelling == 0,
            searched=onset_searched,
        )

    # Heat-dissipation scale factor for the cavitation/starvation effects:
    # the film fraction of the mass-conserving film.
    if film is not None:
        s["scale_dissip"] = np.array(film.film_fraction, copy=True)
        return s
    s["scale_dissip"] = cavitation_scale(
        total_pads,
        mesh,
        operating_type,
        s["film_onset"],
        s["q_in"],
        s["q_out"],
        s["q_x"],
        s["x_hmin"],
        s["scale_dissip"],
    )

    return s


def film_boundary_flows(film, mesh, total_pads, film_onset, hotoil_lamda):
    """Pad flows of the mass-conserving film from its boundary flows.

    The finite-volume solution gives the flow each prescribed-pressure node
    supplies to the film (``CavitationSolution.boundary_flow``, positive in);
    a face between two such nodes carries no film flow. Per pad, the inlet
    nodes (the leading edge and the nodes upstream of the film onset) give the
    net draw ``q_in``, the trailing edge the outflow ``q_out`` and the two
    axial edges the side leakage, so ``q_in = q_out + q_sides`` to the mass
    balance of the solve. The carry-over is ``hotoil_lamda * q_out``: the oil
    that leaves the trailing edge, not the flow past the minimum film used by
    :func:`flow_rates` for the Reynolds film.

    Parameters
    ----------
    film : MassConservingFilm
        Film state of the case, holding the last solution of each pad.
    mesh : ReynoldsMesh
        Film (x-z) mesh.
    total_pads : int
        Number of pads.
    film_onset : numpy.ndarray of int
        Per-pad film-onset element index.
    hotoil_lamda : float
        Hot-oil carry-over fraction.

    Returns
    -------
    dict
        ``q_in``, ``q_out``, ``q_sides``, ``q_carryover``, ``q_sidea``,
        ``q_sideb``: per-pad flows, m^3/s, signed as those of
        :func:`flow_rates` (``q_sidea`` negative when oil leaves through the
        ``z = 0`` edge, ``q_sideb`` positive when it leaves through the other).
    """
    nx = mesh.total_e_x_film + 1
    nz = mesh.total_e_z_film + 1
    out = {
        key: np.zeros(total_pads, dtype=np.float64)
        for key in ("q_in", "q_out", "q_sides", "q_carryover", "q_sidea", "q_sideb")
    }
    for p in range(total_pads):
        solution = film.records[p]
        if solution is None:
            continue
        flow = np.asarray(solution.boundary_flow, dtype=float).reshape(nx, nz)
        inlet = int(film_onset[p]) + 1
        side_a = float(np.sum(flow[inlet : nx - 1, 0]))
        side_b = float(np.sum(flow[inlet : nx - 1, nz - 1]))
        strip_in, strip_out = film.edge_flow[p]
        out["q_in"][p] = float(np.sum(flow[:inlet])) + strip_in
        out["q_out"][p] = -float(np.sum(flow[nx - 1])) + strip_out
        out["q_sides"][p] = strip_in - strip_out - (side_a + side_b)
        out["q_sidea"][p] = side_a
        out["q_sideb"][p] = -side_b
        out["q_carryover"][p] = float(hotoil_lamda) * out["q_out"][p]
    return out


_FLOODED_TYPES = ("regular_flooded", "axial_flow", "high_ambient_pressure")


def _film_closed(h_min, total_pads, cb):
    """Whether the film of any pad is closed (the journal crosses a pad)."""
    return bool(np.min(np.asarray(h_min, dtype=float)[:total_pads]) <= CLOSED_FILM * cb)


def _record_newton(monitor, iterations, residual, tolerance, exit_kind, singular):
    """Record the last pass of the equilibrium Newton loop.

    The pass converged when its final force residual ``|F_net| / |F_ext|`` is
    below the tolerance, whatever the divergence counter of the loop says.
    """
    if exit_kind == "not_applicable":
        monitor.record("newton", iterations, np.nan, tolerance, True, "not_applicable")
        return
    converged = bool(np.isfinite(residual) and residual < tolerance)
    if converged:
        reason = "converged"
    elif np.isnan(residual):
        reason = "non_finite"
    elif singular:
        reason = "singular_jacobian"
    elif exit_kind == "diverged":
        reason = "diverged"
    else:
        reason = "max_iterations"
    monitor.record("newton", iterations, residual, tolerance, converged, reason)


def _newton_step(kxx_e, kxy_e, kyx_e, kyy_e, fx_net, fy_net, cb):
    """One Newton step on the journal position, with its fallbacks.

    Handles near-singular / decoupled stiffness, inverts the 2x2 system
    otherwise, and limits the step length to ``0.2 * cb``. Returns ``(None, None)`` for the
    fully singular case so the caller applies the special reset. A non-finite
    stiffness, or a zero direct stiffness that the decoupled branches below
    would divide by (``kyy_e == 0``, or ``kxx_e == 0`` with
    ``|kyy_e| <= 1e-6``), is treated as singular: it occurs when no pad carries
    a film force, e.g. every pad pushed out of contact by a soft pivot. A zero
    ``kxx_e`` with a finite ``|kyy_e| > 1e-6`` keeps the decoupled y-step.
    """
    stiffness = (kxx_e, kxy_e, kyx_e, kyy_e)
    if (
        not np.all(np.isfinite(stiffness))
        or kyy_e == 0.0
        or (kxx_e == 0.0 and abs(kyy_e) <= 1.0e-6)
    ):
        return None, None
    if abs(kxx_e / kyy_e) < 1.0e-4 and abs(kyy_e) > 1.0e-6:
        delta_xj = 0.0
        delta_yj = fy_net / kyy_e
    elif abs(kyy_e / kxx_e) < 1.0e-4 and abs(kxx_e) > 1.0e-6:
        delta_xj = fx_net / kxx_e
        delta_yj = 0.0
    elif (
        abs(kxy_e) < 1.0e-6
        and abs(kyx_e) < 1.0e-6
        and abs(kxx_e) > 1.0e-6
        and abs(kyy_e) > 1.0e-6
    ):
        delta_xj = fx_net / kxx_e
        delta_yj = fy_net / kyy_e
    elif abs(kxx_e * kyy_e - kxy_e * kyx_e) < 1.0e-6:
        return None, None
    else:
        det = kxx_e * kyy_e - kxy_e * kyx_e
        delta_xj = (kyy_e * fx_net - kxy_e * fy_net) / det
        delta_yj = (-kyx_e * fx_net + kxx_e * fy_net) / det

    if np.sqrt(delta_xj**2 + delta_yj**2) > 0.2 * cb:
        # Sequential rather than simultaneous rescaling: delta_xj is scaled
        # then delta_yj uses the NEW delta_xj in its norm denominator. This is
        # not a true uniform clamp, but it is deliberate: the pinned regression
        # fixtures depend on it, and it is required for journal-equilibrium
        # agreement on pressure-dam variants, where this clamp activates
        # during the Newton iteration. Do not "fix" it into a uniform clamp.
        delta_xj = delta_xj * 0.2 * cb / np.sqrt(delta_xj**2 + delta_yj**2)
        delta_yj = delta_yj * 0.2 * cb / np.sqrt(delta_xj**2 + delta_yj**2)
    return delta_xj, delta_yj


def forces(
    total_pads,
    mesh,
    leading_angle_rad,
    nodal_pressure,
    integrate_xz,
):
    """Integrate the pad pressure into the global fluid forces.

    The returned forces are the *negative* of the pressure resultant (the force
    the film exerts on the journal), summed over pads, plus per-pad components.

    Parameters
    ----------
    total_pads : int
        Number of pads.
    mesh : ReynoldsMesh
        Film (x-z) mesh: coordinates, connectivity and index maps.
    leading_angle_rad : numpy.ndarray
        Per-pad leading-edge angle, rad.
    nodal_pressure : numpy.ndarray
        Film pressure at Reynolds nodes, Pa, shape ``(total_pads, dim_xz)``.
    integrate_xz : callable
        ``integrate_xz(pad_index, mesh, f)`` -> the surface integral of the
        nodal field ``f`` over the Reynolds mesh.

    Returns
    -------
    dict
        ``fx_hydro``, ``fy_hydro`` : float
            Total fluid forces, N.
        ``fx_hydroi``, ``fy_hydroi`` : numpy.ndarray, shape ``(total_pads,)``
            Per-pad fluid forces, N.
    """
    x_reynolds_rad = mesh.x_rad
    nodal_pressure = np.asarray(nodal_pressure, dtype=float)
    x_reynolds_rad = np.asarray(x_reynolds_rad, dtype=float)
    idx = np.asarray(mesh.n_index[: mesh.total_nodes], dtype=np.int64)

    fx_hydroi = np.zeros(total_pads, dtype=float)
    fy_hydroi = np.zeros(total_pads, dtype=float)
    fx_hydro = 0.0
    fy_hydro = 0.0

    for pad_index in range(total_pads):
        px_i = np.zeros(mesh.dim_xz, dtype=float)
        py_i = np.zeros(mesh.dim_xz, dtype=float)
        angle = leading_angle_rad[pad_index] + x_reynolds_rad[pad_index, idx]
        press_neg = -nodal_pressure[pad_index, idx]
        px_i[idx] = press_neg * np.cos(angle)
        py_i[idx] = press_neg * np.sin(angle)

        force_x_i = integrate_xz(
            pad_index,
            mesh,
            px_i,
        )
        force_y_i = integrate_xz(
            pad_index,
            mesh,
            py_i,
        )
        fx_hydro += force_x_i
        fy_hydro += force_y_i
        fx_hydroi[pad_index] = force_x_i
        fy_hydroi[pad_index] = force_y_i

    return {
        "fx_hydro": fx_hydro,
        "fy_hydro": fy_hydro,
        "fx_hydroi": fx_hydroi,
        "fy_hydroi": fy_hydroi,
    }


def forces_groove(
    total_pads,
    pads,
    operating,
):
    """Hydrostatic forces from the inter-pad grooves.

    The groove pressure is the average of the two ambient pressures; the
    resultant on each groove acts radially through its angular centre.

    Parameters
    ----------
    total_pads : int
        Number of pads.
    pads : PadGeometry
        Per-pad geometry; the leading-edge angle, arc length, axial length
        and journal radius set the extent of each groove.
    operating : OperatingPoint
        Speed and pressure conditions of the case; the groove pressure is the
        average of its two ambient pressures.

    Returns
    -------
    tuple of float
        ``(fx_groove, fy_groove)`` total groove forces, N.
    """
    press_groove = 0.5 * (operating.ambient_press1 + operating.ambient_press2)
    fx_groove = 0.0
    fy_groove = 0.0

    for pad_index in range(total_pads):
        if pad_index == 0:
            # First pad: the upstream groove wraps to the last pad.
            leading_angle = (
                pads.leading_angle_rad[total_pads - 1]
                + pads.arc_length_rad[total_pads - 1]
            )
            trailing_angle = pads.leading_angle_rad[pad_index]
            arc_length_groove = trailing_angle - leading_angle
            if arc_length_groove < 0.0:
                arc_length_groove = 2.0 * PI + arc_length_groove
        else:
            leading_angle = (
                pads.leading_angle_rad[pad_index - 1]
                + pads.arc_length_rad[pad_index - 1]
            )
            trailing_angle = pads.leading_angle_rad[pad_index]
            arc_length_groove = trailing_angle - leading_angle
            if arc_length_groove < 0.0:
                # The groove crosses 0 rad (pads not listed in increasing
                # angle). This branch used to add 360 (degrees) to a value in
                # radians; no fixture reaches it.
                arc_length_groove = 2.0 * PI + arc_length_groove
        alpha = 0.5 * arc_length_groove
        groove_angle = trailing_angle - alpha

        fr = (
            2.0
            * pads.axial_length[pad_index]
            * pads.journal_radius
            * press_groove
            * np.sin(alpha)
        )
        fx_groove += -fr * np.cos(groove_angle)
        fy_groove += -fr * np.sin(groove_angle)

    return fx_groove, fy_groove


def effective_stiffness(
    total_pads,
    kxx,
    kxy,
    kyx,
    kyy,
    kdeltax,
    kdeltay,
    kxdelta,
    kydelta,
    kdeltadelta,
    bearing_type,
    k_rotate,
):
    """Effective journal stiffness used for the equilibrium Newton search.

    For tilting-pad bearings the pad-tilt degrees of freedom are statically
    condensed out of the journal stiffness.

    Parameters
    ----------
    total_pads : int
        Number of pads.
    kxx, kxy, kyx, kyy : float
        Journal-to-journal stiffness components, N/m.
    kdeltax, kdeltay, kxdelta, kydelta, kdeltadelta, k_rotate : numpy.ndarray
        Per-pad tilt-coupling, tilt-tilt and pivot-rotational stiffness.
    bearing_type : str
        Bearing geometry, one of
        :data:`~ross.bearings.fluid_film.constants.BEARING_TYPES`; only the
        tilting-pad types trigger the condensation.

    Returns
    -------
    tuple of float
        ``(kxx_effect, kxy_effect, kyx_effect, kyy_effect)``.
    """
    kxx_effect = kxx
    kyx_effect = kyx
    kxy_effect = kxy
    kyy_effect = kyy

    if bearing_type in (
        "conventional_tilting_pad",
        "inlet_groove_tilting_pad",
        "spray_bar_tilting_pad",
    ):
        for pad_index in range(total_pads):
            if abs(kdeltadelta[pad_index]) > 1.0e-8:
                denom = kdeltadelta[pad_index] + k_rotate[pad_index]
                kxx_effect -= kxdelta[pad_index] * kdeltax[pad_index] / denom
                kyx_effect -= kydelta[pad_index] * kdeltax[pad_index] / denom
                kxy_effect -= kxdelta[pad_index] * kdeltay[pad_index] / denom
                kyy_effect -= kydelta[pad_index] * kdeltay[pad_index] / denom

    return kxx_effect, kxy_effect, kyx_effect, kyy_effect


def velocity(
    total_pads,
    mesh,
    pads,
    dpdx_n,
    dpdz_n,
    speed_surface,
    h_n,
    dhdx_n,
    vis_effect_3d,
    velocity_x_n,
    velocity_y_n,
    velocity_z_n,
):
    """Velocity components of the film flow field.

    The circumferential (``U``) and axial (``W``) velocities follow
    analytically from the Reynolds derivation; the radial (``V``) velocity is
    approximated assuming Couette flow. Within the pocket the radial integral
    is taken from the pad surface (limit ``1``), in the dam region from the dam
    floor (limit ``limit1``).

    Parameters
    ----------
    total_pads : int
        Number of pads.
    mesh : ReynoldsMesh
        Film (x-z) mesh: coordinates, connectivity and index maps, plus the
        radial coordinate ``y_3d`` of the 3-D film nodes.
    pads : PadGeometry
        Per-pad geometry.
    dpdx_n, dpdz_n : numpy.ndarray
        Pressure gradients at Reynolds nodes, Pa/m, shape
        ``(total_pads, dim_xz)``.
    speed_surface : float
        Journal surface speed, m/s.
    h_n, dhdx_n : numpy.ndarray
        Film thickness (m) and its circumferential derivative, shape
        ``(total_pads, dim_xz)``.
    vis_effect_3d : numpy.ndarray
        Effective viscosity on the 3-D mesh, Pa*s, shape
        ``(total_pads, dim_3d)``.
    velocity_x_n, velocity_y_n, velocity_z_n : numpy.ndarray
        Velocity fields on the 3-D mesh, m/s, shape ``(total_pads, dim_3d)``
        (updated copies returned).

    Returns
    -------
    tuple of numpy.ndarray
        ``(velocity_x_n, velocity_y_n, velocity_z_n)``.
    """
    axial_length = pads.axial_length
    axial_length_dam = pads.axial_length_dam
    axial_length_track = pads.axial_length_track
    depth_track = pads.depth_track
    length_track = pads.length_track
    pad_length = pads.pad_length
    match_nodes_xz = mesh.match_nodes_xz
    n_index_reynolds = mesh.n_index
    total_e_y_trackbl = mesh.total_e_y_trackbl
    total_e_y_trackcore = mesh.total_e_y_trackcore
    x_reynolds = mesh.x
    y_3d = mesh.y_3d
    z_reynolds = mesh.z
    n_index_reynolds = np.ascontiguousarray(n_index_reynolds, dtype=np.int64)
    match_nodes_xz = np.ascontiguousarray(match_nodes_xz, dtype=np.int64)
    dpdx_n = np.ascontiguousarray(dpdx_n, dtype=np.float64)
    dpdz_n = np.ascontiguousarray(dpdz_n, dtype=np.float64)
    x_reynolds = np.ascontiguousarray(x_reynolds, dtype=np.float64)
    z_reynolds = np.ascontiguousarray(z_reynolds, dtype=np.float64)
    h_n = np.ascontiguousarray(h_n, dtype=np.float64)
    dhdx_n = np.ascontiguousarray(dhdx_n, dtype=np.float64)
    vis_effect_3d = np.ascontiguousarray(vis_effect_3d, dtype=np.float64)
    y_3d = np.ascontiguousarray(y_3d, dtype=np.float64)
    pad_length = np.ascontiguousarray(pad_length, dtype=np.float64)
    axial_length = np.ascontiguousarray(axial_length, dtype=np.float64)
    length_track = np.ascontiguousarray(length_track, dtype=np.float64)
    depth_track = np.ascontiguousarray(depth_track, dtype=np.float64)
    axial_length_dam = np.ascontiguousarray(axial_length_dam, dtype=np.float64)
    axial_length_track = np.ascontiguousarray(axial_length_track, dtype=np.float64)
    total_e_y_trackbl = np.ascontiguousarray(total_e_y_trackbl, dtype=np.int64)
    total_e_y_trackcore = np.ascontiguousarray(total_e_y_trackcore, dtype=np.int64)
    velocity_x_n = np.ascontiguousarray(velocity_x_n, dtype=np.float64)
    velocity_y_n = np.ascontiguousarray(velocity_y_n, dtype=np.float64)
    velocity_z_n = np.ascontiguousarray(velocity_z_n, dtype=np.float64)

    return velocity_jit(
        total_pads,
        mesh.dim_yf,
        mesh.total_e_y_film,
        total_e_y_trackbl,
        total_e_y_trackcore,
        mesh.total_nodes,
        n_index_reynolds,
        match_nodes_xz,
        pad_length,
        axial_length,
        float(pads.pad_thickness),
        length_track,
        depth_track,
        axial_length_dam,
        axial_length_track,
        dpdx_n,
        dpdz_n,
        float(speed_surface),
        x_reynolds,
        z_reynolds,
        h_n,
        dhdx_n,
        vis_effect_3d,
        y_3d,
        velocity_x_n,
        velocity_y_n,
        velocity_z_n,
    )


def available_flow_even(total_pads, q_supply, q_carryover):
    """Available oil per pad with the total supply split evenly (model 1).

    Each groove receives an equal share of the supply plus the hot-oil carried
    over from the upstream pad.

    Parameters
    ----------
    total_pads : int
        Number of pads.
    q_supply : float
        Total supply flow.
    q_carryover : numpy.ndarray
        Per-pad carry-over flow (0-based natural).

    Returns
    -------
    numpy.ndarray
        ``q_available`` per pad (0-based natural).
    """
    q_carryover = np.asarray(q_carryover, dtype=float)
    q_available = np.zeros(total_pads, dtype=float)
    for pad_index in range(total_pads):
        if pad_index == 0:
            q_available[pad_index] = q_carryover[total_pads - 1] + q_supply / total_pads
        else:
            q_available[pad_index] = q_carryover[pad_index - 1] + q_supply / total_pads
    return q_available


def inlet_film_fraction(
    film_fraction_in,
    q_available,
    inlet_flow,
    inlet_capacity,
    previous=None,
    outflow=None,
    carryover=0.0,
):
    """Update the film fraction of the oil entering each pad (mass-conserving film).

    The supply conserves the oil: a starved pad draws exactly the oil
    available at its leading edge, and a pad whose flooded draw is less than
    the available oil floods (fraction 1) and spills the rest. The net draw
    of the last solve is the Couette inflow ``theta_in C`` (``C`` the inlet
    capacity) less the backflow ``B`` the pressure gradient drives back into
    the supply once the pressurised zone reaches the leading edge.

    The available oil of pad ``p`` is its share of the supply plus
    ``carryover`` times the trailing-edge outflow of the pad before it
    (pad ``p - 1``, cyclically, as
    :func:`available_flow_even`), and that outflow grows with the inlet
    fraction of that pad. The update is one Newton step on the coupled
    residuals ``r_p = q_available_p - draw_p``,

    ``-a_p d_p + carryover * b_(p-1) d_(p-1) = -r_p``,

    with ``a_p`` the slope of pad ``p``'s draw and ``b_p`` the slope of its
    outflow with its inlet fraction: ``C`` (``a``) and 0 (``b``) without a
    previous iterate, else the secant slopes, kept within ``[0.05 C, C]``
    and ``[0, a]``. A pad that is flooded with oil to spare
    (``theta_in = 1``, ``r_p >= 0``) keeps ``theta_in = 1`` and drops out of
    the system. The fractions are clipped to ``[0, 1]``. Without
    ``outflow`` (or with ``carryover = 0``) the pads decouple and each takes
    the step ``r_p / a_p``.

    Parameters
    ----------
    film_fraction_in : numpy.ndarray
        Inlet film fraction of the last solve of each pad.
    q_available : numpy.ndarray
        Oil available at each pad's leading edge, m**3/s.
    inlet_flow : numpy.ndarray
        Net flow each pad drew through its leading edge in its last solve,
        m**3/s.
    inlet_capacity : numpy.ndarray
        Couette flow each pad draws with a full inlet film, m**3/s.
    previous : tuple of numpy.ndarray, optional
        ``(film_fraction_in, inlet_flow)`` or ``(film_fraction_in,
        inlet_flow, outflow)`` of the iterate before the last; gives the
        secant slopes. Default is None.
    outflow : numpy.ndarray, optional
        Trailing-edge outflow of each pad in its last solve, m**3/s.
        Default is None (no coupling).
    carryover : float, optional
        Fraction of a pad's outflow carried to the next pad. Default is 0.

    Returns
    -------
    numpy.ndarray
        Inlet film fraction per pad, in ``[0, 1]``.

    Examples
    --------
    Without backflow the draw is ``theta_in C``: the first pad (draw 2,
    capacity 2) floods but 1 is available, so it starves to 0.5; the second
    (3 available) stays flooded and spills; the third has no oil.

    >>> inlet_film_fraction(
    ...     np.ones(3), np.array([1.0, 3.0, 0.0]), np.full(3, 2.0), np.full(3, 2.0)
    ... )
    array([0.5, 1. , 0. ])
    """
    theta = np.asarray(film_fraction_in, dtype=float)
    q_available = np.asarray(q_available, dtype=float)
    draw = np.asarray(inlet_flow, dtype=float)
    capacity = np.asarray(inlet_capacity, dtype=float)
    n = theta.size
    draw_slope = capacity.copy()
    out_slope = np.zeros(n)
    coupled = outflow is not None and carryover > 0.0
    if previous is not None:
        d_theta = theta - np.asarray(previous[0], dtype=float)
        moved = np.abs(d_theta) > 1e-12
        d_draw = draw - np.asarray(previous[1], dtype=float)
        secant = np.divide(d_draw, d_theta, out=capacity.copy(), where=moved)
        draw_slope = np.clip(secant, 0.05 * capacity, capacity)
        if coupled and len(previous) > 2:
            d_out = np.asarray(outflow, dtype=float) - np.asarray(
                previous[2], dtype=float
            )
            secant = np.divide(d_out, d_theta, out=np.zeros(n), where=moved)
            out_slope = np.clip(secant, 0.0, draw_slope)
    finite = np.isfinite(q_available)
    residual = np.where(finite, q_available - draw, 0.0)
    free = finite & (draw_slope > 0.0) & ~((theta >= 1.0) & (residual >= 0.0))
    step = np.zeros(n)
    if np.any(free):
        jac = np.diag(-draw_slope)
        if coupled:
            for p in range(n):
                jac[p, (p - 1) % n] += carryover * out_slope[(p - 1) % n]
        index = np.flatnonzero(free)
        step[index] = np.linalg.solve(jac[np.ix_(index, index)], -residual[index])
    step[finite & ~free & (residual >= 0.0)] = 1.0
    step[~finite] = 1.0
    return np.clip(theta + step, 0.0, 1.0)


def cool_oil(total_pads, film_onset, q_x, q_carryover):
    """Cool supply oil required to fill the current film-onset location.

    Parameters
    ----------
    total_pads : int
        Number of pads / circumferential array dimension.
    film_onset : numpy.ndarray of int
        Per-pad film-onset element index (0-based natural).
    q_x : numpy.ndarray
        Circumferential flow rate at each x station, shape
        ``(total_pads, dim_x)``.
    q_carryover : numpy.ndarray
        Per-pad carry-over flow (0-based natural).

    Returns
    -------
    numpy.ndarray
        ``qr_supply`` required supply per pad (0-based natural).
    """
    q_x = np.asarray(q_x, dtype=float)
    q_carryover = np.asarray(q_carryover, dtype=float)
    qr_supply = np.zeros(total_pads, dtype=float)
    for pad_index in range(total_pads):
        if pad_index == 0:
            dq = (
                q_x[pad_index, int(film_onset[pad_index])] - q_carryover[total_pads - 1]
            )
        else:
            dq = q_x[pad_index, int(film_onset[pad_index])] - q_carryover[pad_index - 1]
        qr_supply[pad_index] = dq
    return qr_supply


def sort_ascending(n, a):
    """Stable insertion sort returning the ascending permutation of ``a``.

    Returns the 0-based permutation list ``l`` such that ``a[l[0]] <= a[l[1]]
    <= ...``.

    Parameters
    ----------
    n : int
        Number of entries to sort.
    a : numpy.ndarray
        Values, 0-based natural (entry ``i`` at slot ``i``), at least ``n``
        long.

    Returns
    -------
    numpy.ndarray of int
        Permutation ``l`` (0-based values), sized ``max(n, 10)``.
    """
    a = np.asarray(a, dtype=float)
    size = max(n, 10)
    order = np.zeros(size, dtype=np.int64)
    for i in range(n):
        order[i] = i
    for i in range(1, n):
        for j in range(i - 1, -1, -1):
            if a[order[j + 1]] > a[order[j]]:
                break
            k = order[j]
            order[j] = order[j + 1]
            order[j + 1] = k
    return order


def available_flow_uneven(
    total_pads, order, film_onset, q_supply, qr_supply, q_carryover
):
    """Available oil per pad with extra flow redirected (model 2).

    Supply oil is distributed starting from the pad that requires the least;
    any surplus from a choked groove is spread evenly over the remaining
    grooves.

    Parameters
    ----------
    total_pads : int
        Number of pads.
    order : numpy.ndarray of int
        Ascending permutation from :func:`sort_ascending` (0-based natural).
    film_onset : numpy.ndarray of int
        Per-pad film-onset element index (0-based natural).
    q_supply : float
        Total supply flow.
    qr_supply : numpy.ndarray
        Per-pad required supply (0-based natural).
    q_carryover : numpy.ndarray
        Per-pad carry-over flow (0-based natural).

    Returns
    -------
    numpy.ndarray
        ``q_available`` per pad (0-based natural).
    """
    order = np.asarray(order, dtype=np.int64)
    qr_supply = np.asarray(qr_supply, dtype=float)
    q_carryover = np.asarray(q_carryover, dtype=float)

    q_supply_i = np.zeros(total_pads, dtype=float)
    q_supply_even = q_supply / total_pads
    q_rest = q_supply
    for i in range(total_pads):
        li = int(order[i])
        # ``film_onset`` value 0 == the leading film element.
        if q_supply_even > qr_supply[li] and int(film_onset[li]) == 0:
            q_supply_i[li] = max(qr_supply[li], 0.0)
        else:
            q_supply_i[li] = q_supply_even
        q_rest -= q_supply_i[li]
        if total_pads - 1 - i != 0:
            q_supply_even = q_rest / (total_pads - 1 - i)

    q_available = np.zeros(total_pads, dtype=float)
    for pad_index in range(total_pads):
        if pad_index == 0:
            q_available[pad_index] = q_carryover[total_pads - 1] + q_supply_i[pad_index]
        else:
            q_available[pad_index] = q_carryover[pad_index - 1] + q_supply_i[pad_index]
    return q_available


def update_film_onset(total_pads, total_e_x_film, q_x, q_available, film_onset):
    """Advance / retreat the per-pad film-onset location.

    If the available flow cannot fill the current onset station the onset moves
    one element downstream; if there is surplus it may move one element
    upstream. The onset is capped at ``total_e_x_film - 1`` (fully cavitated
    pad).

    Parameters
    ----------
    total_pads, total_e_x_film : int
        Number of pads / array dimension / circumferential element count.
    q_x : numpy.ndarray
        Circumferential flow rate, shape ``(total_pads, dim_x)``.
    q_available : numpy.ndarray
        Per-pad available flow (0-based natural).
    film_onset : numpy.ndarray of int
        Per-pad film-onset element index (0-based natural; updated copy
        returned).

    Returns
    -------
    numpy.ndarray of int
        Updated ``film_onset``.
    """
    q_x = np.asarray(q_x, dtype=float)
    q_available = np.asarray(q_available, dtype=float)
    film_onset = np.ascontiguousarray(film_onset, dtype=np.int64)

    for pad_index in range(total_pads):
        fo = int(film_onset[pad_index])
        if q_available[pad_index] < q_x[pad_index, fo]:
            film_onset[pad_index] = fo + 1
        else:
            # ``fo`` is a 0-based element value; the onset may not retreat past
            # the leading element (value 0), so require ``fo - 1 >= 0``.
            if (
                q_available[pad_index] > q_x[pad_index, fo]
                or abs(q_available[pad_index] - q_x[pad_index, fo]) < 1.0e-6
            ) and (fo - 1) >= 0:
                film_onset[pad_index] = fo - 1
        # Cap a fully cavitated pad at the last interior element (0-based
        # value ``total_e_x_film - 2``).
        if int(film_onset[pad_index]) >= total_e_x_film - 1:
            film_onset[pad_index] = total_e_x_film - 2
    return film_onset


def cavitation_scale(
    total_pads,
    mesh,
    operating_type,
    film_onset,
    q_in,
    q_out,
    q_x,
    x_hmin,
    scale_dissip,
):
    """Heat-dissipation scale factor accounting for cavitation/starvation.

    In the starved upstream region and in the downstream cavitated region the
    dissipation is scaled by the ratio of the local continuous flow to the
    circumferential flow; in the continuous film region the factor is unity.
    With full cavitation suppressed (``"axial_flow"`` /
    ``"high_ambient_pressure"``) the factor is unity everywhere.

    Parameters
    ----------
    total_pads : int
        Number of pads.
    mesh : ReynoldsMesh
        Film (x-z) mesh: coordinates, connectivity and index maps.
    operating_type : str
        Lubrication model, one of
        :data:`~ross.bearings.fluid_film.constants.OPERATING_TYPES`.
    film_onset : numpy.ndarray of int
        Per-pad film-onset element index.
    q_in, q_out : numpy.ndarray
        Per-pad inlet / exit flow, m^3/s.
    q_x : numpy.ndarray
        Circumferential flow rate, m^3/s, shape ``(total_pads, dim_x)``.
    x_hmin : numpy.ndarray
        Per-pad circumferential location of the minimum film, m.
    scale_dissip : numpy.ndarray
        Dissipation scale, shape ``(total_pads, dim_xz)`` (updated copy
        returned).

    Returns
    -------
    numpy.ndarray
        Updated ``scale_dissip``.
    """
    n_index_reynolds = mesh.n_index
    x_reynolds = mesh.x
    n_index_reynolds = np.asarray(n_index_reynolds, dtype=np.int64)
    x_reynolds = np.asarray(x_reynolds, dtype=float)
    q_in = np.asarray(q_in, dtype=float)
    q_out = np.asarray(q_out, dtype=float)
    q_x = np.asarray(q_x, dtype=float)
    x_hmin = np.asarray(x_hmin, dtype=float)
    scale_dissip = np.ascontiguousarray(scale_dissip, dtype=np.float64)

    step = mesh.total_e_z_film + 1
    for pad_index in range(total_pads):
        if operating_type in (
            "regular_flooded",
            "starved_condition_even",
            "starved_condition_uneven",
            "oil_ring_lubricated",
        ):
            for i in range(mesh.total_nodes):
                node = int(n_index_reynolds[i])
                # ``node`` is a 0-based node value; its 0-based x-station is
                # ``node // step``. The upstream starved region is the nodes
                # before the film onset, i.e. ``node + 1 <= film_onset * step``
                # (the value-shifted form of
                # ``node <= (film_onset - 1) * step``).
                k = node // step
                if node < int(film_onset[pad_index]) * step:
                    scale_dissip[pad_index, node] = q_in[pad_index] / q_x[pad_index, k]
                elif (
                    x_reynolds[pad_index, node] < x_hmin[pad_index]
                    or abs(x_reynolds[pad_index, node] - x_hmin[pad_index]) < 1.0e-6
                ):
                    scale_dissip[pad_index, node] = 1.0
                else:
                    scale_dissip[pad_index, node] = q_out[pad_index] / q_x[pad_index, k]
        elif operating_type in ("axial_flow", "high_ambient_pressure"):
            for i in range(mesh.total_nodes):
                node = int(n_index_reynolds[i])
                scale_dissip[pad_index, node] = 1.0
    return scale_dissip


def flow_rates(
    total_pads,
    mesh,
    operating,
    film_onset,
    pads,
    velocity_x_n,
    velocity_z_n,
    h_n,
    x_hmin,
    hotoil_lamda,
    temp_3d,
    temp_inlet,
    lube,
    xj,
    yj,
    q_available,
):
    """Inlet/exit/side flow rates and carry-over per pad.

    The radially averaged velocities and temperature are integrated across the
    film, the circumferential flow ``Q_x`` is accumulated at every x station,
    the exit flow is taken at the minimum-film location, and the side leakage /
    enthalpy bookkeeping is computed for the loading check.

    Parameters
    ----------
    total_pads : int
        Number of pads.
    mesh : ReynoldsMesh
        Film (x-z) mesh: coordinates, element sizes, connectivity and index
        maps, plus the radial coordinate ``y_3d`` of the 3-D film nodes.
    operating : OperatingPoint
        Speed and pressure conditions of the case.
    film_onset : numpy.ndarray of int
        Per-pad film-onset element index.
    pads : PadGeometry
        Per-pad geometry.
    velocity_x_n, velocity_z_n : numpy.ndarray
        Circumferential and axial velocity on the 3-D mesh, m/s, shape
        ``(total_pads, dim_3d)``.
    h_n : numpy.ndarray
        Film thickness, m, shape ``(total_pads, dim_xz)``.
    x_hmin : numpy.ndarray
        Per-pad circumferential location of the minimum film, m.
    hotoil_lamda : float
        Hot-oil carry-over fraction.
    temp_3d : numpy.ndarray
        Temperature on the 3-D mesh, K, shape ``(total_pads, dim_3d)``.
    temp_inlet : numpy.ndarray
        Per-pad inlet temperature, K.
    lube : Lubricant
        Lubricant properties.
    xj, yj : float
        Journal-centre displacements, m (used only for the centred-dam
        exception).
    q_available : numpy.ndarray
        Per-pad available flow, m^3/s. Read-only; used only to cap
        ``q_in`` on an unloaded pad under flooded/starved operation. On the
        first pass (before the starvation routines run) the orchestrator passes
        ``+inf`` per pad so ``min(Q_x, Q_available)`` reduces to ``Q_x``,
        for the cases where ``q_available`` is otherwise uninitialised.

    Returns
    -------
    dict
        ``q_x`` : numpy.ndarray, shape ``(total_pads, dim_x)``
            Circumferential flow at each x station, m^3/s.
        ``q_in``, ``q_out``, ``q_sides``, ``q_carryover``, ``q_sidea``,
        ``q_sideb`` : numpy.ndarray, shape ``(total_pads,)``
            Per-pad flow quantities, m^3/s.
        ``t_average`` : numpy.ndarray, shape ``(total_pads, dim_xz)``
            Radially averaged nodal temperature, K.
    """
    e_index_reynolds = mesh.e_index
    e_length_reynolds = mesh.e_length
    e_width_reynolds = mesh.e_width
    match_nodes_xz = mesh.match_nodes_xz
    n_index_reynolds = mesh.n_index
    node_i_reynolds = mesh.node_i
    node_j_reynolds = mesh.node_j
    node_k_reynolds = mesh.node_k
    node_l_reynolds = mesh.node_l
    x_reynolds = mesh.x
    y_3d = mesh.y_3d
    z_reynolds = mesh.z
    n_index_reynolds = np.asarray(n_index_reynolds, dtype=np.int64)
    match_nodes_xz = np.asarray(match_nodes_xz, dtype=np.int64)
    e_index_reynolds = np.asarray(e_index_reynolds, dtype=np.int64)
    node_i_reynolds = np.asarray(node_i_reynolds, dtype=np.int64)
    node_j_reynolds = np.asarray(node_j_reynolds, dtype=np.int64)
    node_k_reynolds = np.asarray(node_k_reynolds, dtype=np.int64)
    node_l_reynolds = np.asarray(node_l_reynolds, dtype=np.int64)
    x_reynolds = np.asarray(x_reynolds, dtype=float)
    z_reynolds = np.asarray(z_reynolds, dtype=float)
    e_length_reynolds = np.asarray(e_length_reynolds, dtype=float)
    e_width_reynolds = np.asarray(e_width_reynolds, dtype=float)
    velocity_x_n = np.asarray(velocity_x_n, dtype=float)
    velocity_z_n = np.asarray(velocity_z_n, dtype=float)
    h_n = np.asarray(h_n, dtype=float)
    x_hmin = np.asarray(x_hmin, dtype=float)
    y_3d = np.asarray(y_3d, dtype=float)
    temp_3d = np.asarray(temp_3d, dtype=float)
    temp_inlet = np.asarray(temp_inlet, dtype=float)
    q_available = np.asarray(q_available, dtype=float)

    q_x = np.zeros((total_pads, mesh.dim_x), dtype=np.float64)
    q_in = np.zeros(total_pads, dtype=np.float64)
    q_out = np.zeros(total_pads, dtype=np.float64)
    q_sides = np.zeros(total_pads, dtype=np.float64)
    q_carryover = np.zeros(total_pads, dtype=np.float64)
    q_sidea = np.zeros(total_pads, dtype=np.float64)
    q_sideb = np.zeros(total_pads, dtype=np.float64)
    t_average = np.zeros((total_pads, mesh.dim_xz), dtype=np.float64)

    flow_rates_jit(
        int(total_pads),
        int(mesh.dim_yf),
        int(mesh.dim_xz),
        operating.operating_type,
        int(mesh.total_e_x_film),
        int(mesh.total_e_y_film),
        int(mesh.total_e_z_film),
        np.ascontiguousarray(mesh.total_e_y_trackbl, dtype=np.int64),
        np.ascontiguousarray(mesh.total_e_y_trackcore, dtype=np.int64),
        int(mesh.total_nodes),
        n_index_reynolds,
        match_nodes_xz,
        int(mesh.total_elements),
        e_index_reynolds,
        node_i_reynolds,
        node_j_reynolds,
        node_k_reynolds,
        node_l_reynolds,
        np.ascontiguousarray(film_onset, dtype=np.int64),
        float(pads.pad_thickness),
        np.ascontiguousarray(pads.length_track, dtype=np.float64),
        np.ascontiguousarray(pads.depth_track, dtype=np.float64),
        np.ascontiguousarray(pads.pad_length, dtype=np.float64),
        np.ascontiguousarray(pads.axial_length, dtype=np.float64),
        np.ascontiguousarray(pads.axial_length_track, dtype=np.float64),
        np.ascontiguousarray(pads.axial_length_dam, dtype=np.float64),
        x_reynolds,
        z_reynolds,
        e_length_reynolds,
        e_width_reynolds,
        velocity_x_n,
        velocity_z_n,
        h_n,
        x_hmin,
        y_3d,
        float(hotoil_lamda),
        temp_3d,
        temp_inlet,
        float(lube.density),
        float(lube.cp),
        operating.bearing_type,
        float(xj),
        float(yj),
        q_available,
        q_x,
        q_in,
        q_out,
        q_sides,
        q_carryover,
        q_sidea,
        q_sideb,
        t_average,
    )

    return {
        "q_x": q_x,
        "q_in": q_in,
        "q_out": q_out,
        "q_sides": q_sides,
        "q_carryover": q_carryover,
        "q_sidea": q_sidea,
        "q_sideb": q_sideb,
        "t_average": t_average,
    }


# ----------------------------------------------------------------------------
# Fixed journal/pad position: film, regime, pressure, forces
# ----------------------------------------------------------------------------
def tilt_angle_range(
    pad_index,
    total_n_reynolds,
    n_index_reynolds,
    pads,
    xj,
    yj,
    cb,
    cb_new,
    dh_n,
):
    """Admissible range of the pad tilt angle.

    The bounds correspond to the trailing edge and the leading edge just
    touching the journal (preload taken as zero, so the bound is conservative).
    :class:`PadTiltSearch` looks for the pad's moment balance inside it.

    Parameters
    ----------
    pad_index : int
        0-based pad index.
    total_n_reynolds : int
        Number of Reynolds nodes.
    n_index_reynolds : numpy.ndarray of int
        Reynolds node-number map.
    pads : PadGeometry
        Per-pad geometry; the arc length, leading-edge and pivot angles, the
        journal radius and the pad thickness set the bounds.
    xj, yj : float
        Journal-centre displacements, m.
    cb : float
        Bore clearance, m.
    cb_new : numpy.ndarray
        Per-pad bore clearance after pivot deformation, m.
    dh_n : numpy.ndarray
        Surface-deformation film perturbation at Reynolds nodes, m, shape
        ``(total_pads, dim_xz)``.

    Returns
    -------
    tuple of float
        ``(tilt_angle_max, tilt_angle_min)``.
    """
    n_index_reynolds = np.asarray(n_index_reynolds, dtype=np.int64)
    dh_n = np.asarray(dh_n, dtype=float)
    p = pad_index

    dh_n_min = 0.0
    for i in range(total_n_reynolds):
        node = int(n_index_reynolds[i])
        dh_n_min = min(dh_n_min, dh_n[p, node])

    radius = pads.journal_radius + cb + pads.pad_thickness
    tilt_angle_max = (
        cb_new[p]
        + dh_n_min
        - xj * np.cos(pads.leading_angle_rad[p] + pads.arc_length_rad[p])
        - yj * np.sin(pads.leading_angle_rad[p] + pads.arc_length_rad[p])
    ) / (radius * np.sin(pads.arc_length_rad[p] - pads.x_pivot_rad[p]))
    tilt_angle_min = (
        cb_new[p]
        + dh_n_min
        - xj * np.cos(pads.leading_angle_rad[p])
        - yj * np.sin(pads.leading_angle_rad[p])
    ) / (radius * np.sin(-pads.x_pivot_rad[p]))

    return tilt_angle_max, tilt_angle_min


class PadTiltSearch:
    """Root of one pad's pivot moment over its tilt angle.

    Finds the tilt ``delta`` at which the pivot moment ``M(delta)`` of a pad
    changes sign, inside the admissible range of :func:`tilt_angle_range`.
    The moment decreases with the tilt (the old bisection relied on it), so
    the root is bracketed by stepping from a start point towards it, then
    refined by Brent's method (inverse quadratic interpolation and secant
    steps with a bisection safeguard, after the ``brentq`` of SciPy [1]_).
    Each moment evaluation is one Reynolds solve of the pad.

    A pad whose film diverges from its leading edge on (``full_cavitate``,
    called *flat* here) carries the cavitation pressure only: its moment
    needs no Reynolds solve, and it jumps where the pad leaves that state
    (an unloaded pad: zero moment up to a tilt, a finite negative moment
    beyond). When the sign change lies between a flat and a non-flat tilt,
    the search first locates the edge of the flat range from the film
    thickness alone (no solve), then decides from the two tilts at the edge
    whether the sign changes at the edge (the answer is the tilt there with
    the non-negative moment, as for the bisection) or on either side of it
    (Brent's method continues there).

    The search is driven by the caller, which owns the film state:
    :meth:`propose` gives the next trial tilt; the caller builds the film
    there and calls :meth:`reject` when the film closes (``h_min`` at or
    below ``CLOSED_FILM * C_b``), then :meth:`skip` with the flat flag (True:
    no solve needed, propose again), and otherwise solves the pressure and
    calls :meth:`update` with the moment. The search ends on the last
    evaluated tilt, so the caller's film and pressure fields belong to the
    returned tilt.

    Parameters
    ----------
    lower, upper : float
        Admissible tilt range, rad.
    tilt_tolerance : float
        Width of the root bracket at which the search stops, rad.
    moment_tolerance : float
        Pivot moment, relative to the moment scale passed to :meth:`update`,
        at which the search stops (an exactly zero moment, or any moment of a
        flat film, does not stop it; a flat film's moment below it, and a
        round-off moment off a root bracket, count as zero).
    start : float, optional
        Start tilt, rad (the pad's tilt of the previous film solve). Default
        is None: the search starts at the middle of the range.
    slope : float, optional
        Estimate of ``dM/d(delta)`` from the previous solve, N*m/rad. With a
        start point it sets the first step: four times the secant distance
        ``|M / slope|`` towards the root. Default is None (a fixed fraction of
        the range).

    Attributes
    ----------
    iterations : int
        Moment evaluations so far.
    converged : bool
        The last evaluated tilt meets a tolerance, or is the end of the
        admissible range towards which the moment keeps its sign.
    residual : float
        ``min(|M| / (moment_tolerance * scale), width / tilt_tolerance)`` of
        the last evaluation: at most 1 when converged.
    moment_ratio : float
        ``|M| / scale`` of the last evaluation.
    width : float
        Width of the last root bracket, rad.
    slope : float or None
        Secant estimate of ``dM/d(delta)`` from the final bracket.

    References
    ----------
    .. [1] Brent, R. P. (1973). Algorithms for minimization without
           derivatives. Prentice-Hall, chapter 4.
    """

    #: Expansion factor of the start step and of each further step while the
    #: root is not bracketed.
    expand = 4.0
    #: First step of a start point without a slope estimate, as a fraction of
    #: the range.
    start_fraction = 1.0 / 64.0
    #: Moment, relative to the moment scale, below which it is round-off and
    #: counts as zero off a root bracket (64 machine epsilons; 70 times
    #: below the tightest moment tolerance used, ``1e-12``).
    roundoff = 64.0 * np.finfo(float).eps

    def __init__(
        self, lower, upper, tilt_tolerance, moment_tolerance, start=None, slope=None
    ):
        lower, upper = float(lower), float(upper)
        if upper < lower:
            lower, upper = upper, lower
        self._lo = lower
        self._hi = upper
        self._tol_x = float(tilt_tolerance)
        self._tol_m = float(moment_tolerance)
        self._cold = start is None or not np.isfinite(start)
        if self._cold:
            self._next = 0.5 * (lower + upper)
        else:
            self._next = min(max(float(start), lower), upper)
        self.slope = (
            float(slope)
            if slope is not None and np.isfinite(slope) and slope < 0.0
            else None
        )
        self._phase = "bracket"
        self._ref = None  # last admissible point of the bracketing phase
        self._dir = 0.0
        self._step = 0.0
        self._brent = None
        self._flip = None
        self._last = None
        self._plateau = False  # a round-off moment has been counted as zero
        self.iterations = 0
        self.converged = False
        self.at_bound = False
        self.residual = np.inf
        self.moment_ratio = np.inf
        self.width = upper - lower

    # -- caller protocol ---------------------------------------------------
    def propose(self):
        """Next trial tilt, rad."""
        return self._next

    def reject(self, delta):
        """Exclude a trial tilt at which the film closes.

        The film is closed when ``h_min <= CLOSED_FILM * C_b`` (see
        :data:`~ross.bearings.fluid_film.constants.CLOSED_FILM`).

        Parameters
        ----------
        delta : float
            The trial tilt, rad.

        Returns
        -------
        bool
            True when another trial is proposed; False when the admissible
            range has shrunk below the tilt tolerance (the caller keeps the
            closed film, as the bisection did).
        """
        if self._hi - self._lo < self._tol_x:
            return False
        if self._phase != "bracket":
            # Between two admissible tilts the film cannot close (h_min is
            # concave in the tilt); bisect towards the last point if it
            # happens anyway.
            last = self._last[0]
            if abs(delta - last) < self._tol_x:
                return False
            self._next = 0.5 * (delta + last)
            return True
        ref = self._ref
        if ref is None:
            # The bisection's rule: a positive tilt closes the trailing edge.
            if delta > 0.0:
                self._hi = delta
            else:
                self._lo = delta
            self._next = 0.5 * (self._lo + self._hi)
        elif delta > ref[0]:
            self._hi = delta
            self._next = 0.5 * (ref[0] + delta)
        else:
            self._lo = delta
            self._next = 0.5 * (ref[0] + delta)
        return True

    def skip(self, delta, flat):
        """Report whether the film at a trial tilt is flat; skip its solve.

        Parameters
        ----------
        delta : float
            The trial tilt, rad.
        flat : bool
            The film at ``delta`` diverges from the leading edge on
            (``full_cavitate``).

        Returns
        -------
        bool
            True when the trial only located the edge of the flat range and
            another trial is proposed (no pressure solve); False when the
            caller solves the pressure and calls :meth:`update`.
        """
        if self._phase != "flip":
            return False
        f = self._flip
        if bool(flat):
            f["flat_x"] = float(delta)
        else:
            f["solid_x"] = float(delta)
        if abs(f["solid_x"] - f["flat_x"]) < self._tol_x:
            self._phase = "flip_solid"
            self._next = f["solid_x"]
        else:
            self._next = 0.5 * (f["flat_x"] + f["solid_x"])
        return True

    def update(self, delta, moment, scale, flat=False):
        """Record the pivot moment at the last proposed tilt.

        Parameters
        ----------
        delta : float
            The evaluated tilt, rad.
        moment : float
            Pivot moment there, N*m.
        scale : float
            Moment scale of the pad, N*m (see :func:`moment_scale`).
        flat : bool, optional
            The film at ``delta`` is flat (see :meth:`skip`). Default is
            False.

        Returns
        -------
        bool
            True when the search has ended at ``delta``.
        """
        self.iterations += 1
        tol_m = self._tol_m * scale
        flat = bool(flat)
        moment = float(moment)
        # A film of uniform pressure (the cavitation pressure of a flat or
        # fully clamped film, the axial ambient profile of an unloaded pad
        # under high ambient pressure) has no moment about a central pivot,
        # computed as round-off of either sign whenever that pressure is not
        # zero, against a moment scale that holds the full ``|p| * area``.
        # Such a moment counts as exactly zero, so the answer does not hang
        # on the sign of the round-off: a flat film's moment below the
        # moment tolerance, and a moment below ``roundoff`` times the scale
        # while the root is not yet bracketed or once such a range has been
        # met. Inside a bracket between moments of genuine sign a moment that
        # small is a root and ends the search.
        if flat and abs(moment) <= tol_m:
            moment = 0.0
        elif (
            moment != 0.0
            and abs(moment) <= self.roundoff * scale
            and (self._plateau or self._phase != "brent")
        ):
            moment = 0.0
            self._plateau = True
        point = (float(delta), moment, flat)
        self._last = point
        self.moment_ratio = abs(point[1]) / scale if scale > 0.0 else np.inf
        if self._phase == "final":
            # Re-evaluation of the chosen end of a collapsed bracket.
            return self._finish(self.residual)
        # A flat film never meets the moment test, nor does a zero moment
        # (which counts as positive, as in the bisection): a pad whose film
        # carries a uniform pressure over a range of tilts settles at the
        # edge of that range, where the moment jumps. A tilt passing the test
        # there would end the search wherever it first lands in the range;
        # inside such a range the search ends on the bracket width.
        if not point[2] and point[1] != 0.0 and abs(point[1]) <= tol_m:
            return self._finish(abs(point[1]) / tol_m)
        self.residual = self.moment_ratio / self._tol_m if self._tol_m > 0.0 else np.inf
        if self._phase == "bracket":
            return self._bracket(point)
        if self._phase == "flip_solid":
            self._flip["solid"] = point
            self._phase = "flip_flat"
            self._next = self._flip["flat_x"]
            return False
        if self._phase == "flip_flat":
            return self._flip_decide(point)
        return self._brent_step(point)

    # -- phases --------------------------------------------------------------
    @staticmethod
    def _positive(moment):
        return moment >= 0.0

    def _finish(self, residual, width=None):
        self.converged = True
        self.residual = residual
        if width is not None:
            self.width = width
        return True

    def _start_root(self, a, b):
        """Refine the sign change between the evaluated points ``a``, ``b``."""
        if a[2] != b[2]:
            flat, solid = (a, b) if a[2] else (b, a)
            self._flip = {
                "flat_end": flat,
                "solid_end": solid,
                "flat_x": flat[0],
                "solid_x": solid[0],
            }
            self._phase = "flip"
            self.width = abs(solid[0] - flat[0])
            if self.width < self._tol_x:
                self._phase = "flip_solid"
                self._next = solid[0]
            else:
                self._next = 0.5 * (flat[0] + solid[0])
            return False
        self._phase = "brent"
        self._brent = {
            "xpre": a[0],
            "fpre": a[1],
            "xcur": b[0],
            "fcur": b[1],
            "xblk": 0.0,
            "fblk": 0.0,
            "spre": 0.0,
            "scur": 0.0,
        }
        return self._brent_next(self._brent)

    def _bracket(self, point):
        delta, moment = point[0], point[1]
        ref = self._ref
        if ref is not None and self._positive(moment) != self._positive(ref[1]):
            return self._start_root(ref, point)
        if ref is None:
            self._dir = 1.0 if self._positive(moment) else -1.0
            if self._cold:
                step = 0.125 * (self._hi - self._lo)
            elif self.slope is not None:
                step = abs(moment / self.slope)
            else:
                step = self.start_fraction * (self._hi - self._lo)
            self._step = self.expand * max(step, self._tol_x)
        else:
            self._step *= self.expand
        self._ref = point
        target = delta + self._dir * self._step
        target = min(max(target, self._lo), self._hi)
        if abs(target - delta) < self._tol_x:
            # The moment keeps its sign up to the end of the range: the end
            # is the answer, as the bisection converged to it.
            self.at_bound = True
            return self._finish(0.0, width=0.0)
        self._next = target
        return False

    def _flip_decide(self, flat_point):
        f = self._flip
        solid_point = f["solid"]
        if self._positive(flat_point[1]) != self._positive(f["flat_end"][1]):
            # The sign changes inside the flat range (pivot spring moment).
            return self._start_root(f["flat_end"], flat_point)
        if self._positive(solid_point[1]) == self._positive(flat_point[1]):
            # The sign changes beyond the edge, in the solid range.
            return self._start_root(f["solid_end"], solid_point)
        # The sign changes at the edge: keep the tilt with the non-negative
        # moment.
        width = abs(solid_point[0] - flat_point[0])
        residual = width / self._tol_x if self._tol_x > 0.0 else 0.0
        self.width = width
        if self._positive(flat_point[1]):
            return self._finish(residual, width=width)
        self._phase = "final"
        self.residual = residual
        self._next = solid_point[0]
        return False

    def _brent_step(self, point):
        b = self._brent
        b["xcur"] = point[0]
        b["fcur"] = point[1]
        return self._brent_next(b)

    def _brent_next(self, b):
        # One pass of SciPy's ``brentq`` loop (``Zeros/brentq.c``) after the
        # evaluation of ``xcur``, with a zero moment counted as positive and
        # no exit on an exact zero (see :meth:`update`).
        newest = self._last[0]
        if self._positive(b["fpre"]) != self._positive(b["fcur"]):
            b["xblk"] = b["xpre"]
            b["fblk"] = b["fpre"]
            b["spre"] = b["scur"] = b["xcur"] - b["xpre"]
        if abs(b["fblk"]) < abs(b["fcur"]):
            b["xpre"], b["xcur"], b["xblk"] = b["xcur"], b["xblk"], b["xcur"]
            b["fpre"], b["fcur"], b["fblk"] = b["fcur"], b["fblk"], b["fcur"]
        width = abs(b["xblk"] - b["xcur"])
        self.width = width
        if b["xblk"] != b["xcur"]:
            slope = (b["fblk"] - b["fcur"]) / (b["xblk"] - b["xcur"])
            if np.isfinite(slope) and slope < 0.0:
                self.slope = slope
        tol = self._tol_x + 4.0 * np.finfo(float).eps * abs(b["xcur"])
        delta_tol = 0.5 * tol
        sbis = 0.5 * (b["xblk"] - b["xcur"])
        if abs(sbis) < delta_tol:
            if b["xcur"] == newest:
                return self._finish(width / tol, width=width)
            # The bracket has collapsed but the newest tilt is not its end
            # with the smaller moment: evaluate that end once more.
            self._phase = "final"
            self.residual = width / tol
            self._next = b["xcur"]
            return False
        if abs(b["spre"]) > delta_tol and abs(b["fcur"]) < abs(b["fpre"]):
            if b["xpre"] == b["xblk"]:
                stry = -b["fcur"] * (b["xcur"] - b["xpre"]) / (b["fcur"] - b["fpre"])
            else:
                dpre = (b["fpre"] - b["fcur"]) / (b["xpre"] - b["xcur"])
                dblk = (b["fblk"] - b["fcur"]) / (b["xblk"] - b["xcur"])
                stry = (
                    -b["fcur"]
                    * (b["fblk"] * dblk - b["fpre"] * dpre)
                    / (dblk * dpre * (b["fblk"] - b["fpre"]))
                )
            if 2.0 * abs(stry) < min(abs(b["spre"]), 3.0 * abs(sbis) - delta_tol):
                b["spre"] = b["scur"]
                b["scur"] = stry
            else:
                b["spre"] = b["scur"] = sbis
        else:
            b["spre"] = b["scur"] = sbis
        b["xpre"] = b["xcur"]
        b["fpre"] = b["fcur"]
        if abs(b["scur"]) > delta_tol:
            self._next = b["xcur"] + b["scur"]
        else:
            self._next = b["xcur"] + (delta_tol if sbis > 0.0 else -delta_tol)
        self.residual = min(self.residual, width / tol)
        return False


def moment_scale(
    mesh,
    operating,
    pad_index,
    pads,
    nodal_pressure,
    pressback_n,
    k_rotate,
    tilt_angle,
    integrate_xz,
):
    """Scale of a pad's pivot moment, the reference of its tilt tolerance.

    The sum of the magnitudes of the terms of :func:`moment`: the film
    pressure and the back pressure, each as ``(R + t) * integral |p| dA``
    (the moment of the pressure force with the full pivot radius as lever),
    the inlet-groove pocket moment and the pivot spring moment
    ``|k_rotate * tilt|``.

    Parameters
    ----------
    mesh : ReynoldsMesh
        Film (x-z) mesh.
    operating : OperatingPoint
        Speed and pressure conditions of the case.
    pad_index : int
        0-based pad index.
    pads : PadGeometry
        Per-pad geometry.
    nodal_pressure, pressback_n : numpy.ndarray
        Film and back pressure at Reynolds nodes, Pa, shape
        ``(total_pads, dim_xz)``.
    k_rotate : numpy.ndarray
        Per-pad pivot rotational stiffness, N*m/rad.
    tilt_angle : numpy.ndarray
        Per-pad tilt angle, rad.
    integrate_xz : callable
        ``integrate_xz(pad_index, mesh, f)`` -> the surface integral of the
        nodal field ``f`` over the Reynolds mesh.

    Returns
    -------
    float
        Moment scale, N*m.
    """
    p = pad_index
    idx = np.asarray(mesh.n_index, dtype=np.int64)[: mesh.total_nodes]
    integrand = np.zeros(mesh.dim_xz, dtype=float)
    integrand[idx] = np.abs(np.asarray(nodal_pressure)[p, idx]) + np.abs(
        np.asarray(pressback_n)[p, idx]
    )
    lever = pads.journal_radius + pads.pad_thickness
    scale = lever * abs(integrate_xz(pad_index, mesh, integrand))
    if operating.bearing_type == "inlet_groove_tilting_pad":
        scale += abs(
            operating.press_supply
            * (pads.length_pocket * pads.axial_length_pocket)
            * pads.x_pivot[p]
        )
    return scale + abs(k_rotate[p] * tilt_angle[p])


def film_thickness(
    mesh,
    pad_index,
    operating,
    energy_mesh,
    total_e_y_dambl,
    total_e_y_damcore,
    weight_h,
    pads,
    cb,
    tilt_angle,
    preload_new,
    xj,
    yj,
    h_n,
    dh_n,
    dhdx_n,
    cp_new,
    unloaded,
):
    """Film thickness and the deformed y-coordinates of the energy/3-D meshes.

    Evaluates the nodal film thickness ``h_n`` (large in the pocket, small in
    the dam) including journal eccentricity, preload, pad tilt, surface
    deformation ``dh_n`` and any LE/TE taper; tracks the minimum film thickness
    and its location; flags full cavitation; and rebuilds the radial node
    coordinates of the energy mesh (``y_energy``) and, where in the film, the
    3-D mesh (``y_3d``).

    Parameters
    ----------
    mesh : ReynoldsMesh
        Film (x-z) mesh: coordinates, connectivity and index maps. Its
        ``y_3d`` is rebuilt in place.
    pad_index : int
        0-based pad index.
    operating : OperatingPoint
        Speed and pressure conditions of the case.
    energy_mesh : EnergyMesh
        Film+pad cross-section (x-y) mesh. Its ``y`` is rebuilt in place.
    total_e_y_dambl, total_e_y_damcore : numpy.ndarray of int
        Per-pad through-film element counts in the dam babbitt / core layers.
    weight_h : float
        Boundary-layer thickness fraction placing the cross-film nodes.
    pads : PadGeometry
        Per-pad geometry.
    cb : float
        Bore clearance, m.
    tilt_angle : numpy.ndarray
        Per-pad tilt angle, rad.
    preload_new : numpy.ndarray
        Per-pad preload after pivot deformation, dimensionless.
    xj, yj : float
        Journal-centre displacements, m.
    h_n, dh_n, dhdx_n : numpy.ndarray
        Film thickness (m), surface deformation (m) and the circumferential
        film-thickness derivative at Reynolds nodes, shape
        ``(total_pads, dim_xz)`` (``h_n`` and ``dhdx_n`` returned).
    cp_new : numpy.ndarray
        Per-pad machined clearance after pivot deformation, m.
    unloaded : numpy.ndarray of bool
        Per-pad unloaded flag.

    Returns
    -------
    dict
        ``h_n``, ``dhdx_n`` : numpy.ndarray, shape ``(total_pads, dim_xz)``
        ``y_energy`` : numpy.ndarray, shape ``(total_pads, dim_xy)``
        ``y_3d`` : numpy.ndarray, shape ``(total_pads, dim_3d)``
        ``h_min``, ``x_hmin`` : float
            Minimum film thickness (m) and its circumferential location (m)
            for this pad.
        ``full_cavitate`` : bool
            Whether the pad is fully cavitated.
    """
    match_nodes_xy = energy_mesh.match_nodes_xy
    n_index_energy = energy_mesh.n_index
    y_energy = energy_mesh.y
    axial_length = pads.axial_length
    axial_length_dam = pads.axial_length_dam
    axial_length_track = pads.axial_length_track
    length_track = pads.length_track
    pad_length = pads.pad_length
    n_index_reynolds = mesh.n_index
    x_reynolds = mesh.x
    x_reynolds_rad = mesh.x_rad
    y_3d = mesh.y_3d
    z_reynolds = mesh.z
    n_index_reynolds = np.ascontiguousarray(n_index_reynolds, dtype=np.int64)
    n_index_energy = np.ascontiguousarray(n_index_energy, dtype=np.int64)
    match_nodes_xy = np.ascontiguousarray(match_nodes_xy, dtype=np.int64)
    x_reynolds = np.ascontiguousarray(x_reynolds, dtype=np.float64)
    x_reynolds_rad = np.ascontiguousarray(x_reynolds_rad, dtype=np.float64)
    z_reynolds = np.ascontiguousarray(z_reynolds, dtype=np.float64)
    dh_n = np.ascontiguousarray(dh_n, dtype=np.float64)
    h_n = np.ascontiguousarray(h_n, dtype=np.float64)
    dhdx_n = np.ascontiguousarray(dhdx_n, dtype=np.float64)
    y_energy = np.ascontiguousarray(y_energy, dtype=np.float64)
    y_3d = np.ascontiguousarray(y_3d, dtype=np.float64)
    pad_length = np.ascontiguousarray(pad_length, dtype=np.float64)
    axial_length = np.ascontiguousarray(axial_length, dtype=np.float64)
    axial_length_track = np.ascontiguousarray(axial_length_track, dtype=np.float64)
    axial_length_dam = np.ascontiguousarray(axial_length_dam, dtype=np.float64)
    length_track = np.ascontiguousarray(length_track, dtype=np.float64)

    p = pad_index
    h_min, x_hmin, full_cavitate_int, h_ns = film_thickness_jit(
        p,
        int(mesh.total_e_z_film),
        float(pads.pad_thickness),
        float(pads.leading_angle_rad[p]),
        float(cp_new[p]),
        float(pad_length[p]),
        float(preload_new[p]),
        float(pads.x_pivot_rad[p]),
        float(pads.depth_track[p]),
        float(pads.journal_radius),
        float(cb),
        float(tilt_angle[p]),
        float(xj),
        float(yj),
        float(pads.dh_ramp_le[p]),
        float(pads.length_ramp_le[p]),
        float(pads.dh_ramp_te[p]),
        float(pads.length_ramp_te[p]),
        int(mesh.total_nodes),
        n_index_reynolds,
        x_reynolds,
        x_reynolds_rad,
        z_reynolds,
        pad_length,
        axial_length,
        axial_length_track,
        axial_length_dam,
        length_track,
        dh_n,
        h_n,
        dhdx_n,
        operating.operating_type,
        operating.bearing_type,
        bool(unloaded[p]),
        int(mesh.dim_xz),
    )
    full_cavitate = bool(full_cavitate_int)
    film_thickness_rebuild_jit(
        p,
        int(energy_mesh.total_e_y_pad),
        int(mesh.total_e_x_film),
        int(mesh.total_e_y_film),
        int(mesh.total_e_z_film),
        int(mesh.total_e_y_trackbl[p]),
        int(mesh.total_e_y_trackcore[p]),
        int(total_e_y_dambl[p]),
        int(total_e_y_damcore[p]),
        float(pads.depth_track[p]),
        float(pads.pad_thickness),
        float(weight_h),
        h_ns,
        n_index_energy,
        match_nodes_xy,
        y_energy,
        y_3d,
    )

    return {
        "h_n": h_n,
        "dhdx_n": dhdx_n,
        "y_energy": y_energy,
        "y_3d": y_3d,
        "h_min": h_min,
        "x_hmin": x_hmin,
        "full_cavitate": full_cavitate,
    }


def flow_regime(
    mesh,
    pad_index,
    flow_regime_track,
    flow_regime_dam,
    pads,
    lube_density,
    speed_surface,
    turbulence,
    vis_n_3d,
    h_n,
    scale_turb_track,
    scale_turb_dam,
    vis_n_average,
    re_max,
):
    """Flow regime and turbulence scaling factor for one pad.

    Computes the radially averaged viscosity and the local Reynolds number at
    each node, the maximum Reynolds number in the track and dam regions, and
    the per-region flow regime (0 laminar, 1 transition, 2 turbulent) plus the
    turbulence scaling factor.

    Parameters
    ----------
    mesh : ReynoldsMesh
        Film (x-z) mesh: coordinates, connectivity and index maps, plus the
        radial coordinate ``y_3d`` of the 3-D film nodes.
    pad_index : int
        0-based pad index.
    flow_regime_track, flow_regime_dam : numpy.ndarray of int
        Per-pad flow-regime flag, 0 laminar / 1 transition / 2 turbulent
        (updated copies returned).
    pads : PadGeometry
        Per-pad geometry.
    lube_density : float
        Lubricant density, kg/m^3.
    speed_surface : float
        Journal surface speed, m/s.
    turbulence : Turbulence
        Turbulence-model constants.
    vis_n_3d : numpy.ndarray
        Molecular viscosity on the 3-D mesh, Pa*s, shape
        ``(total_pads, dim_3d)``.
    h_n : numpy.ndarray
        Film thickness, m, shape ``(total_pads, dim_xz)``.
    scale_turb_track, scale_turb_dam : numpy.ndarray
        Per-pad turbulence scaling factor (updated copies returned).
    vis_n_average : numpy.ndarray
        Radially averaged nodal viscosity, Pa*s, shape
        ``(total_pads, dim_xz)`` (updated copy returned).
    re_max : numpy.ndarray
        Per-pad maximum Reynolds number (updated copy returned).

    Returns
    -------
    dict
        ``flow_regime_track``, ``flow_regime_dam``, ``scale_turb_track``,
        ``scale_turb_dam``, ``re_max`` : numpy.ndarray (0-based natural)
        ``vis_n_average`` : numpy.ndarray, shape ``(total_pads, dim_xz)``
    """
    axial_length = pads.axial_length
    axial_length_dam = pads.axial_length_dam
    axial_length_track = pads.axial_length_track
    depth_track = pads.depth_track
    length_track = pads.length_track
    pad_length = pads.pad_length
    match_nodes_xz = mesh.match_nodes_xz
    n_index_reynolds = mesh.n_index
    x_reynolds = mesh.x
    y_3d = mesh.y_3d
    z_reynolds = mesh.z
    n_index_reynolds = np.asarray(n_index_reynolds, dtype=np.int64)
    match_nodes_xz = np.asarray(match_nodes_xz, dtype=np.int64)
    x_reynolds = np.asarray(x_reynolds, dtype=float)
    z_reynolds = np.asarray(z_reynolds, dtype=float)
    h_n = np.asarray(h_n, dtype=float)
    vis_n_3d = np.asarray(vis_n_3d, dtype=float)
    y_3d = np.asarray(y_3d, dtype=float)
    flow_regime_track = np.ascontiguousarray(flow_regime_track, dtype=np.int64)
    flow_regime_dam = np.ascontiguousarray(flow_regime_dam, dtype=np.int64)
    scale_turb_track = np.ascontiguousarray(scale_turb_track, dtype=np.float64)
    scale_turb_dam = np.ascontiguousarray(scale_turb_dam, dtype=np.float64)
    re_max = np.ascontiguousarray(re_max, dtype=np.float64)
    vis_n_average = np.ascontiguousarray(vis_n_average, dtype=np.float64)

    p = pad_index
    pad_length = np.ascontiguousarray(pad_length, dtype=np.float64)
    axial_length = np.ascontiguousarray(axial_length, dtype=np.float64)
    axial_length_track = np.ascontiguousarray(axial_length_track, dtype=np.float64)
    axial_length_dam = np.ascontiguousarray(axial_length_dam, dtype=np.float64)
    length_track = np.ascontiguousarray(length_track, dtype=np.float64)
    depth_track = np.ascontiguousarray(depth_track, dtype=np.float64)
    (
        ft_p,
        st_p,
        fd_p,
        sd_p,
        rem_p,
    ) = flow_regime_jit(
        p,
        mesh.dim_yf,
        int(mesh.total_e_y_trackbl[p]),
        int(mesh.total_e_y_trackcore[p]),
        mesh.total_e_y_film,
        mesh.total_nodes,
        n_index_reynolds,
        match_nodes_xz,
        pad_length,
        axial_length,
        axial_length_track,
        axial_length_dam,
        length_track,
        depth_track,
        float(pads.pad_thickness),
        x_reynolds,
        z_reynolds,
        float(lube_density),
        float(speed_surface),
        float(turbulence.re_lower),
        float(turbulence.re_upper),
        vis_n_3d,
        y_3d,
        h_n,
        vis_n_average,
        float(turbulence.scale_factor_exponent),
    )
    flow_regime_track[p] = ft_p
    scale_turb_track[p] = st_p
    flow_regime_dam[p] = fd_p
    scale_turb_dam[p] = sd_p
    re_max[p] = rem_p

    return {
        "flow_regime_track": flow_regime_track,
        "flow_regime_dam": flow_regime_dam,
        "scale_turb_track": scale_turb_track,
        "scale_turb_dam": scale_turb_dam,
        "re_max": re_max,
        "vis_n_average": vis_n_average,
    }


def effective_viscosity(
    pad_index,
    mesh,
    total_e_y_dambl,
    total_e_y_damcore,
    pads,
    vis_n_3d,
    vis_n_average,
    shear_stress,
    h_n,
    lube_density,
    scale_turb_track,
    scale_turb_dam,
    vis_eddy_3d,
    vis_effect_3d,
    turbulence,
):
    """Effective viscosity including the turbulent eddy contribution.

    The eddy viscosity follows Reichardt's formula in the dimensionless wall
    distance ``y_plus``, evaluated for the lower / upper half of the channel;
    the effective viscosity is the molecular viscosity scaled by ``(1 +
    eddy)``.

    Parameters
    ----------
    pad_index : int
        0-based pad index.
    mesh : ReynoldsMesh
        Film (x-z) mesh: coordinates, connectivity and index maps, plus the
        radial coordinate ``y_3d`` of the 3-D film nodes.
    total_e_y_dambl, total_e_y_damcore : numpy.ndarray of int
        Per-pad through-film element counts in the dam babbitt / core layers.
    pads : PadGeometry
        Per-pad geometry.
    vis_n_3d : numpy.ndarray
        Molecular viscosity on the 3-D mesh, Pa*s, shape
        ``(total_pads, dim_3d)``.
    vis_n_average : numpy.ndarray
        Radially averaged nodal viscosity, Pa*s, shape
        ``(total_pads, dim_xz)``.
    shear_stress : numpy.ndarray
        Shear stress on the 3-D mesh, Pa, shape ``(total_pads, dim_3d)``.
    h_n : numpy.ndarray
        Film thickness, m, shape ``(total_pads, dim_xz)``.
    lube_density : float
        Lubricant density, kg/m^3.
    scale_turb_track, scale_turb_dam : numpy.ndarray
        Per-pad turbulence scaling factor.
    vis_eddy_3d, vis_effect_3d : numpy.ndarray
        Eddy-viscosity ratio (dimensionless) and the resulting effective
        viscosity (Pa*s) on the 3-D mesh, shape ``(total_pads, dim_3d)``
        (updated copies returned).
    turbulence : Turbulence
        Turbulence-model constants.

    Returns
    -------
    dict
        ``vis_eddy_3d``, ``vis_effect_3d`` : numpy.ndarray, shape
        ``(total_pads, dim_3d)``
        ``y_plus_max`` : float
            Maximum dimensionless wall distance encountered on this pad.
    """
    axial_length = pads.axial_length
    axial_length_dam = pads.axial_length_dam
    axial_length_track = pads.axial_length_track
    depth_track = pads.depth_track
    length_track = pads.length_track
    pad_length = pads.pad_length
    match_nodes_xz = mesh.match_nodes_xz
    n_index_reynolds = mesh.n_index
    x_reynolds = mesh.x
    y_3d = mesh.y_3d
    z_reynolds = mesh.z
    n_index_reynolds = np.ascontiguousarray(n_index_reynolds, dtype=np.int64)
    match_nodes_xz = np.ascontiguousarray(match_nodes_xz, dtype=np.int64)
    x_reynolds = np.ascontiguousarray(x_reynolds, dtype=np.float64)
    z_reynolds = np.ascontiguousarray(z_reynolds, dtype=np.float64)
    h_n = np.ascontiguousarray(h_n, dtype=np.float64)
    vis_n_3d = np.ascontiguousarray(vis_n_3d, dtype=np.float64)
    shear_stress = np.ascontiguousarray(shear_stress, dtype=np.float64)
    y_3d = np.ascontiguousarray(y_3d, dtype=np.float64)
    vis_n_average = np.ascontiguousarray(vis_n_average, dtype=np.float64)
    pad_length = np.ascontiguousarray(pad_length, dtype=np.float64)
    axial_length = np.ascontiguousarray(axial_length, dtype=np.float64)
    length_track = np.ascontiguousarray(length_track, dtype=np.float64)
    axial_length_dam = np.ascontiguousarray(axial_length_dam, dtype=np.float64)
    axial_length_track = np.ascontiguousarray(axial_length_track, dtype=np.float64)
    depth_track = np.ascontiguousarray(depth_track, dtype=np.float64)
    vis_eddy_3d = np.ascontiguousarray(vis_eddy_3d, dtype=np.float64)
    vis_effect_3d = np.ascontiguousarray(vis_effect_3d, dtype=np.float64)

    p = pad_index
    y_plus_max = effective_viscosity_jit(
        p,
        mesh.total_e_y_film,
        int(total_e_y_dambl[p]),
        int(total_e_y_damcore[p]),
        int(mesh.total_e_y_trackbl[p]),
        int(mesh.total_e_y_trackcore[p]),
        mesh.total_nodes,
        n_index_reynolds,
        match_nodes_xz,
        pad_length,
        axial_length,
        length_track,
        axial_length_track,
        axial_length_dam,
        depth_track,
        float(pads.pad_thickness),
        x_reynolds,
        z_reynolds,
        vis_n_3d,
        vis_n_average,
        shear_stress,
        y_3d,
        h_n,
        float(lube_density),
        float(scale_turb_track[p]),
        float(scale_turb_dam[p]),
        vis_eddy_3d,
        vis_effect_3d,
        float(turbulence.reichardt_delta),
        float(turbulence.reichardt_kappa),
    )

    return {
        "vis_eddy_3d": vis_eddy_3d,
        "vis_effect_3d": vis_effect_3d,
        "y_plus_max": y_plus_max,
    }


def dudy_dwdy(
    mesh,
    pad_index,
    pads,
    speed_surface,
    h_n,
    dpdx_n,
    dpdz_n,
    vis_effect_3d,
    dudy_n,
    dwdy_n,
):
    """Velocity gradients ``dU/dy`` and ``dW/dy`` across the film.

    Used to evaluate the shear stress and the heat dissipation. Within the
    pocket the integral runs from the pad surface (limit ``1``), in the dam
    region from the dam floor (limit ``limit1``).

    Parameters
    ----------
    mesh : ReynoldsMesh
        Film (x-z) mesh: coordinates, connectivity and index maps, plus the
        radial coordinate ``y_3d`` of the 3-D film nodes.
    pad_index : int
        0-based pad index.
    pads : PadGeometry
        Per-pad geometry.
    speed_surface : float
        Journal surface speed, m/s.
    h_n : numpy.ndarray
        Film thickness, m, shape ``(total_pads, dim_xz)``.
    dpdx_n, dpdz_n : numpy.ndarray
        Pressure gradients at Reynolds nodes, Pa/m, shape
        ``(total_pads, dim_xz)``.
    vis_effect_3d : numpy.ndarray
        Effective viscosity on the 3-D mesh, Pa*s, shape
        ``(total_pads, dim_3d)``.
    dudy_n, dwdy_n : numpy.ndarray
        Velocity gradients on the 3-D mesh, 1/s, shape
        ``(total_pads, dim_3d)`` (updated copies returned).

    Returns
    -------
    tuple of numpy.ndarray
        ``(dudy_n, dwdy_n)``.
    """
    axial_length = pads.axial_length
    axial_length_dam = pads.axial_length_dam
    axial_length_track = pads.axial_length_track
    depth_track = pads.depth_track
    length_track = pads.length_track
    pad_length = pads.pad_length
    match_nodes_xz = mesh.match_nodes_xz
    n_index_reynolds = mesh.n_index
    x_reynolds = mesh.x
    y_3d = mesh.y_3d
    z_reynolds = mesh.z
    n_index_reynolds = np.ascontiguousarray(n_index_reynolds, dtype=np.int64)
    match_nodes_xz = np.ascontiguousarray(match_nodes_xz, dtype=np.int64)
    x_reynolds = np.ascontiguousarray(x_reynolds, dtype=np.float64)
    z_reynolds = np.ascontiguousarray(z_reynolds, dtype=np.float64)
    h_n = np.ascontiguousarray(h_n, dtype=np.float64)
    dpdx_n = np.ascontiguousarray(dpdx_n, dtype=np.float64)
    dpdz_n = np.ascontiguousarray(dpdz_n, dtype=np.float64)
    vis_effect_3d = np.ascontiguousarray(vis_effect_3d, dtype=np.float64)
    y_3d = np.ascontiguousarray(y_3d, dtype=np.float64)
    pad_length = np.ascontiguousarray(pad_length, dtype=np.float64)
    axial_length = np.ascontiguousarray(axial_length, dtype=np.float64)
    length_track = np.ascontiguousarray(length_track, dtype=np.float64)
    depth_track = np.ascontiguousarray(depth_track, dtype=np.float64)
    axial_length_dam = np.ascontiguousarray(axial_length_dam, dtype=np.float64)
    axial_length_track = np.ascontiguousarray(axial_length_track, dtype=np.float64)
    dudy_n = np.ascontiguousarray(dudy_n, dtype=np.float64)
    dwdy_n = np.ascontiguousarray(dwdy_n, dtype=np.float64)

    p = pad_index
    return dudy_dwdy_jit(
        p,
        mesh.dim_yf,
        mesh.total_e_y_film,
        int(mesh.total_e_y_trackbl[p]),
        int(mesh.total_e_y_trackcore[p]),
        n_index_reynolds,
        match_nodes_xz,
        mesh.total_nodes,
        float(pads.pad_thickness),
        float(speed_surface),
        pad_length,
        axial_length,
        length_track,
        depth_track,
        axial_length_dam,
        axial_length_track,
        x_reynolds,
        z_reynolds,
        dpdx_n,
        dpdz_n,
        vis_effect_3d,
        y_3d,
        dudy_n,
        dwdy_n,
    )


def update_shear(
    pad_index,
    mesh,
    dudy_n,
    dwdy_n,
    vis_effect_3d,
    shear_stress,
    relaxp,
):
    """Relax the shear-stress field toward the latest pressure solution.

    The new local shear is ``vis_effect * sqrt(dUdy^2 + dWdy^2)``; the stored
    field is relaxed by ``relaxp``. Returns the normalized RMS change used for
    the convergence test.

    Parameters
    ----------
    pad_index : int
        0-based pad index.
    mesh : ReynoldsMesh
        Film (x-z) mesh: coordinates, connectivity and index maps.
    dudy_n, dwdy_n : numpy.ndarray
        Velocity gradients on the 3-D mesh, 1/s, shape
        ``(total_pads, dim_3d)``.
    vis_effect_3d : numpy.ndarray
        Effective viscosity on the 3-D mesh, Pa*s, shape
        ``(total_pads, dim_3d)``.
    shear_stress : numpy.ndarray
        Shear stress on the 3-D mesh, Pa, shape ``(total_pads, dim_3d)``
        (updated copy returned).
    relaxp : float
        Relaxation factor (``1.0`` for the laminar, no-iteration path).

    Returns
    -------
    tuple
        ``(shear_stress, rms_shear)`` -- updated field and normalized RMS
        change.
    """
    match_nodes_xz = mesh.match_nodes_xz
    n_index_reynolds = mesh.n_index
    n_index_reynolds = np.ascontiguousarray(n_index_reynolds, dtype=np.int64)
    match_nodes_xz = np.ascontiguousarray(match_nodes_xz, dtype=np.int64)
    dudy_n = np.ascontiguousarray(dudy_n, dtype=np.float64)
    dwdy_n = np.ascontiguousarray(dwdy_n, dtype=np.float64)
    vis_effect_3d = np.ascontiguousarray(vis_effect_3d, dtype=np.float64)
    shear_stress = np.ascontiguousarray(shear_stress, dtype=np.float64)

    return update_shear_jit(
        pad_index,
        mesh.total_nodes,
        n_index_reynolds,
        mesh.total_e_y_film,
        match_nodes_xz,
        dudy_n,
        dwdy_n,
        vis_effect_3d,
        shear_stress,
        float(relaxp),
    )


def press_special_flow(
    pad_index,
    total_n_reynolds,
    n_index_reynolds,
    axial_length,
    operating,
    z_reynolds,
    pressback_n,
):
    """Back-of-pad pressure under axial flow / high ambient pressure.

    For tilting pads the back pressure varies linearly between the two ambient
    pressures along the axial coordinate; for fixed geometry it is zero.

    Parameters
    ----------
    pad_index : int
        0-based pad index.
    total_n_reynolds : int
        Number of Reynolds nodes.
    n_index_reynolds : numpy.ndarray of int
        Reynolds node-number map.
    axial_length : numpy.ndarray
        Per-pad axial length, m.
    operating : OperatingPoint
        Speed and pressure conditions of the case; supplies the bearing type
        and the two ambient pressures.
    z_reynolds : numpy.ndarray
        Axial nodal coordinate, m, shape ``(total_pads, dim_xz)``.
    pressback_n : numpy.ndarray
        Back pressure at Reynolds nodes, Pa, shape ``(total_pads, dim_xz)``
        (updated copy returned).

    Returns
    -------
    numpy.ndarray
        Updated ``pressback_n``.
    """
    n_index_reynolds = np.asarray(n_index_reynolds, dtype=np.int64)
    z_reynolds = np.asarray(z_reynolds, dtype=float)
    pressback_n = np.ascontiguousarray(pressback_n, dtype=np.float64)
    p = pad_index

    if operating.bearing_type in (
        "conventional_tilting_pad",
        "inlet_groove_tilting_pad",
        "spray_bar_tilting_pad",
    ):
        for i in range(total_n_reynolds):
            node = int(n_index_reynolds[i])
            pressback_n[p, node] = (
                z_reynolds[p, node]
                * (operating.ambient_press2 - operating.ambient_press1)
            ) / axial_length[p] + operating.ambient_press1
    else:
        for i in range(total_n_reynolds):
            node = int(n_index_reynolds[i])
            pressback_n[p, node] = 0.0
    return pressback_n


def film_moment(mesh, pad_index, pads, nodal_pressure, integrate_xz):
    """Moment of the film pressure about the pad pivot.

    Parameters
    ----------
    mesh : ReynoldsMesh
        Film (x-z) mesh: coordinates, connectivity and index maps.
    pad_index : int
        0-based pad index.
    pads : PadGeometry
        Per-pad geometry (pivot angle, journal radius, pad thickness).
    nodal_pressure : numpy.ndarray
        Film pressure at Reynolds nodes, Pa, shape ``(total_pads, dim_xz)``.
    integrate_xz : callable
        ``integrate_xz(pad_index, mesh, f)`` -> the surface integral of the
        nodal field ``f`` over the Reynolds mesh.

    Returns
    -------
    float
        ``-(R + t) * integral of p sin(theta - theta_pivot)``, N*m, with the
        lever arm ``R + t`` from the journal centre to the pivot.
    """
    n_index_reynolds = np.asarray(mesh.n_index, dtype=np.int64)
    nodal_pressure = np.asarray(nodal_pressure, dtype=float)
    x_reynolds_rad = np.asarray(mesh.x_rad, dtype=float)
    p = pad_index
    idx = n_index_reynolds[: mesh.total_nodes]
    integrand = np.zeros(mesh.dim_xz, dtype=float)
    integrand[idx] = nodal_pressure[p, idx] * np.sin(
        x_reynolds_rad[p, idx] - pads.x_pivot_rad[p]
    )
    inte_f = integrate_xz(pad_index, mesh, integrand)
    return -(pads.journal_radius + pads.pad_thickness) * inte_f


def moment(
    mesh,
    operating,
    pad_index,
    pads,
    nodal_pressure,
    pressback_n,
    k_rotate,
    tilt_angle,
    integrate_xz,
):
    """Net moment about the pad pivot for a tilting pad.

    Integrates the film pressure (and, under special flow, the back pressure)
    into a moment about the pivot, adds the LEG-bearing pocket moment and the
    pivot rotational spring term.

    Parameters
    ----------
    mesh : ReynoldsMesh
        Film (x-z) mesh: coordinates, connectivity and index maps.
    operating : OperatingPoint
        Speed and pressure conditions of the case.
    pad_index : int
        0-based pad index.
    pads : PadGeometry
        Per-pad geometry, including the pivot position and the inlet-groove
        pocket dimensions.
    nodal_pressure : numpy.ndarray
        Film pressure at Reynolds nodes, Pa, shape ``(total_pads, dim_xz)``.
    pressback_n : numpy.ndarray
        Back pressure at Reynolds nodes, Pa, shape ``(total_pads, dim_xz)``
        (updated copy returned).
    k_rotate : numpy.ndarray
        Per-pad pivot rotational stiffness, N*m/rad.
    tilt_angle : numpy.ndarray
        Per-pad tilt angle, rad.
    integrate_xz : callable
        ``integrate_xz(pad_index, mesh, f)`` -> the surface integral of the
        nodal field ``f`` over the Reynolds mesh.

    Returns
    -------
    tuple
        ``(moment_pivot, pressback_n)``, the pivot moment in N*m and the
        updated back-pressure field.
    """
    n_index_reynolds = np.asarray(mesh.n_index, dtype=np.int64)
    x_reynolds_rad = np.asarray(mesh.x_rad, dtype=float)
    pressback_n = np.ascontiguousarray(pressback_n, dtype=np.float64)
    p = pad_index

    idx = n_index_reynolds[: mesh.total_nodes]
    integrand = np.zeros(mesh.dim_xz, dtype=float)
    moment_pivot = film_moment(mesh, pad_index, pads, nodal_pressure, integrate_xz)

    if operating.bearing_type == "inlet_groove_tilting_pad":
        moment_pocket = (
            operating.press_supply
            * (pads.length_pocket * pads.axial_length_pocket)
            * pads.x_pivot[p]
        )
    else:
        moment_pocket = 0.0

    if operating.operating_type in ("axial_flow", "high_ambient_pressure"):
        pressback_n = press_special_flow(
            pad_index,
            mesh.total_nodes,
            n_index_reynolds,
            pads.axial_length,
            operating,
            mesh.z,
            pressback_n,
        )
        integrand[idx] = pressback_n[p, idx] * np.sin(
            x_reynolds_rad[p, idx] - pads.x_pivot_rad[p]
        )
        inte_f = integrate_xz(
            pad_index,
            mesh,
            integrand,
        )
        moment_back = (pads.journal_radius + pads.pad_thickness) * inte_f
    else:
        moment_back = 0.0
        pressback_n[p, idx] = 0.0

    moment_pivot = moment_pivot + moment_pocket + moment_back
    moment_pivot = moment_pivot + (-k_rotate[p] * tilt_angle[p])

    return moment_pivot, pressback_n


def loading_condition(
    mesh,
    pad_index,
    nodal_pressure,
    pressback_n,
    x_pivot_rad,
    unloaded,
    integrate_xz,
):
    """Determine whether the pad is unloaded (negative net pivot force).

    Integrates the film and back pressures into the radial force through the
    pivot; a negative net force flags the pad as unloaded.

    Parameters
    ----------
    mesh : ReynoldsMesh
        Film (x-z) mesh: coordinates, connectivity and index maps.
    pad_index : int
        0-based pad index.
    nodal_pressure, pressback_n : numpy.ndarray
        Film and back pressure at Reynolds nodes, Pa, shape
        ``(total_pads, dim_xz)``.
    x_pivot_rad : numpy.ndarray
        Per-pad pivot angle, rad.
    unloaded : numpy.ndarray of bool
        Per-pad unloaded flag (updated copy returned).
    integrate_xz : callable
        ``integrate_xz(pad_index, mesh, f)`` -> the surface integral of the
        nodal field ``f`` over the Reynolds mesh.

    Returns
    -------
    numpy.ndarray of bool
        Updated ``unloaded``.
    """
    n_index_reynolds = mesh.n_index
    x_reynolds_rad = mesh.x_rad
    n_index_reynolds = np.asarray(n_index_reynolds, dtype=np.int64)
    nodal_pressure = np.asarray(nodal_pressure, dtype=float)
    pressback_n = np.asarray(pressback_n, dtype=float)
    x_reynolds_rad = np.asarray(x_reynolds_rad, dtype=float)
    unloaded = np.array(unloaded, dtype=bool)
    p = pad_index

    integrand = np.zeros(mesh.dim_xz, dtype=float)
    for i in range(mesh.total_nodes):
        node = int(n_index_reynolds[i])
        integrand[node] = nodal_pressure[p, node] * np.cos(
            x_reynolds_rad[p, node] - x_pivot_rad[p]
        )
    inte_f_babbitt = integrate_xz(
        pad_index,
        mesh,
        integrand,
    )

    for i in range(mesh.total_nodes):
        node = int(n_index_reynolds[i])
        integrand[node] = pressback_n[p, node] * np.cos(
            x_reynolds_rad[p, node] - x_pivot_rad[p]
        )
    inte_f_back = integrate_xz(
        pad_index,
        mesh,
        integrand,
    )

    frp = inte_f_babbitt - inte_f_back
    if frp < 0.0:
        unloaded[p] = True
    return unloaded


# ----------------------------------------------------------------------------
# Drivers
# ----------------------------------------------------------------------------
def fixed_brg(
    total_pads,
    mesh,
    operating,
    flow_regime_track,
    flow_regime_dam,
    pads,
    cb,
    tilt_angle,
    energy_mesh,
    total_e_y_dambl,
    total_e_y_damcore,
    weight_h,
    film_onset,
    xj,
    yj,
    h_min,
    x_hmin,
    h_n,
    dh_n,
    dhdx_n,
    lube_density,
    vis_n_3d,
    vis_n_average,
    shear_stress,
    scale_turb_track,
    scale_turb_dam,
    vis_eddy_3d,
    vis_effect_3d,
    nodal_pressure,
    dpdx_n,
    dpdz_n,
    dudy_n,
    dwdy_n,
    relaxp,
    equilpost_index,
    turbulence,
    unloaded,
    pressback_n,
    deform_pivot,
    re_max,
    k_rotate,
    press=None,
    integrate_xz=None,
    monitor=None,
    tolerance_scale=1.0,
    held_tilts=None,
    cavitated=None,
    tilt_memory=None,
):
    """Solve the bearing film at a *fixed* journal position (isoviscous).

    For each pad it finds the tilt angle at which the pivot moment vanishes
    (:class:`PadTiltSearch`; zero range for fixed geometry and pressure
    dams), and for each trial angle:
    builds the film thickness (:func:`film_thickness`), the flow regime
    (:func:`flow_regime`) and effective viscosity (:func:`effective_viscosity`),
    solves the pressure (injected ``press``), computes velocity gradients
    (:func:`dudy_dwdy`), iterates pressure/shear for super-laminar flow
    (:func:`update_shear`), and evaluates the pivot moment (:func:`moment`).
    For tilting pads under axial flow / high ambient pressure it also checks the
    loading condition (:func:`loading_condition`) and, if unloaded, recomputes
    everything at the constant clearance ``Cp``.

    All array arguments follow this module's 0-based convention.

    Parameters
    ----------
    total_pads : int
        Number of pads.
    mesh : ReynoldsMesh
        Film (x-z) mesh: coordinates, connectivity and index maps.
    operating : OperatingPoint
        Speed and pressure conditions of the case.
    flow_regime_track, flow_regime_dam : numpy.ndarray of int
        Per-pad flow-regime flag, 0 laminar / 1 transition / 2 turbulent
        (updated copies returned).
    pads : PadGeometry
        Per-pad geometry.
    cb : float
        Bore clearance, m.
    tilt_angle : numpy.ndarray
        Per-pad tilt angle, rad (updated copy returned).
    energy_mesh : EnergyMesh
        Film+pad cross-section (x-y) mesh.
    total_e_y_dambl, total_e_y_damcore : numpy.ndarray of int
        Per-pad through-film element counts in the dam babbitt / core layers.
    weight_h : float
        Boundary-layer thickness fraction placing the cross-film nodes.
    film_onset : numpy.ndarray of int
        Per-pad film-onset element index.
    xj, yj : float
        Journal-centre displacements, m.
    h_min : numpy.ndarray
        Per-pad minimum film thickness, m (updated copy returned).
    x_hmin : numpy.ndarray
        Per-pad circumferential location of the minimum film, m (updated copy
        returned).
    h_n, dh_n, dhdx_n : numpy.ndarray
        Film thickness (m), surface deformation (m) and the circumferential
        film-thickness derivative at Reynolds nodes, shape
        ``(total_pads, dim_xz)`` (``h_n`` and ``dhdx_n`` returned).
    lube_density : float
        Lubricant density, kg/m^3.
    vis_n_3d : numpy.ndarray
        Molecular viscosity on the 3-D mesh, Pa*s, shape
        ``(total_pads, dim_3d)``.
    vis_n_average : numpy.ndarray
        Radially averaged nodal viscosity, Pa*s, shape
        ``(total_pads, dim_xz)`` (updated copy returned).
    shear_stress : numpy.ndarray
        Shear stress on the 3-D mesh, Pa, shape ``(total_pads, dim_3d)``
        (updated copy returned).
    scale_turb_track, scale_turb_dam : numpy.ndarray
        Per-pad turbulence scaling factor (updated copies returned).
    vis_eddy_3d, vis_effect_3d : numpy.ndarray
        Eddy-viscosity ratio (dimensionless) and the resulting effective
        viscosity (Pa*s) on the 3-D mesh, shape ``(total_pads, dim_3d)``
        (updated copies returned).
    nodal_pressure : numpy.ndarray
        Film pressure at Reynolds nodes, Pa, shape ``(total_pads, dim_xz)``
        (updated copy returned).
    dpdx_n, dpdz_n : numpy.ndarray
        Pressure gradients at Reynolds nodes, Pa/m, shape
        ``(total_pads, dim_xz)`` (updated copies returned).
    dudy_n, dwdy_n : numpy.ndarray
        Velocity gradients on the 3-D mesh, 1/s, shape
        ``(total_pads, dim_3d)`` (updated copies returned).
    relaxp : float
        Relaxation factor for the pressure/shear iteration.
    equilpost_index : int
        Equilibrium-iteration counter, 1-based; on the first iteration the
        per-pad unloaded flags are reset.
    turbulence : Turbulence
        Turbulence-model constants.
    unloaded : numpy.ndarray of bool
        Per-pad unloaded flag (updated copy returned).
    pressback_n : numpy.ndarray
        Back pressure at Reynolds nodes, Pa, shape ``(total_pads, dim_xz)``
        (updated copy returned).
    deform_pivot : numpy.ndarray
        Per-pad pivot deformation added to the bore clearance, m.
    re_max : numpy.ndarray
        Per-pad maximum Reynolds number (updated copy returned).
    k_rotate : numpy.ndarray
        Per-pad pivot rotational stiffness, N*m/rad.
    press : callable, optional
        The Reynolds pressure solver (``ross.bearings.fluid_film.pressure.press``).
        Imported lazily if not supplied.
    integrate_xz : callable
        ``integrate_xz(pad_index, mesh, f)`` -> the surface integral of the
        nodal field ``f`` over the Reynolds mesh. Required by the moment and
        loading-condition routines.
    monitor : LoopMonitor, optional
        Receives the ``tilt`` and ``turbulence`` records of this call (worst
        pad). Default is None (nothing recorded).
    tolerance_scale : float, optional
        Factor on the tilt (``TILT_ERROR`` rad, ``TILT_MOMENT_ERROR`` of the
        moment scale) and shear-stress tolerances. Default is 1.0.
    held_tilts : array_like of float, optional
        Tilt angle of each pad, rad, held instead of solving the pad moment
        balance (one film solve per pad at that tilt). Ignored for fixed
        geometry. Default is None (tilts solved).
    cavitated : array_like of bool, optional
        Cavitated-node mask of the previous solve, shape
        ``(total_pads, dim_xz)``; each pad's row is replaced by its new solve.
        Default is None (all False).
    tilt_memory : dict, optional
        The ``_tilt_memory`` of a previous call: ``known`` (bool per pad, the
        pad's ``tilt_angle`` is a solved tilt) and ``slope`` (moment slope
        ``dM/d(delta)`` per pad, N*m/rad, NaN if unknown). A known pad starts
        its search at its current tilt with a step from the slope. Default
        is None (every search starts at the middle of the admissible range).

    Returns
    -------
    dict
        The fields mutated by the pressure / film / regime / viscosity / shear
        solve, all 0-based natural:

        ``tilt_angle``, ``h_min``, ``x_hmin``, ``re_max`` : numpy.ndarray
            Per-pad (length ``total_pads``).
        ``h_n``, ``dhdx_n``, ``vis_n_average``, ``nodal_pressure``, ``dpdx_n``,
        ``dpdz_n``, ``pressback_n`` : numpy.ndarray, shape
        ``(total_pads, dim_xz)``
        ``y_energy`` : numpy.ndarray, shape ``(total_pads, dim_xy)``
        ``y_3d``, ``vis_eddy_3d``, ``vis_effect_3d``, ``shear_stress``,
        ``dudy_n``, ``dwdy_n`` : numpy.ndarray, shape
        ``(total_pads, dim_3d)``
        ``flow_regime_track``, ``flow_regime_dam`` : numpy.ndarray of int
        ``scale_turb_track``, ``scale_turb_dam`` : numpy.ndarray
        ``unloaded`` : numpy.ndarray of bool
        ``full_cavitate`` : numpy.ndarray of bool, per pad
        ``cavitated`` : numpy.ndarray of bool, shape ``(total_pads, dim_xz)``
            Nodes held at the cavitation pressure (see
            :func:`ross.bearings.fluid_film.pressure.press`).
        ``_tilt_memory`` : dict
            Start information of the next call (see ``tilt_memory``).
    """
    y_energy = energy_mesh.y
    y_3d = mesh.y_3d
    from ross.bearings.fluid_film.pressure import press_full_cavitation

    if press is None:
        from ross.bearings.fluid_film.pressure import press

    # Work on copies so the per-pad updates accumulate without aliasing inputs.
    tilt_angle = np.ascontiguousarray(tilt_angle, dtype=np.float64)
    h_min = np.ascontiguousarray(h_min, dtype=np.float64)
    x_hmin = np.ascontiguousarray(x_hmin, dtype=np.float64)
    re_max = np.ascontiguousarray(re_max, dtype=np.float64)
    h_n = np.ascontiguousarray(h_n, dtype=np.float64)
    dh_n = np.asarray(dh_n, dtype=float)
    dhdx_n = np.ascontiguousarray(dhdx_n, dtype=np.float64)
    vis_n_average = np.ascontiguousarray(vis_n_average, dtype=np.float64)
    nodal_pressure = np.ascontiguousarray(nodal_pressure, dtype=np.float64)
    dpdx_n = np.ascontiguousarray(dpdx_n, dtype=np.float64)
    dpdz_n = np.ascontiguousarray(dpdz_n, dtype=np.float64)
    pressback_n = np.ascontiguousarray(pressback_n, dtype=np.float64)
    y_energy = np.ascontiguousarray(y_energy, dtype=np.float64)
    y_3d = np.ascontiguousarray(y_3d, dtype=np.float64)
    vis_eddy_3d = np.ascontiguousarray(vis_eddy_3d, dtype=np.float64)
    vis_effect_3d = np.ascontiguousarray(vis_effect_3d, dtype=np.float64)
    shear_stress = np.ascontiguousarray(shear_stress, dtype=np.float64)
    dudy_n = np.ascontiguousarray(dudy_n, dtype=np.float64)
    dwdy_n = np.ascontiguousarray(dwdy_n, dtype=np.float64)
    flow_regime_track = np.ascontiguousarray(flow_regime_track, dtype=np.int64)
    flow_regime_dam = np.ascontiguousarray(flow_regime_dam, dtype=np.int64)
    scale_turb_track = np.ascontiguousarray(scale_turb_track, dtype=np.float64)
    scale_turb_dam = np.ascontiguousarray(scale_turb_dam, dtype=np.float64)
    unloaded = np.array(unloaded, dtype=bool)
    if cavitated is None:
        cavitated = np.zeros(nodal_pressure.shape, dtype=bool)
    else:
        cavitated = np.array(cavitated, dtype=bool, copy=True)

    cb_new = np.zeros(total_pads, dtype=float)
    cp_new = np.zeros(total_pads, dtype=float)
    preload_new = np.zeros(total_pads, dtype=float)

    def _film(xj_local, yj_local):
        return film_thickness(
            mesh,
            pad_index,
            operating,
            energy_mesh,
            total_e_y_dambl,
            total_e_y_damcore,
            weight_h,
            pads,
            cb,
            tilt_angle,
            preload_new,
            xj_local,
            yj_local,
            h_n,
            dh_n,
            dhdx_n,
            cp_new,
            unloaded,
        )

    def _regime():
        return flow_regime(
            mesh,
            pad_index,
            flow_regime_track,
            flow_regime_dam,
            pads,
            lube_density,
            operating.speed_surface,
            turbulence,
            vis_n_3d,
            h_n,
            scale_turb_track,
            scale_turb_dam,
            vis_n_average,
            re_max,
        )

    def _eff_vis():
        return effective_viscosity(
            pad_index,
            mesh,
            total_e_y_dambl,
            total_e_y_damcore,
            pads,
            vis_n_3d,
            vis_n_average,
            shear_stress,
            h_n,
            lube_density,
            scale_turb_track,
            scale_turb_dam,
            vis_eddy_3d,
            vis_effect_3d,
            turbulence,
        )

    def _press():
        if full_cavitate[pad_index] and operating.cavitation_model == "reynolds":
            # A fully cavitated pad is at the cavitation pressure: no
            # Reynolds solve (the fields equal those of ``press``).
            return press_full_cavitation(
                mesh,
                operating,
                pad_index,
                pads,
                h_n,
                h_min,
                x_hmin,
                nodal_pressure,
                dpdx_n,
                dpdz_n,
                cavitated,
            )
        return press(
            mesh,
            operating,
            pad_index,
            pads,
            vis_effect_3d,
            film_onset,
            h_n,
            h_min,
            x_hmin,
            nodal_pressure,
            dpdx_n,
            dpdz_n,
            full_cavitate,
            cavitated,
        )

    def _grad():
        return dudy_dwdy(
            mesh,
            pad_index,
            pads,
            operating.speed_surface,
            h_n,
            dpdx_n,
            dpdz_n,
            vis_effect_3d,
            dudy_n,
            dwdy_n,
        )

    def _is_superlaminar():
        return (
            operating.bearing_type != "pressure_dam" and flow_regime_dam[pad_index] > 0
        ) or (
            operating.bearing_type == "pressure_dam"
            and flow_regime_track[pad_index] > 0
        )

    full_cavitate = np.zeros(total_pads, dtype=bool)
    tilt_tolerance = TILT_ERROR * tolerance_scale
    moment_tolerance = TILT_MOMENT_ERROR * tolerance_scale
    shear_tolerance = SHEAR_ERROR * tolerance_scale
    tilt_width = np.zeros(total_pads)
    tilt_moment = np.zeros(total_pads)
    tilt_residual = np.zeros(total_pads)
    tilt_converged = np.ones(total_pads, dtype=bool)
    tilt_iterations = np.zeros(total_pads, dtype=np.int64)
    if tilt_memory is not None:
        memory_known = np.array(tilt_memory["known"], dtype=bool, copy=True)
        memory_slope = np.array(tilt_memory["slope"], dtype=float, copy=True)
    else:
        memory_known = np.zeros(total_pads, dtype=bool)
        memory_slope = np.full(total_pads, np.nan)
    shear_residual = np.zeros(total_pads)
    shear_iterations = np.ones(total_pads, dtype=np.int64)

    for pad_index in range(total_pads):
        cb_new[pad_index] = cb + deform_pivot[pad_index]
        cp_new[pad_index] = pads.cp[pad_index]
        preload_new[pad_index] = 1.0 - cb_new[pad_index] / cp_new[pad_index]

        if (
            operating.operating_type in ("axial_flow", "high_ambient_pressure")
            and operating.bearing_type
            in (
                "conventional_tilting_pad",
                "inlet_groove_tilting_pad",
                "spray_bar_tilting_pad",
            )
            and equilpost_index == 1
        ):
            unloaded[pad_index] = False

        if operating.bearing_type in ("fixed_geometry", "pressure_dam"):
            lower_limit = 0.0
            upper_limit = 0.0
        elif held_tilts is not None:
            lower_limit = upper_limit = float(held_tilts[pad_index])
        else:
            tilt_angle_max, tilt_angle_min = tilt_angle_range(
                pad_index,
                mesh.total_nodes,
                mesh.n_index,
                pads,
                xj,
                yj,
                cb,
                cb_new,
                dh_n,
            )
            lower_limit = tilt_angle_min
            upper_limit = tilt_angle_max
            if operating.operating_type in ("axial_flow", "high_ambient_pressure"):
                lower_limit *= 0.9
                upper_limit *= 0.9

        # Tilt search: one film solve at the held (or zero) tilt, or the
        # root of the pivot moment inside the admissible range.
        search = None
        if lower_limit != upper_limit:
            warm_tilt = None
            warm_slope = None
            if tilt_memory is not None and bool(tilt_memory["known"][pad_index]):
                warm_tilt = float(tilt_angle[pad_index])
                warm_slope = float(tilt_memory["slope"][pad_index])
            search = PadTiltSearch(
                lower_limit,
                upper_limit,
                tilt_tolerance,
                moment_tolerance,
                start=warm_tilt,
                slope=warm_slope,
            )
        for _tilt_index in range(1, MAX_ITERATION + 1):
            tilt_iterations[pad_index] = _tilt_index
            while True:
                if search is None:
                    tilt_angle[pad_index] = lower_limit
                else:
                    tilt_angle[pad_index] = search.propose()

                res = _film(xj, yj)
                h_n = res["h_n"]
                dhdx_n = res["dhdx_n"]
                y_energy = res["y_energy"]
                y_3d = res["y_3d"]
                h_min[pad_index] = res["h_min"]
                x_hmin[pad_index] = res["x_hmin"]
                full_cavitate[pad_index] = res["full_cavitate"]

                # A closed film (h_min <= CLOSED_FILM * cb: negative, or of
                # round-off thickness at the touching bound of a zero-preload
                # pad) is not admissible; when no tilt is left to open it
                # (the journal position itself crosses the pad surface) the
                # closed film is kept.
                if (
                    h_min[pad_index] <= CLOSED_FILM * cb
                    and search is not None
                    and search.reject(tilt_angle[pad_index])
                ):
                    continue
                # Locating the edge of a fully cavitated range needs the
                # film only.
                if search is not None and search.skip(
                    tilt_angle[pad_index], full_cavitate[pad_index]
                ):
                    continue
                break

            res = _regime()
            flow_regime_track = res["flow_regime_track"]
            flow_regime_dam = res["flow_regime_dam"]
            scale_turb_track = res["scale_turb_track"]
            scale_turb_dam = res["scale_turb_dam"]
            re_max = res["re_max"]
            vis_n_average = res["vis_n_average"]

            res = _eff_vis()
            vis_eddy_3d = res["vis_eddy_3d"]
            vis_effect_3d = res["vis_effect_3d"]

            nodal_pressure, dpdx_n, dpdz_n, cavitated = _press()
            dudy_n, dwdy_n = _grad()

            if _is_superlaminar():
                for _press_index in range(1, MAX_ITERATION + 1):
                    shear_stress, rms_shear = update_shear(
                        pad_index,
                        mesh,
                        dudy_n,
                        dwdy_n,
                        vis_effect_3d,
                        shear_stress,
                        relaxp,
                    )
                    res = _eff_vis()
                    vis_eddy_3d = res["vis_eddy_3d"]
                    vis_effect_3d = res["vis_effect_3d"]
                    nodal_pressure, dpdx_n, dpdz_n, cavitated = _press()
                    dudy_n, dwdy_n = _grad()
                    shear_iterations[pad_index] = _press_index
                    shear_residual[pad_index] = rms_shear
                    if rms_shear < shear_tolerance:
                        break
            else:
                shear_stress, _rms = update_shear(
                    pad_index,
                    mesh,
                    dudy_n,
                    dwdy_n,
                    vis_effect_3d,
                    shear_stress,
                    1.0,
                )
                shear_iterations[pad_index] = 1
                shear_residual[pad_index] = 0.0

            moment_pivot, pressback_n = moment(
                mesh,
                operating,
                pad_index,
                pads,
                nodal_pressure,
                pressback_n,
                k_rotate,
                tilt_angle,
                integrate_xz,
            )

            if search is None:
                tilt_residual[pad_index] = 0.0
                tilt_converged[pad_index] = True
                break
            done = search.update(
                tilt_angle[pad_index],
                moment_pivot,
                moment_scale(
                    mesh,
                    operating,
                    pad_index,
                    pads,
                    nodal_pressure,
                    pressback_n,
                    k_rotate,
                    tilt_angle,
                    integrate_xz,
                ),
                flat=full_cavitate[pad_index],
            )
            tilt_residual[pad_index] = search.residual
            tilt_converged[pad_index] = search.converged
            tilt_width[pad_index] = search.width
            tilt_moment[pad_index] = search.moment_ratio
            if done:
                break
        if search is not None:
            memory_known[pad_index] = True
            if search.slope is not None:
                memory_slope[pad_index] = search.slope

        # Tilting pad under axial flow / high ambient pressure: loading check.
        if operating.operating_type in (
            "axial_flow",
            "high_ambient_pressure",
        ) and operating.bearing_type in (
            "conventional_tilting_pad",
            "inlet_groove_tilting_pad",
            "spray_bar_tilting_pad",
        ):
            unloaded = loading_condition(
                mesh,
                pad_index,
                nodal_pressure,
                pressback_n,
                pads.x_pivot_rad,
                unloaded,
                integrate_xz,
            )
            if unloaded[pad_index]:
                res = _film(0.0, 0.0)
                h_n = res["h_n"]
                dhdx_n = res["dhdx_n"]
                y_energy = res["y_energy"]
                y_3d = res["y_3d"]
                h_min[pad_index] = res["h_min"]
                x_hmin[pad_index] = res["x_hmin"]
                full_cavitate[pad_index] = res["full_cavitate"]

                res = _regime()
                flow_regime_track = res["flow_regime_track"]
                flow_regime_dam = res["flow_regime_dam"]
                scale_turb_track = res["scale_turb_track"]
                scale_turb_dam = res["scale_turb_dam"]
                re_max = res["re_max"]
                vis_n_average = res["vis_n_average"]

                res = _eff_vis()
                vis_eddy_3d = res["vis_eddy_3d"]
                vis_effect_3d = res["vis_effect_3d"]

                nodal_pressure, dpdx_n, dpdz_n, cavitated = _press()
                dudy_n, dwdy_n = _grad()

                if _is_superlaminar():
                    for _press_index in range(1, MAX_ITERATION + 1):
                        shear_stress, rms_shear = update_shear(
                            pad_index,
                            mesh,
                            dudy_n,
                            dwdy_n,
                            vis_effect_3d,
                            shear_stress,
                            relaxp,
                        )
                        res = _eff_vis()
                        vis_eddy_3d = res["vis_eddy_3d"]
                        vis_effect_3d = res["vis_effect_3d"]
                        nodal_pressure, dpdx_n, dpdz_n, cavitated = _press()
                        dudy_n, dwdy_n = _grad()
                        shear_iterations[pad_index] = _press_index
                        shear_residual[pad_index] = rms_shear
                        if rms_shear < shear_tolerance:
                            break
                else:
                    shear_stress, _rms = update_shear(
                        pad_index,
                        mesh,
                        dudy_n,
                        dwdy_n,
                        vis_effect_3d,
                        shear_stress,
                        1.0,
                    )
                    shear_iterations[pad_index] = 1
                    shear_residual[pad_index] = 0.0

    if monitor is not None:
        unconverged = ~tilt_converged
        if np.any(unconverged):
            worst = int(np.flatnonzero(unconverged)[0])
        else:
            worst = int(np.argmax(tilt_residual))
        monitor.record(
            "tilt",
            tilt_iterations[worst],
            tilt_residual[worst],
            1.0,
            bool(tilt_converged[worst]),
            pad=worst,
            moment=float(tilt_moment[worst]),
            width=float(tilt_width[worst]),
        )
        worst = int(np.argmax(shear_residual))
        monitor.record(
            "turbulence",
            shear_iterations[worst],
            shear_residual[worst],
            shear_tolerance,
            shear_residual[worst] < shear_tolerance,
            pad=worst,
        )

    return {
        "tilt_angle": tilt_angle,
        "h_min": h_min,
        "x_hmin": x_hmin,
        "re_max": re_max,
        "h_n": h_n,
        "dhdx_n": dhdx_n,
        "vis_n_average": vis_n_average,
        "nodal_pressure": nodal_pressure,
        "dpdx_n": dpdx_n,
        "dpdz_n": dpdz_n,
        "pressback_n": pressback_n,
        "y_energy": y_energy,
        "y_3d": y_3d,
        "vis_eddy_3d": vis_eddy_3d,
        "vis_effect_3d": vis_effect_3d,
        "shear_stress": shear_stress,
        "dudy_n": dudy_n,
        "dwdy_n": dwdy_n,
        "flow_regime_track": flow_regime_track,
        "flow_regime_dam": flow_regime_dam,
        "scale_turb_track": scale_turb_track,
        "scale_turb_dam": scale_turb_dam,
        "unloaded": unloaded,
        "full_cavitate": full_cavitate,
        "cavitated": cavitated,
        "_tilt_memory": {"known": memory_known, "slope": memory_slope},
    }
