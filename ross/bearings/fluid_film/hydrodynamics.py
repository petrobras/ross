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
    MAX_ITERATION,
    PI,
    SHEAR_ERROR,
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
        Relaxation factor for the journal-position update.
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

    f = 0.0
    f_old = 0.0
    xj_old = s["xj"]
    yj_old = s["yj"]
    hd_converged = 0

    for filmonset_index in range(1, total_e_x_film - 1 + 1):
        unconverge_number = 0
        hd_converged = 0

        # ``equilpost_index`` is a pure iteration counter (compared ``> 1`` and
        # passed through to ``fixed_brg``); kept 1-based for readability.
        for equilpost_index in range(1, MAX_ITERATION + 1):
            fb = fixed_brg_fn(s, equilpost_index, relaxp, press=press)
            s.update(fb)

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

            ext_mag = np.sqrt(fx_ext**2 + fy_ext**2)
            if (
                np.sqrt(fx_net**2 + fy_net**2) / ext_mag < 2.0e-3
                and equilpost_index > 1
            ):
                hd_converged = 0 if unconverge_number > 0 else 1
                break
            elif f > f_old and equilpost_index > 1:
                unconverge_number += 1
                if unconverge_number > 5:
                    break
                f_min = f_old
                xj_min = xj_old
                yj_min = yj_old
                if unconverge_number == 5:
                    s["xj"] = xj_min
                    s["yj"] = yj_min
                del f_min
            else:
                jac = jacobian_fn(s)
                # Stash the last in-loop Jacobian so the orchestrator can
                # consume it after the equilibrium loop exits. The
                # orchestrator does NOT re-call jacobian after this driver
                # returns, so the stiffness block consumed by the dynamic
                # reduction is the one computed here, on the
                # *pre-convergence* h_n. Recomputing on the converged
                # post-equilibrium h_n (as the orchestrator used to do)
                # shifted the K's ~0.15% on M1 (task #15) and broke pdam by
                # ~260% (task #18).
                s["_last_jacobian"] = jac

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

                if equilibrium_type == "match_load":
                    delta_xj, delta_yj = _newton_step(
                        kxx_e, kxy_e, kyx_e, kyy_e, fx_net, fy_net, s["cb"]
                    )
                    if delta_xj is None:
                        # Singular Jacobian fallback.
                        s["xj"] = 0.0
                        s["yj"] = 0.0
                        delta_xj = 0.0
                        delta_yj = -0.2 * s["cb"]
                elif equilibrium_type == "match_eccentricity":
                    break

                xj_old = s["xj"]
                yj_old = s["yj"]
                xj_new = xj_old + delta_xj
                yj_new = yj_old + delta_yj
                s["xj"] = (1.0 - relaxp) * xj_old + relaxp * xj_new
                s["yj"] = (1.0 - relaxp) * yj_old + relaxp * yj_new

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
        s.update(flow)

        # Starvation: stop here when flooded / suppressed.
        if operating_type in ("regular_flooded", "axial_flow", "high_ambient_pressure"):
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

        s["film_onset"] = update_film_onset(
            total_pads,
            total_e_x_film,
            s["q_x"],
            s["q_available"],
            s["film_onset"],
        )

    # Heat-dissipation scale factor for the cavitation/starvation effects.
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


def _newton_step(kxx_e, kxy_e, kyx_e, kyy_e, fx_net, fy_net, cb):
    """One Newton step on the journal position, with its fallbacks.

    Handles near-singular / decoupled stiffness, inverts the 2x2 system
    otherwise, and limits the step length to ``0.2 * cb``. Returns ``(None, None)`` for the
    fully singular case so the caller applies the special reset.
    """
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
                # NOTE: 360.0 (degrees) is added here while the
                # branch above adds 2*Pi (radians). The inconsistency is
                # kept as-is; the pinned regression fixtures depend on it.
                arc_length_groove = 360.0 + arc_length_groove
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
    """Bisection search bounds for the pad tilt angle.

    The bounds correspond to the trailing edge and the leading edge just
    touching the journal (preload taken as zero, so the bound is conservative).

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
    n_index_reynolds = mesh.n_index
    x_reynolds_rad = mesh.x_rad
    n_index_reynolds = np.asarray(n_index_reynolds, dtype=np.int64)
    nodal_pressure = np.asarray(nodal_pressure, dtype=float)
    x_reynolds_rad = np.asarray(x_reynolds_rad, dtype=float)
    pressback_n = np.ascontiguousarray(pressback_n, dtype=np.float64)
    p = pad_index

    idx = n_index_reynolds[: mesh.total_nodes]
    integrand = np.zeros(mesh.dim_xz, dtype=float)
    integrand[idx] = nodal_pressure[p, idx] * np.sin(
        x_reynolds_rad[p, idx] - pads.x_pivot_rad[p]
    )
    inte_f = integrate_xz(
        pad_index,
        mesh,
        integrand,
    )
    moment_pivot = -(pads.journal_radius + pads.pad_thickness) * inte_f

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
):
    """Solve the bearing film at a *fixed* journal position (isoviscous).

    For each pad it bisection-searches the tilt angle (zero range for fixed
    geometry and pressure dams), and for each trial angle:
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
    """
    y_energy = energy_mesh.y
    y_3d = mesh.y_3d
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

        # Bisection search on the tilt angle.
        for _tilt_index in range(1, MAX_ITERATION + 1):
            while True:
                if abs(upper_limit - lower_limit) < 1.0e-8:
                    tilt_angle[pad_index] = lower_limit
                else:
                    tilt_angle[pad_index] = 0.5 * (lower_limit + upper_limit)

                res = _film(xj, yj)
                h_n = res["h_n"]
                dhdx_n = res["dhdx_n"]
                y_energy = res["y_energy"]
                y_3d = res["y_3d"]
                h_min[pad_index] = res["h_min"]
                x_hmin[pad_index] = res["x_hmin"]
                full_cavitate[pad_index] = res["full_cavitate"]

                if h_min[pad_index] < 0.0:
                    if tilt_angle[pad_index] > 0.0:
                        upper_limit = tilt_angle[pad_index]
                    else:
                        lower_limit = tilt_angle[pad_index]
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

            nodal_pressure, dpdx_n, dpdz_n = _press()
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
                    nodal_pressure, dpdx_n, dpdz_n = _press()
                    dudy_n, dwdy_n = _grad()
                    if rms_shear < SHEAR_ERROR:
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

            if abs(upper_limit - lower_limit) < 1.0e-8:
                break
            if moment_pivot >= 0.0:
                lower_limit = tilt_angle[pad_index]
            else:
                upper_limit = tilt_angle[pad_index]

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

                nodal_pressure, dpdx_n, dpdz_n = _press()
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
                        nodal_pressure, dpdx_n, dpdz_n = _press()
                        dudy_n, dwdy_n = _grad()
                        if rms_shear < SHEAR_ERROR:
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
    }
