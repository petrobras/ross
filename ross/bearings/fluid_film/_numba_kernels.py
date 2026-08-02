"""Numba-jitted kernels for the hot inner loops of the solver.

A plain Python transcription of these kernels is unavoidably slow because the
per-element loops execute many times per case (each thd iteration calls
``hydrodynamics`` / ``thermal_full`` and each Newton step calls a fresh
``press``). ``cProfile`` on the M1 isoviscous run identifies the dominant cost
in just a handful of leaf functions: ``trapezoid``, ``assemble_press``,
the banded LU factor and its two substitution variants.

This module provides ``@njit`` versions of those kernels. They are functionally
identical to their pure-Python counterparts in :mod:`pressure`,
:mod:`coefficients` and :mod:`driver`; the corresponding wrappers in those
modules delegate to the kernels here. Keeping the kernels in a single module
keeps Numba's compile-time hits localized (one cold-start per process) and
isolates the only places where the convention (``float64`` arrays, scalar int
loop bounds, no Python objects) is enforced.

All arrays are 0-based natural along their mesh axes: node/element ids are
0-based values used directly as indices, per-pad fields are shaped
``(total_pads, dim...)`` and indexed ``[pad_index, node]`` with both axes
0-based, and the band diagonal of the banded systems sits at column
``bandwidth - 1``. The kernels do not re-index; they read/write exactly the
slots their pure-Python equivalents do. Loop counters that walk a cross-film
column position (``j``) or a band-storage offset remain local 1-based position
counters and are unrelated to the 0-based mesh numbering.
"""

import numpy as np
from numba import njit


@njit(cache=True, fastmath=False)
def trapezoid_jit(t, f, start, stop):
    """Trapezoid rule over the (possibly unequal) grid ``t[start:stop]``.

    ``start`` / ``stop`` are ordinary Python slice bounds. Fewer than two
    samples integrate to zero.
    """
    if stop - start < 2:
        return 0.0
    total = 0.0
    for i in range(start + 1, stop):
        h = t[i] - t[i - 1]
        total += 0.5 * h * (f[i] + f[i - 1])
    return total


@njit(cache=True, fastmath=False)
def assemble_press_jit(
    e_matrix_reynolds,
    e_column_reynolds,
    local_coordinates_4,
    bandwidth_reynolds,
    global_matrix_p,
    global_column_p,
):
    """Assemble one Q4 element into a 0-based banded global system.

    Node ``k`` sits at row ``k`` (0-based) and the band diagonal at column
    ``bandwidth - 1``. ``local_coordinates_4`` is length 4 carrying 0-based
    node ids (slots 0..3 == local corners 0..3).
    """
    for i in range(4):
        irow = local_coordinates_4[i]
        for j in range(4):
            icol = local_coordinates_4[j]
            jcol = icol - irow + bandwidth_reynolds - 1
            global_matrix_p[irow, jcol] += e_matrix_reynolds[i, j]
        global_column_p[irow] += e_column_reynolds[i]
    return global_matrix_p, global_column_p


@njit(cache=True, fastmath=False)
def include_press_jit(
    global_matrix_p,
    global_column_p,
    bandwidth_reynolds,
    total_bc_reynolds,
    press_bc_index,
    prescribed_press,
    total_n_reynolds,
):
    """Impose prescribed nodal pressures on the banded global system.

    Mirrors :func:`ross.bearings.fluid_film.pressure.include_press`. ``press_bc_index``
    (0-based node ids) and ``prescribed_press`` are 0-based length
    ``total_bc_reynolds``. Node ``k`` sits at row ``k`` (0-based).
    """
    twb = 2 * bandwidth_reynolds
    for i in range(total_bc_reynolds):
        irow = press_bc_index[i]

        # Move the coupled terms to the right-hand side and zero them.
        for j in range(2, twb + 1):
            jrow = irow - bandwidth_reynolds + j - 1
            jcol = twb - j + 1
            if 0 <= jrow <= total_n_reynolds - 1:
                global_column_p[jrow] -= (
                    global_matrix_p[jrow, jcol - 1] * prescribed_press[i]
                )
                if jrow != irow:
                    global_matrix_p[jrow, jcol - 1] = 0.0

        # Zero the constrained row except the diagonal, set the RHS.
        for jcol in range(1, twb):
            if jcol != bandwidth_reynolds:
                global_matrix_p[irow, jcol - 1] = 0.0
        global_column_p[irow] = (
            global_matrix_p[irow, bandwidth_reynolds - 1] * prescribed_press[i]
        )
    return global_matrix_p, global_column_p


@njit(cache=True, fastmath=False)
def lu_factor_band_jit(a, total_n_reynolds, bandwidth_reynolds):
    """Banded LU decomposition (Numerical Recipes ``bandec``), 0-based natural.

    Backs :func:`ross.bearings.fluid_film.banded.lu_factor`. ``a`` has shape
    ``(dim_xz, >= total_column)`` and is overwritten with the decomposition;
    the band-storage repacking loops are 0-based natural. ``a_lower`` only
    ever holds the ``bandwidth - 1`` sub-diagonal multipliers per row, so it
    is allocated at that width (NR ``bandec``'s ``al(n, m1)``).
    """
    dim_xz = a.shape[0]
    a_lower = np.zeros((dim_xz, bandwidth_reynolds - 1), dtype=np.float64)
    index1 = np.zeros(dim_xz, dtype=np.int64)

    m1 = bandwidth_reynolds - 1
    total_column = 2 * bandwidth_reynolds - 1

    # Rearrange the storage so the band is left-justified.
    ll = m1
    for i in range(1, m1 + 1):
        for j in range(m1 + 2 - i, total_column + 1):
            a[i - 1, j - ll - 1] = a[i - 1, j - 1]
        ll -= 1
        for j in range(total_column - ll, total_column + 1):
            a[i - 1, j - 1] = 0.0

    d = 1.0
    ll = m1
    for k in range(1, total_n_reynolds + 1):
        dum = a[k - 1, 0]
        i = k
        if ll < total_n_reynolds:
            ll += 1
        for j in range(k + 1, ll + 1):
            if abs(a[j - 1, 0]) > abs(dum):
                dum = a[j - 1, 0]
                i = j
        index1[k - 1] = i
        if i != k:
            d = -d
            for j in range(1, total_column + 1):
                tmp = a[k - 1, j - 1]
                a[k - 1, j - 1] = a[i - 1, j - 1]
                a[i - 1, j - 1] = tmp
        for ii in range(k + 1, ll + 1):
            dum = a[ii - 1, 0] / a[k - 1, 0]
            a_lower[k - 1, ii - k - 1] = dum
            for j in range(2, total_column + 1):
                a[ii - 1, j - 2] = a[ii - 1, j - 1] - dum * a[k - 1, j - 1]
            a[ii - 1, total_column - 1] = 0.0

    return a, a_lower, index1, d


@njit(cache=True, fastmath=False)
def lu_solve_band_cavitating_jit(
    a,
    total_n_reynolds,
    bandwidth_reynolds,
    a_lower,
    index1,
    b,
    press_cavitate,
):
    """Banded LU back/forward solve with the Reynolds cavitation clamp (0-based).

    ``a`` has shape ``(dim_xz, >= total_column)`` and ``a_lower`` shape
    ``(dim_xz, bandwidth - 1)`` (the factors returned by
    :func:`lu_factor_band_jit`);
    ``index1`` carries 1-based pivot rows at slot ``k - 1``; ``b`` is 0-based
    (node ``k`` at slot ``k - 1``) and is overwritten in place.
    """
    total_column = 2 * bandwidth_reynolds - 1

    # Forward substitution, unscrambling the permuted rows.
    ll = bandwidth_reynolds - 1
    for k in range(1, total_n_reynolds + 1):
        ip = index1[k - 1]
        if ip != k:
            tmp = b[k - 1]
            b[k - 1] = b[ip - 1]
            b[ip - 1] = tmp
        if ll < total_n_reynolds:
            ll += 1
        for i in range(k + 1, ll + 1):
            b[i - 1] -= a_lower[k - 1, i - k - 1] * b[k - 1]

    # Back substitution, applying the Reynolds cavitation condition.
    ll = 1
    for i in range(total_n_reynolds, 0, -1):
        dum = b[i - 1]
        for k in range(2, ll + 1):
            dum -= a[i - 1, k - 1] * b[k + i - 2]
        b[i - 1] = dum / a[i - 1, 0]
        if ll < total_column:
            ll += 1
        b[i - 1] = max(b[i - 1], press_cavitate)
    return b


@njit(cache=True, fastmath=False)
def gamma_g_node_jit(
    nid,
    lo,
    hi,
    offset,
    p,
    match_nodes_xz,
    y_3d,
    h_n,
    vis_effect_3d,
):
    """Per-node Gamma/G integrals for the perturbation Reynolds system.

    ``nid`` is a 0-based Reynolds node id; ``match_nodes_xz`` stores 0-based
    3-D node ids and is indexed ``[nid, jj - 1]`` (the cross-film column ``jj``
    is 1-based, slot ``jj - 1``). ``y_3d``/``vis_effect_3d``/``h_n`` are indexed
    directly with the 0-based pad row ``p`` and 0-based node ids. ``lo``/``hi``
    are 1-based inclusive cross-film column bounds (unchanged: they index local
    scratch buffers, not a mesh array).
    """
    t = np.zeros(hi, dtype=np.float64)
    f1 = np.zeros(hi, dtype=np.float64)
    f2 = np.zeros(hi, dtype=np.float64)

    h_at_nid = h_n[p, nid]

    for jj in range(lo, hi + 1):
        m = match_nodes_xz[nid, jj - 1]
        t[jj - 1] = (y_3d[p, m] - offset) / h_at_nid
        ve = vis_effect_3d[p, m]
        f1[jj - 1] = 1.0 / ve
        f2[jj - 1] = t[jj - 1] / ve

    xi1h = 0.0
    xi2h = 0.0
    for i in range(lo, hi):
        h = t[i] - t[i - 1]
        xi1h += 0.5 * h * (f1[i] + f1[i - 1])
        xi2h += 0.5 * h * (f2[i] + f2[i - 1])

    integrand_gamma = np.zeros(hi, dtype=np.float64)
    integrand_g = np.zeros(hi, dtype=np.float64)
    cum_xi1 = 0.0
    cum_xi2 = 0.0
    integrand_gamma[lo - 1] = 0.0
    integrand_g[lo - 1] = 0.0
    for jj in range(lo + 1, hi + 1):
        h = t[jj - 1] - t[jj - 2]
        cum_xi1 += 0.5 * h * (f1[jj - 1] + f1[jj - 2])
        cum_xi2 += 0.5 * h * (f2[jj - 1] + f2[jj - 2])
        integrand_gamma[jj - 1] = cum_xi2 - (xi2h / xi1h) * cum_xi1
        integrand_g[jj - 1] = cum_xi1 / xi1h

    gamma_node = 0.0
    g_node = 0.0
    for i in range(lo, hi):
        h = t[i] - t[i - 1]
        gamma_node += 0.5 * h * (integrand_gamma[i] + integrand_gamma[i - 1])
        g_node += 0.5 * h * (integrand_g[i] + integrand_g[i - 1])

    return gamma_node, g_node


@njit(cache=True, fastmath=False)
def gamma_g_loop_jit(
    p,
    total_n_reynolds,
    n_index_reynolds,
    total_e_y_film,
    match_nodes_xz,
    total_e_y_trackbl_p,
    total_e_y_trackcore_p,
    pad_thickness,
    pad_length_p,
    axial_length_p,
    depth_track_p,
    length_track_p,
    axial_length_track_p,
    axial_length_dam_p,
    x_reynolds,
    z_reynolds,
    y_3d,
    h_n,
    vis_effect_3d,
    gamma_out,
    g_out,
):
    """Drive :func:`gamma_g_node_jit` over every Reynolds node on the pad.

    ``p`` is 0-based; ``n_index_reynolds`` stores 0-based node ids;
    ``match_nodes_xz`` stores 0-based 3-D node ids. ``gamma_out``/``g_out`` are
    written at slot ``nid`` (0-based). The per-pad scalars are passed already
    indexed.
    """
    limit1 = total_e_y_trackbl_p + total_e_y_trackcore_p + 1
    limit2 = total_e_y_film + 1

    for i in range(total_n_reynolds):
        nid = n_index_reynolds[i]

        zr = z_reynolds[p, nid]
        xr = x_reynolds[p, nid]

        in_pocket = (
            zr > axial_length_dam_p
            and zr < axial_length_dam_p + axial_length_track_p
            and xr < length_track_p
        )
        in_dam = (
            xr > length_track_p
            or zr < axial_length_dam_p
            or zr > axial_length_dam_p + axial_length_track_p
        )

        if in_pocket:
            gn, gn2 = gamma_g_node_jit(
                nid,
                1,
                limit2,
                pad_thickness,
                p,
                match_nodes_xz,
                y_3d,
                h_n,
                vis_effect_3d,
            )
        elif in_dam:
            gn, gn2 = gamma_g_node_jit(
                nid,
                limit1,
                limit2,
                pad_thickness + depth_track_p,
                p,
                match_nodes_xz,
                y_3d,
                h_n,
                vis_effect_3d,
            )
        else:
            edge_is_pad_edge = (
                abs(xr - pad_length_p) < 1.0e-6
                and zr > axial_length_dam_p
                and zr < axial_length_track_p + axial_length_dam_p
            ) or (
                (abs(zr) < 1.0e-6 or abs(zr - axial_length_p) < 1.0e-6)
                and xr < length_track_p
            )
            if edge_is_pad_edge:
                gn, gn2 = gamma_g_node_jit(
                    nid,
                    1,
                    limit2,
                    pad_thickness,
                    p,
                    match_nodes_xz,
                    y_3d,
                    h_n,
                    vis_effect_3d,
                )
            else:
                gn, gn2 = gamma_g_node_jit(
                    nid,
                    limit1,
                    limit2,
                    pad_thickness + depth_track_p,
                    p,
                    match_nodes_xz,
                    y_3d,
                    h_n,
                    vis_effect_3d,
                )
        gamma_out[nid] = gn
        g_out[nid] = gn2


@njit(cache=True, fastmath=False)
def temp_xy_assemble_all_jit(
    pad,
    total_e_energy,
    e_index_energy,
    node_1_energy,
    node_2_energy,
    node_3_energy,
    node_4_energy,
    x_energy,
    y_energy,
    kx_n,
    ky_n,
    mx_n,
    my_n,
    p_n,
    q_n,
    pad_thickness,
    bandwidth_energy,
    global_matrix,
    global_column,
):
    """Assemble every energy-equation element into the 0-based banded global
    system in a single JIT call.

    Fuses ``element_temp``'s interior 2x2 Gauss quadrature and the 0-based
    ``_assemble`` over all ``total_e_energy`` elements. ``pad`` is already
    0-based. Mutates ``global_matrix`` and ``global_column`` in place.

    This kernel covers ONLY the interior contributions. The edge line
    integrals (LE/TE/back convection) are assembled afterwards by
    :func:`temp_xy_boundary_all_jit`; since both passes only ever *add* into
    the global system, splitting them preserves the original per-element
    (interior + edges before assemble) result.
    """
    # See element_temp_interior_jit for the inlined Gauss arithmetic.
    g_gauss = 1.0 / 3.0**0.5

    for ie in range(total_e_energy):
        ce = e_index_energy[ie]
        n1 = node_1_energy[ce]
        n2 = node_2_energy[ce]
        n3 = node_3_energy[ce]
        n4 = node_4_energy[ce]

        kx_e = (kx_n[n1] + kx_n[n2] + kx_n[n3] + kx_n[n4]) * 0.25
        ky_e = (ky_n[n1] + ky_n[n2] + ky_n[n3] + ky_n[n4]) * 0.25
        mx_e = (mx_n[n1] + mx_n[n2] + mx_n[n3] + mx_n[n4]) * 0.25
        my_e = (my_n[n1] + my_n[n2] + my_n[n3] + my_n[n4]) * 0.25
        p_e = (p_n[n1] + p_n[n2] + p_n[n3] + p_n[n4]) * 0.25
        q_e = (q_n[n1] + q_n[n2] + q_n[n3] + q_n[n4]) * 0.25

        # Dissipation is zero in the solid (back-of-pad nodes).
        if (
            y_energy[pad, n3] < pad_thickness
            or abs(y_energy[pad, n3] - pad_thickness) < 1.0e-6
        ):
            p_e = 0.0
            q_e = 0.0

        # element_temp_interior_jit body inlined (same arithmetic).
        x1 = x_energy[pad, n1]
        x2 = x_energy[pad, n2]
        x3 = x_energy[pad, n3]
        x4 = x_energy[pad, n4]
        y1 = y_energy[pad, n1]
        y2 = y_energy[pad, n2]
        y3 = y_energy[pad, n3]
        y4 = y_energy[pad, n4]

        e_mat = np.zeros((4, 4), dtype=np.float64)
        e_col = np.zeros(4, dtype=np.float64)

        for gp in range(4):
            if gp == 0:
                r = -g_gauss
                s = -g_gauss
            elif gp == 1:
                r = g_gauss
                s = -g_gauss
            elif gp == 2:
                r = -g_gauss
                s = g_gauss
            else:
                r = g_gauss
                s = g_gauss

            n0_ = (1.0 - r) * (1.0 - s) * 0.25
            n1v = (1.0 + r) * (1.0 - s) * 0.25
            n2v = (1.0 + r) * (1.0 + s) * 0.25
            n3v = (1.0 - r) * (1.0 + s) * 0.25

            f00 = -(1.0 - s) * 0.25
            f01 = (1.0 - s) * 0.25
            f02 = (1.0 + s) * 0.25
            f03 = -(1.0 + s) * 0.25
            f10 = -(1.0 - r) * 0.25
            f11 = -(1.0 + r) * 0.25
            f12 = (1.0 + r) * 0.25
            f13 = (1.0 - r) * 0.25

            jac00 = f00 * x1 + f01 * x2 + f02 * x3 + f03 * x4
            jac01 = f00 * y1 + f01 * y2 + f02 * y3 + f03 * y4
            jac10 = f10 * x1 + f11 * x2 + f12 * x3 + f13 * x4
            jac11 = f10 * y1 + f11 * y2 + f12 * y3 + f13 * y4

            det_j = jac00 * jac11 - jac01 * jac10
            inv00 = jac11 / det_j
            inv01 = -jac01 / det_j
            inv10 = -jac10 / det_j
            inv11 = jac00 / det_j

            b00 = inv00 * f00 + inv01 * f10
            b01 = inv00 * f01 + inv01 * f11
            b02 = inv00 * f02 + inv01 * f12
            b03 = inv00 * f03 + inv01 * f13
            b10 = inv10 * f00 + inv11 * f10
            b11 = inv10 * f01 + inv11 * f11
            b12 = inv10 * f02 + inv11 * f12
            b13 = inv10 * f03 + inv11 * f13

            n_arr = (n0_, n1v, n2v, n3v)
            br0 = (b00, b01, b02, b03)
            br1 = (b10, b11, b12, b13)

            for i in range(4):
                for j in range(4):
                    btkb = br0[i] * kx_e * br0[j] + br1[i] * ky_e * br1[j]
                    ntvb = n_arr[i] * (mx_e * br0[j] + my_e * br1[j])
                    pntn = p_e * n_arr[i] * n_arr[j]
                    e_mat[i, j] += (btkb - ntvb - pntn) * det_j
                e_col[i] += n_arr[i] * q_e * det_j

        # Note: line integrals (LE/TE/back-of-pad convection) are added by
        # the Python wrapper *after* this kernel returns -- see the caller
        # in :func:`thermal.temp_xy`.

        # Assemble e_mat into global_matrix (0-based band: irow=node-1,
        # jcol=icol-irow+bw-1).
        nodes = (n1, n2, n3, n4)
        for i in range(4):
            irow = nodes[i]
            for j in range(4):
                icol = nodes[j]
                jcol = icol - irow + bandwidth_energy - 1
                global_matrix[irow, jcol] += e_mat[i, j]
            global_column[irow] += e_col[i]


@njit(cache=True, fastmath=False)
def _temp_line_gauss_jit(
    e_mat, e_col, x1, x2, x3, x4, y1, y2, y3, y4, r, s, h, t_ambient
):
    """Accumulate one edge Gauss point into ``e_mat`` / ``e_col``.

    Same arithmetic, term order and ``dl`` branch structure as
    :func:`thermal.integrand_line1` so results are bit-identical.
    """
    n0 = (1.0 - r) * (1.0 - s) / 4.0
    n1 = (1.0 + r) * (1.0 - s) / 4.0
    n2 = (1.0 + r) * (1.0 + s) / 4.0
    n3 = (1.0 - r) * (1.0 + s) / 4.0

    f00 = -(1.0 - s) / 4.0
    f01 = (1.0 - s) / 4.0
    f02 = (1.0 + s) / 4.0
    f03 = -(1.0 + s) / 4.0
    f10 = -(1.0 - r) / 4.0
    f11 = -(1.0 + r) / 4.0
    f12 = (1.0 + r) / 4.0
    f13 = (1.0 - r) / 4.0

    jac00 = f00 * x1 + f01 * x2 + f02 * x3 + f03 * x4
    jac10 = f10 * x1 + f11 * x2 + f12 * x3 + f13 * x4
    jac11 = f10 * y1 + f11 * y2 + f12 * y3 + f13 * y4

    if abs(r - 1.0) < 1.0e-6 or abs(r + 1.0) < 1.0e-6:
        dl = np.sqrt(jac00**2 + jac11**2)
    elif abs(s - 1.0) < 1.0e-6 or abs(s + 1.0) < 1.0e-6:
        dl = np.sqrt(jac10**2 + jac11**2)
    else:  # pragma: no cover - only edges are integrated
        dl = 0.0

    n_arr = (n0, n1, n2, n3)
    for i in range(4):
        for j in range(4):
            e_mat[i, j] += (h * (n_arr[i] * n_arr[j])) * dl
        e_col[i] += ((h * t_ambient) * n_arr[i]) * dl


@njit(cache=True, fastmath=False)
def temp_xy_boundary_all_jit(
    pad,
    total_e_energy,
    e_index_energy,
    node_1_energy,
    node_2_energy,
    node_3_energy,
    node_4_energy,
    x_energy,
    y_energy,
    pad_length_p,
    pad_thickness,
    t_ambient,
    convec_edges,
    convec_back_p,
    bandwidth_energy,
    global_matrix,
    global_column,
):
    """Boundary-line (LE / TE / back convection) pass of ``temp_xy`` in one
    JIT call.

    Fuses the per-element edge predicates, the two-point Gauss line integrals
    (:func:`thermal.integrand_line1`, inlined via
    :func:`_temp_line_gauss_jit`) and the 0-based banded ``_assemble`` over
    all ``total_e_energy`` elements. Edge order (LE, TE, back) and Gauss-point
    order (-g, +g) match the original Python loop so accumulation is
    bit-identical. Mutates ``global_matrix`` / ``global_column`` in place.
    """
    g_line = 1.0 / np.sqrt(3.0)

    for ie in range(total_e_energy):
        ce = e_index_energy[ie]
        n1 = node_1_energy[ce]
        n2 = node_2_energy[ce]
        n3 = node_3_energy[ce]
        n4 = node_4_energy[ce]

        in_solid = (
            y_energy[pad, n3] < pad_thickness
            or abs(y_energy[pad, n3] - pad_thickness) < 1.0e-6
        )
        on_le = abs(x_energy[pad, n1]) < 1.0e-6 and in_solid
        on_te = abs(x_energy[pad, n3] - pad_length_p) < 1.0e-6 and in_solid
        on_back = abs(y_energy[pad, n1]) < 1.0e-6

        if not (on_le or on_te or on_back):
            continue

        x1 = x_energy[pad, n1]
        x2 = x_energy[pad, n2]
        x3 = x_energy[pad, n3]
        x4 = x_energy[pad, n4]
        y1 = y_energy[pad, n1]
        y2 = y_energy[pad, n2]
        y3 = y_energy[pad, n3]
        y4 = y_energy[pad, n4]

        e_mat = np.zeros((4, 4), dtype=np.float64)
        e_col = np.zeros(4, dtype=np.float64)

        if on_le:
            for gp in range(2):
                g2 = -g_line if gp == 0 else g_line
                _temp_line_gauss_jit(
                    e_mat,
                    e_col,
                    x1,
                    x2,
                    x3,
                    x4,
                    y1,
                    y2,
                    y3,
                    y4,
                    -1.0,
                    g2,
                    -convec_edges,
                    t_ambient,
                )
        if on_te:
            for gp in range(2):
                g2 = -g_line if gp == 0 else g_line
                _temp_line_gauss_jit(
                    e_mat,
                    e_col,
                    x1,
                    x2,
                    x3,
                    x4,
                    y1,
                    y2,
                    y3,
                    y4,
                    1.0,
                    g2,
                    -convec_edges,
                    t_ambient,
                )
        if on_back:
            for gp in range(2):
                g2 = -g_line if gp == 0 else g_line
                _temp_line_gauss_jit(
                    e_mat,
                    e_col,
                    x1,
                    x2,
                    x3,
                    x4,
                    y1,
                    y2,
                    y3,
                    y4,
                    g2,
                    -1.0,
                    -convec_back_p,
                    t_ambient,
                )

        nodes = (n1, n2, n3, n4)
        for i in range(4):
            irow = nodes[i]
            for j in range(4):
                icol = nodes[j]
                jcol = icol - irow + bandwidth_energy - 1
                global_matrix[irow, jcol] += e_mat[i, j]
            global_column[irow] += e_col[i]


@njit(cache=True, fastmath=False)
def temp_journal_film_average_jit(
    total_pads,
    total_n_reynolds,
    n_index_reynolds,
    total_e_y_trackbl,
    total_e_y_trackcore,
    match_nodes_xz,
    total_e_y_film,
    total_e_reynolds,
    e_index_reynolds,
    node_i_reynolds,
    node_j_reynolds,
    node_k_reynolds,
    node_l_reynolds,
    pad_thickness,
    pad_length,
    axial_length,
    length_track,
    depth_track,
    axial_length_dam,
    axial_length_track,
    x_reynolds,
    z_reynolds,
    y_3d,
    h_n,
    temp_3d,
    e_length_reynolds,
    e_width_reynolds,
    dim_xz,
    dim_yf,
):
    """Inner body of :func:`thd.temp_journal_film_average`.

    All arrays 0-based (thd convention); ``pad`` loops over [0, total_pads).
    Returns the journal surface temperature (scalar).
    """
    integrand = np.zeros(dim_xz, dtype=np.float64)
    t = np.zeros(dim_yf, dtype=np.float64)
    f = np.zeros(dim_yf, dtype=np.float64)

    total_pad_area = 0.0
    sum_temp = 0.0
    for pad in range(total_pads):
        total_pad_area += pad_length[pad] * axial_length[pad]

        limit1 = total_e_y_trackbl[pad] + total_e_y_trackcore[pad] + 1
        limit2 = total_e_y_film + 1

        for i in range(total_n_reynolds):
            integrand[i] = 0.0

        for i in range(total_n_reynolds):
            node = n_index_reynolds[i]
            x = x_reynolds[pad, node]
            z = z_reynolds[pad, node]

            # Inlined _node_region.
            in_pocket = (
                z > axial_length_dam[pad]
                and z < axial_length_dam[pad] + axial_length_track[pad]
                and x < length_track[pad]
            )
            in_dam = (
                x > length_track[pad]
                or z < axial_length_dam[pad]
                or z > axial_length_dam[pad] + axial_length_track[pad]
            )
            if in_pocket:
                pocket = True
            elif in_dam:
                pocket = False
            else:
                cond_x = (
                    abs(x - pad_length[pad]) < 1.0e-6
                    and z > axial_length_dam[pad]
                    and z < axial_length_track[pad] + axial_length_dam[pad]
                )
                cond_z = (
                    abs(z) < 1.0e-6 or abs(z - axial_length[pad]) < 1.0e-6
                ) and x < length_track[pad]
                pocket = cond_x or cond_z

            if pocket:
                lo = 1
                offset = pad_thickness
            else:
                lo = limit1
                offset = pad_thickness + depth_track[pad]

            for j in range(lo, limit2 + 1):
                m = match_nodes_xz[node, j - 1]
                t[j - 1] = y_3d[pad, m] - offset
                f[j - 1] = temp_3d[pad, m]

            inte_trap = 0.0
            for ii in range(lo, limit2):
                h = t[ii] - t[ii - 1]
                inte_trap += 0.5 * h * (f[ii] + f[ii - 1])
            integrand[node] = inte_trap / h_n[pad, node]

        # integrate_xz inline (0-based variant).
        inte_temp = 0.0
        for i in range(total_e_reynolds):
            e = e_index_reynolds[i]
            area = e_length_reynolds[pad, e] * e_width_reynolds[pad, e]
            ni = node_i_reynolds[e]
            nj = node_j_reynolds[e]
            nk = node_k_reynolds[e]
            nl = node_l_reynolds[e]
            inte_temp += (
                area
                * (integrand[ni] + integrand[nj] + integrand[nk] + integrand[nl])
                * 0.25
            )
        sum_temp += inte_temp

    return sum_temp / total_pad_area


@njit(cache=True, fastmath=False)
def press_bc_pert_jit(
    p,
    dim_xz,
    total_e_z_film,
    film_onset_p,
    pad_length_p,
    axial_length_p,
    arc_length_rad_0,
    nodal_pressure,
    press_cavitate,
    total_n_reynolds,
    n_index_reynolds,
    x_reynolds,
    z_reynolds,
):
    """Inner body of :func:`coefficients.press_bc_pert`.

    Fully 0-based natural: ``p`` is 0-based; ``n_index_reynolds`` stores 0-based
    node ids; ``film_onset_p`` is 0-based. Returns
    ``(total_bc, press_bc_index, prescribed_press)`` with ``press_bc_index``
    carrying 0-based node ids (only first ``total_bc`` entries meaningful).
    """
    press_bc_index = np.zeros(dim_xz, dtype=np.int64)
    prescribed_press = np.zeros(dim_xz, dtype=np.float64)
    pi2 = 6.283185307179586

    j = 0
    if abs(arc_length_rad_0 - pi2) < 1.0e-6:
        for i in range(total_n_reynolds):
            nid = n_index_reynolds[i]
            zr = z_reynolds[p, nid]
            if abs(zr) < 1.0e-6 or abs(zr - axial_length_p) < 1.0e-6:
                press_bc_index[j] = nid
                prescribed_press[j] = 0.0
                j += 1
    else:
        for i in range(total_n_reynolds):
            nid = n_index_reynolds[i]
            zr = z_reynolds[p, nid]
            xr = x_reynolds[p, nid]
            if (
                abs(zr) < 1.0e-6
                or abs(zr - axial_length_p) < 1.0e-6
                or nid + 1 <= (film_onset_p + 1) * (total_e_z_film + 1)
                or abs(xr - pad_length_p) < 1.0e-6
                or abs(nodal_pressure[p, nid] - press_cavitate) < 1.0e-6
            ):
                press_bc_index[j] = nid
                prescribed_press[j] = 0.0
                j += 1
    return j, press_bc_index, prescribed_press


@njit(cache=True, fastmath=False)
def integrate_forces_jit(
    p,
    dim_xz,
    total_e_reynolds,
    e_index_reynolds,
    node_i_reynolds,
    node_j_reynolds,
    node_k_reynolds,
    node_l_reynolds,
    e_length_reynolds,
    e_width_reynolds,
    n_index_reynolds,
    total_n_reynolds,
    n_press_pert,
    leading_angle_rad_p,
    x_reynolds_rad,
    x_pivot_rad_p,
    journal_radius,
    pad_thickness,
):
    """Build the three perturbation-force integrands and integrate each.

    Fuses :func:`coefficients._integrate_forces` -- the
    ``dpx = pp * cos``, ``dpy = pp * sin``, ``mp = pp * sin(theta - pivot)``
    nodal field assembly + three calls to ``integrate_xz`` (each was a
    Python wrapper around a JIT kernel) all run in one JIT call. ``p``
    is already 0-based; all arrays uniformly 0-based natural (0-based node
    and element values).
    """
    dpx = np.zeros(dim_xz, dtype=np.float64)
    dpy = np.zeros(dim_xz, dtype=np.float64)
    mp = np.zeros(dim_xz, dtype=np.float64)
    for i in range(total_n_reynolds):
        nid = n_index_reynolds[i]
        pp = n_press_pert[nid]
        theta_node = x_reynolds_rad[p, nid]
        dpx[nid] = pp * np.cos(leading_angle_rad_p + theta_node)
        dpy[nid] = pp * np.sin(leading_angle_rad_p + theta_node)
        mp[nid] = pp * np.sin(theta_node - x_pivot_rad_p)

    fx = 0.0
    fy = 0.0
    mp_int = 0.0
    for i in range(total_e_reynolds):
        e = e_index_reynolds[i]
        area = e_length_reynolds[p, e] * e_width_reynolds[p, e]
        ni = node_i_reynolds[e]
        nj = node_j_reynolds[e]
        nk = node_k_reynolds[e]
        nl = node_l_reynolds[e]
        fx += area * (dpx[ni] + dpx[nj] + dpx[nk] + dpx[nl]) * 0.25
        fy += area * (dpy[ni] + dpy[nj] + dpy[nk] + dpy[nl]) * 0.25
        mp_int += area * (mp[ni] + mp[nj] + mp[nk] + mp[nl]) * 0.25
    moment = (journal_radius + pad_thickness) * mp_int
    return fx, fy, moment


@njit(cache=True, fastmath=False)
def integrate_xz_jit(
    pad_index,
    total_e,
    e_index,
    ni,
    nj,
    nk,
    nl,
    elen,
    ewid,
    f,
):
    """Surface integral of ``f``, fully 0-based natural layout.

    Mirrors the inner loop of ``driver._integrate_xz_coeff``. All arrays
    0-based natural; ``pad_index`` is the 0-based pad row, ``e_index`` and the
    ``ni``..``nl`` connectivity store 0-based ids, ``f`` is 0-based by node.
    """
    p = pad_index
    inte = 0.0
    for i in range(total_e):
        e = e_index[i]
        inte += (
            elen[p, e] * ewid[p, e] * (f[ni[e]] + f[nj[e]] + f[nk[e]] + f[nl[e]]) * 0.25
        )
    return inte


@njit(cache=True, fastmath=False)
def pert_press_assemble_all_jit(
    p,
    total_e_reynolds,
    e_index_reynolds,
    node_i_reynolds,
    node_j_reynolds,
    node_k_reynolds,
    node_l_reynolds,
    gamma,
    h_n,
    e_length_reynolds,
    e_width_reynolds,
    q_per_element,
    oil_seal,
    bandwidth_reynolds,
    global_matrix_p,
    global_column_p,
):
    """Assemble every perturbation-Reynolds element into the 0-based banded
    global system in a single JIT call.

    Fuses ``element_press`` (closed-form Allaire stencil) and the 0-based
    ``assemble_press`` for the K/C perturbation source. ``q_per_element`` is
    a length-``total_e_reynolds`` array of pre-computed source values (the
    Python callback in ``_solve_pert_pressure`` runs once per element to
    fill it before this kernel runs). All arrays uniformly 0-based natural
    (0-based node/element values); ``p`` is already 0-based.
    ``global_matrix_p`` / ``global_column_p`` are 0-based-shaped
    (``(dim_xz, dim_xz)`` / ``(dim_xz,)``); mutated in place.
    """
    for i in range(total_e_reynolds):
        ce = e_index_reynolds[i]
        ni = node_i_reynolds[ce]
        nj = node_j_reynolds[ce]
        nk = node_k_reynolds[ce]
        nl = node_l_reynolds[ce]

        h_e = (h_n[p, ni] + h_n[p, nj] + h_n[p, nk] + h_n[p, nl]) * 0.25
        gamma_e = (gamma[ni] + gamma[nj] + gamma[nk] + gamma[nl]) * 0.25
        l_e = e_length_reynolds[p, ce]
        w_e = e_width_reynolds[p, ce]

        k_x = h_e * h_e * h_e * gamma_e
        k_z = k_x
        if oil_seal:
            k_x = 0.0
        q = q_per_element[i]

        a11 = (k_x * w_e) / (3.0 * l_e) + (k_z * l_e) / (3.0 * w_e)
        a12 = -(k_x * w_e) / (3.0 * l_e) + (k_z * l_e) / (6.0 * w_e)
        a13 = -(k_x * w_e) / (6.0 * l_e) - (k_z * l_e) / (6.0 * w_e)
        a14 = (k_x * w_e) / (6.0 * l_e) - (k_z * l_e) / (3.0 * w_e)
        rhs = q * l_e * w_e * 0.25

        nodes = (ni, nj, nk, nl)
        em = np.empty(16, dtype=np.float64)
        em[0] = a11
        em[1] = a12
        em[2] = a13
        em[3] = a14
        em[4] = a12
        em[5] = a11
        em[6] = a14
        em[7] = a13
        em[8] = a13
        em[9] = a14
        em[10] = a11
        em[11] = a12
        em[12] = a14
        em[13] = a13
        em[14] = a12
        em[15] = a11

        for ii in range(4):
            irow = nodes[ii]
            for jj in range(4):
                icol = nodes[jj]
                jcol = icol - irow + bandwidth_reynolds - 1
                global_matrix_p[irow, jcol] += em[ii * 4 + jj]
            global_column_p[irow] += rhs


@njit(cache=True, fastmath=False)
def press_assemble_all_jit(
    p,
    total_e_reynolds,
    e_index_reynolds,
    node_i_reynolds,
    node_j_reynolds,
    node_k_reynolds,
    node_l_reynolds,
    gamma,
    g,
    h_n,
    dx_reynolds,
    e_length_reynolds,
    e_width_reynolds,
    speed_surface,
    is_360,
    bandwidth_reynolds,
    global_matrix_p,
    global_column_p,
):
    """Assemble every Reynolds element into the banded global system in one
    JIT call.

    Fuses :func:`pressure.element_press` and :func:`pressure.assemble_press`
    over all ``total_e_reynolds`` elements, eliminating ~14k per-element
    Python<->JIT boundary crossings per :func:`pressure.press` call.
    All arrays 0-based natural; ``p`` is 0-based; ``e_index_reynolds`` and the
    ``node_*`` connectivity store 0-based ids. The band diagonal sits at
    column ``bandwidth_reynolds - 1``. Mutates ``global_matrix_p`` /
    ``global_column_p`` in place.
    """
    for ie in range(total_e_reynolds):
        ce = e_index_reynolds[ie]
        ni = node_i_reynolds[ce]
        nj = node_j_reynolds[ce]
        nk = node_k_reynolds[ce]
        nl = node_l_reynolds[ce]

        gamma_e = (gamma[ni] + gamma[nj] + gamma[nk] + gamma[nl]) * 0.25
        g_e = (g[ni] + g[nj] + g[nk] + g[nl]) * 0.25
        h_e = (h_n[p, ni] + h_n[p, nj] + h_n[p, nk] + h_n[p, nl]) * 0.25
        dhdx_e = (
            dx_reynolds[p, ce, 0] * h_n[p, ni]
            + dx_reynolds[p, ce, 1] * h_n[p, nj]
            + dx_reynolds[p, ce, 2] * h_n[p, nk]
            + dx_reynolds[p, ce, 3] * h_n[p, nl]
        )
        l_e = e_length_reynolds[p, ce]
        w_e = e_width_reynolds[p, ce]

        k_x = h_e * h_e * h_e * gamma_e
        k_z = k_x
        q = speed_surface * g_e * dhdx_e
        if is_360:
            k_x = 0.0

        # element_press inlined.
        a11 = (k_x * w_e) / (3.0 * l_e) + (k_z * l_e) / (3.0 * w_e)
        a12 = -(k_x * w_e) / (3.0 * l_e) + (k_z * l_e) / (6.0 * w_e)
        a13 = -(k_x * w_e) / (6.0 * l_e) - (k_z * l_e) / (6.0 * w_e)
        a14 = (k_x * w_e) / (6.0 * l_e) - (k_z * l_e) / (3.0 * w_e)
        rhs = q * l_e * w_e * 0.25
        # e_matrix laid out per the closed-form Allaire stencil; assembled
        # directly into the band rather than via an intermediate 4x4 buffer.
        # Row/col mapping (1-based corner -> nodal id):
        nodes = (ni, nj, nk, nl)
        # The closed-form e_matrix has the symmetric block pattern
        #     [a11 a12 a13 a14]
        #     [a12 a11 a14 a13]
        #     [a13 a14 a11 a12]
        #     [a14 a13 a12 a11]
        # so we precompute a flattened 4x4 to read by [i, j].
        em = np.empty(16, dtype=np.float64)
        em[0] = a11
        em[1] = a12
        em[2] = a13
        em[3] = a14
        em[4] = a12
        em[5] = a11
        em[6] = a14
        em[7] = a13
        em[8] = a13
        em[9] = a14
        em[10] = a11
        em[11] = a12
        em[12] = a14
        em[13] = a13
        em[14] = a12
        em[15] = a11

        for i in range(4):
            irow = nodes[i]
            for j in range(4):
                icol = nodes[j]
                jcol = icol - irow + bandwidth_reynolds - 1
                global_matrix_p[irow, jcol] += em[i * 4 + j]
            global_column_p[irow] += rhs


@njit(cache=True, fastmath=False)
def t_outlet_jit(
    total_pads,
    total_e_x_film,
    total_e_y_film,
    total_e_z_film,
    total_e_y_trackbl,
    total_e_y_trackcore,
    total_n_reynolds,
    n_index_reynolds,
    total_e_reynolds,
    e_index_reynolds,
    node_j_reynolds,
    node_k_reynolds,
    match_nodes_xz,
    pad_thickness,
    y_3d,
    length_track,
    depth_track,
    axial_length,
    axial_length_dam,
    axial_length_track,
    pad_length,
    x_reynolds,
    z_reynolds,
    h_n,
    temp_3d,
    e_width_reynolds,
    velocity_x_n,
    q_x,
    dim_xz,
    dim_yf,
):
    """Inner body of :func:`thd.t_outlet`.

    All arrays 0-based (thd convention); ``pad`` loops over [0, total_pads).
    Returns ``(temp_outlet, temp_outlet_bulk)``, both length ``total_pads``.
    """
    temp_outlet = np.zeros(total_pads, dtype=np.float64)
    temp_outlet_bulk = np.zeros(total_pads, dtype=np.float64)

    temp_n_average = np.zeros(dim_xz, dtype=np.float64)
    temp_n_average_bulk = np.zeros(dim_xz, dtype=np.float64)
    t = np.zeros(dim_yf, dtype=np.float64)
    f1 = np.zeros(dim_yf, dtype=np.float64)
    f2 = np.zeros(dim_yf, dtype=np.float64)

    for pad in range(total_pads):
        limit1 = total_e_y_trackbl[pad] + total_e_y_trackcore[pad] + 1
        limit2 = total_e_y_film + 1

        for i in range(dim_xz):
            temp_n_average[i] = 0.0
            temp_n_average_bulk[i] = 0.0

        for i in range(total_n_reynolds):
            node = n_index_reynolds[i]
            region = node_region_jit(
                pad,
                node,
                x_reynolds,
                z_reynolds,
                pad_length,
                axial_length,
                axial_length_track,
                axial_length_dam,
                length_track,
            )
            if region == 0:  # pocket
                lo = 1
                offset = pad_thickness
            else:
                lo = limit1
                offset = pad_thickness + depth_track[pad]

            for j in range(lo, limit2 + 1):
                m = match_nodes_xz[node, j - 1]
                t[j - 1] = y_3d[pad, m] - offset
                f1[j - 1] = temp_3d[pad, m]
                f2[j - 1] = temp_3d[pad, m] * velocity_x_n[pad, m]

            temp_n_average[node] = trapezoid_jit(t, f1, lo - 1, limit2) / h_n[pad, node]
            temp_n_average_bulk[node] = trapezoid_jit(t, f2, lo - 1, limit2)

        # Sum over the trailing-edge element row.
        sum1 = 0.0
        sum2 = 0.0
        trailing_edge_area = 0.0
        for i in range(total_e_reynolds - total_e_z_film, total_e_reynolds):
            elem = e_index_reynolds[i]
            node_j = node_j_reynolds[elem]
            node_k = node_k_reynolds[elem]
            width = e_width_reynolds[pad, elem]
            half_h = 0.5 * (h_n[pad, node_j] + h_n[pad, node_k])

            trailing_edge_area += width * half_h
            sum1 += (
                0.5 * (temp_n_average[node_j] + temp_n_average[node_k]) * width * half_h
            )
            sum2 += (
                0.5
                * (temp_n_average_bulk[node_j] + temp_n_average_bulk[node_k])
                * width
            )

        temp_outlet[pad] = sum1 / trailing_edge_area
        temp_outlet_bulk[pad] = sum2 / q_x[pad, total_e_x_film]

    return temp_outlet, temp_outlet_bulk


@njit(cache=True, fastmath=False)
def deform_assemble_all_jit(
    p,
    total_e_pad,
    e_index_pad,
    node_1_pad,
    node_2_pad,
    node_3_pad,
    node_4_pad,
    x_pad,
    y_pad,
    pad_temp,
    temp_ref,
    young,
    poisson,
    pad_expand,
    bandwidth_deform,
    global_matrix_d,
    global_column_d,
):
    """Assemble every pad element into the banded deformation system in one
    JIT call.

    Fuses :func:`deform.integrand_e_deform` (2x2 Gauss),
    :func:`deform.element_deform` and :func:`deform.assemble_pad` over all
    ``total_e_pad`` elements, eliminating ~16k per-element/per-Gauss-point
    Python<->JIT crossings per :func:`deform.deformation` call. All arrays are
    0-based natural; ``p`` is the 0-based pad row and the connectivity stores
    0-based node ids. DOFs are ``2*node`` (x) and ``2*node + 1`` (y); the band
    diagonal sits at column ``bandwidth_deform - 1``. Mutates
    ``global_matrix_d`` / ``global_column_d`` in place.
    """
    # Plane-strain material matrix -- invariant across elements and Gauss
    # points, so it is built once rather than per integrand evaluation.
    alpha = young / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    e_mat = np.zeros((3, 3), dtype=np.float64)
    e_mat[0, 0] = alpha * (1.0 - poisson)
    e_mat[0, 1] = alpha * poisson
    e_mat[1, 0] = alpha * poisson
    e_mat[1, 1] = alpha * (1.0 - poisson)
    e_mat[2, 2] = alpha * (1.0 - 2.0 * poisson) * 0.5

    g = 1.0 / np.sqrt(3.0)
    gauss_r = np.array([-g, g, -g, g], dtype=np.float64)
    gauss_s = np.array([-g, -g, g, g], dtype=np.float64)

    # Scratch reused across elements. ``dn``'s zero entries never change, so
    # only its 16 populated slots are rewritten per Gauss point.
    global_coord = np.zeros((2, 4), dtype=np.float64)
    f_deriv = np.zeros((2, 4), dtype=np.float64)
    b = np.zeros((2, 4), dtype=np.float64)
    dn = np.zeros((3, 8), dtype=np.float64)
    dn_e = np.zeros((8, 3), dtype=np.float64)
    ee0 = np.zeros(3, dtype=np.float64)
    e_matrix_pad = np.zeros((8, 8), dtype=np.float64)
    e_column_pad = np.zeros(8, dtype=np.float64)

    for ie in range(total_e_pad):
        ce = e_index_pad[ie]
        n1 = node_1_pad[ce]
        n2 = node_2_pad[ce]
        n3 = node_3_pad[ce]
        n4 = node_4_pad[ce]

        global_coord[0, 0] = x_pad[p, n1]
        global_coord[0, 1] = x_pad[p, n2]
        global_coord[0, 2] = x_pad[p, n3]
        global_coord[0, 3] = x_pad[p, n4]
        global_coord[1, 0] = y_pad[n1]
        global_coord[1, 1] = y_pad[n2]
        global_coord[1, 2] = y_pad[n3]
        global_coord[1, 3] = y_pad[n4]

        delta_t_e = (
            pad_temp[p, n1] + pad_temp[p, n2] + pad_temp[p, n3] + pad_temp[p, n4]
        ) / 4.0 - temp_ref

        # e_mat @ epsilon0 with epsilon0 = [pad_expand*dT, pad_expand*dT, 0].
        eps = pad_expand * delta_t_e
        ee0[0] = e_mat[0, 0] * eps + e_mat[0, 1] * eps
        ee0[1] = e_mat[1, 0] * eps + e_mat[1, 1] * eps
        ee0[2] = 0.0

        for i in range(8):
            e_column_pad[i] = 0.0
            for j in range(8):
                e_matrix_pad[i, j] = 0.0

        for gp in range(4):
            r = gauss_r[gp]
            s = gauss_s[gp]

            f_deriv[0, 0] = -(1.0 - s) / 4.0
            f_deriv[0, 1] = (1.0 - s) / 4.0
            f_deriv[0, 2] = (1.0 + s) / 4.0
            f_deriv[0, 3] = -(1.0 + s) / 4.0
            f_deriv[1, 0] = -(1.0 - r) / 4.0
            f_deriv[1, 1] = -(1.0 + r) / 4.0
            f_deriv[1, 2] = (1.0 + r) / 4.0
            f_deriv[1, 3] = (1.0 - r) / 4.0

            # jac[i, j] = sum_k f_deriv[i, k] * global_coord[j, k]
            jac00 = 0.0
            jac01 = 0.0
            jac10 = 0.0
            jac11 = 0.0
            for k in range(4):
                jac00 += f_deriv[0, k] * global_coord[0, k]
                jac01 += f_deriv[0, k] * global_coord[1, k]
                jac10 += f_deriv[1, k] * global_coord[0, k]
                jac11 += f_deriv[1, k] * global_coord[1, k]
            det_j = jac00 * jac11 - jac01 * jac10

            ji00 = jac11 / det_j
            ji01 = -jac01 / det_j
            ji10 = -jac10 / det_j
            ji11 = jac00 / det_j

            for k in range(4):
                b[0, k] = ji00 * f_deriv[0, k] + ji01 * f_deriv[1, k]
                b[1, k] = ji10 * f_deriv[0, k] + ji11 * f_deriv[1, k]

            for k in range(4):
                dn[0, 2 * k] = b[0, k]
                dn[1, 2 * k + 1] = b[1, k]
                dn[2, 2 * k] = b[1, k]
                dn[2, 2 * k + 1] = b[0, k]

            # dn_e = dn.T @ e_mat  (8x3)
            for i in range(8):
                for j in range(3):
                    acc = 0.0
                    for k in range(3):
                        acc += dn[k, i] * e_mat[k, j]
                    dn_e[i, j] = acc

            # e_matrix_pad += (dn_e @ dn) * det_j ; e_column_pad += (dn.T @ ee0) * det_j
            for i in range(8):
                for j in range(8):
                    acc = 0.0
                    for k in range(3):
                        acc += dn_e[i, k] * dn[k, j]
                    e_matrix_pad[i, j] += acc * det_j
                acc = 0.0
                for k in range(3):
                    acc += dn[k, i] * ee0[k]
                e_column_pad[i] += acc * det_j

        # Assemble into the band (DOF x = 2*node, y = 2*node + 1).
        nodes = (n1, n2, n3, n4)
        for i in range(4):
            ix = 2 * i
            iy = ix + 1
            irow_x = nodes[i] * 2
            irow_y = irow_x + 1
            for j in range(4):
                jx = 2 * j
                jy = jx + 1
                icol_x = nodes[j] * 2
                jcol_xx = icol_x - irow_x + (bandwidth_deform - 1)
                global_matrix_d[irow_x, jcol_xx] += e_matrix_pad[ix, jx]
                global_matrix_d[irow_y, jcol_xx] += e_matrix_pad[iy, jy]
                global_matrix_d[irow_x, jcol_xx + 1] += e_matrix_pad[ix, jy]
                global_matrix_d[irow_y, jcol_xx - 1] += e_matrix_pad[iy, jx]
            global_column_d[irow_x] += e_column_pad[ix]
            global_column_d[irow_y] += e_column_pad[iy]
    return global_matrix_d, global_column_d


@njit(cache=True, fastmath=False)
def element_press_jit(k_x, k_z, q, l_e, w_e):
    """Q4 element matrix/column of the Reynolds equation.

    Mirrors :func:`ross.bearings.fluid_film.pressure.element_press` (closed-form
    Allaire Q4 stencil). Pure scalar arithmetic — JIT cuts ~150k-call
    overhead in the M2 hot loop.
    """
    a11 = (k_x * w_e) / (3.0 * l_e) + (k_z * l_e) / (3.0 * w_e)
    a12 = -(k_x * w_e) / (3.0 * l_e) + (k_z * l_e) / (6.0 * w_e)
    a13 = -(k_x * w_e) / (6.0 * l_e) - (k_z * l_e) / (6.0 * w_e)
    a14 = (k_x * w_e) / (6.0 * l_e) - (k_z * l_e) / (3.0 * w_e)

    e_matrix = np.empty((4, 4), dtype=np.float64)
    e_matrix[0, 0] = a11
    e_matrix[0, 1] = a12
    e_matrix[0, 2] = a13
    e_matrix[0, 3] = a14
    e_matrix[1, 0] = a12
    e_matrix[1, 1] = a11
    e_matrix[1, 2] = a14
    e_matrix[1, 3] = a13
    e_matrix[2, 0] = a13
    e_matrix[2, 1] = a14
    e_matrix[2, 2] = a11
    e_matrix[2, 3] = a12
    e_matrix[3, 0] = a14
    e_matrix[3, 1] = a13
    e_matrix[3, 2] = a12
    e_matrix[3, 3] = a11

    e_column = np.empty(4, dtype=np.float64)
    rhs = q * l_e * w_e * 0.25
    e_column[0] = rhs
    e_column[1] = rhs
    e_column[2] = rhs
    e_column[3] = rhs
    return e_matrix, e_column


@njit(cache=True, fastmath=False)
def _film_element_heights_jit(
    pad,
    depth_track,
    thickness_bl,
    h_n1,
    total_e_y_trackbl,
    total_e_y_trackcore,
    total_e_y_dambl,
    total_e_y_damcore,
):
    """Numba-friendly translation of :func:`mesh._film_element_heights`."""
    if abs(depth_track[pad]) < 1.0e-6:
        if thickness_bl < 1.0e-8:
            e_track_bl = 0.0
            e_track_core = 0.0
            e_dam_bl = 0.0
            e_dam_core = h_n1 / total_e_y_damcore[pad]
        else:
            e_track_bl = 0.0
            e_track_core = 0.0
            e_dam_bl = thickness_bl / total_e_y_dambl[pad]
            e_dam_core = (h_n1 - 2.0 * thickness_bl) / total_e_y_damcore[pad]
    else:
        if thickness_bl < 1.0e-8:
            e_track_bl = 0.0
            e_track_core = depth_track[pad] / total_e_y_trackcore[pad]
            e_dam_bl = 0.0
            e_dam_core = h_n1 / total_e_y_damcore[pad]
        else:
            e_track_bl = thickness_bl / total_e_y_trackbl[pad]
            e_track_core = (depth_track[pad] - thickness_bl) / total_e_y_trackcore[pad]
            e_dam_bl = thickness_bl / total_e_y_dambl[pad]
            e_dam_core = (h_n1 - 2.0 * thickness_bl) / total_e_y_damcore[pad]
    return e_track_bl, e_track_core, e_dam_bl, e_dam_core


@njit(cache=True, fastmath=False)
def mesh_3d_jit(
    total_pads,
    total_e_x_film,
    total_e_y_film,
    total_e_z_film,
    total_e_x_track,
    total_e_z_track,
    total_e_x_dam,
    total_e_z_dam,
    total_e_y_trackbl,
    total_e_y_trackcore,
    total_e_y_dambl,
    total_e_y_damcore,
    axial_length,
    pad_thickness,
    leading_angle_rad,
    cp,
    arc_length_rad,
    pad_length,
    offset,
    preload,
    length_track,
    length_track_rad,
    depth_track,
    length_dam,
    axial_length_track,
    axial_length_dam,
    xj,
    yj,
    total_n_reynolds,
    n_index_reynolds,
    x_reynolds,
    z_reynolds,
    total_n_energy,
    n_index_energy,
    x_energy,
    y_energy,
    weight_h,
    n_index_3d,
    x_3d,
    y_3d,
    z_3d,
    match_nodes_xz,
    match_nodes_xy,
):
    """Core of :func:`ross.bearings.fluid_film.mesh.mesh_3d`: triple-nested 3-D node
    construction plus the two matching loops (xz <-> 3d, xy <-> 3d).

    All arrays are 0-based natural: ``pad`` is a 0-based pad index; the
    per-pad / per-node input arrays are indexed directly; ``n_index_*`` store
    0-based node ids; ``match_nodes_*`` store 0-based 3-D node ids with ``-1``
    marking unused fill slots (initialised by the caller). All output arrays
    are pre-allocated and mutated in place.
    """
    for pad in range(total_pads):
        if abs(depth_track[pad]) < 1.0e-6:
            dx_track = 0.0
            dx_track_rad = 0.0
            dz_track = 0.0
            dz_edge = 0.0
        else:
            dx_track = (length_track[pad] - 0.005 * pad_length[pad]) / (
                total_e_x_track[pad] - 1
            )
            dx_track_rad = (length_track_rad[pad] - 0.005 * arc_length_rad[pad]) / (
                total_e_x_track[pad] - 1
            )
            dz_track = (axial_length_track[pad] - 0.01 * axial_length[pad]) / (
                total_e_z_track[pad] - 2
            )
            dz_edge = 0.005 * axial_length[pad]
        dx_dam = length_dam[pad] / total_e_x_dam[pad]
        dx_dam_rad = (arc_length_rad[pad] - length_track_rad[pad]) / total_e_x_dam[pad]
        if total_e_z_dam[pad] == 0:
            dz_dam = 0.0
        else:
            dz_dam = axial_length_dam[pad] / total_e_z_dam[pad]

        n = 0
        for i in range(1, total_e_x_film + 1 + 1):
            if i <= total_e_x_track[pad]:
                x1 = (i - 1) * dx_track
                x2 = (i - 1) * dx_track_rad
            elif i == total_e_x_track[pad] + 1:
                x1 = length_track[pad]
                x2 = length_track_rad[pad]
            else:
                x1 = length_track[pad] + (i - total_e_x_track[pad] - 1) * dx_dam
                x2 = length_track_rad[pad] + (i - total_e_x_track[pad] - 1) * dx_dam_rad

            h_n1 = (
                cp[pad]
                - xj * np.cos(leading_angle_rad[pad] + x2)
                - yj * np.sin(leading_angle_rad[pad] + x2)
                - preload[pad]
                * cp[pad]
                * np.cos(x2 - offset[pad] * arc_length_rad[pad])
            )
            thickness_bl = weight_h * h_n1

            (
                e_film_track_bl,
                e_film_track_core,
                e_film_dam_bl,
                e_film_dam_core,
            ) = _film_element_heights_jit(
                pad,
                depth_track,
                thickness_bl,
                h_n1,
                total_e_y_trackbl,
                total_e_y_trackcore,
                total_e_y_dambl,
                total_e_y_damcore,
            )

            tbl = total_e_y_trackbl[pad]
            tcore = total_e_y_trackcore[pad]
            dbl = total_e_y_dambl[pad]
            dcore = total_e_y_damcore[pad]

            for j in range(1, total_e_y_film + 1 + 1):
                if j <= tbl + 1:
                    y1 = pad_thickness + (j - 1) * e_film_track_bl
                elif j > tbl + 1 and j <= tbl + tcore + 1:
                    y1 = (
                        pad_thickness + thickness_bl + (j - tbl - 1) * e_film_track_core
                    )
                elif j > tbl + tcore + 1 and j <= tbl + tcore + dbl + 1:
                    y1 = (
                        pad_thickness
                        + depth_track[pad]
                        + (j - tbl - tcore - 1) * e_film_dam_bl
                    )
                elif j > tbl + tcore + dbl + 1 and j <= tbl + tcore + dbl + dcore + 1:
                    y1 = (
                        pad_thickness
                        + depth_track[pad]
                        + thickness_bl
                        + (j - tbl - tcore - dbl - 1) * e_film_dam_core
                    )
                else:
                    y1 = (
                        pad_thickness
                        + depth_track[pad]
                        + h_n1
                        - thickness_bl
                        + (j - tbl - tcore - dbl - dcore - 1) * e_film_dam_bl
                    )

                ezt = total_e_z_track[pad]
                ezd = total_e_z_dam[pad]
                for k in range(1, total_e_z_film + 1 + 1):
                    if k <= ezd + 1:
                        z1 = (k - 1) * dz_dam
                    elif k == ezd + 2 and k < ezd + ezt:
                        z1 = axial_length_dam[pad] + dz_edge
                    elif k > ezd + 2 and k <= ezd + ezt:
                        z1 = axial_length_dam[pad] + dz_edge + (k - ezd - 2) * dz_track
                    elif k == ezd + ezt + 1 and k > ezd + 2:
                        z1 = axial_length_dam[pad] + axial_length_track[pad]
                    else:
                        z1 = (
                            axial_length_dam[pad]
                            + axial_length_track[pad]
                            + (k - ezd - ezt - 1) * dz_dam
                        )
                    n_index_3d[n] = n
                    x_3d[pad, n_index_3d[n]] = x1
                    y_3d[pad, n_index_3d[n]] = y1
                    z_3d[pad, n_index_3d[n]] = z1
                    n += 1

    # Matching loop 1: Reynolds xz nodes -> 3D nodes (same x, z; uses pad 0).
    total_n_3d = (total_e_x_film + 1) * (total_e_y_film + 1) * (total_e_z_film + 1)
    for i in range(total_n_reynolds):
        nri = n_index_reynolds[i]
        xri = x_reynolds[0, nri]
        zri = z_reynolds[0, nri]
        kc = 0
        for jj in range(total_n_3d):
            n3 = n_index_3d[jj]
            if abs(x_3d[0, n3] - xri) < 1.0e-6 and abs(z_3d[0, n3] - zri) < 1.0e-6:
                match_nodes_xz[nri, kc] = n3
                kc += 1

    # Matching loop 2: energy xy nodes -> 3D nodes (same x, y; uses pad 0).
    for i in range(total_n_energy):
        nei = n_index_energy[i]
        yei = y_energy[0, nei]
        if yei > pad_thickness or abs(yei - pad_thickness) < 1.0e-6:
            xei = x_energy[0, nei]
            kc = 0
            for jj in range(total_n_3d):
                n3 = n_index_3d[jj]
                if abs(x_3d[0, n3] - xei) < 1.0e-6 and abs(y_3d[0, n3] - yei) < 1.0e-6:
                    match_nodes_xy[nei, kc] = n3
                    kc += 1


@njit(cache=True, fastmath=False)
def stiffness_source_terms_jit(
    p,
    current_element,
    ni,
    nj,
    nk,
    nl,
    gamma,
    g_e,
    speed_surface,
    dx_reynolds,
    dz_reynolds,
    x_reynolds_rad,
    h_n,
    dpdx_n,
    dpdz_n,
    angle_kind,
    scale,
    angle_offset_0,
    angle_offset_1,
    angle_offset_2,
    angle_offset_3,
):
    """JIT-friendly version of :func:`coefficients._stiffness_source_terms`.

    ``angle_kind`` is 0 for ``cos`` (K1) and 1 for ``sin`` (K2/K3); the four
    per-corner ``angle_offset_*`` values are the (typically identical) angle
    offsets for the four local nodes. Fully 0-based natural (``p`` 0-based,
    ``ni``..``nl`` and ``current_element`` 0-based values).
    """
    nodes_arr = (ni, nj, nk, nl)
    angle_offsets = (
        angle_offset_0,
        angle_offset_1,
        angle_offset_2,
        angle_offset_3,
    )

    term_i = 0.0
    term_ii = 0.0
    term_iii_sum = 0.0
    for k in range(4):
        node = nodes_arr[k]
        theta = angle_offsets[k] + x_reynolds_rad[p, node]
        if angle_kind == 0:
            ang = np.cos(theta)
        else:
            ang = np.sin(theta)
        h2 = h_n[p, node] * h_n[p, node]
        gam = gamma[node]
        dx = dx_reynolds[p, current_element, k]
        dz = dz_reynolds[p, current_element, k]
        term_i += dx * ang * h2 * gam * dpdx_n[p, node]
        term_ii += dz * ang * h2 * gam * dpdz_n[p, node]
        term_iii_sum += dx * ang

    term_i_e = 3.0 * scale * term_i
    term_ii_e = 3.0 * scale * term_ii
    term_iii_e = speed_surface * g_e * scale * term_iii_sum
    return term_i_e, term_ii_e, term_iii_e


@njit(cache=True, fastmath=False)
def stiffness_source_all_jit(
    p,
    elem,
    ni_arr,
    nj_arr,
    nk_arr,
    nl_arr,
    gamma,
    g_e_arr,
    speed_surface,
    dx_reynolds,
    dz_reynolds,
    x_reynolds_rad,
    h_n,
    dpdx_n,
    dpdz_n,
    angle_kind,
    scale,
    angle_offset,
    oil_seal,
):
    """Fused ``stiffness_source_terms_jit`` over every Reynolds element.

    Computes the perturbation source ``Q`` for all elements of one pad in a
    single JIT call (element-by-element arithmetic identical to
    :func:`stiffness_source_terms_jit`, so results are bit-identical).
    ``elem`` / ``ni_arr``..``nl_arr`` are the pre-gathered element ids and
    local node ids (0-based); ``g_e_arr`` the per-element nodal average of
    ``g``. ``angle_offset`` is the single scalar offset shared by the four
    local nodes (``leading_angle`` for K1/K2, ``-x_pivot`` for K3). For the
    oil-seal short-bearing case Term I is dropped.
    """
    n_e = elem.shape[0]
    q = np.empty(n_e, dtype=np.float64)
    for i in range(n_e):
        current_element = elem[i]
        term_i = 0.0
        term_ii = 0.0
        term_iii_sum = 0.0
        for k in range(4):
            if k == 0:
                node = ni_arr[i]
            elif k == 1:
                node = nj_arr[i]
            elif k == 2:
                node = nk_arr[i]
            else:
                node = nl_arr[i]
            theta = angle_offset + x_reynolds_rad[p, node]
            if angle_kind == 0:
                ang = np.cos(theta)
            else:
                ang = np.sin(theta)
            h2 = h_n[p, node] * h_n[p, node]
            gam = gamma[node]
            dx = dx_reynolds[p, current_element, k]
            dz = dz_reynolds[p, current_element, k]
            term_i += dx * ang * h2 * gam * dpdx_n[p, node]
            term_ii += dz * ang * h2 * gam * dpdz_n[p, node]
            term_iii_sum += dx * ang

        term_i_e = 3.0 * scale * term_i
        term_ii_e = 3.0 * scale * term_ii
        term_iii_e = speed_surface * g_e_arr[i] * scale * term_iii_sum
        if oil_seal:
            q[i] = -term_ii_e - term_iii_e
        else:
            q[i] = -term_i_e - term_ii_e - term_iii_e
    return q


@njit(cache=True, fastmath=False)
def velocity_jit(
    total_pads,
    dim_yf,
    total_e_y_film,
    total_e_y_trackbl,
    total_e_y_trackcore,
    total_n_reynolds,
    n_index_reynolds,
    match_nodes_xz,
    pad_length,
    axial_length,
    pad_thickness,
    length_track,
    depth_track,
    axial_length_dam,
    axial_length_track,
    dpdx_n,
    dpdz_n,
    speed_surface,
    x_reynolds,
    z_reynolds,
    h_n,
    dhdx_n,
    vis_effect_3d,
    y_3d,
    velocity_x_n,
    velocity_y_n,
    velocity_z_n,
):
    """Inner body of :func:`hydrodynamics.velocity` (all three components).

    0-based natural arrays throughout: ``pad_index``/``node`` are 0-based values
    used directly as indices; ``n_index_reynolds`` stores 0-based node ids;
    ``match_nodes_xz`` stores 0-based 3-D ids and is indexed ``[node, j - 1]``
    (cross-film column ``j`` is a 1-based position counter into the local scratch
    buffers ``t``/``f``). Mutates ``velocity_*_n`` in place.
    """
    t = np.zeros(dim_yf + 1, dtype=np.float64)
    f1 = np.zeros(dim_yf + 1, dtype=np.float64)
    f2 = np.zeros(dim_yf + 1, dtype=np.float64)

    for pad_index in range(total_pads):
        limit1 = total_e_y_trackbl[pad_index] + total_e_y_trackcore[pad_index] + 1
        limit2 = total_e_y_film + 1

        for i in range(total_n_reynolds):
            node = n_index_reynolds[i]
            region = node_region_jit(
                pad_index,
                node,
                x_reynolds,
                z_reynolds,
                pad_length,
                axial_length,
                axial_length_track,
                axial_length_dam,
                length_track,
            )
            if region == 0:
                jlo = 1
                offset = pad_thickness
            else:
                jlo = limit1
                offset = pad_thickness + depth_track[pad_index]

            for j in range(jlo, limit2 + 1):
                m = match_nodes_xz[node, j - 1]
                t[j] = y_3d[pad_index, m] - offset
                f1[j] = 1.0 / vis_effect_3d[pad_index, m]
                f2[j] = t[j] / vis_effect_3d[pad_index, m]

            # xi1h, xi2h: integral over [jlo, limit2].
            xi1h = 0.0
            xi2h = 0.0
            for ii in range(jlo, limit2):
                h = t[ii + 1] - t[ii]
                xi1h += 0.5 * h * (f1[ii + 1] + f1[ii])
                xi2h += 0.5 * h * (f2[ii + 1] + f2[ii])

            # Running cumulative xi1, xi2 from jlo up to j.
            xi2h_over_xi1h = xi2h / xi1h
            speed_over_xi1h = speed_surface / xi1h
            cum_xi1 = 0.0
            cum_xi2 = 0.0
            # j == jlo: integral from jlo to jlo is zero.
            m = match_nodes_xz[node, jlo - 1]
            velocity_x_n[pad_index, m] = 0.0
            velocity_z_n[pad_index, m] = 0.0
            velocity_y_n[pad_index, m] = (
                ((y_3d[pad_index, m] - pad_thickness) / h_n[pad_index, node]) ** 2
                * speed_surface
                * dhdx_n[pad_index, node]
            )
            for j in range(jlo + 1, limit2 + 1):
                h = t[j] - t[j - 1]
                cum_xi1 += 0.5 * h * (f1[j] + f1[j - 1])
                cum_xi2 += 0.5 * h * (f2[j] + f2[j - 1])
                m = match_nodes_xz[node, j - 1]
                dpdx_node = dpdx_n[pad_index, node]
                dpdz_node = dpdz_n[pad_index, node]
                velocity_x_n[pad_index, m] = (
                    dpdx_node * cum_xi2
                    + (speed_over_xi1h - dpdx_node * xi2h_over_xi1h) * cum_xi1
                )
                velocity_z_n[pad_index, m] = (
                    dpdz_node * cum_xi2 - dpdz_node * xi2h_over_xi1h * cum_xi1
                )
                velocity_y_n[pad_index, m] = (
                    ((y_3d[pad_index, m] - pad_thickness) / h_n[pad_index, node]) ** 2
                    * speed_surface
                    * dhdx_n[pad_index, node]
                )
    return velocity_x_n, velocity_y_n, velocity_z_n


@njit(cache=True, fastmath=False)
def expand_film_temp_flooded_jit(
    pad,
    total_n_energy,
    n_index_energy,
    match_nodes_xy,
    total_e_z_film,
    y_energy,
    pad_thickness,
    temp_full,
    temp_3d,
):
    """Inner body of :func:`thermal.expand_film_temp_flooded` (constant axial profile)."""
    for i in range(total_n_energy):
        node = n_index_energy[i]
        y = y_energy[pad, node]
        if y > pad_thickness or abs(y - pad_thickness) < 1.0e-6:
            for j in range(1, total_e_z_film + 1 + 1):
                m = match_nodes_xy[node, j - 1]
                temp_3d[pad, m] = temp_full[pad, node]


@njit(cache=True, fastmath=False)
def film_thickness_jit(
    p,
    total_e_z_film,
    pad_thickness,
    leading_angle_rad_p,
    cp_new_p,
    pad_length_p,
    preload_new_p,
    x_pivot_rad_p,
    depth_track_p,
    journal_radius,
    cb,
    tilt_angle_p,
    xj,
    yj,
    dh_ramp_le_p,
    length_ramp_le_p,
    dh_ramp_te_p,
    length_ramp_te_p,
    total_n_reynolds,
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
    operating_type,
    bearing_type,
    unloaded_p,
    dim_xz,
):
    """Inner body of :func:`hydrodynamics.film_thickness`.

    All arrays 0-based natural; ``p`` and the stored node ids are 0-based.
    Mutates ``h_n`` / ``dhdx_n`` / ``y_energy`` / ``y_3d`` in place. Returns
    ``(h_min, x_hmin, full_cavitate_int, h_ns)`` where ``full_cavitate_int`` is
    1 if the pad fully cavitates.
    """
    h_ns = np.zeros(dim_xz, dtype=np.float64)
    radius = journal_radius + cb + pad_thickness
    special = (
        operating_type == "axial_flow" or operating_type == "high_ambient_pressure"
    ) and (
        bearing_type == "conventional_tilting_pad"
        or bearing_type == "inlet_groove_tilting_pad"
        or bearing_type == "spray_bar_tilting_pad"
    )

    h_min = 0.0
    x_hmin = 0.0

    # Per-node loop: film thickness + taper + h_min tracking.
    for i in range(total_n_reynolds):
        node = n_index_reynolds[i]
        region = node_region_jit(
            p,
            node,
            x_reynolds,
            z_reynolds,
            pad_length,
            axial_length,
            axial_length_track,
            axial_length_dam,
            length_track,
        )
        xr_rad = x_reynolds_rad[p, node]
        base_partial = (
            cp_new_p
            - xj * np.cos(leading_angle_rad_p + xr_rad)
            - yj * np.sin(leading_angle_rad_p + xr_rad)
            - preload_new_p * cp_new_p * np.cos(xr_rad - x_pivot_rad_p)
            - radius * tilt_angle_p * np.sin(xr_rad - x_pivot_rad_p)
            + dh_n[p, node]
        )
        if region == 0:  # pocket
            h_n[p, node] = base_partial + depth_track_p
            if unloaded_p and special:
                h_n[p, node] = cp_new_p + depth_track_p
        else:
            h_n[p, node] = base_partial
            if unloaded_p and special:
                h_n[p, node] = cp_new_p

        # Small clearance excluding the pocket depth.
        h_ns[node] = base_partial
        if unloaded_p and special:
            h_ns[node] = cp_new_p

        # Leading-edge taper.
        if x_reynolds[p, node] < length_ramp_le_p and length_ramp_le_p > 1.0e-6:
            ramp = dh_ramp_le_p * (1.0 - x_reynolds[p, node] / length_ramp_le_p)
            h_n[p, node] += ramp
            h_ns[node] += ramp

        # Trailing-edge taper.
        if (
            x_reynolds[p, node] > (pad_length_p - length_ramp_te_p)
            and length_ramp_te_p > 1.0e-6
        ):
            ramp = dh_ramp_te_p * (
                1.0 - ((pad_length_p - x_reynolds[p, node]) / length_ramp_te_p)
            )
            h_n[p, node] += ramp
            h_ns[node] += ramp

        if node == 0 or h_n[p, node] < h_min:
            h_min = h_n[p, node]
            x_hmin = x_reynolds[p, node]

    # Circumferential dh/dx (forward / backward / central).
    step = total_e_z_film + 1
    for i in range(total_n_reynolds):
        node = n_index_reynolds[i]
        if abs(x_reynolds[p, node]) < 1.0e-6:
            dhdx_n[p, node] = (-h_n[p, node] + h_n[p, node + step]) / (
                x_reynolds[p, node + step] - x_reynolds[p, node]
            )
        elif abs(x_reynolds[p, node] - pad_length_p) < 1.0e-6:
            dhdx_n[p, node] = (h_n[p, node] - h_n[p, node - step]) / (
                x_reynolds[p, node] - x_reynolds[p, node - step]
            )
        else:
            dhdx_n[p, node] = (h_n[p, node + step] - h_n[p, node - step]) / (
                x_reynolds[p, node + step] - x_reynolds[p, node - step]
            )

    if (
        operating_type == "axial_flow"
        or operating_type == "high_ambient_pressure"
        or bearing_type == "pressure_dam"
    ):
        full_cavitate_int = 0
    else:
        full_cavitate_int = 1 if x_hmin < 1.0e-6 else 0

    return h_min, x_hmin, full_cavitate_int, h_ns


@njit(cache=True, fastmath=False)
def film_thickness_rebuild_jit(
    p,
    total_e_y_pad,
    total_e_x_film,
    total_e_y_film,
    total_e_z_film,
    total_e_y_trackbl_p,
    total_e_y_trackcore_p,
    total_e_y_dambl_p,
    total_e_y_damcore_p,
    depth_track_p,
    pad_thickness,
    weight_h,
    h_ns,
    n_index_energy,
    match_nodes_xy,
    y_energy,
    y_3d,
):
    """Rebuild the energy / 3-D radial coordinates for one pad.

    Split from :func:`film_thickness_jit` to keep each kernel small. 0-based
    natural: ``p`` and the stored ids are 0-based; ``n_index_energy`` stores
    0-based ids and is walked by the position counter ``n`` (0-based);
    ``match_nodes_xy`` stores 0-based ids and is indexed ``[ne, k - 1]``. The
    x-station counter ``i`` and cross-film counter ``j`` stay 1-based position
    counters.
    """
    e_height_pad = pad_thickness / total_e_y_pad
    tbl = total_e_y_trackbl_p
    tcore = total_e_y_trackcore_p
    dbl = total_e_y_dambl_p
    dcore = total_e_y_damcore_p

    n = -1
    for i in range(1, total_e_x_film + 1 + 1):
        # First node of x-station ``i`` (0-based node value).
        h_n1 = h_ns[(i - 1) * (total_e_z_film + 1)]
        thickness_bl = weight_h * h_n1

        if abs(depth_track_p) < 1.0e-6:
            if thickness_bl < 1.0e-8:
                e_track_bl = 0.0
                e_track_core = 0.0
                e_dam_bl = 0.0
                e_dam_core = h_n1 / total_e_y_damcore_p
            else:
                e_track_bl = 0.0
                e_track_core = 0.0
                e_dam_bl = thickness_bl / total_e_y_dambl_p
                e_dam_core = (h_n1 - 2.0 * thickness_bl) / total_e_y_damcore_p
        else:
            if thickness_bl < 1.0e-8:
                e_track_bl = 0.0
                e_track_core = depth_track_p / total_e_y_trackcore_p
                e_dam_bl = 0.0
                e_dam_core = h_n1 / total_e_y_damcore_p
            else:
                e_track_bl = thickness_bl / total_e_y_trackbl_p
                e_track_core = (depth_track_p - thickness_bl) / total_e_y_trackcore_p
                e_dam_bl = thickness_bl / total_e_y_dambl_p
                e_dam_core = (h_n1 - 2.0 * thickness_bl) / total_e_y_damcore_p

        for j in range(1, total_e_y_pad + total_e_y_film + 1 + 1):
            if j <= total_e_y_pad + 1:
                y1 = (j - 1) * e_height_pad
            elif total_e_y_pad + 1 < j <= total_e_y_pad + tbl + 1:
                y1 = pad_thickness + (j - total_e_y_pad - 1) * e_track_bl
            elif total_e_y_pad + tbl + 1 < j <= total_e_y_pad + tbl + tcore + 1:
                y1 = (
                    pad_thickness
                    + thickness_bl
                    + (j - total_e_y_pad - tbl - 1) * e_track_core
                )
            elif (
                total_e_y_pad + tbl + tcore + 1
                < j
                <= total_e_y_pad + tbl + tcore + dbl + 1
            ):
                y1 = (
                    pad_thickness
                    + depth_track_p
                    + (j - total_e_y_pad - tbl - tcore - 1) * e_dam_bl
                )
            elif (
                total_e_y_pad + tbl + tcore + dbl + 1
                < j
                <= total_e_y_pad + tbl + tcore + dbl + dcore + 1
            ):
                y1 = (
                    pad_thickness
                    + depth_track_p
                    + thickness_bl
                    + (j - total_e_y_pad - tbl - tcore - dbl - 1) * e_dam_core
                )
            else:
                y1 = (
                    pad_thickness
                    + depth_track_p
                    + h_n1
                    - thickness_bl
                    + (j - total_e_y_pad - tbl - tcore - dbl - dcore - 1) * e_dam_bl
                )
            n += 1
            ne = n_index_energy[n]
            y_energy[p, ne] = y1
            if (
                y_energy[p, ne] > pad_thickness
                or abs(y_energy[p, ne] - pad_thickness) < 1.0e-6
            ):
                for k in range(1, total_e_z_film + 1 + 1):
                    y_3d[p, match_nodes_xy[ne, k - 1]] = y_energy[p, ne]


@njit(cache=True, fastmath=False)
def press_gradient_node_jit(
    p,
    total_n_reynolds,
    n_index_reynolds,
    nodal_pressure,
    axial_length,
    x_reynolds,
    z_reynolds,
    total_e_x_film,
    total_e_z_film,
    dpdx_n,
    dpdz_n,
):
    """Inner body of :func:`pressure.press_gradient_node`.

    All arrays 0-based natural; ``p`` is 0-based; ``n_index_reynolds`` stores
    0-based node ids. Three-point forward / backward / central differences on
    x and z.
    """
    step = total_e_z_film + 1
    al = axial_length[p]

    for i in range(total_n_reynolds):
        node = n_index_reynolds[i]

        # Circumferential derivative.
        if node >= (total_e_z_film + 1) * (total_e_x_film - 1):
            n0 = node
            nb = node - step
            nbb = node - 2 * step
            xn0 = x_reynolds[p, n0]
            xnb = x_reynolds[p, nb]
            xnbb = x_reynolds[p, nbb]
            dpdx_n[p, node] = (
                nodal_pressure[p, nbb] * (xn0 - xnb) / ((xnbb - xnb) * (xnbb - xn0))
                + nodal_pressure[p, nb] * (xn0 - xnbb) / ((xnb - xnbb) * (xnb - xn0))
                + nodal_pressure[p, n0]
                * (xn0 - xnbb + xn0 - xnb)
                / ((xn0 - xnbb) * (xn0 - xnb))
            )
        else:
            n0 = node
            nf = node + step
            nff = node + 2 * step
            xn0 = x_reynolds[p, n0]
            xnf = x_reynolds[p, nf]
            xnff = x_reynolds[p, nff]
            dpdx_n[p, node] = (
                nodal_pressure[p, n0]
                * (xn0 - xnf + xn0 - xnff)
                / ((xn0 - xnf) * (xn0 - xnff))
                + nodal_pressure[p, nf] * (xn0 - xnff) / ((xnf - xn0) * (xnf - xnff))
                + nodal_pressure[p, nff] * (xn0 - xnf) / ((xnff - xn0) * (xnff - xnf))
            )

        # Axial derivative.
        zn = z_reynolds[p, node]
        if zn < 0.5 * al:
            n0 = node
            nf = node + 1
            nff = node + 2
            zn0 = z_reynolds[p, n0]
            znf = z_reynolds[p, nf]
            znff = z_reynolds[p, nff]
            dpdz_n[p, node] = (
                nodal_pressure[p, n0]
                * (zn0 - znf + zn0 - znff)
                / ((zn0 - znf) * (zn0 - znff))
                + nodal_pressure[p, nf] * (zn0 - znff) / ((znf - zn0) * (znf - znff))
                + nodal_pressure[p, nff] * (zn0 - znf) / ((znff - zn0) * (znff - znf))
            )
        elif zn > 0.5 * al:
            n0 = node
            nb = node - 1
            nbb = node - 2
            zn0 = z_reynolds[p, n0]
            znb = z_reynolds[p, nb]
            znbb = z_reynolds[p, nbb]
            dpdz_n[p, node] = (
                nodal_pressure[p, nbb] * (zn0 - znb) / ((znbb - znb) * (znbb - zn0))
                + nodal_pressure[p, nb] * (zn0 - znbb) / ((znb - znbb) * (znb - zn0))
                + nodal_pressure[p, n0]
                * (zn0 - znbb + zn0 - znb)
                / ((zn0 - znbb) * (zn0 - znb))
            )
        else:
            n0 = node
            nb = node - 1
            nf = node + 1
            zn0 = z_reynolds[p, n0]
            znb = z_reynolds[p, nb]
            znf = z_reynolds[p, nf]
            dpdz_n[p, node] = (
                nodal_pressure[p, nb] * (zn0 - znf) / ((znb - zn0) * (znb - znf))
                + nodal_pressure[p, n0]
                * (zn0 - znb + zn0 - znf)
                / ((zn0 - znb) * (zn0 - znf))
                + nodal_pressure[p, nf] * (zn0 - znb) / ((znf - znb) * (znf - zn0))
            )


@njit(cache=True, fastmath=False)
def effective_conduct_jit(
    pad,
    total_n_reynolds,
    n_index_reynolds,
    total_e_y_film,
    match_nodes_xz,
    lube_conduct,
    lube_cp,
    vis_n_3d,
    conduct_effect,
    vis_eddy_3d,
    pr_turb,
):
    """Inner body of :func:`thermal.effective_conduct`.

    ``pad`` is 0-based. Mutates ``conduct_effect`` in place.
    """
    for i in range(total_n_reynolds):
        node = n_index_reynolds[i]
        for j in range(1, total_e_y_film + 1 + 1):
            m = match_nodes_xz[node, j - 1]
            conduct_effect[pad, m] = (
                lube_conduct
                + vis_n_3d[pad, m] * lube_cp * vis_eddy_3d[pad, m] / pr_turb
            )


@njit(cache=True, fastmath=False)
def flow_rates_jit(
    total_pads,
    dim_yf,
    dim_xz,
    operating_type,
    total_e_x_film,
    total_e_y_film,
    total_e_z_film,
    total_e_y_trackbl,
    total_e_y_trackcore,
    total_n_reynolds,
    n_index_reynolds,
    match_nodes_xz,
    total_e_reynolds,
    e_index_reynolds,
    node_i_reynolds,
    node_j_reynolds,
    node_k_reynolds,
    node_l_reynolds,
    film_onset,
    pad_thickness,
    length_track,
    depth_track,
    pad_length,
    axial_length,
    axial_length_track,
    axial_length_dam,
    x_reynolds,
    z_reynolds,
    e_length_reynolds,
    e_width_reynolds,
    velocity_x_n,
    velocity_z_n,
    h_n,
    x_hmin,
    y_3d,
    hotoil_lamda,
    temp_3d,
    temp_inlet,
    lube_density,
    lube_cp,
    bearing_type,
    xj,
    yj,
    q_available,
    q_x,
    q_in,
    q_out,
    q_sides,
    q_carryover,
    q_sidea,
    q_sideb,
    t_average,
):
    """Inner body of :func:`hydrodynamics.flow_rates`.

    0-based natural throughout: ``pad_index`` / ``node`` / element ids / the
    ``film_onset`` value and the ``q_x`` column index are all 0-based;
    ``match_nodes_xz`` stores 0-based ids indexed ``[node, j - 1]`` (cross-film
    column ``j`` is a 1-based position counter); ``e_index_reynolds`` is walked
    by the 0-based position counter ``n``. Mutates ``q_*`` and ``t_average`` in
    place.
    """
    # SI: ``lube_density [kg/m^3] * lube_cp [J/(kg K)]`` gives J/(m^3 K) directly.
    # The imperial *3600/2545 (BTU/s -> hp / degF -> degR equivalent) factor is
    # gone after Phase 3.
    enth_coeff = lube_density * lube_cp

    u_average = np.zeros(dim_xz + 1, dtype=np.float64)
    w_average = np.zeros(dim_xz + 1, dtype=np.float64)
    hc_n = np.zeros(dim_xz + 1, dtype=np.float64)
    f1 = np.zeros(dim_yf + 1, dtype=np.float64)
    f2 = np.zeros(dim_yf + 1, dtype=np.float64)
    f3 = np.zeros(dim_yf + 1, dtype=np.float64)
    f4 = np.zeros(dim_yf + 1, dtype=np.float64)
    f5 = np.zeros(dim_yf + 1, dtype=np.float64)
    t = np.zeros(dim_yf + 1, dtype=np.float64)

    for pad_index in range(total_pads):
        limit1 = total_e_y_trackbl[pad_index] + total_e_y_trackcore[pad_index] + 1
        limit2 = total_e_y_film + 1

        # Reset per-pad scratch.
        for k in range(dim_xz + 1):
            u_average[k] = 0.0
            w_average[k] = 0.0
            hc_n[k] = 0.0

        for i in range(total_n_reynolds):
            node = n_index_reynolds[i]
            for j in range(1, total_e_y_film + 1 + 1):
                m = match_nodes_xz[node, j - 1]
                f1[j] = velocity_x_n[pad_index, m]
                f2[j] = velocity_z_n[pad_index, m]
                f3[j] = temp_3d[pad_index, m]
                f4[j] = enth_coeff * f2[j] * (f3[j] - temp_inlet[pad_index])
                f5[j] = enth_coeff * f1[j] * (f3[j] - temp_inlet[pad_index])

            region = node_region_jit(
                pad_index,
                node,
                x_reynolds,
                z_reynolds,
                pad_length,
                axial_length,
                axial_length_track,
                axial_length_dam,
                length_track,
            )
            if region == 0:
                jlo = 1
                offset = pad_thickness
            else:
                jlo = limit1
                offset = pad_thickness + depth_track[pad_index]

            for j in range(jlo, limit2 + 1):
                m = match_nodes_xz[node, j - 1]
                t[j] = y_3d[pad_index, m] - offset

            i1 = 0.0
            i2 = 0.0
            i3 = 0.0
            i4 = 0.0
            i5 = 0.0
            for k in range(jlo, limit2):
                h = t[k + 1] - t[k]
                i1 += 0.5 * h * (f1[k + 1] + f1[k])
                i2 += 0.5 * h * (f2[k + 1] + f2[k])
                i3 += 0.5 * h * (f3[k + 1] + f3[k])
                i4 += 0.5 * h * (f4[k + 1] + f4[k])
                i5 += 0.5 * h * (f5[k + 1] + f5[k])

            hc_n[node] = i5
            u_average[node] = i1 / h_n[pad_index, node]
            w_average[node] = i2 / h_n[pad_index, node]
            t_average[pad_index, node] = i3 / h_n[pad_index, node]

        # Circumferential flow at all x stations (0-based columns).
        n = -1
        q_x[pad_index, total_e_x_film] = 0.0
        for ii in range(total_e_x_film):
            q_x[pad_index, ii] = 0.0
            for _j in range(1, total_e_z_film + 1):
                n += 1
                el = e_index_reynolds[n]
                ni = node_i_reynolds[el]
                nj = node_j_reynolds[el]
                nk = node_k_reynolds[el]
                nl = node_l_reynolds[el]
                q_x_e = (
                    0.5
                    * (u_average[ni] + u_average[nl])
                    * e_width_reynolds[pad_index, el]
                    * 0.5
                    * (h_n[pad_index, ni] + h_n[pad_index, nl])
                )
                q_x[pad_index, ii] += q_x_e
                if ii == total_e_x_film - 1:
                    q_x_e = (
                        0.5
                        * (u_average[nj] + u_average[nk])
                        * e_width_reynolds[pad_index, el]
                        * 0.5
                        * (h_n[pad_index, nj] + h_n[pad_index, nk])
                    )
                    q_x[pad_index, ii + 1] += q_x_e

        # Exit flow at the minimum-film location. ``ii`` is a 1-based x-station
        # counter; the centre node of station ``ii`` is the 0-based node value
        # ``(total_e_z_film // 2) + (ii - 1) * step``; its 0-based q_x column is
        # ``ii - 1``.
        q_out[pad_index] = 0.0
        for ii in range(1, total_e_x_film + 1 + 1):
            node = (total_e_z_film // 2) + (ii - 1) * (total_e_z_film + 1)
            if abs(x_reynolds[pad_index, node] - x_hmin[pad_index]) < 1.0e-6:
                q_out[pad_index] = q_x[pad_index, ii - 1]
                break

        # Centred pressure-dam exception.
        if bearing_type == "pressure_dam" and abs(xj) < 1.0e-8 and abs(yj) < 1.0e-8:
            q_out[pad_index] = q_x[pad_index, total_e_x_film]

        # Side leakage.
        q_w1 = 0.0
        q_w2 = 0.0
        for i in range(total_e_reynolds):
            el = e_index_reynolds[i]
            ni = node_i_reynolds[el]
            nj = node_j_reynolds[el]
            nk = node_k_reynolds[el]
            nl = node_l_reynolds[el]
            if abs(z_reynolds[pad_index, ni]) < 1.0e-6:
                q_w_e = (
                    0.5
                    * (w_average[ni] + w_average[nj])
                    * e_length_reynolds[pad_index, el]
                    * 0.5
                    * (h_n[pad_index, ni] + h_n[pad_index, nj])
                )
                q_w1 += q_w_e
            elif abs(z_reynolds[pad_index, nl] - axial_length[pad_index]) < 1.0e-6:
                q_w_e = (
                    0.5
                    * (w_average[nk] + w_average[nl])
                    * e_length_reynolds[pad_index, el]
                    * 0.5
                    * (h_n[pad_index, nk] + h_n[pad_index, nl])
                )
                q_w2 += q_w_e

        q_sides1 = abs(q_w1) + abs(q_w2)
        q_sidea[pad_index] = q_w1
        q_sideb[pad_index] = q_w2

        if (
            operating_type == "regular_flooded"
            or operating_type == "starved_condition_even"
            or operating_type == "starved_condition_uneven"
            or operating_type == "oil_ring_lubricated"
        ):
            if q_sides1 < 1.0e-6:
                qa = q_x[pad_index, film_onset[pad_index]]
                qb = q_available[pad_index]
                q_in[pad_index] = min(qb, qa)
            else:
                q_in[pad_index] = q_x[pad_index, film_onset[pad_index]]
            q_out[pad_index] = min(q_out[pad_index], q_in[pad_index])
            q_sides[pad_index] = q_in[pad_index] - q_out[pad_index]
            q_carryover[pad_index] = hotoil_lamda * q_out[pad_index]
        elif operating_type == "high_ambient_pressure":
            q_in[pad_index] = q_x[pad_index, 0]
            q_out[pad_index] = q_x[pad_index, total_e_x_film]
            q_sides[pad_index] = q_in[pad_index] - q_out[pad_index]
            q_carryover[pad_index] = hotoil_lamda * q_out[pad_index]


@njit(cache=True, fastmath=False)
def viscosity_temp_jit(viscosity1, viscosity2, temp1, temp2, temp):
    """Reynolds two-point viscosity-temperature law (scalar, JIT)."""
    beta1 = np.log(viscosity2 / viscosity1) / (temp2 - temp1)
    return viscosity1 * np.exp(beta1 * (temp - temp1))


@njit(cache=True, fastmath=False)
def update_vis_jit(
    pad,
    total_n_reynolds,
    n_index_reynolds,
    total_e_y_film,
    match_nodes_xz,
    lube_density,
    speed_surface,
    temp_3d,
    pad_thickness,
    y_3d,
    vis_n_3d,
    vis_n_average,
    h_n,
    viscosity1,
    viscosity2,
    temp1,
    temp2,
    dim_yf,
    dim_xz,
):
    """Inner loop body of :func:`thermal.update_vis`.

    ``pad`` is 0-based. Mutates ``vis_n_3d`` and ``vis_n_average`` in place;
    returns ``re_max_dam`` so the caller can call ``_update_flow_regime``
    with it.
    """
    re_n = np.zeros(dim_xz, dtype=np.float64)
    t = np.zeros(dim_yf, dtype=np.float64)
    f = np.zeros(dim_yf, dtype=np.float64)

    limit = total_e_y_film + 1
    re_max_dam = 0.0

    for i in range(total_n_reynolds):
        node = n_index_reynolds[i]
        for j in range(1, limit + 1):
            m = match_nodes_xz[node, j - 1]
            vis_n_3d[pad, m] = viscosity_temp_jit(
                viscosity1, viscosity2, temp1, temp2, temp_3d[pad, m]
            )
            t[j - 1] = y_3d[pad, m] - pad_thickness
            f[j - 1] = vis_n_3d[pad, m]

        # Trapezoid integral over [1, limit] in 0-based slots.
        inte = 0.0
        for ii in range(1, limit):
            h = t[ii] - t[ii - 1]
            inte += 0.5 * h * (f[ii] + f[ii - 1])
        h_node = h_n[pad, node]
        vis_n_average[pad, node] = inte / h_node
        re_n[node] = lube_density * speed_surface * h_node / vis_n_average[pad, node]
        re_max_dam = max(re_max_dam, re_n[node])

    return re_max_dam


@njit(cache=True, fastmath=False)
def energy_coeffs_flooded_jit(
    pad,
    total_n_energy,
    n_index_energy,
    match_nodes_xy,
    total_e_z_film,
    pad_thickness,
    y_energy,
    lube_conduct,
    pad_conduct,
    lube_density,
    lube_cp,
    al,
    vis_effect_3d,
    conduct_effect,
    dudy_n,
    dwdy_n,
    velocity_x_n,
    velocity_y_n,
    z_3d,
    dim_xy,
    dim_z,
    dissip_factor,
):
    """Inner body of :func:`thermal.energy_coeffs_flooded`.

    ``pad`` is 0-based; arrays are 0-based (matching the caller convention).
    Returns ``(kx_n, ky_n, mx_n, my_n, p_n, q_n)`` as length-``dim_xy``
    arrays.
    """
    kx_n = np.zeros(dim_xy, dtype=np.float64)
    ky_n = np.zeros(dim_xy, dtype=np.float64)
    mx_n = np.zeros(dim_xy, dtype=np.float64)
    my_n = np.zeros(dim_xy, dtype=np.float64)
    p_n = np.zeros(dim_xy, dtype=np.float64)
    q_n = np.zeros(dim_xy, dtype=np.float64)

    t = np.zeros(dim_z, dtype=np.float64)
    f1 = np.zeros(dim_z, dtype=np.float64)
    f2 = np.zeros(dim_z, dtype=np.float64)
    f3 = np.zeros(dim_z, dtype=np.float64)
    f4 = np.zeros(dim_z, dtype=np.float64)

    limit = total_e_z_film + 1

    for i in range(total_n_energy):
        node = n_index_energy[i]
        y = y_energy[pad, node]

        if y < pad_thickness:
            kx_n[node] = -pad_conduct
            ky_n[node] = -pad_conduct
            # mx, my, p, q already zero.
        elif y > pad_thickness:
            kx_n[node] = -lube_conduct
            for j in range(1, limit + 1):
                m = match_nodes_xy[node, j - 1]
                t[j - 1] = z_3d[pad, m]
                f1[j - 1] = conduct_effect[pad, m]
                f2[j - 1] = velocity_x_n[pad, m]
                f3[j - 1] = velocity_y_n[pad, m]
                f4[j - 1] = (
                    vis_effect_3d[pad, m]
                    * (
                        dudy_n[pad, m] * dudy_n[pad, m]
                        + dwdy_n[pad, m] * dwdy_n[pad, m]
                    )
                    / dissip_factor
                )
            # Trapezoid integrals over [1, limit] (0-based slot k - 1).
            i1 = 0.0
            i2 = 0.0
            i3 = 0.0
            i4 = 0.0
            for ii in range(1, limit):
                h = t[ii] - t[ii - 1]
                i1 += 0.5 * h * (f1[ii] + f1[ii - 1])
                i2 += 0.5 * h * (f2[ii] + f2[ii - 1])
                i3 += 0.5 * h * (f3[ii] + f3[ii - 1])
                i4 += 0.5 * h * (f4[ii] + f4[ii - 1])
            ky_n[node] = -i1 / al
            mx_n[node] = lube_density * lube_cp * i2 / al
            my_n[node] = lube_density * lube_cp * i3 / al
            q_n[node] = -i4 / al

        if abs(y - pad_thickness) < 1.0e-6:
            kx_n[node] = -(pad_conduct * lube_conduct) / (pad_conduct + lube_conduct)
            for j in range(1, limit + 1):
                m = match_nodes_xy[node, j - 1]
                t[j - 1] = z_3d[pad, m]
                ce = conduct_effect[pad, m]
                f1[j - 1] = (pad_conduct * ce) / (pad_conduct + ce)
                f2[j - 1] = velocity_x_n[pad, m]
                f3[j - 1] = velocity_y_n[pad, m]
                f4[j - 1] = (
                    vis_effect_3d[pad, m]
                    * (
                        dudy_n[pad, m] * dudy_n[pad, m]
                        + dwdy_n[pad, m] * dwdy_n[pad, m]
                    )
                    / dissip_factor
                )
            i1 = 0.0
            i2 = 0.0
            i3 = 0.0
            i4 = 0.0
            for ii in range(1, limit):
                h = t[ii] - t[ii - 1]
                i1 += 0.5 * h * (f1[ii] + f1[ii - 1])
                i2 += 0.5 * h * (f2[ii] + f2[ii - 1])
                i3 += 0.5 * h * (f3[ii] + f3[ii - 1])
                i4 += 0.5 * h * (f4[ii] + f4[ii - 1])
            ky_n[node] = -i1 / al
            mx_n[node] = lube_density * lube_cp * i2 / al
            my_n[node] = lube_density * lube_cp * i3 / al
            q_n[node] = -i4 / al

    return kx_n, ky_n, mx_n, my_n, p_n, q_n


@njit(cache=True, fastmath=False)
def flow_regime_jit(
    p,
    dim_yf,
    total_e_y_trackbl_p,
    total_e_y_trackcore_p,
    total_e_y_film,
    total_n_reynolds,
    n_index_reynolds,
    match_nodes_xz,
    pad_length,
    axial_length,
    axial_length_track,
    axial_length_dam,
    length_track,
    depth_track,
    pad_thickness,
    x_reynolds,
    z_reynolds,
    lube_density,
    speed_surface,
    re_lower,
    re_upper,
    vis_n_3d,
    y_3d,
    h_n,
    vis_n_average,
    turb_scal_fac_exp,
):
    """Inner body of :func:`hydrodynamics.flow_regime`.

    0-based natural; ``p`` and the stored node ids are 0-based;
    ``match_nodes_xz`` is indexed ``[node, j - 1]`` (cross-film column ``j`` is a
    1-based position counter). Mutates ``vis_n_average`` in place; returns
    ``(flow_regime_track, scale_turb_track, flow_regime_dam, scale_turb_dam,
    re_max)`` as scalars for the caller to scatter.
    """
    limit1 = total_e_y_trackbl_p + total_e_y_trackcore_p + 1
    limit2 = total_e_y_film + 1

    t = np.zeros(dim_yf + 1, dtype=np.float64)
    f = np.zeros(dim_yf + 1, dtype=np.float64)

    re_max_track = 0.0
    re_max_dam = 0.0
    for i in range(total_n_reynolds):
        node = n_index_reynolds[i]
        region = node_region_jit(
            p,
            node,
            x_reynolds,
            z_reynolds,
            pad_length,
            axial_length,
            axial_length_track,
            axial_length_dam,
            length_track,
        )
        if region == 0:
            jlo = 1
            offset = pad_thickness
        else:
            jlo = limit1
            offset = pad_thickness + depth_track[p]

        for j in range(jlo, limit2 + 1):
            m = match_nodes_xz[node, j - 1]
            t[j] = y_3d[p, m] - offset
            f[j] = vis_n_3d[p, m]
        inte_trap = 0.0
        for ii in range(jlo, limit2):
            h = t[ii + 1] - t[ii]
            inte_trap += 0.5 * h * (f[ii + 1] + f[ii])
        vis_n_average[p, node] = inte_trap / h_n[p, node]
        re_n = lube_density * speed_surface * h_n[p, node] / vis_n_average[p, node]
        if region == 0:
            re_max_track = max(re_max_track, re_n)
        else:
            re_max_dam = max(re_max_dam, re_n)

    if re_max_track < re_lower:
        flow_regime_track_p = 0
        scale_turb_track_p = 0.0
    elif re_lower < re_max_track < re_upper:
        flow_regime_track_p = 1
        scale_turb_track_p = (
            1.0
            - ((re_upper - re_max_track) / (re_upper - re_lower)) ** turb_scal_fac_exp
        )
    else:
        flow_regime_track_p = 2
        scale_turb_track_p = 1.0

    if re_max_dam < re_lower:
        flow_regime_dam_p = 0
        scale_turb_dam_p = 0.0
    elif re_lower < re_max_dam < re_upper:
        flow_regime_dam_p = 1
        scale_turb_dam_p = (
            1.0 - ((re_upper - re_max_dam) / (re_upper - re_lower)) ** turb_scal_fac_exp
        )
    else:
        flow_regime_dam_p = 2
        scale_turb_dam_p = 1.0

    re_max_p = max(re_max_track, re_max_dam)

    return (
        flow_regime_track_p,
        scale_turb_track_p,
        flow_regime_dam_p,
        scale_turb_dam_p,
        re_max_p,
    )


@njit(cache=True, fastmath=False)
def effective_viscosity_jit(
    p,
    total_e_y_film,
    total_e_y_dambl_p,
    total_e_y_damcore_p,
    total_e_y_trackbl_p,
    total_e_y_trackcore_p,
    total_n_reynolds,
    n_index_reynolds,
    match_nodes_xz,
    pad_length,
    axial_length,
    length_track,
    axial_length_track,
    axial_length_dam,
    depth_track,
    pad_thickness,
    x_reynolds,
    z_reynolds,
    vis_n_3d,
    vis_n_average,
    shear_stress,
    y_3d,
    h_n,
    lube_density,
    scale_turb_track_p,
    scale_turb_dam_p,
    vis_eddy_3d,
    vis_effect_3d,
    reichardt_delta,
    reichardt_kappa,
):
    """Inner loop body of :func:`hydrodynamics.effective_viscosity`.

    0-based natural; ``p`` and the stored node ids are 0-based;
    ``match_nodes_xz`` is indexed ``[node, j - 1]`` (cross-film column ``j`` is a
    1-based position counter). ``vis_eddy_3d`` and ``vis_effect_3d`` are mutated
    in place. Returns ``y_plus_max``.
    """
    y_plus_max = 0.0
    nf = total_e_y_film + 1

    for i in range(total_n_reynolds):
        node = n_index_reynolds[i]
        region = node_region_jit(
            p,
            node,
            x_reynolds,
            z_reynolds,
            pad_length,
            axial_length,
            axial_length_track,
            axial_length_dam,
            length_track,
        )
        if region == 0:
            scale_turb = scale_turb_track_p
        else:
            scale_turb = scale_turb_dam_p

        tbl = total_e_y_trackbl_p
        tcore = total_e_y_trackcore_p
        dbl = total_e_y_dambl_p
        dcore = total_e_y_damcore_p

        for j in range(1, nf + 1):
            m = match_nodes_xz[node, j - 1]
            if region == 0:
                if j <= nf // 2:
                    wall = match_nodes_xz[node, 0]
                    y_plus = (
                        (y_3d[p, m] - pad_thickness)
                        * (shear_stress[p, wall] * lube_density) ** 0.5
                        / vis_n_average[p, node]
                    )
                else:
                    wall = match_nodes_xz[node, nf - 1]
                    y_plus = (
                        (h_n[p, node] - y_3d[p, m] + pad_thickness)
                        * (shear_stress[p, wall] * lube_density) ** 0.5
                        / vis_n_average[p, node]
                    )
            else:
                mid = tbl + tcore + dbl + (dcore + 1) // 2
                if tbl + tcore < j <= mid:
                    wall = match_nodes_xz[node, tbl + tcore + 1 - 1]
                    y_plus = (
                        (y_3d[p, m] - pad_thickness - depth_track[p])
                        * (shear_stress[p, wall] * lube_density) ** 0.5
                        / vis_n_average[p, node]
                    )
                elif mid < j <= nf:
                    wall = match_nodes_xz[node, nf - 1]
                    y_plus = (
                        (h_n[p, node] - y_3d[p, m] + pad_thickness + depth_track[p])
                        * (shear_stress[p, wall] * lube_density) ** 0.5
                        / vis_n_average[p, node]
                    )
                else:
                    y_plus = 0.0

            y_plus_max = max(y_plus_max, y_plus)
            vis_eddy_3d[p, m] = (
                scale_turb
                * reichardt_kappa
                * (y_plus - reichardt_delta * np.tanh(y_plus / reichardt_delta))
            )
            vis_effect_3d[p, m] = vis_n_3d[p, m] * (1.0 + vis_eddy_3d[p, m])

    return y_plus_max


@njit(cache=True, fastmath=False)
def node_region_jit(
    p,
    node,
    x_reynolds,
    z_reynolds,
    pad_length,
    axial_length,
    axial_length_track,
    axial_length_dam,
    length_track,
):
    """Numba-friendly version of :func:`hydrodynamics._node_region`.

    Returns 0 for ``"pocket"``, 1 for ``"dam"``. ``p`` and ``node`` are 0-based
    values indexing the 0-based natural per-pad coordinate arrays directly.
    """
    xr = x_reynolds[p, node]
    zr = z_reynolds[p, node]
    if (
        zr > axial_length_dam[p]
        and zr < axial_length_dam[p] + axial_length_track[p]
        and xr < length_track[p]
    ):
        return 0  # pocket
    if (
        xr > length_track[p]
        or zr < axial_length_dam[p]
        or zr > axial_length_dam[p] + axial_length_track[p]
    ):
        return 1  # dam
    edge_is_pad_edge = (
        abs(xr - pad_length[p]) < 1.0e-6
        and zr > axial_length_dam[p]
        and zr < axial_length_track[p] + axial_length_dam[p]
    ) or (
        (abs(zr) < 1.0e-6 or abs(zr - axial_length[p]) < 1.0e-6)
        and xr < length_track[p]
    )
    return 0 if edge_is_pad_edge else 1


@njit(cache=True, fastmath=False)
def dudy_dwdy_jit(
    p,
    dim_yf,
    total_e_y_film,
    total_e_y_trackbl_p,
    total_e_y_trackcore_p,
    n_index_reynolds,
    match_nodes_xz,
    total_n_reynolds,
    pad_thickness,
    speed_surface,
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
):
    """Inner loop body of :func:`hydrodynamics.dudy_dwdy`.

    0-based natural; ``p`` and the stored node ids are 0-based;
    ``match_nodes_xz`` is indexed ``[node, j - 1]`` (cross-film column ``j`` is a
    1-based position counter into the local scratch grid). ``dudy_n``/``dwdy_n``
    are mutated in place.
    """
    limit1 = total_e_y_trackbl_p + total_e_y_trackcore_p + 1
    limit2 = total_e_y_film + 1

    t = np.zeros(dim_yf + 1, dtype=np.float64)
    f1 = np.zeros(dim_yf + 1, dtype=np.float64)
    f2 = np.zeros(dim_yf + 1, dtype=np.float64)

    for i in range(total_n_reynolds):
        node = n_index_reynolds[i]
        region = node_region_jit(
            p,
            node,
            x_reynolds,
            z_reynolds,
            pad_length,
            axial_length,
            axial_length_track,
            axial_length_dam,
            length_track,
        )
        if region == 0:  # pocket
            jlo = 1
            offset = pad_thickness
        else:
            jlo = limit1
            offset = pad_thickness + depth_track[p]

        for j in range(jlo, limit2 + 1):
            m = match_nodes_xz[node, j - 1]
            t[j] = y_3d[p, m] - offset
            f1[j] = 1.0 / vis_effect_3d[p, m]
            f2[j] = t[j] / vis_effect_3d[p, m]

        # Trapezoid over the local scratch grid for xi1h and xi2h over
        # [jlo, limit2].
        xi1h = 0.0
        xi2h = 0.0
        for ii in range(jlo, limit2):
            h = t[ii + 1] - t[ii]
            xi1h += 0.5 * h * (f1[ii + 1] + f1[ii])
            xi2h += 0.5 * h * (f2[ii + 1] + f2[ii])

        ratio = xi2h / xi1h
        for j in range(jlo, limit2 + 1):
            m = match_nodes_xz[node, j - 1]
            vis = vis_effect_3d[p, m]
            yo = y_3d[p, m] - offset
            dudy_n[p, m] = (
                dpdx_n[p, node] * yo / vis
                + (speed_surface / xi1h - dpdx_n[p, node] * ratio) / vis
            )
            dwdy_n[p, m] = dpdz_n[p, node] * (yo - ratio) / vis

    return dudy_n, dwdy_n


@njit(cache=True, fastmath=False)
def update_shear_jit(
    p,
    total_n_reynolds,
    n_index_reynolds,
    total_e_y_film,
    match_nodes_xz,
    dudy_n,
    dwdy_n,
    vis_effect_3d,
    shear_stress,
    relaxp,
):
    """Inner loop body of :func:`ross.bearings.fluid_film.hydrodynamics.update_shear`.

    0-based natural; ``p`` and the stored node ids are 0-based;
    ``match_nodes_xz`` is indexed ``[node, j - 1]`` (cross-film column ``j`` is a
    1-based position counter). Mutates ``shear_stress`` in place and returns
    ``rms_shear``.
    """
    nf = total_e_y_film + 1
    total = 0.0
    total1 = 0.0
    for i in range(total_n_reynolds):
        node = n_index_reynolds[i]
        for j in range(1, nf + 1):
            m = match_nodes_xz[node, j - 1]
            shear_old = shear_stress[p, m]
            shear1 = (
                vis_effect_3d[p, m]
                * (dudy_n[p, m] * dudy_n[p, m] + dwdy_n[p, m] * dwdy_n[p, m]) ** 0.5
            )
            shear_stress[p, m] = (1.0 - relaxp) * shear_old + relaxp * shear1
            total += (shear1 - shear_old) * (shear1 - shear_old)
            total1 += shear_old
    count = total_n_reynolds * nf
    rms_shear = (total / count) ** 0.5 / (total1 / count)
    return shear_stress, rms_shear


@njit(cache=True, fastmath=False)
def element_temp_interior_jit(
    pad,
    ce,
    node_1_energy,
    node_2_energy,
    node_3_energy,
    node_4_energy,
    x_energy,
    y_energy,
    kx_e,
    ky_e,
    mx_e,
    my_e,
    p_e,
    q_e,
):
    """Sum the four interior Gauss-point integrands of a thermal Q4 element.

    Mirrors the (-g,-g)/(g,-g)/(-g,g)/(g,g) loop inside
    :func:`ross.bearings.fluid_film.thermal.element_temp` and inlines
    :func:`ross.bearings.fluid_film.thermal.integrand_e_temp` at each point. ``pad``
    and ``ce`` are already 0-based.
    """
    n1 = node_1_energy[ce]
    n2 = node_2_energy[ce]
    n3 = node_3_energy[ce]
    n4 = node_4_energy[ce]

    # Geometry columns for the four corner nodes.
    x1 = x_energy[pad, n1]
    x2 = x_energy[pad, n2]
    x3 = x_energy[pad, n3]
    x4 = x_energy[pad, n4]
    y1 = y_energy[pad, n1]
    y2 = y_energy[pad, n2]
    y3 = y_energy[pad, n3]
    y4 = y_energy[pad, n4]

    g = 1.0 / 3.0**0.5

    e_matrix = np.zeros((4, 4), dtype=np.float64)
    e_column = np.zeros(4, dtype=np.float64)

    for gp in range(4):
        if gp == 0:
            r = -g
            s = -g
        elif gp == 1:
            r = g
            s = -g
        elif gp == 2:
            r = -g
            s = g
        else:
            r = g
            s = g

        # Shape functions n[0..3] and derivatives f[0..1, 0..3].
        n0 = (1.0 - r) * (1.0 - s) * 0.25
        n1v = (1.0 + r) * (1.0 - s) * 0.25
        n2v = (1.0 + r) * (1.0 + s) * 0.25
        n3v = (1.0 - r) * (1.0 + s) * 0.25

        # f[0, :] = dN/dr; f[1, :] = dN/ds.
        f00 = -(1.0 - s) * 0.25
        f01 = (1.0 - s) * 0.25
        f02 = (1.0 + s) * 0.25
        f03 = -(1.0 + s) * 0.25
        f10 = -(1.0 - r) * 0.25
        f11 = -(1.0 + r) * 0.25
        f12 = (1.0 + r) * 0.25
        f13 = (1.0 - r) * 0.25

        # jac = f @ gc.T  -> 2x2.
        # gc.T row k = (x_k, y_k).
        jac00 = f00 * x1 + f01 * x2 + f02 * x3 + f03 * x4
        jac01 = f00 * y1 + f01 * y2 + f02 * y3 + f03 * y4
        jac10 = f10 * x1 + f11 * x2 + f12 * x3 + f13 * x4
        jac11 = f10 * y1 + f11 * y2 + f12 * y3 + f13 * y4

        det_j = jac00 * jac11 - jac01 * jac10
        inv00 = jac11 / det_j
        inv01 = -jac01 / det_j
        inv10 = -jac10 / det_j
        inv11 = jac00 / det_j

        # b = j_inv @ f  -> 2x4.
        b00 = inv00 * f00 + inv01 * f10
        b01 = inv00 * f01 + inv01 * f11
        b02 = inv00 * f02 + inv01 * f12
        b03 = inv00 * f03 + inv01 * f13
        b10 = inv10 * f00 + inv11 * f10
        b11 = inv10 * f01 + inv11 * f11
        b12 = inv10 * f02 + inv11 * f12
        b13 = inv10 * f03 + inv11 * f13

        # b_t @ kb where kb = [kx*b[0,:]; ky*b[1,:]] -> 4x4 with entry
        # (i, j) = b[0, i] * kx * b[0, j] + b[1, i] * ky * b[1, j].
        # n_tv @ b = outer(n, [mx, my]) @ b = n[i] * (mx*b[0,j] + my*b[1,j]).
        # pn_tn = p_e * outer(n, n).
        n_arr = np.empty(4, dtype=np.float64)
        n_arr[0] = n0
        n_arr[1] = n1v
        n_arr[2] = n2v
        n_arr[3] = n3v
        b_row0 = np.empty(4, dtype=np.float64)
        b_row1 = np.empty(4, dtype=np.float64)
        b_row0[0] = b00
        b_row0[1] = b01
        b_row0[2] = b02
        b_row0[3] = b03
        b_row1[0] = b10
        b_row1[1] = b11
        b_row1[2] = b12
        b_row1[3] = b13

        for i in range(4):
            for j in range(4):
                btkb = b_row0[i] * kx_e * b_row0[j] + b_row1[i] * ky_e * b_row1[j]
                ntvb = n_arr[i] * (mx_e * b_row0[j] + my_e * b_row1[j])
                pntn = p_e * n_arr[i] * n_arr[j]
                e_matrix[i, j] += (btkb - ntvb - pntn) * det_j
            e_column[i] += n_arr[i] * q_e * det_j

    return e_matrix, e_column


@njit(cache=True, fastmath=False)
def lu_solve_band_jit(
    a,
    total_n_reynolds,
    bandwidth_reynolds,
    a_lower,
    index1,
    b,
):
    """Banded LU solve for a perturbation right-hand side (no cavitation clamp).

    Backs :func:`ross.bearings.fluid_film.banded.lu_solve`. ``b`` is the
    0-based-shaped vector used by ``coefficients`` (slot ``k - 1`` carries node
    ``k``); ``a`` / ``a_lower`` / ``index1`` are the 0-based-shaped factors the
    coefficients module owns. The solution overwrites ``b`` in place.
    """
    total_column = 2 * bandwidth_reynolds - 1

    # Forward substitution with row unscrambling.
    ll = bandwidth_reynolds - 1
    for k in range(1, total_n_reynolds + 1):
        ip = index1[k - 1]
        if ip != k:
            tmp = b[k - 1]
            b[k - 1] = b[ip - 1]
            b[ip - 1] = tmp
        if ll < total_n_reynolds:
            ll += 1
        for i in range(k + 1, ll + 1):
            b[i - 1] -= a_lower[k - 1, i - k - 1] * b[k - 1]

    # Back substitution (no clamp).
    ll = 1
    for i in range(total_n_reynolds, 0, -1):
        dum = b[i - 1]
        for k in range(2, ll + 1):
            dum -= a[i - 1, k - 1] * b[k + i - 2]
        b[i - 1] = dum / a[i - 1, 0]
        if ll < total_column:
            ll += 1
    return b
