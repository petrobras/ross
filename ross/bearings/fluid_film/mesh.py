"""Finite-element mesh generation.

Builds the node coordinates and element connectivity the rest of the solver
runs on:

* :func:`mesh_reynolds` -- 2-D mesh in the circumferential/axial (``x``-``z``)
  plane for the Reynolds (pressure) solution.
* :func:`mesh_deform` -- 2-D mesh in the circumferential/radial (``x``-``y``)
  plane for the pad elastic deformation.
* :func:`mesh_energy` -- 2-D mesh in the ``x``-``y`` plane spanning both the
  solid pad and the lubricant film for the energy (temperature) equation.
* :func:`mesh_3d` -- the 3-D film mesh used to transfer data between the two
  orthogonal 2-D meshes, plus the node-matching maps ``match_nodes_xz`` and
  ``match_nodes_xy``.

Indexing / data-structure convention
-------------------------------------
Every array is 0-based: it is allocated exactly as large as needed and indexed
with 0-based node / element / pad numbers. Connectivity arrays store node and
element numbers that are used directly as indices into the coordinate arrays --
there is no padding and no value shift anywhere in the package. Thus

* ``e_index_reynolds[e]`` for ``e = 0 .. total_e_reynolds - 1`` (== ``e``),
* ``node_i_reynolds[e]`` is the node number of element ``e``,
* ``x_reynolds[pad, n]`` for ``pad = 0 .. total_pads - 1`` and node ``n``,

with element/node loops running ``range(total_*)``. The array sizes are::

    dim_x   = total_e_x_film + 1
    dim_yf  = total_e_y_film + 1
    dim_z   = total_e_z_film + 1
    dim_yp  = total_e_y_pad + 1
    dim_xz  = dim_x * dim_z
    dim_xy  = dim_x * (total_e_y_film + total_e_y_pad + 1)
    dim_xy2 = 2 * dim_x * dim_yp
    dim_3d  = dim_x * dim_yf * dim_z

Per-pad integer arrays such as ``total_e_x_track`` are shaped
``(total_pads,)``. The ``match_nodes_*`` maps store 3-D node numbers with
``-1`` marking an unused slot (``0`` is a valid node number).

Every function *returns* its results as numpy arrays or tuples rather than
mutating its arguments; the caller assigns them back.
"""

import numpy as np

from ross.bearings.fluid_film._numba_kernels import mesh_3d_jit


def mesh_reynolds(
    total_pads,
    dim_xz,
    arc_length_rad,
    pad_length,
    length_track_rad,
    length_track,
    length_dam,
    axial_length,
    axial_length_track,
    axial_length_dam,
    depth_track,
    total_e_x_film,
    total_e_z_film,
    total_e_x_track,
    total_e_z_track,
    total_e_x_dam,
    total_e_z_dam,
):
    """Generate the 2-D Reynolds (pressure) mesh in the ``x``-``z`` plane.

    Builds node coordinates and quadrilateral connectivity for the film of a
    pad that may carry a pressure-dam track/pocket; a smooth pad is the special
    case of zero track size. Element size near the pocket edges is fixed at 1%
    of the global length. The connectivity is identical for every pad even
    though the coordinates may differ.

    Parameters
    ----------
    total_pads : int
        Number of pads.
    dim_xz : int
        Allocated size of the ``x``-``z`` arrays, ``(total_e_x_film + 1) *
        (total_e_z_film + 1)``.
    arc_length_rad : array_like
        Per-pad pad arc length, radians (shape ``(total_pads,)``).
    pad_length : array_like
        Per-pad circumferential pad length, m.
    length_track_rad : array_like
        Per-pad circumferential track length, radians.
    length_track : array_like
        Per-pad circumferential track length, m.
    length_dam : array_like
        Per-pad circumferential dam length, m.
    axial_length : array_like
        Per-pad total axial length, m.
    axial_length_track : array_like
        Per-pad axial track length, m.
    axial_length_dam : array_like
        Per-pad axial dam length, m.
    depth_track : array_like
        Per-pad track (pocket) depth, m; ``|depth| < 1e-6`` means smooth.
    total_e_x_film : int
        Number of film elements in the circumferential direction.
    total_e_z_film : int
        Number of film elements in the axial direction.
    total_e_x_track, total_e_z_track, total_e_x_dam, total_e_z_dam : array_like
        Per-pad element counts in the track/dam regions. Passed in but
        Ignored on input: recomputed and returned.

    Returns
    -------
    dict
        Keys (all numpy arrays are 0-based natural):

        ``total_e_x_track``, ``total_e_z_track``, ``total_e_x_dam``,
        ``total_e_z_dam`` : ndarray of int, shape ``(total_pads,)``
            Recomputed per-pad region element counts.
        ``total_e_reynolds`` : int
            Total number of elements, ``total_e_x_film * total_e_z_film``.
        ``e_index_reynolds`` : ndarray of int, shape ``(dim_xz,)``
            Identity element-number map (``e_index_reynolds[e] == e``).
        ``node_i_reynolds``, ``node_j_reynolds``, ``node_k_reynolds``,
        ``node_l_reynolds`` : ndarray of int, shape ``(dim_xz,)``
            0-based node ids of each element's four corners.
        ``e_length_reynolds``, ``e_width_reynolds`` : ndarray of float,
        shape ``(total_pads, dim_xz)``
            Element circumferential length / axial width per pad.
        ``total_n_reynolds`` : int
            Total number of nodes, ``(total_e_x_film+1)*(total_e_z_film+1)``.
        ``n_index_reynolds`` : ndarray of int, shape ``(dim_xz,)``
            Identity node-number map (``n_index_reynolds[n] == n``).
        ``x_reynolds``, ``z_reynolds``, ``x_reynolds_rad`` : ndarray of float,
        shape ``(total_pads, dim_xz)``
            Nodal coordinates: circumferential m, axial m, and
            circumferential rad.
        ``dx_reynolds``, ``dz_reynolds`` : ndarray of float,
        shape ``(total_pads, dim_xz, 4)``
            Shape-function derivatives at the element centre ``(0, 0)``; the
            trailing axis (size 4) is 0-based slots ``[..., 0:4]`` for the
            four element corners.
    """
    total_e_x_track = np.array(total_e_x_track, dtype=np.int64)
    total_e_z_track = np.array(total_e_z_track, dtype=np.int64)
    total_e_x_dam = np.array(total_e_x_dam, dtype=np.int64)
    total_e_z_dam = np.array(total_e_z_dam, dtype=np.int64)

    total_n_reynolds = (total_e_x_film + 1) * (total_e_z_film + 1)
    total_e_reynolds = total_e_x_film * total_e_z_film

    n_index_reynolds = np.zeros(dim_xz, dtype=np.int64)
    e_index_reynolds = np.zeros(dim_xz, dtype=np.int64)
    node_i_reynolds = np.zeros(dim_xz, dtype=np.int64)
    node_j_reynolds = np.zeros(dim_xz, dtype=np.int64)
    node_k_reynolds = np.zeros(dim_xz, dtype=np.int64)
    node_l_reynolds = np.zeros(dim_xz, dtype=np.int64)

    x_reynolds = np.zeros((total_pads, dim_xz))
    z_reynolds = np.zeros((total_pads, dim_xz))
    x_reynolds_rad = np.zeros((total_pads, dim_xz))
    e_length_reynolds = np.zeros((total_pads, dim_xz))
    e_width_reynolds = np.zeros((total_pads, dim_xz))
    dx_reynolds = np.zeros((total_pads, dim_xz, 4))
    dz_reynolds = np.zeros((total_pads, dim_xz, 4))

    # Calculate global node numbers and their coordinates, pad by pad.
    for pad in range(total_pads):
        # Number of elements in each region (truncating toward zero;
        # arguments are positive so int(...) == floor here).
        total_e_x_track[pad] = int(
            total_e_x_film * (length_track_rad[pad] / arc_length_rad[pad])
        )
        total_e_x_dam[pad] = total_e_x_film - total_e_x_track[pad]
        total_e_z_dam[pad] = int(
            total_e_z_film
            * (0.5 * (axial_length[pad] - axial_length_track[pad]) / axial_length[pad])
        )
        total_e_z_track[pad] = total_e_z_film - 2 * total_e_z_dam[pad]

        # Element size in each region.
        if abs(depth_track[pad]) < 1.0e-6:
            dx_track = 0.0
            dx_track_rad = 0.0
            dx_edge = 0.0
            dx_edge_rad = 0.0
            dx_dam = length_dam[pad] / total_e_x_dam[pad]
            dx_dam_rad = (arc_length_rad[pad] - length_track_rad[pad]) / total_e_x_dam[
                pad
            ]
            dz_track = 0.0
            dz_edge = 0.0
            dz_dam = axial_length_dam[pad] / total_e_z_dam[pad]
        else:
            dx_track = (length_track[pad] - 0.005 * pad_length[pad]) / (
                total_e_x_track[pad] - 1
            )
            dx_track_rad = (length_track_rad[pad] - 0.005 * arc_length_rad[pad]) / (
                total_e_x_track[pad] - 1
            )
            dx_edge = 0.005 * pad_length[pad]
            dx_edge_rad = 0.005 * arc_length_rad[pad]
            dx_dam = length_dam[pad] / total_e_x_dam[pad]
            dx_dam_rad = (arc_length_rad[pad] - length_track_rad[pad]) / total_e_x_dam[
                pad
            ]
            dz_track = (axial_length_track[pad] - 0.01 * axial_length[pad]) / (
                total_e_z_track[pad] - 2
            )
            dz_edge = 0.005 * axial_length[pad]
            if total_e_z_dam[pad] == 0:
                dz_dam = 0.0
            else:
                dz_dam = axial_length_dam[pad] / total_e_z_dam[pad]

        # dx_edge / dx_edge_rad / dz_edge are computed here but never
        # used in this routine (the edge node sits at the track length); kept
        # above only to mirror the source.
        del dx_edge, dx_edge_rad

        n = 0
        # Circumferential sweep.
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

            # Axial sweep.
            for j in range(1, total_e_z_film + 1 + 1):
                if j <= total_e_z_dam[pad] + 1:
                    z1 = (j - 1) * dz_dam
                elif (
                    j == total_e_z_dam[pad] + 2
                    and j < total_e_z_dam[pad] + total_e_z_track[pad]
                ):
                    z1 = axial_length_dam[pad] + dz_edge
                elif (
                    j > total_e_z_dam[pad] + 2
                    and j <= total_e_z_dam[pad] + total_e_z_track[pad]
                ):
                    z1 = (
                        axial_length_dam[pad]
                        + dz_edge
                        + (j - total_e_z_dam[pad] - 2) * dz_track
                    )
                elif (
                    j == total_e_z_dam[pad] + total_e_z_track[pad] + 1
                    and j > total_e_z_dam[pad] + 2
                ):
                    z1 = axial_length_dam[pad] + axial_length_track[pad]
                else:
                    z1 = (
                        axial_length_dam[pad]
                        + axial_length_track[pad]
                        + (j - total_e_z_dam[pad] - total_e_z_track[pad] - 1) * dz_dam
                    )
                n_index_reynolds[n] = n
                x_reynolds[pad, n] = x1
                x_reynolds_rad[pad, n] = x2
                z_reynolds[pad, n] = z1
                n += 1

    # Nodal connectivity (same for all pads). First element (0-based index 0).
    e_index_reynolds[0] = 0
    node_i_reynolds[0] = 0
    node_j_reynolds[0] = node_i_reynolds[0] + total_e_z_film + 1
    node_k_reynolds[0] = node_j_reynolds[0] + 1
    node_l_reynolds[0] = node_i_reynolds[0] + 1

    # Remaining elements.
    for e in range(1, total_e_reynolds):
        e_index_reynolds[e] = e
        if e % total_e_z_film == 0:
            # Element moves to the next circumferential layer.
            node_i_reynolds[e] = node_j_reynolds[e - total_e_z_film]
        else:
            node_i_reynolds[e] = node_l_reynolds[e - 1]
        node_j_reynolds[e] = node_i_reynolds[e] + total_e_z_film + 1
        node_k_reynolds[e] = node_j_reynolds[e] + 1
        node_l_reynolds[e] = node_i_reynolds[e] + 1

    # Bandwidth: largest node-number difference, sampled from element 0.
    bandwidth_reynolds = node_k_reynolds[0] - node_i_reynolds[0] + 1

    # Element sizes and shape-function derivatives at the element centre.
    for pad in range(total_pads):
        for e in range(total_e_reynolds):
            ei = e_index_reynolds[e]
            length = abs(
                x_reynolds[pad, node_j_reynolds[ei]]
                - x_reynolds[pad, node_i_reynolds[ei]]
            )
            width = abs(
                z_reynolds[pad, node_l_reynolds[ei]]
                - z_reynolds[pad, node_i_reynolds[ei]]
            )
            e_length_reynolds[pad, ei] = length
            e_width_reynolds[pad, ei] = width
            # Trailing axis slots 0..3 hold the four derivative components.
            dx_reynolds[pad, ei, 0] = -1.0 / (2.0 * length)
            dx_reynolds[pad, ei, 1] = 1.0 / (2.0 * length)
            dx_reynolds[pad, ei, 2] = 1.0 / (2.0 * length)
            dx_reynolds[pad, ei, 3] = -1.0 / (2.0 * length)
            dz_reynolds[pad, ei, 0] = -1.0 / (2.0 * width)
            dz_reynolds[pad, ei, 1] = -1.0 / (2.0 * width)
            dz_reynolds[pad, ei, 2] = 1.0 / (2.0 * width)
            dz_reynolds[pad, ei, 3] = 1.0 / (2.0 * width)

    return {
        "total_e_x_track": total_e_x_track,
        "total_e_z_track": total_e_z_track,
        "total_e_x_dam": total_e_x_dam,
        "total_e_z_dam": total_e_z_dam,
        "total_e_reynolds": total_e_reynolds,
        "e_index_reynolds": e_index_reynolds,
        "node_i_reynolds": node_i_reynolds,
        "node_j_reynolds": node_j_reynolds,
        "node_k_reynolds": node_k_reynolds,
        "node_l_reynolds": node_l_reynolds,
        "e_length_reynolds": e_length_reynolds,
        "e_width_reynolds": e_width_reynolds,
        "total_n_reynolds": total_n_reynolds,
        "n_index_reynolds": n_index_reynolds,
        "x_reynolds": x_reynolds,
        "z_reynolds": z_reynolds,
        "x_reynolds_rad": x_reynolds_rad,
        "dx_reynolds": dx_reynolds,
        "dz_reynolds": dz_reynolds,
        "bandwidth_reynolds": bandwidth_reynolds,
    }


def mesh_deform(
    total_pads,
    dim_xy2,
    pad_thickness,
    pad_length,
    length_track,
    length_dam,
    depth_track,
    total_e_x_film,
    total_e_y_pad,
    total_e_x_track,
    total_e_x_dam,
):
    """Generate the 2-D pad-deformation mesh in the ``x``-``y`` plane.

    Quadrilateral mesh through the solid pad (circumferential ``x`` by radial
    ``y``). It circumferentially matches the other meshes. Only generated when
    pad deformation is active.

    Parameters
    ----------
    total_pads : int
        Number of pads.
    dim_xy2 : int
        Allocated array size, ``2 * (total_e_x_film + 1) * (total_e_y_pad + 1)``.
    pad_thickness : float
        Radial pad thickness, m (single scalar, same for all pads).
    pad_length : array_like
        Per-pad circumferential pad length, m (shape ``(total_pads,)``).
    length_track : array_like
        Per-pad circumferential track length, m.
    length_dam : array_like
        Per-pad circumferential dam length, m.
    depth_track : array_like
        Per-pad track depth, m; ``|depth| < 1e-6`` means smooth.
    total_e_x_film : int
        Number of circumferential elements.
    total_e_y_pad : int
        Number of radial elements through the pad.
    total_e_x_track, total_e_x_dam : array_like
        Per-pad circumferential region element counts (shape ``(total_pads,)``).
        Read only; typically supplied from :func:`mesh_reynolds`.

    Returns
    -------
    dict
        Keys:

        ``total_e_pad`` : int
            Total elements, ``total_e_x_film * total_e_y_pad``.
        ``e_index_pad`` : ndarray of int, shape ``(dim_xy2,)``
            Identity element-number map.
        ``node_1_pad``, ``node_2_pad``, ``node_3_pad``, ``node_4_pad`` :
        ndarray of int, shape ``(dim_xy2,)``
            0-based node ids of each element's four corners.
        ``total_n_pad`` : int
            Total nodes, ``(total_e_x_film+1)*(total_e_y_pad+1)``.
        ``n_index_pad`` : ndarray of int, shape ``(dim_xy2,)``
            Identity node-number map.
        ``x_pad`` : ndarray of float, shape ``(total_pads, dim_xy2)``
            Per-pad circumferential nodal coordinate, m.
        ``y_pad`` : ndarray of float, shape ``(dim_xy2,)``
            Radial nodal coordinate, m (shared across pads).
        ``bandwidth_deform`` : int
            Half-bandwidth of the deformation system (with 2 DOF/node factor).
    """
    total_e_x_track = np.asarray(total_e_x_track, dtype=np.int64)
    total_e_x_dam = np.asarray(total_e_x_dam, dtype=np.int64)

    total_n_pad = (total_e_x_film + 1) * (total_e_y_pad + 1)
    total_e_pad = total_e_x_film * total_e_y_pad

    n_index_pad = np.zeros(dim_xy2, dtype=np.int64)
    e_index_pad = np.zeros(dim_xy2, dtype=np.int64)
    node_1_pad = np.zeros(dim_xy2, dtype=np.int64)
    node_2_pad = np.zeros(dim_xy2, dtype=np.int64)
    node_3_pad = np.zeros(dim_xy2, dtype=np.int64)
    node_4_pad = np.zeros(dim_xy2, dtype=np.int64)
    x_pad = np.zeros((total_pads, dim_xy2))
    y_pad = np.zeros(dim_xy2)

    for pad in range(total_pads):
        if abs(depth_track[pad]) < 1.0e-6:
            dx_track = 0.0
            dx_edge = 0.0
        else:
            dx_track = (length_track[pad] - 0.005 * pad_length[pad]) / (
                total_e_x_track[pad] - 1
            )
            dx_edge = 0.005 * pad_length[pad]
        dx_dam = length_dam[pad] / total_e_x_dam[pad]
        del dx_edge  # computed above but unused in this routine.

        n = 0
        for i in range(1, total_e_x_film + 1 + 1):
            if i <= total_e_x_track[pad]:
                x1 = (i - 1) * dx_track
            elif i == total_e_x_track[pad] + 1:
                x1 = length_track[pad]
            else:
                x1 = length_track[pad] + (i - total_e_x_track[pad] - 1) * dx_dam

            for j in range(1, total_e_y_pad + 1 + 1):
                y1 = (j - 1) * pad_thickness / total_e_y_pad
                n_index_pad[n] = n
                x_pad[pad, n] = x1
                y_pad[n] = y1
                n += 1

    # Nodal connectivity (same for all pads). First element (0-based index 0).
    e_index_pad[0] = 0
    node_1_pad[0] = 0
    node_2_pad[0] = node_1_pad[0] + total_e_y_pad + 1
    node_3_pad[0] = node_2_pad[0] + 1
    node_4_pad[0] = node_1_pad[0] + 1

    for e in range(1, total_e_pad):
        e_index_pad[e] = e
        if e % total_e_y_pad == 0:
            node_1_pad[e] = node_2_pad[e - total_e_y_pad]
        else:
            node_1_pad[e] = node_4_pad[e - 1]
        node_2_pad[e] = node_1_pad[e] + total_e_y_pad + 1
        node_3_pad[e] = node_2_pad[e] + 1
        node_4_pad[e] = node_1_pad[e] + 1

    bandwidth_deform = (node_3_pad[0] - node_1_pad[0] + 1) * 2

    return {
        "total_e_pad": total_e_pad,
        "e_index_pad": e_index_pad,
        "node_1_pad": node_1_pad,
        "node_2_pad": node_2_pad,
        "node_3_pad": node_3_pad,
        "node_4_pad": node_4_pad,
        "total_n_pad": total_n_pad,
        "n_index_pad": n_index_pad,
        "x_pad": x_pad,
        "y_pad": y_pad,
        "bandwidth_deform": bandwidth_deform,
    }


def _film_element_heights(
    pad,
    depth_track,
    thickness_bl,
    h_n1,
    total_e_y_trackbl,
    total_e_y_trackcore,
    total_e_y_dambl,
    total_e_y_damcore,
):
    """Return film element heights for the four through-film bands.

    Shared logic used by both :func:`mesh_energy` and :func:`mesh_3d` to size
    the boundary-layer / core elements of the track (pocket) and dam regions.

    Parameters
    ----------
    pad : int
        0-based pad index.
    depth_track : array_like
        Per-pad track depth (shape ``(total_pads,)``); ``|.| < 1e-6`` means
        smooth.
    thickness_bl : float
        Boundary-layer thickness at the current circumferential station.
    h_n1 : float
        Film thickness at the current circumferential station.
    total_e_y_trackbl, total_e_y_trackcore, total_e_y_dambl,
    total_e_y_damcore : array_like
        Per-pad through-film element counts (shape ``(total_pads,)``).

    Returns
    -------
    tuple of float
        ``(e_track_bl, e_track_core, e_dam_bl, e_dam_core)`` element heights.
    """
    if abs(depth_track[pad]) < 1.0e-6:
        # Smooth pad: no pocket region.
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
        # Pocketed pad.
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


def mesh_energy(
    total_pads,
    dim_xy,
    pad_thickness,
    leading_angle_rad,
    cp,
    arc_length_rad,
    pad_length,
    offset,
    preload,
    length_track,
    length_dam,
    length_track_rad,
    depth_track,
    xj,
    yj,
    total_e_x_film,
    total_e_y_film,
    total_e_y_pad,
    total_e_x_track,
    total_e_x_dam,
    weight_e,
    weight_h,
):
    """Generate the 2-D energy-equation mesh in the ``x``-``y`` plane.

    Spans the solid pad and the lubricant film. The through-film elements are
    split between dam and pocket regions, with a fraction concentrated in
    boundary layers near the two solid walls. The initial film thickness is
    evaluated assuming zero tilt and deformation.

    Parameters
    ----------
    total_pads : int
        Number of pads.
    dim_xy : int
        Allocated array size,
        ``(total_e_x_film + 1) * (total_e_y_film + total_e_y_pad + 1)``.
    pad_thickness : float
        Radial pad thickness, m.
    leading_angle_rad : array_like
        Per-pad pad leading-edge angle, radians (shape ``(total_pads,)``).
    cp : array_like
        Per-pad assembled clearance, m.
    arc_length_rad : array_like
        Per-pad pad arc length, radians.
    pad_length : array_like
        Per-pad circumferential pad length, m.
    offset : array_like
        Per-pad pivot offset fraction.
    preload : array_like
        Per-pad preload.
    length_track, length_dam, length_track_rad : array_like
        Per-pad track/dam circumferential lengths (m, m, rad).
    depth_track : array_like
        Per-pad track depth, m.
    xj, yj : float
        Journal-centre displacement in the global ``x`` / ``y`` directions,
        m.
    total_e_x_film : int
        Circumferential film element count.
    total_e_y_film : int
        Through-film element count.
    total_e_y_pad : int
        Through-pad element count.
    total_e_x_track, total_e_x_dam : array_like
        Per-pad circumferential region element counts (shape ``(total_pads,)``).
    weight_e : float
        Fraction of through-film elements placed in each boundary layer.
    weight_h : float
        Boundary-layer thickness as a fraction of the local film thickness.

    Returns
    -------
    dict
        Keys:

        ``total_e_y_trackbl``, ``total_e_y_trackcore``, ``total_e_y_dambl``,
        ``total_e_y_damcore`` : ndarray of int, shape ``(total_pads,)``
            Per-pad through-film element counts for the four bands.
        ``total_e_energy`` : int
            Total elements,
            ``total_e_x_film * (total_e_y_pad + total_e_y_film)``.
        ``e_index_energy`` : ndarray of int, shape ``(dim_xy,)``
            Identity element-number map.
        ``node_1_energy``, ``node_2_energy``, ``node_3_energy``,
        ``node_4_energy`` : ndarray of int, shape ``(dim_xy,)``
            0-based node ids of each element's four corners.
        ``total_n_energy`` : int
            Total nodes,
            ``(total_e_x_film+1)*(total_e_y_pad+total_e_y_film+1)``.
        ``n_index_energy`` : ndarray of int, shape ``(dim_xy,)``
            Identity node-number map.
        ``x_energy``, ``y_energy`` : ndarray of float,
        shape ``(total_pads, dim_xy)``
            Per-pad circumferential / radial nodal coordinates, m.
        ``bandwidth_energy`` : int
            Half-bandwidth of the energy system.
    """
    total_e_x_track = np.asarray(total_e_x_track, dtype=np.int64)
    total_e_x_dam = np.asarray(total_e_x_dam, dtype=np.int64)

    total_n_energy = (total_e_x_film + 1) * (total_e_y_pad + total_e_y_film + 1)
    total_e_energy = total_e_x_film * (total_e_y_pad + total_e_y_film)

    # Number of elements concentrated near each solid wall (boundary layer).
    # Truncate toward zero.
    total_e_bl = int(weight_e * total_e_y_film)

    total_e_y_trackbl = np.zeros(total_pads, dtype=np.int64)
    total_e_y_trackcore = np.zeros(total_pads, dtype=np.int64)
    total_e_y_dambl = np.zeros(total_pads, dtype=np.int64)
    total_e_y_damcore = np.zeros(total_pads, dtype=np.int64)

    n_index_energy = np.zeros(dim_xy, dtype=np.int64)
    e_index_energy = np.zeros(dim_xy, dtype=np.int64)
    node_1_energy = np.zeros(dim_xy, dtype=np.int64)
    node_2_energy = np.zeros(dim_xy, dtype=np.int64)
    node_3_energy = np.zeros(dim_xy, dtype=np.int64)
    node_4_energy = np.zeros(dim_xy, dtype=np.int64)
    x_energy = np.zeros((total_pads, dim_xy))
    y_energy = np.zeros((total_pads, dim_xy))

    for pad in range(total_pads):
        # Number of elements in each through-film region.
        if abs(depth_track[pad]) < 1.0e-6:
            total_e_y_trackbl[pad] = 0
            total_e_y_trackcore[pad] = 0
            total_e_y_dambl[pad] = total_e_bl
            total_e_y_damcore[pad] = total_e_y_film - 2 * total_e_bl
        else:
            total_e_y_trackbl[pad] = total_e_bl
            # Integer division, truncating toward zero (numerator >= 0).
            total_e_y_trackcore[pad] = (total_e_y_film - 3 * total_e_bl) // 2
            total_e_y_dambl[pad] = total_e_bl
            total_e_y_damcore[pad] = (
                total_e_y_film
                - 2 * total_e_bl
                - total_e_y_trackbl[pad]
                - total_e_y_trackcore[pad]
            )

        # Circumferential element sizes.
        if abs(depth_track[pad]) < 1.0e-6:
            dx_track = 0.0
            dx_track_rad = 0.0
            dx_edge = 0.0
            dx_edge_rad = 0.0
        else:
            dx_track = (length_track[pad] - 0.005 * pad_length[pad]) / (
                total_e_x_track[pad] - 1
            )
            dx_track_rad = (length_track_rad[pad] - 0.005 * arc_length_rad[pad]) / (
                total_e_x_track[pad] - 1
            )
            dx_edge = 0.005 * pad_length[pad]
            dx_edge_rad = 0.005 * arc_length_rad[pad]
        dx_dam = length_dam[pad] / total_e_x_dam[pad]
        dx_dam_rad = (arc_length_rad[pad] - length_track_rad[pad]) / total_e_x_dam[pad]
        del dx_edge, dx_edge_rad  # computed above but unused here.

        e_height_pad = pad_thickness / total_e_y_pad

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

            # Film thickness in the dam region (zero tilt / deformation).
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
            ) = _film_element_heights(
                pad,
                depth_track,
                thickness_bl,
                h_n1,
                total_e_y_trackbl,
                total_e_y_trackcore,
                total_e_y_dambl,
                total_e_y_damcore,
            )

            for j in range(1, total_e_y_pad + total_e_y_film + 1 + 1):
                if j <= total_e_y_pad + 1:
                    # Solid pad.
                    y1 = (j - 1) * e_height_pad
                elif (
                    j > total_e_y_pad + 1
                    and j <= total_e_y_pad + total_e_y_trackbl[pad] + 1
                ):
                    # Pocket boundary layer.
                    y1 = pad_thickness + (j - total_e_y_pad - 1) * e_film_track_bl
                elif (
                    j > total_e_y_pad + total_e_y_trackbl[pad] + 1
                    and j
                    <= total_e_y_pad
                    + total_e_y_trackbl[pad]
                    + total_e_y_trackcore[pad]
                    + 1
                ):
                    # Pocket core.
                    y1 = (
                        pad_thickness
                        + thickness_bl
                        + (j - total_e_y_pad - total_e_y_trackbl[pad] - 1)
                        * e_film_track_core
                    )
                elif (
                    j
                    > total_e_y_pad
                    + total_e_y_trackbl[pad]
                    + total_e_y_trackcore[pad]
                    + 1
                    and j
                    <= total_e_y_pad
                    + total_e_y_trackbl[pad]
                    + total_e_y_trackcore[pad]
                    + total_e_y_dambl[pad]
                    + 1
                ):
                    # Lower dam boundary layer.
                    y1 = (
                        pad_thickness
                        + depth_track[pad]
                        + (
                            j
                            - total_e_y_pad
                            - total_e_y_trackbl[pad]
                            - total_e_y_trackcore[pad]
                            - 1
                        )
                        * e_film_dam_bl
                    )
                elif (
                    j
                    > total_e_y_pad
                    + total_e_y_trackbl[pad]
                    + total_e_y_trackcore[pad]
                    + total_e_y_dambl[pad]
                    + 1
                    and j
                    <= total_e_y_pad
                    + total_e_y_trackbl[pad]
                    + total_e_y_trackcore[pad]
                    + total_e_y_dambl[pad]
                    + total_e_y_damcore[pad]
                    + 1
                ):
                    # Dam core.
                    y1 = (
                        pad_thickness
                        + depth_track[pad]
                        + thickness_bl
                        + (
                            j
                            - total_e_y_pad
                            - total_e_y_trackbl[pad]
                            - total_e_y_trackcore[pad]
                            - total_e_y_dambl[pad]
                            - 1
                        )
                        * e_film_dam_core
                    )
                else:
                    # Upper boundary layer.
                    y1 = (
                        pad_thickness
                        + depth_track[pad]
                        + h_n1
                        - thickness_bl
                        + (
                            j
                            - total_e_y_pad
                            - total_e_y_trackbl[pad]
                            - total_e_y_trackcore[pad]
                            - total_e_y_dambl[pad]
                            - total_e_y_damcore[pad]
                            - 1
                        )
                        * e_film_dam_bl
                    )
                n_index_energy[n] = n
                x_energy[pad, n] = x1
                y_energy[pad, n] = y1
                n += 1

    # Nodal connectivity (same for all pads). First element (0-based index 0).
    layer = total_e_y_film + total_e_y_pad
    e_index_energy[0] = 0
    node_1_energy[0] = 0
    node_2_energy[0] = node_1_energy[0] + total_e_y_pad + total_e_y_film + 1
    node_3_energy[0] = node_2_energy[0] + 1
    node_4_energy[0] = node_1_energy[0] + 1

    for e in range(1, total_e_energy):
        e_index_energy[e] = e
        if e % layer == 0:
            node_1_energy[e] = node_2_energy[e - layer]
        else:
            node_1_energy[e] = node_4_energy[e - 1]
        node_2_energy[e] = node_1_energy[e] + total_e_y_pad + total_e_y_film + 1
        node_3_energy[e] = node_2_energy[e] + 1
        node_4_energy[e] = node_1_energy[e] + 1

    bandwidth_energy = node_3_energy[0] - node_1_energy[0] + 1

    return {
        "total_e_y_trackbl": total_e_y_trackbl,
        "total_e_y_trackcore": total_e_y_trackcore,
        "total_e_y_dambl": total_e_y_dambl,
        "total_e_y_damcore": total_e_y_damcore,
        "total_e_energy": total_e_energy,
        "e_index_energy": e_index_energy,
        "node_1_energy": node_1_energy,
        "node_2_energy": node_2_energy,
        "node_3_energy": node_3_energy,
        "node_4_energy": node_4_energy,
        "total_n_energy": total_n_energy,
        "n_index_energy": n_index_energy,
        "x_energy": x_energy,
        "y_energy": y_energy,
        "bandwidth_energy": bandwidth_energy,
    }


def mesh_3d(
    total_pads,
    dim_yf,
    dim_z,
    dim_xy,
    dim_xz,
    dim_3d,
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
):
    """Generate the 3-D film mesh and the node-matching maps.

    The 3-D film mesh (circumferential ``x`` by radial ``y`` by axial ``z``) is
    used only to transfer data between the two orthogonal 2-D meshes (the
    Reynolds ``x``-``z`` mesh and the energy ``x``-``y`` mesh); it is not used
    for finite-element computation. Because the node relationship is the same
    for every pad, only pad #1 is used to build the matching maps. For a
    pressure-dam pad the mesh covers some solid region as well.

    Parameters
    ----------
    total_pads : int
        Number of pads.
    dim_yf, dim_z, dim_xy, dim_xz, dim_3d : int
        Allocated array dimensions (see module docstring).
    total_e_x_film, total_e_y_film, total_e_z_film : int
        Film element counts in the three directions.
    total_e_x_track, total_e_z_track, total_e_x_dam, total_e_z_dam :
    array_like
        Per-pad circumferential/axial region element counts
        (shape ``(total_pads,)``).
    total_e_y_trackbl, total_e_y_trackcore, total_e_y_dambl,
    total_e_y_damcore : array_like
        Per-pad through-film band element counts (shape ``(total_pads,)``),
        from :func:`mesh_energy`.
    axial_length : array_like
        Per-pad total axial length, m.
    pad_thickness : float
        Radial pad thickness, m.
    leading_angle_rad, cp, arc_length_rad, pad_length, offset, preload :
    array_like
        Per-pad geometry (rad / m / fractions), as in
        :func:`mesh_energy`.
    length_track, length_track_rad, depth_track, length_dam,
    axial_length_track, axial_length_dam : array_like
        Per-pad track/dam dimensions.
    xj, yj : float
        Journal-centre displacements, m.
    total_n_reynolds : int
        Number of Reynolds-mesh nodes (from :func:`mesh_reynolds`).
    n_index_reynolds : array_like
        Reynolds node-number map (shape ``(dim_xz,)``).
    x_reynolds, z_reynolds : array_like
        Reynolds nodal coordinates, shape ``(total_pads, dim_xz)``.
    total_n_energy : int
        Number of energy-mesh nodes (from :func:`mesh_energy`).
    n_index_energy : array_like
        Energy node-number map (shape ``(dim_xy,)``).
    x_energy, y_energy : array_like
        Energy nodal coordinates, shape ``(total_pads, dim_xy)``.
    weight_h : float
        Boundary-layer thickness fraction of the local film thickness.

    Returns
    -------
    dict
        Keys:

        ``total_n_3d`` : int
            Total 3-D nodes,
            ``(total_e_x_film+1)*(total_e_y_film+1)*(total_e_z_film+1)``.
        ``n_index_3d`` : ndarray of int, shape ``(dim_3d,)``
            Identity node-number map (0-based).
        ``x_3d``, ``y_3d``, ``z_3d`` : ndarray of float,
        shape ``(total_pads, dim_3d)``
            Per-pad 3-D nodal coordinates, m.
        ``match_nodes_xz`` : ndarray of int, shape ``(dim_xz, dim_yf)``
            For each Reynolds node (first index, 0-based), the matching 3-D
            node ids stacked along the second axis (slot ``k`` filled for
            ``k = 0, 1, ...``; unused slots hold ``-1``).
        ``match_nodes_xy`` : ndarray of int, shape ``(dim_xy, dim_z)``
            For each film energy node (first index, 0-based), the matching 3-D
            node ids stacked along the second axis (unused slots hold ``-1``).

    Notes
    -----
    The ``match_nodes_*`` arrays are shaped ``(dim_xz, dim_yf)`` and
    ``(DimXY, DimZ)`` with the first index running over Reynolds / energy node
    numbers and the second index ``k`` accumulating matches. Here both axes are
    0-based natural; unused fill slots hold ``-1`` (``0`` is a valid node id).
    """
    total_e_x_track = np.asarray(total_e_x_track, dtype=np.int64)
    total_e_z_track = np.asarray(total_e_z_track, dtype=np.int64)
    total_e_x_dam = np.asarray(total_e_x_dam, dtype=np.int64)
    total_e_z_dam = np.asarray(total_e_z_dam, dtype=np.int64)

    total_n_3d = (total_e_x_film + 1) * (total_e_y_film + 1) * (total_e_z_film + 1)

    n_index_3d = np.zeros(dim_3d, dtype=np.int64)
    x_3d = np.zeros((total_pads, dim_3d))
    y_3d = np.zeros((total_pads, dim_3d))
    z_3d = np.zeros((total_pads, dim_3d))

    match_nodes_xz = np.full((dim_xz, dim_yf), -1, dtype=np.int64)
    match_nodes_xy = np.full((dim_xy, dim_z), -1, dtype=np.int64)

    # Triple-nested 3-D node construction + the two matching loops live in the
    # JIT kernel; full case mesh_3d drops from ~31 s (pure Python) to <0.5 s.
    mesh_3d_jit(
        int(total_pads),
        int(total_e_x_film),
        int(total_e_y_film),
        int(total_e_z_film),
        total_e_x_track,
        np.ascontiguousarray(total_e_z_track, dtype=np.int64),
        total_e_x_dam,
        np.ascontiguousarray(total_e_z_dam, dtype=np.int64),
        np.ascontiguousarray(total_e_y_trackbl, dtype=np.int64),
        np.ascontiguousarray(total_e_y_trackcore, dtype=np.int64),
        np.ascontiguousarray(total_e_y_dambl, dtype=np.int64),
        np.ascontiguousarray(total_e_y_damcore, dtype=np.int64),
        np.ascontiguousarray(axial_length, dtype=np.float64),
        float(pad_thickness),
        np.ascontiguousarray(leading_angle_rad, dtype=np.float64),
        np.ascontiguousarray(cp, dtype=np.float64),
        np.ascontiguousarray(arc_length_rad, dtype=np.float64),
        np.ascontiguousarray(pad_length, dtype=np.float64),
        np.ascontiguousarray(offset, dtype=np.float64),
        np.ascontiguousarray(preload, dtype=np.float64),
        np.ascontiguousarray(length_track, dtype=np.float64),
        np.ascontiguousarray(length_track_rad, dtype=np.float64),
        np.ascontiguousarray(depth_track, dtype=np.float64),
        np.ascontiguousarray(length_dam, dtype=np.float64),
        np.ascontiguousarray(axial_length_track, dtype=np.float64),
        np.ascontiguousarray(axial_length_dam, dtype=np.float64),
        float(xj),
        float(yj),
        int(total_n_reynolds),
        np.ascontiguousarray(n_index_reynolds, dtype=np.int64),
        np.ascontiguousarray(x_reynolds, dtype=np.float64),
        np.ascontiguousarray(z_reynolds, dtype=np.float64),
        int(total_n_energy),
        np.ascontiguousarray(n_index_energy, dtype=np.int64),
        np.ascontiguousarray(x_energy, dtype=np.float64),
        np.ascontiguousarray(y_energy, dtype=np.float64),
        float(weight_h),
        n_index_3d,
        x_3d,
        y_3d,
        z_3d,
        match_nodes_xz,
        match_nodes_xy,
    )

    return {
        "total_n_3d": total_n_3d,
        "n_index_3d": n_index_3d,
        "x_3d": x_3d,
        "y_3d": y_3d,
        "z_3d": z_3d,
        "match_nodes_xz": match_nodes_xz,
        "match_nodes_xy": match_nodes_xy,
    }
