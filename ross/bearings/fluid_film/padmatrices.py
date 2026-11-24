"""Unreduced pad matrices of a journal bearing and their exact condensation.

A journal bearing solved at one speed has the journal coordinates
``u = (x, y)`` and, for tilting pads, the coordinates of each pad ``k``: the
tilt ``delta_k`` about its pivot and, with a flexible pivot, the pivot
displacement ``v_k`` along the pivot direction. :class:`PadMatrices` keeps
the stiffness, damping and mass matrix of every pad on ``[u; eta_k]`` and
condenses the pad coordinates exactly at any frequency through
:mod:`ross.bearings.film.condense`::

    D(s) = sum_k [ Z_uu - Z_ue Z_ee^-1 Z_eu ]_k,   Z = K + s C + s^2 M
    K(w) = Re D(i w),   C(w) = Im D(i w) / w

with the closed forms at ``w = 0`` and ``w -> infinity``.
:func:`pad_inertia_about_pivot` gives the pad mass block from the annular
sector of each pad, taken about its pivot.
"""

from dataclasses import dataclass

import numpy as np

__all__ = [
    "PadMatrices",
    "SingularPadStiffness",
    "pad_inertia_about_pivot",
]

# Condition number above which a pad block K_ee has no static condensation.
_SINGULAR_CONDITION = 1e12


class SingularPadStiffness(ValueError):
    """The pad stiffness block ``K_ee`` of a coupled pad is singular.

    Raised by the static (``w = 0``) condensation: a pad that is free to move
    (for example on an unloaded Hertz pivot) has no static stiffness.

    Attributes
    ----------
    pads : list of int
        0-based indices of the singular pads.
    """

    def __init__(self, pads, message):
        super().__init__(message)
        self.pads = list(pads)


def pad_inertia_about_pivot(
    journal_radius,
    pad_clearance,
    pad_thickness,
    pad_arc,
    axial_length,
    density,
    offset,
    pivot_radius=None,
):
    """Mass, inertia about the pivot and tilt-translation coupling of the pads.

    Each pad is an annular sector of bore radius ``R_i = R + C_p``, back
    radius ``R_o = R_i + t``, arc ``beta`` and axial length ``L``. Its pivot
    ``P`` sits at the radius ``r_P`` (default ``R + t``, the lever arm of the
    film moment in the engine) at the angle ``phi = (offset - 0.5) beta``
    from the pad centre line, positive in the direction of the pad
    coordinate (from the leading to the trailing edge). With ``r_g`` the
    radius of the centroid ``G``::

        m   = rho L beta (R_o^2 - R_i^2) / 2
        I_O = m (R_o^2 + R_i^2) / 2                       (about the sector centre)
        r_g = 4 sin(beta/2) (R_o^3 - R_i^3) / (3 beta (R_o^2 - R_i^2))
        I_P = I_O + m r_P^2 - 2 m r_P r_g cos(phi)        (= I_G + m |G - P|^2)
        c   = m r_g sin(phi)

    For the tilt ``delta`` (rotation of the pad about ``P``, positive
    counter-clockwise, the engine's convention) and the pivot displacement
    ``v`` (outwards along the pivot direction ``e_P``), the kinetic energy
    ``1/2 int |v_dot e_P + delta_dot z x (r - r_P)|^2 dm`` is
    ``1/2 [delta_dot, v_dot] [[I_P, c], [c, m]] [delta_dot, v_dot]^T``.
    A centred pivot has no coupling (``c = 0``).

    The sector is centred on the bearing centre: the offset of a preloaded
    pad's centre of curvature (``C_p - C_b``) and the clearance between the
    pivot radius ``R + t`` and the pad back ``R + C_p + t`` are of the order
    of the clearance and are neglected next to ``t`` (a relative change of
    about ``2 C_p / t`` in the parallel-axis term).

    Parameters
    ----------
    journal_radius : float
        Journal radius ``R``, m.
    pad_clearance : float or array_like
        Pad machined clearance ``C_p = C_b / (1 - preload)``, m, per pad.
    pad_thickness : float
        Radial pad thickness ``t``, m.
    pad_arc : float or array_like
        Pad arc ``beta``, rad, per pad.
    axial_length : float or array_like
        Pad axial length ``L``, m, per pad.
    density : float
        Pad material density, kg/m**3.
    offset : float or array_like
        Pivot offset, fraction of the arc from the leading edge, per pad.
    pivot_radius : float, optional
        Pivot radius ``r_P``, m. Default is ``R + t``.

    Returns
    -------
    mass : numpy.ndarray
        Pad mass ``m``, kg.
    inertia : numpy.ndarray
        Moment of inertia about the pivot axis ``I_P``, kg*m**2.
    coupling : numpy.ndarray
        Tilt-translation mass coupling ``c``, kg*m.

    Examples
    --------
    A full ring (``beta = 2 pi``) has its centroid on the axis, so the
    inertia about a point at radius ``r_P`` is ``I_O + m r_P^2``:

    >>> import numpy as np
    >>> m, i_p, c = pad_inertia_about_pivot(1.0, 0.0, 1.0, 2 * np.pi, 1.0, 1.0, 0.5)
    >>> bool(np.isclose(m[0], 3 * np.pi)), bool(np.isclose(i_p[0], 3 * np.pi * 6.5))
    (True, True)
    >>> float(c[0])
    0.0
    """
    radius = float(journal_radius)
    thickness = float(pad_thickness)
    beta, clearance, length, offset = np.broadcast_arrays(
        *(
            np.atleast_1d(np.asarray(a, dtype=float))
            for a in (pad_arc, pad_clearance, axial_length, offset)
        )
    )
    r_p = radius + thickness if pivot_radius is None else float(pivot_radius)
    r_i = radius + clearance
    r_o = r_i + thickness
    area = r_o**2 - r_i**2
    mass = density * length * beta * area / 2.0
    inertia_centre = mass * (r_o**2 + r_i**2) / 2.0
    r_g = 4.0 * np.sin(beta / 2.0) * (r_o**3 - r_i**3) / (3.0 * beta * area)
    phi = (offset - 0.5) * beta
    inertia = inertia_centre + mass * r_p**2 - 2.0 * mass * r_p * r_g * np.cos(phi)
    coupling = mass * r_g * np.sin(phi)
    return mass, inertia, coupling


@dataclass(frozen=True, eq=False)
class PadMatrices:
    """Unreduced stiffness, damping and mass matrices of a journal bearing.

    Entry ``k`` of ``K``, ``C`` and ``M`` is the matrix of pad ``k`` on its
    coordinates ``labels[k]``: the ``n_u = 2`` journal coordinates ``x``,
    ``y`` (m) first, then the pad coordinates, ``tilt_k`` (rad) and, with a
    flexible pivot, ``pivot_k`` (m, outwards along the pivot direction).
    Fixed-geometry pads have no pad coordinates. Summing the journal blocks
    with the pad coordinates held fixed gives the frozen coefficients; the
    methods below condense the pad coordinates instead.

    Pads whose coupling to the journal is zero in both ``K`` and ``C`` (an
    unloaded or fully cavitated pad, whose film blocks the engine sets to 0)
    contribute their journal block only: they are condensed without their
    pad coordinates, which is exact and keeps ``w = 0`` defined.

    Attributes
    ----------
    K, C, M : tuple of numpy.ndarray
        Per-pad stiffness (N/m, N/rad, N*m/rad), damping (same units times
        s) and mass (kg, kg*m, kg*m**2) matrices, read-only.
    labels : tuple of tuple of str
        Coordinate names of each entry.
    pad_mass, pad_inertia, pad_coupling : numpy.ndarray or None
        Pad mass (kg), moment of inertia about the pivot (kg*m**2) and
        tilt-translation coupling (kg*m) of each pad (see
        :func:`pad_inertia_about_pivot`); None for fixed-geometry pads.
    n_u : int
        Number of journal coordinates, 2.

    Examples
    --------
    >>> import numpy as np
    >>> K = np.array([[2.0, 0.0, 1.0], [0.0, 2.0, 0.0], [1.0, 0.0, 1.0]])
    >>> pads = PadMatrices(K=(K,), C=(np.eye(3),), M=(np.zeros((3, 3)),),
    ...                    labels=(("x", "y", "tilt_0"),))
    >>> K0, C0 = pads.condense(0.0)
    >>> float(K0[0, 0]), float(C0[0, 0])
    (1.0, 2.0)
    """

    K: tuple
    C: tuple
    M: tuple
    labels: tuple
    pad_mass: np.ndarray = None
    pad_inertia: np.ndarray = None
    pad_coupling: np.ndarray = None
    n_u: int = 2

    def __post_init__(self):
        for name in ("K", "C", "M"):
            matrices = tuple(np.array(m, dtype=float) for m in getattr(self, name))
            for m in matrices:
                m.setflags(write=False)
            object.__setattr__(self, name, matrices)
        if not (len(self.K) == len(self.C) == len(self.M) == len(self.labels)):
            raise ValueError("K, C, M and labels must have one entry per pad")
        for k, (stiffness, damping, mass, labels) in enumerate(
            zip(self.K, self.C, self.M, self.labels, strict=True)
        ):
            size = len(labels)
            if not (stiffness.shape == damping.shape == mass.shape == (size, size)):
                raise ValueError(
                    f"entry {k}: K, C and M must be {size} x {size} to match its labels"
                )
            if size < self.n_u:
                raise ValueError(
                    f"entry {k} has fewer than n_u = {self.n_u} coordinates"
                )
        object.__setattr__(self, "labels", tuple(tuple(lb) for lb in self.labels))
        for name in ("pad_mass", "pad_inertia", "pad_coupling"):
            value = getattr(self, name)
            if value is not None:
                value = np.array(value, dtype=float)
                value.setflags(write=False)
                object.__setattr__(self, name, value)

    @classmethod
    def from_blocks(
        cls,
        stiffness,
        damping,
        pad_mass=None,
        pad_inertia=None,
        pad_coupling=None,
        k_rotate=None,
        k_pivot=None,
        pivot_direction=None,
    ):
        """Build the pad matrices from the engine's coefficient blocks.

        Rigid pivot, pad ``k`` on ``[x, y, tilt_k]``::

            K_k = [[K_uu^(k), K_ue^(k)], [K_eu^(k), K_ee^(k) + k_rotate]]
            M_k = diag(0, 0, I_P)

        with ``K_uu^(k)`` the pad's journal block (``xxi`` ... ``yyi``),
        ``K_ue = (xdelta, ydelta)``, ``K_eu = (deltax, deltay)`` and
        ``K_ee = deltadelta``; ``C_k`` likewise without ``k_rotate``.
        Flexible pivot, pad ``k`` on ``[x, y, tilt_k, pivot_k]``: the film
        sees the journal relative to the pivot, ``u - v n`` with
        ``n = (cos theta_P, sin theta_P)``, so
        ``K_k = T^T K_film T + diag(0, 0, k_rotate, k_pivot)`` with
        ``T = [[I, 0, -n], [0, 1, 0]]``, and ``M_ee = [[I_P, c], [c, m]]``.
        Without ``pad_inertia`` the pads do not tilt (fixed geometry): they
        have no pad coordinate, or only ``pivot_k`` with a flexible pivot.

        Parameters
        ----------
        stiffness, damping : CoefficientBlock
            Stiffness and damping blocks of
            :func:`~ross.bearings.fluid_film.coefficients.jacobian` and
            :func:`~ross.bearings.fluid_film.coefficients.damping`. Without
            the per-pad journal blocks (``xxi`` is None) the summed journal
            block becomes a first entry with no pad coordinates.
        pad_mass : array_like, optional
            Per-pad mass, kg; needed with ``k_pivot``.
        pad_inertia : array_like, optional
            Per-pad moment of inertia about the pivot, kg*m**2 (see
            :func:`pad_inertia_about_pivot`). Default is None (pads that do
            not tilt).
        pad_coupling : array_like, optional
            Per-pad tilt-translation mass coupling, kg*m. Default is 0.
        k_rotate : array_like, optional
            Per-pad rotational stiffness of the pivot, N*m/rad. Default is 0.
        k_pivot : array_like, optional
            Per-pad pivot translational stiffness, N/m. None (default) for a
            rigid pivot.
        pivot_direction : array_like, optional
            Per-pad angle ``theta_P`` of the pivot, rad (pad leading edge
            plus the pivot position); needed with ``k_pivot``.

        Returns
        -------
        PadMatrices
        """
        n_pads = len(np.atleast_1d(stiffness.deltadelta))

        def per_pad(value, default=0.0):
            value = default if value is None else value
            return np.broadcast_to(np.asarray(value, dtype=float), (n_pads,))

        per_pad_uu = stiffness.xxi is not None and damping.xxi is not None
        tilting = pad_inertia is not None
        flexible = k_pivot is not None
        if flexible and (pad_mass is None or pivot_direction is None):
            raise ValueError("a flexible pivot needs pad_mass and pivot_direction")
        inertia = per_pad(pad_inertia)
        mass = per_pad(pad_mass)
        coupling = per_pad(pad_coupling)
        rotate = per_pad(k_rotate)
        pivot = per_pad(k_pivot)
        direction = per_pad(pivot_direction)

        K, C, M, labels = [], [], [], []
        if not per_pad_uu:
            for out, block in ((K, stiffness), (C, damping)):
                out.append(
                    np.array([[block.xx, block.xy], [block.yx, block.yy]], dtype=float)
                )
            M.append(np.zeros((2, 2)))
            labels.append(("x", "y"))
        keep = [0, 1] + ([2] if tilting else []) + ([3] if flexible else [])
        index = np.ix_(keep, keep)
        for k in range(n_pads):
            n = np.array([np.cos(direction[k]), np.sin(direction[k])])
            T = np.zeros((3, 4))
            T[:2, :2] = np.eye(2)
            T[:2, 3] = -n
            T[2, 2] = 1.0
            for out, block, is_stiffness in ((K, stiffness, True), (C, damping, False)):
                film = np.zeros((3, 3))
                if per_pad_uu:
                    film[:2, :2] = [
                        [block.xxi[k], block.xyi[k]],
                        [block.yxi[k], block.yyi[k]],
                    ]
                film[:2, 2] = (block.xdelta[k], block.ydelta[k])
                film[2, :2] = (block.deltax[k], block.deltay[k])
                film[2, 2] = block.deltadelta[k]
                full = T.T @ film @ T if flexible else np.pad(film, ((0, 1), (0, 1)))
                if is_stiffness:
                    full[2, 2] += rotate[k]
                    full[3, 3] += pivot[k]
                out.append(full[index])
            full_mass = np.zeros((4, 4))
            full_mass[2, 2] = inertia[k]
            full_mass[2, 3] = full_mass[3, 2] = coupling[k]
            full_mass[3, 3] = mass[k]
            M.append(full_mass[index])
            names = ("x", "y", f"tilt_{k}", f"pivot_{k}")
            labels.append(tuple(names[i] for i in keep))
        return cls(
            K=tuple(K),
            C=tuple(C),
            M=tuple(M),
            labels=tuple(labels),
            pad_mass=mass if (tilting or flexible) else None,
            pad_inertia=inertia if tilting else None,
            pad_coupling=coupling if tilting else None,
        )

    def _condensed_inputs(self):
        """Per-entry matrices with the decoupled pad coordinates removed."""
        n = self.n_u
        K, C, M = [], [], []
        for stiffness, damping, mass in zip(self.K, self.C, self.M, strict=True):
            decoupled = not (
                np.any(stiffness[:n, n:])
                or np.any(stiffness[n:, :n])
                or np.any(damping[:n, n:])
                or np.any(damping[n:, :n])
            )
            keep = n if decoupled else stiffness.shape[0]
            K.append(stiffness[:keep, :keep])
            C.append(damping[:keep, :keep])
            M.append(mass[:keep, :keep])
        return K, C, M

    def _check_static(self, K):
        n = self.n_u
        singular = []
        for k, stiffness in enumerate(K):
            block = stiffness[n:, n:]
            if block.size and (
                not np.all(np.isfinite(block))
                or np.linalg.cond(block) > _SINGULAR_CONDITION
            ):
                singular.append(k)
        if singular:
            raise SingularPadStiffness(
                singular,
                f"entries {singular}: the pad stiffness block K_ee is singular "
                "(the pad is free to move, e.g. on an unloaded pivot), so the "
                "static (w = 0) condensation does not exist",
            )

    def impedance(self, s):
        """Condensed dynamic stiffness ``D(s)`` on the journal coordinates.

        Parameters
        ----------
        s : complex or array_like
            Laplace variable(s), 1/s; ``1j * w`` on the imaginary axis. A
            complex ``s = lambda`` evaluates the bearing at a damped rotor
            eigenvalue.

        Returns
        -------
        numpy.ndarray
            ``(2, 2)`` complex matrix ``[[D_xx, D_xy], [D_yx, D_yy]]``, N/m,
            or ``(len(s), 2, 2)``.
        """
        from ross.bearings.film.condense import dynamic_matrix

        K, C, M = self._condensed_inputs()
        return dynamic_matrix(K, C, M, self.n_u, s)

    def condense(self, frequency):
        """Condensed stiffness and damping at whirl frequencies.

        Parameters
        ----------
        frequency : float or array_like
            Whirl frequencies ``w >= 0``, rad/s. ``w = 0`` uses the static
            closed form.

        Returns
        -------
        K, C : numpy.ndarray
            ``(2, 2)`` matrices ``[[kxx, kxy], [kyx, kyy]]`` (N/m) and
            ``[[cxx, cxy], [cyx, cyy]]`` (N*s/m), or ``(n, 2, 2)`` for an
            array of frequencies.

        Raises
        ------
        SingularPadStiffness
            If a frequency is 0 and a coupled pad has a singular ``K_ee``.
        """
        from ross.bearings.film.condense import condense

        K, C, M = self._condensed_inputs()
        if np.any(np.asarray(frequency, dtype=float) == 0.0):
            self._check_static(K)
        return condense(K, C, M, self.n_u, frequency)

    def condense_static(self):
        """Condensed stiffness and damping at ``w = 0`` (closed form).

        Returns
        -------
        K, C : numpy.ndarray
            ``(2, 2)`` matrices, N/m and N*s/m.

        Raises
        ------
        SingularPadStiffness
            If a coupled pad has a singular ``K_ee``.
        """
        from ross.bearings.film.condense import condense_static

        K, C, _ = self._condensed_inputs()
        self._check_static(K)
        return condense_static(K, C, self.n_u)

    def condense_limit(self):
        """Condensed stiffness and damping as ``w -> infinity`` (closed form).

        With pad inertia the limit is ``K_uu - C_ue M_ee^-1 C_eu`` and
        ``C_uu`` (see :func:`ross.bearings.film.condense.condense_limit`).

        This is the asymptote of the condensation, not a stiffness to use in
        a rotor model. It is approached only far above any whirl frequency a
        rotor reaches (thousands of times the speed for a flexible pivot),
        and its stiffness can have the opposite sign of the static value and
        a magnitude hundreds to thousands of times larger.

        Returns
        -------
        K, C : numpy.ndarray
            ``(2, 2)`` matrices, N/m and N*s/m.
        """
        from ross.bearings.film.condense import condense_limit

        K, C, M = self._condensed_inputs()
        return condense_limit(K, C, M, self.n_u)
