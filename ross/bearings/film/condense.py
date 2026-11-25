"""Condensation of pad degrees of freedom at any whirl frequency.

Each pad ``k`` contributes matrices on ``[u; eta_k]``: the ``n_u`` shared
(collar) coordinates first, then its own ``n_e`` pad coordinates (tilts,
pivot displacement). With ``Z_k(s) = K_k + s C_k + s^2 M_k``::

    D(s) = sum_k [ Z_uu - Z_ue Z_ee^-1 Z_eu ]_k
    K(w) = Re D(i w),   C(w) = Im D(i w) / w   (w > 0)

At ``w = 0``, with ``Kb = K_ee``::

    K(0) = K_uu - K_ue Kb^-1 K_eu
    C(0) = C_uu - C_ue Kb^-1 K_eu - K_ue Kb^-1 C_eu + K_ue Kb^-1 C_ee Kb^-1 K_eu

High-frequency limits (``Cb = C_ee``)::

    M_ee = 0:  K_inf = K_uu - K_ue Cb^-1 C_eu - C_ue Cb^-1 K_eu + C_ue Cb^-1 K_ee Cb^-1 C_eu
               C_inf = C_uu - C_ue Cb^-1 C_eu
    M_ee > 0:  K_inf = K_uu - C_ue M_ee^-1 C_eu,   C_inf = C_uu

The frozen values ``sum_k K_uu``, ``sum_k C_uu`` (pad coordinates held fixed)
are frequency independent and are not a limit of ``D``.
"""

import numpy as np

__all__ = [
    "dynamic_matrix",
    "condense",
    "condense_static",
    "condense_limit",
]


def _pads(K, C, M):
    """Normalise per-pad inputs to lists of 2-D arrays."""

    def as_list(X):
        if X is None:
            return None
        if isinstance(X, np.ndarray) and X.ndim == 2:
            return [np.asarray(X, dtype=float)]
        return [np.asarray(x, dtype=float) for x in X]

    K, C = as_list(K), as_list(C)
    M = as_list(M) if M is not None else [np.zeros_like(k) for k in K]
    if not (len(K) == len(C) == len(M)):
        raise ValueError("K, C and M must have one matrix per pad")
    for k, c, m in zip(K, C, M, strict=True):
        if not (k.shape == c.shape == m.shape and k.shape[0] == k.shape[1]):
            raise ValueError("per-pad K, C and M must be square and of equal shape")
    return K, C, M


def _blocks(X, n_u):
    return X[:n_u, :n_u], X[:n_u, n_u:], X[n_u:, :n_u], X[n_u:, n_u:]


def dynamic_matrix(K, C, M, n_u, s):
    """Condensed dynamic matrix ``D(s)`` on the shared coordinates.

    Parameters
    ----------
    K, C, M : array_like or sequence of array_like
        Per-pad stiffness, damping and mass matrices on ``[u; eta_k]`` (one
        2-D array, or a sequence/stack of them). ``M`` may be None (no pad
        inertia).
    n_u : int
        Number of shared coordinates (first rows/columns).
    s : complex or array_like
        Laplace variable(s), for example ``1j * w``.

    Returns
    -------
    numpy.ndarray
        ``(n_u, n_u)`` complex matrix, or ``(len(s), n_u, n_u)`` for an array
        of ``s``.

    Examples
    --------
    >>> import numpy as np
    >>> K = np.array([[2.0, 1.0], [1.0, 1.0]])
    >>> C = np.eye(2)
    >>> complex(dynamic_matrix(K, C, None, 1, 0.0)[0, 0])
    (1+0j)
    """
    Ks, Cs, Ms = _pads(K, C, M)
    s_arr = np.atleast_1d(np.asarray(s, dtype=complex))
    D = np.zeros((s_arr.size, n_u, n_u), dtype=complex)
    for k, c, m in zip(Ks, Cs, Ms, strict=True):
        Z = (
            k[None]
            + s_arr[:, None, None] * c[None]
            + (s_arr**2)[:, None, None] * m[None]
        )
        Zuu, Zue, Zeu, Zee = (
            Z[:, :n_u, :n_u],
            Z[:, :n_u, n_u:],
            Z[:, n_u:, :n_u],
            Z[:, n_u:, n_u:],
        )
        if Zee.shape[-1]:
            D += Zuu - Zue @ np.linalg.solve(Zee, Zeu)
        else:
            D += Zuu
    return D if np.ndim(s) else D[0]


def condense_static(K, C, n_u):
    """Condensed stiffness and damping at ``w = 0`` (closed form).

    Parameters
    ----------
    K, C : array_like or sequence of array_like
        Per-pad stiffness and damping on ``[u; eta_k]``.
    n_u : int
        Number of shared coordinates.

    Returns
    -------
    K0, C0 : numpy.ndarray
        ``(n_u, n_u)`` limits of ``K(w)`` and ``C(w)`` as ``w -> 0``.

    Examples
    --------
    >>> import numpy as np
    >>> K0, C0 = condense_static(np.array([[2.0, 1.0], [1.0, 1.0]]), np.eye(2), 1)
    >>> float(K0[0, 0]), float(C0[0, 0])
    (1.0, 2.0)
    """
    Ks, Cs, _ = _pads(K, C, None)
    K0 = np.zeros((n_u, n_u))
    C0 = np.zeros((n_u, n_u))
    for k, c in zip(Ks, Cs, strict=True):
        Kuu, Kue, Keu, Kee = _blocks(k, n_u)
        Cuu, Cue, Ceu, Cee = _blocks(c, n_u)
        if Kee.shape[0] == 0:
            K0 += Kuu
            C0 += Cuu
            continue
        A = np.linalg.solve(Kee, Keu)  # Kb^-1 K_eu
        B = np.linalg.solve(Kee, Ceu)  # Kb^-1 C_eu
        Kb_inv_Kue_T = np.linalg.solve(Kee.T, Kue.T).T  # K_ue Kb^-1
        K0 += Kuu - Kue @ A
        C0 += Cuu - Cue @ A - Kue @ B + Kb_inv_Kue_T @ Cee @ A
    return K0, C0


def condense_limit(K, C, M, n_u):
    """Condensed stiffness and damping as ``w -> infinity`` (closed form).

    Parameters
    ----------
    K, C, M : array_like or sequence of array_like
        Per-pad matrices on ``[u; eta_k]``; ``M`` may be None. A pad whose
        ``M_ee`` is zero uses the no-inertia limit, otherwise ``M_ee`` must be
        non-singular.
    n_u : int
        Number of shared coordinates.

    Returns
    -------
    K_inf, C_inf : numpy.ndarray
        ``(n_u, n_u)`` limits of ``K(w)`` and ``C(w)``.

    Examples
    --------
    >>> import numpy as np
    >>> K = np.array([[2.0, 1.0], [1.0, 1.0]])
    >>> C = np.array([[1.0, 0.5], [0.5, 1.0]])
    >>> Ki, Ci = condense_limit(K, C, None, 1)
    >>> float(Ci[0, 0])
    0.75
    """
    Ks, Cs, Ms = _pads(K, C, M)
    Ki = np.zeros((n_u, n_u))
    Ci = np.zeros((n_u, n_u))
    for k, c, m in zip(Ks, Cs, Ms, strict=True):
        Kuu, Kue, Keu, Kee = _blocks(k, n_u)
        Cuu, Cue, Ceu, Cee = _blocks(c, n_u)
        Mee = m[n_u:, n_u:]
        if Kee.shape[0] == 0:
            Ki += Kuu
            Ci += Cuu
            continue
        if not np.any(m):
            A = np.linalg.solve(Cee, Ceu)  # Cb^-1 C_eu
            G = np.linalg.solve(Cee.T, Cue.T).T  # C_ue Cb^-1
            Ki += Kuu - Kue @ A - G @ Keu + G @ Kee @ A
            Ci += Cuu - Cue @ A
        else:
            if np.any(m[:n_u, :]) or np.any(m[:, :n_u]):
                raise ValueError("the shared coordinates must carry no mass in M")
            if np.linalg.matrix_rank(Mee) < Mee.shape[0]:
                raise ValueError("M_ee must be zero or non-singular")
            Ki += Kuu - Cue @ np.linalg.solve(Mee, Ceu)
            Ci += Cuu
    return Ki, Ci


def condense(K, C, M, n_u, frequency):
    """Condensed stiffness and damping on a frequency axis.

    ``w = 0`` uses :func:`condense_static` (no division by ``w``).

    Parameters
    ----------
    K, C, M : array_like or sequence of array_like
        Per-pad matrices on ``[u; eta_k]``; ``M`` may be None.
    n_u : int
        Number of shared coordinates.
    frequency : float or array_like
        Whirl frequencies ``w >= 0``, rad/s.

    Returns
    -------
    Kw, Cw : numpy.ndarray
        ``(n_f, n_u, n_u)`` stiffness and damping (``(n_u, n_u)`` for a
        scalar frequency).

    Examples
    --------
    >>> import numpy as np
    >>> K = np.array([[2.0, 1.0], [1.0, 1.0]])
    >>> Kw, Cw = condense(K, np.eye(2), None, 1, [0.0, 1.0])
    >>> np.round(Kw[:, 0, 0], 12).tolist(), np.round(Cw[:, 0, 0], 12).tolist()
    ([1.0, 1.5], [2.0, 1.5])
    """
    w = np.atleast_1d(np.asarray(frequency, dtype=float))
    if np.any(w < 0.0) or not np.all(np.isfinite(w)):
        raise ValueError("frequencies must be finite and non-negative")
    Kw = np.empty((w.size, n_u, n_u))
    Cw = np.empty((w.size, n_u, n_u))
    zero = w == 0.0
    if zero.any():
        K0, C0 = condense_static(K, C, n_u)
        Kw[zero], Cw[zero] = K0, C0
    if (~zero).any():
        D = dynamic_matrix(K, C, M, n_u, 1j * w[~zero])
        Kw[~zero] = D.real
        Cw[~zero] = D.imag / w[~zero][:, None, None]
    if np.ndim(frequency) == 0:
        return Kw[0], Cw[0]
    return Kw, Cw
