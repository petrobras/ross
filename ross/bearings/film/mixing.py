"""Hot-oil carry-over mixing at the pad inlets.

The conventional mixing model (Ettles 1969): the oil entering pad ``k`` at its
leading edge is the hot oil carried over from the trailing edge of the upstream
pad ``u(k)`` (the previous pad in the ring; pad 0 is fed by the last pad)
mixed with fresh oil, weighted by the flows::

    T_in,k = w_k T_out,u + (1 - w_k) T_fresh,   w_k = min(q_c,u / q_in,k, 1)

with ``q_c,u`` the carried-over flow of the upstream pad (the fraction
``lambda`` of its trailing-edge flow) and ``q_in,k`` the inflow of pad ``k``.
When the carry-over exceeds the inlet flow (``w_k = 1``) the inlet temperature
is the upstream outlet temperature, and the hot oil that enters the pad is
``min(q_c,u, q_in,k)``.

This module is the one implementation of the rule:

- :func:`upstream_pads` gives ``u(k)``;
- :func:`carryover_weights` gives ``w_k`` and :func:`carryover_flows` the hot
  oil ``min(q_c,u, q_in,k)``, the form in which the mixing is linear in the
  inlet temperatures;
- :func:`temp_mixing_carryover` evaluates ``T_in`` from given outlet
  temperatures; the journal engine
  (``ross.bearings.fluid_film.thd.temp_mixing_carryover``) imports it from here.
"""

import numpy as np

__all__ = [
    "upstream_pads",
    "carryover_weights",
    "carryover_flows",
    "temp_mixing_carryover",
]


def upstream_pads(n_pads):
    """Upstream pad of each pad of a ring: ``k - 1``, the first fed by the last.

    Parameters
    ----------
    n_pads : int
        Number of pads in the ring (1: the pad feeds itself).

    Returns
    -------
    numpy.ndarray of int
        ``u[k]``, the pad whose trailing-edge oil reaches pad ``k``.

    Examples
    --------
    >>> upstream_pads(4).tolist()
    [3, 0, 1, 2]
    >>> upstream_pads(1).tolist()
    [0]
    """
    return (np.arange(int(n_pads)) - 1) % int(n_pads)


def carryover_weights(q_in, q_carryover, upstream=None):
    """Hot-oil fraction of each pad inlet, ``w_k = min(q_c,u / q_in,k, 1)``.

    Parameters
    ----------
    q_in : array_like
        Leading-edge inflow of each pad, m^3/s.
    q_carryover : array_like
        Carry-over flow leaving each pad (``lambda`` times its trailing-edge
        flow), m^3/s.
    upstream : array_like of int, optional
        Upstream pad of each pad. Default :func:`upstream_pads`.

    Returns
    -------
    numpy.ndarray
        ``w_k`` in ``[0, 1]``; 0 for a pad without inflow.

    Examples
    --------
    >>> carryover_weights([2.0, 1.0], [1.0, 3.0]).tolist()
    [1.0, 1.0]
    >>> carryover_weights([2.0, 4.0], [1.0, 1.0]).tolist()
    [0.5, 0.25]
    """
    q_in = np.asarray(q_in, dtype=float)
    q_c = np.asarray(q_carryover, dtype=float)
    if upstream is None:
        upstream = upstream_pads(q_in.size)
    w = np.zeros(q_in.size)
    flowing = q_in > 0.0
    w[flowing] = np.minimum(q_c[upstream][flowing] / q_in[flowing], 1.0)
    return w


def carryover_flows(q_in, q_carryover, upstream=None):
    """Hot oil entering each pad from its upstream pad, ``min(q_c,u, q_in,k)``.

    Parameters
    ----------
    q_in : array_like
        Leading-edge inflow of each pad, m^3/s.
    q_carryover : array_like
        Carry-over flow leaving each pad, m^3/s.
    upstream : array_like of int, optional
        Upstream pad of each pad. Default :func:`upstream_pads`.

    Returns
    -------
    numpy.ndarray
        Hot oil entering each pad, m^3/s.

    Examples
    --------
    >>> carryover_flows([2.0, 1.0], [1.5, 0.5]).tolist()
    [0.5, 1.0]
    """
    q_in = np.asarray(q_in, dtype=float)
    q_c = np.asarray(q_carryover, dtype=float)
    if upstream is None:
        upstream = upstream_pads(q_in.size)
    return np.minimum(q_c[upstream], q_in)


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

    The inlet temperature of pad ``p`` mixes the hot oil carried over from the
    upstream pad (:func:`upstream_pads`) with fresh oil at ``temp_sump``,
    weighted by flow, ``(q_c,u T_out,u + (q_in,p - q_c,u) T_sump) / q_in,p``.
    If the carry-over exceeds the pad inlet flow the inlet is the upstream
    outlet. For ``operating_type`` equal to ``"axial_flow"`` or
    ``"oil_ring_lubricated"`` (journal lubrication models) every inlet is the
    sump temperature. The journal engine uses this function for every bearing
    type except the spray bar.

    Parameters
    ----------
    total_pads : int
        Number of pads.
    operating_type : str
        Lubrication model (only ``"axial_flow"`` and
        ``"oil_ring_lubricated"`` change the result).
    q_in, q_carryover : array_like of float
        Per-pad inlet and carry-over flow rates, m^3/s.
    temp_sump : float
        Fresh-oil temperature, K.
    temp_outlet : array_like of float
        Per-pad outlet temperature, K.
    temp_inlet : array_like of float
        Per-pad inlet temperature (a copy is updated and returned).

    Returns
    -------
    numpy.ndarray
        Updated per-pad inlet temperature.

    Examples
    --------
    >>> temp_mixing_carryover(2, "flooded", [2.0, 2.0], [1.0, 1.0], 300.0,
    ...                       [340.0, 360.0], [0.0, 0.0]).tolist()
    [330.0, 320.0]
    """
    temp_inlet = np.array(temp_inlet, dtype=float, copy=True)
    upstream = upstream_pads(total_pads)
    for p in range(total_pads):
        u = upstream[p]
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
