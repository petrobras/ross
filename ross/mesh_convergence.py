"""Mesh convergence module.

This module searches the shaft discretization of a rotor whose lowest natural
frequencies are within a relative tolerance of a converged reference. See
:py:meth:`ross.Rotor.run_mesh_convergence`.
"""

import warnings

import numpy as np

from ross.coupling_element import CouplingElement
from ross.results import MeshConvergenceResults

__all__ = ["mesh_convergence", "MeshSearch"]


class MeshSearch:
    """Evaluate shaft discretizations of a rotor against a converged reference.

    A discretization is the number of equal elements each shaft interval (the
    length between consecutive shaft nodes of the rotor) is split into (see
    :py:meth:`ross.Rotor.refine`). Two one-parameter families are searched:

    - uniform: every element is at most ``L_max / m`` long, where ``L_max`` is
      the longest interval of the rotor;
    - graded: every interval gets ``t`` elements per local bending wavelength
      at the highest reference frequency, so thin sections get shorter
      elements than thick ones.

    Parameters
    ----------
    rotor : ross.Rotor
        Rotor whose shaft elements are the coarsest discretization.
    frequencies : int
        Number of natural frequencies that must converge.
    speed : float
        Rotor speed of the modal analyses (rad/s).
    max_elements : int
        Maximum number of shaft elements of a discretization.
    """

    def __init__(self, rotor, frequencies, speed, max_elements):
        self.rotor = rotor
        self.frequencies = frequencies
        self.speed = speed
        self.max_elements = max_elements

        self.first_node = min(elm.n for elm in rotor.shaft_elements)
        n_intervals = max(elm.n for elm in rotor.shaft_elements) - self.first_node + 1
        self.lengths = np.zeros(n_intervals)
        self.layers = np.zeros(n_intervals, dtype=int)
        self.coupling = np.zeros(n_intervals, dtype=bool)
        self.bending_stiffness = np.zeros(n_intervals)
        self.mass_per_length = np.zeros(n_intervals)
        for elm in rotor.shaft_elements:
            i = elm.n - self.first_node
            self.layers[i] += 1
            if isinstance(elm, CouplingElement):
                self.coupling[i] = True
                continue
            self.lengths[i] = elm.L
            o_d = (elm.odl + elm.odr) / 2
            i_d = (elm.idl + elm.idr) / 2
            self.bending_stiffness[i] += elm.material.E * np.pi * (o_d**4 - i_d**4) / 64
            self.mass_per_length[i] += elm.material.rho * np.pi * (o_d**2 - i_d**2) / 4

        self.wn = {}
        self.families = {}
        self.reference = None
        self.reference_m = None
        self.wn_reference = None

    def _subdivisions(self, elements_per_length):
        subdivisions = np.ceil(self.lengths * elements_per_length - 1e-9)
        subdivisions = np.maximum(subdivisions, 1).astype(int)
        subdivisions[self.coupling] = 1
        return subdivisions

    def uniform(self, m):
        """Return the subdivisions whose elements are at most ``L_max / m`` long."""
        return self._subdivisions(m / self.lengths.max())

    def graded(self, t):
        """Return the subdivisions with ``t`` elements per bending wavelength."""
        w = self.wn_reference.max()
        wavelength = np.ones(len(self.lengths))
        valid = ~self.coupling
        wavelength[valid] = (
            2
            * np.pi
            * (self.bending_stiffness[valid] / (self.mass_per_length[valid] * w**2))
            ** 0.25
        )
        return self._subdivisions(t / wavelength)

    def n_elements(self, subdivisions):
        """Return the number of shaft elements of a discretization."""
        return int(np.sum(self.layers * subdivisions))

    def evaluate(self, subdivisions, family):
        """Return the lowest natural frequencies of a discretization.

        Parameters
        ----------
        subdivisions : array
            Number of elements of each shaft interval.
        family : str
            Family of the discretization, stored for the plot.

        Returns
        -------
        wn : array
            The ``frequencies`` lowest natural frequencies in ascending order
            (rad/s). It is shorter when the discretization has fewer modes.
        """
        key = tuple(subdivisions)
        self.families.setdefault(key, family)
        if key not in self.wn:
            rotor = self.rotor.refine(subdivisions)
            n_modes = min(self.frequencies + 4, (rotor.ndof - 2) // 2)
            with warnings.catch_warnings():
                # ModalResults classifies each mode by its components above 8%
                # of the norm, and fine meshes spread a mode over so many
                # degrees of freedom that none passes, which divides 0 by 0.
                warnings.filterwarnings(
                    "ignore",
                    message="invalid value encountered in scalar divide",
                    category=RuntimeWarning,
                )
                modal = rotor.run_modal(self.speed, num_modes=2 * n_modes)
            self.wn[key] = np.sort(modal.wn)[: self.frequencies]

        return self.wn[key]

    def compare(self, subdivisions, family):
        """Return the relative error of each natural frequency of a discretization.

        The k-th lowest natural frequency is compared with the k-th lowest
        frequency of the reference.

        Parameters
        ----------
        subdivisions : array
            Number of elements of each shaft interval.
        family : str
            Family of the discretization, stored for the plot.

        Returns
        -------
        wn : array
            Natural frequencies (rad/s).
        error : array
            Relative error of each natural frequency. It is infinite when the
            discretization has fewer modes than the reference.
        """
        wn = self.evaluate(subdivisions, family)
        if len(wn) < self.frequencies:
            return np.full(self.frequencies, np.nan), np.full(self.frequencies, np.inf)
        return wn, np.abs(wn / self.wn_reference - 1)

    def converged(self, subdivisions, family, rtol):
        """Return True when every natural frequency is within rtol of the reference."""
        return bool(np.max(self.compare(subdivisions, family)[1]) <= rtol)

    def find_reference(self, rtol):
        """Halve the element length until the frequencies change less than rtol.

        Parameters
        ----------
        rtol : float
            Relative change of the natural frequencies between two consecutive
            uniform discretizations that defines the converged reference.
        """
        m = 1
        subdivisions = self.uniform(m)
        wn = self.evaluate(subdivisions, "uniform")

        while True:
            finer = self.uniform(2 * m)
            if self.n_elements(finer) > self.max_elements:
                warnings.warn(
                    "The natural frequencies did not converge to the reference "
                    f"tolerance within max_elements={self.max_elements}. The "
                    "finest discretization evaluated is the reference."
                )
                break

            m *= 2
            wn_finer = self.evaluate(finer, "uniform")
            converged = (
                len(wn) == len(wn_finer) and np.max(np.abs(wn / wn_finer - 1)) <= rtol
            )
            subdivisions, wn = finer, wn_finer
            if converged:
                break

        if len(wn) < self.frequencies:
            raise ValueError(
                f"The rotor has fewer than {self.frequencies} natural frequencies "
                "to converge."
            )

        self.reference = subdivisions
        self.reference_m = m
        self.wn_reference = wn

    def search_uniform(self, rtol):
        """Return the uniform discretization with the fewest elements within rtol."""
        if self.converged(self.uniform(1), "uniform", rtol):
            return self.uniform(1)

        low, high = 1, self.reference_m
        while high - low > 1:
            middle = (low + high) // 2
            if self.converged(self.uniform(middle), "uniform", rtol):
                high = middle
            else:
                low = middle

        return self.uniform(high)

    def search_graded(self, rtol, iterations=12):
        """Return the graded discretization with the fewest elements within rtol.

        Returns None when the graded discretizations reach ``max_elements``
        before converging.
        """
        low, high = 0.0, 1.0
        while True:
            subdivisions = self.graded(high)
            if self.n_elements(subdivisions) > self.max_elements:
                return None
            if self.converged(subdivisions, "graded", rtol):
                break
            low, high = high, 2 * high

        for _ in range(iterations):
            middle = (low + high) / 2
            if self.converged(self.graded(middle), "graded", rtol):
                high = middle
            else:
                low = middle

        return self.graded(high)


def mesh_convergence(rotor, rtol, frequencies, strategy, speed, max_elements):
    """Find the shaft discretization with converged natural frequencies.

    See :py:meth:`ross.Rotor.run_mesh_convergence` for the parameters.

    Returns
    -------
    results : ross.MeshConvergenceResults
        Mesh convergence results.
    """
    if rtol <= 0:
        raise ValueError("rtol must be greater than zero.")
    if frequencies < 1:
        raise ValueError("frequencies must be at least 1.")
    if strategy not in ("cheapest", "uniform"):
        raise ValueError('strategy must be "cheapest" or "uniform".')

    search = MeshSearch(rotor, frequencies, speed, max_elements)
    search.find_reference(rtol / 10)

    candidates = [search.search_uniform(rtol)]
    if strategy == "cheapest":
        graded = search.search_graded(rtol)
        if graded is not None:
            candidates.append(graded)
    subdivisions = min(candidates, key=search.n_elements)

    wn, _ = search.compare(subdivisions, search.families[tuple(subdivisions)])
    evaluated = list(search.wn)
    errors = [
        np.max(search.compare(np.array(key), search.families[key])[1])
        for key in evaluated
    ]

    return MeshConvergenceResults(
        rotor=rotor.refine(subdivisions),
        subdivisions=subdivisions,
        wn=wn,
        wn_reference=search.wn_reference,
        rtol=rtol,
        elements=[search.n_elements(np.array(key)) for key in evaluated],
        errors=errors,
        families=[search.families[key] for key in evaluated],
        reference_elements=search.n_elements(search.reference),
    )
