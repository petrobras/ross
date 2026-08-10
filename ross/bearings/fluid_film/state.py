"""State objects passed between the solver modules.

The solver threads a lot of data through a lot of routines. Grouping it by
*lifetime* keeps the signatures readable and makes the mutability contract
explicit: every object here is frozen, so a routine that takes one cannot
perturb its caller's state, and the reader knows at a glance that it will not
change under an iteration.

Built once per analysis
-----------------------
``ReynoldsMesh``
    The circumferential/axial (``x``-``z``) film mesh and its connectivity,
    from :func:`~ross.bearings.fluid_film.mesh.mesh_reynolds` and
    :func:`~ross.bearings.fluid_film.mesh.mesh_3d`.
``EnergyMesh``
    The circumferential/radial (``x``-``y``) mesh spanning the film *and* the
    pad solid, used by the full energy equation, from
    :func:`~ross.bearings.fluid_film.mesh.mesh_energy`.
``PadGeometry``
    The pad shapes: arc lengths, pivot positions, clearances, taper and
    pressure-dam pocket dimensions. It also carries the two predicates that
    classify a node against the pocket, :meth:`PadGeometry.node_in_pocket`
    and :meth:`PadGeometry.pocket_edge_is_pad`.
``Lubricant``
    The oil: its two-point viscosity-temperature law (see
    :meth:`Lubricant.viscosity_at`) and its bulk thermal properties.
``Turbulence``
    The transition Reynolds numbers and the Reichardt eddy-viscosity
    constants.

Built once per speed case
-------------------------
``OperatingPoint``
    Bearing type, operating model, surface speed, and the pressure and
    temperature boundary conditions.

Results
-------
``CoefficientBlock``
    One family of dynamic coefficients -- the direct journal 2x2, the pad-tilt
    coupling, and the per-pad direct blocks a flexible pivot needs. The layout
    is the same for stiffness and damping, so
    :func:`~ross.bearings.fluid_film.coefficients.jacobian` and
    :func:`~ross.bearings.fluid_film.coefficients.damping` both return one.

Everything not here -- the film thickness, pressure, temperature and viscosity
fields -- is what the iterations actually update, and still travels through the
orchestrator's state dict.

Indexing follows the package convention: node and element numbers index the
coordinate and field arrays directly, per-pad arrays are shaped
``(total_pads,)`` or ``(total_pads, dim_*)``, and the ``match_nodes_*`` maps
use ``-1`` for unused slots. All values are SI.
"""

from dataclasses import dataclass, fields

import numpy as np

__all__ = [
    "CoefficientBlock",
    "EnergyMesh",
    "Lubricant",
    "OperatingPoint",
    "PadGeometry",
    "ReynoldsMesh",
    "Turbulence",
]


class _FromState:
    """Mixin: build a state object from the orchestrator's state dict."""

    @classmethod
    def from_state(cls, g, names):
        """Build from ``g``, taking each field from ``g[names[field]]``.

        Parameters
        ----------
        g : dict
            Solver state, keyed by the solver-wide parameter names.
        names : dict
            ``{field_name: state_key}`` covering every field of ``cls``.

        Returns
        -------
        object
            An instance of ``cls``.
        """
        return cls(**{f.name: g[names[f.name]] for f in fields(cls)})


@dataclass(frozen=True)
class ReynoldsMesh(_FromState):
    """The ``x``-``z`` film mesh and its connectivity.

    Attributes
    ----------
    dim_x, dim_z, dim_yf, dim_xz, dim_3d : int
        Array dimensions: circumferential, axial and through-film node counts,
        their ``x``-``z`` product, and the 3-D film node count.
    bandwidth : int
        Half-bandwidth of the banded global system.
    total_e_x_film, total_e_y_film, total_e_z_film : int
        Film element counts per direction.
    total_e_y_trackbl, total_e_y_trackcore : numpy.ndarray of int
        Per-pad through-film element counts in the track babbitt / core
        layers, used to pick the pad-surface layer of a node.
    total_nodes, total_elements : int
        Number of active Reynolds nodes and elements.
    n_index, e_index : numpy.ndarray of int
        Active node and element numbers.
    node_i, node_j, node_k, node_l : numpy.ndarray of int
        Element connectivity: the four corner node numbers per element.
    match_nodes_xz : numpy.ndarray of int
        ``[node, j]`` -> the 3-D film node in cross-film column ``j``, with
        ``-1`` marking an unused slot.
    x, z : numpy.ndarray
        Nodal circumferential and axial coordinates, shape
        ``(total_pads, dim_xz)``, in m.
    x_rad : numpy.ndarray
        Nodal circumferential coordinate in rad.
    y_3d : numpy.ndarray
        Radial coordinate of the 3-D film nodes, shape
        ``(total_pads, dim_3d)``.
    e_length, e_width : numpy.ndarray
        Element circumferential length and axial width.
    dx, dz : numpy.ndarray
        Nodal coordinate derivatives, shape ``(total_pads, dim_xz, 4)``.
    """

    dim_x: int
    dim_z: int
    dim_yf: int
    dim_xz: int
    dim_3d: int
    bandwidth: int
    total_e_x_film: int
    total_e_y_film: int
    total_e_z_film: int
    total_e_y_trackbl: np.ndarray
    total_e_y_trackcore: np.ndarray
    total_nodes: int
    total_elements: int
    n_index: np.ndarray
    e_index: np.ndarray
    node_i: np.ndarray
    node_j: np.ndarray
    node_k: np.ndarray
    node_l: np.ndarray
    match_nodes_xz: np.ndarray
    x: np.ndarray
    z: np.ndarray
    x_rad: np.ndarray
    y_3d: np.ndarray
    e_length: np.ndarray
    e_width: np.ndarray
    dx: np.ndarray
    dz: np.ndarray


@dataclass(frozen=True)
class PadGeometry(_FromState):
    """Pad shapes and pivot positions, in SI.

    Every per-pad array is shaped ``(total_pads,)`` and indexed by pad number.

    Attributes
    ----------
    journal_radius, pad_thickness : float
        Journal radius and pad (or shell) thickness, m.
    arc_length_rad : numpy.ndarray
        Pad arc length, rad.
    pad_length, axial_length : numpy.ndarray
        Pad circumferential length and axial length, m.
    leading_angle_rad : numpy.ndarray
        Angular position of the pad leading edge, rad.
    x_pivot_rad, x_pivot : numpy.ndarray
        Pivot position along the pad, in rad and in m from the leading edge.
    cp : numpy.ndarray
        Machined (pad) clearance, m.
    preload, offset : numpy.ndarray
        Pad preload and pivot offset ratio, dimensionless.
    depth_track, length_track, length_track_rad, axial_length_track : numpy.ndarray
        Pressure-dam pocket depth, its circumferential length (in m and rad)
        and its axial length.
    length_dam, axial_length_dam : numpy.ndarray
        Dam circumferential and axial length, m.
    length_pocket, axial_length_pocket : float
        Inlet-groove pocket dimensions, m.
    length_ramp_le, length_ramp_te : numpy.ndarray
        Taper length at the leading and trailing edges, m.
    dh_ramp_le, dh_ramp_te : numpy.ndarray
        Taper depth at the leading and trailing edges, m.
    """

    journal_radius: float
    pad_thickness: float
    arc_length_rad: np.ndarray
    pad_length: np.ndarray
    axial_length: np.ndarray
    leading_angle_rad: np.ndarray
    x_pivot_rad: np.ndarray
    x_pivot: np.ndarray
    cp: np.ndarray
    preload: np.ndarray
    offset: np.ndarray
    depth_track: np.ndarray
    length_track: np.ndarray
    length_track_rad: np.ndarray
    axial_length_track: np.ndarray
    length_dam: np.ndarray
    axial_length_dam: np.ndarray
    length_pocket: float
    axial_length_pocket: float
    length_ramp_le: np.ndarray
    length_ramp_te: np.ndarray
    dh_ramp_le: np.ndarray
    dh_ramp_te: np.ndarray

    def pocket_edge_is_pad(self, x, z, pad):
        """Whether a node on the pocket edge behaves as a pocket node.

        True when the edge coincides with a pad boundary, so the pocket is not
        shrouded there: either the circumferential pad edge inside the pocket's
        axial span, or an axial pad end inside the track's circumferential
        span.

        Parameters
        ----------
        x, z : float
            Circumferential and axial coordinate of the node, m.
        pad : int
            Pad number.

        Returns
        -------
        bool
        """
        on_circumferential_edge = (
            abs(x - self.pad_length[pad]) < 1.0e-6
            and z > self.axial_length_dam[pad]
            and z < self.axial_length_track[pad] + self.axial_length_dam[pad]
        )
        on_axial_end = (
            abs(z) < 1.0e-6 or abs(z - self.axial_length[pad]) < 1.0e-6
        ) and x < self.length_track[pad]
        return on_circumferential_edge or on_axial_end

    def node_in_pocket(self, x, z, pad):
        """Classify a node as pocket (``True``) or dam (``False``).

        Pocket nodes integrate over the full film thickness, dam nodes over the
        reduced one. A node exactly on the boundary is resolved by
        :meth:`pocket_edge_is_pad`.

        Parameters
        ----------
        x, z : float
            Circumferential and axial coordinate of the node, m.
        pad : int
            Pad number.

        Returns
        -------
        bool
        """
        if (
            z > self.axial_length_dam[pad]
            and z < self.axial_length_dam[pad] + self.axial_length_track[pad]
            and x < self.length_track[pad]
        ):
            return True
        if (
            x > self.length_track[pad]
            or z < self.axial_length_dam[pad]
            or z > self.axial_length_dam[pad] + self.axial_length_track[pad]
        ):
            return False
        return self.pocket_edge_is_pad(x, z, pad)


@dataclass(frozen=True)
class EnergyMesh(_FromState):
    """The ``x``-``y`` film-plus-pad mesh used by the energy equation.

    Attributes
    ----------
    dim_xy, dim_xy2 : int
        Node count of the energy mesh, and of the doubled (two-layer) pad
        deformation mesh built on the same circumferential stations.
    bandwidth : int
        Half-bandwidth of the banded energy system.
    total_e_y_pad : int
        Number of radial elements across the pad solid.
    total_nodes, total_elements : int
        Number of active energy nodes and elements.
    n_index, e_index : numpy.ndarray of int
        Active node and element numbers.
    node_1, node_2, node_3, node_4 : numpy.ndarray of int
        Element connectivity: the four corner node numbers per element.
    match_nodes_xy : numpy.ndarray of int
        ``[node, j]`` -> the 3-D film node in cross-film column ``j``, with
        ``-1`` marking an unused slot.
    x, y : numpy.ndarray
        Nodal circumferential and radial coordinates, m. ``y`` runs from the
        back of the pad, through the pad solid, out to the film surface.
    """

    dim_xy: int
    dim_xy2: int
    bandwidth: int
    total_e_y_pad: int
    total_nodes: int
    total_elements: int
    n_index: np.ndarray
    e_index: np.ndarray
    node_1: np.ndarray
    node_2: np.ndarray
    node_3: np.ndarray
    node_4: np.ndarray
    match_nodes_xy: np.ndarray
    x: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class Lubricant(_FromState):
    """Oil properties, in SI.

    The viscosity-temperature law is the two-point Reynolds (exponential) fit
    through ``(temp1, viscosity1)`` and ``(temp2, viscosity2)``.

    Attributes
    ----------
    viscosity1, viscosity2 : float
        Dynamic viscosity at ``temp1`` and ``temp2``, Pa*s.
    temp1, temp2 : float
        The two reference temperatures, K.
    density : float
        Density, kg/m^3.
    cp : float
        Specific heat capacity, J/(kg*K).
    conduct : float
        Thermal conductivity, W/(m*K).
    """

    viscosity1: float
    viscosity2: float
    temp1: float
    temp2: float
    density: float
    cp: float
    conduct: float

    def viscosity_at(self, temp):
        """Viscosity at ``temp`` (K), from the two-point Reynolds law.

        Parameters
        ----------
        temp : float or array_like
            Temperature(s), K.

        Returns
        -------
        float or numpy.ndarray
            Dynamic viscosity, Pa*s.

        Examples
        --------
        >>> oil = Lubricant(2.758e-6, 1.119e-6, 323.0, 353.0, 870.0, 1950.0, 0.13)
        >>> float(oil.viscosity_at(323.0))
        2.758e-06
        """
        beta = np.log(self.viscosity2 / self.viscosity1) / (self.temp2 - self.temp1)
        return self.viscosity1 * np.exp(beta * (temp - self.temp1))


@dataclass(frozen=True)
class OperatingPoint(_FromState):
    """What the bearing is doing, for one speed case. SI.

    Fixed for the whole case: the journal position and the temperature field
    move during the equilibrium and thermal iterations, but none of these do.

    Attributes
    ----------
    bearing_type : str
        Bearing geometry, one of
        :data:`~ross.bearings.fluid_film.constants.BEARING_TYPES`.
    operating_type : str
        Lubrication model, one of
        :data:`~ross.bearings.fluid_film.constants.OPERATING_TYPES`.
    speed_surface : float
        Journal surface speed, m/s.
    temp_supply : float
        Oil supply temperature, K.
    press_supply : float
        Oil supply pressure, Pa.
    press_cavitate : float
        Cavitation pressure, Pa.
    ambient_press1, ambient_press2 : float
        Ambient pressure at the two axial ends, Pa.
    """

    bearing_type: str
    operating_type: str
    speed_surface: float
    temp_supply: float
    press_supply: float
    press_cavitate: float
    ambient_press1: float
    ambient_press2: float


@dataclass(frozen=True)
class CoefficientBlock:
    """One family of dynamic coefficients, in SI.

    Holds either stiffness (N/m) or damping (N*s/m); the layout is the same
    for both, so :func:`~ross.bearings.fluid_film.coefficients.jacobian` and
    :func:`~ross.bearings.fluid_film.coefficients.damping` both return one of these
    and the dynamic reduction consumes them interchangeably.

    Attributes
    ----------
    xx, yx, xy, yy : float
        The direct journal 2x2 block.
    deltax, deltay, xdelta, ydelta, deltadelta : numpy.ndarray
        Coupling between the journal translation and the pad tilt degrees of
        freedom, per pad.
    xxi, yxi, xyi, yyi : numpy.ndarray or None
        Per-pad direct blocks, needed only by
        :func:`~ross.bearings.fluid_film.coefficients.dynamic_reduction_pivot` when
        the pivot is flexible. ``None`` for the rigid-pivot reduction.
    """

    xx: float
    yx: float
    xy: float
    yy: float
    deltax: np.ndarray
    deltay: np.ndarray
    xdelta: np.ndarray
    ydelta: np.ndarray
    deltadelta: np.ndarray
    xxi: np.ndarray = None
    yxi: np.ndarray = None
    xyi: np.ndarray = None
    yyi: np.ndarray = None

    @classmethod
    def from_tuple(cls, values):
        """Build from the flat tuple ``jacobian`` / ``damping`` returns.

        Parameters
        ----------
        values : sequence
            Nine or thirteen coefficients, in signature order.

        Returns
        -------
        CoefficientBlock
        """
        return cls(*values)


@dataclass(frozen=True)
class Turbulence:
    """Turbulence-model constants.

    The film is laminar below ``re_lower`` and fully turbulent above
    ``re_upper``, with a blend in between. The Reichardt constants set the
    eddy-viscosity profile across the film.

    Attributes
    ----------
    re_lower, re_upper : float
        Reynolds numbers bounding the laminar-turbulent transition.
    reichardt_delta, reichardt_kappa : float
        Reichardt eddy-viscosity constants.
    scale_factor_exponent : float
        Exponent of the turbulent shear scaling factor.
    """

    re_lower: float
    re_upper: float
    reichardt_delta: float
    reichardt_kappa: float
    scale_factor_exponent: float
