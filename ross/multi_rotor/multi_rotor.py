"""Multi Rotor module.

This module defines the MultiRotor class which is used to couple two rotors
through a pair of gears, allowing lateral-torsional coupled analyses of
geared rotor-dynamic systems.
"""

import numpy as np
import pandas as pd
from re import search
from scipy.optimize import newton
from copy import deepcopy as copy

import ross as rs
from ross.element import Element
from ross.rotor_assembly import Rotor
from ross.units import Q_, check_units
from ross.utils import make_speed_array
from ross.results import CriticalSpeedResults, ForcedResponseResults, SummaryResults

from .gear_element import GearElement
from .mesh import Mesh
from .results import BacklashResults

__all__ = ["MultiRotor"]


class MultiRotor(Rotor):
    """A class representing a multi-rotor system.

    This class creates a system comprising multiple rotors, with the specified
    driving rotor and driven rotor. For systems with more than two rotors,
    multiple multi-rotors can be nested.

    Parameters
    ----------
    driving_rotor : rs.Rotor
        The driving rotor object.
    driven_rotor : rs.Rotor
        The driven rotor object.
    coupled_nodes : tuple of int
        Tuple specifying the coupled nodes, where the first node corresponds to
        the driving rotor and the second node corresponds to the driven rotor.
    gear_mesh_stiffness : float, optional
        Directly specify the stiffness of the gear mesh.
        If not provided, it can be calculated automatically
        when using `GearElementTVMS` instead of `GearElement`.
        Default is None.
    update_mesh_stiffness : bool, optional
        Applicable only when using `GearElementTVMS`.
        If True, the gear mesh stiffness is recalculated
        at each time step. If False, the maximum stiffness
        value is used throughout the simulation.
        Default is False.
    square_varying_stiffness : dict, optional
        Dictionary to enable and configure a square-shaped time-varying
        mesh stiffness. Keys are:

        - enable : bool
            If True, a square-shaped time-varying mesh stiffness is used.
        - amplitude_ratio : float
            Ratio of the stiffness amplitude based on the mean value of the
            mesh stiffness.

        All keys are required.
        Default is `{"enable": False, "amplitude_ratio": 0}`.
    backlash : dict, optional
        Dictionary to enable and configure the backlash model between the
        coupled gears. Keys are:

        - enable : bool
            If True, the backlash model is used.
        - initial_value : float
            Initial backlash of the gear pair (m).
        - error_amp : float
            Error amplitude used in the backlash force model.
        - smooth_operator : bool
            If True, a smooth (hyperbolic tangent) approximation is used for
            the backlash force.
        - sigma : float
            Parameter related to the regularization of the smooth approach.

        All keys are required.
        Default is `{"enable": False, "initial_value": 0.0, "error_amp": 0.0,
        "smooth_operator": False, "sigma": 1e4}`.
    mesh_damping_ratio : float, pint.Quantity, optional
        Damping ratio ζ of the gear mesh (dimensionless), giving the mesh damping
        coefficient :math:`c_m = 2 \\zeta \\sqrt{k_m M_{eq}}`. Without backlash,
        :math:`c_m` is added to the global damping matrix; with backlash, it is
        used in the nonlinear mesh force. Default is 0.0, no mesh damping
        (ROSS 3.0 used 0.07 for backlash models).
    orientation_angle : float, pint.Quantity, optional
        The angle between the line of gear centers and x-axis. Default is 0.0 rad.
    position : {'above', 'below'}, optional
        The relative position of the driven rotor with respect to the driving rotor
        when plotting the multi-rotor. Default is 'above'.
    min_w : float, optional
        The minimum angular velocity of the multi-rotor system. Default is None.
    max_w : float, optional
        The maximum angular velocity of the multi-rotor system. Default is None.
    rated_w : float, optional
        The rated angular velocity of the multi-rotor system. Default is None.
    tag : str, optional
        A tag to identify the multi-rotor. Default is None.

    Returns
    -------
    rotor : rs.Rotor
        The created multi-rotor object.

    References
    ----------
    The mesh model and its references (Kaplan et al., 2013; Yi et al., 2019;
    Visnadi, 2022) are described in the MultiRotor tutorial, with the full
    entries in ``docs/references.bib``.

    Examples
    --------
    >>> import ross as rs
    >>> steel = rs.materials.steel
    >>> # Rotor 1:
    >>> L1 = [0.1, 4.24, 1.16, 0.3]
    >>> d1 = [0.3, 0.3, 0.22, 0.22]
    >>> shaft1 = [
    ...     rs.ShaftElement(
    ...         L=L1[i],
    ...         idl=0.0,
    ...         odl=d1[i],
    ...         material=steel,
    ...     )
    ...     for i in range(len(L1))
    ... ]
    >>> generator = rs.DiskElement(n=1, m=525.7, Id=16.1, Ip=32.2)
    >>> disk = rs.DiskElement(n=2, m=116.04, Id=3.115, Ip=6.23)
    >>> gear1 = rs.GearElement(
    ...     n=4, m=726.4, Id=56.95, Ip=113.9, n_teeth=328,
    ...     pitch_diameter=1.1, pr_angle=rs.Q_(22.5, 'deg'),
    ... )
    >>> bearing1 = rs.BearingElement(n=0, kxx=183.9e6, kyy=200.4e6, cxx=3e3)
    >>> bearing2 = rs.BearingElement(n=3, kxx=183.9e6, kyy=200.4e6, cxx=3e3)
    >>> rotor1 = rs.Rotor(shaft1, [generator, disk, gear1], [bearing1, bearing2],)

    >>> # Rotor 2:
    >>> L2 = [0.3, 5, 0.1]
    >>> d2 = [0.15, 0.15, 0.15]
    >>> shaft2 = [
    ...     rs.ShaftElement(
    ...         L=L2[i],
    ...         idl=0.0,
    ...         odl=d2[i],
    ...         material=steel,
    ...     )
    ...     for i in range(len(L2))
    ... ]
    >>> gear2 = rs.GearElement(
    ...     n=0, m=5, Id=0.002, Ip=0.004, n_teeth=23,
    ...     base_diameter=0.03567 * 2, pr_angle=rs.Q_(22.5, 'deg'),
    ... )
    >>> turbine = rs.DiskElement(n=2, m=7.45, Id=0.0745, Ip=0.149)
    >>> bearing3 = rs.BearingElement(n=1, kxx=10.1e6, kyy=41.6e6, cxx=3e3)
    >>> bearing4 = rs.BearingElement(n=3, kxx=10.1e6, kyy=41.6e6, cxx=3e3)
    >>> rotor2 = rs.Rotor(shaft2, [gear2, turbine], [bearing3, bearing4],)

    >>> # Multi rotor:
    >>> multi_rotor = rs.MultiRotor(
    ...     rotor1,
    ...     rotor2,
    ...     coupled_nodes=(4, 0),
    ...     gear_mesh_stiffness=1e8,
    ...     orientation_angle=0.0,
    ...     position="below"
    ... )
    >>> modal = multi_rotor.run_modal(speed=0)
    >>> modal.wd[0] # doctest: +ELLIPSIS
    74.160...
    """

    @check_units
    def __init__(
        self,
        driving_rotor,
        driven_rotor,
        coupled_nodes,
        gear_mesh_stiffness=None,
        update_mesh_stiffness=False,
        square_varying_stiffness={"enable": False, "amplitude_ratio": 0},
        backlash={
            "enable": False,
            "initial_value": 0.0,
            "error_amp": 0.0,
            "smooth_operator": False,
            "sigma": 1e4,
        },
        mesh_damping_ratio=0.0,
        orientation_angle=0.0,
        position="above",
        min_w=None,
        max_w=None,
        rated_w=None,
        tag=None,
    ):
        self.rotors = {"driving": copy(driving_rotor), "driven": copy(driven_rotor)}

        self.coupled_nodes = tuple(int(node) for node in coupled_nodes)
        self.gear_mesh_stiffness = gear_mesh_stiffness
        self.mesh_damping_ratio = mesh_damping_ratio
        self.update_mesh_stiffness = update_mesh_stiffness
        self.square_varying_stiffness = copy(square_varying_stiffness)
        self.backlash = copy(backlash)
        self.orientation_angle = orientation_angle
        self.position = position

        R1 = copy(driving_rotor)
        R2 = copy(driven_rotor)

        gear_1 = self._locate_gear_by_node(R1, self.coupled_nodes[0])
        gear_2 = self._locate_gear_by_node(R2, self.coupled_nodes[1])

        if gear_1 is None or gear_2 is None:
            raise TypeError("Each rotor needs a GearElement in the coupled nodes!")

        self.coupled_tags = (gear_1.tag, gear_2.tag)

        gear1_plot = self._locate_gear_on_plot(R1, gear_1)
        gear2_plot = self._locate_gear_on_plot(R2, gear_2)

        if position == "above":
            ymax = max(y for y in gear1_plot["y"] if y is not None)
            ymin = min(y for y in gear2_plot["y"] if y is not None)
            self.dy_pos = +abs(ymax - ymin)
        else:
            ymax = max(y for y in gear2_plot["y"] if y is not None)
            ymin = min(y for y in gear1_plot["y"] if y is not None)
            self.dy_pos = -abs(ymax - ymin)

        idx1 = R1.nodes.index(gear_1.n)
        idx2 = R2.nodes.index(gear_2.n)
        self.dz_pos = float(R1.nodes_pos[idx1] - R2.nodes_pos[idx2])

        R1_max_node = max([*R1.nodes, *R1.link_nodes])
        R2_min_node = min([*R2.nodes, *R2.link_nodes])
        driven_offset = 0

        if R1_max_node >= R2_min_node:
            driven_offset = R1_max_node + 1
            for elm in R2.elements:
                elm.n += driven_offset

                if getattr(elm, "n_link", None) is not None:
                    elm.n_link += driven_offset

                elm.tag = None

        self.driven_nodes = [int(n + driven_offset) for n in R2.nodes]

        shaft_elements = [*R1.shaft_elements, *R2.shaft_elements]
        disk_elements = [*R1.disk_elements, *R2.disk_elements]
        bearing_elements = [*R1.bearing_elements, *R2.bearing_elements]
        point_mass_elements = [*R1.point_mass_elements, *R2.point_mass_elements]

        motor_element = None
        if R1.motor_element is not None and R2.motor_element is not None:
            motor_element = R1.motor_element
            raise Warning("""Rotor supports only one motor element.
            Using the motor element of the driving rotor.\n""")
        elif R1.motor_element is not None:
            motor_element = R1.motor_element
        elif R2.motor_element is not None:
            motor_element = R2.motor_element

        super().__init__(
            shaft_elements,
            disk_elements,
            bearing_elements,
            point_mass_elements,
            motor_element,
            min_w=min_w,
            max_w=max_w,
            rated_w=rated_w,
            tag=tag,
        )

        # Create mesh
        self.mesh = Mesh(
            gear_1,
            gear_2,
            gear_mesh_stiffness=gear_mesh_stiffness,
            square_varying_stiffness=square_varying_stiffness,
            damping_ratio=mesh_damping_ratio,
            backlash=backlash,
            orientation_angle=orientation_angle,
        )

        self._mesh_stiffness = self.mesh.stiffness

        self.K_coupling = self.compute_coupling_matrix()
        self.coupling_matrix = self.compute_coupling_matrix()

        if self.mesh.backlash:
            self.add_coupling_stiffness = lambda K0: K0
            self.add_coupling_damping = lambda C0: C0
        else:
            self.add_coupling_stiffness = self.K_mesh
            self.add_coupling_damping = self.C_mesh

        self._gear_ratio = self._node_gear_ratio(self.driven_nodes[0])

    def split_elements(self, elements, node_scale=1):
        """Split elements into driving and driven rotor elements.

        This method takes a list of elements and separates them into two lists:
        one for the driving rotor and another for the driven rotor. The separation
        is based on the node numbers of the elements.

        Parameters
        ----------
        elements : list
            List of elements to be separated.
        node_scale : int, optional
            Scaling factor for node numbers. Default is 1.

        Returns
        -------
        driving_elements : list
            List of elements belonging to the driving rotor.
        driven_elements : list
            List of elements belonging to the driven rotor.
        """
        driven_offset = (
            self.driven_nodes[0] - self.rotors["driven"].nodes[0]
        ) * node_scale

        driving_nodes = set(
            range(
                min(self.rotors["driving"].nodes) * node_scale,
                (max(self.rotors["driving"].nodes) + 1) * node_scale,
            )
        ) | {int(node) * node_scale for node in self.rotors["driving"].link_nodes}

        driven_nodes = set(
            range(
                min(self.driven_nodes) * node_scale,
                (max(self.driven_nodes) + 1) * node_scale,
            )
        ) | {
            int(node) * node_scale + driven_offset
            for node in self.rotors["driven"].link_nodes
        }

        driving_elements = []
        driven_elements = []

        for element in elements:
            elm = copy(element)

            if elm.n is None:
                raise ValueError(
                    "Element node `n` must be set when adding to a MultiRotor."
                )

            if elm.n in driving_nodes:
                driving_elements.append(elm)
            elif elm.n in driven_nodes:
                elm.n -= driven_offset
                if getattr(elm, "n_link", None) is not None:
                    elm.n_link -= driven_offset
                driven_elements.append(elm)
            else:
                raise ValueError(
                    f"Node {elm.n} does not belong to the driving or driven rotor."
                )

        return driving_elements, driven_elements

    def _parameters(self, skip=[]):
        """Return keyword arguments to reconstruct this multi-rotor.

        Collects every ``__init__`` parameter except the driving and driven
        rotors, so callers can pass ``**self._parameters()`` when
        building a new multi-rotor from updated component rotors.

        Parameters
        ----------
        skip : list, optional
            List of parameter names to skip. Default is empty.

        Returns
        -------
        dict
            Mapping of parameter names to the current configuration values.
        """
        skip = set(skip).union({"driving_rotor", "driven_rotor"})

        return super()._parameters(skip=skip)

    @classmethod
    def concatenate(cls, *rotors, tag=None):
        """Concatenate rotors to the shafts of a multi-rotor.

        Rotors placed before the multi-rotor are concatenated to the start of
        the driving rotor, and rotors placed after it are concatenated to the
        end of the driven rotor.

        Parameters
        ----------
        *rotors : Rotor
            Sequence of rotors with exactly one MultiRotor, in the order they
            should be combined. A single list or tuple of rotors is also
            accepted as one argument.
        tag : str, optional
            Tag for the new multi-rotor. Default is None.

        Returns
        -------
        MultiRotor object.

        Examples
        --------
        >>> import ross as rs
        >>> from ross.multi_rotor.multi_rotor import two_shaft_rotor_example
        >>> multi_rotor = two_shaft_rotor_example()
        >>> rotor = rs.rotor_example()
        >>> new_multi_rotor = multi_rotor + rotor
        >>> len(new_multi_rotor.rotors["driven"].shaft_elements) == (
        ...     len(multi_rotor.rotors["driven"].shaft_elements)
        ...     + len(rotor.shaft_elements)
        ... )
        True
        """
        if len(rotors) == 1 and isinstance(rotors[0], (list, tuple)):
            rotors = rotors[0]

        rotor_list = list(rotors)
        multi_indices = [
            i for i, rotor in enumerate(rotor_list) if isinstance(rotor, MultiRotor)
        ]
        if len(multi_indices) != 1:
            raise ValueError("Exactly one MultiRotor must be provided.")

        k = multi_indices[0]
        multi_rotor = rotor_list[k]
        driving = multi_rotor.rotors["driving"]
        driven = multi_rotor.rotors["driven"]
        gear_tags = multi_rotor._locate_coupled_gears(driving, driven)
        gear_tags = [gear.tag for gear in gear_tags]
        coupled_nodes = list(multi_rotor.coupled_nodes)

        before = rotor_list[:k]
        after = rotor_list[k + 1 :]

        if before:
            driving = Rotor.concatenate(*before, driving)
            gear_tags[0] = f"{gear_tags[0]} (R{len(before)})"
        if after:
            driven = Rotor.concatenate(driven, *after)
            gear_tags[1] = f"{gear_tags[1]} (R0)"

        for i, rotor in enumerate([driving, driven]):
            coupled_nodes[i] = next(
                elm.n for elm in rotor.disk_elements if elm.tag == gear_tags[i]
            )

        parameters = multi_rotor._parameters()
        parameters["coupled_nodes"] = tuple(coupled_nodes)
        parameters["tag"] = tag or "Concatenated MultiRotor"

        return cls(driving, driven, **parameters)

    def _new_rotor(
        self,
        shaft_elements,
        disk_elements,
        bearing_elements,
        point_mass_elements,
        node_scale=1,
        **kwargs,
    ):
        """Rebuild a multi-rotor while preserving its two-rotor topology."""
        driving_shaft, driven_shaft = self.split_elements(
            shaft_elements, node_scale=node_scale
        )
        driving_disk, driven_disk = self.split_elements(
            disk_elements, node_scale=node_scale
        )
        driving_bearing, driven_bearing = self.split_elements(
            bearing_elements, node_scale=node_scale
        )
        driving_point_mass, driven_point_mass = self.split_elements(
            point_mass_elements, node_scale=node_scale
        )

        driving_rotor = self.rotors["driving"]._new_rotor(
            driving_shaft,
            driving_disk,
            driving_bearing,
            driving_point_mass,
        )
        driven_rotor = self.rotors["driven"]._new_rotor(
            driven_shaft,
            driven_disk,
            driven_bearing,
            driven_point_mass,
        )

        return self._rebuild(
            driving_rotor=driving_rotor,
            driven_rotor=driven_rotor,
            node_scale=node_scale,
            **kwargs,
        )

    def _to_dict(self, file_format=".toml"):
        """Return the complete multi-rotor topology in the format written by ``save()``.

        The two component rotors are nested under the ``rotors`` key, so the
        whole multi-rotor fits in a single file.  The data also contains an
        explicit type marker and all mesh constructor options, so loading
        through either ``MultiRotor.load`` or ``Rotor.load`` restores a real
        ``MultiRotor`` instead of an assembled plain ``Rotor``.

        Parameters
        ----------
        file_format : str, optional
            File format used to serialize the elements (".toml" or ".json").
            Default is ".toml".

        Returns
        -------
        data : dict
            Dictionary with the ROSS version, the rotor type, the multi-rotor
            parameters and the data of both component rotors.
        """
        import ross

        return {
            "ross_version": ross.__version__,
            "rotor_type": self.__class__.__name__,
            "multi_rotor": self._parameters(),
            "rotors": {
                name: rotor._to_dict(file_format) for name, rotor in self.rotors.items()
            },
        }

    @classmethod
    def _from_dict(cls, data):
        """Restore a multi-rotor from data in the format written by ``save()``.

        Parameters
        ----------
        data : dict
            Dictionary obtained from ``_to_dict()``, toml.load() or json.load().

        Returns
        -------
        multi_rotor : ross.MultiRotor
        """
        from ross.rotor_assembly import Rotor

        config = dict(data["multi_rotor"])
        config["coupled_nodes"] = tuple(config["coupled_nodes"])

        return cls(
            driving_rotor=Rotor._from_dict(data["rotors"]["driving"]),
            driven_rotor=Rotor._from_dict(data["rotors"]["driven"]),
            **config,
        )

    def __eq__(self, other):
        """Compare both assembled elements and multi-rotor configuration."""
        if not isinstance(other, MultiRotor):
            return False

        return (
            self.elements == other.elements
            and self._parameters({"tag"}) == other._parameters({"tag"})
            and self.rotors["driving"] == other.rotors["driving"]
            and self.rotors["driven"] == other.rotors["driven"]
        )

    def set_tag(self, tag):
        """Set the tag for the current multi-rotor."""
        self.tag = tag or "MultiRotor 0"

    def add_nodes(self, new_nodes_pos):
        """Add new nodes to the multi-rotor.

        This method allows adding new nodes to the multi-rotor system. It takes
        a list of new node positions and adds them to the existing nodes of the
        multi-rotor.

        Parameters
        ----------
        new_nodes_pos : list of float
            List of new node positions to be added to the multi-rotor.

        Returns
        -------
        multi_rotor : MultiRotor
            The multi-rotor object with the new nodes added.
        """
        driving_rotor = self.rotors["driving"].add_nodes(new_nodes_pos)

        new_nodes_pos_2 = [pos - self.dz_pos for pos in new_nodes_pos]
        driven_rotor = self.rotors["driven"].add_nodes(new_nodes_pos_2)

        return self._rebuild(driving_rotor, driven_rotor)

    def refine_mesh(self, max_ld=0.5, min_ld=0.1, parts=None):
        """Subdivide the shaft elements of both rotors by their L/D ratio.

        Each rotor is refined on its own, by :meth:`Rotor.refine_mesh`, and the
        multi-rotor is built again from the two: a node added to one rotor is
        not added to the other, unlike :meth:`add_nodes`, which cuts both at
        the same positions.

        Parameters
        ----------
        max_ld : float, optional
            Largest length-to-diameter ratio left in the model. Default is 0.5.
        min_ld : float, optional
            Smallest length-to-diameter ratio that is not warned about.
            Default is 0.1.
        parts : int, optional
            When given, every shaft element of both rotors is cut into this
            many equal parts instead, and ``max_ld`` is not used.

        Returns
        -------
        multi_rotor : MultiRotor
            A new multi-rotor; the original is not modified.

        Examples
        --------
        >>> multi_rotor = two_shaft_rotor_example()
        >>> refined = multi_rotor.refine_mesh(max_ld=0.5)
        >>> ratios = [elm.L / max(elm.odl, elm.odr)
        ...     for rotor in refined.rotors.values() for elm in rotor.shaft_elements]
        >>> round(max(ratios), 9) <= 0.5
        True
        """
        driving_rotor = self.rotors["driving"].refine_mesh(max_ld, min_ld, parts)
        driven_rotor = self.rotors["driven"].refine_mesh(max_ld, min_ld, parts)
        return self._rebuild(driving_rotor, driven_rotor)

    def add_elements(self, new_elements):
        """Add elements to the multi-rotor.

        Elements are routed to the driving or driven rotor according to
        their node number in the multi-rotor numbering.

        Parameters
        ----------
        new_elements : list
            List with the new elements.

        Returns
        -------
        multi_rotor : MultiRotor
            The multi-rotor object with the new elements added.

        Examples
        --------
        >>> import ross as rs
        >>> multi_rotor = two_shaft_rotor_example()
        >>> n_elm = len(multi_rotor.elements)
        >>> disk = rs.DiskElement(n=2, m=10, Id=0.1, Ip=0.2)
        >>> new_rotor = multi_rotor.add_elements([disk])
        >>> len(new_rotor.elements) == n_elm + 1
        True
        """
        for el in new_elements:
            if not isinstance(el, Element):
                raise ValueError(f"{el} is not a valid element.")

        driving_elements, driven_elements = self.split_elements(new_elements)

        driving_rotor = self.rotors["driving"].add_elements(driving_elements)
        driven_rotor = self.rotors["driven"].add_elements(driven_elements)

        return self._rebuild(driving_rotor, driven_rotor)

    def _rebuild(self, driving_rotor, driven_rotor, node_scale=None, **kwargs):
        """Rebuild the multi-rotor from updated driving and driven rotors."""
        parameters = self._parameters()

        if node_scale is not None:
            parameters["coupled_nodes"] = tuple(
                int(node * node_scale) for node in self.coupled_nodes
            )
        else:
            gear_1, gear_2 = self._locate_coupled_gears(driving_rotor, driven_rotor)
            parameters["coupled_nodes"] = (gear_1.n, gear_2.n)

        parameters.update(kwargs)

        return self.__class__(driving_rotor, driven_rotor, **parameters)

    def _locate_coupled_gears(self, driving_rotor, driven_rotor):
        """Locate the gear elements in rotors that are coupled."""
        gears = []

        for i, rotor in enumerate([driving_rotor, driven_rotor]):
            gears.append(
                next(
                    elm
                    for elm in rotor.disk_elements
                    if elm.tag == self.coupled_tags[i]
                )
            )

        return gears

    @staticmethod
    def _locate_gear_by_node(rotor, node):
        """Locate the gear elements in the rotor that are coupled to the given node."""
        return next(
            (
                elm
                for elm in rotor.disk_elements
                if elm.n == node and isinstance(elm, GearElement)
            ),
            None,
        )

    @staticmethod
    def _locate_gear_on_plot(rotor, gear):
        """Locate the gear trace on the plot that is coupled to the given gear element."""
        return next(
            (
                elm
                for elm in rotor.plot_rotor().data
                if elm["legendgroup"] == GearElement._legend_group
                and int(search(r"Gear Node: (\d+)", elm.text).group(1)) == gear.n
            ),
            None,
        )

    def _fix_nodes_pos(self, index, node, nodes_pos_l):
        """Adjust node positions of the driven rotor."""
        if node < self.driven_nodes[0]:
            nodes_pos_l[index] = self.rotors["driving"].nodes_pos[
                self.rotors["driving"].nodes.index(node)
            ]
        elif node == self.driven_nodes[0]:
            nodes_pos_l[index] = self.rotors["driven"].nodes_pos[0] + self.dz_pos

    def _set_nodes(self, df_shaft):
        """Set nodes and nodes_pos lists."""
        self.nodes = [*self.rotors["driving"].nodes, *self.driven_nodes]

        R2_nodes_pos = [pos + self.dz_pos for pos in self.rotors["driven"].nodes_pos]
        self.nodes_pos = [*self.rotors["driving"].nodes_pos, *R2_nodes_pos]

        R2_center_line = [
            pos + self.dy_pos for pos in self.rotors["driven"].center_line_pos
        ]

        self.center_line_pos = [
            *self.rotors["driving"].center_line_pos,
            *R2_center_line,
        ]

    def _join_matrices(self, driving_matrix, driven_matrix):
        """Join matrices from the driving rotor and driven rotor to form the matrix of
        the coupled system.

        Parameters
        ----------
        driving_matrix : np.ndarray
            The matrix from the driving rotor.
        driven_matrix : np.ndarray
            The matrix from the driven rotor.

        Returns
        -------
        global_matrix : np.ndarray
            The combined matrix of the coupled system.
        """

        global_matrix = np.zeros((self.ndof, self.ndof))

        first_ndof = self.rotors["driving"].ndof
        global_matrix[:first_ndof, :first_ndof] = driving_matrix
        global_matrix[first_ndof:, first_ndof:] = driven_matrix

        return global_matrix

    def _unbalance_force(self, node, magnitude, phase, omega):
        """Calculate unbalance forces.

        This is an auxiliary function to calculate unbalance forces. It takes the
        force magnitude and phase and generates an array with complex values of forces
        on each degree of freedom of the given node.

        Parameters
        ----------
        node : int
            Node where the unbalance is applied.
        magnitude : float
            Unbalance magnitude (kg.m)
        phase : float
            Unbalance phase (rad)
        omega : list, float
            Array with the desired range of frequencies

        Returns
        -------
        F0 : list
            Unbalance force in each degree of freedom for each value in omega
        """
        ratio = self._node_gear_ratio(node)

        F0 = super()._unbalance_force(
            node, magnitude, phase, np.abs(ratio) * np.asarray(omega)
        )

        if ratio < 0:
            F0 = F0.conj()

        return F0

    def unbalance_force_over_time(
        self, node, magnitude, phase, omega, t, return_all=False
    ):
        """Calculate unbalance forces for each time step.

        This auxiliary function calculates the unbalanced forces by taking
        into account the magnitude and phase of the force. It generates an
        array of force values at each degree of freedom for the specified
        nodes at each time step, while also considering a range of
        frequencies.

        Parameters
        ----------
        node : list, int
            Nodes where the unbalance is applied.
        magnitude : list, float
            Unbalance magnitude (kg.m) for each node.
        phase : list, float
            Unbalance phase (rad) for each node.
        omega : float, np.darray
            Constant velocity or desired range of velocities (rad/s).
        t : np.darray
            Time array (s).
        return_all : bool, optional
            If True, returns F0, theta, omega, and alpha.
            If False, returns only F0.
            Default is False.

        Returns
        -------
        F0 : np.ndarray
            Unbalance force at each degree of freedom for each time step.
        theta : np.ndarray
            Angular positions for each time step.
        omega : np.ndarray
            Angular velocities for each time step.
        alpha : np.ndarray
            Angular accelerations for each time step.
        """

        omega, theta, alpha = make_speed_array(omega, t)

        F0 = np.zeros((self.ndof, len(t)))

        for i, n in enumerate(node):
            phi = phase[i] + self.to_shaft_rotation(theta, n)
            w = self.to_shaft_rotation(omega, n)
            a = self.to_shaft_rotation(alpha, n)

            Fx = magnitude[i] * ((w**2) * np.cos(phi) + a * np.sin(phi))
            Fy = magnitude[i] * ((w**2) * np.sin(phi) - a * np.cos(phi))

            F0[n * self.number_dof + 0, :] += Fx
            F0[n * self.number_dof + 1, :] += Fy

        if return_all:
            return F0, theta, omega, alpha
        else:
            return F0

    def summary(self):
        """Plot the multi-rotor summary.

        The rotor data table has one row per shaft, since the center of
        gravity and the moments of inertia of geared shafts are not those of a
        single rigid rotor. Shafts are numbered from the main driving shaft
        along the gear train. Nodes and the center of gravity are given in the
        multi-rotor numbering and axial positions.

        Returns
        -------
        results : ross.SummaryResults class
            An instance of SummaryResults class to build the summary table.

        Examples
        --------
        >>> multi_rotor = two_shaft_rotor_example()
        >>> summary = multi_rotor.summary()
        >>> [round(cg, 3) for cg in summary.CG]
        [2.899, 8.508]
        >>> table = summary.plot()
        """
        df_disks = pd.merge(
            self.df_disks, self.df[["tag", "nodes_pos_l"]], on="tag", how="left"
        )
        df_bearings = pd.merge(
            self.df_bearings, self.df[["tag", "nodes_pos_l"]], on="tag", how="left"
        )
        self.run_static()

        rotors = self._single_rotors()

        return SummaryResults(
            self.df_shaft,
            df_disks,
            df_bearings,
            self.bearing_forces_tag,
            CG=[rotor.CG + pos_offset for rotor, _, pos_offset in rotors],
            Ip=[rotor.Ip for rotor, _, _ in rotors],
            It=[rotor.It for rotor, _, _ in rotors],
            tag=[f"{rotor.tag} (shaft {i})" for i, (rotor, _, _) in enumerate(rotors)],
            nodes=[
                (rotor.nodes[0] + node_offset, rotor.nodes[-1] + node_offset)
                for rotor, node_offset, _ in rotors
            ],
        )

    def _single_rotors(self):
        """Return the single rotors of this multi-rotor with their offsets.

        Nested MultiRotor objects are expanded recursively.

        Returns
        -------
        rotors : list
            List of (rotor, node_offset, pos_offset) tuples, where the offsets
            map the rotor node numbers and axial positions (m) to those of
            this multi-rotor.
        """
        rotors = []
        driven = self.rotors["driven"]
        node_offsets = {
            "driving": 0,
            "driven": self.driven_nodes[0] - driven.nodes[0],
        }

        for key, rotor in self.rotors.items():
            node_offset = node_offsets[key]
            first_node = self.nodes.index(rotor.nodes[0] + node_offset)
            pos_offset = self.nodes_pos[first_node] - rotor.nodes_pos[0]

            if isinstance(rotor, MultiRotor):
                for sub_rotor, sub_node, sub_pos in rotor._single_rotors():
                    rotors.append(
                        (sub_rotor, sub_node + node_offset, sub_pos + pos_offset)
                    )
            else:
                rotors.append((rotor, node_offset, pos_offset))

        return rotors

    def to_shaft_rotation(self, value, node):
        """Convert a rotation of the driving rotor to the shaft of the node.

        The value is multiplied by the signed speed ratio of the node, so a
        driven shaft rotates by ``-gear_ratio * value``. Nested MultiRotor
        objects accumulate the ratio along the gear train. It applies to
        angular positions, speeds and accelerations.

        Parameters
        ----------
        value : float or np.ndarray
            Angular position (rad), speed (rad/s) or acceleration (rad/s²) of
            the driving rotor.
        node : int, optional
            Node index in this MultiRotor numbering.

        Returns
        -------
        float or np.ndarray
            The same quantity for the shaft of the node.

        Examples
        --------
        >>> multi_rotor = two_shaft_rotor_example()
        >>> multi_rotor.to_shaft_rotation(100.0, 7) # doctest: +ELLIPSIS
        -1426.08...
        """
        return self._node_gear_ratio(node) * value

    def _node_gear_ratio(self, node):
        """Return the signed speed ratio of the specified node.

        The ratio relates the rotational speed at the node to the speed of the
        main driving shaft, i.e. the speed given to the analysis methods. Each
        gear mesh contributes a factor of -gear_ratio, since meshed gears spin
        in opposite directions. Nested MultiRotor objects are handled
        recursively, so the ratio accumulates along the whole gear train:

        - nodes of the driving rotor take the ratio from the driving rotor,
          which is 1.0 when it is a single Rotor;
        - nodes of the driven rotor take the ratio of the driving gear node,
          multiplied by -gear_ratio of this mesh and by the ratio of the node
          inside the driven rotor.

        Parameters
        ----------
        node : int
            The node index in this MultiRotor numbering.

        Returns
        -------
        ratio : float
            Signed speed ratio at the node. A negative value means the node
            spins opposite to the main driving shaft.
        """
        ratio = 1.0
        driving_rotor = self.rotors["driving"]

        if node in self.driven_nodes:
            if isinstance(driving_rotor, MultiRotor):
                ratio = driving_rotor._node_gear_ratio(self.coupled_nodes[0])

            ratio *= -self.mesh.gear_ratio

            driven_rotor = self.rotors["driven"]
            node -= self.driven_nodes[0] - driven_rotor.nodes[0]

            if isinstance(driven_rotor, MultiRotor):
                ratio *= driven_rotor._node_gear_ratio(node)

        elif isinstance(driving_rotor, MultiRotor):
            ratio = driving_rotor._node_gear_ratio(node)

        return ratio

    @property
    def _ucs_mode_step(self):
        """Step between the modes drawn as curves in the UCS map.

        The gear mesh couples the lateral planes, so the modes do not come in
        pairs with the same natural frequency and every mode is drawn.
        """
        return 1

    def compute_coupling_matrix(self):
        """Coupling matrix of two coupled gears.

        Returns
        -------
        coupling_matrix : np.ndarray
            Dimensionless coupling matrix of two coupled gears

        References
        ----------
        STRINGER, D. B. Geared Rotor Dynamic Methodologies for Advancing Prognostic
        Modeling Capabilities in Rotary-Wing Transmission Systems.
        Thesis (Dissertation) — University of Virginia, Charlottesville, VA, 2008.

        Examples
        --------
        >>> multi_rotor = two_shaft_rotor_example()
        >>> np.round(multi_rotor.compute_coupling_matrix(),8)[:4, :4]
        array([[ 0.14644661,  0.35355339, -0.        , -0.        ],
               [ 0.35355339,  0.85355339, -0.        , -0.        ],
               [-0.        , -0.        ,  0.        ,  0.        ],
               [-0.        , -0.        ,  0.        ,  0.        ]])
        """

        pressure_angle = self.mesh.pressure_angle  # normal angle

        helix_angle = self.mesh.helix_angle - np.pi  # adjusted according to formulation

        cx = np.cos(pressure_angle) * np.cos(helix_angle)
        cy = np.sin(pressure_angle)
        cz = np.sin(pressure_angle) * np.sin(helix_angle)

        cp = np.cos(self.mesh.orientation_angle)
        sp = np.sin(self.mesh.orientation_angle)

        r1 = self.mesh.driving_gear.pitch_diameter / 2
        r2 = self.mesh.driven_gear.pitch_diameter / 2

        Kii = np.zeros((self.number_dof, self.number_dof))
        Kji = np.zeros((self.number_dof, self.number_dof))
        Kjj = np.zeros((self.number_dof, self.number_dof))

        # Kii
        Kii[0, 0] = (sp * cx + cp * cy) ** 2
        Kii[1, 0] = (sp * cx + cp * cy) * (sp * cy - cp * cx)
        Kii[2, 0] = cz * (sp * cx + cp * cy)
        Kii[3, 0] = sp * cz * r1 * (sp * cx + cp * cy)
        Kii[4, 0] = -cp * cz * r1 * (sp * cx + cp * cy)
        Kii[5, 0] = -cx * r1 * (cp**2 + sp**2) * (sp * cx + cp * cy)

        Kii[0, 1] = Kii[1, 0]
        Kii[1, 1] = (sp * cy - cp * cx) ** 2
        Kii[2, 1] = cz * (sp * cy - cp * cx)
        Kii[3, 1] = sp * cz * r1 * (sp * cy - cp * cx)
        Kii[4, 1] = -cp * cz * r1 * (sp * cy - cp * cx)
        Kii[5, 1] = -cx * r1 * (cp**2 + sp**2) * (sp * cy - cp * cx)

        Kii[0, 2] = Kii[2, 0]
        Kii[1, 2] = Kii[2, 1]
        Kii[2, 2] = cz**2
        Kii[3, 2] = sp * (cz**2) * r1
        Kii[4, 2] = -cp * (cz**2) * r1
        Kii[5, 2] = -cx * cz * r1 * (cp**2 + sp**2)

        Kii[0, 3] = Kii[3, 0]
        Kii[1, 3] = Kii[3, 1]
        Kii[2, 3] = Kii[3, 2]
        Kii[3, 3] = (sp * cz * r1) ** 2
        Kii[4, 3] = -cp * sp * (cz * r1) ** 2
        Kii[5, 3] = -sp * cx * cz * (r1**2) * (cp**2 + sp**2)

        Kii[0, 4] = Kii[4, 0]
        Kii[1, 4] = Kii[4, 1]
        Kii[2, 4] = Kii[4, 2]
        Kii[3, 4] = Kii[4, 3]
        Kii[4, 4] = (cp * cz * r1) ** 2
        Kii[5, 4] = cp * cx * cz * (r1**2) * (cp**2 + sp**2)

        Kii[0, 5] = Kii[5, 0]
        Kii[1, 5] = Kii[5, 1]
        Kii[2, 5] = Kii[5, 2]
        Kii[3, 5] = Kii[5, 3]
        Kii[4, 5] = Kii[5, 4]
        Kii[5, 5] = ((cx * r1) ** 2) * (cp**4 + 2 * (cp * sp) ** 2 + sp**4)

        # Kji
        Kji[0, 0] = -((sp * cx + cp * cy) ** 2)
        Kji[1, 0] = -(sp * cx + cp * cy) * (sp * cy - cp * cx)
        Kji[2, 0] = -cz * (sp * cx + cp * cy)
        Kji[3, 0] = -sp * cz * r2 * (sp * cx + cp * cy)
        Kji[4, 0] = cp * cz * r2 * (sp * cx + cp * cy)
        Kji[5, 0] = cx * r2 * (cp**2 + sp**2) * (sp * cx + cp * cy)

        Kji[0, 1] = Kji[1, 0]
        Kji[1, 1] = -((sp * cy - cp * cx) ** 2)
        Kji[2, 1] = -cz * (sp * cy - cp * cx)
        Kji[3, 1] = -sp * cz * r2 * (sp * cy - cp * cx)
        Kji[4, 1] = cp * cz * r2 * (sp * cy - cp * cx)
        Kji[5, 1] = cx * r2 * (cp**2 + sp**2) * (sp * cy - cp * cx)

        Kji[0, 2] = Kji[2, 0]
        Kji[1, 2] = Kji[2, 1]
        Kji[2, 2] = -(cz**2)
        Kji[3, 2] = -sp * (cz**2) * r2
        Kji[4, 2] = cp * (cz**2) * r2
        Kji[5, 2] = cx * cz * r2 * (cp**2 + sp**2)

        Kji[0, 3] = -sp * cz * r1 * (sp * cx + cp * cy)
        Kji[1, 3] = -sp * cz * r1 * (sp * cy - cp * cx)
        Kji[2, 3] = -sp * (cz**2) * r1
        Kji[3, 3] = -((sp * cz) ** 2) * r1 * r2
        Kji[4, 3] = cp * sp * (cz**2) * r1 * r2
        Kji[5, 3] = sp * cx * cz * r1 * r2 * (cp**2 + sp**2)

        Kji[0, 4] = cp * cz * r1 * (sp * cx + cp * cy)
        Kji[1, 4] = cp * cz * r1 * (sp * cy - cp * cx)
        Kji[2, 4] = cp * (cz**2) * r1
        Kji[3, 4] = cp * sp * (cz**2) * r1 * r2
        Kji[4, 4] = -((cp * cz) ** 2) * r1 * r2
        Kji[5, 4] = -cp * cx * cz * r1 * r2 * (cp**2 + sp**2)

        Kji[0, 5] = cx * r1 * (cp**2 + sp**2) * (sp * cx + cp * cy)
        Kji[1, 5] = cx * r1 * (cp**2 + sp**2) * (sp * cy - cp * cx)
        Kji[2, 5] = cx * cz * r1 * (cp**2 + sp**2)
        Kji[3, 5] = sp * cx * cz * r1 * r2 * (cp**2 + sp**2)
        Kji[4, 5] = -cp * cx * cz * r1 * r2 * (cp**2 + sp**2)
        Kji[5, 5] = -(cx**2) * r1 * r2 * (cp**4 + 2 * (cp * sp) ** 2 + sp**4)

        # Kjj
        Kjj[0, 0] = (sp * cx + cp * cy) ** 2
        Kjj[1, 0] = (sp * cx + cp * cy) * (sp * cy - cp * cx)
        Kjj[2, 0] = cz * (sp * cx + cp * cy)
        Kjj[3, 0] = sp * cz * r2 * (sp * cx + cp * cy)
        Kjj[4, 0] = -cp * cz * r2 * (sp * cx + cp * cy)
        Kjj[5, 0] = -cx * r2 * (cp**2 + sp**2) * (sp * cx + cp * cy)

        Kjj[0, 1] = Kjj[1, 0]
        Kjj[1, 1] = (sp * cy - cp * cx) ** 2
        Kjj[2, 1] = cz * (sp * cy - cp * cx)
        Kjj[3, 1] = sp * cz * r2 * (sp * cy - cp * cx)
        Kjj[4, 1] = -cp * cz * r2 * (sp * cy - cp * cx)
        Kjj[5, 1] = -cx * r2 * (cp**2 + sp**2) * (sp * cy - cp * cx)

        Kjj[0, 2] = Kjj[2, 0]
        Kjj[1, 2] = Kjj[2, 1]
        Kjj[2, 2] = cz**2
        Kjj[3, 2] = sp * (cz**2) * r2
        Kjj[4, 2] = -cp * (cz**2) * r2
        Kjj[5, 2] = -cx * cz * r2 * (cp**2 + sp**2)

        Kjj[0, 3] = Kjj[3, 0]
        Kjj[1, 3] = Kjj[3, 1]
        Kjj[2, 3] = Kjj[3, 2]
        Kjj[3, 3] = (sp * cz * r2) ** 2
        Kjj[4, 3] = -cp * sp * (cz * r2) ** 2
        Kjj[5, 3] = -sp * cx * cz * (r2**2) * (cp**2 + sp**2)

        Kjj[0, 4] = Kjj[4, 0]
        Kjj[1, 4] = Kjj[4, 1]
        Kjj[2, 4] = Kjj[4, 2]
        Kjj[3, 4] = Kjj[4, 3]
        Kjj[4, 4] = (cp * cz * r2) ** 2
        Kjj[5, 4] = cp * cx * cz * (r2**2) * (cp**2 + sp**2)

        Kjj[0, 5] = Kjj[5, 0]
        Kjj[1, 5] = Kjj[5, 1]
        Kjj[2, 5] = Kjj[5, 2]
        Kjj[3, 5] = Kjj[5, 3]
        Kjj[4, 5] = Kjj[5, 4]
        Kjj[5, 5] = ((cx * r2) ** 2) * (cp**4 + 2 * (cp * sp) ** 2 + sp**4)

        coupling_matrix = np.block([[Kii, Kji.T], [Kji, Kjj]])

        return coupling_matrix

    def K_mesh(self, K0):
        """Add the gear mesh stiffness contribution to a stiffness matrix.

        Parameters
        ----------
        K0 : np.ndarray
            Stiffness matrix to which the gear mesh stiffness will be added.

        Returns
        -------
        K0 : np.ndarray
            Stiffness matrix with the gear mesh stiffness contribution added.
        """
        dofs_1 = self.mesh.driving_gear.dof_global_index.values()
        dofs_2 = self.mesh.driven_gear.dof_global_index.values()
        dofs = [*dofs_1, *dofs_2]

        K0[np.ix_(dofs, dofs)] += self.coupling_matrix * self._mesh_stiffness

        return K0

    def C_mesh(self, C0):
        """Add the gear mesh damping contribution to a damping matrix.

        The mesh damping coefficient :math:`c_m = 2 \\zeta \\sqrt{k_m M_{eq}}`
        (Yi et al., 2019) multiplies the same coupling matrix as the mesh
        stiffness.

        Parameters
        ----------
        C0 : np.ndarray
            Damping matrix to which the mesh damping will be added.

        Returns
        -------
        C0 : np.ndarray
            Damping matrix with the gear mesh damping contribution added.

        """
        dofs_1 = self.mesh.driving_gear.dof_global_index.values()
        dofs_2 = self.mesh.driven_gear.dof_global_index.values()
        dofs = [*dofs_1, *dofs_2]

        c_m = (
            2.0
            * self.mesh_damping_ratio
            * np.sqrt(self._mesh_stiffness * self.mesh.M_eq)
        )
        C0[np.ix_(dofs, dofs)] += self.coupling_matrix * c_m

        return C0

    def K(self, frequency, speed=None):
        """Stiffness matrix for a multi-rotor.

        The excitation frequency is global to the coupled system, while the
        driven rotor speed is scaled by the gear ratio.

        Parameters
        ----------
        frequency : float
            Excitation (whirl) frequency.
        speed : float, optional
            Driving rotor speed. Default is the excitation frequency
            (synchronous evaluation).

        Returns
        -------
        K0 : np.ndarray
            Stiffness matrix for the multi-rotor.

        Examples
        --------
        >>> multi_rotor = two_shaft_rotor_example()
        >>> multi_rotor.K(0)[:4, :4] / 1e10
        array([[ 4.7609372 ,  0.        ,  0.        ,  0.        ],
               [ 0.        ,  4.7625872 ,  0.        , -0.23712736],
               [ 0.        ,  0.        , 14.63196778,  0.        ],
               [ 0.        , -0.23712736,  0.        ,  0.09416119]])
        """

        if speed is None:
            speed = frequency

        return self.add_coupling_stiffness(
            self._join_matrices(
                self.rotors["driving"].K(frequency, speed),
                self.rotors["driven"].K(frequency, speed * abs(self._gear_ratio)),
            )
        )

    def Ksdt(self):
        """Dynamic stiffness matrix for a multi-rotor.

        Stiffness matrix associated with the transient motion of the
        shaft and disks. For time-dependent analyses, this matrix needs to be
        multiplied by the angular acceleration. Therefore, the stiffness matrix
        of the driven rotor is scaled by the gear ratio before being combined
        with the driving rotor matrix.

        Returns
        -------
        Ksdt0 : np.ndarray
            Dynamic stiffness matrix for the multi-rotor.

        Examples
        --------
        >>> multi_rotor = two_shaft_rotor_example()
        >>> multi_rotor.Ksdt()[:6, :4]
        array([[  0.        , -74.43218395,   0.        ,   0.6202682 ],
               [  0.        ,   0.        ,   0.        ,   0.        ],
               [  0.        ,   0.        ,   0.        ,   0.        ],
               [  0.        ,   0.        ,   0.        ,   0.        ],
               [  0.        ,  -0.6202682 ,   0.        ,   0.08270243],
               [  0.        ,   0.        ,   0.        ,   0.        ]])
        """

        return self._join_matrices(
            self.rotors["driving"].Ksdt(),
            self._gear_ratio * self.rotors["driven"].Ksdt(),
        )

    def M(self, frequency=None, speed=None, synchronous=False):
        """Mass matrix for a multi-rotor.

        The excitation frequency is global to the coupled system, while the
        driven rotor speed is scaled by the gear ratio.

        Parameters
        ----------
        frequency : float, optional
            Excitation (whirl) frequency. Default is 0.
        speed : float, optional
            Driving rotor speed. Default is the excitation frequency
            (synchronous evaluation).
        synchronous : bool, optional
            If True the gyroscopic matrix is folded into the mass matrix
            (Rouch's formulation for a synchronous analysis).
            Default is False.

        Returns
        -------
        M0 : np.ndarray
            Mass matrix for the multi-rotor.

        Examples
        --------
        >>> multi_rotor = two_shaft_rotor_example()
        >>> multi_rotor.M(0)[:4, :4]
        array([[18.55298224,  0.        ,  0.        ,  0.        ],
               [ 0.        , 18.55298224,  0.        , -0.16179571],
               [ 0.        ,  0.        , 18.37831702,  0.        ],
               [ 0.        , -0.16179571,  0.        ,  0.10074262]])
        """

        if frequency is None:
            frequency = 0
        if speed is None:
            speed = frequency

        return self._join_matrices(
            self.rotors["driving"].M(frequency, speed, synchronous),
            self.rotors["driven"].M(
                frequency, speed * abs(self._gear_ratio), synchronous
            ),
        )

    def C(self, frequency, speed=None):
        """Damping matrix for a multi-rotor rotor.

        The excitation frequency is global to the coupled system, while the
        driven rotor speed is scaled by the gear ratio.

        Parameters
        ----------
        frequency : float
            Excitation (whirl) frequency.
        speed : float, optional
            Driving rotor speed. Default is the excitation frequency
            (synchronous evaluation).

        Returns
        -------
        C0 : np.ndarray
            Damping matrix for the multi-rotor.

        Examples
        --------
        >>> multi_rotor = two_shaft_rotor_example()
        >>> multi_rotor.C(0)[:4, :4] / 1e3
        array([[ 3.,  0.,  0.,  0.],
               [ 0.,  3.,  0., -0.],
               [ 0.,  0.,  0.,  0.],
               [ 0., -0.,  0.,  0.]])
        """

        if speed is None:
            speed = frequency

        return self.add_coupling_damping(
            self._join_matrices(
                self.rotors["driving"].C(frequency, speed),
                self.rotors["driven"].C(frequency, speed * abs(self._gear_ratio)),
            )
        )

    def G(self):
        """Gyroscopic matrix for a multi-rotor.

        For time-dependent analyses, this matrix needs to be multiplied by the
        rotor speed. Therefore, the gyroscopic matrix of the driven rotor is
        scaled by the gear ratio before being combined with the driving rotor matrix.

        Returns
        -------
        G0 : np.ndarray
            Gyroscopic matrix for the multi-rotor.

        Examples
        --------
        >>> multi_rotor = two_shaft_rotor_example()
        >>> multi_rotor.G()[:4, :4]
        array([[ 0.        ,  0.17162125,  0.        ,  0.1403395 ],
               [-0.17162125,  0.        ,  0.        ,  0.        ],
               [ 0.        ,  0.        ,  0.        ,  0.        ],
               [-0.1403395 ,  0.        ,  0.        ,  0.        ]])
        """

        return self._join_matrices(
            self.rotors["driving"].G(),
            self._gear_ratio * self.rotors["driven"].G(),
        )

    def _rotor_system_for_integrate(
        self, rotor, speed, t, reduce_model, forces, **kwargs
    ):
        """Build rotor system for integrate method."""

        updated_forces = forces

        if self.mesh.backlash:
            speed, theta, _ = make_speed_array(speed, t)

            updated_forces = lambda step, **curr_state: (
                forces(step, **curr_state)
                + reduce_model[1](
                    self.mesh.backlash.compute_force(
                        step=step,
                        disp_resp=reduce_model[2](curr_state.get("disp_resp")),
                        velc_resp=reduce_model[2](curr_state.get("velc_resp")),
                        time=t[step],
                        angular_pos=theta[step],
                        speed=speed[step],
                    )
                )
            )

        elif self.update_mesh_stiffness:
            speed, theta, accel = make_speed_array(speed, t)
            reduce_matrix = reduce_model[0]

            # Assemble matrices
            M = reduce_matrix(kwargs.get("M", self.M()))
            C2 = reduce_matrix(kwargs.get("G", self.G()))
            K2 = reduce_matrix(kwargs.get("Ksdt", self.Ksdt()))

            def rotor_system(step, **current_state):
                # Update mesh stiffness
                self._mesh_stiffness = self.mesh.interpolate_stiffness(theta[step])

                C1 = reduce_matrix(self.C(speed[step]))
                K1 = reduce_matrix(self.K(speed[step]))

                return (
                    M,
                    C1 + C2 * speed[step],
                    K1 + K2 * accel[step],
                    updated_forces(step, **current_state),
                )

            return rotor_system, updated_forces

        return super()._rotor_system_for_integrate(
            rotor, speed, t, reduce_model, updated_forces, **kwargs
        )

    @check_units
    def run_time_response(self, speed, F, t, method="default", **kwargs):
        """Calculate the time response of the multi-rotor.

        This function will take the multi-rotor object and calculate its time
        response given a force and a time. If the gear mesh has backlash
        enabled, the returned results also include the time evolution of the
        mesh dynamics (transmission error, backlash, mesh force, mesh
        stiffness, center distance, pressure angle and contact ratio).

        Parameters
        ----------
        speed : float or array_like, pint.Quantity
            Rotor speed. Automatically, the Newmark method is chosen if `speed`
            has an array_like type.
        F : array
            Force array (needs to have the same number of rows as time array).
            Each column corresponds to a dof and each row to a time.
        t : array
            Time array.
        method : str, optional
            The Newmark method can be chosen by setting `method='newmark'`.
        **kwargs : optional
            Additional keyword arguments can be passed to define the parameters
            of the Newmark method if it is used (e.g. gamma, beta, tol, ...).
            See `ross.utils.newmark` for more details.
            Other keyword arguments can also be passed to be used in numerical
            integration (e.g. model_reduction, add_to_RHS).
            See `Rotor.integrate_system` for more details.

        Returns
        -------
        results : ross.TimeResponseResults
            Time response results for the multi-rotor.
        """
        if self.mesh.backlash:
            t_, yout, xout = self.time_response(speed, F, t, method=method, **kwargs)
            results = BacklashResults(self, t, yout, xout, speed=speed)
        else:
            results = super().run_time_response(speed, F, t, method=method, **kwargs)

        if self.update_mesh_stiffness:
            self._mesh_stiffness = self.mesh.stiffness

        return results

    @check_units
    def run_unbalance_response(
        self,
        node,
        unbalance_magnitude,
        unbalance_phase,
        speed_range=None,
        modes=None,
    ):
        """Unbalanced response for a multi-rotor.
        Each unbalance excites at the speed of its own shaft.
        The response is evaluated at that frequency, with the
        rotor at driving rotor speed.
        Parameters
        ----------
        node : list, int
            Node where the unbalance is applied.
        unbalance_magnitude : list, float, pint.Quantity
            Unbalance magnitude (kg.m).
        unbalance_phase : list, float, pint.Quantity
            Unbalance phase (rad).
        speed_range : list, pint.Quantity
            Driving rotor speeds (rad/s).
            Default is 0 to 1.5 x highest damped natural frequency.
        modes : list, optional
            Modes that will be used to calculate the frequency response
            (all modes will be used if a list is not given).
        Returns
        -------
        results : ross.ForcedResponseResults
            For more information on attributes and methods available see:
            :py:class:`ross.ForcedResponseResults`
        Raises
        ------
        ValueError
            If there are unbalances on both shafts and ``abs(speed_ratio)``
            is not 1. The response then has two harmonics, which a single
            result cannot hold, so each shaft must be run separately.
        Examples
        --------
        >>> multi_rotor = two_shaft_rotor_example()
        >>> speed = np.linspace(0, 100, 11)
        >>> response = multi_rotor.run_unbalance_response(node=7,
        ...                                               unbalance_magnitude=1e-3,
        ...                                               unbalance_phase=0,
        ...                                               speed_range=speed)
        >>> abs(response.forced_resp[7 * multi_rotor.number_dof, :3])
        array([0.00000000e+00, 2.01389384e-06, 2.11118132e-06])
        """
        if speed_range is None:
            modal = self.run_modal(0)
            speed_range = np.linspace(0, max(modal.evalues.imag) * 1.5, 1000)

        speed_range = np.asarray(speed_range)

        node = np.atleast_1d(node)
        unbalance_magnitude = np.atleast_1d(unbalance_magnitude)
        unbalance_phase = np.atleast_1d(unbalance_phase)

        frequency_ratios = {abs(self._node_gear_ratio(n)) for n in node}
        if len(frequency_ratios) > 1:
            raise ValueError(
                "Unbalances on both shafts excite at different frequencies. "
                "Run the unbalance response for each shaft separately."
            )

        frequency_range = frequency_ratios.pop() * speed_range

        self._check_coefficient_axes(speed=speed_range, frequency=frequency_range)

        force = np.zeros((self.ndof, len(speed_range)), dtype=complex)
        for n, m, p in zip(node, unbalance_magnitude, unbalance_phase, strict=True):
            force += self._unbalance_force(n, m, p, speed_range)

        forced_resp = np.zeros((self.ndof, len(speed_range)), dtype=complex)
        for i, (speed, frequency) in enumerate(
            zip(speed_range, frequency_range, strict=True)
        ):
            H = self.transfer_matrix(speed=speed, frequency=frequency, modes=modes)
            forced_resp[:, i] = H @ force[:, i]

        return ForcedResponseResults(
            rotor=self,
            forced_resp=forced_resp,
            velc_resp=1j * frequency_range * forced_resp,
            accl_resp=-(frequency_range**2) * forced_resp,
            speed_range=speed_range,
            unbalance=np.vstack((node, unbalance_magnitude, unbalance_phase)),
        )

    def run_critical_speed(self, speed_range=None, num_modes=12, rtol=0.005, shaft=0):
        """Calculate the critical speeds and damping ratios for the multi-rotor.

        A critical speed is a driving rotor speed at which the synchronous
        (1X) excitation of one shaft equals a natural frequency. The chosen
        shaft turns ``abs(speed_ratio)`` times faster than the driving
        rotor (see :py:meth:`to_shaft_rotation`), so its critical speeds are
        the driving rotor speeds ``s`` with ``abs(speed_ratio) * s = wn(s)``.
        The frequency-dependent coefficients are evaluated at that excitation
        frequency.

        Parameters
        ----------
        speed_range : tuple, optional, pint.Quantity
            Tuple (start, end) with the range of driving rotor speeds (rad/s).
            The function returns all critical speeds within this range.
        num_modes : int, optional
            The number of eigenvalues and eigenvectors to be calculated using
            ARPACK. If speed_range is not None, num_modes is overrided.
            Default is 12.
        rtol : float, optional
            Tolerance (relative) for termination. Applied to
            scipy.optimize.newton. Default is 0.005 (0.5%).
        shaft : int, optional
            Index of the shaft whose synchronous excitation is considered,
            counted along the gear train from the driving shaft (0). Nested
            multi-rotors are expanded, so a three-shaft train has shafts 0, 1
            and 2. Default is 0.

        Returns
        -------
        results : ross.CriticalSpeedResults
            Critical speeds of the driving rotor. For more information on
            attributes and methods available see:
            :py:class:`ross.CriticalSpeedResults`

        Examples
        --------
        >>> multi_rotor = two_shaft_rotor_example()

        Critical speeds excited by the driving shaft
        >>> results = multi_rotor.run_critical_speed(num_modes=8)
        >>> np.round(results.wn("rpm"))
        array([ 702.,  734., 1048., 1099.])

        Critical speeds excited by the driven shaft, given as driving shaft
        speeds
        >>> results = multi_rotor.run_critical_speed(num_modes=8, shaft=1)
        >>> np.round(results.wn("rpm"))
        array([49., 51., 75., 76.])
        """
        shafts = self._single_rotors()
        if not 0 <= shaft < len(shafts):
            raise ValueError(
                f"shaft must be between 0 and {len(shafts) - 1}, got {shaft}."
            )

        rotor, node_offset, _ = shafts[shaft]
        node = rotor.nodes[0] + node_offset
        ratio = abs(self.to_shaft_rotation(1, node))

        num_modes = (self.ndof - 4) * 2 if speed_range is not None else num_modes

        modal = self.run_modal(0, num_modes)

        def modal_at(speed):
            frequency = None if ratio == 1 else ratio * speed
            return self.run_modal(speed, num_modes, frequency=frequency)

        def w_crit(speed_type):
            w0 = getattr(modal, speed_type)
            w = np.zeros_like(w0)

            for i in range(len(w)):
                func = lambda s: ratio * s - getattr(modal_at(s), speed_type)[i]
                w[i] = newton(func=func, x0=w0[i] / ratio, rtol=rtol)

            return w

        return self._run_critical_speed(modal_at, w_crit, speed_range)

    @check_units
    def run_harmonic_balance_response(
        self,
        speed,
        t,
        harmonic_forces,
        gravity=False,
        n_harmonics=1,
    ):
        """
        Compute the steady-state response of the rotor using the Harmonic Balance
        method.

        The speed and frequency dependent coefficients are evaluated at the rotor
        speed for every harmonic. In a geared multi-rotor, ``speed`` is the
        driving rotor speed.

        Parameters
        ----------
        speed : float, pint.Quantity
            Rotational speed of the rotor (rad/s).
        t : array_like
            Time vector (s).
        harmonic_forces : list of dict
            List of harmonic force definitions. Each dictionary should contain the
            following keys:

            - 'node': int
                Node index where the force is applied.

            - 'magnitudes': list, float
                List of excitation magnitudes.
                Interpretation depends on the type of excitation:

                - For direct harmonic forces: force amplitudes (N).
                - For unbalance-type excitations: ``m * e * shaft_speed**2`` (N),
                  where ``m`` is the unbalance mass (kg) and ``e`` is the
                  eccentricity (m).

            - 'phases': list of float
                List of phase angles (rad).

            - 'harmonics': list of int
                List of harmonic orders of the speed of the node's shaft
                (1 for fundamental, 2 for second, etc.). The force rotates
                with that shaft. In a geared multi-rotor, harmonic ``h`` of a
                shaft turning ``r`` times as fast as the driving rotor is
                harmonic ``abs(r) * h`` of the rotor speed. When that is not
                an integer, a warning is issued and the force is applied at
                harmonic ``h`` of the rotor speed, whirling forward.
        gravity : bool, optional
            If True, include the effect of gravity in the response.
            Default is False.
        n_harmonics : int, optional
            Number of harmonics to consider in the Harmonic Balance solution.
            Default is 1 (only fundamental harmonic is considered).

        Returns
        -------
        results : ross.results.HarmonicBalanceResponse
            Object containing the steady-state response.

        Examples
        --------
        """
        return super().run_harmonic_balance_response(
            speed=speed,
            t=t,
            harmonic_forces=harmonic_forces,
            gravity=gravity,
            n_harmonics=n_harmonics,
        )


def two_shaft_rotor_example():
    """Create a multi-rotor as example.

    This function returns an instance of two-shaft rotor system from Rao et al.
    This typical example is a turbo-alternator rotor system, which consists of
    a generator rotor, a turbine rotor and a spur gear pair connecting two rotors.
    Each rotor is supported by a pair of bearing two shaft elements, one disk and
    two simple bearings.

    The purpose of this is to make available a simple model so that doctest can
    be written using this.

    Returns
    -------
    An instance of a rotor object.

    References
    ----------
    Rao, J. S., Shiau, T. N., chang, J. R. (1998). Theoretical analysis of lateral
    response due to torsional excitation of geared rotors. Mechanism and Machine
    Theory, 33 (6), 761-783. doi: 10.1016/S0094-114X(97)00056-6

    Examples
    --------
    >>> multi_rotor = two_shaft_rotor_example()
    >>> modal = multi_rotor.run_modal(speed=0)
    >>> np.round(modal.wd[:4])
    array([ 74.,  77., 112., 113.])
    """
    # A spur geared two-shaft rotor system.
    material = rs.Material(name="mat_steel", rho=7800, E=207e9, G_s=79.5e9)

    # Rotor 1
    L1 = [0.1, 4.24, 1.16, 0.3]
    d1 = [0.3, 0.3, 0.22, 0.22]
    shaft1 = [
        rs.ShaftElement(
            L=L1[i],
            idl=0.0,
            odl=d1[i],
            material=material,
            shear_effects=True,
            rotary_inertia=True,
            gyroscopic=True,
        )
        for i in range(len(L1))
    ]

    generator = rs.DiskElement(
        n=1,
        m=525.7,
        Id=16.1,
        Ip=32.2,
    )
    disk = rs.DiskElement(
        n=2,
        m=116.04,
        Id=3.115,
        Ip=6.23,
    )

    pressure_angle = Q_(22.5, "deg")

    gear1 = rs.GearElement(
        n=4,
        m=726.4,
        Id=56.95,
        Ip=113.9,
        n_teeth=328,
        base_diameter=0.5086 * 2,
        pr_angle=pressure_angle,
    )

    bearing1 = rs.BearingElement(n=0, kxx=183.9e6, kyy=200.4e6, cxx=3e3)
    bearing2 = rs.BearingElement(n=3, kxx=183.9e6, kyy=200.4e6, cxx=3e3)

    rotor1 = rs.Rotor(
        shaft1,
        [generator, disk, gear1],
        [bearing1, bearing2],
    )

    # Rotor 2
    L2 = [0.3, 5, 0.1]
    d2 = [0.15, 0.15, 0.15]
    shaft2 = [
        rs.ShaftElement(
            L=L2[i],
            idl=0.0,
            odl=d2[i],
            material=material,
            shear_effects=True,
            rotary_inertia=True,
            gyroscopic=True,
        )
        for i in range(len(L2))
    ]

    gear2 = rs.GearElement(
        n=0,
        m=5,
        Id=0.002,
        Ip=0.004,
        n_teeth=23,
        base_diameter=0.03567 * 2,
        pr_angle=pressure_angle,
    )

    turbine = rs.DiskElement(n=2, m=7.45, Id=0.0745, Ip=0.149)

    bearing3 = rs.BearingElement(n=1, kxx=10.1e6, kyy=41.6e6, cxx=3e3)
    bearing4 = rs.BearingElement(n=3, kxx=10.1e6, kyy=41.6e6, cxx=3e3)

    rotor2 = rs.Rotor(
        shaft2,
        [gear2, turbine],
        [bearing3, bearing4],
    )

    return rs.MultiRotor(
        rotor1,
        rotor2,
        coupled_nodes=(4, 0),
        gear_mesh_stiffness=1e8,
        orientation_angle=0.0,
        position="below",
    )
