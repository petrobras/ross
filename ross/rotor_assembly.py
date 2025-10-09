import warnings
from collections.abc import Iterable
from copy import copy, deepcopy
from itertools import chain, cycle
from pathlib import Path

import numpy as np
import pandas as pd
import toml
from methodtools import lru_cache
from plotly import express as px
from plotly import graph_objects as go
from scipy import io as sio
from scipy import linalg as la
from scipy import signal as signal
from scipy.integrate import cumulative_trapezoid as integrate
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import newton
from scipy.signal import chirp
from scipy.sparse import linalg as las

from ross.bearing_seal_element import (
    BallBearingElement,
    BearingElement,
    BearingFluidFlow,
    CylindricalBearing,
    MagneticBearingElement,
    RollerBearingElement,
    SealElement,
)
from ross.coupling_element import CouplingElement
from ross.disk_element import DiskElement
from ross.faults import Crack, MisalignmentFlex, MisalignmentRigid, Rubbing
from ross.materials import Material, steel
from ross.point_mass import PointMass
from ross.results import (
    CampbellResults,
    ConvergenceResults,
    CriticalSpeedResults,
    ForcedResponseResults,
    FrequencyResponseResults,
    Level1Results,
    ModalResults,
    StaticResults,
    SummaryResults,
    TimeResponseResults,
    UCSResults,
    SensitivityResults,
)
from ross.shaft_element import ShaftElement
from ross.units import Q_, check_units
from ross.utils import (
    assemble_C_K_matrices,
    convert_6dof_to_4dof,
    convert_6dof_to_torsional,
    intersection,
    newmark,
    remove_dofs,
)
from ross.seals.labyrinth_seal import LabyrinthSeal

from ross.model_reduction import ModelReduction

__all__ = [
    "Rotor",
    "CoAxialRotor",
    "rotor_example",
    "compressor_example",
    "coaxrotor_example",
    "rotor_example_6dof",
    "rotor_example_with_damping",
    "rotor_amb_example",
]

# set Plotly palette of colors
colors = px.colors.qualitative.Dark24


class Rotor(object):
    r"""A rotor object.

    This class will create a rotor with the shaft,
    disk, bearing and seal elements provided.

    Parameters
    ----------
    shaft_elements : list
        List with the shaft elements
    disk_elements : list
        List with the disk elements
    bearing_elements : list
        List with the bearing elements
    point_mass_elements: list
        List with the point mass elements
    tag : str
        A tag for the rotor

    Returns
    -------
    A rotor object.

    Attributes
    ----------
    evalues : array
        Rotor's eigenvalues.
    evectors : array
        Rotor's eigenvectors.
    wn : array
        Rotor's natural frequencies in rad/s.
    wd : array
        Rotor's damped natural frequencies in rad/s.

    Examples
    --------
    >>> #  Rotor without damping with 2 shaft elements 1 disk and 2 bearings
    >>> import ross as rs
    >>> steel = rs.materials.steel
    >>> z = 0
    >>> le = 0.25
    >>> i_d = 0
    >>> o_d = 0.05
    >>> tim0 = rs.ShaftElement(le, i_d, o_d,
    ...                        material=steel,
    ...                        shear_effects=True,
    ...                        rotary_inertia=True,
    ...                        gyroscopic=True)
    >>> tim1 = rs.ShaftElement(le, i_d, o_d,
    ...                        material=steel,
    ...                        shear_effects=True,
    ...                        rotary_inertia=True,
    ...                        gyroscopic=True)
    >>> shaft_elm = [tim0, tim1]
    >>> disk0 = rs.DiskElement.from_geometry(1, steel, 0.07, 0.05, 0.28)
    >>> stf = 1e6
    >>> bearing0 = rs.BearingElement(0, kxx=stf, cxx=0)
    >>> bearing1 = rs.BearingElement(2, kxx=stf, cxx=0)
    >>> rotor = rs.Rotor(shaft_elm, [disk0], [bearing0, bearing1])
    >>> modal = rotor.run_modal(speed=0)
    >>> modal.wd[0] # doctest: +ELLIPSIS
    215.3707...
    """

    def __init__(
        self,
        shaft_elements,
        disk_elements=None,
        bearing_elements=None,
        point_mass_elements=None,
        min_w=None,
        max_w=None,
        rated_w=None,
        tag=None,
    ):
        self.parameters = {"min_w": min_w, "max_w": max_w, "rated_w": rated_w}
        isMultiRotor = type(self) not in (Rotor, CoAxialRotor)

        if tag is None:
            self.tag = "MultiRotor 0" if isMultiRotor else "Rotor 0"
        else:
            self.tag = tag

        ####################################################
        # Config attributes
        ####################################################

        # operational speeds
        self.min_w = min_w
        self.max_w = max_w
        self.rated_w = rated_w

        ####################################################

        # flatten shaft_elements
        def flatten(l):
            for el in l:
                if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
                    yield from flatten(el)
                else:
                    yield el

        # flatten and make a copy for shaft elements to avoid altering
        # attributes for elements that might be used in different rotors
        # e.g. altering shaft_element.n
        shaft_elements = [copy(el) for el in flatten(shaft_elements)]

        # set n for each shaft element
        for i, sh in enumerate(shaft_elements):
            if sh.n is None:
                sh.n = i
            if sh.tag is None or isMultiRotor:
                sh.tag = sh.get_class_name_prefix(i)

        if disk_elements is None:
            disk_elements = []
        if bearing_elements is None:
            bearing_elements = []
        if point_mass_elements is None:
            point_mass_elements = []

        for i, disk in enumerate(disk_elements):
            if disk.tag is None or isMultiRotor:
                disk.tag = disk.get_class_name_prefix(i)

        for i, brg in enumerate(bearing_elements):
            # add n_l and n_r to bearing elements
            brg.n_l = brg.n
            brg.n_r = brg.n
            if brg.tag is None or isMultiRotor:
                brg.tag = brg.get_class_name_prefix(i)

        for i, p_mass in enumerate(point_mass_elements):
            if p_mass.tag is None or isMultiRotor:
                p_mass.tag = p_mass.get_class_name_prefix(i)

        self.shaft_elements = sorted(shaft_elements, key=lambda el: el.n)
        self.bearing_elements = sorted(bearing_elements, key=lambda el: el.n)
        self.disk_elements = disk_elements
        self.point_mass_elements = point_mass_elements
        self.elements = [
            el
            for el in flatten(
                [
                    self.shaft_elements,
                    self.disk_elements,
                    self.bearing_elements,
                    self.point_mass_elements,
                ]
            )
        ]

        # check if tags are unique
        tags_list = [el.tag for el in self.elements]
        if len(tags_list) != len(set(tags_list)):
            raise ValueError("Tags should be unique.")

        self.number_dof = self._check_number_dof()

        ####################################################
        # Rotor summary
        ####################################################
        columns = [
            "type",
            "n",
            "n_link",
            "L",
            "node_pos",
            "node_pos_r",
            "idl",
            "odl",
            "idr",
            "odr",
            "i_d",
            "o_d",
            "beam_cg",
            "axial_cg_pos",
            "y_pos",
            "material",
            "rho",
            "volume",
            "m",
            "tag",
        ]

        df_shaft = pd.DataFrame([el.summary() for el in self.shaft_elements])
        df_disks = pd.DataFrame([el.summary() for el in self.disk_elements])
        df_bearings = pd.DataFrame(
            [
                el.summary()
                for el in self.bearing_elements
                if not (isinstance(el, SealElement))
            ]
        )
        df_seals = pd.DataFrame(
            [
                el.summary()
                for el in self.bearing_elements
                if (isinstance(el, SealElement))
            ]
        )
        df_point_mass = pd.DataFrame([el.summary() for el in self.point_mass_elements])

        nodes_pos_l = np.zeros(len(df_shaft.n_l))
        nodes_pos_r = np.zeros(len(df_shaft.n_l))
        axial_cg_pos = np.zeros(len(df_shaft.n_l))

        for i, sh in enumerate(self.shaft_elements):
            if i == 0:
                nodes_pos_r[i] = nodes_pos_r[i] + df_shaft.loc[i, "L"]
                axial_cg_pos[i] = sh.beam_cg + nodes_pos_l[i]
                sh.axial_cg_pos = axial_cg_pos[i]
                continue
            if df_shaft.loc[i, "n_l"] == df_shaft.loc[i - 1, "n_l"]:
                nodes_pos_l[i] = nodes_pos_l[i - 1]
                nodes_pos_r[i] = nodes_pos_r[i - 1]
            else:
                nodes_pos_l[i] = nodes_pos_r[i - 1]
                if isMultiRotor:
                    self._fix_nodes_pos(i, sh.n, nodes_pos_l)
                nodes_pos_r[i] = nodes_pos_l[i] + df_shaft.loc[i, "L"]
            axial_cg_pos[i] = sh.beam_cg + nodes_pos_l[i]
            sh.axial_cg_pos = axial_cg_pos[i]

        df_shaft["nodes_pos_l"] = nodes_pos_l
        df_shaft["nodes_pos_r"] = nodes_pos_r
        df_shaft["axial_cg_pos"] = axial_cg_pos

        df = pd.concat(
            [df_shaft, df_disks, df_bearings, df_point_mass, df_seals], sort=True
        )
        df = df.sort_values(by="n_l")
        df = df.reset_index(drop=True)
        df["shaft_number"] = np.zeros(len(df))

        df_shaft["shaft_number"] = np.zeros(len(df_shaft))
        df_disks["shaft_number"] = np.zeros(len(df_disks))
        df_bearings["shaft_number"] = np.zeros(len(df_bearings))
        df_seals["shaft_number"] = np.zeros(len(df_seals))
        df_point_mass["shaft_number"] = np.zeros(len(df_point_mass))

        self.df_disks = df_disks
        self.df_bearings = df_bearings
        self.df_shaft = df_shaft
        self.df_point_mass = df_point_mass
        self.df_seals = df_seals

        # check consistence for disks and bearings location
        if len(df_point_mass) > 0:
            max_loc_point_mass = df_point_mass.n.max()
        else:
            max_loc_point_mass = 0
        max_location = max(df_shaft.n_r.max(), max_loc_point_mass)
        if df.n_l.max() > max_location:
            raise ValueError("Trying to set disk or bearing outside shaft")

        # nodes axial position and diameter
        self.nodes_pos = list(df_shaft.groupby("n_l")["nodes_pos_l"].max())
        self.nodes_pos.append(df_shaft["nodes_pos_r"].iloc[-1])

        self.nodes = list(df_shaft.groupby("n_l")["n_l"].max())
        self.nodes.append(df_shaft["n_r"].iloc[-1])

        self.center_line_pos = [0] * len(self.nodes)

        if isMultiRotor:
            self._fix_nodes()

        nodes_i_d = []
        for n in self.nodes:
            nodes_i_d.append(
                self.df_shaft[
                    (self.df_shaft.n_l == n) | (self.df_shaft.n_r == n)
                ].i_d.min()
            )
        self.nodes_i_d = nodes_i_d

        nodes_o_d = []
        for n in self.nodes:
            nodes_o_d.append(
                self.df_shaft[
                    (self.df_shaft.n_l == n) | (self.df_shaft.n_r == n)
                ].o_d.max()
            )
        self.nodes_o_d = nodes_o_d

        shaft_elements_length = list(df_shaft.groupby("n_l")["L"].min())
        self.shaft_elements_length = shaft_elements_length

        self.L = self.nodes_pos[-1]

        if "n_link" in df.columns:
            self.link_nodes = list(df["n_link"].dropna().unique().astype(int))
        else:
            self.link_nodes = []

        # rotor mass can also be calculated with self.M()[::4, ::4].sum()
        self.m_disks = np.sum([disk.m for disk in self.disk_elements])
        self.m_shaft = np.sum([sh_el.m for sh_el in self.shaft_elements])
        self.m = self.m_disks + self.m_shaft

        # rotor center of mass and total inertia
        CG_sh = np.sum(
            [(sh.m * sh.axial_cg_pos) / self.m for sh in self.shaft_elements]
        )
        CG_dsk = np.sum(
            [
                disk.m * self.nodes_pos[self.nodes.index(disk.n)] / self.m
                for disk in self.disk_elements
            ]
        )
        self.CG = CG_sh + CG_dsk

        Ip_sh = np.sum([sh.Im for sh in self.shaft_elements])
        Ip_dsk = np.sum([disk.Ip for disk in self.disk_elements])
        self.Ip = Ip_sh + Ip_dsk

        # number of dofs
        half_ndof = self.number_dof / 2
        self.ndof = int(
            self.number_dof * len(self.nodes)
            + half_ndof * len(self.point_mass_elements)
        )

        # global indexes for dofs
        n_last = self.shaft_elements[-1].n
        for elm in self.elements:
            dof_mapping = elm.dof_mapping()
            global_dof_mapping = {}
            for k, v in dof_mapping.items():
                dof_letter, dof_number = k.split("_")
                global_dof_mapping[dof_letter + "_" + str(int(dof_number) + elm.n)] = (
                    int(v)
                )

            if elm.n <= n_last + 1:
                for k, v in global_dof_mapping.items():
                    global_dof_mapping[k] = int(self.number_dof * elm.n + v)
            else:
                for k, v in global_dof_mapping.items():
                    global_dof_mapping[k] = int(
                        half_ndof * n_last + half_ndof * elm.n + self.number_dof + v
                    )

            if hasattr(elm, "n_link") and elm.n_link is not None:
                if elm.n_link <= n_last + 1:
                    global_dof_mapping[f"x_{elm.n_link}"] = int(
                        self.number_dof * elm.n_link
                    )
                    global_dof_mapping[f"y_{elm.n_link}"] = int(
                        self.number_dof * elm.n_link + 1
                    )
                    global_dof_mapping[f"z_{elm.n_link}"] = int(
                        self.number_dof * elm.n_link + 2
                    )
                else:
                    global_dof_mapping[f"x_{elm.n_link}"] = int(
                        half_ndof * n_last + half_ndof * elm.n_link + self.number_dof
                    )
                    global_dof_mapping[f"y_{elm.n_link}"] = int(
                        half_ndof * n_last
                        + half_ndof * elm.n_link
                        + self.number_dof
                        + 1
                    )
                    global_dof_mapping[f"z_{elm.n_link}"] = int(
                        half_ndof * n_last
                        + half_ndof * elm.n_link
                        + self.number_dof
                        + 2
                    )

            elm.dof_global_index = global_dof_mapping
            df.at[df.loc[df.tag == elm.tag].index[0], "dof_global_index"] = (
                elm.dof_global_index
            )

        # define positions for disks
        for elm in self.disk_elements:
            i = self.nodes.index(elm.n)
            z_pos = self.nodes_pos[i]
            y_pos = self.nodes_o_d[i] / 2
            df.loc[df.tag == elm.tag, "nodes_pos_l"] = z_pos
            df.loc[df.tag == elm.tag, "nodes_pos_r"] = z_pos
            df.loc[df.tag == elm.tag, "y_pos"] = y_pos

        # define positions for bearings
        for elm in self.bearing_elements:
            node = elm.n
            if node in self.link_nodes:
                node = self._find_linked_bearing_node(node)

            i = self.nodes.index(node)
            z_pos = self.nodes_pos[i]
            df.loc[df.tag == elm.tag, "nodes_pos_l"] = z_pos
            df.loc[df.tag == elm.tag, "nodes_pos_r"] = z_pos

        bclass = BearingElement
        classes = [cls.__name__ for cls in ([bclass] + bclass.get_subclasses())]

        dfb = df[df.type.isin(classes)]
        z_positions = [pos for pos in dfb["nodes_pos_l"]]
        z_positions = list(dict.fromkeys(z_positions))
        for z_pos in z_positions:
            dfb_z_pos = dfb[dfb.nodes_pos_l == z_pos]
            dfb_z_pos = dfb_z_pos.sort_values(by="n_l")

            mean_od = np.mean(self.nodes_o_d)
            scale_size = dfb["scale_factor"] * mean_od

            for i in range(len(dfb_z_pos)):
                t = dfb_z_pos.iloc[i].tag

                n_l = df.loc[df.tag == t, "n_l"].values[0]
                if n_l in self.link_nodes:
                    scale_size_link = (
                        df["scale_factor"][df.tag == t].values[0] * mean_od
                    )

                    y_pos = df.loc[df.n_link == n_l, "y_pos_sup"].values[
                        0
                    ]  # equal to y_pos_sup of linked bearing

                    df.loc[df.tag == t, "y_pos"] = y_pos
                    df.loc[df.tag == t, "y_pos_sup"] = y_pos + scale_size_link

                else:
                    try:
                        y_pos = (
                            max(
                                df_shaft["odl"][
                                    df_shaft.n_l == int(dfb_z_pos.iloc[i]["n_l"])
                                ].values
                            )
                            / 2
                        )
                    except ValueError:
                        try:
                            y_pos = (
                                max(
                                    df_shaft["odr"][
                                        df_shaft.n_r == int(dfb_z_pos.iloc[i]["n_r"])
                                    ].values
                                )
                                / 2
                            )
                        except ValueError:
                            y_pos = (
                                max(
                                    [
                                        max(
                                            df_shaft["odl"][
                                                df_shaft._n
                                                == int(dfb_z_pos.iloc[i]["n_l"])
                                            ].values
                                        ),
                                        max(
                                            df_shaft["odr"][
                                                df_shaft._n
                                                == int(dfb_z_pos.iloc[i]["n_l"]) - 1
                                            ].values
                                        ),
                                    ]
                                )
                                / 2
                            )

                    df.loc[df.tag == t, "y_pos"] = y_pos
                    df.loc[df.tag == t, "y_pos_sup"] = y_pos + scale_size

        # define position for point mass elements
        dfb = df[df.type.isin(classes)]
        for p in point_mass_elements:
            z_pos = dfb[dfb.n_l == p.n]["nodes_pos_l"].values[0]
            y_pos = dfb[dfb.n_l == p.n]["y_pos"].values[0]
            df.loc[df.tag == p.tag, "nodes_pos_l"] = z_pos
            df.loc[df.tag == p.tag, "nodes_pos_r"] = z_pos
            df.loc[df.tag == p.tag, "y_pos"] = y_pos

        self.df = df

        # Base matrices:
        M0 = np.zeros((self.ndof, self.ndof))
        C0 = np.zeros((self.ndof, self.ndof))
        K0 = np.zeros((self.ndof, self.ndof))
        G0 = np.zeros((self.ndof, self.ndof))
        Ksdt0 = np.zeros((self.ndof, self.ndof))

        elements = list(set(self.elements).difference(self.bearing_elements))

        for elm in elements:
            dofs = list(elm.dof_global_index.values())

            M0[np.ix_(dofs, dofs)] += elm.M()
            C0[np.ix_(dofs, dofs)] += elm.C()
            K0[np.ix_(dofs, dofs)] += elm.K()
            G0[np.ix_(dofs, dofs)] += elm.G()

            if elm in self.shaft_elements:
                Ksdt0[np.ix_(dofs, dofs)] += elm.Kst()
            elif elm in self.disk_elements:
                Ksdt0[np.ix_(dofs, dofs)] += elm.Kdt()

        self.M0 = M0
        self.C0 = C0
        self.K0 = K0
        self.G0 = G0
        self.Ksdt0 = Ksdt0

    def _check_number_dof(self):
        """Verify the consistency of degrees of freedom.

        This method loops for all the elements, checking if the number of degrees of
        freedom is consistent.
        E.g.: inputting 2 shaft elements, one with 4 dof and one with 6, will raise
        an error.

        Raises
        ------
        Exception
            Error pointing out difference between the number of DoF's from each element
            type.

        Returns
        -------
        number_dof : int
            Number of degrees of freedom from the adopted shaft element.
        """
        number_dof = len(self.shaft_elements[0].dof_mapping()) / 2

        if any(len(sh.dof_mapping()) != number_dof * 2 for sh in self.shaft_elements):
            raise Exception(
                "The number of degrees o freedom of all elements must be the same! There are SHAFT elements with discrepant DoFs."
            )

        if any(len(disk.dof_mapping()) != number_dof for disk in self.disk_elements):
            raise Exception(
                "The number of degrees o freedom of all elements must be the same! There are DISK elements with discrepant DoFs."
            )

        if any(
            len(brg.dof_mapping()) != number_dof / 2 for brg in self.bearing_elements
        ):
            raise Exception(
                "The number of degrees o freedom of all elements must be the same! There are BEARING elements with discrepant DoFs."
            )

        return int(number_dof)

    def _find_linked_bearing_node(self, node):
        """Find the linked bearing element by node

        Parameters
        ----------
        node : int
            Node number to search for a linked bearing element.

        Returns
        -------
        node_found : int or None
            The bearing element node linked to the specified node, or None if not found.
        """
        for brg in self.bearing_elements:
            if brg.n_link == node:
                node_found = self._find_linked_bearing_node(brg.n)
                if node_found is not None:
                    return node_found
                else:
                    return brg.n
        return None

    def __eq__(self, other):
        """Equality method for comparasions.

        Parameters
        ----------
        other : obj
            parameter for comparasion

        Returns
        -------
        True if other is equal to the reference parameter.
        False if not.
        """
        if self.elements == other.elements and self.parameters == other.parameters:
            return True
        else:
            return False

    def add_nodes(self, new_nodes_pos):
        """Add nodes to rotor.

        This method returns the modified rotor with additional nodes according to
        the positions of the new nodes provided.

        Parameters
        ----------
        new_nodes_pos : list
            List with the position of the new nodes.

        Returns
        -------
        A rotor object.

        Examples
        --------
        >>> import ross as rs
        >>> rotor = rs.rotor_example()
        >>> new_rotor = rotor.add_nodes([0.62, 1.11])
        >>> shaft_elements = new_rotor.shaft_elements
        >>> len(shaft_elements)
        8
        >>> round(shaft_elements[3].L, 2)
        0.13
        >>> round(shaft_elements[6].L, 2)
        0.14
        """
        new_nodes_pos.sort()

        shaft_elements = deepcopy(self.shaft_elements)
        disk_elements = deepcopy(self.disk_elements)
        bearing_elements = deepcopy(self.bearing_elements)
        point_mass_elements = deepcopy(self.point_mass_elements)

        elements = [
            *shaft_elements,
            *disk_elements,
            *bearing_elements,
            *point_mass_elements,
        ]

        target_elements = []
        new_elems_length = []

        for new_pos in new_nodes_pos:
            for elm in shaft_elements:
                elm.tag = None

                pos_l = self.nodes_pos[self.nodes.index(elm.n_l)]
                pos_r = self.nodes_pos[self.nodes.index(elm.n_r)]

                if new_pos > pos_l and new_pos < pos_r:
                    target_elements.append(elm)
                    new_elems_length.append(pos_r - new_pos)

        prev_left_node = -1

        for i in range(len(target_elements)):
            elem = target_elements[i]

            left_elem = elem.create_modified(L=(elem.L - new_elems_length[i]))
            right_elem = elem.create_modified(L=new_elems_length[i], n=(elem.n + 1))

            if left_elem.n != prev_left_node:
                for elm in elements:
                    if elm.n >= right_elem.n:
                        elm.n += 1
                        if elm in shaft_elements:
                            elm._n = elm.n
                            elm.n_l = elm.n
                            elm.n_r = elm.n + 1
                        if elm in point_mass_elements:
                            for brg in bearing_elements:
                                if elm.n - 1 == brg.n_link:
                                    brg.n_link += 1

            for j in range(i + 1, len(target_elements)):
                if target_elements[j] == target_elements[i]:
                    target_elements[j] = right_elem

            idx_left = shaft_elements.index(elem)
            shaft_elements[idx_left] = left_elem

            idx_right = idx_left + len(
                [k for k, elm in enumerate(shaft_elements) if elm.n == left_elem.n]
            )
            shaft_elements.insert(idx_right, right_elem)

            prev_left_node = left_elem.n

        return Rotor(
            shaft_elements,
            disk_elements=disk_elements,
            bearing_elements=bearing_elements,
            point_mass_elements=point_mass_elements,
            min_w=self.min_w,
            max_w=self.max_w,
            rated_w=self.rated_w,
            tag=self.tag,
        )

    @lru_cache()
    @check_units
    def run_modal(self, speed, num_modes=12, sparse=True, synchronous=False):
        """Run modal analysis.

        Method to calculate eigenvalues and eigvectors for a given rotor system.
        The natural frequencies and dampings ratios are calculated for a given
        rotor speed. It means that for each speed input there's a different set of
        eigenvalues and eigenvectors, hence, different natural frequencies and damping
        ratios are returned.
        This method will return a ModalResults object which stores all data generated
        and also provides methods for plotting.

        Available plotting methods:
            .plot_mode_2d()
            .plot_mode_3d()

        Parameters
        ----------
        speed : float
            Speed at which the eigenvalues and eigenvectors will be calculated.
        num_modes : int, optional
            The number of eigenvalues and eigenvectors to be calculated using ARPACK.
            If sparse=True, it determines the number of eigenvalues and eigenvectors
            to be calculated. It must be smaller than Rotor.ndof - 1. It is not
            possible to compute all eigenvectors of a matrix with ARPACK.
            If sparse=False, num_modes does not have any effect over the method.
            Default is 12.
        sparse : bool, optional
            If True, ARPACK is used to calculate a desired number (according to
            num_modes) or eigenvalues and eigenvectors.
            If False, `scipy.linalg.eig()` is used to calculate all the eigenvalues and
            eigenvectors.
            Default is True.
        synchronous : bool, optional
            If True a synchronous analysis is carried out.
            Default is False.

        Returns
        -------
        results : ross.ModalResults
            For more information on attributes and methods available see:
            :py:class:`ross.ModalResults`

        Examples
        --------
        >>> import ross as rs
        >>> rotor = rs.rotor_example()
        >>> modal = rotor.run_modal(speed=0, sparse=False)
        >>> modal.wn[:2]
        array([91.79655318, 96.28899977])
        >>> modal.wd[:2]
        array([91.79655318, 96.28899977])
        >>> # Plotting 3D mode shape
        >>> mode1 = 0  # First mode
        >>> fig = modal.plot_mode_3d(mode1)
        >>> # Plotting 2D mode shape
        >>> mode2 = 1  # Second mode
        >>> fig = modal.plot_mode_2d(mode2)
        """
        evalues, evectors = self._eigen(
            speed, num_modes=num_modes, sparse=sparse, synchronous=synchronous
        )

        wn_len = num_modes // 2
        wn = (np.absolute(evalues))[:wn_len]
        wd = (np.imag(evalues))[:wn_len]
        damping_ratio = (-np.real(evalues) / np.absolute(evalues))[:wn_len]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            log_dec = 2 * np.pi * damping_ratio / np.sqrt(1 - damping_ratio**2)

        modal_results = ModalResults(
            speed,
            evalues,
            evectors,
            wn,
            wd,
            damping_ratio,
            log_dec,
            self.ndof,
            self.nodes,
            self.nodes_pos,
            self.shaft_elements_length,
            self.number_dof,
        )

        return modal_results

    @check_units
    def run_critical_speed(self, speed_range=None, num_modes=12, rtol=0.005):
        """Calculate the critical speeds and damping ratios for the rotor model.

        This function runs an iterative method over "run_modal()" to minimize
        (using scipy.optimize.newton) the error between the rotor speed and the rotor
        critical speeds (rotor speed - critical speed).

        Differently from run_modal(), this function doesn't take a speed input because
        it iterates over the natural frequencies calculated in the last iteration.
        The initial value is considered to be the undamped natural frequecies for
        speed = 0 (no gyroscopic effect).

        Once the error is within an acceptable range defined by "rtol", it returns the
        approximated critical speed.

        With the critical speeds calculated, the function uses the results to
        calculate the log dec and damping ratios for each critical speed.

        Parameters
        ----------
        speed_range : tuple, optional, pint.Quantity
            Tuple (start, end) with the desired range of frequencies (rad/s).
            The function returns all eigenvalues within this range.
        num_modes : int, optional
            The number of eigenvalues and eigenvectors to be calculated using ARPACK.
            If sparse=True, it determines the number of eigenvalues and eigenvectors
            to be calculated. It must be smaller than Rotor.ndof - 1. It is not
            possible to compute all eigenvectors of a matrix with ARPACK.
            If speed_range is not None, num_modes is overrided.
            Default is 12.
        rtol : float, optional
            Tolerance (relative) for termination. Applied to scipy.optimize.newton.
            Default is 0.005 (0.5%).

        Returns
        -------
        results : ross.CriticalSpeedResults
            For more information on attributes and methods available see:
            :py:class:`ross.CriticalSpeedResults`

        Examples
        --------
        >>> import ross as rs
        >>> rotor = rs.rotor_example()

        Finding the first Nth critical speeds
        >>> results = rotor.run_critical_speed(num_modes=8)
        >>> np.round(results.wd())
        array([ 92.,  96., 271., 300.])
        >>> np.round(results.wn())
        array([ 92.,  96., 271., 300.])

        Finding the first critical speeds within a speed range
        >>> results = rotor.run_critical_speed(speed_range=(100, 1000))
        >>> np.round(results.wd())
        array([271., 300., 636., 774., 867.])

        Changing output units
        >>> np.round(results.wd("rpm"))
        array([2590., 2868., 6074., 7394., 8278.])

        Retrieving whirl directions
        >>> results.whirl_direction # doctest: +ELLIPSIS
        array([...
        """
        num_modes = (self.ndof - 4) * 2 if speed_range is not None else num_modes

        modal = self.run_modal(0, num_modes)
        _wn = modal.wn
        _wd = modal.wd
        wn = np.zeros_like(_wn)
        wd = np.zeros_like(_wd)

        for i in range(len(wn)):
            wn_func = lambda s: (s - self.run_modal(s, num_modes).wn[i])
            wn[i] = newton(func=wn_func, x0=_wn[i], rtol=rtol)

        for i in range(len(wd)):
            wd_func = lambda s: (s - self.run_modal(s, num_modes).wd[i])
            wd[i] = newton(func=wd_func, x0=_wd[i], rtol=rtol)

        log_dec = np.zeros_like(wn)
        damping_ratio = np.zeros_like(wn)
        whirl_direction = list(np.zeros_like(wn))
        for i, s in enumerate(wd):
            modal = self.run_modal(s, num_modes)
            log_dec[i] = modal.log_dec[i]
            damping_ratio[i] = modal.damping_ratio[i]
            whirl_direction[i] = modal.whirl_direction()[i]

        whirl_direction = np.array(whirl_direction)
        if speed_range is not None:
            vmin, vmax = speed_range
            idx = np.where((wd >= vmin) & (wd <= vmax))
            wn = wn[idx]
            wd = wd[idx]
            log_dec = log_dec[idx]
            damping_ratio = damping_ratio[idx]
            whirl_direction = whirl_direction[idx]

        return CriticalSpeedResults(wn, wd, log_dec, damping_ratio, whirl_direction)

    def convergence(self, n_eigval=0, err_max=1e-02):
        """Run convergence analysis.

        Function to analyze the eigenvalues convergence through the number of
        shaft elements. Every new run doubles the number os shaft elements.

        Parameters
        ----------
        n_eigval : int
            The nth eigenvalue which the convergence analysis will run.
            Default is 0 (the first eigenvalue).
        err_max : float
            Maximum allowable convergence error.
            Default is 1e-02

        Returns
        -------
        results : An instance of ConvergenceResults class, which is used to post-process
        results. Attributes stored:
            el_num : array
                Array with number of elements in each iteraction
            eigv_arr : array
                Array with the n'th natural frequency in each iteraction
            error_arr : array
                Array with the relative error in each iteraction

        Example
        -------
        >>> import ross as rs
        >>> i_d = 0
        >>> o_d = 0.05
        >>> n = 6
        >>> L = [0.25 for _ in range(n)]
        ...
        >>> shaft_elem = [rs.ShaftElement(l, i_d, o_d, material=steel,
        ... shear_effects=True, rotary_inertia=True, gyroscopic=True) for l in L]
        >>> disk0 = DiskElement.from_geometry(2, steel, 0.07, 0.05, 0.28)
        >>> disk1 = DiskElement.from_geometry(4, steel, 0.07, 0.05, 0.35)
        >>> bearing0 = BearingElement(0, kxx=1e6, kyy=8e5, cxx=2e3)
        >>> bearing1 = BearingElement(6, kxx=1e6, kyy=8e5, cxx=2e3)
        >>> rotor0 = Rotor(shaft_elem, [disk0, disk1], [bearing0, bearing1])
        >>> len(rotor0.shaft_elements)
        6
        >>> convergence = rotor0.convergence(n_eigval=0, err_max=1e-08)
        >>> len(rotor0.shaft_elements)
        96

        Plotting convergence graphics
        >>> fig = convergence.plot()
        """
        el_num = np.array([len(self.shaft_elements)])
        eigv_arr = np.array([])
        error_arr = np.array([0])

        modal = self.run_modal(speed=0)
        eigv_arr = np.append(eigv_arr, modal.wn[n_eigval])

        # this value is up to start the loop while
        error = 1.0e10
        nel_r = 2

        while error > err_max:
            shaft_elem = []
            disk_elem = []
            brgs_elem = []

            for shaft in self.shaft_elements:
                le = shaft.L / nel_r
                odl = shaft.odl
                odr = shaft.odr
                idl = shaft.idl
                idr = shaft.idr

                # loop to double the number of element
                for j in range(nel_r):
                    odr = ((nel_r - j - 1) * odl + (j + 1) * odr) / nel_r
                    idr = ((nel_r - j - 1) * idl + (j + 1) * idr) / nel_r
                    odl = ((nel_r - j) * odl + j * odr) / nel_r
                    idl = ((nel_r - j) * idl + j * idr) / nel_r
                    shaft_elem.append(
                        ShaftElement(
                            L=le,
                            idl=idl,
                            odl=odl,
                            idr=idr,
                            odr=odr,
                            material=shaft.material,
                            shear_effects=shaft.shear_effects,
                            rotary_inertia=shaft.rotary_inertia,
                            gyroscopic=shaft.gyroscopic,
                        )
                    )

            for DiskEl in self.disk_elements:
                aux_DiskEl = deepcopy(DiskEl)
                aux_DiskEl.n = nel_r * DiskEl.n
                disk_elem.append(aux_DiskEl)

            for Brg_SealEl in self.bearing_elements:
                aux_Brg_SealEl = deepcopy(Brg_SealEl)
                aux_Brg_SealEl.n = nel_r * Brg_SealEl.n
                brgs_elem.append(aux_Brg_SealEl)

            aux_rotor = Rotor(shaft_elem, disk_elem, brgs_elem)
            aux_modal = aux_rotor.run_modal(speed=0)

            eigv_arr = np.append(eigv_arr, aux_modal.wn[n_eigval])
            el_num = np.append(el_num, len(shaft_elem))

            error = abs(1 - eigv_arr[-1] / eigv_arr[-2])

            error_arr = np.append(error_arr, 100 * error)
            nel_r *= 2

        self.__dict__ = aux_rotor.__dict__
        self.error_arr = error_arr

        results = ConvergenceResults(el_num[1:], eigv_arr[1:], error_arr[1:])

        return results

    def M(self, frequency=None, synchronous=False):
        """Mass matrix for an instance of a rotor.

        Parameters
        ----------
        synchronous : bool, optional
            If True a synchronous analysis is carried out.
            Default is False.

        Returns
        -------
        M0 : np.ndarray
            Mass matrix for the rotor.

        Examples
        --------
        >>> rotor = rotor_example()
        >>> rotor.M(0)[:4, :4]
        array([[ 1.42050794,  0.        ,  0.        ,  0.        ],
               [ 0.        ,  1.42050794,  0.        , -0.04931719],
               [ 0.        ,  0.        ,  1.27790826,  0.        ],
               [ 0.        , -0.04931719,  0.        ,  0.00231392]])
        """
        # if frequency is None, we assume the rotor does not have any elements
        # with frequency dependent mass matrices
        if frequency is None:
            frequency = 0

        M0 = self.M0.copy()

        for elm in self.bearing_elements:
            dofs = list(elm.dof_global_index.values())
            M0[np.ix_(dofs, dofs)] += elm.M(frequency)

        if synchronous:
            for elm in self.shaft_elements:
                dofs = list(elm.dof_global_index.values())
                x0 = elm.dof_mapping()["x_0"]
                y0 = elm.dof_mapping()["y_0"]
                a0 = elm.dof_mapping()["alpha_0"]
                b0 = elm.dof_mapping()["beta_0"]
                x1 = elm.dof_mapping()["x_1"]
                y1 = elm.dof_mapping()["y_1"]
                a1 = elm.dof_mapping()["alpha_1"]
                b1 = elm.dof_mapping()["beta_1"]
                G = elm.G()
                for i in range(2 * self.number_dof):
                    if i in (x0, b0, x1, b1):
                        M0[dofs[i], dofs[x0]] -= G[i, y0]
                        M0[dofs[i], dofs[b0]] += G[i, a0]
                        M0[dofs[i], dofs[x1]] -= G[i, y1]
                        M0[dofs[i], dofs[b1]] += G[i, a1]
                    else:
                        M0[dofs[i], dofs[y0]] += G[i, x0]
                        M0[dofs[i], dofs[a0]] -= G[i, b0]
                        M0[dofs[i], dofs[y1]] += G[i, x1]
                        M0[dofs[i], dofs[a1]] -= G[i, b1]
            for elm in self.disk_elements:
                dofs = list(elm.dof_global_index.values())
                a0 = elm.dof_mapping()["alpha_0"]
                b0 = elm.dof_mapping()["beta_0"]
                G = elm.G()
                M0[dofs[a0], dofs[a0]] -= G[a0, b0]
                M0[dofs[b0], dofs[b0]] += G[b0, a0]

        return M0

    def K(self, frequency):
        """Stiffness matrix for an instance of a rotor.

        Parameters
        ----------
        frequency : float, optional
            Excitation frequency.

        Returns
        -------
        K0 : np.ndarray
            Stiffness matrix for the rotor.

        Examples
        --------
        >>> rotor = rotor_example()
        >>> np.round(rotor.K(0)[:4, :4] / 1e6)
        array([[ 4.700e+01,  0.000e+00,  0.000e+00,  0.000e+00],
               [ 0.000e+00,  4.600e+01,  0.000e+00, -6.000e+00],
               [ 0.000e+00,  0.000e+00,  1.657e+03,  0.000e+00],
               [ 0.000e+00, -6.000e+00,  0.000e+00,  1.000e+00]])
        """
        K0 = self.K0.copy()

        for elm in self.bearing_elements:
            dofs = list(elm.dof_global_index.values())
            K0[np.ix_(dofs, dofs)] += elm.K(frequency)

        return K0

    def Ksdt(self):
        """Dynamic stiffness matrix for an instance of a rotor.

        Stiffness matrix associated with the transient motion of the
        shaft and disks. It needs to be multiplied by the angular
        acceleration when considered in time dependent analyses.

        Returns
        -------
        Ksdt0 : np.ndarray
            Dynamic stiffness matrix for the rotor.

        Examples
        --------
        >>> rotor = rotor_example_6dof()
        >>> np.round(rotor.Ksdt()[:6, :6] * 1e3, 2)
        array([[  0.  , -23.  ,   0.  ,   0.48,   0.  ,   0.  ],
               [  0.  ,   0.  ,   0.  ,   0.  ,   0.  ,   0.  ],
               [  0.  ,   0.  ,   0.  ,   0.  ,   0.  ,   0.  ],
               [  0.  ,   0.  ,   0.  ,   0.  ,   0.  ,   0.  ],
               [  0.  ,  -0.48,   0.  ,   0.16,   0.  ,   0.  ],
               [  0.  ,   0.  ,   0.  ,   0.  ,   0.  ,   0.  ]])
        """
        Ksdt0 = self.Ksdt0.copy()

        return Ksdt0

    def C(self, frequency):
        """Damping matrix for an instance of a rotor.

        Parameters
        ----------
        frequency : float
            Excitation frequency.

        Returns
        -------
        C0 : np.ndarray
            Damping matrix for the rotor.

        Examples
        --------
        >>> rotor = compressor_example()
        >>> rotor.C(0)[:4, :4]
        array([[0., 0., 0., 0.],
               [0., 0., 0., 0.],
               [0., 0., 0., 0.],
               [0., 0., 0., 0.]])
        """
        C0 = self.C0.copy()

        for elm in self.bearing_elements:
            dofs = list(elm.dof_global_index.values())
            C0[np.ix_(dofs, dofs)] += elm.C(frequency)

        return C0

    def G(self):
        """Gyroscopic matrix for an instance of a rotor.

        Returns
        -------
        G0 : np.ndarray
            Gyroscopic matrix for the rotor.

        Examples
        --------
        >>> rotor = rotor_example()
        >>> rotor.G()[:4, :4]
        array([[ 0.        ,  0.01943344,  0.        , -0.00022681],
               [-0.01943344,  0.        ,  0.        ,  0.        ],
               [ 0.        ,  0.        ,  0.        ,  0.        ],
               [ 0.00022681,  0.        ,  0.        ,  0.        ]])
        """
        G0 = self.G0.copy()

        return G0

    def A(self, speed=0, frequency=None, synchronous=False):
        """State space matrix for an instance of a rotor.

        Parameters
        ----------
        speed: float, optional
            Rotor speed.
            Default is 0.
        frequency : float, optional
            Excitation frequency. Default is rotor speed.
        synchronous : bool, optional
            If True a synchronous analysis is carried out.
            Default is False.

        Returns
        -------
        A : np.ndarray
            State space matrix for the rotor.

        Examples
        --------
        >>> rotor = rotor_example()
        >>> np.round(rotor.A()[75:83, :2]) + 0.
        array([[     0.,  10927.],
               [-10924.,      0.],
               [     0.,      0.],
               [  -174.,      0.],
               [     0.,   -174.],
               [     0.,      0.],
               [     0.,  10723.],
               [-10719.,      0.]])
        """
        if frequency is None:
            frequency = speed

        M = self.M(frequency, synchronous=synchronous)
        size = M.shape[0]

        Z = np.zeros((size, size))
        I = np.eye(size)

        # fmt: off
        A = np.vstack(
            [np.hstack([Z, I]),
             np.hstack([la.solve(-M, self.K(frequency)), la.solve(-M, (self.C(frequency) + self.G() * speed))])])
        # fmt: on

        return A

    def _check_frequency_array(self, frequency_range):
        """Verify if bearing elements coefficients are extrapolated.

        This method takes the frequency / speed range array applied to a particular
        method (run_campbell, run_freq_response) and checks if it's extrapolating the
        bearing rotordynamics coefficients.

        If any value of frequency_range argument is out of any bearing frequency
        parameter, the warning is raised.
        If none of the bearings has a frequency argument assigned, no warning will be
        raised.

        Parameters
        ----------
        frequency_range : array
            The array of frequencies or speeds used in particular method.

        Warnings
        --------
            It warns the user if the frequency_range causes the bearing coefficients
            to be extrapolated.
        """
        for bearing in self.bearing_elements:
            if bearing.frequency is not None:
                if (np.max(frequency_range) > max(bearing.frequency)) or (
                    np.min(frequency_range) < min(bearing.frequency)
                ):
                    warnings.warn(
                        "Extrapolating bearing coefficients. Be careful when post-processing the results."
                    )
                    break

    def _clustering_points(self, num_modes=12, num_points=10, modes=None, rtol=0.005):
        """Create an array with points clustered close to the natural frequencies.

        This method generates an automatic array to run frequency response analyses.
        The frequency points are calculated based on the damped natural frequencies and
        their respective damping ratios. The greater the damping ratio, the more spread
        the points are. If the damping ratio, for a given critical speed, is smaller
        than 0.005, it is redefined to be 0.005 (for this method only).

        Parameters
        ----------
        num_modes : int, optional
            The number of eigenvalues and eigenvectors to be calculated using ARPACK.
            It also defines the range for the output array, since the method generates
            points only for the critical speed calculated by run_critical_speed().
            Default is 12.
        num_points : int, optional
            The number of points generated for each critical speed.
            The method set the same number of points for slightly less and slightly
            higher than the natural circular frequency. It means there'll be num_points
            greater and num_points smaller than a given critical speed.
            num_points may be between 2 and 12. Anything above this range defaults
            to 10 and anything below this range defaults to 4.
            The default is 10.
        modes : list, optional
            Modes that will be used to calculate the frequency response.
            The possibilities are limited by the num_modes argument.
            (all modes will be used if a list is not given).
        rtol : float, optional
            Tolerance (relative) for termination. Applied to scipy.optimize.newton in
            run_critical_speed() method.
            Default is 0.005 (0.5%).

        Returns
        -------
        speed_range : array
            Range of frequencies (or speed).

        Examples
        --------
        >>> rotor = rotor_example()
        >>> speed_range = rotor._clustering_points(num_modes=12, num_points=5)
        >>> speed_range.shape
        (61,)
        """
        critical_speeds = self.run_critical_speed(num_modes=num_modes, rtol=rtol)
        omega = critical_speeds._wd
        damping = critical_speeds.damping_ratio
        damping = np.array([d if d >= 0.005 else 0.005 for d in damping])

        if num_points > 12:
            num_points = 10
        elif num_points < 2:
            num_points = 4

        if modes is not None:
            omega = omega[modes]
            damping = damping[modes]

        a = np.zeros((len(omega), num_points))
        for i in range(len(omega)):
            for j in range(num_points):
                b = 2 * (num_points - j + 1) / (num_points - 1)
                a[i, j] = 1 + damping[i] ** b

        omega = omega.reshape((len(omega), 1))
        speed_range = np.sort(np.ravel(np.concatenate((omega / a, omega * a))))
        speed_range = np.insert(speed_range, 0, 0)

        return speed_range

    @staticmethod
    def _index(eigenvalues):
        """Generate indexes to sort eigenvalues and eigenvectors.

        Function used to generate an index that will sort
        eigenvalues and eigenvectors based on the imaginary (wd)
        part of the eigenvalues. Positive eigenvalues will be
        positioned at the first half of the array.

        Parameters
        ----------
        eigenvalues: array
            Array with the eigenvalues.

        Returns
        -------
        idx:
            An array with indices that will sort the
            eigenvalues and eigenvectors.

        Examples
        --------
        >>> rotor = rotor_example()
        >>> evalues, evectors = rotor._eigen(0, sorted_=True, sparse=False)
        >>> idx = rotor._index(evalues)
        >>> idx[:6] # doctest: +ELLIPSIS
        array([0, 1, 2, 3, 4, ...
        """
        # avoid float point errors when sorting
        evals_truncated = np.around(eigenvalues, decimals=10)
        a = np.imag(evals_truncated)  # First column
        b = np.absolute(evals_truncated)  # Second column
        ind = np.lexsort((b, a))  # Sort by imag (wd), then by absolute (wn)
        # Positive eigenvalues first
        positive = [i for i in ind[len(a) // 2 :]]
        negative = [i for i in ind[: len(a) // 2]]

        idx = np.array([*positive, *negative])

        return idx

    @check_units
    def _eigen(
        self,
        speed,
        num_modes=12,
        frequency=None,
        sorted_=True,
        A=None,
        sparse=None,
        synchronous=False,
    ):
        """Calculate eigenvalues and eigenvectors.

        This method will return the eigenvalues and eigenvectors of the
        state space matrix A, sorted by the index method which considers
        the imaginary part (wd) of the eigenvalues for sorting.
        To avoid sorting use sorted_=False

        Parameters
        ----------
        speed : float, pint.Quantity
            Rotor speed. Default unit is rad/s.
        num_modes : int, optional
            The number of eigenvalues and eigenvectors to be calculated using ARPACK.
            If sparse=True, it determines the number of eigenvalues and eigenvectors
            to be calculated. It must be smaller than Rotor.ndof - 1. It is not
            possible to compute all eigenvectors of a matrix with ARPACK.
            If sparse=False, num_modes does not have any effect over the method.
            Default is 12.
        frequency: float, pint.Quantity
            Excitation frequency. Default units is rad/s.
        sorted_ : bool, optional
            Sort considering the imaginary part (wd).
            Default is True.
        A : np.array, optional
            Matrix for which eig will be calculated.
            Defaul is the rotor A matrix.
        sparse : bool, optional
            If True, eigenvalues are computed using ARPACK. If False, they are
            computed with `scipy.linalg.eig()`. When sparse is False, eigenvalues
            are filtered to exclude rigid body modes. If sparse is None, no filtering
            is applied. Default is None.
        synchronous : bool, optional
            If True a synchronous analysis is carried out.
            Default is False.

        Returns
        -------
        evalues: array
            An array with the eigenvalues
        evectors array
            An array with the eigenvectors

        Examples
        --------
        >>> rotor = rotor_example()
        >>> evalues, evectors = rotor._eigen(0, sparse=False)
        >>> evalues[0].imag # doctest: +ELLIPSIS
        91.796...
        """
        if A is None:
            A = self.A(speed=speed, frequency=frequency, synchronous=synchronous)

        filter_eigenpairs = lambda values, vectors, indices: (
            values[indices],
            vectors[:, indices],
        )

        if synchronous:
            evalues, evectors = la.eig(A)

            idx = np.where(np.imag(evalues) != 0)[0]
            evalues, evectors = filter_eigenpairs(evalues, evectors, idx)
            idx = np.where(np.abs(np.real(evalues) / np.imag(evalues)) < 1000)[0]
            evalues, evectors = filter_eigenpairs(evalues, evectors, idx)
        else:
            if sparse:
                try:
                    evalues, evectors = las.eigs(
                        A,
                        k=min(2 * num_modes, max(num_modes, A.shape[0] - 2)),
                        sigma=1,
                        which="LM",
                        v0=np.ones(A.shape[0]),
                    )
                except las.ArpackError:
                    evalues, evectors = la.eig(A)
            else:
                evalues, evectors = la.eig(A)

        if sparse is not None and not synchronous:
            idx = np.where(np.abs(evalues) > 1e-1)[0]
            evalues, evectors = filter_eigenpairs(evalues, evectors, idx)

        if sorted_:
            idx = self._index(evalues)
            evalues, evectors = filter_eigenpairs(evalues, evectors, idx)

        return evalues, evectors

    def _lti(self, speed, frequency=None):
        """Continuous-time linear time invariant system.

        This method is used to create a Continuous-time linear
        time invariant system for the mdof system.
        From this system we can obtain poles, impulse response,
        generate a bode, etc.

        Parameters
        ----------
        speed: float
            Rotor speed.
        frequency: float, optional
            Excitation frequency.
            Default is rotor speed.

        Returns
        -------
        sys : StateSpaceContinuous
            Space State Continuos with A, B, C and D matrices

        Example
        -------
        >>> rotor = rotor_example()
        >>> A = rotor._lti(speed=0).A
        >>> B = rotor._lti(speed=0).B
        >>> C = rotor._lti(speed=0).C
        >>> D = rotor._lti(speed=0).D
        """
        Z = np.zeros((self.ndof, self.ndof))
        I = np.eye(self.ndof)

        # x' = Ax + Bu
        B2 = I
        if frequency is None:
            frequency = speed

        A = self.A(speed=speed, frequency=frequency)
        M = self.M(frequency)

        # fmt: off
        B = np.vstack([Z,
                       la.solve(M, B2)])
        # fmt: on

        # y = Cx + Du
        # Observation matrices
        Cd = I
        Cv = Z
        Ca = Z

        # fmt: off
        C = np.hstack((Cd - Ca @ la.solve(M, self.K(frequency)), Cv - Ca @ la.solve(M, self.C(frequency))))
        # fmt: on
        D = Ca @ la.solve(M, B2)

        sys = signal.lti(A, B, C, D)

        return sys

    def transfer_matrix(self, speed=None, frequency=None, modes=None):
        """Calculate the fer matrix for the frequency response function (FRF).

        Paramenters
        -----------
        frequency : float, optional
            Excitation frequency. Default is rotor speed.
        speed : float, optional
            Rotating speed. Default is rotor speed (frequency).

        Returns
        -------
        H : matrix
            System transfer matrix

        Example
        -------
        >>> rotor = rotor_example()
        >>> speed = 100.0
        >>> H = rotor.transfer_matrix(speed=speed)
        """
        if frequency is None:
            frequency = speed

        I = np.eye(self.M().shape[0])

        lu, piv = lu_factor(
            -(frequency**2) * self.M(frequency=frequency)
            + 1j * frequency * (self.C(frequency=frequency) + frequency * self.G())
            + self.K(frequency=frequency)
        )
        H = lu_solve((lu, piv), I)

        if np.isnan(H).any():
            H = np.zeros((H.shape))

        return H

    @check_units
    def run_freq_response(
        self,
        speed_range=None,
        modes=None,
        cluster_points=False,
        num_modes=12,
        num_points=10,
        rtol=0.005,
        free_free=False,
    ):
        """Frequency response for a mdof system.

        This method returns the frequency response for a mdof system given a range of
        frequencies and the modes that will be used.

        Available plotting methods:
            .plot()
            .plot_magnitude()
            .plot_phase()
            .plot_polar_bode()

        Parameters
        ----------
        speed_range : array, optional, pint.Quantity
            Array with the desired range of frequencies.
            Default is 0 to 1.5 x highest damped natural frequency.
        modes : list, optional
            Modes that will be used to calculate the frequency response
            (all modes will be used if a list is not given).
        cluster_points : bool, optional
            boolean to activate the automatic frequency spacing method. If True, the
            method uses _clustering_points() to create an speed_range.
            Default is False
        num_points : int, optional
            The number of points generated per critical speed.
            The method set the same number of points for slightly less and slightly
            higher than the natural circular frequency. It means there'll be num_points
            greater and num_points smaller than a given critical speed.
            num_points may be between 2 and 12. Anything above this range defaults
            to 10 and anything below this range defaults to 4.
            The default is 10.
        num_modes
            The number of eigenvalues and eigenvectors to be calculated using ARPACK.
            It also defines the range for the output array, since the method generates
            points only for the critical speed calculated by run_critical_speed().
            Default is 12.
        rtol : float, optional
            Tolerance (relative) for termination. Applied to scipy.optimize.newton to
            calculate the approximated critical speeds.
            Default is 0.005 (0.5%).
        free_free : bool, optional
            If True, the method will consider the rotor system as free-free.
            Default is False.

        Returns
        -------
        results : ross.FrequencyResponseResults
            For more information on attributes and methods available see:
            :py:class:`ross.FrequencyResponseResults`

        Examples
        --------
        >>> import ross as rs
        >>> rotor = rs.rotor_example()
        >>> speed =np.linspace(0, 1000, 101)
        >>> response = rotor.run_freq_response(speed_range=speed)

        Return the response amplitude
        >>> abs(response.freq_resp) # doctest: +ELLIPSIS
        array([[[0.00000000e+00, 1.00261725e-06, 1.01076952e-06, ...

        Return the response phase
        >>> np.angle(response.freq_resp) # doctest: +ELLIPSIS
        array([[[...

        Using clustered points option.
        Set `cluster_points=True` and choose how many modes the method must search and
        how many points to add just before and after each critical speed.

        >>> response = rotor.run_freq_response(cluster_points=True, num_points=5)
        >>> response.speed_range.shape
        (61,)

        Selecting the disirable modes, if you want a reduced model:
        >>> response = rotor.run_freq_response(speed_range=speed, modes=[0, 1, 2, 3, 4])
        >>> abs(response.freq_resp) # doctest: +ELLIPSIS
        array([[[0.00000000e+00, 1.00261725e-06, 1.01076952e-06, ...

        Plotting frequency response function:
        >>> fig = response.plot(inp=13, out=13)

        To plot velocity and acceleration responses, you must change amplitude_units
        from "[length]/[force]" units to "[speed]/[force]" or "[acceleration]/[force]"
        respectively

        Plotting velocity response
        >>> fig = response.plot(inp=13, out=13, amplitude_units="m/s/N")

        Plotting acceleration response
        >>> fig = response.plot(inp=13, out=13, amplitude_units="m/s**2/N")
        """

        if speed_range is not None:
            speed_range = tuple(speed_range)

        if modes is not None:
            modes = tuple(modes)

        return self._run_freq_response(
            speed_range=speed_range,
            modes=modes,
            cluster_points=cluster_points,
            num_modes=num_modes,
            num_points=num_points,
            rtol=rtol,
            free_free=free_free,
        )

    @lru_cache()
    def _run_freq_response(
        self,
        speed_range=None,
        modes=None,
        cluster_points=False,
        num_modes=12,
        num_points=10,
        rtol=0.005,
        free_free=False,
    ):
        """Frequency response for a mdof system.

        The `run_freq_response()` has been split into two separate methods. This change
        was made to convert `speed_range` and `modes` to a tuple format and to enable
        the use of the `@lru_cache()` method, which requires hashable arguments to cache
        results effectively.
        """

        if speed_range is None:
            if not cluster_points:
                modal = self.run_modal(0)
                speed_range = np.linspace(0, max(modal.evalues.imag) * 1.5, 1000)
            else:
                speed_range = self._clustering_points(
                    num_modes, num_points, modes, rtol
                )

        self._check_frequency_array(speed_range)

        freq_resp = np.empty((self.ndof, self.ndof, len(speed_range)), dtype=complex)
        velc_resp = np.empty((self.ndof, self.ndof, len(speed_range)), dtype=complex)
        accl_resp = np.empty((self.ndof, self.ndof, len(speed_range)), dtype=complex)

        if free_free:
            transfer_matrix = lambda s: self.transfer_matrix(speed=0, frequency=s)
        else:
            transfer_matrix = lambda s: self.transfer_matrix(speed=s)

        for i, speed in enumerate(speed_range):
            H = transfer_matrix(speed)
            freq_resp[..., i] = H
            velc_resp[..., i] = 1j * speed * H
            accl_resp[..., i] = -(speed**2) * H

        results = FrequencyResponseResults(
            freq_resp=freq_resp,
            velc_resp=velc_resp,
            accl_resp=accl_resp,
            speed_range=np.array(speed_range),
            number_dof=self.number_dof,
        )

        return results

    def run_amb_sensitivity(
        self,
        speed,
        t_max,
        dt,
        disturbance_amplitude=10e-6,
        disturbance_min_frequency=0.001,
        disturbance_max_frequency=150,
        amb_tags=None,
        sensors_theta=0.7853981633974483,
        verbose=1,
    ):
        """Run Active Magnetic Bearing (AMB) sensitivity analysis.

        This method performs a frequency-domain sensitivity analysis of the rotor system
        equipped with active magnetic bearings (AMBs). The analysis uses a logarithmic
        chirp excitation applied as an external disturbance force to compute the system's
        frequency response at the AMB-controlled degrees of freedom (DoFs). The results
        provide magnitude and phase sensitivity functions for each AMB in both x and y
        directions.

        Parameters
        ----------
        speed : float
            Rotational speed of the rotor in rad/s.
        t_max : float
            Total time duration of the simulation in seconds.
        dt : float
            Time step for the simulation in seconds.
        disturbance_amplitude : float, optional
            Amplitude of the excitation chirp signal applied as a disturbance.
            Default is 10e-6.
        disturbance_min_frequency : float, optional
            Minimum frequency (in Hz) of the logarithmic chirp signal used for excitation.
            The chirp sweeps from this frequency up to `disturbance_max_frequency`.
            Default is 1e-3 Hz.
        disturbance_max_frequency : float, optional
            Maximum frequency (in Hz) of the logarithmic chirp signal used for excitation.
            Default is 150 Hz.
        amb_tags : list of str, optional
            List of magnetic bearing tags to include in the sensitivity analysis.
            If None or empty, all `MagneticBearingElement` instances in the rotor are used.
            If provided, only the AMBs matching the specified tags will be analyzed.
            Raises a RuntimeError if no AMB with the given tag is found.
        sensors_theta : float, optional
            Angular position of the Active Magnetic Bearing (AMB) sensors, in radians.
            This angle defines the orientation of the sensor coordinate system (v, w)
            relative to the global coordinate system (x, y). A positive angle
            corresponds to a counter-clockwise rotation. Default is 45 degrees (π/4 rad).
        verbose : int, optional
            Controls the verbosity of the method. If `1` or greater, both the simulation
            time and the forces produced by the AMBs are presented. If `0`, no output is
            shown. Default is `1`.

        Returns
        -------
        results : SensitivityResults
            Object containing sensitivity magnitude, phase, and frequency vectors
            for each magnetic bearing tag and direction ('x', 'y'). Also includes
            the excitation, disturbed, and sensor signals used in the computation.

        Notes
        -----
        - The excitation is a logarithmic chirp sweeping from `disturbance_min_frequency`
          to `disturbance_max_frequency` (Hz).
        - The excitation is applied individually to each DoF controlled by an AMB.
        - The method assumes that the rotor contains `MagneticBearingElement` instances.
        - A Newmark time integration scheme is used internally via `run_time_response()`.

        Examples
        --------
        >>> import ross as rs
        >>> rotor = rs.rotor_amb_example()

        >>> # Run sensitivity for all magnetic bearings in the rotor (default sweep)
        >>> sensitivity_results = rotor.run_amb_sensitivity(speed=314.16, t_max=5e-4, dt=1e-4) # doctest: +ELLIPSIS
        Running direct method...

        >>> # Run sensitivity only for a specific AMB tag (e.g., "Magnetic Bearing 0")
        >>> sensitivity_results = rotor.run_amb_sensitivity(
        ...     speed=314.16, t_max=5e-4, dt=1e-4, amb_tags=["Magnetic Bearing 0"]
        ... ) # doctest: +ELLIPSIS
        Running direct method...

        >>> # Run sensitivity with a custom chirp band (0.1 Hz to 200 Hz)
        >>> sensitivity_results = rotor.run_amb_sensitivity(
        ...     speed=314.16, t_max=5e-4, dt=1e-4,
        ...     disturbance_min_frequency=0.1, disturbance_max_frequency=200.0
        ... ) # doctest: +ELLIPSIS
        Running direct method...

        >>> # Accessing maximum absolute sensitivities for "Magnetic Bearing 0"
        >>> max_sens_bearing_0_x = sensitivity_results.max_abs_sensitivities["Magnetic Bearing 0"]["x"]
        >>> max_sens_bearing_0_y = sensitivity_results.max_abs_sensitivities["Magnetic Bearing 0"]["y"]

        >>> # Plotting the sensitivities for all AMBs and axes
        >>> fig = sensitivity_results.plot(
        ...     frequency_units="Hz", phase_unit="degree",
        ...     magnitude_scale="decibel", xaxis_type="log"
        ... )

        >>> # Plotting the time results used in sensitivity calculation
        >>> fig = sensitivity_results.plot_time_results()
        """

        if amb_tags is not None and not isinstance(amb_tags, list):
            raise ValueError("`amb_tags` must be a list of strings.")

        t = np.arange(0, t_max, dt)
        f = np.zeros((len(t), self.ndof))

        all_magnetic_bearings = [
            brg
            for brg in self.bearing_elements
            if isinstance(brg, MagneticBearingElement)
        ]

        if amb_tags is not None and len(amb_tags) > 0:
            magnetic_bearings = [
                amb for amb in all_magnetic_bearings if amb.tag in amb_tags
            ]
            if len(magnetic_bearings) == 0:
                raise RuntimeError("No Magnetic Bearing with the given tag was found.")
        else:
            magnetic_bearings = all_magnetic_bearings

        sensitivity_compute_dofs = {
            magnetic_bearing.tag: {
                "x": self.number_dof * magnetic_bearing.n,
                "y": self.number_dof * magnetic_bearing.n + 1,
            }
            for magnetic_bearing in magnetic_bearings
        }

        sensitivity_data = {
            magnetic_bearing.tag: {
                "x": {},
                "y": {},
            }
            for magnetic_bearing in magnetic_bearings
        }

        chirp_signal = disturbance_amplitude * chirp(
            t,
            f0=disturbance_min_frequency,  # frequência no instante t = 0
            f1=disturbance_max_frequency,  # frequência no instante t = t_f
            t1=float(t[-1]),  # instante final
            method="logarithmic",
            phi=-90,
        )

        progress_interval = t_max / 25 if verbose >= 1 else 2 * t_max

        for amb_tag in sensitivity_compute_dofs.keys():
            for axis in sensitivity_compute_dofs[amb_tag].keys():
                sensitivity_result_values = {}
                self.run_time_response(
                    speed,
                    f,
                    t,
                    progress_interval=progress_interval,
                    method="newmark",
                    sensitivity_disturbance=chirp_signal,
                    sensitivity_result_values=sensitivity_result_values,
                    sensitivity_compute_dof=sensitivity_compute_dofs[amb_tag][axis],
                    sensors_theta=sensors_theta,
                )
                sensitivity_data[amb_tag][axis] = dict(sensitivity_result_values)

        results = SensitivityResults(
            sensitivity_data=sensitivity_data,
            sensitivity_compute_dofs=sensitivity_compute_dofs,
            number_dof=self.number_dof,
            t=t,
        )

        return results

    @check_units
    def run_forced_response(
        self,
        force=None,
        speed_range=None,
        modes=None,
        cluster_points=False,
        num_modes=12,
        num_points=10,
        rtol=0.005,
        unbalance=None,
    ):
        """Forced response for a mdof system.

        This method returns the unbalanced response for a mdof system
        given magnitude and phase of the unbalance, the node where it's
        applied and a frequency range.

        Available plotting methods:
            .plot()
            .plot_magnitude()
            .plot_phase()
            .plot_polar_bode()
            .plot_deflected_shape()
            .plot_bending_moment()
            .plot_deflected_shape_3d()
            .plot_deflected_shape_2d()

        Parameters
        ----------
        force : list, array, pint.Quantity
            Unbalance force in each degree of freedom for each value in omega
        speed_range : list, array, pint.Quantity
            Array with the desired range of frequencies
        modes : list, optional
            Modes that will be used to calculate the frequency response
            (all modes will be used if a list is not given).
        unbalance : array, optional
            Array with the unbalance data (node, magnitude and phase) to be plotted
            with deflected shape. This argument is set only if running an unbalance
            response analysis.
            Default is None.
        cluster_points : bool, optional
            boolean to activate the automatic frequency spacing method. If True, the
            method uses _clustering_points() to create an speed_range.
            Default is False
        num_points : int, optional
            The number of points generated per critical speed.
            The method set the same number of points for slightly less and slightly
            higher than the natural circular frequency. It means there'll be num_points
            greater and num_points smaller than a given critical speed.
            num_points may be between 2 and 12. Anything above this range defaults
            to 10 and anything below this range defaults to 4.
            The default is 10.
        num_modes
            The number of eigenvalues and eigenvectors to be calculated using ARPACK.
            It also defines the range for the output array, since the method generates
            points only for the critical speed calculated by run_critical_speed().
            Default is 12.
        rtol : float, optional
            Tolerance (relative) for termination. Applied to scipy.optimize.newton to
            calculate the approximated critical speeds.
            Default is 0.005 (0.5%).

        Returns
        -------
        results : ross.ForcedResponseResults
            For more information on attributes and methods available see:
            :py:class:`ross.ForcedResponseResults`

        Examples
        --------
        >>> rotor = rotor_example()
        >>> speed = np.linspace(0, 1000, 101)
        >>> force = rotor._unbalance_force(3, 10.0, 0.0, speed)
        >>> resp = rotor.run_forced_response(force=force, speed_range=speed)
        >>> abs(resp.forced_resp) # doctest: +ELLIPSIS
        array([[0.00000000e+00, 5.06073311e-04, 2.10044826e-03, ...

        Using clustered points option.
        Set `cluster_points=True` and choose how many modes the method must search and
        how many points to add just before and after each critical speed.

        >>> response = rotor.run_forced_response(
        ...     force=force, cluster_points=True, num_modes=12, num_points=5
        ... )
        >>> response.speed_range.shape
        (61,)
        """
        if speed_range is None:
            if cluster_points:
                speed_range = self._clustering_points(
                    num_modes, num_points, modes, rtol
                )

        freq_resp = self.run_freq_response(
            speed_range, modes, cluster_points, num_modes, num_points, rtol
        )

        forced_resp = np.zeros((self.ndof, len(freq_resp.speed_range)), dtype=complex)
        velc_resp = np.zeros((self.ndof, len(freq_resp.speed_range)), dtype=complex)
        accl_resp = np.zeros((self.ndof, len(freq_resp.speed_range)), dtype=complex)

        for i in range(len(freq_resp.speed_range)):
            forced_resp[:, i] = freq_resp.freq_resp[..., i] @ force[..., i]
            velc_resp[:, i] = freq_resp.velc_resp[..., i] @ force[..., i]
            accl_resp[:, i] = freq_resp.accl_resp[..., i] @ force[..., i]
        forced_resp = ForcedResponseResults(
            rotor=self,
            forced_resp=forced_resp,
            velc_resp=velc_resp,
            accl_resp=accl_resp,
            speed_range=speed_range,
            unbalance=unbalance,
        )

        return forced_resp

    def _unbalance_force(self, node, magnitude, phase, omega):
        """Calculate unbalance forces.

        This is an auxiliary function the calculate unbalance forces. It takes the
        force magnitude and phase and generate an array with complex values of forces
        on each degree degree of freedom of the given node.

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

        Examples
        --------
        >>> rotor = rotor_example()
        >>> speed = np.linspace(0, 1000, 101)
        >>> rotor._unbalance_force(3, 10.0, 0.0, speed)[18] # doctest: +ELLIPSIS
        array([0.000e+00+0.j, 1.000e+03+0.j, 4.000e+03+0.j, ...
        """
        F0 = np.zeros((self.ndof, len(omega)), dtype=np.complex128)

        b0 = np.zeros((self.number_dof), dtype=np.complex128)
        b0[0] = magnitude * np.exp(1j * phase)
        b0[1] = -1j * magnitude * np.exp(1j * phase)
        # b0[2] 1j*(Id - Ip)*beta*np.exp(1j*gamma)

        n0 = self.number_dof * node
        n1 = n0 + self.number_dof
        for i, w in enumerate(omega):
            F0[n0:n1, i] += w**2 * b0

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

        Examples
        --------
        >>> rotor = rotor_example()
        >>> t = np.linspace(0, 10, 31)
        >>> omega = np.linspace(0, 1000, 31)
        >>> F = rotor.unbalance_force_over_time([3], [10.0], [0.0], omega, t)
        >>> F[18, :3]
        array([     0.        ,   7632.15353293, -43492.18127561])
        """

        if not isinstance(omega, Iterable):
            omega = np.full_like(t, omega)

        theta = integrate(omega, t, initial=0)
        alpha = np.gradient(omega, t)

        F0 = np.zeros((self.ndof, len(t)))

        for i, n in enumerate(node):
            phi = phase[i] + theta

            Fx = magnitude[i] * ((omega**2) * np.cos(phi) + alpha * np.sin(phi))
            Fy = magnitude[i] * ((omega**2) * np.sin(phi) - alpha * np.cos(phi))

            F0[n * self.number_dof + 0, :] += Fx
            F0[n * self.number_dof + 1, :] += Fy

        if return_all:
            return F0, theta, omega, alpha
        else:
            return F0

    @check_units
    def run_unbalance_response(
        self,
        node,
        unbalance_magnitude,
        unbalance_phase,
        frequency=None,
        modes=None,
        cluster_points=False,
        num_modes=12,
        num_points=10,
        rtol=0.005,
    ):
        """Unbalanced response for a mdof system.

        This method returns the unbalanced response for a mdof system
        given magnitide and phase of the unbalance, the node where it's
        applied and a frequency range.

        Available plotting methods:
            .plot()
            .plot_magnitude()
            .plot_phase()
            .plot_polar_bode()
            .plot_deflected_shape()
            .plot_bending_moment()
            .plot_deflected_shape_3d()
            .plot_deflected_shape_2d()

        Parameters
        ----------
        node : list, int
            Node where the unbalance is applied.
        unbalance_magnitude : list, float, pint.Quantity
            Unbalance magnitude (kg.m).
        unbalance_phase : list, float, pint.Quantity
            Unbalance phase (rad).
        frequency : list, pint.Quantity
            List with the desired range of frequencies (rad/s).
        modes : list, optional
            Modes that will be used to calculate the frequency response
            (all modes will be used if a list is not given).
        cluster_points : bool, optional
            boolean to activate the automatic frequency spacing method. If True, the
            method uses _clustering_points() to create an speed_range.
            Default is False
        num_points : int, optional
            The number of points generated per critical speed.
            The method set the same number of points for slightly less and slightly
            higher than the natural circular frequency. It means there'll be num_points
            greater and num_points smaller than a given critical speed.
            num_points may be between 2 and 12. Anything above this range defaults
            to 10 and anything below this range defaults to 4.
            The default is 10.
        num_modes
            The number of eigenvalues and eigenvectors to be calculated using ARPACK.
            It also defines the range for the output array, since the method generates
            points only for the critical speed calculated by run_critical_speed().
            Default is 12.
        rtol : float, optional
            Tolerance (relative) for termination. Applied to scipy.optimize.newton to
            calculate the approximated critical speeds.
            Default is 0.005 (0.5%).

        Returns
        -------
        results : ross.ForcedResponseResults
            For more information on attributes and methods available see:
            :py:class:`ross.ForcedResponseResults`

        Examples
        --------
        >>> import ross as rs
        >>> rotor = rs.rotor_example()
        >>> speed = np.linspace(0, 1000, 101)
        >>> response = rotor.run_unbalance_response(node=3,
        ...                                         unbalance_magnitude=10.0,
        ...                                         unbalance_phase=0.0,
        ...                                         frequency=speed)

        Return the response amplitude
        >>> abs(response.forced_resp) # doctest: +ELLIPSIS
        array([[0.00000000e+00, 5.06073311e-04, 2.10044826e-03, ...

        Return the response phase
        >>> np.angle(response.forced_resp) # doctest: +ELLIPSIS
        array([[ 0.        ,  0.        ,  0.        , ...

        Using clustered points option.
        Set `cluster_points=True` and choose how many modes the method must search and
        how many points to add just before and after each critical speed.

        >>> response2 = rotor.run_unbalance_response(
        ...     node=3, unbalance_magnitude=0.01, unbalance_phase=0.0, cluster_points=True, num_points=5
        ... )
        >>> response2.speed_range.shape
        (61,)

        plot unbalance response:
        >>> probe_node = 3
        >>> probe_angle = np.pi / 2
        >>> probe_tag = "my_probe"  # optional
        >>> fig = response.plot(probe=[rs.Probe(probe_node, probe_angle, tag=probe_tag)])

        plot response for major or minor axis:
        >>> probe_node = 3
        >>> probe_angle = "major"   # for major axis
        >>> # probe_angle = "minor" # for minor axis
        >>> probe_tag = "my_probe"  # optional
        >>> fig = response.plot(probe=[rs.Probe(probe_node, probe_angle, tag=probe_tag)])

        To plot velocity and acceleration responses, you must change amplitude_units
        from "[length]" units to "[length]/[time]" or "[length]/[time] ** 2" respectively
        Plotting velocity response
        >>> fig = response.plot(
        ...     probe=[rs.Probe(probe_node, probe_angle)],
        ...     amplitude_units="m/s"
        ... )

        Plotting acceleration response
        >>> fig = response.plot(
        ...     probe=[rs.Probe(probe_node, probe_angle)],
        ...     amplitude_units="m/s**2"
        ... )

        Plotting deflected shape configuration
        Speed value must be in speed_range.
        >>> value = 600
        >>> fig = response.plot_deflected_shape(speed=value)
        """
        if frequency is None:
            if cluster_points:
                frequency = self._clustering_points(num_modes, num_points, modes, rtol)

        force = np.zeros((self.ndof, len(frequency)), dtype=complex)

        try:
            for n, m, p in zip(node, unbalance_magnitude, unbalance_phase):
                force += self._unbalance_force(n, m, p, frequency)
        except TypeError:
            force = self._unbalance_force(
                node, unbalance_magnitude, unbalance_phase, frequency
            )

        # fmt: off
        ub = np.vstack((node, unbalance_magnitude, unbalance_phase))
        forced_response = self.run_forced_response(
            force, frequency, modes, cluster_points, num_modes, num_points, rtol, ub
        )
        # fmt: on

        return forced_response

    def magnetic_bearing_controller(
        self, step, magnetic_bearings, time_step, disp_resp, **kwargs
    ):
        """Compute control forces for Active Magnetic Bearings (AMBs).

        This method calculates the magnetic control forces generated by active
        magnetic bearings (AMBs) at each time step using a PID control law. The
        forces are based on the measured displacements and can optionally include
        external disturbances for sensitivity analysis.

        If sensitivity analysis is enabled via keyword arguments, the method injects
        a known disturbance at a specific DoF and logs excitation, disturbed, and
        sensor signals for post-processing.

        Parameters
        ----------
        step : int
            Current time step index in the simulation.
        magnetic_bearings : list
            List of `MagneticBearingElement` objects used for force computation.
        time_step : float
            Time increment used in the numerical integration scheme (in seconds).
        disp_resp : ndarray
            Displacement response vector of the rotor at the current time step.
            The size must match the number of rotor DoFs.

        Other Parameters
        ----------------
        sensitivity_compute_dof : int, optional
            Index of the DoF where a disturbance signal is applied (for sensitivity analysis).
        sensitivity_disturbance : ndarray, optional
            Disturbance signal array (e.g., chirp) to be injected at the specified DoF.
        sensitivity_result_values : dict, optional
            Dictionary to store the time history of:
                - "excitation_signal"
                - "disturbed_signal"
                - "sensor_signal"
            for post-processing in sensitivity computations.

        Returns
        -------
        magnetic_force : ndarray
            Force vector containing control forces applied by each magnetic bearing
            in the rotor system. Has the same length as `self.ndof`.

        Notes
        -----
        - The control forces are applied in both x and y directions at each AMB location.
        - The actual PID computation is delegated to the `compute_pid_amb` function.
        - If `sensitivity_compute_dof` is provided, the excitation is applied to that DoF only.

        Examples
        --------
        >>> import ross as rs
        >>> import numpy as np
        >>> rotor = rs.rotor_amb_example()
        >>> dt, speed, step = 1e-4, 1000, 1
        >>> t = np.arange(0, 5 * dt, dt)
        >>> node = [27, 29]
        >>> mass = [10, 10]
        >>> F = np.zeros((len(t), rotor.ndof))
        >>> for n, m in zip(node,mass):
        ...     F[:, 6 * n + 0] = m * np.cos((speed * t))
        ...     F[:, 6 * n + 1] = (m-5) * np.sin((speed * t))
        >>> response = rotor.run_time_response(speed, F, t, method = "newmark")
        Running direct method
        >>> magnetic_bearings = [brg for brg in rotor.bearing_elements if isinstance(brg, rs.bearing_seal_element.MagneticBearingElement)]
        >>> magnetic_force = rotor.magnetic_bearing_controller(step, magnetic_bearings, dt, response.yout[-1,:])
        >>> np.nonzero(magnetic_force)[0]
        array([ 72,  73, 258, 259])
        >>> magnetic_force[np.nonzero(magnetic_force)[0]]
        array([-7.24276404e-04, -1.42153354e-05, -1.17641699e-04,  2.39844354e-05])
        """

        if kwargs.get("sensitivity_result_values", None) == {}:
            kwargs["sensitivity_result_values"].update(
                {"excitation_signal": [], "disturbed_signal": [], "sensor_signal": []}
            )

        sensitivity_compute_dof: None | int = kwargs.get(
            "sensitivity_compute_dof", None
        )
        sensitivity_disturbance: None | np.ndarray = kwargs.get(
            "sensitivity_disturbance", None
        )
        sensors_theta: None | float = kwargs.get("sensors_theta", np.deg2rad(45))
        progress_interval: None | float = kwargs.get("progress_interval", None)

        current_offset = 0
        setpoint = 0
        dt = time_step
        magnetic_force = np.zeros(self.ndof)

        for elm in magnetic_bearings:
            x_dof = self.number_dof * elm.n
            y_dof = self.number_dof * elm.n + 1

            x_disp = disp_resp[x_dof]
            y_disp = disp_resp[y_dof]

            # Transforming the displacements to the sensor reference frame
            v_disp = x_disp * np.cos(sensors_theta) + y_disp * np.sin(sensors_theta)
            w_disp = -x_disp * np.sin(sensors_theta) + y_disp * np.cos(sensors_theta)

            if sensitivity_compute_dof is not None and sensitivity_compute_dof in [
                x_dof,
                y_dof,
            ]:
                sensor_signal = v_disp if x_dof == sensitivity_compute_dof else w_disp

                excitation_signal = sensitivity_disturbance[step]
                v_disp = (
                    v_disp + excitation_signal
                    if x_dof == sensitivity_compute_dof
                    else v_disp
                )
                w_disp = (
                    w_disp + excitation_signal
                    if y_dof == sensitivity_compute_dof
                    else w_disp
                )

                disturbed_signal = (
                    v_disp if x_dof == sensitivity_compute_dof else w_disp
                )

                if "sensitivity_result_values" in kwargs.keys():
                    kwargs["sensitivity_result_values"]["excitation_signal"].append(
                        excitation_signal
                    )
                    kwargs["sensitivity_result_values"]["disturbed_signal"].append(
                        disturbed_signal
                    )
                    kwargs["sensitivity_result_values"]["sensor_signal"].append(
                        sensor_signal
                    )

            # The method compute_pid_amb updates the magnetic_force array internally
            magnetic_force_v = elm.compute_pid_amb(
                dt,
                current_offset=current_offset,
                setpoint=setpoint,
                disp=v_disp,
                dof_index=0,
            )

            magnetic_force_w = elm.compute_pid_amb(
                dt,
                current_offset=current_offset,
                setpoint=setpoint,
                disp=w_disp,
                dof_index=1,
            )

            magnetic_force_x = magnetic_force_v * np.cos(
                sensors_theta
            ) - magnetic_force_w * np.sin(sensors_theta)
            magnetic_force_y = magnetic_force_v * np.sin(
                sensors_theta
            ) + magnetic_force_w * np.cos(sensors_theta)

            elm.magnetic_force_xy[-1][0].append(magnetic_force_x)
            elm.magnetic_force_xy[-1][1].append(magnetic_force_y)
            elm.magnetic_force_vw[-1][0].append(magnetic_force_v)
            elm.magnetic_force_vw[-1][1].append(magnetic_force_w)

            magnetic_force[x_dof] = magnetic_force_x
            magnetic_force[y_dof] = magnetic_force_y

            if progress_interval is not None:
                time_progress_ratio = round((step * dt) / progress_interval, 8)
                if time_progress_ratio.is_integer():
                    print(
                        f"Force x / y (N): {magnetic_force_x:.6f} / {magnetic_force_y:.6f} ({elm.tag})"
                    )

        return magnetic_force

    def gravitational_force(self, g=-9.8065, direction="y", M=None, num_dof=None):
        """Compute the gravitational force vector for the system.

        Parameters
        ----------
        g : float, optional
            Acceleration due to gravity. Default is -9.8065 m/s².
        direction : {"x", "y", "z"}, optional
            Direction in which gravity acts. Default is "y".
        M : ndarray, optional
            Mass matrix of the system. If None, the internal mass matrix is used.
        num_dof : int, optional
            Number of degrees of freedom per node. If None, the internal value is used.

        Returns
        -------
        force : ndarray
            Gravitational force (weight) vector of shape `(ndof,)`.

        Examples
        --------
        >>> rotor = compressor_example()
        >>> force = rotor.gravitational_force()
        >>> force[:4]
        array([ 0.        , -3.12941854,  0.        ,  0.01851573])
        """
        idx = {"x": 0, "y": 1, "z": 2}

        if M is None:
            M = self.M()
            num_dof = self.number_dof

        gravity = np.zeros(len(M))
        gravity[idx[direction] :: num_dof] = g

        return M @ gravity

    def integrate_system(self, speed, F, t, **kwargs):
        """Time integration for a rotor system.

        This method returns the time response for a rotor given a force, time and
        speed based on time integration with the Newmark method.

        Parameters
        ----------
        speed : float or array_like
            Rotor speed.
        F : ndarray
            Force array (needs to have the same length as time array).
        t : ndarray
            Time array.
        **kwargs : optional
            Additional keyword arguments can be passed to define the parameters
            of the Newmark method if it is used (e.g. `gamma`, `beta`, `tol`, ...).
            See `newmark` for more details. Other optional arguments are listed
            below.
        model_reduction : dict, optional
            When `model_reduction` is provided, the corresponding reduction method is initialized.
            Dict keys:
                method : str, optional
                    Reduction method to use, e.g., "guyan" or "pseudomodal".
                    Defaults to "guyan".
                num_modes : int, optional
                    Number of modes to reduce the model to, if pseudo-modal method is considered.
                include_nodes : list of int, optional
                    List of the nodes to be included, if Guyan reduction method is considered.
                dof_mapping : list of str, optional
                    List of the local DOFs to be considered when using Guyan reduction method.
                    Valid values are: 'x', 'y', 'z', 'alpha', 'beta', 'theta', corresponding to:
                        - 'x' and 'y': lateral translations
                        - 'z': axial translation
                        - 'alpha': rotation about the x-axis
                        - 'beta': rotation about the y-axis
                        - 'theta': torsional rotation (about the z-axis)
                    Default is ['x', 'y'].
                include_dofs (list of int, optional):
                    Additional degrees of freedom (DOFs) to include in the reduction, such as DOFs
                    with applied forces or probe locations when using Guyan reduction method.
        add_to_RHS : callable, optional
            An optional function that computes and returns an additional array to be added to
            the right-hand side of the equation of motion. This function should take the time
            step number as argument, and can take optional arguments corresponding to the current
            state of the rotor system, including the displacements `disp_resp`, velocities
            `velc_resp`, and acceleration `accl_resp`. It should return an array of the same
            length as the degrees of freedom of the rotor system `rotor.ndof`. This function
            allows for the incorporation of supplementary terms or external effects in the rotor
            system dynamics beyond the specified force input during the time integration process.

        Returns
        -------
        t : ndarray
            Time values for the output.
        yout : ndarray
            System response.

        Examples
        --------
        >>> import ross as rs
        >>> rotor = rs.compressor_example()
        >>> size = 10000
        >>> node = 3
        >>> speed = 500.0
        >>> accel = 0.0
        >>> t = np.linspace(0, 10, size)
        >>> F = np.zeros((size, rotor.ndof))
        >>> F[:, rotor.number_dof * node + 0] = 10 * np.cos(2 * t)
        >>> F[:, rotor.number_dof * node + 1] = 10 * np.sin(2 * t)
        >>> t, yout = rotor.integrate_system(speed, F, t)
        Running direct method
        >>> yout[:, rotor.number_dof * node + 1] # doctest: +ELLIPSIS
        array([0.00000000e+00, 2.07239823e-10, 7.80952429e-10, ...,
               1.21848307e-07, 1.21957287e-07, 1.22065778e-07])
        """

        # Check if speed is array
        speed_is_array = isinstance(speed, Iterable)
        speed_ref = np.mean(speed) if speed_is_array else speed

        # Check if the model reduction has to be applied
        model_reduction = kwargs.get("model_reduction")
        if model_reduction:
            num_modes = model_reduction.get("num_modes")
            method = model_reduction.get("method", "guyan")

            if num_modes or method == "pseudomodal":
                method = "pseudomodal"
            else:
                force_dofs = list(set(np.where(F != 0)[1]))
                add_dofs = list(model_reduction.get("include_dofs", []))
                model_reduction["include_dofs"] = force_dofs + add_dofs

            model_reduction["method"] = method

            print(f"Running with model reduction: {method}")
            mr = ModelReduction(rotor=self, speed=speed_ref, **model_reduction)
            reduction = [mr.reduce_matrix, mr.reduce_vector, mr.revert_vector]

            kwargs.pop("model_reduction")

        else:
            print("Running direct method")
            return_array = lambda array: array
            reduction = [return_array for j in range(3)]

        # Assemble matrices
        M = reduction[0](kwargs.get("M", self.M()))
        C2 = reduction[0](kwargs.get("G", self.G()))
        K2 = reduction[0](kwargs.get("Ksdt", self.Ksdt()))
        F = reduction[1](F.T).T

        # Check if there is any magnetic bearing
        rotor, magnetic_force = self.init_ambs_for_integrate(**kwargs)

        # Consider any additional RHS function (extra forces)
        add_to_RHS = kwargs.get("add_to_RHS")

        if add_to_RHS is None:
            forces = lambda step, **curr_state: F[step, :] + reduction[1](
                magnetic_force(
                    step,
                    curr_state.get("dt"),
                    reduction[2](curr_state.get("y")),
                )
            )
        else:
            forces = lambda step, **curr_state: F[step, :] + reduction[1](
                add_to_RHS(
                    step,
                    time_step=curr_state.get("dt"),
                    disp_resp=reduction[2](curr_state.get("y")),
                    velc_resp=reduction[2](curr_state.get("ydot")),
                    accl_resp=reduction[2](curr_state.get("y2dot")),
                )
                + magnetic_force(
                    step,
                    curr_state.get("dt"),
                    reduction[2](curr_state.get("y")),
                )
            )

        # Depending on the conditions of the analysis,
        # one of the three options below will be chosen.
        if speed_is_array:
            accel = np.gradient(speed, t)

            brgs_with_var_coeffs = tuple(
                brg for brg in self.bearing_elements if brg.frequency is not None
            )

            if len(brgs_with_var_coeffs):  # Option 1
                if kwargs.get("C") or kwargs.get("K"):
                    raise Warning(
                        "The bearing coefficients vary with speed. Therefore, C and K matrices are not being replaced by the matrices defined as input arguments."
                    )

                def rotor_system(step, **current_state):
                    C1 = reduction[0](rotor.C(speed[step]))
                    K1 = reduction[0](rotor.K(speed[step]))

                    return (
                        M,
                        C1 + C2 * speed[step],
                        K1 + K2 * accel[step],
                        forces(step, **current_state),
                    )

            else:  # Option 2
                C1 = reduction[0](kwargs.get("C", rotor.C(speed_ref)))
                K1 = reduction[0](kwargs.get("K", rotor.K(speed_ref)))

                rotor_system = lambda step, **current_state: (
                    M,
                    C1 + C2 * speed[step],
                    K1 + K2 * accel[step],
                    forces(step, **current_state),
                )

        else:  # Option 3
            C1 = reduction[0](kwargs.get("C", rotor.C(speed_ref)))
            K1 = reduction[0](kwargs.get("K", rotor.K(speed_ref)))

            rotor_system = lambda step, **current_state: (
                M,
                C1 + C2 * speed_ref,
                K1,
                forces(step, **current_state),
            )

        size = len(M)
        response = newmark(rotor_system, t, size, **kwargs)
        yout = reduction[2](response.T).T
        return t, yout

    def init_ambs_for_integrate(self, **kwargs):
        magnetic_bearings = [
            brg
            for brg in self.bearing_elements
            if isinstance(brg, MagneticBearingElement)
        ]
        rotor = deepcopy(self)
        if len(magnetic_bearings):
            magnetic_force = (
                lambda step, time_step, disp_resp: self.magnetic_bearing_controller(
                    step, magnetic_bearings, time_step, disp_resp, **kwargs
                )
            )

            # Initialize storage attributes for magnetic bearings
            for brg in magnetic_bearings:
                brg.magnetic_force_xy.append([[], []])
                brg.magnetic_force_vw.append([[], []])
                brg.control_signal.append([[], []])
                brg.integral = [0, 0]
                brg.e0 = [0, 0]

            rotor.bearing_elements = [
                brg for brg in rotor.bearing_elements if brg not in magnetic_bearings
            ]

        else:
            magnetic_force = lambda step, time_step, disp_resp: np.zeros(self.ndof)

        return rotor, magnetic_force

    def time_response(self, speed, F, t, ic=None, method="default", **kwargs):
        """Time response for a rotor.

        This method returns the time response for a rotor
        given a force, time and initial conditions.

        Parameters
        ----------
        speed : float or array_like
            Rotor speed. Automatically, the Newmark method is chosen if `speed`
            has an array_like type.
        F : array
            Force array (needs to have the same length as time array).
        t : array
            Time array. (must have the same length than lti.B matrix)
        ic : array, optional
            The initial conditions on the state vector (zero by default).
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
        t : array
            Time values for the output.
        yout : array
            System response.
        xout : array
            Time evolution of the state vector.

        Examples
        --------
        >>> rotor = rotor_example()
        >>> speed = 0
        >>> size = 28
        >>> t = np.linspace(0, 5, size)
        >>> F = np.ones((size, rotor.ndof))
        >>> rotor.time_response(speed, F, t) # doctest: +ELLIPSIS
        (array([0.        , 0.18518519, 0.37037037, ...
        """

        if isinstance(speed, Iterable) or method.lower() == "newmark":
            t_, yout = self.integrate_system(speed, F, t, **kwargs)
            return t_, yout, []

        else:
            lti = self._lti(speed)
            return signal.lsim(lti, F, t, X0=ic)

    def plot_rotor(self, nodes=1, check_sld=False, length_units="m", **kwargs):
        """Plot a rotor object.

        This function will take a rotor object and plot its elements representation
        using Plotly.

        Parameters
        ----------
        nodes : int, optional
            Increment that will be used to plot nodes label.
        check_sld : bool
            If True, checks the slenderness ratio for each element.
            The shaft elements which has a slenderness ratio < 1.6 will be displayed in
            yellow color.
        length_units : str, optional
            length units to length and diameter.
            Default is 'm'.
        kwargs : optional
            Additional key word arguments can be passed to change the plot layout only
            (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        fig : plotly.graph_objects.Figure
            The figure object with the rotor representation.

        Example
        -------
        >>> import ross as rs
        >>> rotor = rs.rotor_example()
        >>> figure = rotor.plot_rotor()
        """
        SR = [
            shaft.slenderness_ratio
            for shaft in self.shaft_elements
            if shaft.slenderness_ratio < 1.6
        ]

        if check_sld:
            if len(SR):
                warnings.warn(
                    "The beam elements "
                    + str(SR)
                    + " have slenderness ratio (G*A*L^2 / EI) of less than 1.6."
                    + " Results may not converge correctly"
                )

        nodes_pos = Q_(self.nodes_pos, "m").to(length_units).m
        nodes_o_d = Q_(self.nodes_o_d, "m").to(length_units).m
        center_line_pos = Q_(self.center_line_pos, "m").to(length_units).m

        fig = go.Figure()

        # plot shaft centerline
        fig.add_shape(
            x0=0,
            x1=1,
            y0=0,
            y1=0,
            xref="paper",
            yref="y",
            layer="below",
            opacity=0.7,
            type="line",
            line=dict(width=3.0, color="black", dash="dashdot"),
        )

        # plot nodes icons
        text = []
        x_pos = []
        y_pos = np.linspace(0, 0, len(nodes_pos[::nodes]))
        for i, position in enumerate(nodes_pos[::nodes]):
            node = self.nodes[i]
            text.append("{}".format(node * nodes))
            x_pos.append(position)
            y_pos[i] = center_line_pos[i]

        fig.add_trace(
            go.Scatter(
                x=x_pos,
                y=y_pos,
                text=text,
                mode="markers+text",
                marker=dict(
                    opacity=0.7,
                    size=20,
                    color="#ffcc99",
                    line=dict(width=1.0, color="black"),
                ),
                showlegend=False,
                hoverinfo="none",
            )
        )

        # plot shaft elements
        for sh_elm in self.shaft_elements:
            i = self.nodes.index(sh_elm.n)
            z_pos = self.nodes_pos[i]
            yc_pos = self.center_line_pos[i]

            position = (z_pos, yc_pos)
            fig = sh_elm._patch(position, check_sld, fig, length_units)

        mean_od = np.mean(nodes_o_d)
        # plot disk elements

        # calculate scale factor if disks have scale_factor='mass'
        if self.disk_elements:
            scaled_disks = [
                disk for disk in self.disk_elements if disk.scale_factor == "mass"
            ]
            if scaled_disks:
                max_mass = max([disk.m for disk in scaled_disks])
                for disk in scaled_disks:
                    f = disk.m / max_mass
                    disk._scale_factor_calculated = (1 - f) * 0.5 + f * 1.0

            for disk in self.disk_elements:
                scale_factor = disk.scale_factor
                if scale_factor == "mass":
                    scale_factor = disk._scale_factor_calculated
                step = scale_factor * mean_od

                z_pos = (
                    Q_(self.df[self.df.tag == disk.tag]["nodes_pos_l"].values[0], "m")
                    .to(length_units)
                    .m
                )
                y_pos = (
                    Q_(self.df[self.df.tag == disk.tag]["y_pos"].values[0], "m")
                    .to(length_units)
                    .m
                )
                yc_pos = center_line_pos[self.nodes.index(disk.n)]
                position = (z_pos, y_pos, yc_pos, step)
                fig = disk._patch(position, fig)

        # plot bearings
        for bearing in self.bearing_elements:
            z_pos = (
                Q_(self.df[self.df.tag == bearing.tag]["nodes_pos_l"].values[0], "m")
                .to(length_units)
                .m
            )
            y_pos = (
                Q_(self.df[self.df.tag == bearing.tag]["y_pos"].values[0], "m")
                .to(length_units)
                .m
            )
            y_pos_sup = (
                Q_(self.df[self.df.tag == bearing.tag]["y_pos_sup"].values[0], "m")
                .to(length_units)
                .m
            )
            node = bearing.n
            if node in self.link_nodes:
                node = self._find_linked_bearing_node(node)
            yc_pos = center_line_pos[self.nodes.index(node)]

            position = (z_pos, y_pos, y_pos_sup, yc_pos)
            bearing._patch(position, fig)

        # plot point mass
        for p_mass in self.point_mass_elements:
            z_pos = (
                Q_(self.df[self.df.tag == p_mass.tag]["nodes_pos_l"].values[0], "m")
                .to(length_units)
                .m
            )
            y_pos = (
                Q_(self.df[self.df.tag == p_mass.tag]["y_pos"].values[0], "m")
                .to(length_units)
                .m
            )
            node = p_mass.n
            if node in self.link_nodes:
                node = self._find_linked_bearing_node(node)
            yc_pos = center_line_pos[self.nodes.index(node)]

            position = (z_pos, y_pos, yc_pos)
            fig = p_mass._patch(position, fig)

        fig.update_xaxes(
            title_text=f"Axial location ({length_units})",
            showgrid=False,
            mirror=True,
            scaleanchor="y",
            scaleratio=1.5,
        )
        fig.update_yaxes(
            title_text=f"Shaft radius ({length_units})",
            showgrid=False,
            mirror=True,
        )
        kwargs["title"] = kwargs.get("title", "Rotor Model")
        fig.update_layout(**kwargs)

        return fig

    @check_units
    def run_campbell(
        self, speed_range, frequencies=6, frequency_type="wd", torsional_analysis=False
    ):
        """Calculate the Campbell diagram.

        This function will calculate the damped natural frequencies
        for a speed range.

        Available plotting methods:
            .plot()

        Parameters
        ----------
        speed_range : array, pint.Quantity
            Array with the speed range in rad/s.
        frequencies : int, optional
            Number of frequencies that will be calculated.
            Default is 6.
        frequency_type : str, optional
            Choose between displaying results related to the undamped natural
            frequencies ("wn") or damped natural frequencies ("wd").
            The default is "wd".
        torsional_analysis : bool, optional
            If True, performs a separate torsional analysis and returns the
            respective modes in the Campbell diagram. In this case, a system
            with only torsional degrees of freedom is considered, thus
            disregarding coupled modes (lateral + torsional). Default is False.

        Returns
        -------
        results : ross.CampbellResults
            For more information on attributes and methods available see:
            :py:class:`ross.CampbellResults`

        Examples
        --------
        >>> import ross as rs
        >>> rotor1 = rs.rotor_example()
        >>> speed = np.linspace(0, 400, 11)

        Diagram with undamped natural frequencies
        >>> camp = rotor1.run_campbell(speed, frequency_type="wn")

        Diagram with damped natural frequencies
        >>> camp = rotor1.run_campbell(speed)

        Plotting Campbell Diagram
        >>> fig = camp.plot()
        """

        # store in results [speeds(x axis), frequencies[0] or logdec[1] or
        # whirl[2](y axis), 3]
        self._check_frequency_array(speed_range)

        results = np.zeros([len(speed_range), frequencies, 4])

        # MAC criterion to track modes
        def MAC(u, v):
            H = lambda a: a.T.conj()
            return np.absolute((H(u) @ v) ** 2 / ((H(u) @ u) * (H(v) @ v)))

        num_modes = 2 * (frequencies + 2)  # ensure to get the right modes
        evec_size = int(num_modes / 2)
        mode_order = np.arange(evec_size)
        threshold = 0.9
        evec_u = []

        modal_results = {}
        for i, w in enumerate(speed_range):
            modal = self.run_modal(speed=w, num_modes=num_modes)
            modal_results[w] = modal

            evec_v = modal.evectors[:, :evec_size]

            if i > 0:
                macs = np.zeros((evec_size, evec_size))
                for u in enumerate(evec_u.T):
                    for v in enumerate(evec_v.T):
                        macs[u[0], v[0]] = MAC(u[1], v[1])

                mask = macs > threshold
                found_order = np.where(
                    mask.any(axis=1), np.argmax(macs * mask, axis=1), -1
                )
                modes_not_found = np.where(found_order == -1)[0]

                if len(modes_not_found):
                    missing_modes = sorted(set(mode_order) - set(found_order))
                    found_order[modes_not_found] = missing_modes[: len(modes_not_found)]

                if not (found_order == mode_order).all():
                    modal.evectors = modal.evectors[:, found_order]
                    modal.evalues = modal.evalues[found_order]
                    modal.wd = modal.wd[found_order]
                    modal.wn = modal.wn[found_order]
                    modal.log_dec = modal.log_dec[found_order]
                    modal.damping_ratio = modal.damping_ratio[found_order]
                    modal.shapes = list(np.array(modal.shapes)[found_order])

            evec_u = modal.evectors[:, :evec_size]

            if frequency_type == "wd":
                results[i, :, 0] = modal.wd[:frequencies]
                results[i, :, 1] = modal.log_dec[:frequencies]
                results[i, :, 2] = modal.damping_ratio[:frequencies]
                results[i, :, 3] = modal.whirl_values()[:frequencies]
            else:
                idx = modal.wn.argsort()
                results[i, :, 0] = modal.wn[idx][:frequencies]
                results[i, :, 1] = modal.log_dec[idx][:frequencies]
                results[i, :, 2] = modal.damping_ratio[idx][:frequencies]
                results[i, :, 3] = modal.whirl_values()[idx][:frequencies]

        if torsional_analysis:
            rotor_t = convert_6dof_to_torsional(self)
            campbell_t = rotor_t.run_campbell(
                speed_range=speed_range,
                frequencies=int(frequencies / 6),
                frequency_type=frequency_type,
            )

        results = CampbellResults(
            speed_range=speed_range,
            wd=results[..., 0],
            log_dec=results[..., 1],
            damping_ratio=results[..., 2],
            whirl_values=results[..., 3],
            modal_results=modal_results,
            number_dof=self.number_dof,
            run_modal=lambda w: self.run_modal(speed=w, num_modes=num_modes),
            campbell_torsional=campbell_t if torsional_analysis else None,
        )

        return results

    def run_ucs(
        self,
        stiffness_range=None,
        bearing_frequency_range=None,
        num_modes=16,
        num=20,
        synchronous=False,
        **kwargs,
    ):
        """Run Undamped Critical Speeds analyzes.

        This method will run the undamped critical speed analyzes for a given range
        of stiffness values. If the range is not provided, the bearing
        stiffness at rated speed will be used to create a range.

        Parameters
        ----------
        stiffness_range : tuple, optional
            Tuple with (start, end) for stiffness range in a log scale.
            In linear space, the sequence starts at ``base ** start``
            (`base` to the power of `start`) and ends with ``base ** stop``
            (see `endpoint` below). Here base is 10.0.
        bearing_frequency_range : tuple, optional
            The bearing frequency range used to calculate the intersection points.
            In some cases bearing coefficients will have to be extrapolated.
            The default is None. In this case the bearing frequency attribute is used.
        num_modes : int, optional
            Number of modes to be calculated. This uses scipy.sparse.eigs method.
            Default is 16. In this case 4 modes are plotted, since for each pair
            of eigenvalues calculated we have one wn, and we show only the
            forward mode in the plots.
        num : int
            Number of steps in the range.
            Default is 20.
        synchronous : bool, optional
            If True a synchronous analysis is carried out according to :cite:`rouch1980dynamic`.
            Default is False.

        Returns
        -------
        results : ross.UCSResults
            For more information on attributes and methods available see:
            :py:class:`ross.UCSResults`
        """
        if stiffness_range is None:
            if self.rated_w is not None:
                bearing = self.bearing_elements[0]
                k = bearing.kxx_interpolated(self.rated_w)
                k = int(np.log10(k))
                stiffness_range = (k - 3, k + 3)
            else:
                stiffness_range = (6, 11)

        if bearing_frequency_range:
            bearing_frequency_range = np.linspace(
                bearing_frequency_range[0], bearing_frequency_range[1], 30
            )

        stiffness_log = np.logspace(*stiffness_range, num=num)
        # for each pair of eigenvalues calculated we have one wn, and we show only
        # the forward mode in the plots, therefore we have num_modes / 2 / 2
        rotor_wn = np.zeros((num_modes // 2 // 2, len(stiffness_log)))

        # ensure that no proportional damping is considered
        shaft_elements = deepcopy(self.shaft_elements)
        for sh in shaft_elements:
            sh.alpha = sh.beta = 0

        # exclude the seals
        bearings_elements = [
            b for b in self.bearing_elements if not isinstance(b, SealElement)
        ]

        for i, k in enumerate(stiffness_log):
            bearings = [
                BearingElement(b.n, kxx=k, cxx=0)
                for b in bearings_elements
                if b.n not in self.link_nodes
            ]

            rotor = convert_6dof_to_4dof(
                self.__class__(
                    shaft_elements=shaft_elements,
                    disk_elements=self.disk_elements,
                    bearing_elements=bearings,
                )
            )

            modal = rotor.run_modal(
                speed=0, num_modes=num_modes, synchronous=synchronous
            )
            try:
                rotor_wn[:, i] = modal.wn[::2]
            except ValueError:
                rotor_wn[:, i] = modal.wn[::2][:-1]

        bearing0 = bearings_elements[0]

        # if bearing does not have constant coefficient, check intersection points
        if bearing_frequency_range is None:
            if bearing0.frequency is None:
                bearing_frequency_margin = rotor_wn.min() * 0.1
                bearing_frequency_range = np.linspace(
                    rotor_wn.min() - bearing_frequency_margin,
                    rotor_wn.max() + bearing_frequency_margin,
                    10,
                )
            else:
                bearing_frequency_range = bearing0.frequency

        # calculate interception points
        intersection_points = {"x": [], "y": []}

        # save critical mode shapes in the results
        critical_points_modal = []

        coeffs = (
            ["kxx"] if np.array_equal(bearing0.kxx, bearing0.kyy) else ["kxx", "kyy"]
        )

        for wn in rotor_wn:
            for coeff in coeffs:
                x1 = stiffness_log
                y1 = wn
                x2 = getattr(bearing0, f"{coeff}_interpolated")(bearing_frequency_range)
                y2 = bearing_frequency_range
                x, y = intersection(x1, y1, x2, y2)

                if len(x) > 0:
                    for k, speed in zip(x, y):
                        intersection_points["x"].append(float(k))
                        intersection_points["y"].append(float(speed))

                        # create bearing
                        bearings = [
                            BearingElement(b.n, kxx=k, cxx=0, n_link=b.n_link)
                            for b in bearings_elements
                        ]

                        for b in bearings:
                            if b.n in self.link_nodes:
                                node = self._find_linked_bearing_node(b.n)
                                linked_bearing = [b for b in bearings if b.n == node][0]

                                kxx_brg = np.array(linked_bearing.kxx)
                                kyy_brg = np.array(linked_bearing.kyy)
                                kxx_add = np.array(b.kxx)
                                kyy_add = np.array(b.kyy)

                                with np.errstate(divide="ignore"):
                                    kxx_eq = 1 / (1 / kxx_brg + 1 / kxx_add)
                                    kyy_eq = 1 / (1 / kyy_brg + 1 / kyy_add)
                                    kxx_eq[np.isinf(kxx_eq)] = 0
                                    kyy_eq[np.isinf(kyy_eq)] = 0

                                linked_bearing.kxx = list(kxx_eq)
                                linked_bearing.kyy = list(kyy_eq)

                        bearings = [
                            b
                            for b in bearings
                            if b.n not in self.link_nodes
                            and setattr(b, "n_link", None) is None
                        ]

                        # create rotor
                        rotor_critical = convert_6dof_to_4dof(
                            Rotor(
                                shaft_elements=shaft_elements,
                                disk_elements=self.disk_elements,
                                bearing_elements=bearings,
                            )
                        )

                        modal_critical = rotor_critical.run_modal(speed=speed)
                        critical_points_modal.append(modal_critical)

        results = UCSResults(
            stiffness_range,
            stiffness_log,
            bearing_frequency_range,
            rotor_wn,
            bearing0,
            intersection_points,
            critical_points_modal,
        )

        return results

    def run_level1(self, n=5, stiffness_range=None, num=5, **kwargs):
        """Plot level 1 stability analysis.

        This method will plot the stability 1 analysis for a
        given stiffness range.

        Parameters
        ----------
        n : int
            Number of steps in the range.
            Default is 5.
        stiffness_range : tuple, optional
            Tuple with (start, end) for stiffness range.
            This will be used to create an evenly numbers spaced evenly on a log scale
            to create a better visualization (see np.logspace).
        kwargs : optional
            Additional key word arguments can be passed to change the plot layout only
            (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        results : ross.Level1Results
            For more information on attributes and methods available see:
            :py:class:`ross.Level1Results`

        Example
        -------
        >>> i_d = 0
        >>> o_d = 0.05
        >>> n = 6
        >>> L = [0.25 for _ in range(n)]
        >>> shaft_elem = [
        ...     ShaftElement(
        ...         l, i_d, o_d, material=steel, shear_effects=True,
        ...         rotary_inertia=True, gyroscopic=True
        ...     )
        ...     for l in L
        ... ]
        >>> disk0 = DiskElement.from_geometry(
        ...     n=2, material=steel, width=0.07, i_d=0.05, o_d=0.28
        ... )
        >>> disk1 = DiskElement.from_geometry(
        ...     n=4, material=steel, width=0.07, i_d=0.05, o_d=0.28
        ... )
        >>> stfx = 1e6
        >>> stfy = 0.8e6
        >>> bearing0 = BearingElement(0, kxx=stfx, kyy=stfy, cxx=0)
        >>> bearing1 = BearingElement(6, kxx=stfx, kyy=stfy, cxx=0)
        >>> rotor = Rotor(shaft_elem, [disk0, disk1], [bearing0, bearing1], rated_w=0)
        >>> level1 = rotor.run_level1(n=0, stiffness_range=(1e6, 1e11))
        >>> fig = level1.plot()
        """
        if stiffness_range is None:
            if self.rated_w is not None:
                bearing = self.bearing_elements[0]
                k = bearing.kxx.interpolated(self.rated_w)
                k = int(np.log10(k))
                stiffness_range = (k - 3, k + 3)
            else:
                stiffness_range = (6, 11)

        stiffness = np.linspace(*stiffness_range, num)
        log_dec = np.zeros(len(stiffness))

        # set rotor speed to mcs
        speed = self.rated_w
        modal = self.run_modal(speed=speed)

        for i, Q in enumerate(stiffness):
            bearings = [copy(b) for b in self.bearing_elements]
            cross_coupling = bearings[0].__class__(n=n, kxx=0, cxx=0, kxy=Q, kyx=-Q)
            bearings.append(cross_coupling)

            rotor = self.__class__(self.shaft_elements, self.disk_elements, bearings)

            modal = rotor.run_modal(speed=speed)
            non_backward = modal.whirl_direction() != "Backward"
            log_dec[i] = modal.log_dec[non_backward][0]

        results = Level1Results(stiffness, log_dec)

        return results

    @check_units
    def run_time_response(self, speed, F, t, method="default", **kwargs):
        """Calculate the time response.

        This function will take a rotor object and calculate its time response
        given a force and a time.

        Available plotting methods:
            .plot_1d()
            .plot_2d()
            .plot_3d()

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
            For more information on attributes and methods available see:
            :py:class:`ross.TimeResponseResults`

        Examples
        --------
        >>> from ross.probe import Probe
        >>> rotor = rotor_example()
        >>> speed = 500.0
        >>> size = 1000
        >>> node = 3
        >>> probe1 = Probe(3, 0)
        >>> t = np.linspace(0, 10, size)
        >>> F = np.zeros((size, rotor.ndof))
        >>> F[:, rotor.number_dof * node + 0] = 10 * np.cos(2 * t)
        >>> F[:, rotor.number_dof * node + 1] = 10 * np.sin(2 * t)
        >>> response = rotor.run_time_response(speed, F, t)
        >>> response.yout[:, rotor.number_dof * node + 1] # doctest: +ELLIPSIS
        array([ 0.00000000e+00,  1.86686693e-07,  8.39130663e-07, ...
        >>> # plot time response for a given probe:
        >>> fig1 = response.plot_1d(probe=[probe1])
        >>> # plot orbit response - plotting 2D nodal orbit:
        >>> fig2 = response.plot_2d(node=node)
        >>> # plot orbit response - plotting 3D orbits - full rotor model:
        >>> fig3 = response.plot_3d()
        """
        t_, yout, xout = self.time_response(speed, F, t, method=method, **kwargs)

        results = TimeResponseResults(self, t, yout, xout)

        return results

    @check_units
    def run_misalignment(
        self,
        node,
        unbalance_magnitude,
        unbalance_phase,
        speed,
        t,
        coupling="flex",
        **kwargs,
    ):
        """Run analysis for the rotor system with misalignment given an
        unbalance force.

        Misalignment object is instantiated and system's time response is simulated.
        There are two types of coupling: flexible (flex) and rigid, each with distinct
        parameters. These parameters are passed to the respective method through **kwargs.

        Parameters
        ----------
        node : list, int
            Node where the unbalance is applied.
        unbalance_magnitude : list, float, pint.Quantity
            Unbalance magnitude (kg.m).
        unbalance_phase : list, float, pint.Quantity
            Unbalance phase (rad).
        speed : float or array_like, pint.Quantity
            Rotor speed.
        F : array
            Force array (needs to have the same number of rows as time array).
            Each column corresponds to a dof and each row to a time.
        t : array
            Time array.
        coupling : str
            Coupling type. The avaible types are: "flex" and "rigid".
            Default is "flex".

        **kwargs : dictionary
            If coupling = "flex", **kwargs receives:
                n : float
                    Number of shaft element where the misalignment is ocurring.
                mis_type: string
                    Name of the chosen misalignment type.
                    The avaible types are: "parallel", "angular" and "combined".
                mis_distance_x : float, pint.Quantity
                    Parallel misalignment distance between driving rotor and driven
                    rotor along X direction.
                mis_distance_y : float, pint.Quantity
                    Parallel misalignment distance between driving rotor and driven
                    rotor along Y direction.
                mis_angle : float, pint.Quantity
                    Angular misalignment angle.
                radial_stiffness : float, pint.Quantity
                    Radial stiffness of flexible coupling.
                bending_stiffness : float, pint.Quantity
                    Bending stiffness of flexible coupling. Provide if mis_type is
                    "angular" or "combined".
                input_torque : float, pint.Quantity
                    Driving torque. Default is 0.
                load_torque : float, pint.Quantity
                    Driven torque. Default is 0.

            If coupling = "rigid", **kwargs receives:
                n : float
                    Number of shaft element where the misalignment is ocurring.
                mis_distance : float, pint.Quantity
                    Parallel misalignment distance between driving rotor and driven rotor.
                input_torque : float, pint.Quantity
                    Driving torque. Default is 0.
                load_torque : float, pint.Quantity
                    Driven torque. Default is 0.

            Additional keyword arguments can be passed to define the parameters
            of the Newmark method if it is used (e.g. gamma, beta, tol, ...).
            See `ross.utils.newmark` for more details.
            Other keyword arguments can also be passed to be used in numerical
            integration (e.g. model_reduction).
            See `Rotor.integrate_system` for more details.

        Returns
        -------
        results : ross.TimeResponseResults
            For more information on attributes and methods available see:
            :py:class:`ross.TimeResponseResults`

        Examples
        --------
        >>> import ross as rs
        >>> from ross.probe import Probe
        >>> from ross.units import Q_
        >>> rotor = rotor_example_with_damping()
        >>> n1 = rotor.disk_elements[0].n
        >>> n2 = rotor.disk_elements[1].n
        >>> results = rotor.run_misalignment(
        ...    node=[n1, n2],
        ...    unbalance_magnitude=[5e-4, 0],
        ...    unbalance_phase=[-np.pi / 2, 0],
        ...    speed=Q_(1200, "RPM"),
        ...    t=np.arange(0, 0.5, 0.0001),
        ...    coupling="rigid",
        ...    n=0,
        ...    mis_distance=2e-4,
        ...    input_torque=0,
        ...    load_torque=0,
        ...    model_reduction={"num_modes": 12},  # Pseudo-modal method
        ... )
        Running with model reduction: pseudomodal
        >>> probe1 = Probe(14, 0)
        >>> probe2 = Probe(22, 0)
        >>> fig1 = results.plot_1d([probe1, probe2])
        >>> fig2 = results.plot_dfft(
        ...     [probe1, probe2],
        ...     frequency_range=Q_((0, 200), "Hz"),
        ...     yaxis_type="log",
        ... )
        """

        if coupling == "flex":
            fault = MisalignmentFlex(
                self,
                n=kwargs.get("n"),
                mis_type=kwargs.get("mis_type"),
                mis_distance_x=kwargs.get("mis_distance_x"),
                mis_distance_y=kwargs.get("mis_distance_y"),
                mis_angle=kwargs.get("mis_angle"),
                radial_stiffness=kwargs.get("radial_stiffness"),
                bending_stiffness=kwargs.get("bending_stiffness"),
                input_torque=kwargs.get("input_torque", 0),
                load_torque=kwargs.get("load_torque", 0),
            )

        elif coupling == "rigid":
            fault = MisalignmentRigid(
                self,
                n=kwargs.get("n"),
                mis_distance=kwargs.get("mis_distance"),
                input_torque=kwargs.get("input_torque", 0),
                load_torque=kwargs.get("load_torque", 0),
            )

        else:
            raise Exception("Check the choosed coupling type!")

        results = fault.run(
            node, unbalance_magnitude, unbalance_phase, speed, t, **kwargs
        )

        return results

    @check_units
    def run_rubbing(
        self,
        n,
        distance,
        contact_stiffness,
        contact_damping,
        friction_coeff,
        node,
        unbalance_magnitude,
        unbalance_phase,
        speed,
        t,
        torque=False,
        **kwargs,
    ):
        """Run analysis for the rotor system with rubbing given an unbalance force.

        Rubbing object is instantiated and system's time response is simulated.

        Parameters
        ----------
        n : int
            Number of shaft element where rubbing is ocurring.
        distance : float, pint.Quantity
            Distance between the housing and shaft surface.
        contact_stiffness : float, pint.Quantity
            Contact stiffness.
        contact_damping : float, pint.Quantity
            Contact damping.
        friction_coeff : float
            Friction coefficient.
        node : list, int
            Node where the unbalance is applied.
        unbalance_magnitude : list, float, pint.Quantity
            Unbalance magnitude (kg.m).
        unbalance_phase : list, float, pint.Quantity
            Unbalance phase (rad).
        speed : float or array_like, pint.Quantity
            Rotor speed.
        F : array
            Force array (needs to have the same number of rows as time array).
            Each column corresponds to a dof and each row to a time.
        t : array
            Time array.
        torque : bool, optional
            If True a torque is considered by rubbing.
            Default is False.
        **kwargs : optional
            Additional keyword arguments can be passed to define the parameters
            of the Newmark method if it is used (e.g. gamma, beta, tol, ...).
            See `ross.utils.newmark` for more details.
            Other keyword arguments can also be passed to be used in numerical
            integration (e.g. model_reduction).
            See `Rotor.integrate_system` for more details.

        Returns
        -------
        results : ross.TimeResponseResults
            For more information on attributes and methods available see:
            :py:class:`ross.TimeResponseResults`

        Examples
        --------
        >>> import ross as rs
        >>> from ross.units import Q_
        >>> from ross.probe import Probe
        >>> rotor = rotor_example_with_damping()
        >>> n1 = rotor.disk_elements[0].n
        >>> n2 = rotor.disk_elements[1].n
        >>> results = rotor.run_rubbing(
        ...    n=12,
        ...    distance=7.95e-5,
        ...    contact_stiffness=1.1e6,
        ...    contact_damping=40,
        ...    friction_coeff=0.3,
        ...    torque=False,
        ...    node=[n1, n2],
        ...    unbalance_magnitude=[5e-4, 0],
        ...    unbalance_phase=[-np.pi / 2, 0],
        ...    speed=Q_(1200, "RPM"),
        ...    t=np.arange(0, 0.5, 0.0001),
        ...    model_reduction={"num_modes": 12},  # Pseudo-modal method
        ... )
        Running with model reduction: pseudomodal
        >>> probe1 = Probe(14, 0)
        >>> probe2 = Probe(22, 0)
        >>> fig1 = results.plot_1d([probe1, probe2])
        >>> fig2 = results.plot_dfft(
        ...     [probe1, probe2],
        ...     frequency_range=Q_((0, 200), "Hz"),
        ...     yaxis_type="log",
        ... )
        """
        fault = Rubbing(
            self,
            n,
            distance,
            contact_stiffness,
            contact_damping,
            friction_coeff,
            torque=torque,
        )

        results = fault.run(
            node, unbalance_magnitude, unbalance_phase, speed, t, **kwargs
        )

        return results

    @check_units
    def run_crack(
        self,
        n,
        depth_ratio,
        node,
        unbalance_magnitude,
        unbalance_phase,
        speed,
        t,
        crack_model="Mayes",
        cross_divisions=None,
        **kwargs,
    ):
        """Run analysis for the rotor system with crack given an unbalance force.

        Crack object is instantiated and system's time response is simulated.

        Parameters
        ----------
        n : float
            Element number where the crack is located.
        depth_ratio : float
            Crack depth ratio related to the diameter of the crack container element.
            A depth value of 0.1 is equal to 10%, 0.2 equal to 20%, and so on.
        node : list, int
            Node where the unbalance is applied.
        unbalance_magnitude : list, float, pint.Quantity
            Unbalance magnitude (kg.m).
        unbalance_phase : list, float, pint.Quantity
            Unbalance phase (rad).
        speed : float or array_like, pint.Quantity
            Rotor speed.
        F : array
            Force array (needs to have the same number of rows as time array).
            Each column corresponds to a dof and each row to a time.
        t : array
            Time array.
        crack_model : string, optional
            String containing type of crack model chosed. The available types are: "Mayes",
            "Gasch", "Flex Open" and "Flex Breathing". Default is "Mayes".
        cross_divisions: float, optional
            Number of square divisions into which the cross-section of the cracked element
            will be divided in the analysis conducted for the Flex Breathing model.
        **kwargs : optional
            Additional keyword arguments can be passed to define the parameters
            of the Newmark method if it is used (e.g. gamma, beta, tol, ...).
            See `ross.utils.newmark` for more details.
            Other keyword arguments can also be passed to be used in numerical
            integration (e.g. model_reduction).
            See `Rotor.integrate_system` for more details.

        Returns
        -------
        results : ross.TimeResponseResults
            For more information on attributes and methods available see:
            :py:class:`ross.TimeResponseResults`

        Examples
        --------
        >>> import ross as rs
        >>> from ross.probe import Probe
        >>> from ross.units import Q_
        >>> rotor = rs.rotor_example_with_damping()
        >>> n1 = rotor.disk_elements[0].n
        >>> n2 = rotor.disk_elements[1].n
        >>> results = rotor.run_crack(
        ...    n=18,
        ...    depth_ratio=0.2,
        ...    node=[n1, n2],
        ...    unbalance_magnitude=[5e-4, 0],
        ...    unbalance_phase=[-np.pi / 2, 0],
        ...    crack_model="Mayes",
        ...    speed=Q_(1200, "RPM"),
        ...    t=np.arange(0, 0.5, 0.0001),
        ...    model_reduction={"num_modes": 12},  # Pseudo-modal method
        ... )
        Running with model reduction: pseudomodal
        >>> probe1 = Probe(14, 0)
        >>> probe2 = Probe(22, 0)
        >>> fig1 = results.plot_1d([probe1, probe2])
        >>> fig2 = results.plot_dfft(
        ...     [probe1, probe2],
        ...     frequency_range=Q_((0, 200), "Hz"),
        ...     yaxis_type="log",
        ... )
        """
        fault = Crack(self, n, depth_ratio, crack_model, cross_divisions)

        results = fault.run(
            node, unbalance_magnitude, unbalance_phase, speed, t, **kwargs
        )

        return results

    def save_mat(self, file, speed, frequency=None):
        """Save matrices and rotor model to a .mat file.

        Parameters
        ----------
        file : str, pathlib.Path

        speed: float
            Rotor speed.
        frequency: float, optional
            Excitation frequency.
            Default is rotor speed.

        Examples
        --------
        >>> from tempfile import tempdir
        >>> from pathlib import Path
        >>> # create path for temporary file
        >>> file = Path(tempdir) / 'new_matrices'
        >>> rotor = rotor_example()
        >>> rotor.save_mat(file, speed=0)
        """
        if frequency is None:
            frequency = speed

        dic = {
            "M": self.M(frequency),
            "K": self.K(frequency),
            "C": self.C(frequency),
            "G": self.G(),
            "nodes": self.nodes_pos,
        }

        sio.savemat(file, dic)

    def save(self, file):
        """Save the rotor to a .toml file.

        Parameters
        ----------
        file : str or pathlib.Path

        Examples
        --------
        >>> from tempfile import tempdir
        >>> from pathlib import Path
        >>> # create path for temporary file
        >>> file = Path(tempdir) / 'rotor.toml'
        >>> rotor = rotor_example()
        >>> rotor.save(file)
        """
        with open(file, "w") as f:
            toml.dump({"parameters": self.parameters}, f)
        for el in self.elements:
            el.save(file)

    @classmethod
    def load(cls, file):
        """Load rotor from toml file.

        Parameters
        ----------
        file : str or pathlib.Path
            String or Path for a .toml file.

        Returns
        -------
        rotor : ross.rotor.Rotor

        Example
        -------
        >>> from tempfile import tempdir
        >>> from pathlib import Path
        >>> # create path for temporary file
        >>> file = Path(tempdir) / 'new_rotor1.toml'
        >>> rotor1 = rotor_example()
        >>> rotor1.save(file)
        >>> rotor2 = Rotor.load(file)
        >>> rotor1 == rotor2
        True
        """
        data = toml.load(file)
        parameters = data["parameters"]

        elements = []
        for el_name, el_data in data.items():
            if el_name == "parameters":
                continue
            class_name = el_name.split("_")[0]
            try:
                elements.append(globals()[class_name].read_toml_data(el_data))
            except KeyError:
                import rossxl as rsxl

                elements.append(getattr(rsxl, class_name).read_toml_data(el_data))

        shaft_elements = []
        disk_elements = []
        bearing_elements = []
        point_mass_elements = []
        for el in elements:
            if isinstance(el, ShaftElement):
                shaft_elements.append(el)
            elif isinstance(el, DiskElement):
                disk_elements.append(el)
            elif isinstance(el, BearingElement):
                bearing_elements.append(el)
            elif isinstance(el, PointMass):
                point_mass_elements.append(el)

        return cls(
            shaft_elements=shaft_elements,
            disk_elements=disk_elements,
            bearing_elements=bearing_elements,
            point_mass_elements=point_mass_elements,
            **parameters,
        )

    def run_static(self):
        """Run static analysis.

        Static analysis calculates free-body diagram, deformed shaft, shearing
        force diagram and bending moment diagram.

        Available plotting methods:
            .plot_deformation()
            .plot_bending_moment()
            .plot_shearing_force()
            .plot_free_body_diagram()

        Attributes
        ----------
        shaft_weight: float
            Shaft total weight
        disk_forces_nodal : dict
            Relates the static force at each node due to the weight of disks
        bearing_forces_nodal : dict
            Relates the static force at each node due to the bearing reaction forces.
        bearing_forces_tag : dict
            Indicates the reaction force exerted by each bearing.
        disk_forces_tag : dict
            Indicates the force exerted by each disk.
        displacement_y: array
            The shaft static displacement vector,
        Vx: array
            Shearing force vector
        Bm: array
            Bending moment vector

        Returns
        -------
        results : ross.StaticResults
            For more information on attributes and methods available see:
            :py:class:`ross.StaticResults`

        Raises
        ------
        ValueError
            Error raised if the rotor has no bearing elements.

        Example
        -------
        >>> import ross as rs
        >>> rotor = rs.rotor_example()
        >>> static = rotor.run_static()
        >>> rotor.bearing_forces_nodal
        {'node_0': 432...
        >>> rotor.bearing_forces_tag # doctest: +ELLIPSIS
        {'Bearing 0': 432...

        Plotting static deformation
        >>> fig = static.plot_deformation()

        Plotting bending moment
        >>> fig = static.plot_bending_moment()

        Plotting shearing force
        >>> fig = static.plot_shearing_force()

        Plotting free-body diagram
        >>> fig = static.plot_free_body_diagram()
        """
        if not len(self.df_bearings):
            raise ValueError("Rotor has no bearings")

        aux_brg = []
        aux_brg_1 = []
        for elm in self.bearing_elements:
            if not isinstance(elm, SealElement):
                if elm.n not in self.nodes:
                    pass
                elif elm.n_link in self.nodes:
                    aux_brg.append(
                        elm.__class__(n=elm.n, n_link=elm.n_link, kxx=1e20, cxx=0)
                    )
                    aux_brg_1.append(
                        elm.__class__(n=elm.n, n_link=elm.n_link, kxx=0, cxx=0)
                    )
                else:
                    aux_brg.append(elm.__class__(n=elm.n, kxx=1e20, cxx=0))
                    aux_brg_1.append(elm.__class__(n=elm.n, kxx=0, cxx=0))

        aux_rotor = Rotor(self.shaft_elements, self.disk_elements, aux_brg)
        aux_rotor_1 = Rotor(self.shaft_elements, self.disk_elements, aux_brg_1)

        aux_M = aux_rotor.M(0)
        aux_K = aux_rotor.K(0)
        aux1_K = aux_rotor_1.K(0)

        # convert to 4 dof
        num_dof = 4
        aux_M = remove_dofs(aux_M)
        aux_K = remove_dofs(aux_K)
        aux1_K = remove_dofs(aux1_K)

        # gravity aceleration vector
        g = -9.8065
        weight = self.gravitational_force(g=g, M=aux_M, num_dof=num_dof)

        # calculates u, for [K]*(u) = (F)
        displacement = (la.solve(aux_K, weight)).flatten()
        displacement_y = displacement[1::num_dof]

        # calculate forces
        nodal_forces = aux1_K @ displacement

        bearing_force_nodal = {}
        disk_force_nodal = {}
        bearing_force_tag = {}
        disk_force_tag = {}

        elm_weight = np.zeros((len(self.nodes_pos) - 1, 2))
        nodal_shaft_weight = np.zeros(len(self.nodes_pos))

        vx_axis = np.zeros_like(elm_weight)
        for sh in self.shaft_elements:
            vx_axis[sh.n_l] = [
                self.nodes_pos[sh.n_l],
                self.nodes_pos[sh.n_r],
            ]
            elm_weight[sh.n_l] += g * np.array([0, sh.m])

            nodal_shaft_weight[sh.n_r] += g * sh.m * sh.beam_cg / sh.L
            nodal_shaft_weight[sh.n_l] += g * sh.m * (1 - sh.beam_cg / sh.L)

        elm_weight[-1, 1] = 0
        aux_nodal_forces = nodal_forces[: num_dof * (self.nodes[-1] + 1)]

        reaction_forces = nodal_forces[1::num_dof] - weight[1::num_dof]

        for bearing in aux_rotor.bearing_elements:
            bearing_force_nodal[f"node_{bearing.n:d}"] = reaction_forces[bearing.n]
            bearing_force_tag[f"{bearing.tag}"] = reaction_forces[bearing.n]

        for disk in aux_rotor.disk_elements:
            disk_force_nodal[f"node_{disk.n:d}"] = -disk.m * g
            disk_force_tag[f"{disk.tag}"] = -disk.m * g

        nodal_forces_y = aux_nodal_forces[1::num_dof] - nodal_shaft_weight
        elm_forces_y = np.zeros_like(elm_weight)
        elm_forces_y[:, 0] = nodal_forces_y[:-1]
        elm_forces_y[-1, 1] = -nodal_forces_y[-1]
        elm_forces_y += elm_weight

        # Calculate shearing force
        # Each line represents an element, each column a station from the element
        vx = np.zeros_like(elm_weight)
        for j in range(vx.shape[0]):
            if j == 0:
                vx[j] = [elm_forces_y[j, 0], sum(elm_forces_y[j])]
            elif j == vx.shape[0] - 1:
                vx[j, 0] = vx[j - 1, 1] + elm_forces_y[j, 0]
                vx[j, 1] = elm_forces_y[j, 1]
            else:
                vx[j, 0] = vx[j - 1, 1] + elm_forces_y[j, 0]
                vx[j, 1] = vx[j, 0] + elm_forces_y[j, 1]
        vx = -vx

        # Calculate bending moment
        # Each line represents an element, each column a station from the element
        mx = np.zeros_like(vx)
        for j in range(mx.shape[0]):
            if j == 0:
                mx[j] = [0, 0.5 * sum(vx[j]) * np.diff(vx_axis[j])[0]]
            if j == mx.shape[0] - 1:
                mx[j] = [-0.5 * sum(vx[j]) * np.diff(vx_axis[j])[0], 0]
            else:
                mx[j, 0] = mx[j - 1, 1]
                mx[j, 1] = mx[j, 0] + 0.5 * sum(vx[j]) * np.diff(vx_axis[j])[0]

        # flattening arrays
        vx = vx.flatten()
        vx_axis = vx_axis.flatten()
        mx = mx.flatten()

        self.disk_forces_nodal = disk_force_nodal
        self.bearing_forces_nodal = bearing_force_nodal
        self.bearing_forces_tag = bearing_force_tag
        self.disk_forces_tag = disk_force_tag

        self.w_shaft = sum(self.df_shaft["m"]) * (-g)

        results = StaticResults(
            displacement_y,
            vx,
            mx,
            self.w_shaft,
            self.disk_forces_nodal,
            self.bearing_forces_nodal,
            self.nodes,
            self.nodes_pos,
            vx_axis,
        )

        return results

    def summary(self):
        """Plot the rotor summary.

        This functioncreates a summary of the main parameters and attributes of the
        rotor model. The data is presented in a table format.

        Returns
        -------
        results : ross.SummaryResults class
            An instance of SumarryResults class to build the summary table

        Examples
        --------
        >>> rotor = rotor_example()
        >>> table = rotor.summary().plot()
        >>> # to display the plot use the command:
        >>> # show(table)
        """
        self.df_disks = pd.merge(
            self.df_disks, self.df[["tag", "nodes_pos_l"]], on="tag", how="left"
        )
        self.df_bearings = pd.merge(
            self.df_bearings, self.df[["tag", "nodes_pos_l"]], on="tag", how="left"
        )
        self.run_static()
        forces = self.bearing_forces_tag
        results = SummaryResults(
            self.df_shaft,
            self.df_disks,
            self.df_bearings,
            forces,
            self.CG,
            self.Ip,
            self.tag,
        )
        return results

    @classmethod
    def from_section(
        cls,
        leng_data,
        idl_data,
        odl_data,
        idr_data=None,
        odr_data=None,
        material_data=None,
        disk_data=None,
        brg_seal_data=None,
        min_w=None,
        max_w=None,
        rated_w=None,
        nel_r=1,
        tag=None,
    ):
        """Build rotor from sections.

        This class is an alternative to build rotors from separated
        sections. Each section has the same number (n) of shaft elements.

        Parameters
        ----------
        leng_data : list
            List with the lengths of rotor regions.
        idl_data : list
            List with the inner diameters of rotor regions (Left Station).
        odl_data : list
            List with the outer diameters of rotor regions (Left Station).
        idr_data : list, optional
            List with the inner diameters of rotor regions (Right Station).
            Default is equal to idl_data (cylindrical element).
        odr_data : list, optional
            List with the outer diameters of rotor regions (Right Station).
            Default is equal to odl_data (cylindrical element).
        material_data : ross.material or list of ross.material
            Defines a single material for all sections or each section can be
            defined by a material individually.
        disk_data : dict, optional
            Dict holding disks datas.
            Example : disk_data=DiskElement.from_geometry(n=2,
                                                          material=steel,
                                                          width=0.07,
                                                          i_d=0,
                                                          o_d=0.28
                                                          )
            ***See 'disk_element.py' docstring for more information***
        brg_seal_data : dict, optional
            Dict holding lists of bearings and seals datas.
            Example : brg_seal_data=BearingElement(n=1, kxx=1e6, cxx=0,
                                                   kyy=1e6, cyy=0, kxy=0,
                                                   cxy=0, kyx=0, cyx=0)
            ***See 'bearing_seal_element.py' docstring for more information***
        nel_r : int, optional
            Number or elements per shaft region.
            Default is 1.
        tag : str
            A tag for the rotor

        Raises
        ------
        ValueError
            Error raised if lists size do not match.
        AttributeError
            Error raised if the shaft material is not defined.

        Returns
        -------
        A rotor object

        Example
        -------
        >>> from ross.materials import steel
        >>> rotor = Rotor.from_section(leng_data=[0.5,0.5,0.5],
        ...             odl_data=[0.05,0.05,0.05],
        ...             idl_data=[0,0,0],
        ...             material_data=steel,
        ...             disk_data=[DiskElement.from_geometry(n=1, material=steel, width=0.07, i_d=0, o_d=0.28),
        ...                        DiskElement.from_geometry(n=2, material=steel, width=0.07, i_d=0, o_d=0.35)],
        ...             brg_seal_data=[BearingElement(n=0, kxx=1e6, cxx=0, kyy=1e6, cyy=0, kxy=0, cxy=0, kyx=0, cyx=0),
        ...                            BearingElement(n=3, kxx=1e6, cxx=0, kyy=1e6, cyy=0, kxy=0, cxy=0, kyx=0, cyx=0)],
        ...             nel_r=1)
        >>> modal = rotor.run_modal(speed=0)
        >>> modal.wn.round(4)
        array([ 85.7634,  85.7634, 271.9326, 271.9326, 650.1377, 718.58  ])
        """
        if len(leng_data) != len(odl_data) or len(leng_data) != len(idl_data):
            raise ValueError(
                "The lists size do not match (leng_data, odl_data and idl_data)."
            )

        if material_data is None:
            raise AttributeError("Please define a material or a list of materials")

        if idr_data is None:
            idr_data = idl_data
        if odr_data is None:
            odr_data = odl_data
        else:
            if len(leng_data) != len(odr_data) or len(leng_data) != len(idr_data):
                raise ValueError(
                    "The lists size do not match (leng_data, odr_data and idr_data)."
                )

        def rotor_regions(nel_r):
            """Subroutine to discretize each rotor region into n elements.

            Parameters
            ----------
            nel_r : int
                Number of elements per region

            Returns
            -------
            regions : list
                List with elements
            """
            regions = []
            shaft_elements = []
            disk_elements = []
            bearing_elements = []

            try:
                if len(leng_data) != len(material_data):
                    raise IndexError(
                        "material_data size does not match size of other lists"
                    )

                # loop through rotor regions
                for i, leng in enumerate(leng_data):
                    le = leng / nel_r
                    for j in range(nel_r):
                        idl = (idr_data[i] - idl_data[i]) * j * le / leng + idl_data[i]
                        odl = (odr_data[i] - odl_data[i]) * j * le / leng + odl_data[i]
                        idr = (idr_data[i] - idl_data[i]) * (
                            j + 1
                        ) * le / leng + idl_data[i]
                        odr = (odr_data[i] - odl_data[i]) * (
                            j + 1
                        ) * le / leng + odl_data[i]
                        shaft_elements.append(
                            ShaftElement(
                                le,
                                idl,
                                odl,
                                idr,
                                odr,
                                material=material_data[i],
                                shear_effects=True,
                                rotary_inertia=True,
                                gyroscopic=True,
                            )
                        )
            except TypeError:
                for i, leng in enumerate(leng_data):
                    le = leng / nel_r
                    for j in range(nel_r):
                        idl = (idr_data[i] - idl_data[i]) * j * le / leng + idl_data[i]
                        odl = (odr_data[i] - odl_data[i]) * j * le / leng + odl_data[i]
                        idr = (idr_data[i] - idl_data[i]) * (
                            j + 1
                        ) * le / leng + idl_data[i]
                        odr = (odr_data[i] - odl_data[i]) * (
                            j + 1
                        ) * le / leng + odl_data[i]
                        shaft_elements.append(
                            ShaftElement(
                                le,
                                idl,
                                odl,
                                idr,
                                odr,
                                material=material_data,
                                shear_effects=True,
                                rotary_inertia=True,
                                gyroscopic=True,
                            )
                        )

            regions.extend([shaft_elements])

            for DiskEl in disk_data:
                aux_DiskEl = deepcopy(DiskEl)
                aux_DiskEl.n = nel_r * DiskEl.n
                aux_DiskEl.n_l = nel_r * DiskEl.n_l
                aux_DiskEl.n_r = nel_r * DiskEl.n_r
                disk_elements.append(aux_DiskEl)

            for Brg_SealEl in brg_seal_data:
                aux_Brg_SealEl = deepcopy(Brg_SealEl)
                aux_Brg_SealEl.n = nel_r * Brg_SealEl.n
                aux_Brg_SealEl.n_l = nel_r * Brg_SealEl.n
                aux_Brg_SealEl.n_r = nel_r * Brg_SealEl.n
                bearing_elements.append(aux_Brg_SealEl)

            regions.append(disk_elements)
            regions.append(bearing_elements)

            return regions

        regions = rotor_regions(nel_r)
        shaft_elements = regions[0]
        disk_elements = regions[1]
        bearing_elements = regions[2]

        return cls(
            shaft_elements,
            disk_elements,
            bearing_elements,
            min_w=min_w,
            max_w=max_w,
            rated_w=rated_w,
            tag=tag,
        )

    @classmethod
    def to_ross_only(cls, rotor):
        """Convert rotor with rsxl objects to ross only."""
        bearings_seals_rs = []
        for b in rotor.bearing_elements:
            if isinstance(b, SealElement):
                bearings_seals_rs.append(
                    SealElement(
                        n=b.n,
                        kxx=b.kxx,
                        kxy=b.kxy,
                        kyx=b.kyx,
                        kyy=b.kyy,
                        cxx=b.cxx,
                        cxy=b.cxy,
                        cyx=b.cyx,
                        cyy=b.cyy,
                        frequency=b.frequency,
                        tag=b.tag,
                        color=b.color,
                        n_link=b.n_link,
                        seal_leakage=b.seal_leakage,
                    )
                )
            else:
                bearings_seals_rs.append(
                    BearingElement(
                        n=b.n,
                        kxx=b.kxx,
                        kxy=b.kxy,
                        kyx=b.kyx,
                        kyy=b.kyy,
                        cxx=b.cxx,
                        cxy=b.cxy,
                        cyx=b.cyx,
                        cyy=b.cyy,
                        frequency=b.frequency,
                        tag=b.tag,
                        color=b.color,
                        n_link=b.n_link,
                    )
                )

        return cls(
            rotor.shaft_elements,
            rotor.disk_elements,
            bearings_seals_rs,
            rotor.point_mass_elements,
            min_w=rotor.min_w,
            max_w=rotor.max_w,
            rated_w=rotor.rated_w,
            tag=rotor.tag,
        )


class CoAxialRotor(Rotor):
    r"""A rotor object.

    This class will create a system of co-axial rotors with the shaft,
    disk, bearing and seal elements provided.

    Parameters
    ----------
    shafts : list of lists
        Each list of shaft elements builds a different shaft. The number of
        lists sets the number of shafts.
    disk_elements : list
        List with the disk elements
    bearing_elements : list
        List with the bearing elements
    point_mass_elements: list
        List with the point mass elements
    shaft_start_pos : list
        List indicating the initial node position for each shaft.
        Default is zero for each shaft created.
    tag : str
        A tag for the rotor

    Returns
    -------
    A rotor object.

    Attributes
    ----------
    nodes : list
        List of the model's nodes.
    nodes_pos : list
        List with nodal spatial location.
    CG : float
        Center of gravity

    Examples
    --------
    >>> import ross as rs
    >>> steel = rs.materials.steel
    >>> i_d = 0
    >>> o_d = 0.05
    >>> n = 10
    >>> L = [0.25 for _ in range(n)]
    >>> axial_shaft = [rs.ShaftElement(l, i_d, o_d, material=steel) for l in L]
    >>> i_d = 0.15
    >>> o_d = 0.20
    >>> n = 6
    >>> L = [0.25 for _ in range(n)]
    >>> coaxial_shaft = [rs.ShaftElement(l, i_d, o_d, material=steel) for l in L]
    >>> shaft = [axial_shaft, coaxial_shaft]
    >>> disk0 = rs.DiskElement.from_geometry(n=1,
    ...                                     material=steel,
    ...                                     width=0.07,
    ...                                     i_d=0.05,
    ...                                     o_d=0.28)
    >>> disk1 = rs.DiskElement.from_geometry(n=9,
    ...                                     material=steel,
    ...                                     width=0.07,
    ...                                     i_d=0.05,
    ...                                     o_d=0.28)
    >>> disk2 = rs.DiskElement.from_geometry(n=13,
    ...                                      material=steel,
    ...                                      width=0.07,
    ...                                      i_d=0.20,
    ...                                      o_d=0.48)
    >>> disk3 = rs.DiskElement.from_geometry(n=15,
    ...                                      material=steel,
    ...                                      width=0.07,
    ...                                      i_d=0.20,
    ...                                      o_d=0.48)
    >>> disks = [disk0, disk1, disk2, disk3]
    >>> stfx = 1e6
    >>> stfy = 0.8e6
    >>> bearing0 = rs.BearingElement(0, kxx=stfx, kyy=stfy, cxx=0)
    >>> bearing1 = rs.BearingElement(10, kxx=stfx, kyy=stfy, cxx=0)
    >>> bearing2 = rs.BearingElement(11, kxx=stfx, kyy=stfy, cxx=0)
    >>> bearing3 = rs.BearingElement(8, n_link=17, kxx=stfx, kyy=stfy, cxx=0)
    >>> bearings = [bearing0, bearing1, bearing2, bearing3]
    >>> rotor = rs.CoAxialRotor(shaft, disks, bearings)
    """

    def __init__(
        self,
        shafts,
        disk_elements=None,
        bearing_elements=None,
        point_mass_elements=None,
        min_w=None,
        max_w=None,
        rated_w=None,
        tag=None,
    ):
        self.parameters = {"min_w": min_w, "max_w": max_w, "rated_w": rated_w}
        if tag is None:
            self.tag = "Rotor 0"

        ####################################################
        # Config attributes
        ####################################################

        # operational speeds
        self.min_w = min_w
        self.max_w = max_w
        self.rated_w = rated_w

        ####################################################

        # set n for each shaft element
        aux_n = 0
        aux_n_tag = 0
        for j, shaft in enumerate(shafts):
            for i, sh in enumerate(shaft):
                if sh.n is None:
                    sh.n = i + aux_n
                if sh.tag is None:
                    sh.tag = sh.__class__.__name__ + " " + str(i + aux_n_tag)
            aux_n = shaft[-1].n_r + 1
            aux_n_tag = aux_n - 1 - j

        # flatten and make a copy for shaft elements to avoid altering
        # attributes for elements that might be used in different rotors
        # e.g. altering shaft_element.n
        shafts = [copy(sh) for sh in shafts]
        shaft_elements = list(chain(*shafts))

        if disk_elements is None:
            disk_elements = []
        if bearing_elements is None:
            bearing_elements = []
        if point_mass_elements is None:
            point_mass_elements = []

        for i, disk in enumerate(disk_elements):
            if disk.tag is None:
                disk.tag = "Disk " + str(i)

        for i, brg in enumerate(bearing_elements):
            brg.n_l = brg.n
            brg.n_r = brg.n
            if brg.__class__.__name__ == "BearingElement" and brg.tag is None:
                brg.tag = "Bearing " + str(i)
            if brg.__class__.__name__ == "SealElement" and brg.tag is None:
                brg.tag = "Seal " + str(i)

        for i, p_mass in enumerate(point_mass_elements):
            if p_mass.tag is None:
                p_mass.tag = "Point Mass " + str(i)

        self.shafts = shafts
        self.shaft_elements = sorted(shaft_elements, key=lambda el: el.n)
        self.bearing_elements = sorted(bearing_elements, key=lambda el: el.n)
        self.disk_elements = disk_elements
        self.point_mass_elements = point_mass_elements
        self.elements = list(
            chain(
                *[
                    self.shaft_elements,
                    self.disk_elements,
                    self.bearing_elements,
                    self.point_mass_elements,
                ]
            )
        )
        self.number_dof = self._check_number_dof()

        ####################################################
        # Rotor summary
        ####################################################
        columns = [
            "type",
            "n",
            "n_link",
            "L",
            "node_pos",
            "node_pos_r",
            "idl",
            "odl",
            "idr",
            "odr",
            "i_d",
            "o_d",
            "beam_cg",
            "axial_cg_pos",
            "y_pos",
            "material",
            "rho",
            "volume",
            "m",
            "tag",
        ]

        df_shaft = pd.DataFrame([el.summary() for el in self.shaft_elements])
        df_disks = pd.DataFrame([el.summary() for el in self.disk_elements])
        df_bearings = pd.DataFrame(
            [
                el.summary()
                for el in self.bearing_elements
                if not isinstance(el, SealElement)
            ]
        )
        df_seals = pd.DataFrame(
            [
                el.summary()
                for el in self.bearing_elements
                if isinstance(el, SealElement)
            ]
        )
        df_point_mass = pd.DataFrame([el.summary() for el in self.point_mass_elements])

        nodes_pos_l = np.zeros(len(df_shaft.n_l))
        nodes_pos_r = np.zeros(len(df_shaft.n_l))
        axial_cg_pos = np.zeros(len(df_shaft.n_l))
        shaft_number = np.zeros(len(df_shaft.n_l))

        i = 0
        for j, shaft in enumerate(self.shafts):
            for k, sh in enumerate(shaft):
                shaft_number[k + i] = j
                if k == 0:
                    nodes_pos_r[k + i] = df_shaft.loc[k + i, "L"]
                    axial_cg_pos[k + i] = sh.beam_cg + nodes_pos_l[k + i]
                    sh.axial_cg_pos = axial_cg_pos[k + i]
                if (
                    k > 0
                    and df_shaft.loc[k + i, "n_l"] == df_shaft.loc[k + i - 1, "n_l"]
                ):
                    nodes_pos_l[k + i] = nodes_pos_l[k + i - 1]
                    nodes_pos_r[k + i] = nodes_pos_r[k + i - 1]
                else:
                    nodes_pos_l[k + i] = nodes_pos_r[k + i - 1]
                    nodes_pos_r[k + i] = nodes_pos_l[k + i] + df_shaft.loc[k + i, "L"]

                if sh.n in df_bearings["n_link"].values:
                    idx = df_bearings.loc[df_bearings.n_link == sh.n, "n"].values[0]
                    nodes_pos_l[i : sh.n] += nodes_pos_l[idx] - nodes_pos_l[k + i]
                    nodes_pos_r[i : sh.n] += nodes_pos_r[idx] - nodes_pos_r[k + i]
                    axial_cg_pos[i : sh.n] += nodes_pos_r[idx] - nodes_pos_r[k + i]
                elif sh.n_r in df_bearings["n_link"].values:
                    idx = df_bearings.loc[df_bearings.n_link == sh.n_r, "n"].values[0]
                    nodes_pos_l[i : sh.n_r] += nodes_pos_l[idx - 1] - nodes_pos_l[k + i]
                    nodes_pos_r[i : sh.n_r] += nodes_pos_r[idx - 1] - nodes_pos_r[k + i]
                    axial_cg_pos[i : sh.n_r] += (
                        nodes_pos_r[idx - 1] - nodes_pos_r[k + i]
                    )

                axial_cg_pos[k + i] = sh.beam_cg + nodes_pos_l[k + i]
                sh.axial_cg_pos = axial_cg_pos[k + i]
            i += k + 1

        df_shaft["shaft_number"] = shaft_number
        df_shaft["nodes_pos_l"] = nodes_pos_l
        df_shaft["nodes_pos_r"] = nodes_pos_r
        df_shaft["axial_cg_pos"] = axial_cg_pos

        df = pd.concat(
            [df_shaft, df_disks, df_bearings, df_point_mass, df_seals], sort=True
        )
        df = df.sort_values(by="n_l")
        df = df.reset_index(drop=True)

        # check consistence for disks and bearings location
        if len(df_point_mass) > 0:
            max_loc_point_mass = df_point_mass.n.max()
        else:
            max_loc_point_mass = 0
        max_location = max(df_shaft.n_r.max(), max_loc_point_mass)
        if df.n_l.max() > max_location:
            raise ValueError("Trying to set disk or bearing outside shaft")

        # nodes axial position and diameter
        nodes_pos = list(df_shaft.groupby("n_l")["nodes_pos_l"].max())
        nodes_i_d = list(df_shaft.groupby("n_l")["i_d"].min())
        nodes_o_d = list(df_shaft.groupby("n_l")["o_d"].max())

        for i, shaft in enumerate(self.shafts):
            pos = shaft[-1].n_r
            if i < len(self.shafts) - 1:
                nodes_pos.insert(pos, df_shaft["nodes_pos_r"].iloc[pos - 1])
                nodes_i_d.insert(pos, df_shaft["i_d"].iloc[pos - 1])
                nodes_o_d.insert(pos, df_shaft["o_d"].iloc[pos - 1])
            else:
                nodes_pos.append(df_shaft["nodes_pos_r"].iloc[-1])
                nodes_i_d.append(df_shaft["i_d"].iloc[-1])
                nodes_o_d.append(df_shaft["o_d"].iloc[-1])

        self.nodes_pos = nodes_pos
        self.nodes_i_d = nodes_i_d
        self.nodes_o_d = nodes_o_d

        shaft_elements_length = list(df_shaft.groupby("n_l")["L"].min())
        self.shaft_elements_length = shaft_elements_length

        self.nodes = list(range(len(self.nodes_pos)))
        self.L = nodes_pos[-1]
        self.center_line_pos = [0] * len(self.nodes)

        # rotor mass can also be calculated with self.M()[::4, ::4].sum()
        self.m_disks = np.sum([disk.m for disk in self.disk_elements])
        self.m_shaft = np.sum([sh_el.m for sh_el in self.shaft_elements])
        self.m = self.m_disks + self.m_shaft

        # rotor center of mass and total inertia
        CG_sh = np.sum(
            [(sh.m * sh.axial_cg_pos) / self.m for sh in self.shaft_elements]
        )
        CG_dsk = np.sum(
            [disk.m * nodes_pos[disk.n] / self.m for disk in self.disk_elements]
        )
        self.CG = CG_sh + CG_dsk

        Ip_sh = np.sum([sh.Im for sh in self.shaft_elements])
        Ip_dsk = np.sum([disk.Ip for disk in self.disk_elements])
        self.Ip = Ip_sh + Ip_dsk

        # number of dofs
        half_ndof = self.number_dof / 2
        self.ndof = int(
            self.number_dof * (max([el.n for el in shaft_elements]) + 2)
            + half_ndof * len([el for el in point_mass_elements])
        )

        elm_no_shaft_id = {
            elm
            for elm in self.elements
            if pd.isna(df.loc[df.tag == elm.tag, "shaft_number"]).all()
        }
        for elm in cycle(self.elements):
            if elm_no_shaft_id:
                if elm in elm_no_shaft_id:
                    shnum_l = df.loc[
                        (df.n_l == elm.n) & (df.tag != elm.tag), "shaft_number"
                    ]
                    shnum_r = df.loc[
                        (df.n_r == elm.n) & (df.tag != elm.tag), "shaft_number"
                    ]
                    if len(shnum_l) == 0 and len(shnum_r) == 0:
                        shnum_l = df.loc[
                            (df.n_link == elm.n) & (df.tag != elm.tag), "shaft_number"
                        ]
                        shnum_r = shnum_l
                    if len(shnum_l):
                        df.loc[df.tag == elm.tag, "shaft_number"] = shnum_l.values[0]
                        elm_no_shaft_id.discard(elm)
                    elif len(shnum_r):
                        df.loc[df.tag == elm.tag, "shaft_number"] = shnum_r.values[0]
                        elm_no_shaft_id.discard(elm)
            else:
                break

        df_disks["shaft_number"] = df.loc[
            (df.type == "DiskElement"), "shaft_number"
        ].values
        df_bearings["shaft_number"] = df.loc[
            (df.type == "BearingElement"), "shaft_number"
        ].values
        df_seals["shaft_number"] = df.loc[
            (df.type == "SealElement"), "shaft_number"
        ].values
        df_point_mass["shaft_number"] = df.loc[
            (df.type == "PointMass"), "shaft_number"
        ].values

        self.df_disks = df_disks
        self.df_bearings = df_bearings
        self.df_shaft = df_shaft
        self.df_point_mass = df_point_mass
        self.df_seals = df_seals

        if "n_link" in df.columns and df_point_mass.index.size > 0:
            aux_link = list(df["n_link"].dropna().unique().astype(int))
            aux_node = list(df_point_mass["n"].dropna().unique().astype(int))
            self.link_nodes = list(set(aux_link) & set(aux_node))
        else:
            self.link_nodes = []

        # global indexes for dofs
        n_last = self.shaft_elements[-1].n
        for elm in self.elements:
            dof_mapping = elm.dof_mapping()
            global_dof_mapping = {}
            for k, v in dof_mapping.items():
                dof_letter, dof_number = k.split("_")
                global_dof_mapping[dof_letter + "_" + str(int(dof_number) + elm.n)] = (
                    int(v)
                )

            if elm.n <= n_last + 1:
                for k, v in global_dof_mapping.items():
                    global_dof_mapping[k] = int(self.number_dof * elm.n + v)
            else:
                for k, v in global_dof_mapping.items():
                    global_dof_mapping[k] = int(
                        half_ndof * n_last + half_ndof * elm.n + self.number_dof + v
                    )

            if hasattr(elm, "n_link") and elm.n_link is not None:
                if elm.n_link <= n_last + 1:
                    global_dof_mapping[f"x_{elm.n_link}"] = int(
                        self.number_dof * elm.n_link
                    )
                    global_dof_mapping[f"y_{elm.n_link}"] = int(
                        self.number_dof * elm.n_link + 1
                    )
                    global_dof_mapping[f"z_{elm.n_link}"] = int(
                        self.number_dof * elm.n_link + 2
                    )
                else:
                    global_dof_mapping[f"x_{elm.n_link}"] = int(
                        half_ndof * n_last + half_ndof * elm.n_link + self.number_dof
                    )
                    global_dof_mapping[f"y_{elm.n_link}"] = int(
                        half_ndof * n_last
                        + half_ndof * elm.n_link
                        + self.number_dof
                        + 1
                    )
                    global_dof_mapping[f"z_{elm.n_link}"] = int(
                        half_ndof * n_last
                        + half_ndof * elm.n_link
                        + self.number_dof
                        + 2
                    )

            elm.dof_global_index = global_dof_mapping
            df.at[df.loc[df.tag == elm.tag].index[0], "dof_global_index"] = (
                elm.dof_global_index
            )

        # define positions for disks
        for disk in disk_elements:
            z_pos = nodes_pos[disk.n]
            y_pos = nodes_o_d[disk.n]
            df.loc[df.tag == disk.tag, "nodes_pos_l"] = z_pos
            df.loc[df.tag == disk.tag, "nodes_pos_r"] = z_pos
            df.loc[df.tag == disk.tag, "y_pos"] = y_pos

        # define positions for bearings
        # check if there are bearings without location
        bearings_no_zloc = {
            b
            for b in bearing_elements
            if pd.isna(df.loc[df.tag == b.tag, "nodes_pos_l"]).all()
        }

        # cycle while there are bearings without a z location
        for b in cycle(self.bearing_elements):
            if bearings_no_zloc:
                if b in bearings_no_zloc:
                    # first check if b.n is on list, if not, check for n_link
                    node_l = df.loc[(df.n_l == b.n) & (df.tag != b.tag), "nodes_pos_l"]
                    node_r = df.loc[(df.n_r == b.n) & (df.tag != b.tag), "nodes_pos_r"]
                    if len(node_l) == 0 and len(node_r) == 0:
                        node_l = df.loc[
                            (df.n_link == b.n) & (df.tag != b.tag), "nodes_pos_l"
                        ]
                        node_r = node_l
                    if len(node_l):
                        df.loc[df.tag == b.tag, "nodes_pos_l"] = node_l.values[0]
                        df.loc[df.tag == b.tag, "nodes_pos_r"] = node_l.values[0]
                        bearings_no_zloc.discard(b)
                    elif len(node_r):
                        df.loc[df.tag == b.tag, "nodes_pos_l"] = node_r.values[0]
                        df.loc[df.tag == b.tag, "nodes_pos_r"] = node_r.values[0]
                        bearings_no_zloc.discard(b)
            else:
                break

        dfb = df[df.type == "BearingElement"]
        z_positions = [pos for pos in dfb["nodes_pos_l"]]
        z_positions = list(dict.fromkeys(z_positions))
        mean_od = np.mean(nodes_o_d)
        for z_pos in dfb["nodes_pos_l"]:
            dfb_z_pos = dfb[dfb.nodes_pos_l == z_pos]
            dfb_z_pos = dfb_z_pos.sort_values(by="n_l")
            for n, t, nlink in zip(dfb_z_pos.n, dfb_z_pos.tag, dfb_z_pos.n_link):
                if n in self.nodes:
                    if z_pos == df_shaft["nodes_pos_l"].iloc[0]:
                        y_pos = (np.max(df_shaft["odl"][df_shaft.n_l == n].values)) / 2
                    elif z_pos == df_shaft["nodes_pos_r"].iloc[-1]:
                        y_pos = (np.max(df_shaft["odr"][df_shaft.n_r == n].values)) / 2
                    else:
                        if not len(df_shaft["odl"][df_shaft._n == n].values):
                            y_pos = (
                                np.max(df_shaft["odr"][df_shaft._n == n - 1].values)
                            ) / 2
                        elif not len(df_shaft["odr"][df_shaft._n == n - 1].values):
                            y_pos = (
                                np.max(df_shaft["odl"][df_shaft._n == n].values)
                            ) / 2
                        else:
                            y_pos = (
                                np.max(
                                    [
                                        np.max(
                                            df_shaft["odl"][df_shaft._n == n].values
                                        ),
                                        np.max(
                                            df_shaft["odr"][df_shaft._n == n - 1].values
                                        ),
                                    ]
                                )
                                / 2
                            )
                else:
                    y_pos += 2 * mean_od * df["scale_factor"][df.tag == t].values[0]

                if nlink in self.nodes:
                    if z_pos == df_shaft["nodes_pos_l"].iloc[0]:
                        y_pos_sup = (
                            np.min(df_shaft["idl"][df_shaft.n_l == nlink].values)
                        ) / 2
                    elif z_pos == df_shaft["nodes_pos_r"].iloc[-1]:
                        y_pos_sup = (
                            np.min(df_shaft["idr"][df_shaft.n_r == nlink].values)
                        ) / 2
                    else:
                        if not len(df_shaft["idl"][df_shaft._n == nlink].values):
                            y_pos_sup = (
                                np.min(df_shaft["idr"][df_shaft._n == nlink - 1].values)
                            ) / 2
                        elif not len(df_shaft["idr"][df_shaft._n == nlink - 1].values):
                            y_pos_sup = (
                                np.min(df_shaft["idl"][df_shaft._n == nlink].values)
                            ) / 2
                        else:
                            y_pos_sup = (
                                np.min(
                                    [
                                        np.min(
                                            df_shaft["idl"][df_shaft._n == nlink].values
                                        ),
                                        np.min(
                                            df_shaft["idr"][
                                                df_shaft._n == nlink - 1
                                            ].values
                                        ),
                                    ]
                                )
                                / 2
                            )
                else:
                    y_pos_sup = (
                        y_pos + 2 * mean_od * df["scale_factor"][df.tag == t].values[0]
                    )

                df.loc[df.tag == t, "y_pos"] = y_pos
                df.loc[df.tag == t, "y_pos_sup"] = y_pos_sup

        # define position for point mass elements
        dfb = df[df.type == "BearingElement"]
        for p in point_mass_elements:
            z_pos = dfb[dfb.n_l == p.n]["nodes_pos_l"].values[0]
            y_pos = dfb[dfb.n_l == p.n]["y_pos"].values[0]
            df.loc[df.tag == p.tag, "nodes_pos_l"] = z_pos
            df.loc[df.tag == p.tag, "nodes_pos_r"] = z_pos
            df.loc[df.tag == p.tag, "y_pos"] = y_pos

        self.df = df

        # Build matrices considering all elements excluding bearing_elements:
        M0 = np.zeros((self.ndof, self.ndof))
        C0 = np.zeros((self.ndof, self.ndof))
        K0 = np.zeros((self.ndof, self.ndof))
        G0 = np.zeros((self.ndof, self.ndof))
        Ksdt0 = np.zeros((self.ndof, self.ndof))

        elements = list(set(self.elements).difference(self.bearing_elements))

        for elm in elements:
            dofs = list(elm.dof_global_index.values())

            M0[np.ix_(dofs, dofs)] += elm.M()
            C0[np.ix_(dofs, dofs)] += elm.C()
            K0[np.ix_(dofs, dofs)] += elm.K()
            G0[np.ix_(dofs, dofs)] += elm.G()

            if elm in self.shaft_elements:
                Ksdt0[np.ix_(dofs, dofs)] += elm.Kst()
            elif elm in self.disk_elements:
                Ksdt0[np.ix_(dofs, dofs)] += elm.Kdt()

        self.M0 = M0
        self.C0 = C0
        self.K0 = K0
        self.G0 = G0
        self.Ksdt0 = Ksdt0


def rotor_example():
    """Create a rotor as example.

    This function returns an instance of a simple rotor without
    damping with 6 shaft elements, 2 disks and 2 simple bearings.
    The purpose of this is to make available a simple model
    so that doctest can be written using this.

    Returns
    -------
    An instance of a rotor object.

    Examples
    --------
    >>> rotor = rotor_example()
    >>> modal = rotor.run_modal(speed=0)
    >>> np.round(modal.wd[:4])
    array([ 92.,  96., 275., 297.])
    """
    i_d = 0
    o_d = 0.05
    n = 6
    L = [0.25 for _ in range(n)]

    shaft_elem = [
        ShaftElement(
            l,
            i_d,
            o_d,
            material=steel,
            shear_effects=True,
            rotary_inertia=True,
            gyroscopic=True,
        )
        for l in L
    ]

    disk0 = DiskElement.from_geometry(
        n=2, material=steel, width=0.07, i_d=0.05, o_d=0.28
    )
    disk1 = DiskElement.from_geometry(
        n=4, material=steel, width=0.07, i_d=0.05, o_d=0.28
    )

    stfx = 1e6
    stfy = 0.8e6
    bearing0 = BearingElement(0, kxx=stfx, kyy=stfy, cxx=0)
    bearing1 = BearingElement(6, kxx=stfx, kyy=stfy, cxx=0)

    return Rotor(shaft_elem, [disk0, disk1], [bearing0, bearing1])


def compressor_example():
    """Create a compressor as example.

    This function returns an instance of a rotor with
    91 shaft elements, 7 disks and 2 simple bearings and 12 seals.

    Returns
    -------
    An instance of a rotor object.

    References
    ----------
    Timbó, R., Ritto, T. G. (2019). Impact of damper seal coefficients uncertainties
    in rotor dynamics. Journal of the Brazilian Society of Mechanical Sciences and
    Engineering, 41(4),165. doi: 10.1007/s40430-019-1652-8

    Examples
    --------
    >>> rotor = compressor_example()
    >>> len(rotor.shaft_elements)
    91
    >>> len(rotor.disk_elements)
    7
    >>> len(rotor.bearing_elements)
    14
    """
    compressor_dir = Path(__file__).parent / "tests/data/compressor_example.toml"

    return Rotor.load(compressor_dir)


def coaxrotor_example():
    """Create a coaxial rotor as example.

    This function returns an instance of a coaxial rotor with
    2 shafts, 4 disk and 4 bearings.

    Returns
    -------
    An instance of a rotor object.

    Examples
    --------
    >>> rotor = coaxrotor_example()
    >>> modal = rotor.run_modal(speed=0)
    >>> np.round(modal.wd[:4])
    array([39., 39., 99., 99.])
    """
    i_d = 0
    o_d = 0.05
    n = 10
    L = [0.25 for _ in range(n)]

    axial_shaft = [ShaftElement(l, i_d, o_d, material=steel) for l in L]

    i_d = 0.25
    o_d = 0.30
    n = 6
    L = [0.25 for _ in range(n)]

    coaxial_shaft = [ShaftElement(l, i_d, o_d, material=steel) for l in L]

    disk0 = DiskElement.from_geometry(
        n=1, material=steel, width=0.07, i_d=0.05, o_d=0.28, scale_factor=0.8
    )
    disk1 = DiskElement.from_geometry(
        n=9, material=steel, width=0.07, i_d=0.05, o_d=0.28, scale_factor=0.8
    )
    disk2 = DiskElement.from_geometry(
        n=13, material=steel, width=0.07, i_d=0.20, o_d=0.48, scale_factor=0.8
    )
    disk3 = DiskElement.from_geometry(
        n=15, material=steel, width=0.07, i_d=0.20, o_d=0.48, scale_factor=0.8
    )

    shaft = [axial_shaft, coaxial_shaft]
    disks = [disk0, disk1, disk2, disk3]

    stfx = 1e6
    stfy = 1e6
    bearing0 = BearingElement(0, n_link=18, kxx=stfx, kyy=stfy, cxx=0, scale_factor=0.4)
    bearing1 = BearingElement(
        10, n_link=19, kxx=stfx, kyy=stfy, cxx=0, scale_factor=0.4
    )
    bearing2 = BearingElement(11, kxx=stfx, kyy=stfy, cxx=0, scale_factor=0.4)
    bearing3 = BearingElement(8, n_link=17, kxx=stfx, kyy=stfy, cxx=0, scale_factor=0.4)

    base0 = BearingElement(18, kxx=1e8, kyy=1e8, cxx=0, scale_factor=0.4)
    base1 = BearingElement(19, kxx=1e8, kyy=1e8, cxx=0, scale_factor=0.4)

    pointmass0 = PointMass(n=18, m=20)
    pointmass1 = PointMass(n=19, m=20)

    bearings = [bearing0, bearing1, bearing2, bearing3, base0, base1]
    pointmasses = [pointmass0, pointmass1]

    return CoAxialRotor(shaft, disks, bearings, pointmasses)


def rotor_example_6dof():
    """Create a rotor as example.

    This function returns an instance of a simple rotor with
    6 shaft elements, 2 disks and 2 bearings with stiffness in
    the z direction.

    Returns
    -------
    An instance of a rotor object.

    Examples
    --------
    >>> import ross as rs
    >>> import numpy as np
    >>> rotor = rs.rotor_example_6dof()

    Plotting rotor model
    >>> fig = rotor.plot_rotor()
    >>> # fig.show()

    Running modal
    >>> rotor_speed = 100.0 # rad/s
    >>> modal = rotor.run_modal(rotor_speed)
    >>> print(f"Undamped natural frequencies: {np.round(modal.wn, 2)}") # doctest: +ELLIPSIS
    Undamped natural frequencies: [ 47.62  91.84  96.36 274.44 ...
    >>> print(f"Damped natural frequencies: {np.round(modal.wd, 2)}") # doctest: +ELLIPSIS
    Damped natural frequencies: [ 47.62  91.84  96.36 274.44 ...

    Plotting Campbell Diagram
    >>> camp = rotor.run_campbell(np.linspace(0, 400, 101), frequencies=6)
    >>> fig = camp.plot()
    >>> # fig.show()
    """
    i_d = 0
    o_d = 0.05
    n = 6
    L = [0.25 for _ in range(n)]

    shaft_elem = [
        ShaftElement(
            l,
            i_d,
            o_d,
            material=steel,
            alpha=0,
            beta=0,
            rotary_inertia=False,
            shear_effects=False,
        )
        for l in L
    ]

    disk0 = DiskElement.from_geometry(
        n=2, material=steel, width=0.07, i_d=0.05, o_d=0.28
    )
    disk1 = DiskElement.from_geometry(
        n=4, material=steel, width=0.07, i_d=0.05, o_d=0.28
    )

    kxx = 1e6
    kyy = 0.8e6
    kzz = 0.1e6
    bearing0 = BearingElement(n=0, kxx=kxx, kyy=kyy, kzz=kzz, cxx=0, cyy=0, czz=0)
    bearing1 = BearingElement(n=6, kxx=kxx, kyy=kyy, kzz=kzz, cxx=0, cyy=0, czz=0)

    return Rotor(shaft_elem, [disk0, disk1], [bearing0, bearing1])


def rotor_example_with_damping():
    """Create a rotor as example.

    This function returns an instance of a rotor with internal
    damping, with 33 shaft elements, 2 disks and 2 bearings.

    Returns
    -------
    An instance of a rotor object.

    Examples
    --------
    >>> rotor = rotor_example_with_damping()
    >>> rotor.Ip
    0.015118294226367068
    """
    steel2 = Material(name="Steel", rho=7850, E=2.17e11, G_s=81.2e9)

    # fmt: off
    node_position = np.array([
        0  ,  25,  64, 104, 124, 143, 175, 207, 239, 271, 303, 335, 
        345, 355, 380, 408, 436, 466, 496, 526, 556, 586, 614, 647,
        657, 667, 702, 737, 772, 807, 842, 862, 881, 914
    ]) * 1e-3
    # fmt: on

    L = [node_position[i] - node_position[i - 1] for i in range(1, len(node_position))]

    i_d = 0
    o_d = 0.019

    shaft_elem = [
        ShaftElement(
            l,
            i_d,
            o_d,
            material=steel2,
            alpha=8.0501,
            beta=1.0e-5,
            rotary_inertia=True,
            shear_effects=True,
        )
        for l in L
    ]

    m = 2.6375
    Id = 0.003844540885417
    Ip = 0.007513248437500

    disk0 = DiskElement(n=12, m=m, Id=Id, Ip=Ip)
    disk1 = DiskElement(n=24, m=m, Id=Id, Ip=Ip)

    bearing0 = BearingElement(
        n=4, kxx=4.40e5, kyy=4.6114e5, cxx=27.4, cyy=2.505, kzz=0, czz=0
    )
    bearing1 = BearingElement(
        n=31, kxx=9.50e5, kyy=1.09e8, cxx=50.4, cyy=100.4553, kzz=0, czz=0
    )

    return Rotor(shaft_elem, [disk0, disk1], [bearing0, bearing1])


def rotor_amb_example():
    r"""This function creates the model of a test rig rotor supported by magnetic bearings.
    Details of the model can be found at doi.org/10.14393/ufu.di.2015.186.

    Returns
    -------
    Rotor object.
    """

    from ross.materials import Material

    steel_amb = Material(name="Steel", rho=7850, E=2e11, Poisson=0.3)

    # Shaft elements:
    # fmt: off
    Li = [
        0.0, 0.012, 0.032, 0.052, 0.072, 0.092, 0.112, 0.1208, 0.12724,
        0.13475, 0.14049, 0.14689, 0.15299, 0.159170, 0.16535, 0.180350,
        0.1905, 0.2063, 0.2221, 0.2379, 0.2537, 0.2695, 0.2853, 0.3011,
        0.3169, 0.3327, 0.3363, 0.3485, 0.361, 0.3735, 0.3896, 0.4057,
        0.4218, 0.4379, 0.454, 0.4701, 0.4862, 0.5023, 0.5184, 0.5345,
        0.54465, 0.559650, 0.565830, 0.572010, 0.57811, 0.58451, 0.590250,
        0.59776, 0.6042, 0.613, 0.633, 0.645,
    ]
    Li = [round(i, 4) for i in Li]
    L = [Li[i + 1] - Li[i] for i in range(len(Li) - 1)]
    i_d = [0.0 for i in L]
    o_d1 = [0.0 for i in L]
    o_d1[0] = 6.35
    o_d1[1:5] = [32 for i in range(4)]
    o_d1[5:14] = [34.8 for i in range(9)]
    o_d1[14:16] = [1.2 * 49.9 for i in range(2)]
    o_d1[16:27] = [19.05 for i in range(11)]
    o_d1[27:29] = [0.8 * 49.9 for i in range(2)]
    o_d1[29:39] = [19.05 for i in range(10)]
    o_d1[39:41] = [1.2 * 49.9 for i in range(2)]
    o_d1[41:49] = [34.8 for i in range(8)]
    o_d1[49] = 34.8
    o_d1[50] = 6.35
    o_d = [i * 1e-3 for i in o_d1]

    shaft_elements = [
        ShaftElement(
            L=l,
            idl=idl,
            odl=odl,
            material=steel_amb,
            shear_effects=True,
            rotary_inertia=True,
            gyroscopic=True,
        )
        for l, idl, odl in zip(L, i_d, o_d)
    ]

    # Disk elements:
    n_list = [6, 7, 8, 9, 10, 11, 12, 13, 27, 29, 41, 42, 43, 44, 45, 46, 47, 48]
    width = [
        0.0088, 0.0064, 0.0075, 0.0057,
        0.0064, 0.0061, 0.0062, 0.0062,
        0.0124, 0.0124, 0.0062, 0.0062,
        0.0061, 0.0064, 0.0057, 0.0075,
        0.0064, 0.0088,
    ]
    o_disc = [
        0.0249, 0.0249, 0.0249, 0.0249,
        0.0249, 0.0249, 0.0249, 0.0249,
        0.0600, 0.0600, 0.0249, 0.0249,
        0.0249, 0.0249, 0.0249, 0.0249,
        0.0249, 0.0249,
    ]
    i_disc = [
        0.0139, 0.0139, 0.0139, 0.0139,
        0.0139, 0.0139, 0.0139, 0.0139,
        0.0200, 0.0200, 0.0139, 0.0139,
        0.0139, 0.0139, 0.0139, 0.0139,
        0.0139, 0.0139,
    ]
    # fmt: on
    m_list = [
        np.pi * 7850 * w * ((odisc) ** 2 - (idisc) ** 2)
        for w, odisc, idisc in zip(width, o_disc, i_disc)
    ]
    Id_list = [
        m / 12 * (3 * idisc**2 + 3 * odisc**2 + w**2)
        for m, idisc, odisc, w in zip(m_list, i_disc, o_disc, width)
    ]
    Ip_list = [
        m / 2 * (idisc**2 + odisc**2) for m, idisc, odisc in zip(m_list, i_disc, o_disc)
    ]

    disk_elements = [
        DiskElement(
            n=n,
            m=m,
            Id=Id,
            Ip=Ip,
        )
        for n, m, Id, Ip in zip(n_list, m_list, Id_list, Ip_list)
    ]

    # Bearing elements:
    n_list = [12, 43]
    u0 = 4 * np.pi * 1e-7
    n = 200
    A = 1e-4
    i0 = 1.0
    s0 = 1e-3
    alpha = 0.392
    Kp = 1000
    Ki = 0
    Kd = 5
    k_amp = 1.0
    k_sense = 1.0
    bearing_elements = [
        MagneticBearingElement(
            n=n_list[0],
            g0=s0,
            i0=i0,
            ag=A,
            nw=n,
            alpha=alpha,
            k_amp=k_amp,
            k_sense=k_sense,
            kp_pid=Kp,
            kd_pid=Kd,
            ki_pid=Ki,
            tag="Magnetic Bearing 0",
        ),
        MagneticBearingElement(
            n=n_list[1],
            g0=s0,
            i0=i0,
            ag=A,
            nw=n,
            alpha=alpha,
            k_amp=k_amp,
            k_sense=k_sense,
            kp_pid=Kp,
            kd_pid=Kd,
            ki_pid=Ki,
            tag="Magnetic Bearing 1",
        ),
    ]

    return Rotor(shaft_elements, disk_elements, bearing_elements)
