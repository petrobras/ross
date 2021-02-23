# fmt: off
import inspect
import sys
import warnings
from collections import Counter, namedtuple
from collections.abc import Iterable
from copy import copy, deepcopy
from itertools import chain, cycle
from pathlib import Path

import numpy as np
import pandas as pd
import toml
from plotly import express as px
from plotly import graph_objects as go
from scipy import io as sio
from scipy import linalg as la
from scipy import signal as signal
from scipy.interpolate import UnivariateSpline
from scipy.optimize import newton
from scipy.sparse import linalg as las

from ross.bearing_seal_element import (BallBearingElement, BearingElement,
                                       BearingElement6DoF, BearingFluidFlow,
                                       MagneticBearingElement,
                                       RollerBearingElement, SealElement)
from ross.defects import Crack, MisalignmentFlex, MisalignmentRigid, Rubbing
from ross.disk_element import DiskElement, DiskElement6DoF
from ross.materials import steel
from ross.point_mass import PointMass
from ross.results import (CampbellResults, ConvergenceResults,
                          CriticalSpeedResults, ForcedResponseResults,
                          FrequencyResponseResults, Level1Results,
                          ModalResults, StaticResults, SummaryResults,
                          TimeResponseResults, UCSResults)
from ross.shaft_element import ShaftElement, ShaftElement6DoF
from ross.units import Q_, check_units
from ross.utils import intersection

# fmt: on

__all__ = ["Rotor", "CoAxialRotor", "rotor_example", "coaxrotor_example"]

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
        self.tag = "Rotor 0" if tag is None else tag

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
            if sh.tag is None:
                sh.tag = sh.__class__.__name__ + " " + str(i)

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
            if not isinstance(brg, SealElement) and brg.tag is None:
                brg.tag = "Bearing " + str(i)
            elif isinstance(brg, SealElement) and brg.tag is None:
                brg.tag = "Seal " + str(i)

        for i, p_mass in enumerate(point_mass_elements):
            if p_mass.tag is None:
                p_mass.tag = "Point Mass " + str(i)

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
        nodes_pos = list(df_shaft.groupby("n_l")["nodes_pos_l"].max())
        nodes_pos.append(df_shaft["nodes_pos_r"].iloc[-1])
        self.nodes_pos = nodes_pos

        nodes_i_d = list(df_shaft.groupby("n_l")["i_d"].min())
        nodes_i_d.append(df_shaft["i_d"].iloc[-1])
        self.nodes_i_d = nodes_i_d

        nodes_o_d = list(df_shaft.groupby("n_l")["o_d"].max())
        nodes_o_d.append(df_shaft["o_d"].iloc[-1])
        self.nodes_o_d = nodes_o_d

        shaft_elements_length = list(df_shaft.groupby("n_l")["L"].min())
        self.shaft_elements_length = shaft_elements_length

        self.nodes = list(range(len(self.nodes_pos)))
        self.L = nodes_pos[-1]

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
            [disk.m * nodes_pos[disk.n] / self.m for disk in self.disk_elements]
        )
        self.CG = CG_sh + CG_dsk

        Ip_sh = np.sum([sh.Im for sh in self.shaft_elements])
        Ip_dsk = np.sum([disk.Ip for disk in self.disk_elements])
        self.Ip = Ip_sh + Ip_dsk

        self._v0 = None  # used to call eigs

        # number of dofs
        self.ndof = int(
            self.number_dof * max([el.n for el in shaft_elements])
            + self.number_dof * 2
            + 2 * len([el for el in point_mass_elements])
        )

        # global indexes for dofs
        n_last = self.shaft_elements[-1].n
        for elm in self.elements:
            dof_mapping = elm.dof_mapping()
            global_dof_mapping = {}
            for k, v in dof_mapping.items():
                dof_letter, dof_number = k.split("_")
                global_dof_mapping[
                    dof_letter + "_" + str(int(dof_number) + elm.n)
                ] = int(v)
            dof_tuple = namedtuple("GlobalIndex", global_dof_mapping)

            if elm.n <= n_last + 1:
                for k, v in global_dof_mapping.items():
                    global_dof_mapping[k] = int(self.number_dof * elm.n + v)
            else:
                for k, v in global_dof_mapping.items():
                    global_dof_mapping[k] = int(
                        2 * n_last + self.number_dof / 2 * elm.n + self.number_dof + v
                    )

            if hasattr(elm, "n_link") and elm.n_link is not None:
                if elm.n_link <= n_last + 1:
                    global_dof_mapping[f"x_{elm.n_link}"] = int(
                        self.number_dof * elm.n_link
                    )
                    global_dof_mapping[f"y_{elm.n_link}"] = int(
                        self.number_dof * elm.n_link + 1
                    )
                else:
                    global_dof_mapping[f"x_{elm.n_link}"] = int(
                        2 * n_last + 2 * elm.n_link + self.number_dof
                    )
                    global_dof_mapping[f"y_{elm.n_link}"] = int(
                        2 * n_last + 2 * elm.n_link + self.number_dof + 1
                    )

            dof_tuple = namedtuple("GlobalIndex", global_dof_mapping)
            elm.dof_global_index = dof_tuple(**global_dof_mapping)
            df.at[
                df.loc[df.tag == elm.tag].index[0], "dof_global_index"
            ] = elm.dof_global_index

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

        classes = [
            _class
            for _class, _ in inspect.getmembers(
                sys.modules["ross.bearing_seal_element"], inspect.isclass
            )
        ]
        dfb = df[df.type.isin(classes)]
        z_positions = [pos for pos in dfb["nodes_pos_l"]]
        z_positions = list(dict.fromkeys(z_positions))
        for z_pos in z_positions:
            dfb_z_pos = dfb[dfb.nodes_pos_l == z_pos]
            dfb_z_pos = dfb_z_pos.sort_values(by="n_l")
            if z_pos == df_shaft["nodes_pos_l"].iloc[0]:
                y_pos = (
                    max(
                        df_shaft["odl"][
                            df_shaft.n_l == int(dfb_z_pos.iloc[0]["n_l"])
                        ].values
                    )
                    / 2
                )
            elif z_pos == df_shaft["nodes_pos_r"].iloc[-1]:
                y_pos = (
                    max(
                        df_shaft["odr"][
                            df_shaft.n_r == int(dfb_z_pos.iloc[0]["n_r"])
                        ].values
                    )
                    / 2
                )
            else:
                y_pos = (
                    max(
                        [
                            max(
                                df_shaft["odl"][
                                    df_shaft._n == int(dfb_z_pos.iloc[0]["n_l"])
                                ].values
                            ),
                            max(
                                df_shaft["odr"][
                                    df_shaft._n == int(dfb_z_pos.iloc[0]["n_l"]) - 1
                                ].values
                            ),
                        ]
                    )
                    / 2
                )
            mean_od = np.mean(nodes_o_d)
            scale_size = dfb["scale_factor"] * mean_od
            y_pos_sup = y_pos + 2 * scale_size

            for t in dfb_z_pos.tag:
                df.loc[df.tag == t, "y_pos"] = y_pos
                df.loc[df.tag == t, "y_pos_sup"] = y_pos_sup
                y_pos += 2 * mean_od * df["scale_factor"][df.tag == t].values[0]
                y_pos_sup += 2 * mean_od * df["scale_factor"][df.tag == t].values[0]

        # define position for point mass elements
        dfb = df[df.type.isin(classes)]
        for p in point_mass_elements:
            z_pos = dfb[dfb.n_l == p.n]["nodes_pos_l"].values[0]
            y_pos = dfb[dfb.n_l == p.n]["y_pos"].values[0]
            df.loc[df.tag == p.tag, "nodes_pos_l"] = z_pos
            df.loc[df.tag == p.tag, "nodes_pos_r"] = z_pos
            df.loc[df.tag == p.tag, "y_pos"] = y_pos

        self.df = df

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

    def run_modal(self, speed, num_modes=12, sparse=True):
        """Run modal analysis.

        Method to calculate eigenvalues and eigvectors for a given rotor system.
        Tthe natural frequencies and dampings ratios are calculated for a given
        rotor speed. It means that for each speed input there's a different set of
        eigenvalues and eigenvectors, hence, different natural frequencies and damping
        ratios are returned.
        This method will return a ModalResults object which stores all data generated
        and also provides so methods for plotting.

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
            If False, scipy.linalg.eig() is used to calculate all the eigenvalues and
            eigenvectors.
            Default is True.

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
        evalues, evectors = self._eigen(speed, num_modes=num_modes, sparse=sparse)
        wn_len = num_modes // 2
        wn = (np.absolute(evalues))[:wn_len]
        wd = (np.imag(evalues))[:wn_len]
        damping_ratio = (-np.real(evalues) / np.absolute(evalues))[:wn_len]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            log_dec = 2 * np.pi * damping_ratio / np.sqrt(1 - damping_ratio ** 2)

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
        )

        return modal_results

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
        speed_range : tuple
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
        CriticalSpeedResults : An instance of CriticalSpeedResults class, which is
        used to post-process results. Attributes stored:
            CriticalSpeedResults.wn() : undamped critical speeds.
            CriticalSpeedResults.wd(): damped critical speeds.
            CriticalSpeedResults.log_dec : log_dec for each critical speed.
            CriticalSpeedResults.damping_ratio : damping ratio for each critical speed.
            CriticalSpeedResults.whirl_direction : whirl dir. for each critical speed.

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
        array([271., 300., 636., 867.])

        Changing output units
        >>> np.round(results.wd("rpm"))
        array([2590., 2868., 6074., 8278.])

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

    def M(self):
        """Mass matrix for an instance of a rotor.

        Returns
        -------
        M0 : np.ndarray
            Mass matrix for the rotor.

        Examples
        --------
        >>> rotor = rotor_example()
        >>> rotor.M()[:4, :4]
        array([[ 1.42050794,  0.        ,  0.        ,  0.04931719],
               [ 0.        ,  1.42050794, -0.04931719,  0.        ],
               [ 0.        , -0.04931719,  0.00231392,  0.        ],
               [ 0.04931719,  0.        ,  0.        ,  0.00231392]])
        """
        M0 = np.zeros((self.ndof, self.ndof))

        for elm in self.elements:
            dofs = elm.dof_global_index
            M0[np.ix_(dofs, dofs)] += elm.M()

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
        >>> np.round(rotor.K(0)[:4, :4]/1e6)
        array([[47.,  0.,  0.,  6.],
               [ 0., 46., -6.,  0.],
               [ 0., -6.,  1.,  0.],
               [ 6.,  0.,  0.,  1.]])
        """
        K0 = np.zeros((self.ndof, self.ndof))

        for elm in self.elements:
            dofs = elm.dof_global_index
            try:
                K0[np.ix_(dofs, dofs)] += elm.K(frequency)
            except TypeError:
                K0[np.ix_(dofs, dofs)] += elm.K()

        return K0

    def Kst(self):
        """Dynamic stiffness matrix for an instance of a rotor.

        Returns
        -------
        Kst0 : np.ndarray
            Dynamic stiffness matrix for the rotor.
            This matris IS OMEGA dependent
            Only useable to the 6 DoF model.

        Examples
        --------
        >>> rotor = rotor_example_6dof()
        >>> np.round(rotor.Kst()[:6, :6]*1e6)
        array([[     0., -23002.,      0.,   -479.,      0.,      0.],
               [     0.,      0.,      0.,      0.,      0.,      0.],
               [     0.,      0.,      0.,      0.,      0.,      0.],
               [     0.,      0.,      0.,      0.,      0.,      0.],
               [     0.,    479.,      0.,    160.,      0.,      0.],
               [     0.,      0.,      0.,      0.,      0.,      0.]])
        """

        Kst0 = np.zeros((self.ndof, self.ndof))

        if self.number_dof == 6:

            for elm in self.shaft_elements:
                dofs = elm.dof_global_index
                try:
                    Kst0[np.ix_(dofs, dofs)] += elm.Kst()
                except TypeError:
                    Kst0[np.ix_(dofs, dofs)] += elm.Kst()

        return Kst0

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
        >>> rotor = rotor_example()
        >>> rotor.C(0)[:4, :4]
        array([[0., 0., 0., 0.],
               [0., 0., 0., 0.],
               [0., 0., 0., 0.],
               [0., 0., 0., 0.]])
        """
        C0 = np.zeros((self.ndof, self.ndof))

        for elm in self.elements:
            dofs = elm.dof_global_index

            try:
                C0[np.ix_(dofs, dofs)] += elm.C(frequency)
            except TypeError:
                C0[np.ix_(dofs, dofs)] += elm.C()

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
        array([[ 0.        ,  0.01943344, -0.00022681,  0.        ],
               [-0.01943344,  0.        ,  0.        , -0.00022681],
               [ 0.00022681,  0.        ,  0.        ,  0.0001524 ],
               [ 0.        ,  0.00022681, -0.0001524 ,  0.        ]])
        """
        G0 = np.zeros((self.ndof, self.ndof))

        for elm in self.elements:
            dofs = elm.dof_global_index
            G0[np.ix_(dofs, dofs)] += elm.G()

        return G0

    def A(self, speed=0, frequency=None):
        """State space matrix for an instance of a rotor.

        Parameters
        ----------
        speed: float, optional
            Rotor speed.
            Default is 0.
        frequency : float, optional
            Excitation frequency. Default is rotor speed.

        Returns
        -------
        A : np.ndarray
            State space matrix for the rotor.

        Examples
        --------
        >>> rotor = rotor_example()
        >>> np.round(rotor.A()[50:56, :2])
        array([[     0.,  10927.],
               [-10924.,     -0.],
               [  -174.,      0.],
               [    -0.,   -174.],
               [    -0.,  10723.],
               [-10719.,     -0.]])
        """
        if frequency is None:
            frequency = speed

        Z = np.zeros((self.ndof, self.ndof))
        I = np.eye(self.ndof)

        # fmt: off
        A = np.vstack(
            [np.hstack([Z, I]),
             np.hstack([la.solve(-self.M(), self.K(frequency) + self.Kst()*speed), la.solve(-self.M(), (self.C(frequency) + self.G() * speed))])])
        # fmt: on

        return A

    def _check_frequency_array(self, frequency_range):
        """Verify if bearing elements coefficients are extrapolated.

        This method takes the frequency / speed range array applied to a particular
        method (run_campbell, run_freq_response) and checks if it's extrapolating the
        bearing rotordynamics coefficients.

        If any value of frequency_range argument is out of any bearing frequency
        parameter, the warning is raised.
        If none of the bearings has a frequency argument assinged, no warning will be
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
        # fmt: off
        for bearing in self.bearing_elements:
            if bearing.kxx.frequency is not None:
                if (np.max(frequency_range) > max(bearing.frequency) or
                    np.min(frequency_range) < min(bearing.frequency)):
                    warnings.warn(
                        "Extrapolating bearing coefficients. Be careful when post-processing the results."
                    )
                    break
        # fmt: on

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
        >>> evalues, evectors = rotor._eigen(0, sorted_=True)
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

        idx = np.array([positive, negative]).flatten()

        return idx

    def _eigen(
        self, speed, num_modes=12, frequency=None, sorted_=True, A=None, sparse=True
    ):
        """Calculate eigenvalues and eigenvectors.

        This method will return the eigenvalues and eigenvectors of the
        state space matrix A, sorted by the index method which considers
        the imaginary part (wd) of the eigenvalues for sorting.
        To avoid sorting use sorted_=False

        Parameters
        ----------
        speed : float
            Rotor speed.
        frequency: float
            Excitation frequency.
        sorted_ : bool, optional
            Sort considering the imaginary part (wd)
            Default is True
        A : np.array, optional
            Matrix for which eig will be calculated.
            Defaul is the rotor A matrix.
        sparse : bool, optional
            If sparse, eigenvalues will be calculated with arpack.
            Default is True.

        Returns
        -------
        evalues: array
            An array with the eigenvalues
        evectors array
            An array with the eigenvectors

        Examples
        --------
        >>> rotor = rotor_example()
        >>> evalues, evectors = rotor._eigen(0)
        >>> evalues[0].imag # doctest: +ELLIPSIS
        91.796...
        """
        if A is None:
            A = self.A(speed=speed, frequency=frequency)

        if sparse is True:
            try:
                evalues, evectors = las.eigs(
                    A, k=num_modes, sigma=0, ncv=2 * num_modes, which="LM", v0=self._v0
                )
                # store v0 as a linear combination of the previously
                # calculated eigenvectors to use in the next call to eigs
                self._v0 = np.real(sum(evectors.T))
            except las.ArpackError:
                evalues, evectors = la.eig(A)
        else:
            evalues, evectors = la.eig(A)

        if sorted_ is False:
            return evalues, evectors

        idx = self._index(evalues)

        return evalues[idx], evectors[:, idx]

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
        # fmt: off
        B = np.vstack([Z,
                       la.solve(self.M(), B2)])
        # fmt: on

        # y = Cx + Du
        # Observation matrices
        Cd = I
        Cv = Z
        Ca = Z

        # fmt: off
        C = np.hstack((Cd - Ca @ la.solve(self.M(), self.K(frequency)), Cv - Ca @ la.solve(self.M(), self.C(frequency))))
        # fmt: on
        D = Ca @ la.solve(self.M(), B2)

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
        modes : list, optional
            List with modes used to calculate the matrix.
            (all modes will be used if a list is not given).

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
        lti = self._lti(speed=speed)
        B = lti.B
        C = lti.C
        D = lti.D

        # calculate eigenvalues and eigenvectors using la.eig to get
        # left and right eigenvectors.

        evals, psi = self._eigen(speed=speed, frequency=frequency, sparse=False)

        psi_inv = la.inv(psi)

        if modes is not None:
            n = self.ndof  # n dof -> number of modes
            m = len(modes)  # -> number of desired modes
            # idx to get each evalue/evector and its conjugate
            idx = np.zeros((2 * m), int)
            idx[0:m] = modes  # modes
            idx[m:] = range(2 * n)[-m:]  # conjugates (see how evalues are ordered)
            evals = evals[np.ix_(idx)]
            psi = psi[np.ix_(range(2 * n), idx)]
            psi_inv = psi_inv[np.ix_(idx, range(2 * n))]

        diag = np.diag([1 / (1j * speed - lam) for lam in evals])

        H = C @ psi @ diag @ psi_inv @ B + D

        return H

    def run_freq_response(
        self,
        speed_range=None,
        modes=None,
        cluster_points=False,
        num_modes=12,
        num_points=10,
        rtol=0.005,
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
        speed_range : array, optional
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

        Returns
        -------
        results : object
            An instance of ForcedResponseResult class, which is used to post-process
            results. Attributes stored:
            freq_resp : array
                Array with the frequency response for each node for each pair
                input/output.
            speed_range : array
                Array with the frequencies.
            velc_resp : array
                Array with the velocity response for each node for each pair
                input/output.
            accl_resp : array
                Array with the acceleration response for each node for each pair
                input/output.

        Examples
        --------
        >>> import ross as rs
        >>> rotor = rs.rotor_example()
        >>> speed = np.linspace(0, 1000, 101)
        >>> response = rotor.run_freq_response(speed_range=speed)

        Return the response amplitude
        >>> abs(response.freq_resp) # doctest: +ELLIPSIS
        array([[[1.00000000e-06, 1.00261725e-06, 1.01076952e-06, ...

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
        >>> response = rotor.run_freq_response(speed_range=speed, modes=[0, 1, 2])
        >>> abs(response.freq_resp) # doctest: +ELLIPSIS
        array([[[2.00154633e-07, 2.02422522e-07, 2.09522044e-07, ...

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
        if speed_range is None:
            if not cluster_points:
                modal = self.run_modal(0)
                speed_range = np.linspace(0, max(modal.evalues.imag) * 1.5, 1000)
            else:
                speed_range = self._clustering_points(
                    num_modes, num_points, modes, rtol
                )

        self._check_frequency_array(speed_range)

        freq_resp = np.empty((self.ndof, self.ndof, len(speed_range)), dtype=np.complex)
        velc_resp = np.empty((self.ndof, self.ndof, len(speed_range)), dtype=np.complex)
        accl_resp = np.empty((self.ndof, self.ndof, len(speed_range)), dtype=np.complex)

        for i, speed in enumerate(speed_range):
            H = self.transfer_matrix(speed=speed, modes=modes)
            freq_resp[..., i] = H
            velc_resp[..., i] = 1j * speed * H
            accl_resp[..., i] = -(speed ** 2) * H

        results = FrequencyResponseResults(
            freq_resp=freq_resp,
            velc_resp=velc_resp,
            accl_resp=accl_resp,
            speed_range=speed_range,
            number_dof=self.number_dof,
        )

        return results

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
        force : list, array
            Unbalance force in each degree of freedom for each value in omega
        speed_range : list, array
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
        forced_resp : object
            An instance of ForcedResponseResult class, which is used to post-process
            results. Attributes stored:
            forced_resp : array
                Array with the forced response for each node for each frequency.
            speed_range : array
                Array with the frequencies.
            velc_resp : array
                Array with the velocity response for each node for each frequency.
            accl_resp : array
                Array with the acceleration response for each node for each frequency.

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

        forced_resp = np.zeros(
            (self.ndof, len(freq_resp.speed_range)), dtype=np.complex
        )
        velc_resp = np.zeros((self.ndof, len(freq_resp.speed_range)), dtype=np.complex)
        accl_resp = np.zeros((self.ndof, len(freq_resp.speed_range)), dtype=np.complex)

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
        >>> rotor._unbalance_force(3, 10.0, 0.0, speed)[12] # doctest: +ELLIPSIS
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
            F0[n0:n1, i] += w ** 2 * b0

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
        frequency : list, float, pint.Quantity
            Array with the desired range of frequencies (rad/s).
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
        forced_response : object
            An instance of ForcedResponseResult class, which is used to post-process
            results. Attributes stored:
            forced_resp : array
                Array with the forced response for each node for each frequency.
            speed_range : array
                Array with the frequencies.
            velc_resp : array
                Array with the velocity response for each node for each frequency.
            accl_resp : array
                Array with the acceleration response for each node for each frequency.

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
        array([[ 0.00000000e+00, ...

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
        >>> fig = response.plot(probe=[(probe_node, probe_angle, probe_tag)])

        plot response for major or minor axis:
        >>> probe_node = 3
        >>> probe_angle = "major"   # for major axis
        >>> # probe_angle = "minor" # for minor axis
        >>> probe_tag = "my_probe"  # optional
        >>> fig = response.plot(probe=[(probe_node, probe_angle, probe_tag)])

        To plot velocity and acceleration responses, you must change amplitude_units
        from "[length]" units to "[length]/[time]" or "[length]/[time] ** 2" respectively
        Plotting velocity response
        >>> fig = response.plot(
        ...     probe=[(probe_node, probe_angle)],
        ...     amplitude_units="m/s"
        ... )

        Plotting acceleration response
        >>> fig = response.plot(
        ...     probe=[(probe_node, probe_angle)],
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

        force = np.zeros((self.ndof, len(frequency)), dtype=np.complex)

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

    def time_response(self, speed, F, t, ic=None):
        """Time response for a rotor.

        This method returns the time response for a rotor
        given a force, time and initial conditions.

        Parameters
        ----------
        F : array
            Force array (needs to have the same length as time array).
        t : array
            Time array. (must have the same length than lti.B matrix)
        ic : array, optional
            The initial conditions on the state vector (zero by default).

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

        fig = go.Figure()

        # plot shaft centerline
        shaft_end = max(nodes_pos)
        fig.add_trace(
            go.Scatter(
                x=[-0.2 * shaft_end, 1.2 * shaft_end],
                y=[0, 0],
                mode="lines",
                opacity=0.7,
                line=dict(width=3.0, color="black", dash="dashdot"),
                showlegend=False,
                hoverinfo="none",
            )
        )

        # plot nodes icons
        text = []
        x_pos = []
        y_pos = np.linspace(0, 0, len(nodes_pos[::nodes]))
        for node, position in enumerate(nodes_pos[::nodes]):
            text.append("{}".format(node * nodes))
            x_pos.append(position)

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
            position = self.nodes_pos[sh_elm.n]
            fig = sh_elm._patch(position, check_sld, fig, length_units)

        mean_od = np.mean(nodes_o_d)
        # plot disk elements
        for disk in self.disk_elements:
            step = disk.scale_factor * mean_od
            position = (nodes_pos[disk.n], nodes_o_d[disk.n] / 2, step)
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
            position = (z_pos, y_pos, y_pos_sup)
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
            position = (z_pos, y_pos)
            fig = p_mass._patch(position, fig)

        fig.update_xaxes(
            title_text=f"Axial location ({length_units})",
            range=[-0.1 * shaft_end, 1.1 * shaft_end],
            showgrid=False,
            mirror=True,
        )
        fig.update_yaxes(
            title_text=f"Shaft radius ({length_units})",
            range=[-0.3 * shaft_end, 0.3 * shaft_end],
            showgrid=False,
            mirror=True,
        )
        fig.update_layout(title=dict(text="Rotor Model"), **kwargs)

        return fig

    def run_campbell(self, speed_range, frequencies=6, frequency_type="wd"):
        """Calculate the Campbell diagram.

        This function will calculate the damped natural frequencies
        for a speed range.

        Available plotting methods:
            .plot()

        Parameters
        ----------
        speed_range : array
            Array with the speed range in rad/s.
        frequencies : int, optional
            Number of frequencies that will be calculated.
            Default is 6.
        frequency_type : str, optional
            Choose between displaying results related to the undamped natural
            frequencies ("wn") or damped natural frequencies ("wd").
            The default is "wd".

        Returns
        -------
        results : array
            Array with the damped natural frequencies, log dec and precessions
            corresponding to each speed of the speed_rad array.
            It will be returned if plot=False.

        Examples
        --------
        >>> import ross as rs
        >>> rotor1 = rs.rotor_example()
        >>> speed = np.linspace(0, 400, 101)

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

        results = np.zeros([len(speed_range), frequencies, 5])

        for i, w in enumerate(speed_range):
            modal = self.run_modal(speed=w, num_modes=2 * frequencies)

            if frequency_type == "wd":
                results[i, :, 0] = modal.wd[:frequencies]
                results[i, :, 1] = modal.log_dec[:frequencies]
                results[i, :, 2] = modal.whirl_values()[:frequencies]
            else:
                idx = modal.wn.argsort()
                results[i, :, 0] = modal.wn[idx][:frequencies]
                results[i, :, 1] = modal.log_dec[idx][:frequencies]
                results[i, :, 2] = modal.whirl_values()[idx][:frequencies]

            results[i, :, 3] = w
            results[i, :, 4] = modal.wn[:frequencies]

        results = CampbellResults(
            speed_range=speed_range,
            wd=results[..., 0],
            log_dec=results[..., 1],
            whirl_values=results[..., 2],
        )

        return results

    def run_ucs(
        self,
        stiffness_range=None,
        num_modes=16,
        num=20,
        fig=None,
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
            Tuple with (start, end) for stiffness range.
        num : int
            Number of steps in the range.
            Default is 20.
        num_modes : int, optional
            Number of modes to be calculated. This uses scipy.sparse.eigs method.
            Default is 16.
        synchronous : bool
            If True a synchronous analysis is carried out and the frequency of
            the first forward model will be equal to the speed.
            Default is False.
        """
        if stiffness_range is None:
            if self.rated_w is not None:
                bearing = self.bearing_elements[0]
                k = bearing.kxx.interpolated(self.rated_w)
                k = int(np.log10(k))
                stiffness_range = (k - 3, k + 3)
            else:
                stiffness_range = (6, 11)

        stiffness_log = np.logspace(*stiffness_range, num=num)
        rotor_wn = np.zeros((self.number_dof, len(stiffness_log)))

        bearings_elements = []  # exclude the seals
        for bearing in self.bearing_elements:
            if not isinstance(bearing, SealElement):
                bearings_elements.append(bearing)

        for i, k in enumerate(stiffness_log):
            bearings = [BearingElement(b.n, kxx=k, cxx=0) for b in bearings_elements]
            rotor = self.__class__(self.shaft_elements, self.disk_elements, bearings)
            speed = 0
            if synchronous:

                def wn_diff(x):
                    """Function to evaluate difference between speed and
                    natural frequency for the first mode."""
                    modal = rotor.run_modal(speed=x, num_modes=num_modes)
                    # get first forward mode
                    if modal.whirl_direction()[0] == "Forward":
                        wn0 = modal.wn[0]
                    else:
                        wn0 = modal.wn[1]

                    return wn0 - x

                speed = newton(wn_diff, 0)
            modal = rotor.run_modal(speed=speed, num_modes=num_modes)

            # if sync, select only forward modes
            if synchronous:
                rotor_wn[:, i] = modal.wn[modal.whirl_direction() == "Forward"]
            # if not sync, with speed=0 whirl direction can be confusing, with
            # two close modes being forward or backward, so we select on mode in
            # each 2 modes.
            else:
                rotor_wn[:, i] = modal.wn[
                    : int(self.number_dof * 2) : int(self.number_dof / 2)
                ]

        bearing0 = bearings_elements[0]

        # calculate interception points
        intersection_points = {"x": [], "y": []}

        # if bearing does not have constant coefficient, check intersection points
        if not np.isnan(bearing0.frequency).all():
            for j in range(rotor_wn.shape[0]):
                for coeff in ["kxx", "kyy"]:
                    x1 = rotor_wn[j]
                    y1 = stiffness_log
                    x2 = bearing0.frequency
                    y2 = getattr(bearing0, coeff).coefficient
                    x, y = intersection(x1, y1, x2, y2)
                    try:
                        intersection_points["y"].append(float(x))
                        intersection_points["x"].append(float(y))
                    except TypeError:
                        # pass if x/y is empty
                        pass

        results = UCSResults(
            stiffness_range, stiffness_log, rotor_wn, bearing0, intersection_points
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
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.

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
            cross_coupling = BearingElement(n=n, kxx=0, cxx=0, kxy=Q, kyx=-Q)
            bearings.append(cross_coupling)

            rotor = self.__class__(self.shaft_elements, self.disk_elements, bearings)

            modal = rotor.run_modal(speed=speed)
            non_backward = modal.whirl_direction() != "Backward"
            log_dec[i] = modal.log_dec[non_backward][0]

        results = Level1Results(stiffness, log_dec)

        return results

    def run_time_response(self, speed, F, t):
        """Calculate the time response.

        This function will take a rotor object and calculate its time response
        given a force and a time.

        Available plotting methods:
            .plot_1d()
            .plot_2d()
            .plot_3d()

        Parameters
        ----------
        speed : float
            Rotor speed.
        F : array
            Force array (needs to have the same number of rows as time array).
            Each column corresponds to a dof and each row to a time.
        t : array
            Time array.

        Returns
        -------
        results : array
            Array containing the time array, the system response, and the
            time evolution of the state vector.
            It will be returned if plot=False.

        Examples
        --------
        >>> rotor = rotor_example()
        >>> speed = 500.0
        >>> size = 1000
        >>> node = 3
        >>> probe1 = (3, 0)
        >>> t = np.linspace(0, 10, size)
        >>> F = np.zeros((size, rotor.ndof))
        >>> F[:, 4 * node] = 10 * np.cos(2 * t)
        >>> F[:, 4 * node + 1] = 10 * np.sin(2 * t)
        >>> response = rotor.run_time_response(speed, F, t)
        >>> dof = 13
        >>> response.yout[:, dof] # doctest: +ELLIPSIS
        array([ 0.00000000e+00,  1.86686693e-07,  8.39130663e-07, ...

        # plot time response for a given probe:
        >>> fig1 = response.plot_1d(probe=[probe1])

        # plot orbit response - plotting 2D nodal orbit:
        >>> fig2 = response.plot_2d(node=node)

        # plot orbit response - plotting 3D orbits - full rotor model:
        >>> fig3 = response.plot_3d()
        """
        t_, yout, xout = self.time_response(speed, F, t)

        results = TimeResponseResults(self, t, yout, xout)

        return results

    def run_misalignment(self, coupling="flex", **kwargs):
        """Run an analyzes with misalignment.

        Execute the misalignment defect and generates the misalignment object
        on the back-end. There are two types of coupling, flexible (flex)
        and rigid, which have different entries. These entries are provided
        via **kwargs to the specific method.

        Parameters
        ----------
        coupling : str
            Coupling type. The avaible types are: flex, by default; and rigid.

        **kwargs: dictionary

            In the case of coupling = "flex", **kwargs receives:
                dt : float
                    Time step.
                tI : float
                    Initial time.
                tF : float
                    Final time.
                kd : float
                    Radial stiffness of flexible coupling.
                ks : float
                    Bending stiffness of flexible coupling.
                eCOUPx : float
                    Parallel misalignment offset between driving rotor and driven rotor along X direction.
                eCOUPy : float
                    Parallel misalignment offset between driving rotor and driven rotor along Y direction.
                misalignment_angle : float
                    Angular misalignment angle.
                TD : float
                    Driving torque.
                TL : float
                    Driven torque.
                n1 : float
                    Node where the misalignment is ocurring.
                speed : float, pint.Quantity
                    Operational speed of the machine. Default unit is rad/s.
                unbalance_magnitude : array
                    Array with the unbalance magnitude. The unit is kg.m.
                unbalance_phase : array
                    Array with the unbalance phase. The unit is rad.
                mis_type: string
                    String containing the misalignment type choosed. The avaible types are: parallel, by default; angular; combined.
                print_progress : bool
                    Set it True, to print the time iterations and the total time spent.
                    False by default.

            In the case of coupling = "rigid", **kwargs receives:
                dt : float
                    Time step.
                tI : float
                    Initial time.
                tF : float
                    Final time.
                eCOUP : float
                    Parallel misalignment offset between driving rotor and driven rotor along X direction.
                TD : float
                    Driving torque.
                TL : float
                    Driven torque.
                n1 : float
                    Node where the misalignment is ocurring.
                speed : float, pint.Quantity
                    Operational speed of the machine. Default unit is rad/s.
                unbalance_magnitude : array
                    Array with the unbalance magnitude. The unit is kg.m.
                unbalance_phase : array
                    Array with the unbalance phase. The unit is rad.
                print_progress : bool
                    Set it True, to print the time iterations and the total time spent.
                    False by default.

        Examples
        --------
        >>> from ross.defects.misalignment import misalignment_flex_parallel_example
        >>> probe1 = (14, 0)
        >>> probe2 = (22, 0)
        >>> response = misalignment_flex_parallel_example()
        >>> results = response.run_time_response()
        >>> fig = response.plot_dfft(probe=[probe1, probe2], range_freq=[0, 100], yaxis_type="log")
        >>> # fig.show()
        """

        if coupling == "flex" or coupling == None:
            defect = MisalignmentFlex(**kwargs)
        elif coupling == "rigid":
            defect = MisalignmentRigid(**kwargs)
        else:
            raise Exception("Check the choosed coupling type!")

        defect.run(self)
        return defect

    def run_rubbing(self, **kwargs):
        """Run an analyzes with rubbing.

        Execute the rubbing defect and generates the rubbing object on the back-end.

        Parameters
        ----------
        **kwargs: dictionary

            **kwargs receives:
                dt : float
                    Time step.
                tI : float
                    Initial time.
                tF : float
                    Final time.
                deltaRUB : float
                    Distance between the housing and shaft surface.
                kRUB : float
                    Contact stiffness.
                cRUB : float
                    Contact damping.
                miRUB : float
                    Friction coefficient.
                posRUB : int
                    Node where the rubbing is ocurring.
                speed : float, pint.Quantity
                    Operational speed of the machine. Default unit is rad/s.
                unbalance_magnitude : array
                    Array with the unbalance magnitude. The unit is kg.m.
                unbalance_phase : array
                    Array with the unbalance phase. The unit is rad.
                torque : bool
                    Set it as True to consider the torque provided by the rubbing, by default False.
                print_progress : bool
                    Set it True, to print the time iterations and the total time spent, by default False.

        Examples
        --------
        >>> from ross.defects.rubbing import rubbing_example
        >>> probe1 = (14, 0)
        >>> probe2 = (22, 0)
        >>> response = rubbing_example()
        >>> results = response.run_time_response()
        >>> fig = response.plot_dfft(probe=[probe1, probe2], range_freq=[0, 100], yaxis_type="log")
        >>> # fig.show()
        """

        defect = Rubbing(**kwargs)
        defect.run(self)
        return defect

    def run_crack(self, **kwargs):
        """Run an analyzes with rubbing.

        Execute the crack defect and generates the crack object on the back-end.

        Parameters
        ----------
        **kwargs: dictionary

            **kwargs receives:
                dt : float
                    Time step
                tI : float
                    Initial time
                tF : float
                    Final time
                depth_ratio : float
                    Crack depth ratio related to the diameter of the crack container element. A depth value of 0.1 is equal to 10%, 0.2 equal to 20%, and so on.
                n_crack : float
                    Element where the crack is located
                speed : float, pint.Quantity
                    Operational speed of the machine. Default unit is rad/s.
                unbalance_magnitude : array
                    Array with the unbalance magnitude. The unit is kg.m.
                unbalance_phase : array
                    Array with the unbalance phase. The unit is rad.
                crack_type : string
                    String containing type of crack model chosed. The avaible types are: Mayes and Gasch.
                print_progress : bool
                    Set it True, to print the time iterations and the total time spent, by default False.
        Examples
        --------
        >>> from ross.defects.crack import crack_example
        >>> probe1 = (14, 0)
        >>> probe2 = (22, 0)
        >>> response = crack_example()
        >>> results = response.run_time_response()
        >>> fig = response.plot_dfft(probe=[probe1, probe2], range_freq=[0, 100], yaxis_type="log")
        >>> # fig.show()
        """
        defect = Crack(**kwargs)
        defect.run(self)
        return defect

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
            "M": self.M(),
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
            elements.append(globals()[class_name].read_toml_data(el_data))

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
        disp_y: array
            The shaft static displacement vector,
        Vx: array
            Shearing force vector
        Bm: array
            Bending moment vector

        Returns
        -------
        results: object
            An instance of StaticResult class, which is used to create plots.

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
        {'node_0': 432.4, 'node_6': 432.4}
        >>> rotor.bearing_forces_tag
        {'Bearing 0': 432.4, 'Bearing 1': 432.4}

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
        for elm in self.bearing_elements:
            if elm.n not in self.nodes:
                pass
            elif elm.n_link in self.nodes:
                aux_brg.append(
                    BearingElement(n=elm.n, n_link=elm.n_link, kxx=1e14, cxx=0)
                )
            else:
                aux_brg.append(BearingElement(n=elm.n, kxx=1e14, cxx=0))

        if isinstance(self, CoAxialRotor):
            aux_rotor = CoAxialRotor(self.shafts, self.disk_elements, aux_brg)
        else:
            aux_rotor = Rotor(self.shaft_elements, self.disk_elements, aux_brg)

        aux_K = aux_rotor.K(0)
        for elm in aux_rotor.bearing_elements:
            if isinstance(elm, SealElement):
                dofs = elm.dof_global_index
                try:
                    aux_K[np.ix_(dofs, dofs)] -= elm.K(0)
                except TypeError:
                    aux_K[np.ix_(dofs, dofs)] -= elm.K()

        df_num = aux_rotor.df["shaft_number"].values
        sh_num = [int(item) for item, count in Counter(df_num).items() if count > 1]

        # gravity aceleration vector
        g = -9.8065
        grav = np.zeros(len(aux_rotor.K(0)))
        grav[1 :: self.number_dof] = g

        # calculates u, for [K]*(u) = (F)
        disp = (la.solve(aux_K, aux_rotor.M() @ grav)).flatten()

        # calculates displacement values in gravity's direction
        shafts_disp_y = disp[1 :: self.number_dof]

        disp_y = []

        # calculate forces
        nodal_forces = self.K(0) @ disp

        Vx_axis, Vx, Mx = [], [], []
        nodes, nodes_pos = [], []

        BrgForce_nodal = {}
        DskForce_nodal = {}
        BrgForce_tag = {}
        DskForce_tag = {}
        for i in sh_num:
            # get indexes for each shaft in the model
            index = self.df_shaft.loc[self.df_shaft.shaft_number == i, "_n"].index
            n_min = min(self.df_shaft.loc[self.df_shaft.shaft_number == i, "n_l"])
            n_max = max(self.df_shaft.loc[(self.df_shaft.shaft_number == i), "n_r"])
            nodes_pos.append(self.nodes_pos[n_min : n_max + 1])
            nodes.append(list(range(n_min, n_max + 1)))

            elm_weight = np.zeros((len(nodes_pos[i]) - 1, 2))
            nodal_shaft_weight = np.zeros(len(nodes_pos[i]))

            # displacements for a single shaft
            shafts_disp = disp[n_min * self.number_dof : (n_max + 1) * self.number_dof]
            disp_y.append(shafts_disp[1 :: self.number_dof])

            aux_Vx_axis = np.zeros_like(elm_weight)
            for sh in np.array(self.shaft_elements)[index]:
                aux_Vx_axis[sh.n_l - n_min] = [
                    self.nodes_pos[sh.n_l],
                    self.nodes_pos[sh.n_r],
                ]
                elm_weight[sh.n_l - n_min] += g * np.array([0, sh.m])

                nodal_shaft_weight[sh.n_r - n_min] += g * sh.m * sh.beam_cg / sh.L
                nodal_shaft_weight[sh.n_l - n_min] += g * sh.m * (1 - sh.beam_cg / sh.L)

            elm_weight[-1, 1] = 0
            aux_nodal_forces = nodal_forces[
                self.number_dof * n_min : self.number_dof * (n_max + 1)
            ]

            nodal_forces_y = aux_nodal_forces[1 :: self.number_dof] - nodal_shaft_weight
            elm_forces_y = np.zeros_like(elm_weight)
            elm_forces_y[:, 0] = nodal_forces_y[:-1]
            elm_forces_y[-1, 1] = -nodal_forces_y[-1]
            elm_forces_y += elm_weight

            # locate and collect bearing and disk forces
            aux_df = aux_rotor.df.loc[
                (aux_rotor.df["type"] != "ShaftElement")
                & (aux_rotor.df["shaft_number"] == i)
            ]
            for j, row in aux_df.iterrows():
                if row["n"] == n_max:
                    force = -np.round(elm_forces_y[-1, 1], 1)
                else:
                    force = np.round(elm_forces_y[int(row["n"]) - n_min, 0], 1)

                if row["type"] == "DiskElement":
                    DskForce_nodal["node_" + str(int(row["n"]))] = force
                    DskForce_tag[row["tag"]] = force
                elif row["type"] == "BearingElement":
                    BrgForce_nodal["node_" + str(int(row["n"]))] = force
                    BrgForce_tag[row["tag"]] = force
                    if not pd.isna(row["n_link"]):
                        BrgForce_nodal["node_" + str(int(row["n_link"]))] = -force

            # Calculate shearing force
            # Each line represents an element, each column a station from the element
            aux_Vx = np.zeros_like(elm_weight)
            for j in range(aux_Vx.shape[0]):
                if j == 0:
                    aux_Vx[j] = [elm_forces_y[j, 0], sum(elm_forces_y[j])]
                elif j == aux_Vx.shape[0] - 1:
                    aux_Vx[j, 0] = aux_Vx[j - 1, 1] + elm_forces_y[j, 0]
                    aux_Vx[j, 1] = elm_forces_y[j, 1]
                else:
                    aux_Vx[j, 0] = aux_Vx[j - 1, 1] + elm_forces_y[j, 0]
                    aux_Vx[j, 1] = aux_Vx[j, 0] + elm_forces_y[j, 1]
            aux_Vx = -aux_Vx

            # Calculate bending moment
            # Each line represents an element, each column a station from the element
            aux_Mx = np.zeros_like(aux_Vx)
            for j in range(aux_Mx.shape[0]):
                if j == 0:
                    aux_Mx[j] = [0, 0.5 * sum(aux_Vx[j]) * np.diff(aux_Vx_axis[j])]
                if j == aux_Mx.shape[0] - 1:
                    aux_Mx[j] = [-0.5 * sum(aux_Vx[j]) * np.diff(aux_Vx_axis[j]), 0]
                else:
                    aux_Mx[j, 0] = aux_Mx[j - 1, 1]
                    aux_Mx[j, 1] = aux_Mx[j, 0] + 0.5 * sum(aux_Vx[j]) * np.diff(
                        aux_Vx_axis[j]
                    )

            # flattening arrays
            aux_Vx = aux_Vx.flatten()
            aux_Vx_axis = aux_Vx_axis.flatten()
            aux_Mx = aux_Mx.flatten()

            Vx.append(aux_Vx)
            Vx_axis.append(aux_Vx_axis)
            Mx.append(aux_Mx)

        self.disk_forces_nodal = DskForce_nodal
        self.bearing_forces_nodal = BrgForce_nodal
        self.bearing_forces_tag = BrgForce_tag
        self.disk_forces_tag = DskForce_tag

        self.w_shaft = [
            sum(self.df_shaft.loc[self.df_shaft.shaft_number == i, "m"]) * (-g)
            for i in sh_num
        ]

        results = StaticResults(
            disp_y,
            Vx,
            Mx,
            self.w_shaft,
            self.disk_forces_nodal,
            self.bearing_forces_nodal,
            nodes,
            nodes_pos,
            Vx_axis,
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
        self.run_static()
        forces = self.bearing_forces_tag
        results = SummaryResults(
            self.df_shaft,
            self.df_disks,
            self.df_bearings,
            self.nodes_pos,
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
        array([ 85.7634,  85.7634, 271.9326, 271.9326, 718.58  , 718.58  ])
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
                aux_Brg_SealEl.n_l = nel_r * Brg_SealEl.n_l
                aux_Brg_SealEl.n_r = nel_r * Brg_SealEl.n_r
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

        self._v0 = None  # used to call eigs

        # number of dofs
        self.ndof = int(
            4 * max([el.n for el in shaft_elements])
            + 8
            + 2 * len([el for el in point_mass_elements])
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
                global_dof_mapping[dof_letter + "_" + str(int(dof_number) + elm.n)] = v

            if elm.n <= n_last + 1:
                for k, v in global_dof_mapping.items():
                    global_dof_mapping[k] = 4 * elm.n + v
            else:
                for k, v in global_dof_mapping.items():
                    global_dof_mapping[k] = 2 * n_last + 2 * elm.n + 4 + v

            if hasattr(elm, "n_link") and elm.n_link is not None:
                if elm.n_link <= n_last + 1:
                    global_dof_mapping[f"x_{elm.n_link}"] = 4 * elm.n_link
                    global_dof_mapping[f"y_{elm.n_link}"] = 4 * elm.n_link + 1
                else:
                    global_dof_mapping[f"x_{elm.n_link}"] = (
                        2 * n_last + 2 * elm.n_link + 4
                    )
                    global_dof_mapping[f"y_{elm.n_link}"] = (
                        2 * n_last + 2 * elm.n_link + 5
                    )

            dof_tuple = namedtuple("GlobalIndex", global_dof_mapping)
            elm.dof_global_index = dof_tuple(**global_dof_mapping)
            df.at[
                df.loc[df.tag == elm.tag].index[0], "dof_global_index"
            ] = elm.dof_global_index

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


def rotor_example():
    """Create a rotor as example.

    This function returns an instance of a simple rotor with
    two shaft elements, one disk and two simple bearings.
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
    #  Rotor without damping with 6 shaft elements 2 disks and 2 bearings
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


def coaxrotor_example():
    """Create a rotor as example.

    This function returns an instance of a simple rotor with
    two shafts, four disk and four bearings.
    The purpose of this is to make available a simple model for co-axial rotors
    so that doctest can be written using this.

    Returns
    -------
    An instance of a rotor object.

    Examples
    --------
    >>> import ross as rs
    >>> rotor = rs.coaxrotor_example()

    Plotting rotor model
    >>> fig = rotor.plot_rotor()

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
    """This function returns an instance of a simple rotor with
    two shaft elements, one disk and two simple bearings.
    The purpose of this is to make available a simple model
    so that doctest can be written using this.

    Parameters
    ----------

    Returns
    -------
    An instance of a 6DoFs rotor object.

    Examples
    --------
    >>> import ross as rs
    >>> import numpy as np
    >>> rotor6 = rs.rotor_assembly.rotor_example_6dof()
    >>> camp6 = rotor6.run_campbell(np.linspace(0,400,101),frequencies=18)

    # plotting Campbell Diagram
    >>> fig = camp6.plot()
    """
    #  Rotor with 6 DoFs, with internal damping, with 10 shaft elements, 2 disks and 2 bearings.
    i_d = 0
    o_d = 0.05
    n = 6
    L = [0.25 for _ in range(n)]

    shaft_elem = [
        ShaftElement6DoF(
            material=steel,
            L=0.25,
            idl=0,
            odl=0.05,
            idr=0,
            odr=0.05,
            alpha=0,
            beta=0,
            rotary_inertia=False,
            shear_effects=False,
        )
        for l in L
    ]

    disk0 = DiskElement6DoF.from_geometry(
        n=2, material=steel, width=0.07, i_d=0.05, o_d=0.28
    )
    disk1 = DiskElement6DoF.from_geometry(
        n=4, material=steel, width=0.07, i_d=0.05, o_d=0.28
    )

    kxx = 1e6
    kyy = 0.8e6
    kzz = 1e5
    cxx = 0
    cyy = 0
    czz = 0
    bearing0 = BearingElement6DoF(
        n=0, kxx=kxx, kyy=kyy, cxx=cxx, cyy=cyy, kzz=kzz, czz=czz
    )
    bearing1 = BearingElement6DoF(
        n=6, kxx=kxx, kyy=kyy, cxx=cxx, cyy=cyy, kzz=kzz, czz=czz
    )

    return Rotor(shaft_elem, [disk0, disk1], [bearing0, bearing1])
