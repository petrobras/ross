# fmt: off
import os
import shutil
import warnings
from collections import Counter, Iterable, namedtuple
from copy import copy, deepcopy
from itertools import chain, cycle
from pathlib import Path

import bokeh.palettes as bp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.linalg as la
import scipy.signal as signal
import scipy.sparse.linalg as las
import toml
from bokeh.models import ColumnDataSource
from bokeh.models.glyphs import Text
from bokeh.plotting import figure, output_file
from cycler import cycler

from ross.bearing_seal_element import BearingElement, SealElement
from ross.disk_element import DiskElement
from ross.materials import steel
from ross.results import (CampbellResults, ConvergenceResults,
                          ForcedResponseResults, FrequencyResponseResults,
                          ModalResults, OrbitResponseResults, StaticResults,
                          SummaryResults, TimeResponseResults)
from ross.shaft_element import ShaftElement
from ross.utils import convert

# fmt: on

__all__ = ["Rotor", "CoAxialRotor", "rotor_example", "coaxrotor_example"]

# set style and colors
plt.style.use("seaborn-white")
plt.style.use(
    {
        "lines.linewidth": 2.5,
        "axes.grid": True,
        "axes.linewidth": 0.1,
        "grid.color": ".9",
        "grid.linestyle": "--",
        "legend.frameon": True,
        "legend.framealpha": 0.2,
    }
)

# set bokeh palette of colors
bokeh_colors = bp.RdGy[11]

_orig_rc_params = mpl.rcParams.copy()

seaborn_colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b2", "#ccb974", "#64b5cd"]


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
    sparse : bool, optional
        If sparse, eigenvalues will be calculated with arpack.
        Default is True.
    n_eigen : int, optional
        Number of eigenvalues calculated by arpack.
        Default is 12.
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
        sparse=True,
        n_eigen=12,
        min_w=None,
        max_w=None,
        rated_w=None,
        tag=None,
    ):

        self.parameters = {
            "sparse": True,
            "n_eigen": n_eigen,
            "min_w": min_w,
            "max_w": max_w,
            "rated_w": rated_w,
        }
        if tag is None:
            self.tag = "Rotor 0"

        ####################################################
        # Config attributes
        ####################################################

        self.sparse = sparse
        self.n_eigen = n_eigen
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
            if brg.__class__.__name__ == "BearingElement" and brg.tag is None:
                brg.tag = "Bearing " + str(i)
            if brg.__class__.__name__ == "SealElement" and brg.tag is None:
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
                if (el.__class__.__name__ == "BearingElement")
            ]
        )
        df_seals = pd.DataFrame(
            [
                el.summary()
                for el in self.bearing_elements
                if (el.__class__.__name__ == "SealElement")
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

        # values for evalues and evectors will be calculated by self.run_modal
        self.evalues = None
        self.evectors = None
        self.wn = None
        self.wd = None
        self.lti = None

        self._v0 = None  # used to call eigs

        # number of dofs
        self.ndof = (
            4 * max([el.n for el in shaft_elements])
            + 8
            + 2 * len([el for el in point_mass_elements])
        )

        # global indexes for dofs
        n_last = self.shaft_elements[-1].n
        for elm in self.elements:
            dof_mapping = elm.dof_mapping()
            global_dof_mapping = {}
            for k, v in dof_mapping.items():
                dof_letter, dof_number = k.split("_")
                global_dof_mapping[dof_letter + "_" + str(int(dof_number) + elm.n)] = v
            dof_tuple = namedtuple("GlobalIndex", global_dof_mapping)

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

        #  values for static analysis will be calculated by def static
        self.Vx = None
        self.Bm = None
        self.disp_y = None

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
        dfb = df[df.type == "BearingElement"]
        for p in point_mass_elements:
            z_pos = dfb[dfb.n_l == p.n]["nodes_pos_l"].values[0]
            y_pos = dfb[dfb.n_l == p.n]["y_pos"].values[0]
            df.loc[df.tag == p.tag, "nodes_pos_l"] = z_pos
            df.loc[df.tag == p.tag, "nodes_pos_r"] = z_pos
            df.loc[df.tag == p.tag, "y_pos"] = y_pos

        self.df = df

    def __eq__(self, other):
        """
        Equality method for comparasions

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

    def run_modal(self, speed):
        """
        Method to calculate eigenvalues and eigvectors for a given rotor system
        This method is automatically called when a rotor is instantiated.

        Parameters
        ----------

        Returns
        -------
        evalues : array
            Eigenvalues array
        evectors : array
            Eigenvectors array
        wn : array
            Undamped natural frequencies array
        wd : array
            Damped natural frequencies array
        log_dec : array
            Logarithmic decrement array

        Example
        -------
        >>> rotor = rotor_example()
        >>> modal = rotor.run_modal(speed=0)
        >>> modal.wn[:2]
        array([91.79655318, 96.28899977])
        >>> modal.wd[:2]
        array([91.79655318, 96.28899977])
        >>> modal.plot_mode3D(0) # doctest: +ELLIPSIS
        (<Figure ...
        """
        evalues, evectors = self._eigen(speed)
        wn_len = len(evalues) // 2
        wn = (np.absolute(evalues))[:wn_len]
        wd = (np.imag(evalues))[:wn_len]
        damping_ratio = (-np.real(evalues) / np.absolute(evalues))[:wn_len]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            log_dec = 2 * np.pi * damping_ratio / np.sqrt(1 - damping_ratio ** 2)
        lti = self._lti(speed)
        modal_results = ModalResults(
            speed,
            evalues,
            evectors,
            wn,
            wd,
            damping_ratio,
            log_dec,
            lti,
            self.ndof,
            self.nodes,
            self.nodes_pos,
            self.shaft_elements_length,
        )

        return modal_results

    def convergence(self, n_eigval=0, err_max=1e-02):
        """
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
        Lists containing the information about:
            The number or elements in each run;
            The relative error calculated in each run;
            The natural frequency calculated in each run.

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

            aux_rotor = Rotor(shaft_elem, disk_elem, brgs_elem, n_eigen=self.n_eigen)
            aux_modal = aux_rotor.run_modal(speed=0)

            eigv_arr = np.append(eigv_arr, aux_modal.wn[n_eigval])
            el_num = np.append(el_num, len(shaft_elem))

            error = min(eigv_arr[-1], eigv_arr[-2]) / max(eigv_arr[-1], eigv_arr[-2])
            error = 1 - error

            error_arr = np.append(error_arr, 100 * error)
            nel_r *= 2

        self.__dict__ = aux_rotor.__dict__
        self.error_arr = error_arr

        results = ConvergenceResults(el_num[1:], eigv_arr[1:], error_arr[1:])

        return results

    def M(self):
        r"""Mass matrix for an instance of a rotor.

        Returns
        -------
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

    def C(self, frequency):
        """Damping matrix for an instance of a rotor.

        Parameters
        ----------
        frequency : float
            Excitation frequency.

        Returns
        -------
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
             np.hstack([la.solve(-self.M(), self.K(frequency)), la.solve(-self.M(), (self.C(frequency) + self.G() * speed))])])
        # fmt: on

        return A

    @staticmethod
    def _index(eigenvalues):
        r"""Function used to generate an index that will sort
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

    def _eigen(self, speed, frequency=None, sorted_=True, A=None):
        r"""This method will return the eigenvalues and eigenvectors of the
        state space matrix A, sorted by the index method which considers
        the imaginary part (wd) of the eigenvalues for sorting.
        To avoid sorting use sorted_=False

        Parameters
        ----------
        speed: float
            Rotor speed.
        frequency: float
            Excitation frequency.
        sorted_: bool, optional
            Sort considering the imaginary part (wd)
            Default is True
        A: np.array, optional
            Matrix for which eig will be calculated.
            Defaul is the rotor A matrix.


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

        if self.sparse is True:
            try:
                evalues, evectors = las.eigs(
                    A,
                    k=self.n_eigen,
                    sigma=0,
                    ncv=2 * self.n_eigen,
                    which="LM",
                    v0=self._v0,
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
        """
        Calculates the fer matrix for the frequency response function (FRF)

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
        modal = self.run_modal(speed=speed)
        B = modal.lti.B
        C = modal.lti.C
        D = modal.lti.D

        # calculate eigenvalues and eigenvectors using la.eig to get
        # left and right eigenvectors.

        evals, psi, = la.eig(self.A(speed, frequency))

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

    def run_freq_response(self, speed_range=None, modes=None):
        """Frequency response for a mdof system.

        This method returns the frequency response for a mdof system
        given a range of frequencies and the modes that will be used.

        Parameters
        ----------
        speed_range : array, optional
            Array with the desired range of frequencies (the default
             is 0 to 1.5 x highest damped natural frequency.
        modes : list, optional
            Modes that will be used to calculate the frequency response
            (all modes will be used if a list is not given).

        Returns
        -------
        results : array
            Array with the frequencies, magnitude (dB) of the frequency
            response for each pair input/output, and
            phase of the frequency response for each pair input/output..
            It will be returned if plot=False.

        Examples
        --------
        >>> rotor = rotor_example()
        >>> speed = np.linspace(0, 1000, 101)
        >>> response = rotor.run_freq_response(speed_range=speed)
        >>> response.magnitude # doctest: +ELLIPSIS
        array([[[1.00000000e-06, 1.00261725e-06, 1.01076952e-06, ...
        """
        if speed_range is None:
            modal = self.run_modal(0)
            speed_range = np.linspace(0, max(modal.evalues.imag) * 1.5, 1000)

        freq_resp = np.empty((self.ndof, self.ndof, len(speed_range)), dtype=np.complex)

        for i, speed in enumerate(speed_range):
            H = self.transfer_matrix(speed=speed, modes=modes)
            freq_resp[..., i] = H

        results = FrequencyResponseResults(
            freq_resp=freq_resp,
            speed_range=speed_range,
            magnitude=abs(freq_resp),
            phase=np.angle(freq_resp),
        )

        return results

    def run_forced_response(self, force=None, speed_range=None, modes=None):
        """Unbalanced response for a mdof system.

        This method returns the unbalanced response for a mdof system
        given magnitude and phase of the unbalance, the node where it's
        applied and a frequency range.

        Parameters
        ----------
        force : list
            Unbalance force in each degree of freedom for each value in omega
        speed_range : list, float
            Array with the desired range of frequencies
        modes : list, optional
            Modes that will be used to calculate the frequency response
            (all modes will be used if a list is not given).

        Returns
        -------
        force_resp : array
            Array with the force response for each node for each frequency
        speed_range : array
            Array with the frequencies
        magnitude : array
            Magnitude (dB) of the frequency response for node for each frequency
        phase : array
            Phase of the frequency response for node for each frequency

        Examples
        --------
        >>> rotor = rotor_example()
        >>> speed = np.linspace(0, 1000, 101)
        >>> force = rotor._unbalance_force(3, 10.0, 0.0, speed)
        >>> resp = rotor.run_forced_response(force=force, speed_range=speed)
        >>> resp.magnitude # doctest: +ELLIPSIS
        array([[0.00000000e+00, 5.06073311e-04, 2.10044826e-03, ...
        """
        freq_resp = self.run_freq_response(speed_range=speed_range, modes=modes)

        forced_resp = np.zeros(
            (self.ndof, len(freq_resp.speed_range)), dtype=np.complex
        )

        for i in range(len(freq_resp.speed_range)):
            forced_resp[:, i] = freq_resp.freq_resp[..., i] @ force[..., i]

        forced_resp = ForcedResponseResults(
            forced_resp=forced_resp,
            speed_range=speed_range,
            magnitude=abs(forced_resp),
            phase=np.angle(forced_resp),
        )

        return forced_resp

    def _unbalance_force(self, node, magnitude, phase, omega):
        """
        Function to calculate unbalance force

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
        me = magnitude
        delta = phase
        b0 = np.array(
            [
                me * np.exp(1j * delta),
                -1j * me * np.exp(1j * delta),
                0,  # 1j*(Id - Ip)*beta*np.exp(1j*gamma),
                0,
            ]
        )  # (Id - Ip)*beta*np.exp(1j*gamma)])

        n0 = 4 * node
        n1 = n0 + 4
        for i, w in enumerate(omega):
            F0[n0:n1, i] += w ** 2 * b0

        return F0

    def unbalance_response(self, node, magnitude, phase, frequency_range=None):
        """Unbalanced response for a mdof system.

        This method returns the unbalanced response for a mdof system
        given magnitide and phase of the unbalance, the node where it's
        applied and a frequency range.

        Parameters
        ----------
        node : list, int
            Node where the unbalance is applied.
        magnitude : list, float
            Unbalance magnitude (kg.m)
        phase : list, float
            Unbalance phase (rad)
        frequency_range : list, float
            Array with the desired range of frequencies

        Returns
        -------
        force_resp : array
            Array with the force response for each node for each frequency
        speed_range : array
            Array with the frequencies
        magdb : array
            Magnitude (dB) of the frequency response for each pair input/output.
            The order of the array is: [output, input, magnitude]
        phase : array
            Phase of the frequency response for each pair input/output.
            The order of the array is: [output, input, phase]

        Examples
        --------
        >>> rotor = rotor_example()
        >>> speed = np.linspace(0, 1000, 101)
        >>> response = rotor.unbalance_response(node=3, magnitude=10.0, phase=0.0, frequency_range=speed)
        >>> response.magnitude # doctest: +ELLIPSIS
        array([[0.00000000e+00, 5.06073311e-04, 2.10044826e-03, ...
        """
        force = np.zeros((self.ndof, len(frequency_range)), dtype=np.complex)

        try:
            for n, m, p in zip(node, magnitude, phase):
                force += self._unbalance_force(n, m, p, frequency_range)
        except TypeError:
            force = self._unbalance_force(node, magnitude, phase, frequency_range)

        forced_response = self.run_forced_response(force, frequency_range)

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
        modal = self.run_modal(speed=speed)
        return signal.lsim(modal.lti, F, t, X0=ic)

    def _plot_rotor_matplotlib(self, nodes=1, check_sld=False, ax=None):
        """Plots a rotor object.

        This function will take a rotor object and plot its shaft,
        disks and bearing elements

        Parameters
        ----------
        nodes : int, optional
            Increment that will be used to plot nodes label.
        check_sld : bool
            If True, checks the slenderness ratio for each element
        ax : matplotlib plotting axes, optional
            Axes in which the plot will be drawn.

        Returns
        -------
        ax : matplotlib axes
            Returns the axes object with the plot.

        Example
        -------
        >>> import ross as rs
        >>> rotor = rs.rotor_example()
        >>> rotor._plot_rotor_matplotlib() # doctest: +ELLIPSIS
        <matplotlib.axes...
        """
        if ax is None:
            ax = plt.gca()

        #  plot shaft centerline
        shaft_end = max(self.nodes_pos)
        ax.plot([-0.2 * shaft_end, 1.2 * shaft_end], [0, 0], "k-.")

        try:
            max_diameter = max([disk.o_d for disk in self.disk_elements])
        except (ValueError, AttributeError):
            max_diameter = max([shaft.odl for shaft in self.shaft_elements])

        ax.set_ylim(-1.2 * max_diameter, 1.2 * max_diameter)
        ax.axis("equal")
        ax.set_xlabel("Axial location (m)")
        ax.set_ylabel("Shaft radius (m)")

        # plot nodes
        for node, position in enumerate(self.nodes_pos[::nodes]):
            ax.plot(
                position,
                0,
                zorder=2,
                ls="",
                marker="D",
                color="#6caed6",
                markersize=10,
                alpha=0.6,
            )
            ax.text(
                position,
                0,
                f"{node*nodes}",
                size="smaller",
                horizontalalignment="center",
                verticalalignment="center",
            )

        # plot shaft elements
        for sh_elm in self.shaft_elements:
            position = self.nodes_pos[sh_elm.n]
            sh_elm.patch(position, check_sld, ax)

        mean_od = np.mean(self.nodes_o_d)
        # plot disk elements
        for disk in self.disk_elements:
            position = (self.nodes_pos[disk.n], self.nodes_o_d[disk.n] / 2, mean_od)
            disk.patch(position, ax)

        # plot bearings
        for bearing in self.bearing_elements:
            z_pos = self.df[self.df.tag == bearing.tag]["nodes_pos_l"].values[0]
            y_pos = self.df[self.df.tag == bearing.tag]["y_pos"].values[0]
            y_pos_sup = self.df[self.df.tag == bearing.tag]["y_pos_sup"].values[0]
            position = (z_pos, y_pos, y_pos_sup)
            bearing.patch(position, ax)

        # plot point mass
        for p_mass in self.point_mass_elements:
            z_pos = self.df[self.df.tag == p_mass.tag]["nodes_pos_l"].values[0]
            y_pos = self.df[self.df.tag == p_mass.tag]["y_pos"].values[0]
            position = (z_pos, y_pos)
            p_mass.patch(position, ax)

        return ax

    def _plot_rotor_bokeh(self, nodes=1, check_sld=False, bk_ax=None):
        """Plots a rotor object.

        This function will take a rotor object and plot its shaft,
        disks and bearing elements

        Parameters
        ----------
        nodes : int, optional
            Increment that will be used to plot nodes label.
        check_sld : bool
            If True, checks the slenderness ratio for each element
        bk_ax : bokeh plotting axes, optional
            Axes in which the plot will be drawn.

        Returns
        -------
        bk_ax : bokeh plotting axes
            Returns the axes object with the plot.

        Example
        -------
        >>> import ross as rs
        >>> rotor = rs.rotor_example()
        >>> figure = rotor._plot_rotor_bokeh()
        """
        #  plot shaft centerline
        shaft_end = max(self.nodes_pos)

        # bokeh plot - create a new plot
        bk_ax = figure(
            tools="pan, wheel_zoom, reset, save",
            width=800,
            height=600,
            x_range=[-0.1 * shaft_end, 1.1 * shaft_end],
            y_range=[-0.3 * shaft_end, 0.3 * shaft_end],
            title="Rotor model",
            x_axis_label="Axial location (m)",
            y_axis_label="Shaft radius (m)",
            match_aspect=True,
        )
        bk_ax.xaxis.axis_label_text_font_size = "14pt"
        bk_ax.yaxis.axis_label_text_font_size = "14pt"

        # bokeh plot - plot shaft centerline
        bk_ax.line(
            [-0.2 * shaft_end, 1.2 * shaft_end],
            [0, 0],
            line_width=3,
            line_dash="dotdash",
            line_color=bokeh_colors[0],
        )

        # plot nodes
        text = []
        x_pos = []
        for node, position in enumerate(self.nodes_pos[::nodes]):
            # bokeh plot
            text.append(str(node * nodes))
            x_pos.append(position)

        # bokeh plot - plot nodes
        y_pos = np.linspace(0, 0, len(self.nodes_pos[::nodes]))

        source = ColumnDataSource(dict(x=x_pos, y=y_pos, text=text))

        bk_ax.circle(
            x=x_pos, y=y_pos, size=30, fill_alpha=0.8, fill_color=bokeh_colors[6]
        )

        glyph = Text(
            x="x",
            y="y",
            text="text",
            text_font_style="bold",
            text_baseline="middle",
            text_align="center",
            text_alpha=1.0,
            text_color=bokeh_colors[0],
        )
        bk_ax.add_glyph(source, glyph)

        # plot shaft elements
        for sh_elm in self.shaft_elements:
            position = self.nodes_pos[sh_elm.n]
            hover = sh_elm.bokeh_patch(position, check_sld, bk_ax)

        bk_ax.add_tools(hover)

        mean_od = np.mean(self.nodes_o_d)
        # plot disk elements
        for disk in self.disk_elements:
            position = (self.nodes_pos[disk.n], self.nodes_o_d[disk.n] / 2, mean_od)
            hover = disk.bokeh_patch(position, bk_ax)

        bk_ax.add_tools(hover)

        # plot bearings
        for bearing in self.bearing_elements:
            z_pos = self.df[self.df.tag == bearing.tag]["nodes_pos_l"].values[0]
            y_pos = self.df[self.df.tag == bearing.tag]["y_pos"].values[0]
            y_pos_sup = self.df[self.df.tag == bearing.tag]["y_pos_sup"].values[0]
            position = (z_pos, y_pos, y_pos_sup)
            bearing.bokeh_patch(position, bk_ax)

        # plot point mass
        for p_mass in self.point_mass_elements:
            z_pos = self.df[self.df.tag == p_mass.tag]["nodes_pos_l"].values[0]
            y_pos = self.df[self.df.tag == p_mass.tag]["y_pos"].values[0]
            position = (z_pos, y_pos)
            hover = p_mass.bokeh_patch(position, bk_ax)

        bk_ax.add_tools(hover)

        return bk_ax

    def plot_rotor(self, nodes=1, *args, plot_type="bokeh", **kwargs):
        """Plots a rotor object.

        This function will take a rotor object and plot its shaft,
        disks and bearing elements

        Parameters
        ----------
        nodes : int, optional
            Increment that will be used to plot nodes label.
        plot_type : str
            Matplotlib or bokeh.
            Default is matplotlib.
        ax : matplotlib axes, optional
            Axes in which the plot will be drawn.
        bk_ax : bokeh plotting axes, optional
            Axes in which the plot will be drawn.

        Returns
        -------
        ax : matplotlib axes
            Returns the axes object with the plot.
        bk_ax : bokeh plotting axes
            Returns the axes object with the plot.

        Examples:
        >>> import ross as rs
        >>> rotor = rs.rotor_example()
        >>> rotor.plot_rotor() # doctest: +ELLIPSIS
        Figure...
        """
        if plot_type == "matplotlib":
            return self._plot_rotor_matplotlib(
                nodes=nodes, check_sld=False, *args, **kwargs
            )
        elif plot_type == "bokeh":
            return self._plot_rotor_bokeh(nodes=nodes, check_sld=False, *args, **kwargs)
        else:
            raise ValueError(f"{plot_type} is not a valid plot type.")

    def check_slenderness_ratio(self, nodes=1, *args, plot_type="matplotlib", **kwargs):
        """Plots a rotor object and check the slenderness ratio

        Parameters
        ----------
        nodes : int, optional
            Increment that will be used to plot nodes label.
        plot_type : str
            Matplotlib or bokeh.
            Default is matplotlib.

        Returns
        -------
        ax : matplotlib axes
            Returns the axes object with the plot.
        bk_ax : bokeh plotting axes
            Returns the axes object with the plot.

        Example
        -------
        >>> import ross as rs
        >>> rotor = rs.rotor_example()
        >>> rotor.check_slenderness_ratio() # doctest: +ELLIPSIS
        <matplotlib.axes...
        """

        # check slenderness ratio of beam elements
        SR = np.array([])
        for shaft in self.shaft_elements:
            if shaft.slenderness_ratio < 1.6:
                SR = np.append(SR, shaft.n)
        if len(SR) != 0:
            warnings.warn(
                "The beam elements "
                + str(SR)
                + " have slenderness ratio (G*A*L^2 / EI) of less than 1.6."
                + " Results may not converge correctly"
            )

        if plot_type == "matplotlib":
            return self._plot_rotor_matplotlib(
                nodes=nodes, check_sld=True, *args, **kwargs
            )
        elif plot_type == "bokeh":
            return self._plot_rotor_bokeh(nodes=nodes, check_sld=True, *args, **kwargs)
        else:
            raise ValueError(f"{plot_type} is not a valid plot type.")

    def run_campbell(self, speed_range, frequencies=6, frequency_type="wd"):
        """Calculates the Campbell diagram.

        This function will calculate the damped natural frequencies
        for a speed range.

        Parameters
        ----------
        speed_range : array
            Array with the speed range in rad/s.
        frequencies : int, optional
            Number of frequencies that will be calculated.
            Default is 6.

        Returns
        -------
        results : array
            Array with the damped natural frequencies, log dec and precessions
            corresponding to each speed of the speed_rad array.
            It will be returned if plot=False.

        Examples
        --------
        >>> rotor1 = rotor_example()
        >>> speed = np.linspace(0, 400, 101)
        >>> camp = rotor1.run_campbell(speed)
        >>> camp.plot() # doctest: +ELLIPSIS
        Figure...
        """
        # store in results [speeds(x axis), frequencies[0] or logdec[1] or
        # whirl[2](y axis), 3]
        results = np.zeros([len(speed_range), frequencies, 5])

        for i, w in enumerate(speed_range):
            modal = self.run_modal(speed=w)

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

    def plot_ucs(self, stiffness_range=None, num=20, ax=None, output_html=False):
        """Plot undamped critical speed map.

        This method will plot the undamped critical speed map for a given range
        of stiffness values. If the range is not provided, the bearing
        stiffness at rated speed will be used to create a range.

        Parameters
        ----------
        stiffness_range : tuple, optional
            Tuple with (start, end) for stiffness range.
        num : int
            Number of steps in the range.
            Default is 20.
        ax : matplotlib axes, optional
            Axes in which the plot will be drawn.
        output_html : Boolean, optional
            outputs a html file.
            Default is False

        Returns
        -------
        ax : matplotlib axes
            Returns the axes object with the plot.
        bk_ax : bokeh plot axes
            Returns the axes object with the plot.

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
        >>> stfx = [1e6, 2e7, 3e8]
        >>> stfy = [0.8e6, 1.6e7, 2.4e8]
        >>> bearing0 = BearingElement(0, kxx=stfx, kyy=stfy, cxx=0, frequency=[0,1000, 2000])
        >>> bearing1 = BearingElement(6, kxx=stfx, kyy=stfy, cxx=0, frequency=[0,1000, 2000])
        >>> rotor = Rotor(shaft_elem, [disk0, disk1], [bearing0, bearing1])
        >>> rotor.plot_ucs() # doctest: +ELLIPSIS
        <matplotlib.axes._subplots.AxesSubplot ...
        """
        if ax is None:
            ax = plt.gca()

        if stiffness_range is None:
            if self.rated_w is not None:
                bearing = self.bearing_elements[0]
                k = bearing.kxx.interpolated(self.rated_w)
                k = int(np.log10(k))
                stiffness_range = (k - 3, k + 3)
            else:
                stiffness_range = (6, 11)

        stiffness_log = np.logspace(*stiffness_range, num=num)
        rotor_wn = np.zeros((4, len(stiffness_log)))

        bearings_elements = []  # exclude the seals
        for bearing in self.bearing_elements:
            if type(bearing) == BearingElement:
                bearings_elements.append(bearing)

        for i, k in enumerate(stiffness_log):
            bearings = [BearingElement(b.n, kxx=k, cxx=0) for b in bearings_elements]
            rotor = self.__class__(
                self.shaft_elements, self.disk_elements, bearings, n_eigen=16
            )
            modal = rotor.run_modal(speed=0)
            rotor_wn[:, i] = modal.wn[:8:2]

        ax.set_prop_cycle(cycler("color", seaborn_colors))
        ax.loglog(stiffness_log, rotor_wn.T)
        ax.set_xlabel("Bearing Stiffness (N/m)")
        ax.set_ylabel("Critical Speed (rad/s)")

        bearing0 = bearings_elements[0]

        ax.plot(
            bearing0.kxx.interpolated(bearing0.frequency),
            bearing0.frequency,
            marker="o",
            color="k",
            alpha=0.25,
            markersize=5,
            lw=0,
            label="kxx",
        )
        ax.plot(
            bearing0.kyy.interpolated(bearing0.frequency),
            bearing0.frequency,
            marker="s",
            color="k",
            alpha=0.25,
            markersize=5,
            lw=0,
            label="kyy",
        )
        ax.legend()

        # bokeh plot - output to static HTML file
        if output_html:
            output_file("Plot_UCS.html")

        # bokeh plot - create a new plot
        bk_ax = figure(
            tools="pan, box_zoom, wheel_zoom, reset, save",
            width=1200,
            height=900,
            title="Undamped critical speed map",
            x_axis_label="Bearing Stiffness (N/m)",
            y_axis_label="Critical Speed (rad/s)",
            x_axis_type="log",
            y_axis_type="log",
        )
        bk_ax.xaxis.axis_label_text_font_size = "14pt"
        bk_ax.yaxis.axis_label_text_font_size = "14pt"

        # bokeh plot - plot shaft centerline
        bk_ax.circle(
            bearing0.kxx.interpolated(bearing0.frequency),
            bearing0.frequency,
            size=5,
            fill_alpha=0.5,
            fill_color=bokeh_colors[0],
            legend_label="Kxx",
        )
        bk_ax.square(
            bearing0.kyy.interpolated(bearing0.frequency),
            bearing0.frequency,
            size=5,
            fill_alpha=0.5,
            fill_color=bokeh_colors[0],
            legend_label="Kyy",
        )
        for j in range(rotor_wn.T.shape[1]):
            bk_ax.line(
                stiffness_log,
                np.transpose(rotor_wn.T)[j],
                line_width=3,
                line_color=bokeh_colors[-j + 1],
            )

        return ax

    def plot_level1(
        self, n=None, stiffness_range=None, num=5, ax=None, output_html=False, **kwargs
    ):
        """Plot level 1 stability analysis.

        This method will plot the stability 1 analysis for a
        given stiffness range.

        Parameters
        ----------
        stiffness_range : tuple, optional
            Tuple with (start, end) for stiffness range.
        num : int
            Number of steps in the range.
            Default is 5.
        ax : matplotlib axes, optional
            Axes in which the plot will be drawn.
        output_html : Boolean, optional
            outputs a html file.
            Default is False

        Returns
        -------
        ax : matplotlib axes
            Returns the axes object with the plot.
        bk_ax : bokeh plot axes
            Returns the axes object with the plot.
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
        >>> rotor.plot_level1(n=0, stiffness_range=(1e6, 1e11)) # doctest: +ELLIPSIS
        (<matplotlib.axes._subplots.AxesSubplot ...
        """
        if ax is None:
            ax = plt.gca()

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

        ax.plot(stiffness, log_dec, "--", **kwargs)
        ax.set_xlabel("Applied Cross Coupled Stiffness, Q (N/m)")
        ax.set_ylabel("Log Dec")

        # bokeh plot - output to static HTML file
        if output_html:
            output_file("Plot_level1.html")

        # bokeh plot - create a new plot
        bk_ax = figure(
            tools="pan, box_zoom, wheel_zoom, reset, save",
            width=1200,
            height=900,
            title="Level 1 stability analysis",
            x_axis_label="Applied Cross Coupled Stiffness, Q (N/m)",
            y_axis_label="Log Dec",
        )
        bk_ax.xaxis.axis_label_text_font_size = "14pt"
        bk_ax.yaxis.axis_label_text_font_size = "14pt"

        # bokeh plot - plot shaft centerline
        bk_ax.line(stiffness, log_dec, line_width=3, line_color=bokeh_colors[0])

        return ax, bk_ax

    def run_time_response(self, speed, F, t, dof):
        """Calculates the time response.

        This function will take a rotor object and plot its time response
        given a force and a time.

        Parameters
        ----------
        F : array
            Force array (needs to have the same number of rows as time array).
            Each column corresponds to a dof and each row to a time.
        t : array
            Time array.
        dof : int
            Degree of freedom that will be observed.

        Returns
        -------
        results : array
            Array containing the time array, the system response, and the
            time evolution of the state vector.
            It will be returned if plot=False.

        Examples
        --------
        >>> rotor = rotor_example()
        >>> speed = 0
        >>> size = 28
        >>> t = np.linspace(0, 5, size)
        >>> F = np.ones((size, rotor.ndof))
        >>> dof = 13
        >>> response = rotor.run_time_response(speed, F, t, dof)
        >>> response.yout[:, dof] # doctest: +ELLIPSIS
        array([ 0.00000000e+00,  1.06327334e-05,  1.54684988e-05, ...
        """
        t_, yout, xout = self.time_response(speed, F, t)

        results = TimeResponseResults(t, yout, xout, dof)

        return results

    def run_orbit_response(self, speed, F, t):
        """Calculates the orbit for a given node.

        This function will take a rotor object and plot the orbit for a single
        (2D graph) or all nodes (3D graph).

        Parameters
        ----------
        speed: float
            Rotor speed
        F: array
            Force array (needs to have the same number of rows as time array).
            Each column corresponds to a dof and each row to a time.
        t: array
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
        >>> t = np.linspace(0, 10, size)
        >>> F = np.zeros((size, rotor.ndof))
        >>> F[:, 4 * node] = 10 * np.cos(2 * t)
        >>> F[:, 4 * node + 1] = 10 * np.sin(2 * t)
        >>> response = rotor.run_orbit_response(speed, F, t)
        >>> response.yout[:, 4 * node] # doctest: +ELLIPSIS
        array([ 0.00000000e+00,  6.94968863e-06,  2.13014440e-05, ...
        """
        t_, yout, xout = self.time_response(speed, F, t)

        results = OrbitResponseResults(t, yout, xout, self.nodes, self.nodes_pos)

        return results

    def save_mat(self, file_path, speed, frequency=None):
        """Save matrices and rotor model to a .mat file.

        Parameters
        ----------
        file_path : str

        speed: float
            Rotor speed.
        frequency: float, optional
            Excitation frequency.
            Default is rotor speed.

        Examples
        --------
        >>> rotor = rotor_example()
        >>> rotor.save_mat('new_matrices.mat', speed=0)
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

        sio.savemat("%s/%s.mat" % (os.getcwd(), file_path), dic)

    def save(self, rotor_name="rotor", file_path=Path(".")):
        """Save rotor to toml file.

        Parameters
        ----------
        file_path : str

        Examples
        --------
        >>> rotor = rotor_example()
        >>> rotor.save('new_rotor')
        >>> Rotor.remove('new_rotor')
        """
        path_rotor = Path(file_path)

        if os.path.isdir(path_rotor / rotor_name):
            if int(
                input(
                    "There is a rotor with this file_path, do you want to overwrite it? (1 for yes and 0 for no)"
                )
            ):
                shutil.rmtree(path_rotor / rotor_name)
            else:
                return "The rotor was not saved."

        os.mkdir(path_rotor / rotor_name)
        rotor_folder = path_rotor / rotor_name
        os.mkdir(rotor_folder / "results")
        os.mkdir(rotor_folder / "elements")

        with open(rotor_folder / "properties.toml", "w") as f:
            toml.dump({"parameters": self.parameters}, f)

        elements_folder = rotor_folder / "elements"

        for element in self.elements:
            element.save(elements_folder)

    @staticmethod
    def load(file_path):
        """Load rotor from toml file.

        Parameters
        ----------
        file_path : str

        Returns
        -------
        rotor : ross.rotor.Rotor

        Example
        -------
        >>> rotor1 = rotor_example()
        >>> rotor1.save(Path('.')/'new_rotor1')
        >>> rotor2 = Rotor.load(Path('.')/'new_rotor1')
        >>> rotor1 == rotor2
        True
        >>> Rotor.remove('new_rotor1')
        """
        rotor_path = Path(file_path)

        if os.path.isdir(rotor_path / "elements"):
            elements_path = rotor_path / "elements"
        else:
            raise FileNotFoundError("Elements folder not found.")

        with open(rotor_path / "properties.toml", "r") as f:
            parameters = toml.load(f)["parameters"]

        global_elements = {}
        for el in os.listdir(elements_path):
            elements = []
            if ".toml" in el:
                with open(Path(elements_path) / el, "r") as f:
                    el_dict = toml.load(f)
                    element_class = list(el_dict.keys())[0]
                    for el_number in el_dict[element_class]:
                        element = (
                            element_class + f"(**{el_dict[element_class][el_number]})"
                        )
                        elements.append(eval(element))
            global_elements[convert(element_class + "s")] = elements

        return Rotor(**global_elements, **parameters)

    @staticmethod
    def remove(file_path):
        """
        Remove a previously saved rotor in rotors folder.

        Parameters
        ----------
        file_path : str

        Example
        -------
        >>> rotor = rotor_example()
        >>> rotor.save('new_rotor2')
        >>> Rotor.remove('new_rotor2')
        """
        try:
            Rotor.load(file_path)
            shutil.rmtree(Path(file_path))
        except:
            return "This is not a valid rotor."

    def run_static(self):
        """Rotor static analysis.
        Static analysis calculates free-body diagram, deformed shaft, shearing
        force diagram and bending moment diagram.

        Parameters
        ----------

        Attributes
        ----------
        shaft_weight: float
            Shaft total weight
        disk_weigth_force: list
            Weight forces of each disk
        bearing_reaction_force: list
            The static reaction forces on each bearing
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

        Example
        -------
        >>> rotor = rotor_example()
        >>> static = rotor.run_static()
        >>> rotor.bearing_forces_nodal
        {'node_0': 432.4, 'node_6': 432.4}
        >>> rotor.bearing_forces_tag
        {'Bearing 0': 432.4, 'Bearing 1': 432.4}
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
        aux_M = aux_rotor.M()

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
        g = 9.8065
        grav = np.zeros(len(aux_K))

        # place gravity effect on shaft and disks nodes
        for node_y in range(int(len(aux_K) / 4)):
            grav[4 * node_y + 1] = -g

        # calculates x, for [K]*(x) = [M]*(g)
        disp = (la.solve(aux_K, aux_M @ grav)).flatten()

        # calculates displacement values in gravity's direction
        # dof = degree of freedom
        disp_y = np.array([])
        for node_dof in range(int(len(disp) / 4)):
            disp_y = np.append(disp_y, disp[4 * node_dof + 1])

        # Shearing Force
        BRG = [0] * len(self.nodes_pos)
        DSK = [0] * len(self.nodes_pos)
        SCH = [0] * len(self.nodes_pos)
        BrgForce_nodal = {"node_" + str(i): 0 for i in self.nodes}
        DskForce_nodal = {"node_" + str(i): 0 for i in self.nodes}
        BrgForce_tag = {"node_" + str(i): 0 for i in self.nodes}
        DskForce_tag = {"node_" + str(i): 0 for i in self.nodes}

        # Bearing Forces
        for i, node in enumerate(aux_rotor.df_bearings["n"]):
            if not pd.isna(aux_rotor.df_bearings.loc[i, "n_link"]):
                BRG[node] = (
                    BRG[node]
                    + disp_y[node]
                    * self.df_bearings.loc[
                        self.df_bearings.tag == aux_rotor.df_bearings.tag, "kyy"
                    ][0].coefficient[0]
                )
                BrgForce_nodal["node_" + str(node)] = np.around(
                    BrgForce_nodal["node_" + str(node)]
                    + disp_y[node]
                    * self.df_bearings.loc[
                        self.df_bearings.tag == aux_rotor.df_bearings.tag, "kyy"
                    ][0].coefficient[0],
                    decimals=1,
                )
                BrgForce_tag[aux_rotor.df_bearings.loc[i, "tag"]] = BrgForce_nodal[
                    "node_" + str(node)
                ]

                node = int(aux_rotor.df_bearings.loc[i, "n_link"])
                BRG[node] = (
                    BRG[node]
                    - disp_y[node]
                    * self.df_bearings.loc[self.df_bearings.n_link == node, "kyy"]
                    .values[0]
                    .coefficient[0]
                )
                BrgForce_nodal["node_" + str(node)] = np.around(
                    BrgForce_nodal["node_" + str(node)]
                    - disp_y[node]
                    * self.df_bearings.loc[self.df_bearings.n_link == node, "kyy"]
                    .values[0]
                    .coefficient[0],
                    decimals=1,
                )
                BrgForce_tag[aux_rotor.df_bearings.loc[i, "tag"]] = BrgForce_nodal[
                    "node_" + str(node)
                ]

            else:
                BRG[node] = (
                    BRG[node]
                    - disp_y[node] * aux_rotor.df_bearings.loc[i, "kyy"].coefficient[0]
                )
                BrgForce_nodal["node_" + str(node)] = np.around(
                    BrgForce_nodal["node_" + str(node)]
                    - disp_y[node] * aux_rotor.df_bearings.loc[i, "kyy"].coefficient[0],
                    decimals=1,
                )
                BrgForce_tag[aux_rotor.df_bearings.loc[i, "tag"]] = BrgForce_nodal[
                    "node_" + str(node)
                ]

        # counting nodes with more than 1 bearing attached to
        node_b = list(aux_rotor.df_bearings["n"])
        node_b.extend(list(aux_rotor.df_bearings["n_link"]))
        count = len(node_b) - len(Counter(node_b))

        # Disk Forces
        if len(self.df_disks):
            for i, node in enumerate(self.df_disks["n"]):
                DSK[node] = self.df_disks.loc[i, "m"] * -g
                DskForce_nodal["node_" + str(node)] = np.around(
                    self.df_disks.loc[i, "m"] * -g, decimals=1
                )
                DskForce_tag[aux_rotor.df_disks.loc[i, "tag"]] = DskForce_nodal[
                    "node_" + str(node)
                ]

        # Shaft Weight Forces
        for i, node in enumerate(self.df_shaft["_n"]):
            SCH[node + 1] = self.df_shaft.loc[i, "m"] * -g

        # Organizing data for each shaft
        BrgForce = []
        DskForce = []
        SchForce = []
        nodes = []
        nodes_pos = []
        displacement = []
        dsk = []
        brg = []
        for i in sh_num:
            n_min = min(
                aux_rotor.df_shaft.loc[aux_rotor.df_shaft.shaft_number == i, "n_l"]
            )
            n_max = max(
                aux_rotor.df_shaft.loc[(aux_rotor.df_shaft.shaft_number == i), "n_r"]
            )
            BrgForce.append(BRG[n_min : n_max + 1])
            DskForce.append(DSK[n_min : n_max + 1])
            SchForce.append(SCH[n_min : n_max + 1])
            nodes_pos.append(self.nodes_pos[n_min : n_max + 1])
            displacement.append(disp_y[n_min : n_max + 1])
            nodes.append(list(range(n_min, n_max + 1)))

            # get bearings and disks for each shaft
            dsk.append(
                aux_rotor.df_disks.loc[
                    aux_rotor.df_disks.shaft_number == i, "tag"
                ].values
            )
            brg.append(
                aux_rotor.df_bearings.loc[
                    (aux_rotor.df_bearings.shaft_number == i)
                    | (
                        aux_rotor.df_bearings.n_link.isin(list(range(n_min, n_max + 1)))
                    ),
                    "tag",
                ].values
            )

        Mx = []
        Vx = []
        Bm = []
        Vx_axis = []
        for j in sh_num:
            # Shearing Force vector
            aux_Vx = [0] * (len(nodes_pos[j]))
            aux_Vx_axis = [0] * (len(nodes_pos[j]))

            for i in range(int(len(nodes_pos[j]))):
                aux_Vx_axis[i] = nodes_pos[j][i]
                aux_Vx[i] = (
                    aux_Vx[i - 1] + BrgForce[j][i] + DskForce[j][i] + SchForce[j][i]
                )

            for i in range(len(aux_Vx_axis) + len(dsk[j]) + len(brg[j]) - count):
                if DskForce[j][i] != 0:
                    aux_Vx.insert(i, aux_Vx[i - 1] + SchForce[j][i])
                    DskForce[j].insert(i + 1, 0)
                    SchForce[j].insert(i + 1, 0)
                    BrgForce[j].insert(i + 1, 0)
                    aux_Vx_axis.insert(i, aux_Vx_axis[i])

                if BrgForce[j][i] != 0:
                    aux_Vx.insert(i, aux_Vx[i - 1] + SchForce[j][i])
                    BrgForce[j].insert(i + 1, 0)
                    DskForce[j].insert(i + 1, 0)
                    SchForce[j].insert(i + 1, 0)
                    aux_Vx_axis.insert(i, aux_Vx_axis[i])

            aux_Vx = [x * -1 for x in aux_Vx]
            Vx.append(np.array(aux_Vx))
            Vx_axis.append(np.array(aux_Vx_axis))

            # Bending Moment vector
            aux_Mx = []
            for i in range(len(aux_Vx) - 1):
                if aux_Vx_axis[i] == aux_Vx_axis[i + 1]:
                    pass
                else:
                    aux_Mx.append(
                        (
                            (aux_Vx_axis[i + 1] * aux_Vx[i + 1])
                            + (aux_Vx_axis[i + 1] * aux_Vx[i])
                            - (aux_Vx_axis[i] * aux_Vx[i + 1])
                            - (aux_Vx_axis[i] * aux_Vx[i])
                        )
                        / 2
                    )
            Mx.append(aux_Mx)

            aux_Bm = np.zeros(1)
            for i in range(len(Mx[j])):
                aux_Bm = np.append(aux_Bm, aux_Bm[i] + aux_Mx[i])
            Bm.append(aux_Bm)

        self.Vx = Vx
        self.Bm = Bm
        self.disp_y = displacement

        self.w_shaft = [
            sum(self.df_shaft.loc[self.df_shaft.shaft_number == i, "m"]) * g
            for i in sh_num
        ]

        DskForce_nodal = {k: v for k, v in DskForce_nodal.items() if v != 0}
        BrgForce_nodal = {k: v for k, v in BrgForce_nodal.items() if v != 0}
        BrgForce_tag = {k: v for k, v in BrgForce_tag.items() if v != 0}
        DskForce_tag = {k: v for k, v in DskForce_tag.items() if v != 0}

        self.disk_forces_nodal = DskForce_nodal
        self.bearing_forces_nodal = BrgForce_nodal
        self.bearing_forces_tag = BrgForce_tag
        self.disk_forces_tag = DskForce_tag

        results = StaticResults(
            self.disp_y,
            self.Vx,
            self.Bm,
            self.w_shaft,
            self.disk_forces_nodal,
            self.bearing_forces_nodal,
            nodes,
            nodes_pos,
            Vx_axis,
        )

        return results

    def summary(self):
        """Rotor summary.

        This creates a summary of the main parameters and attributes from the
        rotor model. The data is presented in a table format.

        Parameters
        ----------

        Returns
        -------
        results : class instance
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
        sparse=True,
        min_w=None,
        max_w=None,
        rated_w=None,
        n_eigen=12,
        nel_r=1,
        tag=None,
    ):
        """This class is an alternative to build rotors from separated
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
        n_eigen : int, optional
            Number of eigenvalues calculated by arpack.
            Default is 12.
        tag : str
            A tag for the rotor

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
            """
            A subroutine to discretize each rotor region into n elements

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
            sparse=sparse,
            n_eigen=n_eigen,
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
    sparse : bool, optional
        If sparse, eigenvalues will be calculated with arpack.
        Default is True.
    n_eigen : int, optional
        Number of eigenvalues calculated by arpack.
        Default is 12.
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
        sparse=True,
        n_eigen=12,
        min_w=None,
        max_w=None,
        rated_w=None,
        tag=None,
    ):

        self.parameters = {
            "sparse": True,
            "n_eigen": n_eigen,
            "min_w": min_w,
            "max_w": max_w,
            "rated_w": rated_w,
        }
        if tag is None:
            self.tag = "Rotor 0"

        ####################################################
        # Config attributes
        ####################################################

        self.sparse = sparse
        self.n_eigen = n_eigen
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

        # values for evalues and evectors will be calculated by self.run_modal
        self.evalues = None
        self.evectors = None
        self.wn = None
        self.wd = None
        self.lti = None

        self._v0 = None  # used to call eigs

        # number of dofs
        self.ndof = (
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

        #  values for static analysis will be calculated by def static
        self.Vx = None
        self.Bm = None
        self.disp_y = None

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
    """This function returns an instance of a simple rotor with
    two shaft elements, one disk and two simple bearings.
    The purpose of this is to make available a simple model
    so that doctest can be written using this.

    Parameters
    ----------

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
    """This function returns an instance of a simple rotor with
    two shafts, four disk and four bearings.
    The purpose of this is to make available a simple model for co-axial rotors
    so that doctest can be written using this.

    Parameters
    ----------

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
        n=1, material=steel, width=0.07, i_d=0.05, o_d=0.28
    )
    disk1 = DiskElement.from_geometry(
        n=9, material=steel, width=0.07, i_d=0.05, o_d=0.28
    )
    disk2 = DiskElement.from_geometry(
        n=13, material=steel, width=0.07, i_d=0.20, o_d=0.48
    )
    disk3 = DiskElement.from_geometry(
        n=15, material=steel, width=0.07, i_d=0.20, o_d=0.48
    )

    shaft = [axial_shaft, coaxial_shaft]
    disks = [disk0, disk1, disk2, disk3]

    stfx = 1e6
    stfy = 1e6
    bearing0 = BearingElement(0, kxx=stfx, kyy=stfy, cxx=0)
    bearing1 = BearingElement(10, kxx=stfx, kyy=stfy, cxx=0)
    bearing2 = BearingElement(11, kxx=stfx, kyy=stfy, cxx=0)
    bearing3 = BearingElement(8, n_link=17, kxx=stfx, kyy=stfy, cxx=0)
    bearings = [bearing0, bearing1, bearing2, bearing3]

    return CoAxialRotor(shaft, disks, bearings)


def MAC(u, v):
    """MAC - Modal Assurance Criterion

    MAC for a single pair of vectors.
    The Modal Assurance Criterion (MAC) analysis is used to determine
    the similarity of two mode shapes.

    Parameters
    ----------
    u : array
        complex modal vector
    v : array
        complex modal vector

    Returns
    -------
    MAC from 'u' and 'v'
    """
    H = lambda a: a.T.conj()
    return np.absolute((H(u) @ v) ** 2 / ((H(u) @ u) * (H(v) @ v)))


def MAC_modes(U, V, n=None, plot=True):
    """MAC - Modal Assurance Criterion

    MAC for multiple vectors
    The Modal Assurance Criterion (MAC) analysis is used to determine
    the similarity of two mode shapes.

    Parameters
    ----------
    U : matrix
        complex modal matrix
    V : matrix
        complex modal matrix
    n : int
        number of vectors to be analyzed
    plot : bool
        if True, returns a plot
        if False, returns the macs values

    Returns
    -------
    The macs values from 'U' and 'V'
    """
    # n is the number of modes to be evaluated
    if n is None:
        n = U.shape[1]
    macs = np.zeros((n, n))
    for u in enumerate(U.T[:n]):
        for v in enumerate(V.T[:n]):
            macs[u[0], v[0]] = MAC(u[1], v[1])

    if not plot:
        return macs

    xpos, ypos = np.meshgrid(range(n), range(n))
    xpos, ypos = 0.5 + xpos.flatten(), 0.5 + ypos.flatten()
    zpos = np.zeros_like(xpos)
    dx = 0.75 * np.ones_like(xpos)
    dy = 0.75 * np.ones_like(xpos)
    dz = macs.T.flatten()

    fig = plt.figure(figsize=(12, 8))
    # fig.suptitle('MAC - %s vs %s' % (U.name, V.name), fontsize=12)
    ax = fig.add_subplot(111, projection="3d")
    ax.bar3d(
        xpos, ypos, zpos, dx, dy, dz, color=plt.cm.viridis(dz), alpha=0.7, zsort="max"
    )
    ax.set_xticks(range(1, n + 1))
    ax.set_yticks(range(1, n + 1))
    ax.set_zlim(0, 1)
    # ax.set_xlabel('%s  modes' % U.name)
    # ax.set_ylabel('%s  modes' % V.name)

    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=1))
    # fake up the array of the scalar mappable
    sm._A = []
    cbar = fig.colorbar(sm, shrink=0.5, aspect=10)
    cbar.set_label("MAC")

    return macs
