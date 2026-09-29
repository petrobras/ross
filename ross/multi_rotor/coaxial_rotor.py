import numpy as np
import pandas as pd
import ross as rs

from collections.abc import Iterable
from itertools import chain, cycle
from copy import copy

from ross.rotor_assembly import Rotor
from ross.bearing_seal_element import BearingElement, SealElement
from ross.results import ForcedResponseResults
from ross.units import check_units
from ross.utils import make_speed_array

__all__ = ["CoAxialRotor", "coaxrotor_example"]


class CoAxialRotor(Rotor):
    """A rotor object.

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
    speed_ratio: float, optional
        Speed ratio of the outer rotor with respect to the inner rotor
    modal_damping_ratio: list, optional
        List of modal damping ratio(s) for the first modes
    default_damping_ratio: float, optional
        Default modal damping ratio for the remaining modes.
        Default is zero.
    alpha : float, optional
        Mass proportional damping factor.
        Default is zero.
    beta : float, optional
        Stiffness proportional damping factor.
        Default is zero.
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
        speed_ratio=1,
        min_w=None,
        max_w=None,
        rated_w=None,
        modal_damping_ratio=None,
        default_damping_ratio=0.0,
        alpha=0.0,
        beta=0.0,
        tag=None,
    ):
        self.parameters = {
            "min_w": min_w,
            "max_w": max_w,
            "rated_w": rated_w,
            "modal_damping_ratio": (
                None
                if modal_damping_ratio is None
                else [float(xi) for xi in np.atleast_1d(modal_damping_ratio)]
            ),
            "default_damping_ratio": float(default_damping_ratio),
            "alpha": float(alpha) if alpha is not None else 0.0,
            "beta": float(beta) if beta is not None else 0.0,
        }

        self.set_tag(tag)

        ####################################################
        # Config attributes
        ####################################################

        # operational speeds
        self.speed_ratio = speed_ratio
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

        # set n for each shaft element
        aux_n = 0
        for shaft in shafts:
            for i, sh in enumerate(shaft):
                if sh.n is None:
                    sh.n = i + aux_n
            aux_n = shaft[-1].n_r + 1

        # flatten and make a copy for shaft elements to avoid altering
        # attributes for elements that might be used in different rotors
        # e.g. altering shaft_element.n
        shafts = [copy(sh) for sh in shafts]
        shaft_elements = list(chain(*shafts))

        for i, sh in enumerate(shaft_elements):
            sh.set_tag(i)

        if disk_elements is None:
            disk_elements = []
        if bearing_elements is None:
            bearing_elements = []
        if point_mass_elements is None:
            point_mass_elements = []

        elm_dict = {}
        for elm in disk_elements + bearing_elements + point_mass_elements:
            class_name = elm.__class__.__name__
            elm_dict[class_name] = elm_dict.get(class_name, 0) + 1

            elm.set_tag(elm_dict[class_name] - 1)

            if isinstance(elm, BearingElement):
                # add n_l and n_r to bearing elements
                elm.n_l = elm.n
                elm.n_r = elm.n

        self.shafts = shafts
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
                    shift = nodes_pos_l[idx] - nodes_pos_l[k + i]
                    nodes_pos_l[i : sh.n] += shift
                    nodes_pos_r[i : sh.n] += shift
                    axial_cg_pos[i : sh.n] += shift

                elif sh.n_r in df_bearings["n_link"].values:
                    idx = df_bearings.loc[df_bearings.n_link == sh.n_r, "n"].values[0]
                    shift = nodes_pos_r[idx - 1] - nodes_pos_r[k + i]
                    nodes_pos_l[i : sh.n_r] += shift
                    nodes_pos_r[i : sh.n_r] += shift
                    axial_cg_pos[i : sh.n_r] += shift

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

        self.inner_nodes = sorted({n for sh in self.shafts[0] for n in (sh.n, sh.n_r)})
        self.outer_nodes = sorted({n for sh in self.shafts[1] for n in (sh.n, sh.n_r)})

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

        self.inner_dofs = self._get_inner_global_dofs(self.shaft_elements)
        self.outer_dofs = self._get_outer_global_dofs(self.shaft_elements)

        # define positions for disks
        for disk in disk_elements:
            z_pos = nodes_pos[disk.n]
            y_pos = nodes_o_d[disk.n] / 2.0
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
            for n, t, nlink in zip(
                dfb_z_pos.n, dfb_z_pos.tag, dfb_z_pos.n_link, strict=True
            ):
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
        for pm in point_mass_elements:
            dfb_pm = dfb[dfb.n_l == pm.n]

            if not dfb_pm.empty:
                z_pos = dfb_pm["nodes_pos_l"].values[0]
                y_pos = dfb_pm["y_pos"].values[0]
            else:
                i = self.nodes.index(pm.n)
                z_pos = nodes_pos[i]
                y_pos = nodes_o_d[i] / 2

            df.loc[df.tag == pm.tag, "nodes_pos_l"] = z_pos
            df.loc[df.tag == pm.tag, "nodes_pos_r"] = z_pos
            df.loc[df.tag == pm.tag, "y_pos"] = y_pos

        self.df = df

        # Base matrices:
        self._build_base_matrices(
            modal_damping_ratio, default_damping_ratio, alpha, beta
        )
    
    def _get_inner_elements(self, elements=None):
        elements = elements or self.elements

        return [el for el in elements if el.n in self.inner_nodes]
    
    def _get_outer_elements(self, elements=None):
        elements = elements or self.elements

        return [el for el in elements if el.n in self.outer_nodes]

    def _get_inner_global_dofs(self, elements=None):
        if elements is None:
            return self.inner_dofs
        else:
            return sorted(
                {
                    dof
                    for el in elements
                    if el.n in self.inner_nodes
                    for dof in el.dof_global_index.values()
                }
            )

    def _get_outer_global_dofs(self, elements=None):
        if elements is None:
            return self.outer_dofs
        else:
            return sorted(
                {
                    dof
                    for el in elements
                    if el.n in self.outer_nodes
                    for dof in el.dof_global_index.values()
                }
            )

    def G(self):
        G0 = self.G0.copy()
        dofs = self.outer_dofs

        G0[np.ix_(dofs, dofs)] *= self.speed_ratio

        return G0


    def _node_speed_ratio(self, node):
        """Return the speed ratio of the shaft the node belongs to.

        Parameters
        ----------
        node : int
            Node index.

        Returns
        -------
        ratio : float
            ``speed_ratio`` for a node on the outer shaft, 1 otherwise.
        """
        return self.speed_ratio if node in self.outer_nodes else 1

    def _unbalance_force(self, node, magnitude, phase, omega):
        """Calculate unbalance forces at the node's own excitation frequency.

        The force rotates with the shaft the node belongs to, so its
        frequency is ``abs(ratio) * omega``. For a counter-rotating shaft the
        force whirls backward, which flips the phase and the sign of the y
        component.

        Parameters
        ----------
        node : int
            Node where the unbalance is applied.
        magnitude : float
            Unbalance magnitude (kg.m).
        phase : float
            Unbalance phase (rad).
        omega : list, float
            Inner shaft speeds (rad/s).

        Returns
        -------
        F0 : np.ndarray
            Unbalance force in each degree of freedom for each value in omega.
        """
        ratio = self._node_speed_ratio(node)
        frequency = abs(ratio) * np.asarray(omega)

        if ratio >= 0:
            return super()._unbalance_force(node, magnitude, phase, frequency)
        else:
            F0 = super()._unbalance_force(node, magnitude, -phase, frequency)
            F0[node * self.number_dof + 1] *= -1

            return F0

    @check_units
    def run_unbalance_response(
        self,
        node,
        unbalance_magnitude,
        unbalance_phase,
        speed_range=None,
        modes=None,
    ):
        """Unbalanced response for a coaxial rotor.

        Each unbalance excites at the speed of its own shaft, so an unbalance
        on the outer shaft excites at ``abs(speed_ratio)`` times the inner
        shaft speed. The response is evaluated at that frequency, with the
        rotor at the inner shaft speed.

        Parameters
        ----------
        node : list, int
            Node where the unbalance is applied.
        unbalance_magnitude : list, float, pint.Quantity
            Unbalance magnitude (kg.m).
        unbalance_phase : list, float, pint.Quantity
            Unbalance phase (rad).
        speed_range : list, pint.Quantity
            Inner shaft speeds (rad/s).
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
        >>> rotor = coaxrotor_example()
        >>> speed = np.linspace(0, 150, 31)
        >>> response = rotor.run_unbalance_response(node=[3, 13],
        ...                                         unbalance_magnitude=[1e-4, 1e-4],
        ...                                         unbalance_phase=[0, 0],
        ...                                         speed_range=speed)
        """
        if speed_range is None:
            modal = self.run_modal(0)
            speed_range = np.linspace(0, max(modal.evalues.imag) * 1.5, 1000)
        
        speed_range = np.asarray(speed_range)

        node = np.atleast_1d(node)
        unbalance_magnitude = np.atleast_1d(unbalance_magnitude)
        unbalance_phase = np.atleast_1d(unbalance_phase)

        frequency_ratios = {abs(self._node_speed_ratio(n)) for n in node}
        if len(frequency_ratios) > 1:
            raise ValueError(
                "Unbalances on both shafts excite at different frequencies "
                f"(abs(speed_ratio) = {abs(self.speed_ratio)}). Run the unbalance "
                "response for each shaft separately."
            )
        
        frequency_range = frequency_ratios.pop() * speed_range

        self._check_coefficient_axes(speed=speed_range, frequency=frequency_range)

        force = np.zeros((self.ndof, len(speed_range)), dtype=complex)
        for n, m, p in zip(node, unbalance_magnitude, unbalance_phase, strict=True):
            force += self._unbalance_force(n, m, p, speed_range)

        forced_resp = np.zeros((self.ndof, len(speed_range)), dtype=complex)
        for i, (speed, frequency) in enumerate(zip(speed_range, frequency_range, strict=True)):
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
            phi = phase[i] + theta * self._node_speed_ratio(n)
            w = omega * self._node_speed_ratio(n)
            a = alpha * self._node_speed_ratio(n)

            Fx = magnitude[i] * ((w**2) * np.cos(phi) + a * np.sin(phi))
            Fy = magnitude[i] * ((w**2) * np.sin(phi) - a * np.cos(phi))

            F0[n * self.number_dof + 0, :] += Fx
            F0[n * self.number_dof + 1, :] += Fy

        if return_all:
            return F0, theta, omega, alpha
        else:
            return F0

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
    steel = rs.materials.steel

    i_d = 0
    o_d = 0.05
    n = 10
    L = [0.25 for _ in range(n)]

    axial_shaft = [rs.ShaftElement(l, i_d, o_d, material=steel) for l in L]

    i_d = 0.25
    o_d = 0.30
    n = 6
    L = [0.25 for _ in range(n)]

    coaxial_shaft = [rs.ShaftElement(l, i_d, o_d, material=steel) for l in L]

    disk0 = rs.DiskElement.from_geometry(
        n=1, material=steel, width=0.07, i_d=0.05, o_d=0.28, scale_factor=0.8
    )
    disk1 = rs.DiskElement.from_geometry(
        n=9, material=steel, width=0.07, i_d=0.05, o_d=0.28, scale_factor=0.8
    )
    disk2 = rs.DiskElement.from_geometry(
        n=13, material=steel, width=0.07, i_d=0.20, o_d=0.48, scale_factor=0.8
    )
    disk3 = rs.DiskElement.from_geometry(
        n=15, material=steel, width=0.07, i_d=0.20, o_d=0.48, scale_factor=0.8
    )

    shaft = [axial_shaft, coaxial_shaft]
    disks = [disk0, disk1, disk2, disk3]

    stfx = 1e6
    stfy = 1e6
    bearing0 = rs.BearingElement(
        0, n_link=18, kxx=stfx, kyy=stfy, cxx=0, scale_factor=0.4
    )
    bearing1 = rs.BearingElement(
        10, n_link=19, kxx=stfx, kyy=stfy, cxx=0, scale_factor=0.4
    )
    bearing2 = rs.BearingElement(11, kxx=stfx, kyy=stfy, cxx=0, scale_factor=0.4)
    bearing3 = rs.BearingElement(
        8, n_link=17, kxx=stfx, kyy=stfy, cxx=0, scale_factor=0.4
    )

    base0 = rs.BearingElement(18, kxx=1e8, kyy=1e8, cxx=0, scale_factor=0.4)
    base1 = rs.BearingElement(19, kxx=1e8, kyy=1e8, cxx=0, scale_factor=0.4)

    pointmass0 = rs.PointMass(n=18, m=20)
    pointmass1 = rs.PointMass(n=19, m=20)

    bearings = [bearing0, bearing1, bearing2, bearing3, base0, base1]
    pointmasses = [pointmass0, pointmass1]

    return CoAxialRotor(shaft, disks, bearings, pointmasses)
