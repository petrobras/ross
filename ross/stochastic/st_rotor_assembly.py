"""STOCHASTIC ROSS Module.

This module creates random rotor instances and run stochastic analysis.
"""
# fmt: off
from collections.abc import Iterable

import numpy as np

from ross.rotor_assembly import Rotor
from ross.stochastic.st_bearing_seal_element import ST_BearingElement
from ross.stochastic.st_disk_element import ST_DiskElement
from ross.stochastic.st_point_mass import ST_PointMass
from ross.stochastic.st_results import (ST_CampbellResults,
                                        ST_ForcedResponseResults,
                                        ST_FrequencyResponseResults,
                                        ST_TimeResponseResults)
from ross.stochastic.st_shaft_element import ST_ShaftElement
from ross.units import check_units

# fmt: on

__all__ = ["ST_Rotor", "st_rotor_example"]


class ST_Rotor(object):
    r"""A random rotor object.

    This class will create several rotors according to random elements passed
    to the arguments.
    The number of rotors to be created depends on the amount of random
    elements instantiated and theirs respective sizes.

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

    Attributes
    ----------
    RV_size : int
        Number of random rotor instances.
    ndof : int
        Number of degrees of freedom for random rotor instances.

    Returns
    -------
    Random rotors objects

    Examples
    --------
    # Rotor with 2 shaft elements, 1 random disk element and 2 bearings
    >>> import numpy as np
    >>> import ross as rs
    >>> import ross.stochastic as srs
    >>> steel = rs.materials.steel
    >>> le = 0.25
    >>> i_d = 0
    >>> o_d = 0.05
    >>> tim0 = rs.ShaftElement(le, i_d, o_d, material=steel)
    >>> tim1 = rs.ShaftElement(le, i_d, o_d, material=steel)
    >>> shaft_elm = [tim0, tim1]

    # Building random disk element
    >>> size = 5
    >>> i_d = np.random.uniform(0.05, 0.06, size)
    >>> o_d = np.random.uniform(0.35, 0.39, size)
    >>> disk0 = srs.ST_DiskElement.from_geometry(n=1,
    ...                                          material=steel,
    ...                                          width=0.07,
    ...                                          i_d=i_d,
    ...                                          o_d=o_d,
    ...                                          is_random=["i_d", "o_d"],
    ...                                          )
    >>> stf = 1e6
    >>> bearing0 = rs.BearingElement(0, kxx=stf, cxx=0)
    >>> bearing1 = rs.BearingElement(2, kxx=stf, cxx=0)
    >>> rand_rotor = srs.ST_Rotor(shaft_elm, [disk0], [bearing0, bearing1])
    >>> len(list(iter(rand_rotor)))
    5
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
        if disk_elements is None:
            disk_elements = []
        if bearing_elements is None:
            bearing_elements = []
        if point_mass_elements is None:
            point_mass_elements = []

        # checking for random elements and matching sizes
        is_random = []
        len_list = []

        if any(isinstance(elm, ST_ShaftElement) for elm in shaft_elements):
            is_random.append("shaft_elements")

            it = iter(
                [elm for elm in shaft_elements if isinstance(elm, ST_ShaftElement)]
            )
            len_sh = len(list(next(iter(it))))
            if not all(len(list(l)) == len_sh for l in it):
                raise ValueError(
                    "not all random shaft elements lists have same length."
                )
            len_list.append(len_sh)

        if any(isinstance(elm, ST_DiskElement) for elm in disk_elements):
            is_random.append("disk_elements")

            it = iter([elm for elm in disk_elements if isinstance(elm, ST_DiskElement)])
            len_dk = len(list(next(iter(it))))
            if not all(len(list(l)) == len_dk for l in it):
                raise ValueError("not all random disk elements lists have same length.")
            len_list.append(len_dk)

        if any(isinstance(elm, ST_BearingElement) for elm in bearing_elements):
            is_random.append("bearing_elements")

            it = iter(
                [elm for elm in bearing_elements if isinstance(elm, ST_BearingElement)]
            )
            len_brg = len(list(next(iter(it))))
            if not all(len(list(l)) == len_brg for l in it):
                raise ValueError(
                    "not all random bearing elements lists have same length."
                )
            len_list.append(len_brg)

        if any(isinstance(elm, ST_PointMass) for elm in point_mass_elements):
            is_random.append("point_mass_elements")

            it = iter(
                [elm for elm in point_mass_elements if isinstance(elm, ST_PointMass)]
            )
            len_pm = len(list(next(iter(it))))
            if not all(len(list(l)) == len_pm for l in it):
                raise ValueError("not all random point mass lists have same length.")
            len_list.append(len_pm)

        if len_list.count(len_list[0]) == len(len_list):
            RV_size = len_list[0]
        else:
            raise ValueError("not all the random elements lists have the same length.")

        for i, elm in enumerate(shaft_elements):
            if isinstance(elm, ST_ShaftElement):
                shaft_elements[i] = list(iter(elm))

        for i, elm in enumerate(disk_elements):
            if isinstance(elm, ST_DiskElement):
                disk_elements[i] = list(iter(elm))

        for i, elm in enumerate(bearing_elements):
            if isinstance(elm, ST_BearingElement):
                bearing_elements[i] = list(iter(elm))

        for i, elm in enumerate(point_mass_elements):
            if isinstance(elm, ST_PointMass):
                point_mass_elements[i] = list(iter(elm))

        attribute_dict = dict(
            shaft_elements=shaft_elements,
            disk_elements=disk_elements,
            bearing_elements=bearing_elements,
            point_mass_elements=point_mass_elements,
            min_w=min_w,
            max_w=max_w,
            rated_w=rated_w,
            tag=tag,
        )

        # Assembling random rotors
        self.is_random = is_random
        self.attribute_dict = attribute_dict

        # common parameters
        self.RV_size = RV_size

        # collect a series of attributes from a rotor instance
        self._get_rotor_args()

    def __iter__(self):
        """Return an iterator for the container.

        Returns
        -------
        An iterator over random rotors.

        Examples
        --------
        >>> import ross.stochastic as srs
        >>> rotors = srs.st_rotor_example()
        >>> len(list(iter(rotors)))
        10
        """
        return iter(self.use_random_var(Rotor, self.is_random, self.attribute_dict))

    def __getitem__(self, key):
        """Return the value for a given key from attribute_dict.

        Parameters
        ----------
        key : str
            A class parameter as string.

        Raises
        ------
        KeyError
            Raises an error if the parameter doesn't belong to the class.

        Returns
        -------
        Return the value for the given key.

        Example
        -------
        >>> import ross.stochastic as srs
        >>> rotors = srs.st_rotor_example()
        >>> rotors["shaft_elements"] # doctest: +ELLIPSIS
        [ShaftElement...
        """
        if key not in self.attribute_dict.keys():
            raise KeyError("Object does not have parameter: {}.".format(key))

        return self.attribute_dict[key]

    def __setitem__(self, key, value):
        """Set new parameter values for the object.

        Function to change a parameter value.
        It's not allowed to add new parameters to the object.

        Parameters
        ----------
        key : str
            A class parameter as string.
        value : The corresponding value for the attrbiute_dict's key.
            ***check the correct type for each key in ST_Rotor
            docstring.

        Raises
        ------
        KeyError
            Raises an error if the parameter doesn't belong to the class.

        Example
        -------
        >>> import ross.stochastic as srs
        >>> rotors = srs.st_rotor_example()
        >>> rotors["tag"] = "rotor"
        >>> rotors["tag"]
        'rotor'
        """
        if key not in self.attribute_dict.keys():
            raise KeyError("Object does not have parameter: {}.".format(key))
        self.attribute_dict[key] = value

    def _get_rotor_args(self):
        """Get relevant attributes from a rotor system.

        This auxiliary funtion get some relevant attributes from a rotor system, such as
        the nodes numbers, nodes positions and number of degrees of freedom, and add it
        to the stochastic rotor as attribute. If an attribute is somehow afected by a
        random variable, the function returns its mean.
        """
        self.iter_break = True
        aux_rotor = list(iter(self))[0]

        self.ndof = aux_rotor.ndof
        self.nodes = aux_rotor.nodes
        self.number_dof = aux_rotor.number_dof
        self.link_nodes = aux_rotor.link_nodes

        if "shaft_elements" in self.is_random:
            if any(
                "L" in sh.is_random
                for sh in self.attribute_dict["shaft_elements"]
                if isinstance(sh, ST_ShaftElement)
            ):
                nodes_pos_matrix = np.zeros(len(self.nodes), self.RV_size)
                for i, rotor in enumerate(iter(self)):
                    nodes_pos_matrix[:, i] = rotor.nodes_pos
                self.nodes_pos = np.mean(nodes_pos_matrix, axis=1)
            else:
                self.nodes_pos = aux_rotor.nodes_pos
        else:
            self.nodes_pos = aux_rotor.nodes_pos

    @staticmethod
    def _get_args(idx, *args):
        """Build new list of arguments from a random list of arguments.

        This funtion takes a list with random values or with lists of random
        values and a build an organized list to instantiate functions
        correctly.

        Parameters
        ----------
        idx : int
            iterator index.
        *args : list
            list of mixed arguments.

        Returns
        -------
        new_args : list
            list of arranged arguments.

        Example
        -------
        >>> import ross.stochastic as srs
        >>> rotors = srs.st_rotor_example()
        >>> old_list = [1, 2, [3, 4], 5]
        >>> index = [0, 1]
        >>> new_list = [rotors._get_args(idx, old_list) for idx in index]
        >>> new_list
        [[1, 2, 3, 5], [1, 2, 4, 5]]
        """
        new_args = []
        for arg in list(args[0]):
            if isinstance(arg, Iterable):
                new_args.append(arg[idx])
            else:
                new_args.append(arg)

        return new_args

    def _random_var(self, is_random, *args):
        """Generate a list of random parameters.

        This function creates a list of parameters with random values given
        its own distribution.

        Parameters
        ----------
        is_random : list
            List of the object attributes to become stochastic.
        *args : dict
            Dictionary instanciating the ross.Rotor class.
            The attributes that are supposed to be stochastic should be
            set as lists of random variables.

        Returns
        -------
        new_args : generator
            Generator of random parameters.
        """
        args_dict = args[0]
        new_args = []
        var_size = None

        if self.iter_break is True:
            var_size = 1
            self.iter_break = False
        else:
            for v in list(map(args_dict.get, is_random))[0]:
                if isinstance(v, Iterable):
                    var_size = len(v)
                    break
            if var_size is None:
                var_size = len(list(map(args_dict.get, is_random))[0])

        for i in range(var_size):
            arg = []
            for key, value in args_dict.items():
                if key in is_random and key in self.is_random:
                    arg.append(self._get_args(i, value))
                elif key in is_random and key not in self.is_random:
                    arg.append(value[i])
                else:
                    arg.append(value)
            new_args.append(arg)

        return iter(new_args)

    def use_random_var(self, f, is_random, *args):
        """Generate a list of random objects from random attributes.

        This function creates a list of objects with random values for selected
        attributes from ross.Rotor class or its methods.

        Parameters
        ----------
        f : callable
            Function to be instantiated randomly with its respective *args.
        is_random : list
            List of the object attributes to become stochastic.
        *args : dict
            Dictionary instanciating the ross.Rotor class.
            The attributes that are supposed to be stochastic should be
            set as lists of random variables.

        Returns
        -------
        f_list : generator
            Generator of random objects.
        """
        args_dict = args[0]
        new_args = self._random_var(is_random, args_dict)
        f_list = (f(*arg) for arg in new_args)

        return f_list

    def run_campbell(self, speed_range, frequencies=6, frequency_type="wd"):
        """Stochastic Campbell diagram for multiples rotor systems.

        This function will calculate the damped or undamped natural frequencies
        for a speed range for every rotor instance.

        Parameters
        ----------
        speed_range : array
            Array with the desired range of frequencies.
        frequencies : int, optional
            Number of frequencies that will be calculated.
            Default is 6.
        frequency_type : str, optional
            Choose between displaying results related to the undamped natural
            frequencies ("wn") or damped natural frequencies ("wd").
            The default is "wd".

        Returns
        -------
        results.speed_range : array
            Array with the frequency range
        results.wd : array
            Array with the damped or undamped natural frequencies corresponding to
            each speed of the speed_range array for each rotor instance.
        results.log_dec : array
            Array with the log dec corresponding to each speed of the speed_range
            array for each rotor instance.

        Example
        -------
        >>> import ross.stochastic as srs
        >>> rotors = srs.st_rotor_example()

        # Running Campbell Diagram and saving the results

        >>> speed_range = np.linspace(0, 500, 31)
        >>> results = rotors.run_campbell(speed_range)

        # Plotting Campbell Diagram with Plotly

        >>> fig = results.plot(conf_interval=[90])
        >>> # fig.show()
        """
        CAMP_size = len(speed_range)
        RV_size = self.RV_size
        wd = np.zeros((frequencies, CAMP_size, RV_size))
        log_dec = np.zeros((frequencies, CAMP_size, RV_size))

        # Monte Carlo - results storage
        for i, rotor in enumerate(iter(self)):
            results = rotor.run_campbell(speed_range, frequencies, frequency_type)
            for j in range(frequencies):
                wd[j, :, i] = results.wd[:, j]
                log_dec[j, :, i] = results.log_dec[:, j]

        results = ST_CampbellResults(speed_range, wd, log_dec)

        return results

    def run_freq_response(
        self,
        inp,
        out,
        speed_range=None,
        modes=None,
        cluster_points=False,
        num_modes=12,
        num_points=10,
        rtol=0.005,
    ):
        """Stochastic frequency response for multiples rotor systems.

        This method returns the frequency response for every rotor instance,
        given a range of frequencies, the degrees of freedom to be
        excited and observed and the modes that will be used.

        Parameters
        ----------
        inp : int
            Input DoF.
        out : int
            Output DoF.
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
        results.speed_range : array
            Array with the frequencies.
        results.magnitude : array
            Amplitude response for each rotor system.
        results.phase : array
            Phase response for each rotor system.

        Example
        -------
        >>> import ross.stochastic as srs
        >>> rotors = srs.st_rotor_example()

        # Running Frequency Response and saving the results

        >>> speed_range = np.linspace(0, 500, 31)
        >>> inp = 9
        >>> out = 9
        >>> results = rotors.run_freq_response(inp, out, speed_range)

        # Plotting Frequency Response with Plotly

        >>> fig = results.plot(conf_interval=[90])
        >>> # fig.show()
        """
        FRF_size = len(speed_range)
        RV_size = self.RV_size

        freq_resp = np.empty((FRF_size, RV_size), dtype=np.complex)
        velc_resp = np.empty((FRF_size, RV_size), dtype=np.complex)
        accl_resp = np.empty((FRF_size, RV_size), dtype=np.complex)

        # Monte Carlo - results storage
        for i, rotor in enumerate(iter(self)):
            results = rotor.run_freq_response(
                speed_range,
                modes,
                cluster_points,
                num_modes,
                num_points,
                rtol,
            )
            freq_resp[:, i] = results.freq_resp[inp, out, :]
            velc_resp[:, i] = results.velc_resp[inp, out, :]
            accl_resp[:, i] = results.accl_resp[inp, out, :]

        results = ST_FrequencyResponseResults(
            speed_range, freq_resp, velc_resp, accl_resp
        )

        return results

    def run_time_response(self, speed, force, time_range, ic=None):
        """Stochastic time response for multiples rotor systems.

        This function will take a rotor object and plot its time response
        given a force and a time. This method displays the amplitude vs time or the
        rotor orbits.
        The force and ic parameters can be passed as random variables.

        Parameters
        ----------
        speed: float
            Rotor speed
        force : 2-dimensional array, 3-dimensional array
            Force array (needs to have the same number of rows as time array).
            Each column corresponds to a dof and each row to a time step.
            Inputing a 3-dimensional array, the method considers the force as
            a random variable. The 3rd dimension must have the same size than
            ST_Rotor.RV_size
        time_range : 1-dimensional array
            Time array.
        ic : 1-dimensional array, 2-dimensional array, optional
            The initial conditions on the state vector (zero by default).
            Inputing a 2-dimensional array, the method considers the
            initial condition as a random variable.

        Returns
        -------
        results.time_range : array
            Array containing the time array.
        results.yout : array
            System response.
        results.xout
            Time evolution of the state vector for each rotor system.

        Example
        -------
        >>> import ross.stochastic as srs
        >>> rotors = srs.st_rotor_example()

        # Running Time Response and saving the results

        >>> size = 10
        >>> ndof = rotors.ndof
        >>> node = 3 # node where the force is applied

        >>> dof = 9
        >>> speed = 250.0
        >>> t = np.linspace(0, 10, size)
        >>> F = np.zeros((size, ndof))
        >>> F[:, 4 * node] = 10 * np.cos(2 * t)
        >>> F[:, 4 * node + 1] = 10 * np.sin(2 * t)
        >>> results = rotors.run_time_response(speed, F, t)

        # Plotting Time Response 1D, 2D and 3D

        >>> fig = results.plot_1d(probe=[(3, np.pi / 2)], conf_interval=[90])
        >>> # fig.show()
        >>> fig = results.plot_2d(node=node, conf_interval=[90])
        >>> # fig.show()
        >>> fig = results.plot_3d(conf_interval=[90])
        >>> # fig.show()
        """
        t_size = len(time_range)
        RV_size = self.RV_size
        ndof = self.ndof
        number_dof = self.number_dof
        nodes = self.nodes
        link_nodes = self.link_nodes
        nodes_pos = self.nodes_pos

        xout = np.zeros((RV_size, t_size, 2 * ndof))
        yout = np.zeros((RV_size, t_size, ndof))

        # force is not a random variable
        if len(force.shape) == 2:
            # Monte Carlo - results storage
            for i, rotor in enumerate(iter(self)):
                t_, y, x = rotor.time_response(speed, force, time_range, ic)
                xout[i] = x
                yout[i] = y

        # force is a random variable
        if len(force.shape) == 3:
            # Monte Carlo - results storage
            i = 0
            for rotor, F in zip(iter(self), force):
                t_, y, x = rotor.time_response(speed, F, time_range, ic)
                xout[i] = x
                yout[i] = y
                i += 1

        results = ST_TimeResponseResults(
            time_range,
            yout,
            xout,
            number_dof,
            nodes,
            link_nodes,
            nodes_pos,
        )

        return results

    @check_units
    def run_unbalance_response(
        self,
        node,
        unbalance_magnitude,
        unbalance_phase,
        frequency_range=None,
        modes=None,
        cluster_points=False,
        num_modes=12,
        num_points=10,
        rtol=0.005,
    ):
        """Stochastic unbalance response for multiples rotor systems.

        This method returns the unbalanced response for every rotor instance,
        given magnitide and phase of the unbalance, the node where
        it's applied and a frequency range.

        Magnitude and phase parameters can be passed as random variables.

        Parameters
        ----------
        node : list, int
            Node where the unbalance is applied.
        unbalance_magnitude : list, float, pint.Quantity
            Unbalance magnitude (kg.m).
            If node is int, input a list to make make it random.
            If node is list, input a list of lists to make it random.
            If there're multiple unbalances and not all of the magnitudes are supposed
            to be stochastic, input a list with repeated values to the unbalance
            magnitude considered deterministic.
        unbalance_phase : list, float, pint.Quantity
            Unbalance phase (rad).
            If node is int, input a list to make make it random.
            If node is list, input a list of lists to make it random.
            If there're multiple unbalances and not all of the phases are supposed
            to be stochastic, input a list with repeated values to the unbalance phase
            considered deterministic.
        frequency_range : list, float
            Array with the desired range of frequencies.
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
        results.force_resp : array
            Array with the force response for each node for each frequency
        results.speed_range : array
            Array with the frequencies.
        results.magnitude : array
            Magnitude of the frequency response for node for each frequency.
        results.phase : array
            Phase of the frequency response for node for each frequencye.

        Example
        -------
        >>> import ross.stochastic as srs
        >>> rotors = srs.st_rotor_example()

        # Running Frequency Response and saving the results

        >>> freq_range = np.linspace(0, 500, 31)
        >>> n = 3
        >>> m = np.random.uniform(0.001, 0.002, 10)
        >>> p = 0.0
        >>> results = rotors.run_unbalance_response(n, m, p, freq_range)

        Plot unbalance response:
        >>> probe_node = 3
        >>> probe_angle = np.pi / 2
        >>> probe_tag = "my_probe"  # optional
        >>> fig = results.plot(probe=[(probe_node, probe_angle, probe_tag)])

        To plot velocity and acceleration responses, you must change amplitude_units
        from "[length]" units to "[length]/[time]" or "[length]/[time] ** 2" respectively
        Plotting velocity response:
        >>> fig = results.plot(
        ...     probe=[(probe_node, probe_angle)],
        ...     amplitude_units="m/s"
        ... )

        Plotting acceleration response:
        >>> fig = results.plot(
        ...     probe=[(probe_node, probe_angle)],
        ...     amplitude_units="m/s**2"
        ... )
        """
        RV_size = self.RV_size
        freq_size = len(frequency_range)
        ndof = self.ndof
        args_dict = dict(
            node=node,
            unbalance_magnitude=unbalance_magnitude,
            unbalance_phase=unbalance_phase,
            frequency_range=frequency_range,
            modes=modes,
            cluster_points=cluster_points,
            num_modes=num_modes,
            num_points=num_points,
            rtol=rtol,
        )

        forced_resp = np.zeros((RV_size, ndof, freq_size), dtype=np.complex)
        velc_resp = np.zeros((RV_size, ndof, freq_size), dtype=np.complex)
        accl_resp = np.zeros((RV_size, ndof, freq_size), dtype=np.complex)
        is_random = []

        if (isinstance(node, int) and isinstance(unbalance_magnitude, Iterable)) or (
            isinstance(node, Iterable) and isinstance(unbalance_magnitude[0], Iterable)
        ):
            is_random.append("unbalance_magnitude")

        if (isinstance(node, int) and isinstance(unbalance_phase, Iterable)) or (
            isinstance(node, Iterable) and isinstance(unbalance_phase[0], Iterable)
        ):
            is_random.append("unbalance_phase")

        # Monte Carlo - results storage
        if len(is_random):
            i = 0
            unbalance_args = self._random_var(is_random, args_dict)
            for rotor, args in zip(iter(self), unbalance_args):
                results = rotor.run_unbalance_response(*args)
                forced_resp[i] = results.forced_resp
                velc_resp[i] = results.velc_resp
                accl_resp[i] = results.accl_resp
                i += 1
        else:
            for i, rotor in enumerate(iter(self)):
                results = rotor.run_unbalance_response(
                    node, unbalance_magnitude, unbalance_phase, frequency_range
                )
                forced_resp[i] = results.forced_resp
                velc_resp[i] = results.velc_resp
                accl_resp[i] = results.accl_resp

        results = ST_ForcedResponseResults(
            forced_resp=forced_resp,
            frequency_range=frequency_range,
            velc_resp=velc_resp,
            accl_resp=accl_resp,
            number_dof=self.number_dof,
            nodes=self.nodes,
            link_nodes=self.link_nodes,
        )

        return results


def st_rotor_example():
    """Return an instance of random rotors.

    The purpose of this is to make available a simple model
    so that doctest can be written using this.

    Returns
    -------
    An instance of random rotors.

    Examples
    --------
    >>> import ross.stochastic as srs
    >>> rotors = srs.st_rotor_example()
    >>> len(list(iter(rotors)))
    10
    """
    import ross as rs
    from ross.materials import steel

    i_d = 0
    o_d = 0.05
    n = 6
    L = [0.25 for _ in range(n)]

    shaft_elem = [rs.ShaftElement(l, i_d, o_d, material=steel) for l in L]

    disk0 = rs.DiskElement.from_geometry(
        n=2, material=steel, width=0.07, i_d=0.05, o_d=0.28
    )
    disk1 = rs.DiskElement.from_geometry(
        n=4, material=steel, width=0.07, i_d=0.05, o_d=0.28
    )

    s = 10
    kxx = np.random.uniform(1e6, 2e6, s)
    cxx = np.random.uniform(1e3, 2e3, s)
    bearing0 = ST_BearingElement(n=0, kxx=kxx, cxx=cxx, is_random=["kxx", "cxx"])
    bearing1 = ST_BearingElement(n=6, kxx=kxx, cxx=cxx, is_random=["kxx", "cxx"])

    return ST_Rotor(shaft_elem, [disk0, disk1], [bearing0, bearing1])
