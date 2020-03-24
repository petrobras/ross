# fmt: off
from collections import Iterable

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
    sparse : bool, optional
        If sparse, eigenvalues will be calculated with arpack.
        Default is True.
    n_eigen : int, optional
        Number of eigenvalues calculated by arpack.
        Default is 12.
    tag : str
        A tag for the rotor

    Attributes
    ----------
    rotor_list : list, array
        List with random rotor objects

    Returns
    -------
    Random rotors objects

    Examples
    --------
    >>> # Rotor with 2 shaft elements, 1 random disk element and 2 bearings
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
    >>> i_d=np.random.uniform(0.05, 0.06, size)
    >>> o_d=np.random.uniform(0.35, 0.39, size)
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
    >>> len(rand_rotor.rotor_list)
    5
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

        if disk_elements is None:
            disk_elements = []
        if bearing_elements is None:
            bearing_elements = []
        if point_mass_elements is None:
            point_mass_elements = []

        # checking for random elements and matching sizes
        is_random = []

        if any(isinstance(elm, ST_ShaftElement) for elm in shaft_elements):
            is_random.append("shaft_elements")

            it = iter(
                [elm for elm in shaft_elements if isinstance(elm, ST_ShaftElement)]
            )
            the_len = len(next(it).elements)
            if not all(len(l.elements) == the_len for l in it):
                raise ValueError(
                    "not all random shaft elements lists have same length."
                )

        if any(isinstance(elm, ST_DiskElement) for elm in disk_elements):
            is_random.append("disk_elements")

            it = iter([elm for elm in disk_elements if isinstance(elm, ST_DiskElement)])
            the_len = len(next(it).elements)
            if not all(len(l.elements) == the_len for l in it):
                raise ValueError("not all random disk elements lists have same length.")

        if any(isinstance(elm, ST_BearingElement) for elm in bearing_elements):
            is_random.append("bearing_elements")

            it = iter(
                [elm for elm in bearing_elements if isinstance(elm, ST_BearingElement)]
            )
            the_len = len(next(it).elements)
            if not all(len(l.elements) == the_len for l in it):
                raise ValueError(
                    "not all random bearing elements lists have same length."
                )

        if any(isinstance(elm, ST_PointMass) for elm in point_mass_elements):
            is_random.append("point_mass_elements")

            it = iter(
                [elm for elm in point_mass_elements if isinstance(elm, ST_PointMass)]
            )
            the_len = len(next(it).elements)
            if not all(len(l.elements) == the_len for l in it):
                raise ValueError("not all random point mass lists have same length.")

        for i, elm in enumerate(shaft_elements):
            if isinstance(elm, ST_ShaftElement):
                shaft_elements[i] = elm.elements

        for i, elm in enumerate(disk_elements):
            if isinstance(elm, ST_DiskElement):
                disk_elements[i] = elm.elements

        for i, elm in enumerate(bearing_elements):
            if isinstance(elm, ST_BearingElement):
                bearing_elements[i] = elm.elements

        for i, elm in enumerate(point_mass_elements):
            if isinstance(elm, ST_PointMass):
                point_mass_elements[i] = elm.elements

        attribute_dict = dict(
            shaft_elements=shaft_elements,
            disk_elements=disk_elements,
            bearing_elements=bearing_elements,
            point_mass_elements=point_mass_elements,
            sparse=sparse,
            n_eigen=n_eigen,
            min_w=min_w,
            max_w=max_w,
            rated_w=rated_w,
            tag=tag,
        )

        # Assembling random rotors
        rotor_list = self.use_random_var(Rotor, is_random, attribute_dict)
        self.rotor_list = rotor_list

    def _random_var(self, is_random, *args):
        """Generate a list of random parameters.

        This function creates a list of parameters with random values given
        its own distribution.

        Parameters
        ----------
        is_random : list
            List of the object attributes to become stochastic.
            Default is None
        *args : dict
            Dictionary instanciating the ShaftElement class.
            The attributes that are supposed to be stochastic should be
            set as lists of random variables.

        Returns
        -------
        f_list : list
            List of random parameters.

        Example
        -------
        """
        args_dict = args[0]
        new_args = []

        try:
            for i in range(len(args_dict[is_random[0]][0])):
                arg = []
                for key, value in args_dict.items():
                    if key in is_random:
                        arg.append([value[j][i] for j in range(len(value))])
                    else:
                        arg.append(value)
                new_args.append(arg)

        except TypeError:
            for i in range(len(args_dict[is_random[0]])):
                arg = []
                for key, value in args_dict.items():
                    if key in is_random:
                        arg.append(value[i])
                    else:
                        arg.append(value)
                new_args.append(arg)

        return new_args

    def use_random_var(self, f, is_random, *args):
        """Generate a list of random objects from random attributes.

        This function creates a list of objects with random values for selected
        attributes from a given class or method.

        Parameters
        ----------
        f : callable
            Function to be instantiated randomly with its respective *args.
            Default is instantiating the ST_Rotor class
        is_random : list
            List of the object attributes to become stochastic.
            Default is None
        *args : dict
            Dictionary instanciating the ShaftElement class.
            The attributes that are supposed to be stochastic should be
            set as lists of random variables.

        Returns
        -------
        f_list : list
            List of random objects.

        Example
        -------
        """
        args_dict = args[0]
        new_args = self._random_var(is_random, args_dict)
        f_list = [f(*arg) for arg in new_args]

        return f_list

    def run_campbell(self, speed_range, frequencies=6, frequency_type="wd"):
        """Stochastic Campbell diagram for multiples rotor systems.

        This function will calculate the damped or undamped natural frequencies
        for a speed range for every rotor in rotor_list.

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
        results
            Array with the damped or undamped natural frequencies and log dec
            corresponding to each speed of the speed_range array for each rotor
            in rotor_list.
            It will be returned if plot=False.

        Example
        -------
        >>> import ross.stochastic as srs
        >>> rotors = srs.st_rotor_example()

        # Running Campbell Diagram and saving the results

        >>> speed_range = np.linspace(0, 500, 31)
        >>> results = rotors.run_campbell(speed_range)

        # Plotting Campbell Diagram with bokeh

        >>> results.plot(conf_interval=[90]) # doctest: +ELLIPSIS
        Column...
        """
        CAMP_size = len(speed_range)
        RV_size = len(self.rotor_list)
        wd = np.zeros((frequencies, CAMP_size, RV_size))
        log_dec = np.zeros((frequencies, CAMP_size, RV_size))

        # Monte Carlo - results storage
        for i, rotor in enumerate(self.rotor_list):
            results = rotor.run_campbell(speed_range, frequencies, frequency_type)
            for j in range(frequencies):
                wd[j, :, i] = results.wd[:, j]
                log_dec[j, :, i] = results.log_dec[:, j]

        results = ST_CampbellResults(speed_range, wd, log_dec)

        return results

    def run_freq_response(self, speed_range, inp, out, modes=None):
        """Stochastic frequency response for multiples rotor systems.

        This method returns the frequency response for every rotor system in
        rotor_list given a range of frequencies, the degrees of freedom to be
        excited and observed and the modes that will be used.

        Parameters
        ----------
        speed_range : array
            Array with the desired range of frequencies.
        inp : int
            Degree of freedom to be excited.
        out : int
            Degree of freedom to be observed.
        modes : list, optional
            Modes that will be used to calculate the frequency response
            (all modes will be used if a list is not given).

        Returns
        -------
        results : array
            Array with the frequencies, magnitude and phase of the frequency
            response for the given pair input/output for each rotor in rotor_list.
            It will be returned if plot=False.

        Example
        -------
        >>> import ross.stochastic as srs
        >>> rotors = srs.st_rotor_example()

        # Running Frequency Response and saving the results

        >>> speed_range = np.linspace(0, 500, 31)
        >>> inp = 9
        >>> out = 9
        >>> results = rotors.run_freq_response(speed_range, inp, out)

        # Plotting Frequency Response with bokeh

        >>> results.plot(conf_interval=[90]) # doctest: +ELLIPSIS
        Column...
        """
        FRF_size = len(speed_range)
        RV_size = len(self.rotor_list)
        magnitude = np.zeros(((FRF_size, RV_size)))
        phase = np.zeros(((FRF_size, RV_size)))

        # Monte Carlo - results storage
        for i, rotor in enumerate(self.rotor_list):
            results = rotor.run_freq_response(speed_range, modes)
            magnitude[:, i] = results.magnitude[inp, out, :]
            phase[:, i] = results.phase[inp, out, :]

        results = ST_FrequencyResponseResults(speed_range, magnitude, phase,)

        return results

    def run_time_response(self, speed, force, time_range, dof, ic=None):
        """Stochastic time response for multiples rotor systems.

        This function will take a rotor object and plot its time response
        given a force and a time.
        The force parameter can be passed as random.

        Parameters
        ----------
        speed: float
            Rotor speed
        force : 2-dimensional array, 3-dimensional array
            Force array (needs to have the same number of rows as time array).
            Each column corresponds to a dof and each row to a time step.
            Inputing a 3-dimensional array, the method considers the force as
            a random variable. The 3rd dimension must have the same size than
            ST_Rotor.rotor_list
        time_range : 1-dimensional array
            Time array.
        dof : int
            Degree of freedom that will be observed.
        ic : 1-dimensional array, 2-dimensional array, optional
            The initial conditions on the state vector (zero by default).
            Inputing a 2-dimensional array, the method considers the force as
            a random variable.

        Returns
        -------
        results : array
            Array containing the time array, the system response, and the
            time evolution of the state vector for each rotor system.
            It will be returned if plot=False.

        Example
        -------
        >>> import ross.stochastic as srs
        >>> rotors = srs.st_rotor_example()

        # Running Time Response and saving the results

        >>> size = 1000
        >>> ndof = rotors.rotor_list[0].ndof
        >>> node = 3 # node where the force is applied

        >>> dof = 9
        >>> speed = 250.0
        >>> t = np.linspace(0, 10, size)
        >>> F = np.zeros((size, ndof))
        >>> F[:, 4 * node] = 10 * np.cos(2 * t)
        >>> F[:, 4 * node + 1] = 10 * np.sin(2 * t)
        >>> results = rotors.run_time_response(speed, F, t, dof)

        # Plotting Time Response with bokeh

        >>> results.plot(conf_interval=[90]) # doctest: +ELLIPSIS
        Figure...
        """
        t_size = len(time_range)
        RV_size = len(self.rotor_list)
        ndof = self.rotor_list[0].ndof

        xout = np.zeros((RV_size, t_size, 2 * ndof))
        yout = np.zeros((RV_size, t_size, ndof))

        # force is not a random variable
        if len(force.shape) == 2:
            # Monte Carlo - results storage
            for i, rotor in enumerate(self.rotor_list):
                t_, y, x = rotor.time_response(speed, force, time_range, ic)
                xout[i] = x
                yout[i] = y

        # force is a random variable
        if len(force.shape) == 3:
            # Monte Carlo - results storage
            i = 0
            for rotor, F in zip(self.rotor_list, force):
                t_, y, x = rotor.time_response(speed, F, time_range, ic)
                xout[i] = x
                yout[i] = y
                i += 1

        results = ST_TimeResponseResults(time_range, yout, xout, dof)

        return results

    def run_unbalance_response(self, node, magnitude, phase, frequency_range):
        """Stochastic unbalance response for multiples rotor systems.

        This method returns the unbalanced response for every rotor system in
        rotor_list, given magnitide and phase of the unbalance, the node where
        it's applied and a frequency range.

        Magnitude and phase parameters can be passed as randoms.

        Parameters
        ----------
        node : list, int
            Node where the unbalance is applied.
        magnitude : list, float
            Unbalance magnitude.
            If node is int, input a list to make make it random.
            If node is list, input a list of lists to make it random.
        phase : list, float
            Unbalance phase.
            If node is int, input a list to make make it random.
            If node is list, input a list of lists to make it random.
        frequency_range : list, float
            Array with the desired range of frequencies.

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
        >>> dof = 13
        >>> results = rotors.run_unbalance_response(n, m, p, freq_range)

        # Plotting Frequency Response with bokeh

        >>> results.plot(dof, conf_interval=[90]) # doctest: +ELLIPSIS
        Column...
        """
        RV_size = len(self.rotor_list)
        freq_size = len(frequency_range)
        ndof = self.rotor_list[0].ndof
        args_dict = dict(
            node=node, magnitude=magnitude, phase=phase, frequency_range=frequency_range
        )

        forced_resp = np.zeros((RV_size, freq_size, ndof), dtype=np.complex)
        mag_resp = np.zeros((RV_size, freq_size, ndof))
        phs_resp = np.zeros((RV_size, freq_size, ndof))
        is_random = []

        if (isinstance(node, int) and isinstance(magnitude, Iterable)) or (
            isinstance(node, Iterable) and isinstance(magnitude[0], Iterable)
        ):
            is_random.append("magnitude")

        if (isinstance(node, int) and isinstance(phase, Iterable)) or (
            isinstance(node, Iterable) and isinstance(phase[0], Iterable)
        ):
            is_random.append("phase")

        # Monte Carlo - results storage
        if len(is_random):
            i = 0
            unbalance_args = self._random_var(is_random, args_dict)
            for rotor, args in zip(self.rotor_list, unbalance_args):
                results = rotor.unbalance_response(*args)
                forced_resp[i] = results.forced_resp.T
                mag_resp[i] = results.magnitude.T
                phs_resp[i] = results.phase.T
                i += 1
        else:
            for i, rotor in enumerate(self.rotor_list):
                results = rotor.unbalance_response(
                    node, magnitude, phase, frequency_range
                )
                forced_resp[i] = results.forced_resp.T
                mag_resp[i] = results.magnitude.T
                phs_resp[i] = results.phase.T

        results = ST_ForcedResponseResults(
            forced_resp=forced_resp,
            frequency_range=frequency_range,
            magnitude=mag_resp,
            phase=phs_resp,
        )

        return results


def st_rotor_example():
    """This function returns an instance of random rotors.

    The purpose of this is to make available a simple model
    so that doctest can be written using this.

    Parameters
    ----------

    Returns
    -------
    An instance of random rotors.

    Examples
    --------
    >>> rotors = st_rotor_example()
    >>> len(rotors.rotor_list)
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
