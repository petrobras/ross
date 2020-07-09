"""Rotordynamic Report Configuration File."""
import black

__all__ = ["Config"]


class _Dict:
    """Set keys and values as attribute for the Config object.

    Subclass to organize nested dictionaries and set each key / value as attribute for
    the config object.

    Return
    ------
    A dictionary as attribute for the Config() object.

    Examples
    --------
    >>> param = _Dict({
    ...     "stiffness_range": None,
    ...     "num": 30,
    ...     "num_modes": 16,
    ...     "synchronous": False,
    ... })
    >>> param.num
    30
    """

    def __init__(self, dictionary):
        for k, v in dictionary.items():
            if isinstance(v, dict):
                setattr(self, k, self.__class__(v))
            else:
                setattr(self, k, v)

    def __repr__(self):
        """Return a string representation for the dict attribute.

        This method uses Black code formatting for better visualization.

        Returns
        -------
        A string representation for the dictionary options.

        Examples
        --------
        >>> param = _Dict({
        ...     "oper_clearance": None,
        ...     "min_clearance": None,
        ...     "max_clearance": None,
        ... })
        >>> param # doctest: +ELLIPSIS
        {"oper_clearance": None, "min_clearance": None, "max_clearance": None}...
        """
        return black.format_file_contents(
            repr(self.__dict__), fast=True, mode=black.FileMode()
        )

    def __getitem__(self, option):
        """Return the value for a given option from the dictionary.

        Parameters
        ----------
        option : str
            A dictionary key corresponding to config options as string.

        Raises
        ------
        KeyError
            Raises an error if the parameter doesn't belong to the dictionary.

        Returns
        -------
        Return the value for the given key.

        Examples
        --------
        >>> param = _Dict({
        ...     "stiffness_range": None,
        ...     "num": 30,
        ...     "num_modes": 16,
        ...     "synchronous": False,
        ... })
        >>> param["num"]
        30
        """
        if option not in self.__dict__.keys():
            raise KeyError("Option '{}' not found.".format(option))

        return self.__dict__[option]

    def _update(self, **kwargs):
        """Update the dict values.

        This is an axuliar method for Config.update_config() to set new values for the
        config dictionary according to kwargs input.
        The kwargs must respect Config attributes. It's only possible to update
        existing values from Config dictionary.
        **See Config attributes reference for infos about the dict options.

        Parameters
        ----------
        **kwargs : dict
            Dictionary with new values for corresponding keys.

        Raises
        ------
        KeyError
            Raises an error if the parameter doesn't belong to the dictionary.

        Examples
        --------
        >>> param = _Dict({
        ...     "stiffness_range": None,
        ...     "num": 30,
        ...     "num_modes": 16,
        ...     "synchronous": False,
        ... })
        >>> param._update(num=20, num_modes=10)
        >>> param # doctest: +ELLIPSIS
        {"stiffness_range": None, "num": 20, "num_modes": 10, "synchronous": False}...
        """
        for k, v in kwargs.items():
            if k not in self.__dict__.keys():
                raise KeyError("Option '{}' not found.".format(k))

            if isinstance(v, dict):
                getattr(self, k)._update(**v)
            else:
                self.__dict__[k] = v


class Config:
    """Configurate parameters for rotordynamic report.

    This class generates an object with several preset parameters to run the
    rotordynamics analyses. It's a must to check all the options for a correct
    functioning.

    The attributes are automatically generated and it's not possible to remove then or
    add new ones.

    Attributes
    ----------
    rotor_properties : dict
        Dictionary of rotor properties.

        rotor_speeds : dict
            Dictionary indicating the operational speeds which will be used in the
            analyses.

            min_speed : float
                The machine minimum operational speed.
            max_speed : float
                The machine  maximum operational speed.
            oper_speed : float
                The machine  nominal operational speed.
            trip_speed : float
                The machine overspeed trip.
            speed_factor : float
                Multiplicative factor of the speed range - according to API 684.
                Default is 1.50.
            unit : str
                Unit system to speed values. Options: "rad/s", "rpm".
                Default is "rpm".

        rotor_id : dict
            Dictionary with rotor identifications.

            type : str
                Machine type: Options are: "compressor", "turbine", "axial_flow". Each
                options has it's own considerations (according to API 684). If a different
                option is input, it will be the software will treat as a "compressor".
            tag : str
                Tag for the rotor. If None, it'll copy the tag from rotor object.

    bearings : dict
        The analyses consider different configurations for the bearings. It should be done
        for minimum, maximum and the nominal clearance.

        oper_clearance : list
            List of bearing elements. The coefficients should be calculated for the nominal
            clearance.
        min_clearance : list
            List of bearing elements. The coefficients should be calculated for the minimum
            clearance.
        max_clearance : list
            List of bearing elements. The coefficients should be calculated for the maximum
            clearance.

    run_campbell : dict
        Dictionary configurating run_campbell parameters.

        speed_range : list, array
            Array with the speed range.
        num_modes : float
            Number of frequencies that will be calculated.
            Default is 6.

    plot_ucs : dict
        Dictionary configurating plot_ucs parameters.

        stiffness_range : tuple, optional
            Tuple with (start, end) for stiffness range.
        num : int
            Number of steps in the range.
            Default is 30.
        num_modes : int, optional
            Number of modes to be calculated.
            Default is 16.
        synchronous : bool
            If True a synchronous analysis is carried out and the frequency of
            the first forward model will be equal to the speed.
            Default is False.

    run_unbalance_response : dict
        Dictionary configurating run_unbalance_response parameters.

        probes : dict
            Dictionary with the node where the probe is set and its respective
            orientation angle.

            node : list
                List with the nodes where probes are located.
            orientation : list
                List with the respective orientation angle for the probes.
                0 or π (rad) corresponds to the X orientation and
                π / 2 or 3 * π / 2 (rad) corresponds to the Y orientation.
            unit : str
                Unit system for the orientation angle. Can be "rad" or "degree".
                Default is "rad".

        frequency_range : list, array
            Array with the desired range of frequencies. If None and cluster_points is
            False, it creates an array from 0 to the max continuos speed times the
            speed_factor.
            If None and cluster_points is True, it creates and automatic array based on
            the number of modes selected.
            Default is None with cluster_points False.
        modes : list, optional
            Modes that will be used to calculate the frequency response
            (all modes will be used if a list is not given).
        cluster_points : bool, optional
            Boolean to activate the automatic frequency spacing method. If True, the
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
            points only for the critical speed calculated by Rotor.run_critical_speed().
            Default is 12.
        rtol : float, optional
            Tolerance (relative) for termination. Applied to scipy.optimize.newton to
            calculate the approximated critical speeds.
            Default is 0.005 (0.5%).

    stability_level1 : dict
        Dictionary configurating stability_level_1 parameters.

        D : list, array
            Impeller diameter, m (in.) or Blade pitch diameter, m (in.).
            The disk elements order must be observed to input this list.
        H : list, array
            Minimum diffuser width per impeller, m (in.) or Effective blade height, m (in.).
            The disk elements order must be observed to input this list.
        rated_power : list
            Rated power per stage/impeller, W (HP),
            The disk elements order must be observed to input this list.
        rho_ratio : list
            Density ratio between the discharge gas density and the suction
            gas density per impeller, kg/m3 (lbm/in.3),
            The disk elements order must be observed to input this list.
        rho_suction : float
            Suction gas density in the first stage, kg/m3 (lbm/in.3).
        rho_discharge : float
            Discharge gas density in the last stage, kg/m3 (lbm/in.3),
        unit: str, optional
            Unit system. Options are "m" (meter) and "in" (inch).
            Default is "m".

    Returns
    -------
    A config object to rotordynamic report.

    Examples
    --------
    There are two possible syntax to return the options setup. One is using the
    object.attribute syntax and the other is the dicionary syntax

    First syntax opion:
    >>> configs = Config()
    >>> configs.rotor_properties.rotor_id # doctest: +ELLIPSIS
    {"type": "compressor", "tag": None}...

    Second syntax opion:
    >>> configs = Config()
    >>> configs["rotor_properties"]["rotor_id"] # doctest: +ELLIPSIS
    {"type": "compressor", "tag": None}...
    """

    def __init__(self):
        # fmt: off
        # Configurating rotor properties
        self.rotor_properties = _Dict({
            "rotor_speeds": {
                "min_speed": None,
                "max_speed": None,
                "oper_speed": None,
                "trip_speed": None,
                "speed_factor": 1.50,
                "unit": "rpm",
            },
            "rotor_id": {
                "type": "compressor",
                "tag": None
            },
        })

        # Configurating bearing elements for diferent clearances
        self.bearings = _Dict({
            "oper_clearance": None,
            "min_clearance": None,
            "max_clearance": None,
        })

        # Configurating campbell options
        self.run_campbell = _Dict({
            "speed_range": None,
            "num_modes": 6,
        })

        # Configurating UCS options
        self.plot_ucs = _Dict({
            "stiffness_range": None,
            "num": 30,
            "num_modes": 16,
            "synchronous": False,
        })

        # Configurating unbalance response options
        self.run_unbalance_response = _Dict({
            "probes": {
                "node": None,
                "orientation": None,
                "unit": "rad",
            },
            "frequency_range": None,
            "modes": None,
            "cluster_points": False,
            "num_modes": 12,
            "num_points": 10,
            "rtol": 0.005,
        })

        # Configurating stability level 1 analysis
        self.stability_level1 = _Dict({
            "D": None,
            "H": None,
            "rated_power": None,
            "rho_ratio": None,
            "rho_suction": None,
            "rho_discharge": None,
            "unit": "m",
        })
        # fmt: on

    def __repr__(self):
        """Return a string representation for the config options.

        This method uses Black code formatting for better visualization.

        Returns
        -------
        A string representation for the config dictionary.

        Examples
        --------
        >>> configs = Config()
        >>> configs # doctest: +ELLIPSIS
        {
            "rotor_properties": {
                "rotor_speeds": {
                    "min_speed": None,
                    "max_speed": None,
                    "oper_speed": None...
        """
        return black.format_file_contents(
            repr(self.__dict__), fast=True, mode=black.FileMode()
        )

    def __getitem__(self, option):
        """Return the value for a given option from config dictionary.

        Parameters
        ----------
        option : str
            A dictionary key corresponding to config options as string.

        Raises
        ------
        KeyError
            Raises an error if the parameter doesn't belong to the dictionary.

        Returns
        -------
        Return the value for the given key.

        Examples
        --------
        >>> configs = Config()
        >>> configs["bearings"] # doctest: +ELLIPSIS
        {"oper_clearance": None, "min_clearance": None, "max_clearance": None}...
        """
        if option not in self.__dict__.keys():
            raise KeyError("Option '{}' not found.".format(option))

        return self.__dict__[option]

    def update_config(self, **kwargs):
        """Update the config options.

        This method set new values for the config dictionary according to kwargs input.
        The kwargs must respect Config attributes. It's only possible to update
        existing values from Config dictionary.
        **See Config attributes reference for infos about the dict options.

        Parameters
        ----------
        **kwargs : dict
            Dictionary with new values for corresponding keys.

        Raises
        ------
        KeyError
            Raises an error if the parameter doesn't belong to the dictionary.

        Examples
        --------
        >>> configs = Config()
        >>> configs.update_config(
        ...     rotor_properties=dict(
        ...         rotor_speeds=dict(min_speed=1000.0, max_speed=10000.0),
        ...         rotor_id=dict(type="turbine", tag="Model"),
        ...     )
        ... )
        >>> configs.rotor_properties # doctest: +ELLIPSIS
        {
            "rotor_speeds": {
                "min_speed": 1000.0,
                "max_speed": 10000.0,
                "oper_speed": None,
                "trip_speed": None,
                "speed_factor": 1.5,
                "unit": "rpm",
            },
            "rotor_id": {"type": "turbine", "tag": "Model"},
        }...
        """
        for k, v in kwargs.items():
            if k not in self.__dict__.keys():
                raise KeyError("Option '{}' not found.".format(k))
            else:
                getattr(self, k)._update(**v)
