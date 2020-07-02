"""Rotordynamic Report Configuration File."""


class _Dict:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            if isinstance(v, dict):
                setattr(self, k, self.__class__(v))
            else:
                setattr(self, k, v)

    def __repr__(self):
        return repr(self.__dict__)

    def __getitem__(self, option):
        if option not in self.__dict__.keys():
            raise KeyError("Option '{}' not found.".format(option))

        return self.__dict__[option]

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.__dict__.keys():
                raise KeyError("Option '{}' not found.".format(k))

            if isinstance(v, dict):
                getattr(self, k).update(**v)
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

        rotor_speed : dict
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
            unit : str
                Unit system to speed values. Options: "rad/s", "rpm"

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

        frequency_range : list, float
            Array with the desired range of frequencies. If None and cluster_points is
            False, it will create an array from the min_speed to max_speed
            (rotor_propeties config).
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
            Unit system. Options are "m" (meter) and "in" (inch)
            Default is "m"

    Returns
    -------
    A config object to rotordynamic report.
    """

    def __init__(self):
        # fmt: off
        # Configurating rotor properties
        self.rotor_properties = _Dict({
            "rotor_speeds": {
                "min_speed": 1000.,
                "max_speed": 10000.,
                "oper_speed": 5000.,
                "trip_speed": 12500.,
                "speed_factor ": 1.25,
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
        self.plot_ucs = {
            "stiffness_range": None,
            "num": 30,
            "num_modes": 16,
            "synchronous": False,
        }

        # Configurating unbalance response options
        self.run_unbalance_response = _Dict({
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
        return repr(self.__dict__)

    def __getitem__(self, option):
        if option not in self.__dict__.keys():
            raise KeyError("Option '{}' not found.".format(option))

        return self.__dict__[option]

    def update_config(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.__dict__.keys():
                raise KeyError("Option '{}' not found.".format(k))
            else:
                getattr(self, k).update(**v)
