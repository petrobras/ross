import numpy as np
import math
import time
from ross.bearing_seal_element import BearingElement
from ross.units import Q_, check_units
from ross.bearings.lubricants import lubricants_dict
from ross.bearings.bearing_results import SqueezeFilmDamperResults


class SqueezeFilmDamper(BearingElement):
    """
    Squeeze Film Damper (SFD) element in ROSS standard format.
    Computes damping (co), stiffness (ko), maximum pressure (p_max)
    and pressure angle (theta_m) based on classical short-bearing theory.

    Parameters
    ----------
    Bearing Geometry
    ^^^^^^^^^^^^^^^^
    Describes the geometric characteristics.
    n : int
        Node in which the bearing will be located.
    frequency : list, pint.Quantity
        Array with the frequencies (rad/s).
    axial_length : float, pint.Quantity
        Bearing length. Default unit is meter.
    journal_radius : float
        Rotor radius. The unit is meter.
    radial_clearance : float
        Radial clearence between rotor and bearing. The unit is meter.
    eccentricity_ratio : float
        Normal cases, utilize the eccentricity radio < 0.4, because this
        gap has better propieties. It's a dimensioneless parametre.
    lubricant : str or dict
        Lubricant type. Can be:
        - 'ISOVG32'
        - 'ISOVG46'
        - 'ISOVG68'
        Or a dictionary with lubricant properties.
    geometry : str
        Geometry type. Can be:
        - 'groove': SFD with groove and no end seals
        - 'end_seals': SFD with end seals and no groove
        - 'groove-end_seals': SFD with both groove and end seals
        Default is 'groove'.
    cavitation : boolean
        It can be true or false, depending on configuration of the flow type.
        Default is True.
    tag : str, optional
        Tag for the element.
    scale_factor : float, optional
        Scale factor for the bearing. Default is 1.0.

    Returns
    -------
    Bearing Elements

    Attributes
    ----------
    cxx : ndarray
        Damping coefficient in N*s/m. Shape: (n_freq,).
    kxx : ndarray
        Stiffness coefficient in N/m. Shape: (n_freq,).
    theta : ndarray
        Pressure angle in radians. Shape: (n_freq,).
    p_max : ndarray
        Maximum pressure in Pa. Shape: (n_freq,).

    Example
    -------
    >>> import ross as rs
    >>> Q_ = rs.units.Q_
    >>> SFD = rs.SqueezeFilmDamper(
    ...     n=0,
    ...     frequency=Q_([18600, 20000, 22000], "rpm"),
    ...     axial_length=Q_(0.9, "inches"),
    ...     journal_radius=Q_(2.55, "inches"),
    ...     radial_clearance=Q_(0.003, "inches"),
    ...     eccentricity_ratio=0.5,
    ...     lubricant="ISOVG32",
    ...     geometry="groove",
    ...     cavitation=True,
    ... )
    """

    @check_units
    def __init__(
        self,
        n,
        frequency,
        axial_length,
        journal_radius,
        radial_clearance,
        eccentricity_ratio,
        lubricant,
        geometry="groove",
        cavitation=True,
        tag=None,
        scale_factor=1.0,
    ):
        self.axial_length = axial_length
        self.journal_radius = journal_radius
        self.radial_clearance = radial_clearance
        self.eccentricity_ratio = eccentricity_ratio
        self.frequency = frequency
        self.lubricant = lubricants_dict[lubricant]["liquid_viscosity1"]
        self.cavitation = cavitation
        self.geometry = geometry

        # Start timing
        self.initial_time = time.time()

        # Get number of frequencies
        n_freq = np.shape(self.frequency)[0]

        # Initialize arrays for storing results at each frequency
        kxx = np.zeros(n_freq)
        cxx = np.zeros(n_freq)
        theta_array = np.zeros(n_freq)
        p_max_array = np.zeros(n_freq)

        # Calculate coefficients for each frequency
        for i in range(n_freq):
            freq = self.frequency[i]

            # Calculate coefficients based on geometry
            if self.geometry == "groove":
                co, ko, theta, p_max = self.calculate_coeficients_with_groove(freq)
            elif self.geometry == "end_seals":
                co, ko, theta, p_max = self.calculate_coeficients_with_end_seals(freq)
            elif self.geometry == "groove-end_seals":
                co, ko, theta, p_max = (
                    self.calculate_coeficientes_with_groove_and_end_seals(freq)
                )
            else:
                raise ValueError(
                    f"Invalid geometry type: {geometry}. "
                    "Must be 'groove', 'end_seals', or 'groove-end_seals'"
                )

            # Store results for this frequency
            kxx[i] = ko
            cxx[i] = co
            theta_array[i] = theta
            p_max_array[i] = p_max

        # Store calculated values as instance attributes
        self.theta = theta_array
        self.p_max = p_max_array

        # End timing
        self.final_time = time.time()

        super().__init__(
            n=n,
            frequency=frequency,
            kxx=kxx,
            cxx=cxx,
            tag=tag,
            scale_factor=scale_factor,
        )

        self._results = SqueezeFilmDamperResults(
            frequency=self.frequency,
            kxx=self.kxx,
            cxx=self.cxx,
            theta=self.theta,
            p_max=self.p_max,
            axial_length=self.axial_length,
            journal_radius=self.journal_radius,
            radial_clearance=self.radial_clearance,
            eccentricity_ratio=self.eccentricity_ratio,
            lubricant_viscosity=self.lubricant,
            geometry=self.geometry,
            cavitation=self.cavitation,
            initial_time=self.initial_time,
            final_time=self.final_time,
        )

    def __getattr__(self, name):
        if "_results" in self.__dict__ and hasattr(self._results, name):
            return getattr(self._results, name)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def calculate_coeficients_with_end_seals(self, freq):
        """Calculate coefficients for a sealed SFD without a groove.

        Parameters
        ----------
        freq : float
            Operating frequency in rad/s.

        Returns
        -------
        co : float
            Damping coefficient.
        ko : float
            Stiffness coefficient.
        theta : float
            Pressure angle in radians.
        p_max : float
            Maximum pressure.
        """
        co = (
            12.0
            * np.pi
            * self.axial_length
            * (self.journal_radius / self.radial_clearance) ** 3
            * self.lubricant
        )
        co /= (2.0 + self.eccentricity_ratio**2) * np.sqrt(
            1.0 - self.eccentricity_ratio**2
        )

        ko = (
            24.0
            * self.lubricant
            * self.axial_length
            * (self.journal_radius / self.radial_clearance) ** 3
            * self.eccentricity_ratio
            * freq
        )
        ko /= (2.0 + self.eccentricity_ratio**2) * (1.0 - self.eccentricity_ratio**2)

        theta_m = -80.45 * self.eccentricity_ratio + 268.98
        theta = math.radians(theta_m)

        p_max_num = (
            2.0
            * self.eccentricity_ratio
            * (2.0 + self.eccentricity_ratio * np.cos(theta))
            * np.sin(theta)
        )
        p_max_den = (2.0 + self.eccentricity_ratio**2) * (
            1.0 + self.eccentricity_ratio * np.cos(theta)
        ) ** 2
        p_max = (
            -p_max_num
            / p_max_den
            * 6.0
            * self.lubricant
            * freq
            * (self.journal_radius / self.radial_clearance) ** 2
        )

        if self.cavitation:
            ko = 0.0
        else:
            co = 2.0 * co
            ko = 0.0

        return co, ko, theta, p_max

    def calculate_coeficients_with_groove(self, freq):
        """Calculate coefficients for an SFD with a groove and no end seals.

        Parameters
        ----------
        freq : float
            Operating frequency in rad/s.

        Returns
        -------
        co : float
            Damping coefficient.
        ko : float
            Stiffness coefficient.
        theta : float
            Pressure angle in radians.
        p_max : float
            Maximum pressure.
        """
        if self.cavitation:
            co = (
                self.lubricant
                * (self.axial_length**3)
                * self.journal_radius
                / (2.0 * self.radial_clearance**3)
            )
            co *= np.pi / ((1.0 - self.eccentricity_ratio**2) ** (3.0 / 2.0))
            co /= 4

            ko = (
                2.0
                * self.lubricant
                * freq
                * self.journal_radius
                * (self.axial_length / self.radial_clearance) ** 3
                * self.eccentricity_ratio
            )
            ko /= (1.0 - self.eccentricity_ratio**2) ** 2
            ko /= 4

        theta_m = (
            270.443
            - 191.831 * self.eccentricity_ratio
            + 218.223 * self.eccentricity_ratio**2
            - 114.803 * self.eccentricity_ratio**3
        )
        theta = math.radians(theta_m)

        p_max = (
            -1.5
            * (self.axial_length / self.radial_clearance) ** 2
            * self.lubricant
            * freq
            * self.eccentricity_ratio
            * np.sin(theta)
        )
        p_max /= (1.0 + self.eccentricity_ratio * np.cos(theta)) ** 3
        p_max /= 2

        if not self.cavitation:
            ko = 0.0
            co = (
                self.lubricant
                * (self.axial_length / self.radial_clearance) ** 3
                * self.journal_radius
                * np.pi
            )
            co /= (1.0 - self.eccentricity_ratio**2) ** (3.0 / 2.0)

        return co, ko, theta, p_max

    def calculate_coeficientes_with_groove_and_end_seals(self, freq):
        """Calculate coefficients for an SFD with both a groove and end seals.

        Parameters
        ----------
        freq : float
            Operating frequency in rad/s.

        Returns
        -------
        co : float
            Damping coefficient.
        ko : float
            Stiffness coefficient.
        theta : float
            Pressure angle in radians.
        p_max : float
            Maximum pressure.
        """
        if self.cavitation:
            co = (
                self.lubricant
                * (self.axial_length**3)
                * self.journal_radius
                / (2.0 * self.radial_clearance**3)
            )
            co *= np.pi / ((1.0 - self.eccentricity_ratio**2) ** (3.0 / 2.0))

            ko = (
                2.0
                * self.lubricant
                * freq
                * self.journal_radius
                * (self.axial_length / self.radial_clearance) ** 3
                * self.eccentricity_ratio
            )
            ko /= (1.0 - self.eccentricity_ratio**2) ** 2

        theta_m = (
            270.443
            - 191.831 * self.eccentricity_ratio
            + 218.223 * self.eccentricity_ratio**2
            - 114.803 * self.eccentricity_ratio**3
        )
        theta = math.radians(theta_m)

        p_max = (
            -1.5
            * (self.axial_length / self.radial_clearance) ** 2
            * self.lubricant
            * freq
            * self.eccentricity_ratio
            * np.sin(theta)
        )
        p_max /= (1.0 + self.eccentricity_ratio * np.cos(theta)) ** 3

        if not self.cavitation:
            ko = 0.0
            co = (
                self.lubricant
                * (self.axial_length / self.radial_clearance) ** 3
                * self.journal_radius
                * np.pi
            )
            co /= (1.0 - self.eccentricity_ratio**2) ** (3.0 / 2.0)

        return co, ko, theta, p_max
