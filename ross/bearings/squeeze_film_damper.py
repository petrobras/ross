import numpy as np
import math
from ross.bearing_seal_element import BearingElement
from ross.units import Q_, check_units
from ross.bearings.lubricants import lubricants_dict



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
    groove : boolean
        It can be true or false, depends
    end_seals : boolean
        It can be true or false, depending on configuration of the squeeze film damper.
    cavitation : boolean
        It can be true or false, depending on configuration of the flow type.

    Returns
    -------

    Bearing Elements

    Example
    >>> import ross as rs
    >>> Q_ = rs.units.Q_
    >>> SFD = rs.SqueezeFilmDamper(
    ... n=0,
    ... frequency=Q_([18600], "rpm"),
    ... axial_length=Q_(0.9, "inches"),
    ... journal_radius=Q_(2.55, "inches"),
    ... radial_clearance=Q_(0.003, "inches"),
    ... eccentricity_ratio=0.5,
    ... lubricant = "TEST",
    ... groove=True,
    ... end_seals=True,
    ... cavitation=True,
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
        groove=True,
        end_seals=True,
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
        self.groove = groove
        self.end_seals = end_seals
        self.cavitation = cavitation

        if (not groove) and end_seals:
            co, ko, theta, p_max = self.calculate_coeficients_with_end_seals()
        elif groove and (not end_seals):
            co, ko, theta, p_max = self.calculate_coeficients_with_groove()
        elif groove and end_seals:
            co, ko, theta, p_max = (
                self.calculate_coeficientes_with_groove_and_end_seals()
            )

        super().__init__(
            n=n,
            frequency=frequency,
            kxx=ko,
            cxx=co,
            tag=tag,
            scale_factor=scale_factor,
        )

    def calculate_coeficients_with_end_seals(self):
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
            * self.frequency
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
            * self.frequency
            * (self.journal_radius / self.radial_clearance) ** 2
        )

        if self.cavitation:
            ko = 0.0
        else:
            co = 2.0 * co
            ko = 0.0

        return co, ko, theta, p_max

    def calculate_coeficients_with_groove(self):
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
                * self.frequency
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
            * self.frequency
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

    def calculate_coeficientes_with_groove_and_end_seals(self):
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
                * self.frequency
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
            * self.frequency
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


