import numpy as np
import math
import time
from ross.bearing_seal_element import BearingElement
from ross.units import Q_, check_units
from ross.bearings.lubricants import lubricants_dict
from prettytable import PrettyTable


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

    def show_results(self):
        """Display Squeeze Film Damper calculation results in a formatted table.

        This method prints the main results from the SFD analysis using PrettyTable,
        including geometric parameters, operating conditions, and calculated coefficients
        for each frequency.

        Parameters
        ----------
        None
            This method uses the damper parameters and results stored as
            instance attributes.

        Returns
        -------
        None
            Results are printed to the console in a formatted table.
        """
        if self.frequency.size == 1:
            self._print_single_frequency_results(0)
        else:
            for i in range(self.frequency.size):
                self._print_single_frequency_results(i)

    def _print_single_frequency_results(self, freq_index):
        """Print results for a single frequency."""
        freq = self.frequency[freq_index]

        # Define a fixed width for all columns
        column_width = 20

        table = PrettyTable()
        table.field_names = ["Parameter", "Value", "Unit"]

        for field in table.field_names:
            table.max_width[field] = column_width
            table.min_width[field] = column_width

        # Set column alignment
        table.align["Parameter"] = "l"
        table.align["Value"] = "r"
        table.align["Unit"] = "c"

        # Operating conditions
        table.add_row(["Operating Speed", f"{freq * 30 / np.pi:12.1f}", "RPM"])
        table.add_row(["Geometry Type", f"{self.geometry:>12}", "-"])
        table.add_row(["Cavitation", f"{str(self.cavitation):>12}", "-"])

        # Geometric parameters
        table.add_row(["Axial Length", f"{self.axial_length:12.6f}", "m"])
        table.add_row(["Journal Radius", f"{self.journal_radius:12.6f}", "m"])
        table.add_row(["Radial Clearance", f"{self.radial_clearance:12.6e}", "m"])
        table.add_row(["Eccentricity Ratio", f"{self.eccentricity_ratio:12.4f}", "-"])

        # Lubricant viscosity
        table.add_row(["Lubricant Viscosity", f"{self.lubricant:12.4e}", "Pa*s"])

        # Display stored coefficients (already calculated during __init__)
        table.add_row(["Damping Coefficient", f"{self.cxx[freq_index]:12.4e}", "N*s/m"])
        table.add_row(["Stiffness Coefficient", f"{self.kxx[freq_index]:12.4e}", "N/m"])

        # Display stored pressure angle and max pressure for this frequency
        theta_val = self.theta[freq_index]
        p_max_val = self.p_max[freq_index]

        table.add_row(["Pressure Angle", f"{np.degrees(theta_val):12.2f}", "°"])
        table.add_row(["Pressure Angle", f"{theta_val:12.4f}", "rad"])
        table.add_row(["Maximum Pressure", f"{p_max_val:12.4e}", "Pa"])

        table_str = table.get_string()
        final_width = len(table_str.split("\n")[0])

        print("\n" + "=" * final_width)
        print(
            f"SQUEEZE FILM DAMPER RESULTS - {freq * 30 / np.pi:.1f} RPM".center(
                final_width
            )
        )
        print("=" * final_width)
        print(table)
        print("=" * final_width)

    def show_coefficients_comparison(self):
        """Display SFD coefficients comparison table across frequencies.

        This method creates and displays a formatted table comparing damping
        and stiffness coefficients across different frequencies.

        Parameters
        ----------
        None
            This method uses the frequency array and coefficients stored as
            instance attributes.

        Returns
        -------
        None
            Results are printed to the console in a formatted table.
        """
        freq_rpm = np.atleast_1d(self.frequency).astype(float) * 30.0 / np.pi

        table = PrettyTable()

        headers = [
            "Frequency [RPM]",
            "cxx [N*s/m]",
            "kxx [N/m]",
            "Pressure [Pa]",
            "Angle [°]",
        ]

        table.field_names = headers

        for i in range(len(freq_rpm)):
            theta_val = self.theta[i]
            p_max_val = self.p_max[i]

            row = [
                f"{freq_rpm[i]:.1f}",
                f"{self.cxx[i]:.4e}",
                f"{self.kxx[i]:.4e}",
                f"{p_max_val:.4e}",
                f"{np.degrees(theta_val):.2f}",
            ]

            table.add_row(row)

        # Table width
        desired_width = 20

        table.max_width = desired_width
        table.min_width = desired_width

        table_str = table.get_string()
        table_lines = table_str.split("\n")
        actual_width = len(table_lines[0])

        print("\n" + "=" * actual_width)
        print("SFD COEFFICIENTS COMPARISON TABLE".center(actual_width))
        print("=" * actual_width)
        print(table)
        print("=" * actual_width)

    def show_execution_time(self):
        """Display the simulation execution time.

        This method calculates and displays the total time spent during the
        complete bearing analysis execution, including all frequency calculations.

        Parameters
        ----------
        None
            This method uses the initial_time and final_time attributes
            stored during the simulation execution.

        Returns
        -------
        float
            Total simulation time in seconds. Returns None if simulation
            hasn't been executed yet.
        """
        if hasattr(self, "initial_time") and hasattr(self, "final_time"):
            total_time = self.final_time - self.initial_time
            print(f"Execution time: {total_time:.6f} seconds")
        else:
            print("Simulation hasn't been executed yet.")
