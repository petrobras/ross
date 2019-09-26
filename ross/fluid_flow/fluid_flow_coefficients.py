import numpy as np


def calculate_analytical_stiffness_matrix(load, eccentricity_ratio, radial_clearance):
    """Returns the stiffness matrix calculated analytically.
    Suitable only for short bearings.
    Parameters
    -------
    load: float
        Load applied to the rotor (N).
    eccentricity_ratio: float
        The ratio between the journal displacement, called just eccentricity, and
        the radial clearance.
    radial_clearance: float
        Difference between both stator and rotor radius, regardless of eccentricity.
    Returns
    -------
    list of floats
        A list of length four including stiffness floats in this order: kxx, kxy, kyx, kyy
    Examples
    --------
    >>> from ross.fluid_flow.fluid_flow import pressure_matrix_example
    >>> my_fluid_flow = pressure_matrix_example()
    >>> load = my_fluid_flow.load
    >>> eccentricity_ratio = my_fluid_flow.eccentricity_ratio
    >>> radial_clearance = my_fluid_flow.radial_clearance
    >>> calculate_analytical_stiffness_matrix(load, eccentricity_ratio, radial_clearance) # doctest: +ELLIPSIS
    [...
    """
    # fmt: off
    h0 = 1.0 / (((np.pi ** 2) * (1 - eccentricity_ratio ** 2) + 16 * eccentricity_ratio ** 2) ** 1.5)
    a = load / radial_clearance
    kxx = a * h0 * 4 * ((np.pi ** 2) * (2 - eccentricity_ratio ** 2) + 16 * eccentricity_ratio ** 2)
    kxy = (a * h0 * np.pi * ((np.pi ** 2) * (1 - eccentricity_ratio ** 2) ** 2 -
                             16 * eccentricity_ratio ** 4) /
           (eccentricity_ratio * np.sqrt(1 - eccentricity_ratio ** 2)))
    kyx = (-a * h0 * np.pi * ((np.pi ** 2) * (1 - eccentricity_ratio ** 2) *
                              (1 + 2 * eccentricity_ratio ** 2) +
                              (32 * eccentricity_ratio ** 2) * (1 + eccentricity_ratio ** 2)) /
           (eccentricity_ratio * np.sqrt(1 - eccentricity_ratio ** 2)))
    kyy = (a * h0 * 4 * ((np.pi ** 2) * (1 + 2 * eccentricity_ratio ** 2) +
                         ((32 * eccentricity_ratio ** 2) *
                          (1 + eccentricity_ratio ** 2)) / (1 - eccentricity_ratio ** 2)))
    # fmt: on
    return [kxx, kxy, kyx, kyy]


def calculate_analytical_damping_matrix(load, eccentricity_ratio, radial_clearance, omega):
    """Returns the damping matrix calculated analytically.
    Suitable only for short bearings.
    Returns
    -------
    list of floats
        A list of length four including damping floats in this order: cxx, cxy, cyx, cyy
    Examples
    --------
    >>> from ross.fluid_flow.fluid_flow import pressure_matrix_example
    >>> my_fluid_flow = pressure_matrix_example()
    >>> load = my_fluid_flow.load
    >>> eccentricity_ratio = my_fluid_flow.eccentricity_ratio
    >>> radial_clearance = my_fluid_flow.radial_clearance
    >>> omega = my_fluid_flow.omega
    >>> calculate_analytical_damping_matrix(load, eccentricity_ratio,
    ...                                       radial_clearance, omega) # doctest: +ELLIPSIS
    [...
    """
    # fmt: off
    h0 = 1.0 / (((np.pi ** 2) * (1 - eccentricity_ratio ** 2) + 16 * eccentricity_ratio ** 2) ** 1.5)
    a = load / (radial_clearance * omega)
    cxx = (a * h0 * 2 * np.pi * np.sqrt(1 - eccentricity_ratio ** 2) *
           ((np.pi ** 2) * (
                   1 + 2 * eccentricity_ratio ** 2) - 16 * eccentricity_ratio ** 2) / eccentricity_ratio)
    cxy = (-a * h0 * 8 * (
            (np.pi ** 2) * (1 + 2 * eccentricity_ratio ** 2) - 16 * eccentricity_ratio ** 2))
    cyx = cxy
    cyy = (a * h0 * (2 * np.pi * (
            (np.pi ** 2) * (1 - eccentricity_ratio ** 2) ** 2 + 48 * eccentricity_ratio ** 2)) /
           (eccentricity_ratio * np.sqrt(1 - eccentricity_ratio ** 2)))
    # fmt: on
    return [cxx, cxy, cyx, cyy]

