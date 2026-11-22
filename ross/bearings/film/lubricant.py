"""Lubricant properties of the film core.

Two viscosity laws pass through two table points ``(T_1, mu_1)`` and
``(T_2, mu_2)``:

- ``"walther"`` (ASTM D341): ``log10(log10(nu + 0.7)) = A - B log10(T)``, with
  the kinematic viscosity ``nu = mu / rho(T)`` in mm^2/s (cSt) and ``T`` in K;
- ``"exponential"``: ``mu = mu_1 exp(beta (T - T_1))`` with
  ``beta = ln(mu_2 / mu_1) / (T_2 - T_1)``, the two-point law of the journal
  engine and of ``ThrustPad``.

The density, specific heat and thermal conductivity used by the energy
equation are linear in the temperature about ``T_1``::

    rho(T) = rho_1 (1 - a_v (T - T_1)),   c_p(T) = c_1 + s_c (T - T_1),   k

with the volumetric expansion ``a_v`` and the slope ``s_c`` zero by default
(constant properties, the values of ``ross.bearings.lubricants``). The Walther
constants are fitted on ``nu_i = mu_i / rho(T_i)``, so the two table
viscosities are reproduced for any ``a_v``.

Shared with the journal engine (``ross.bearings.fluid_film``):

- :func:`lubricant_data` resolves a table name or a property dictionary (the
  format of ``ross.bearings.lubricants.lubricants_dict``) with the same errors
  for both engines;
- :func:`exponential_viscosity` is the two-point exponential law, and
  :func:`exponential_viscosity_jit` the same function compiled by numba for the
  journal kernels; :func:`walther_viscosity` and :func:`walther_viscosity_jit`
  are the Walther law of :meth:`Lubricant.viscosity` written for the same use.

Datasheet properties (:meth:`Lubricant.from_datasheet`): an oil datasheet
gives the kinematic viscosities at 40 and 100 C (ISO 3104) and the density at
15 C. The density follows the thermal-expansion form of API MPMS 11.1 (ASTM
D1250) for lubricating oils, :func:`api_density`, and the specific heat and
thermal conductivity the correlations of Cragoe (1929),
:func:`cragoe_specific_heat` and :func:`cragoe_thermal_conductivity`.

References
----------
C. S. Cragoe, "Thermal properties of petroleum products", Bureau of
Standards Miscellaneous Publication No. 97, 1929: equation (4), p. 24
(thermal conductivity) and equation (5), p. 26 (specific heat); Table 20,
p. 47, gives both in metric units.
"""

import numpy as np
from numba import njit

from ross.bearings.lubricants import lubricants_dict
from ross.units import _to_base_unit, check_units

__all__ = [
    "Lubricant",
    "VISCOSITY_MODELS",
    "LUBRICANT_KEYS",
    "lubricant_data",
    "exponential_viscosity",
    "exponential_viscosity_jit",
    "walther_viscosity",
    "walther_viscosity_jit",
    "api_density",
    "cragoe_specific_heat",
    "cragoe_thermal_conductivity",
]

VISCOSITY_MODELS = ("walther", "exponential")
_WALTHER_SHIFT = 0.7  # cSt, ASTM D341 constant
# kg/(m^3 K), lubricating-oil constant of the API MPMS 11.1 expansion form
# (K0 = 0). Not checked against the standard: the value has not been compared
# with a copy of API MPMS 11.1 or ASTM D1250.
_API_K1 = 0.6278
_T15 = 288.15  # K, API reference temperature
_T40, _T100 = 313.15, 373.15  # K, ISO 3104 datasheet temperatures
_RHO_WATER_60F = 999.0  # kg/m^3, water at 15.6 C (60 F), Cragoe's specific gravity

#: Keys of a lubricant dictionary that every film model needs (the format of
#: ``ross.bearings.lubricants.lubricants_dict``, SI, temperatures in K). The
#: thermal models also need ``liquid_specific_heat`` and
#: ``liquid_thermal_conductivity``.
LUBRICANT_KEYS = (
    "liquid_viscosity1",
    "temperature1",
    "liquid_viscosity2",
    "temperature2",
    "liquid_density",
)


def lubricant_data(lubricant, required=LUBRICANT_KEYS):
    """Property dictionary of a lubricant name or dictionary.

    The lubricant resolver of the fluid-film bearings: a name is
    looked up in ``ross.bearings.lubricants.lubricants_dict``, a dictionary is
    taken as is, and both are checked for the ``required`` keys.

    Parameters
    ----------
    lubricant : str or dict
        Key of ``lubricants_dict``, or a dictionary in its format (SI,
        temperatures in K).
    required : sequence of str, optional
        Keys the dictionary must hold. Default :data:`LUBRICANT_KEYS`.

    Returns
    -------
    dict
        The property dictionary (the table entry itself for a name).

    Raises
    ------
    ValueError
        Unknown name, or a dictionary without one of the ``required`` keys.
    TypeError
        Neither a name nor a dictionary.

    Examples
    --------
    >>> lubricant_data("ISOVG46")["liquid_density"]
    862.9
    >>> lubricant_data({"liquid_viscosity1": 0.02})
    Traceback (most recent call last):
        ...
    ValueError: lubricant is missing properties: ['temperature1', 'liquid_viscosity2', 'temperature2', 'liquid_density']
    """
    if isinstance(lubricant, str):
        try:
            properties = lubricants_dict[lubricant]
        except KeyError:
            raise ValueError(
                f"lubricant must be one of {sorted(lubricants_dict)} or a "
                f"dict of properties, not {lubricant!r}"
            ) from None
    elif isinstance(lubricant, dict):
        properties = lubricant
    else:
        raise TypeError(
            "lubricant must be a lubricant name (str) or a dict of "
            f"properties, not {type(lubricant).__name__}"
        )
    missing = [name for name in required if name not in properties]
    if missing:
        raise ValueError(f"lubricant is missing properties: {missing}")
    return properties


def exponential_viscosity(
    viscosity_1, viscosity_2, temperature_1, temperature_2, temperature
):
    """Two-point exponential viscosity law ``mu_1 exp(beta (T - T_1))``.

    ``beta = ln(mu_2 / mu_1) / (T_2 - T_1)``. The journal engine evaluates it
    with numpy (``fluid_film.state.Lubricant.viscosity_at``) and in its numba
    kernels (:func:`exponential_viscosity_jit`, the same source compiled), and
    :class:`Lubricant` with ``viscosity_model="exponential"``.

    Parameters
    ----------
    viscosity_1, viscosity_2 : float
        Dynamic viscosity at ``temperature_1`` and ``temperature_2``, Pa*s.
    temperature_1, temperature_2 : float
        Table temperatures, K (different).
    temperature : float or array_like
        Temperature, K.

    Returns
    -------
    float or numpy.ndarray
        Dynamic viscosity, Pa*s.

    Examples
    --------
    >>> round(float(exponential_viscosity(0.04, 0.01, 313.15, 373.15, 343.15)), 12)
    0.02
    """
    beta = np.log(viscosity_2 / viscosity_1) / (temperature_2 - temperature_1)
    return viscosity_1 * np.exp(beta * (temperature - temperature_1))


#: :func:`exponential_viscosity` compiled by numba (scalar arguments), for the
#: journal kernels.
exponential_viscosity_jit = njit(cache=True, fastmath=False)(exponential_viscosity)


def walther_viscosity(a, b, density, expansion, t_ref, temperature):
    """Walther (ASTM D341) dynamic viscosity.

    ``log10(log10(nu + 0.7)) = a - b log10(T)`` with the kinematic viscosity
    ``nu`` in mm^2/s (cSt) and ``T`` in K, converted to a dynamic viscosity
    with the linear density ``density (1 - expansion (T - t_ref))``. This is
    the law of :meth:`Lubricant.viscosity` with ``viscosity_model="walther"``
    (same arithmetic): the result is ``nu * 1e-6 * rho(T)``, Pa*s.

    Parameters
    ----------
    a, b : float
        Walther constants (:attr:`Lubricant.walther_a`,
        :attr:`Lubricant.walther_b`).
    density : float
        Density at ``t_ref``, kg/m^3.
    expansion : float
        Volumetric expansion coefficient, 1/K.
    t_ref : float
        Reference temperature of the density, K.
    temperature : float or array_like
        Temperature, K.

    Returns
    -------
    float or numpy.ndarray
        Dynamic viscosity, Pa*s.

    Examples
    --------
    >>> oil = Lubricant(0.039, 313.15, 0.0054, 373.15, 855.0)
    >>> mu = walther_viscosity(oil.walther_a, oil.walther_b, 855.0, 0.0, 313.15, 373.15)
    >>> round(float(mu), 12)
    0.0054
    """
    nu = 10.0 ** (10.0 ** (a - b * np.log10(temperature))) - _WALTHER_SHIFT
    return nu * 1e-6 * (density * (1.0 - expansion * (temperature - t_ref)))


#: :func:`walther_viscosity` compiled by numba (scalar arguments), for the
#: journal kernels.
walther_viscosity_jit = njit(cache=True, fastmath=False)(walther_viscosity)


def api_density(rho15, temperature):
    """Density of a lubricating oil, API MPMS 11.1 (ASTM D1250) form.

    ``rho(T) = rho15 exp(-a15 dT (1 + 0.8 a15 dT))`` with
    ``a15 = K1 / rho15``, ``K1 = 0.6278 kg/(m^3 K)`` (``K0 = 0``, the
    lubricating-oil constants) and ``dT = T - 288.15 K``. The form and the
    constant ``K1`` have not been checked against a copy of the standard.

    Parameters
    ----------
    rho15 : float, pint.Quantity
        Density at 15 C, kg/m^3.
    temperature : float, array_like, pint.Quantity
        Temperature, K.

    Returns
    -------
    float or numpy.ndarray
        Density, kg/m^3 (``rho15`` exactly at 15 C).

    Examples
    --------
    >>> api_density(870.0, 288.15)
    870.0
    >>> round(float(api_density(870.0, 313.15)), 3)
    854.223
    """
    rho15 = float(_to_base_unit(rho15, "kg/m**3"))
    t = np.asarray(_to_base_unit(temperature, "K"), dtype=float)
    a15 = _API_K1 / rho15
    dt = t - _T15
    rho = rho15 * np.exp(-a15 * dt * (1.0 + 0.8 * a15 * dt))
    return rho if rho.ndim else float(rho)


def cragoe_specific_heat(specific_gravity, temperature):
    """Specific heat of a petroleum oil, Cragoe (1929).

    ``c_p = (1684 + 3.39 t) / sqrt(d)`` J/(kg K), with ``t`` in C and ``d``
    the specific gravity at 60/60 F: equation (5) of Cragoe (1929),
    ``c = (0.388 + 0.00045 t_F) / sqrt(d)`` Btu/(lb F), converted to SI and
    rounded (1684.8 and 3.391 with the international-table calorie). The
    publication gives it for ``d`` from 0.72 to 0.96 and 0 to 400 C, with
    an estimated accuracy of 5 %.

    Parameters
    ----------
    specific_gravity : float
        Specific gravity ``d`` at 60/60 F.
    temperature : float, array_like, pint.Quantity
        Temperature, K.

    Returns
    -------
    float or numpy.ndarray
        Specific heat, J/(kg K).

    Examples
    --------
    >>> round(float(cragoe_specific_heat(0.8718, 313.15)), 1)
    1948.8
    """
    t = np.asarray(_to_base_unit(temperature, "K"), dtype=float) - 273.15
    cp = (1684.0 + 3.39 * t) / np.sqrt(float(specific_gravity))
    return cp if cp.ndim else float(cp)


def cragoe_thermal_conductivity(specific_gravity, temperature):
    """Thermal conductivity of a petroleum oil, Cragoe (1929).

    ``k = 0.1172 (1 - 0.00054 t) / d`` W/(m K), with ``t`` in C and ``d``
    the specific gravity at 60/60 F: equation (4) of Cragoe (1929),
    ``K = 0.813 [1 - 0.0003 (t_F - 32)] / d`` Btu in/(h ft^2 F), converted
    to SI and rounded (0.11726). The publication gives it for ``d`` from
    0.78 to 0.95 and 0 to 200 C, with an estimated accuracy of 10 %.

    Parameters
    ----------
    specific_gravity : float
        Specific gravity ``d`` at 60/60 F.
    temperature : float, array_like, pint.Quantity
        Temperature, K.

    Returns
    -------
    float or numpy.ndarray
        Thermal conductivity, W/(m K).

    Examples
    --------
    >>> round(float(cragoe_thermal_conductivity(0.8718, 313.15)), 4)
    0.1315
    """
    t = np.asarray(_to_base_unit(temperature, "K"), dtype=float) - 273.15
    k = 0.1172 * (1.0 - 0.00054 * t) / float(specific_gravity)
    return k if k.ndim else float(k)


def _si(value, unit):
    """Magnitude in ``unit`` of a float or pint quantity (None passes)."""
    if value is None:
        return None
    if hasattr(value, "to"):
        return float(value.to(unit).m)
    return float(value)


class Lubricant:
    """Lubricant with a two-point viscosity-temperature law.

    Parameters
    ----------
    viscosity_1, viscosity_2 : float, pint.Quantity
        Dynamic viscosity at ``temperature_1`` and ``temperature_2``, Pa*s.
    temperature_1, temperature_2 : float, pint.Quantity
        Table temperatures, K.
    density : float, pint.Quantity
        Density at ``temperature_1``, kg/m^3 (the Walther law acts on the
        kinematic viscosity).
    viscosity_model : str, optional
        ``"walther"`` (default) or ``"exponential"``.
    name : str, optional
        Label, for example the key in ``lubricants_dict``.
    specific_heat : float, pint.Quantity, optional
        Specific heat at ``temperature_1``, J/(kg K). Required by the thermal
        models; default None.
    thermal_conductivity : float, pint.Quantity, optional
        Thermal conductivity, W/(m K). Default None.
    volumetric_expansion : float, pint.Quantity, optional
        Volumetric expansion coefficient ``a_v``, 1/K, in
        ``rho(T) = rho_1 (1 - a_v (T - T_1))``. Default 0 (constant density).
    specific_heat_slope : float, optional
        ``dc_p/dT``, J/(kg K^2). Default 0 (constant specific heat).

    Attributes
    ----------
    density, specific_heat, thermal_conductivity : float
        Values at ``temperature_1`` (``specific_heat`` and
        ``thermal_conductivity`` may be None).
    walther_a, walther_b : float
        Walther constants ``A`` and ``B`` (``nu`` in cSt, ``T`` in K).
    exponential_b : float
        Exponent ``b = -beta`` of the exponential law
        ``mu = mu_1 exp(-b (T - T_1))``, 1/K.
    isoviscous : bool
        Whether both table viscosities are equal (the viscosity is then
        exactly ``viscosity_1`` at every temperature).

    Examples
    --------
    >>> oil = Lubricant(0.039, 313.15, 0.0054, 373.15, 855.0)
    >>> round(float(oil.viscosity(313.15)), 12), round(float(oil.viscosity(373.15)), 12)
    (0.039, 0.0054)
    >>> round(float(oil.viscosity(333.15)) * 1e3, 2)
    16.91
    >>> oil = Lubricant(0.039, 313.15, 0.0054, 373.15, 855.0, specific_heat=2090.0,
    ...                 thermal_conductivity=0.13)
    >>> float(oil.density_at(350.0)), float(oil.specific_heat_at(350.0))
    (855.0, 2090.0)
    """

    @check_units
    def __init__(
        self,
        viscosity_1,
        temperature_1,
        viscosity_2,
        temperature_2,
        density,
        viscosity_model="walther",
        name=None,
        specific_heat=None,
        thermal_conductivity=None,
        volumetric_expansion=0.0,
        specific_heat_slope=0.0,
    ):
        if viscosity_model not in VISCOSITY_MODELS:
            raise ValueError(
                f"viscosity_model must be one of {VISCOSITY_MODELS}, got {viscosity_model!r}"
            )
        mu1, mu2 = float(viscosity_1), float(viscosity_2)
        t1, t2 = float(temperature_1), float(temperature_2)
        rho = float(density)
        if min(mu1, mu2) <= 0.0:
            raise ValueError("viscosities must be positive")
        if min(t1, t2) <= 0.0:
            raise ValueError("temperatures must be positive (K)")
        if rho <= 0.0:
            raise ValueError("density must be positive")
        if t1 == t2 and mu1 != mu2:
            raise ValueError("two different viscosities at the same temperature")
        self.viscosity_1, self.viscosity_2 = mu1, mu2
        self.temperature_1, self.temperature_2 = t1, t2
        self.density = rho
        self.viscosity_model = viscosity_model
        self.name = name
        self.specific_heat = _si(specific_heat, "J/(kg*K)")
        self.thermal_conductivity = _si(thermal_conductivity, "W/(m*K)")
        self.volumetric_expansion = float(_si(volumetric_expansion, "1/K"))
        self.specific_heat_slope = float(_si(specific_heat_slope, "J/(kg*K**2)"))
        for label, value in (
            ("specific_heat", self.specific_heat),
            ("thermal_conductivity", self.thermal_conductivity),
        ):
            if value is not None and not value > 0.0:
                raise ValueError(f"{label} must be positive")
        self.isoviscous = mu1 == mu2

        self.exponential_b = 0.0 if t1 == t2 else -np.log(mu2 / mu1) / (t2 - t1)
        rho1 = float(self.density_at(t1))
        rho2 = float(self.density_at(t2))
        if min(rho1, rho2) <= 0.0:
            raise ValueError("the density must stay positive at the table temperatures")
        nu1, nu2 = mu1 / rho1 * 1e6, mu2 / rho2 * 1e6
        if viscosity_model == "walther" and min(nu1, nu2) + _WALTHER_SHIFT <= 1.0:
            raise ValueError(
                "the Walther law needs a kinematic viscosity above 0.3 cSt at both "
                "table points; use viscosity_model='exponential'"
            )
        if viscosity_model == "walther":
            z1 = np.log10(np.log10(nu1 + _WALTHER_SHIFT))
            z2 = np.log10(np.log10(nu2 + _WALTHER_SHIFT))
            if t1 == t2:
                self.walther_b = 0.0
            else:
                self.walther_b = (z1 - z2) / (np.log10(t2) - np.log10(t1))
            self.walther_a = z1 + self.walther_b * np.log10(t1)
        else:
            self.walther_a = self.walther_b = np.nan

    @classmethod
    def from_dict(cls, data, viscosity_model="walther", name=None):
        """Lubricant from a ``lubricants_dict``-style dictionary.

        Parameters
        ----------
        data : dict
            Keys ``liquid_viscosity1``, ``temperature1``, ``liquid_viscosity2``,
            ``temperature2`` and ``liquid_density`` (SI, temperatures in K);
            optional ``liquid_specific_heat``, ``liquid_thermal_conductivity``,
            ``volumetric_expansion``, ``specific_heat_slope`` and
            ``viscosity_model`` (the format of :meth:`to_dict`).
        viscosity_model : str or None, optional
            ``"walther"`` (default) or ``"exponential"``. None takes the
            dictionary's ``viscosity_model`` key, ``"walther"`` without it.
        name : str, optional
            Label.

        Returns
        -------
        Lubricant
            The lubricant.

        Raises
        ------
        ValueError, TypeError
            As :func:`lubricant_data` (a key of :data:`LUBRICANT_KEYS` missing,
            not a dictionary).

        Examples
        --------
        >>> from ross.bearings.lubricants import lubricants_dict
        >>> oil = Lubricant.from_dict(lubricants_dict["ISOVG46"])
        >>> round(float(oil.viscosity(313.15)), 6)
        0.039693
        """
        data = lubricant_data(data)
        if viscosity_model is None:
            viscosity_model = data.get("viscosity_model", "walther")
        return cls(
            data["liquid_viscosity1"],
            data["temperature1"],
            data["liquid_viscosity2"],
            data["temperature2"],
            data["liquid_density"],
            viscosity_model=viscosity_model,
            name=name,
            specific_heat=data.get("liquid_specific_heat"),
            thermal_conductivity=data.get("liquid_thermal_conductivity"),
            volumetric_expansion=data.get("volumetric_expansion", 0.0),
            specific_heat_slope=data.get("specific_heat_slope", 0.0),
        )

    @classmethod
    def from_name(cls, name, viscosity_model="walther"):
        """Lubricant from a key of ``ross.bearings.lubricants.lubricants_dict``.

        Parameters
        ----------
        name : str
            Lubricant name, for example ``"ISOVG46"``.
        viscosity_model : str, optional
            ``"walther"`` (default) or ``"exponential"``.

        Returns
        -------
        Lubricant
            The lubricant.

        Raises
        ------
        ValueError
            Unknown name (the message of :func:`lubricant_data`).

        Examples
        --------
        >>> Lubricant.from_name("ISOVG68").name
        'ISOVG68'
        """
        return cls.from_dict(lubricant_data(name), viscosity_model, name=name)

    @classmethod
    def from_datasheet(
        cls,
        nu40,
        nu100,
        rho15,
        viscosity_model="walther",
        name=None,
        specific_heat=None,
        thermal_conductivity=None,
        volumetric_expansion=None,
    ):
        """Lubricant from the values of an oil datasheet.

        The two table points are 40 and 100 C (ISO 3104 kinematic
        viscosities). The density is the API form of :func:`api_density`
        linearised at 40 C: ``density = api_density(rho15, 40 C)`` and
        ``volumetric_expansion = a15 (1 + 1.6 a15 (40 - 15))`` (the slope of
        the API form there), which stays within 0.3 % of :func:`api_density`
        from 15 to 120 C. The table dynamic viscosities are
        ``mu_i = nu_i density_at(T_i)`` with that linear density, so
        :meth:`viscosity` divided by :meth:`density_at` returns ``nu40`` and
        ``nu100`` at 40 and 100 C. The specific heat (at 40 C, with the
        Cragoe slope ``3.39 / sqrt(d)``) and the thermal conductivity (at
        40 C, constant) follow Cragoe with ``d = rho15 / 999`` (water at
        15.6 C).

        Parameters
        ----------
        nu40, nu100 : float, pint.Quantity
            Kinematic viscosity at 40 and 100 C, m^2/s (a pint quantity in
            cSt or mm^2/s is converted).
        rho15 : float, pint.Quantity
            Density at 15 C, kg/m^3.
        viscosity_model : str, optional
            ``"walther"`` (default) or ``"exponential"``.
        name : str, optional
            Label.
        specific_heat : float, pint.Quantity, optional
            Specific heat at 40 C, J/(kg K), used as a constant
            (``specific_heat_slope = 0``). Default None: Cragoe.
        thermal_conductivity : float, pint.Quantity, optional
            Thermal conductivity, W/(m K). Default None: Cragoe at 40 C.
        volumetric_expansion : float, pint.Quantity, optional
            Volumetric expansion coefficient about 40 C, 1/K. Default None:
            the API slope at 40 C.

        Returns
        -------
        Lubricant
            The lubricant, ``temperature_1 = 313.15 K``,
            ``temperature_2 = 373.15 K``.

        Raises
        ------
        ValueError
            A viscosity or the density not positive and finite.

        Examples
        --------
        >>> from ross.units import Q_
        >>> oil = Lubricant.from_datasheet(Q_(32.0, "cSt"), Q_(5.4, "cSt"), 870.0)
        >>> round(float(oil.viscosity(313.15) / oil.density_at(313.15)) * 1e6, 9)
        32.0
        >>> round(oil.specific_heat, 1), round(oil.thermal_conductivity, 4)
        (1949.8, 0.1317)
        """
        nu_40 = _to_base_unit(nu40, "m**2/s")
        nu_100 = _to_base_unit(nu100, "m**2/s")
        rho_15 = _to_base_unit(rho15, "kg/m**3")
        for label, value in (("nu40", nu_40), ("nu100", nu_100), ("rho15", rho_15)):
            if not (np.isfinite(value) and value > 0.0):
                raise ValueError(f"{label} must be positive and finite, got {value!r}")
        nu_40, nu_100, rho_15 = float(nu_40), float(nu_100), float(rho_15)
        if volumetric_expansion is None:
            a15 = _API_K1 / rho_15
            volumetric_expansion = a15 * (1.0 + 1.6 * a15 * (_T40 - _T15))
        else:
            volumetric_expansion = float(_to_base_unit(volumetric_expansion, "1/K"))
        density = api_density(rho_15, _T40)
        gravity = rho_15 / _RHO_WATER_60F
        specific_heat_slope = 0.0
        if specific_heat is None:
            specific_heat = cragoe_specific_heat(gravity, _T40)
            specific_heat_slope = 3.39 / np.sqrt(gravity)
        if thermal_conductivity is None:
            thermal_conductivity = cragoe_thermal_conductivity(gravity, _T40)
        density_100 = density * (1.0 - volumetric_expansion * (_T100 - _T40))
        return cls(
            nu_40 * density,
            _T40,
            nu_100 * density_100,
            _T100,
            density,
            viscosity_model=viscosity_model,
            name=name,
            specific_heat=specific_heat,
            thermal_conductivity=thermal_conductivity,
            volumetric_expansion=volumetric_expansion,
            specific_heat_slope=specific_heat_slope,
        )

    @classmethod
    def create(cls, lubricant, viscosity_model="walther"):
        """Lubricant from a name, a dictionary or a ``Lubricant``.

        Parameters
        ----------
        lubricant : str, dict or Lubricant
            Key of ``lubricants_dict``, a dictionary in its format, or a
            ``Lubricant`` (returned unchanged).
        viscosity_model : str, optional
            Law used for a name or a dictionary. Default is ``"walther"``.

        Returns
        -------
        Lubricant
            The lubricant.

        Raises
        ------
        ValueError, TypeError
            As :func:`lubricant_data` (unknown name, missing key, other type).

        Examples
        --------
        >>> Lubricant.create("ISOVG32", "exponential").viscosity_model
        'exponential'
        """
        if isinstance(lubricant, Lubricant):
            return lubricant
        if isinstance(lubricant, str):
            return cls.from_name(lubricant, viscosity_model)
        # a dict (anything else: the TypeError of lubricant_data)
        return cls.from_dict(lubricant, viscosity_model)

    def viscosity(self, temperature):
        """Dynamic viscosity at ``temperature``.

        Parameters
        ----------
        temperature : float or array_like
            Temperature, K.

        Returns
        -------
        float or numpy.ndarray
            Dynamic viscosity, Pa*s.

        Examples
        --------
        >>> oil = Lubricant(0.04, 313.15, 0.01, 373.15, 860.0, "exponential")
        >>> round(float(oil.viscosity(343.15)), 6)
        0.02
        """
        t = self._temperature(temperature)
        if self.isoviscous:
            mu = np.full(t.shape, self.viscosity_1)
        elif self.viscosity_model == "exponential":
            mu = np.asarray(
                exponential_viscosity(
                    self.viscosity_1,
                    self.viscosity_2,
                    self.temperature_1,
                    self.temperature_2,
                    t,
                )
            )
        else:
            mu = np.asarray(
                walther_viscosity(
                    self.walther_a,
                    self.walther_b,
                    self.density,
                    self.volumetric_expansion,
                    self.temperature_1,
                    t,
                )
            )
        return mu if mu.ndim else float(mu)

    @staticmethod
    def _temperature(temperature):
        t = np.asarray(temperature, dtype=float)
        if np.any(~(t > 0.0)):
            raise ValueError("temperatures must be positive (K)")
        return t

    def density_at(self, temperature):
        """Density ``rho_1 (1 - a_v (T - T_1))``.

        Parameters
        ----------
        temperature : float or array_like
            Temperature, K.

        Returns
        -------
        float or numpy.ndarray
            Density, kg/m^3.

        Examples
        --------
        >>> oil = Lubricant(0.04, 313.15, 0.01, 373.15, 860.0, volumetric_expansion=7e-4)
        >>> round(float(oil.density_at(323.15)), 6)
        853.98
        """
        t = self._temperature(temperature)
        rho = self.density * (
            1.0 - self.volumetric_expansion * (t - self.temperature_1)
        )
        return rho if rho.ndim else float(rho)

    def _require(self, name):
        value = getattr(self, name)
        if value is None:
            raise ValueError(
                f"the lubricant has no {name}; give it (dictionary key "
                f"'liquid_{name}') for the thermal models"
            )
        return value

    def specific_heat_at(self, temperature):
        """Specific heat ``c_1 + s_c (T - T_1)``.

        Parameters
        ----------
        temperature : float or array_like
            Temperature, K.

        Returns
        -------
        float or numpy.ndarray
            Specific heat, J/(kg K).

        Examples
        --------
        >>> oil = Lubricant(0.04, 313.15, 0.01, 373.15, 860.0, specific_heat=1950.0,
        ...                 specific_heat_slope=3.5)
        >>> round(float(oil.specific_heat_at(333.15)), 6)
        2020.0
        """
        t = self._temperature(temperature)
        cp = self._require("specific_heat") + self.specific_heat_slope * (
            t - self.temperature_1
        )
        return cp if cp.ndim else float(cp)

    def thermal_conductivity_at(self, temperature):
        """Thermal conductivity (constant).

        Parameters
        ----------
        temperature : float or array_like
            Temperature, K.

        Returns
        -------
        float or numpy.ndarray
            Thermal conductivity, W/(m K).

        Examples
        --------
        >>> oil = Lubricant(0.04, 313.15, 0.01, 373.15, 860.0, thermal_conductivity=0.13)
        >>> float(oil.thermal_conductivity_at(350.0))
        0.13
        """
        t = self._temperature(temperature)
        k = np.full(t.shape, float(self._require("thermal_conductivity")))
        return k if k.ndim else float(k)

    def volumetric_heat_capacity(self, temperature):
        """Product ``rho(T) c_p(T)``, J/(m^3 K).

        Parameters
        ----------
        temperature : float or array_like
            Temperature, K.

        Returns
        -------
        float or numpy.ndarray
            ``rho c_p``.

        Examples
        --------
        >>> oil = Lubricant(0.04, 313.15, 0.01, 373.15, 800.0, specific_heat=2000.0)
        >>> float(oil.volumetric_heat_capacity(330.0))
        1600000.0
        """
        rho_c = np.asarray(self.density_at(temperature)) * np.asarray(
            self.specific_heat_at(temperature)
        )
        return rho_c if rho_c.ndim else float(rho_c)

    def to_dict(self):
        """Property dictionary in the ``lubricants_dict`` format.

        Holds the five keys of :data:`LUBRICANT_KEYS`, the optional
        ``liquid_specific_heat`` and ``liquid_thermal_conductivity`` (when
        set), ``volumetric_expansion``, ``specific_heat_slope`` and
        ``viscosity_model``; :meth:`from_dict` with ``viscosity_model=None``
        rebuilds the same lubricant.

        Returns
        -------
        dict
            SI values, temperatures in K.

        Examples
        --------
        >>> oil = Lubricant.from_name("ISOVG46", "exponential")
        >>> data = oil.to_dict()
        >>> data["viscosity_model"], data["liquid_density"]
        ('exponential', 862.9)
        >>> Lubricant.from_dict(data, None).viscosity(350.0) == oil.viscosity(350.0)
        True
        """
        data = {
            "liquid_viscosity1": self.viscosity_1,
            "temperature1": self.temperature_1,
            "liquid_viscosity2": self.viscosity_2,
            "temperature2": self.temperature_2,
            "liquid_density": self.density,
        }
        if self.specific_heat is not None:
            data["liquid_specific_heat"] = self.specific_heat
        if self.thermal_conductivity is not None:
            data["liquid_thermal_conductivity"] = self.thermal_conductivity
        data["volumetric_expansion"] = self.volumetric_expansion
        data["specific_heat_slope"] = self.specific_heat_slope
        data["viscosity_model"] = self.viscosity_model
        return data

    def __repr__(self):
        return (
            f"Lubricant(viscosity_1={self.viscosity_1!r}, temperature_1={self.temperature_1!r}, "
            f"viscosity_2={self.viscosity_2!r}, temperature_2={self.temperature_2!r}, "
            f"density={self.density!r}, viscosity_model={self.viscosity_model!r}, "
            f"name={self.name!r}, specific_heat={self.specific_heat!r}, "
            f"thermal_conductivity={self.thermal_conductivity!r}, "
            f"volumetric_expansion={self.volumetric_expansion!r}, "
            f"specific_heat_slope={self.specific_heat_slope!r})"
        )
