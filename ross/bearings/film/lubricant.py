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
  journal kernels.
"""

import numpy as np
from numba import njit

from ross.bearings.lubricants import lubricants_dict
from ross.units import check_units

__all__ = [
    "Lubricant",
    "VISCOSITY_MODELS",
    "LUBRICANT_KEYS",
    "lubricant_data",
    "exponential_viscosity",
    "exponential_viscosity_jit",
]

VISCOSITY_MODELS = ("walther", "exponential")
_WALTHER_SHIFT = 0.7  # cSt, ASTM D341 constant

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
            ``volumetric_expansion`` and ``specific_heat_slope``.
        viscosity_model : str, optional
            ``"walther"`` (default) or ``"exponential"``.
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
            nu = (
                10.0 ** (10.0 ** (self.walther_a - self.walther_b * np.log10(t)))
                - _WALTHER_SHIFT
            )
            mu = nu * 1e-6 * self.density_at(t)
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
