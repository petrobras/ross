.. This file absorbs the unreleased 2.4.0 draft notes (version-2.4.0.rst):
   v2.4.0 was never tagged, so the fluid-film work ships with 3.0.0.
   TODO confirm: remove version-2.4.0.rst (and its include in release_notes.rst)
   if v2.4.0 will indeed not be released separately.

Version 3.0.0
-------------

The following enhancements and bug fixes were implemented for this release:

Enhancements
~~~~~~~~~~~~

Decoupled Rotor Speed and Excitation Frequency in Coefficient Tables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bearing and seal coefficient tables now declare the physical axis they are tabulated on, and the
rotor assembly evaluates them at a ``(frequency, speed)`` pair instead of a single value
(`#1321 <https://github.com/petrobras/ross/issues/1321>`_):

- ``BearingElement(..., speed=[...])`` — a 1-D table over the **rotor speed** (the base flow).
  This is what every fluid-film bearing and seal produces; it is interpolated at the rotor speed
  and is constant with respect to the excitation (whirl) frequency.
- ``BearingElement(..., frequency=[...])`` — a 1-D table over the **excitation (whirl)
  frequency**, kept for the elements that genuinely react to the vibration frequency
  (``MagneticBearingElement``, ``SqueezeFilmDamper``).
- ``BearingElement(..., speed=[...], frequency=[...])`` with 2-D coefficient arrays of shape
  ``(len(speed), len(frequency))`` — a table interpolated on both axes
  (interpolated along the speed axis, then along the frequency axis).

Each coefficient is wrapped in the new ``BearingCoefficient`` class (``brg.kxx_interpolated``),
whose ``kind`` attribute tells which axes it carries and which is evaluated as
``brg.kxx_interpolated(frequency, speed)``. The element matrices ``K``, ``C`` and ``M`` and the
rotor matrices ``Rotor.K``, ``Rotor.C``, ``Rotor.M`` and ``Rotor.A`` take an optional ``speed``
alongside ``frequency``; with ``speed=None`` the synchronous diagonal (``speed == frequency``) is
evaluated, which reproduces the previous numerics exactly. ``plot()`` and ``format_table()`` follow
the tabulated axes (one curve per speed against the frequency axis for 2-D tables), and 2-D tables
round-trip through ``save()`` / ``load()``.

All tables (1-D and 2-D) are interpolated along each axis with a shape-preserving piecewise cubic
Hermite polynomial (PCHIP), which passes through the tabulated values without overshooting between
them; two points give linear interpolation and one point a constant. Outside an axis the
coefficients are extrapolated linearly from the end slope. ``interpolation="linear"`` on any element
selects piecewise-linear interpolation instead. Coefficient tables are smooth and mostly monotonic,
so at least ``MIN_RECOMMENDED_AXIS_POINTS`` (5) points per axis spanning the analysis range give
reliable interpolation: ``run_campbell``, ``run_freq_response`` / forced responses,
``run_modal(frequency=...)`` and ``matched_whirl`` now warn, naming the element and the axis, when
they interpolate a 2- to 4-point table or leave an axis (the previous extrapolation warning was
anonymous and limited to the speed sweeps).

Frequency-Dependent Seal and Bearing Coefficients from the Solvers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The flow solvers now separate the rotor speed, which sets the base flow, from the whirl
(excitation) frequency of the perturbation, and can fill a 2-D table directly
(`#1321 <https://github.com/petrobras/ross/issues/1321>`_):

- ``LabyrinthSeal(speed=..., frequency=...)`` — ``LabyrinthSolver.solve(speed, frequency=None)``
  solves the leakage, cavity pressures and swirl for the speed and the perturbation system for the
  whirl frequency; ``solve_row`` reuses one base flow for several whirl frequencies and
  ``solve_grid`` maps it over the speeds. The synchronous diagonal of the grid reproduces the 1-D
  table exactly.
- ``HolePatternSeal(speed=..., frequency=...)`` — same split for the bulk-flow solver; the grid
  points at ``excitation_ratio * speed`` reproduce the 1-D solve, and ``excitation_ratio`` stays as
  the 1-D convenience.
- ``HybridSeal(speed=..., frequency=...)`` — the interface pressure is matched with the synchronous
  stages, which are then rebuilt at the converged pressure with 2-D tables.
- ``FluidFilmBearing(speed=..., frequency=...)`` (and the configuration classes) — one engine case
  per (speed, frequency) pair with whirl ratio ``frequency / speed`` (nonzero speeds required);
  ``coefficients(speed, frequency=None)`` reads the table on both axes.

Modal and Forced Response with Speed Decoupled from the Excitation Frequency
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``run_modal(speed, frequency=f)`` evaluates the frequency-dependent coefficients at a fixed
  whirl frequency while the gyroscopic effect keeps the rotor speed.
- ``run_modal(speed, matched_whirl=True)`` iterates each mode with a fixed point on its damped
  natural frequency (the mode is tracked between iterates by the modal assurance criterion;
  ``whirl_rtol`` and ``whirl_max_iter`` control the iteration and a warning reports a mode that
  does not converge). This is the relevant analysis for subsynchronous stability, where a mode
  whirls well below the running speed and the synchronous coefficients misestimate its damping.
- ``ModalResults.whirl_frequency`` stores the whirl frequency each mode's coefficients were
  evaluated at (the rotor speed for the default synchronous analysis).
- ``run_campbell(..., matched_whirl=True)`` runs the matched-whirl analysis at every speed and
  keeps ``whirl_frequency`` aligned with the tracked modes.
- ``run_freq_response(speed_range, speed=w)`` and ``run_forced_response(force, speed_range,
  speed=w)`` hold the rotor speed fixed and sweep ``speed_range`` as the excitation frequency; the
  default remains the synchronous sweep.
- The pre-existing ``synchronous=True`` flag of ``run_modal`` / ``run_ucs`` is unrelated: it
  selects Rouch's formulation, which folds the gyroscopic matrix into the mass matrix.

Fluid-Film TEHD Engine for Journal Bearings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The hydrodynamic journal-bearing classes were rebuilt on a shared thermo-elasto-hydro-dynamic (TEHD)
solver (the internal ``ross.bearings.fluid_film`` package). The engine solves the finite-element
Reynolds equation coupled with the film energy equation and, optionally, pad heat conduction,
pad/pivot elastic deformation and shell thermal growth. It models turbulence (Reichardt eddy
viscosity with a laminar-turbulent transition band), Swift–Stieber cavitation, groove hot-oil
carryover mixing, and Hertzian pivot-contact flexibility (ball-in-socket, button and rocker-back
pivots) — capabilities that were not available in the previous solvers
(`#1332 <https://github.com/petrobras/ross/pull/1332>`_).

The public surface is a new family of ``BearingElement`` subclasses:

- ``FluidFilmBearing`` — shared base class: per-pad geometry arrays, the model-flag surface
  (``thermal_type``, ``operating_type``, ``equilibrium_type``, ``pivot_type``, ``deform_type``, ...),
  serial or parallel (``num_processes``) solution of the ``frequency`` table,
  ``coefficients(frequency)`` interpolation, and field plots through ``FluidFilmBearingResults``.
- ``FixedGeometryBearing`` — generic fixed-geometry bearing described by per-pad arrays
  (pivot angle, arc, preload, offset, pockets and tapers).
- ``PartialArcBearing``, ``EllipticalBearing``, ``OffsetHalvesBearing``, ``MultiLobeBearing``,
  ``PressureDamBearing`` — the classic fixed-geometry configurations.
- ``PlainJournal`` and ``TiltingPad`` — rewritten on the engine, keeping the historical parameter
  surface where meaningful (see the migration guide below).

The engine is validated against published results — Lund & Thomsen (1978) two-axial-groove
coefficients, Nicholas, Barrett & Leader (1980) pressure-dam step bearings, and the Fillon et al.
(1992) tilting-pad TEHD benchmark — in ``ross/tests/test_fluid_film_literature.py``, plus a pinned
regression suite in ``ross/tests/test_fluid_film_solver.py``.

The post-processing layer gained ``plot_pad_temperature_3d()``, which draws the pads as real
geometry colored by the solid pad conduction field (``thermal_type="full"`` only), the through-pad
counterpart of ``plot_film_temperature_3d()`` (`#1339 <https://github.com/petrobras/ross/pull/1339>`_).

Gear Mesh Backlash Model for Multi-Rotor Systems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The multi-rotor code was reorganized into the ``ross.multi_rotor`` package (``gear_element``,
``mesh``, ``multi_rotor``, ``results`` and ``utils`` modules) and the gear pair coupling gained a
backlash model. The new ``Mesh`` class concentrates the meshing behavior (stiffness, contact ratio
and, optionally, backlash), and ``MultiRotor`` accepts a ``backlash`` dictionary
(``enable``, ``initial_value``, ``error_amp``, ``smooth_operator``, ...) to activate it.
With backlash enabled, ``MultiRotor.run_time_response()`` uses a numba-accelerated Newmark
integration and returns ``BacklashResults``, which adds the time evolution of the mesh dynamics —
``plot_transmission_error()``, ``plot_backlash()``, ``plot_mesh_force()``, ``plot_mesh_stiffness()``,
``plot_center_distance()``, ``plot_pressure_angle()``, ``plot_contact_ratio()`` and a combined
``plot_dashboard()`` (`#1325 <https://github.com/petrobras/ross/pull/1325>`_).

Global Proportional and Modal Damping
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``Rotor`` now accepts damping to be prescribed at the model level, in addition to the per-element
``alpha`` / ``beta`` factors kept on ``ShaftElement``:

- ``alpha`` and ``beta`` — global proportional (Rayleigh) damping factors applied to the assembled
  mass and stiffness matrices (`#1307 <https://github.com/petrobras/ross/pull/1307>`_).
- ``modal_damping_ratio`` and ``default_damping_ratio`` — modal damping ratios for the first modes
  and a default ratio for the remaining ones (`#1231 <https://github.com/petrobras/ross/pull/1231>`_).

AMB Closed-Loop Time Response with Modal Reduction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The time response of rotors supported by active magnetic bearings is now computed by a dedicated
closed-loop model (``ross/bearings/magnetic/amb_time_response.py``) that converts the rotor to the
modal domain and integrates the plant together with the controller transfer functions and the
linearized magnetic-force relation. ``Rotor.run_time_response()`` detects AMBs on the rotor and
returns ``AmbTimeResponseResults``, which extends ``TimeResponseResults`` with the AMB states and
the plots ``plot_amb_disps()``, ``plot_amb_currents()`` and ``plot_amb_forces()``. The magnetic
modules were reorganized (``amb_controllers``, ``amb_models``, ``amb_time_response``, ``amb_utils``)
and the example rotors are now ``rotor_example_amb_simple()``,
``rotor_example_amb_general_controllers()`` and ``rotor_example_amb_complex_controllers()``
(`#1320 <https://github.com/petrobras/ross/pull/1320>`_).

Real-Gas Model for ``LabyrinthSeal``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``LabyrinthSeal`` accepts ``gas_model="real"`` to evaluate gas properties along an isentrope from a
thermodynamic table built once at construction (requires ``gas_composition``), instead of the
ideal-gas relations used by the default ``gas_model="ideal"``
(`#1317 <https://github.com/petrobras/ross/pull/1317>`_).

API 617 Clearance Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^

``Rotor.run_clearance_analysis`` now follows API 617 (9th edition) 6.8.2.10 and 6.8.2.11:

- the unbalance is placed and sized by the new ``Rotor.api617_unbalance(mode, maximum_continuous_speed)``,
  which reads the antinodes of the selected forward mode shape and applies :math:`U_a = 2 U_r` with the
  journal static loads (single antinode between the bearings), the nearest journal load (conical modes,
  180° out of phase) or the overhung mass (overhung and coupling modes). An explicit ``node`` /
  ``unbalance_magnitude`` / ``unbalance_phase`` can still be given;
- the mechanical test vibration limit is :math:`A_{vl} = \min(25.4, 25.4 \sqrt{12000 / N_{mc}})` µm
  peak to peak (the previous implementation omitted the square root and the 25.4 µm cap);
- :math:`A_{max}` is the largest peak-to-peak amplitude at the machine vibration ``probes`` between
  ``minimum_allowable_speed`` and ``maximum_continuous_speed``, instead of the x-direction amplitude at the
  bearing nodes at a single speed;
- the scale factor :math:`S_{cc} = A_{vl} / A_{max}` is only capped when ``scale_factor_cap`` is given
  (API 617 uses 6; the default applies no cap);
- the scaled **major-axis** peak-to-peak amplitude at every close-clearance location is compared with
  75 % of the minimum **diametral** clearance over the whole ``speed_range`` (previously the x-direction
  amplitude was compared with the radial clearance at one speed).

Close-clearance locations are read from the rotor: every bearing or seal element with a
``radial_clearance``. ``BearingElement`` and ``SealElement`` accept ``radial_clearance=`` so that
coefficient-table elements (including tables written by ``save_coefficient_table``, which now keeps the
clearance) take part in the check.

``ClearanceResults`` stores the probe response, :math:`A_{vl}`, :math:`A_{max}`, :math:`S_{cc}`, the
unbalance used and the scaled response at each location over the speed range, with ``data()`` for a
summary table and ``plot()``, ``plot_response()`` and ``plot_probe_response()`` for the plots
(`#1285 <https://github.com/petrobras/ross/pull/1285>`_ follow-up).

Rotor Composition Helpers
^^^^^^^^^^^^^^^^^^^^^^^^^

Rotors can now be composed and extended after creation:

- ``Rotor.concatenate_rotors(rotor_list)`` (classmethod) and the ``+`` operator (``rotor1 + rotor2``)
  join rotors in series (`#1291 <https://github.com/petrobras/ross/pull/1291>`_).
- ``Rotor.add_elements(new_elements)`` returns a new rotor with extra disk, bearing or seal
  elements attached to an existing model (`#1309 <https://github.com/petrobras/ross/pull/1309>`_).

Units in the Rotor Summary
^^^^^^^^^^^^^^^^^^^^^^^^^^

``SummaryResults.plot()`` accepts ``length_units``, ``mass_units`` and ``force_units``, so the
summary table produced by ``rotor.summary()`` can be displayed in the user's preferred units
(`#1305 <https://github.com/petrobras/ross/pull/1305>`_).

``plot_rotor`` Redesign
^^^^^^^^^^^^^^^^^^^^^^^

The rotor figure was redesigned — cleaner element styling and legend layout — and the plot is now
responsive to the container width (`#1331 <https://github.com/petrobras/ross/pull/1331>`_,
`#1341 <https://github.com/petrobras/ross/pull/1341>`_).

Spline Drawing of Response Curves
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Every plot method with a ``line_shape`` argument now draws a Plotly spline through the sampled points
by default (``line_shape="spline"``) instead of straight segments: ``FrequencyResponseResults.plot_magnitude``,
``ForcedResponseResults.plot_magnitude``, their stochastic counterparts, and
``ClearanceResults.plot_response`` / ``plot_probe_response``, which gained the argument. The sampled
values are not changed; pass ``line_shape="linear"`` to recover the previous drawing.

Documentation
^^^^^^^^^^^^^

- New **Theory & Implementation** section covering the formulation behind the code
  (`#1318 <https://github.com/petrobras/ross/pull/1318>`_).
- New **Validation** page comparing ROSS results with published cases, including the RAPPID
  lumped-mass demonstration rebuilt with ordinary ROSS elements
  (`#1337 <https://github.com/petrobras/ross/pull/1337>`_,
  `#1338 <https://github.com/petrobras/ross/pull/1338>`_).
- The bearings tutorial (``tutorial_bearings_part_2``) was regenerated on the new fluid-film
  engine, the advanced-bearings cookbook recipe was rewritten, and the notebooks based on the
  removed ``fluid_flow`` subpackage were removed (`#1332 <https://github.com/petrobras/ross/pull/1332>`_).
- MultiRotor tutorial reworked with the gear TVMS example
  (`#1293 <https://github.com/petrobras/ross/pull/1293>`_).
- API reference completed with missing objects and new unit tests for the plotting methods
  (`#1308 <https://github.com/petrobras/ross/pull/1308>`_,
  `#1311 <https://github.com/petrobras/ross/pull/1311>`_).
- Sicchieri (2024) thesis added to the ``TiltingPad`` references
  (`#1333 <https://github.com/petrobras/ross/pull/1333>`_).

API Changes and Migration Guide
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Supported Python and Dependency Versions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ROSS now follows `SPEC 0 <https://scientific-python.org/specs/spec-0000/>`_, the
time-based support policy of the scientific Python ecosystem: a Python version is
supported for three years after its initial release and a core dependency version
for two years. Version 3.0.0 requires Python 3.12 or newer and is tested on
Python 3.12, 3.13 and 3.14; support for Python 3.9, 3.10 and 3.11 is dropped.
The minimum dependency versions are ``numpy>=2.2``, ``scipy>=1.15`` and
``pandas>=2.3``. Minimum versions are raised only on major and minor releases,
never on patch releases; see the contributing guide for the policy.

Upgrading with ``ross_2to3``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Version 3.0.0 renames constructor parameters across ``ross.bearings`` and ``ross.seals`` and
changes a few conventions (diameters instead of radii, radians and kelvin instead of the old
plain-number degrees and degrees Celsius) without deprecation shims. The ``ross_2to3`` command
installed with ROSS converts existing assets (`#1358 <https://github.com/petrobras/ross/issues/1358>`_):

.. code-block:: bash

   ross_2to3 my_rotor.toml analysis.py notebooks/   # preview: diff plus a report
   ross_2to3 -w my_rotor.toml analysis.py           # rewrite in place, keeping .bak copies
   ross_2to3 -o converted/ project/                 # write the converted files elsewhere

- **Rotor and element files** (``.toml`` / ``.json`` written by ``save()``): the renamed keys and
  their values are converted, ``frequency`` tables become ``speed`` tables, and every solver-based
  bearing section (``PlainJournal``, ``TiltingPad``, ``ThrustPad``, ``SqueezeFilmDamper``) is
  written as the ``BearingElement`` coefficient table ROSS 3 itself saves, so the file loads
  without re-running a solver. Seal sections keep their class with the renamed parameters. Each
  converted file is loaded with the installed ROSS to prove it is valid.
- **Scripts and notebooks** (``.py`` / ``.ipynb``): the keyword arguments of the constructors
  below, of ``run_unbalance_response`` / ``run_ucs`` and the moved module paths are rewritten in
  place, preserving formatting. Literal values whose convention changed are converted
  (``journal_radius=0.2`` becomes ``journal_diameter=0.4``; ``pad_arc_length=176`` becomes
  ``pad_arc=Q_(176, "deg")``; ``reference_temperature=50`` becomes
  ``oil_supply_temperature=Q_(50, "degC")``; ``iopt1=1`` becomes ``use_jenny_kanki=True``;
  ``load=[fx, fy]`` becomes ``fxs_load=fx, fys_load=fy``), removed parameters are dropped, and
  anything that cannot be rewritten safely — positional arguments, ``**kwargs``, non-literal values
  in changed units, uses of removed classes — is listed in the report with its line number.

The rename map lives in ``ross.ross_2to3.renames`` and is rendered below.

Bearing and seal parameter names
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Geometry is entered as diameters (what drawings and vendor sheets specify), angles in radians
(or any pint angle), temperatures in kelvin (or any pint temperature); clearances stay
``radial_clearance``. Seals keep ``inlet_``/``outlet_pressure`` and ``inlet_temperature`` from the
gas-seal literature, while bearings use ``oil_supply_temperature`` / ``oil_supply_pressure``
from the lubrication literature. ``HolePatternSeal.excitation_ratio`` replaces ``whirl_ratio``
to avoid a clash with the whirl-frequency-ratio stability output (Lund's WFR), which keeps its
name. ``HolePatternSeal.nz`` and ``ThrustPad.n_theta`` / ``n_radial`` are unchanged: they
describe a 1-D bulk-flow grid and a polar thrust-face grid, not the fluid-film mesh family
(`#1359 <https://github.com/petrobras/ross/pull/1359>`_,
`#1360 <https://github.com/petrobras/ross/pull/1360>`_).

.. list-table::
   :header-rows: 1
   :widths: 22 28 50

   * - Class
     - ROSS 2
     - ROSS 3
   * - ``BearingElement``
     - ``frequency``
     - ``speed``
   * - ``SealElement``
     - ``frequency``
     - ``speed``
   * - ``ST_BearingElement``
     - ``frequency``
     - ``speed``
   * - ``FluidFilmBearing``
     - ``frequency``
     - ``speed``
   * - ``PlainJournal``
     - ``frequency``
     - ``speed``
   * - ``PlainJournal``
     - ``axial_length``
     - ``pad_axial_length``
   * - ``PlainJournal``
     - ``journal_radius``
     - ``journal_diameter`` (value ×2, radius to diameter)
   * - ``PlainJournal``
     - ``n_pad``
     - ``n_pads``
   * - ``PlainJournal``
     - ``pad_arc_length``
     - ``pad_arc`` (plain numbers were degrees, now radians)
   * - ``PlainJournal``
     - ``reference_temperature``
     - ``oil_supply_temperature`` (plain numbers were degrees Celsius, now kelvin)
   * - ``PlainJournal``
     - ``initial_guess``
     - ``initial_position`` (now the (x, y) journal position as fractions of the radial clearance)
   * - ``PlainJournal``
     - ``elements_circumferential``
     - ``total_ex_film``
   * - ``PlainJournal``
     - ``elements_axial``
     - ``total_ez_film``
   * - ``PlainJournal``
     - ``operating_type``
     - same name; ``flooded`` becomes ``regular_flooded``, ``starvation`` becomes ``starved_condition_even``
   * - ``PlainJournal``
     - ``geometry``
     - removed: use MultiLobeBearing or EllipticalBearing for non-circular bores
   * - ``PlainJournal``
     - ``model_type``
     - removed: the engine is always thermo-hydro-dynamic
   * - ``PlainJournal``
     - ``sommerfeld_type``
     - removed: the Sommerfeld number is reported directly
   * - ``PlainJournal``
     - ``method``
     - removed: the coefficients come from a single perturbation route
   * - ``PlainJournal``
     - ``groove_factor``
     - removed: groove mixing is set by hot_oil_lambda
   * - ``TiltingPad``
     - ``frequency``
     - ``speed``
   * - ``TiltingPad``
     - ``pre_load``
     - ``preload``
   * - ``TiltingPad``
     - ``nx``
     - ``total_ex_film`` (must be even)
   * - ``TiltingPad``
     - ``nz``
     - ``total_ez_film``
   * - ``TiltingPad``
     - ``nr_pad``
     - ``total_ey_pad``
   * - ``TiltingPad``
     - ``load``
     - ``fxs_load``, ``fys_load`` (the [fx, fy] pair is split into two arguments)
   * - ``TiltingPad``
     - ``hot_oil_carry_over``
     - ``hot_oil_lambda``
   * - ``TiltingPad``
     - ``k_pad``
     - ``pad_conductivity``
   * - ``TiltingPad``
     - ``h_edge``
     - ``edges_convection``
   * - ``TiltingPad``
     - ``relax_t``
     - ``relax_temperature``
   * - ``TiltingPad``
     - ``journal_temperature``
     - ``journal_temperature`` (plain numbers were degrees Celsius, now kelvin)
   * - ``TiltingPad``
     - ``equilibrium_type``
     - same name; ``determine_eccentricity`` becomes ``match_load``
   * - ``TiltingPad``
     - ``initial_pads_angles``
     - removed: the solver owns its iteration strategy and tolerances
   * - ``TiltingPad``
     - ``solver_options``
     - removed: the solver owns its iteration strategy and tolerances
   * - ``TiltingPad``
     - ``inlet_temperature_tolerance``
     - removed: the solver owns its iteration strategy and tolerances
   * - ``TiltingPad``
     - ``max_inlet_iterations``
     - removed: the solver owns its iteration strategy and tolerances
   * - ``TiltingPad``
     - ``h_sump``
     - removed: the solver owns its iteration strategy and tolerances
   * - ``TiltingPad``
     - ``max_jtemp_iter``
     - removed: the solver owns its iteration strategy and tolerances
   * - ``TiltingPad``
     - ``jtemp_error``
     - removed: the solver owns its iteration strategy and tolerances
   * - ``TiltingPad``
     - ``max_relax_change``
     - removed: the solver owns its iteration strategy and tolerances
   * - ``ThrustPad``
     - ``frequency``
     - ``speed``
   * - ``ThrustPad``
     - ``n_pad``
     - ``n_pads``
   * - ``ThrustPad``
     - ``pad_arc_length``
     - ``pad_arc``
   * - ``ThrustPad``
     - ``angular_pivot_position``
     - ``pivot_angle``
   * - ``SqueezeFilmDamper``
     - ``journal_radius``
     - ``journal_diameter`` (value ×2, radius to diameter)
   * - ``LabyrinthSeal``
     - ``frequency``
     - ``speed``
   * - ``LabyrinthSeal``
     - ``shaft_radius``
     - ``shaft_diameter`` (value ×2, radius to diameter)
   * - ``LabyrinthSeal``
     - ``molar``
     - ``molar_mass``
   * - ``LabyrinthSeal``
     - ``tz``
     - ``reference_temperatures``
   * - ``LabyrinthSeal``
     - ``muz``
     - ``reference_viscosities``
   * - ``LabyrinthSeal``
     - ``iopt1``
     - ``use_jenny_kanki`` (0/1 becomes False/True)
   * - ``LabyrinthSeal``
     - ``nprt``
     - removed: printing is controlled by print_results
   * - ``LabyrinthSeal``
     - ``analz``
     - removed: leakage and dynamic coefficients are always computed
   * - ``HolePatternSeal``
     - ``frequency``
     - ``speed``
   * - ``HolePatternSeal``
     - ``shaft_radius``
     - ``shaft_diameter`` (value ×2, radius to diameter)
   * - ``HolePatternSeal``
     - ``molar``
     - ``molar_mass``
   * - ``HolePatternSeal``
     - ``length``
     - ``axial_length``
   * - ``HolePatternSeal``
     - ``roughness``
     - ``relative_roughness``
   * - ``HolePatternSeal``
     - ``whirl_ratio``
     - ``excitation_ratio``
   * - ``HolePatternSeal``
     - ``entr_coef``
     - ``entrance_loss_coefficient``
   * - ``HolePatternSeal``
     - ``exit_coef``
     - ``exit_loss_coefficient``
   * - ``HolePatternSeal``
     - ``rlx_factor``
     - ``relaxation_factor``
   * - ``HolePatternSeal``
     - ``b_suther``
     - ``sutherland_b``
   * - ``HolePatternSeal``
     - ``s_suther``
     - ``sutherland_s``
   * - ``HybridSeal``
     - ``frequency``
     - ``speed``
   * - ``HybridSeal``
     - ``shaft_radius``
     - ``shaft_diameter`` (value ×2, radius to diameter)
   * - ``HybridSeal``
     - ``molar``
     - ``molar_mass``
   * - ``MultiRotor``
     - ``square_stiffness_amplitude_ratio``
     - removed: the mesh stiffness is described by the Mesh class
   * - ``Mesh``
     - ``square_stiffness_amplitude_ratio``
     - removed: the mesh stiffness is described by the Mesh class

``HybridSeal`` takes the ``HolePatternSeal`` / ``LabyrinthSeal`` names inside its
``hole_pattern_parameters`` / ``labyrinth_parameters`` dictionaries. ``TiltingPad.n_link``,
``PlainJournal.operating_type`` and the mesh sizes are forwarded to ``FluidFilmBearing`` as
keyword arguments.

Coefficient table axis: ``frequency=`` becomes ``speed=``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In previous versions every coefficient table was passed as ``frequency=``, although for
fluid-film bearings and seals the values were rotor speeds. The keyword now declares the physical
axis, so speed-tabulated elements take ``speed=`` and only excitation-tabulated elements keep
``frequency=``. In synchronous analyses both kinds give identical results; the distinction only
matters when the excitation frequency is decoupled from the rotor speed.

===================================================  ==========================================================
Old                                                  New
===================================================  ==========================================================
``BearingElement(frequency=w)`` (speed table)        ``BearingElement(speed=w)``
``SealElement(frequency=w)``                         ``SealElement(speed=w)``
``ST_BearingElement(frequency=w)``                   ``ST_BearingElement(speed=w)``
``CylindricalBearing(speed=w)``                      unchanged
``FluidFilmBearing`` / ``PlainJournal`` /            ``speed=w``
``TiltingPad`` / fixed-geometry classes
``(frequency=w)``
``ThrustPad(frequency=w)``                           ``ThrustPad(speed=w)``
``LabyrinthSeal`` / ``HolePatternSeal`` /            ``speed=w``
``HybridSeal(frequency=w)``
``MagneticBearingElement(frequency=w)``              unchanged (excitation-frequency table)
``SqueezeFilmDamper(frequency=w)``                   unchanged (excitation-frequency table)
``BearingElement.from_table`` column ``frequency``   column ``speed`` (``frequency`` still accepted)
``BearingElement.format_table(frequency=...)``       ``format_table(speed=..., frequency=...)``
``brg.frequency`` (speed table)                      ``brg.speed``
``FluidFilmBearing.coefficients(frequency)``         ``coefficients(speed, frequency=None)``
``run_unbalance_response(frequency=w)``              ``run_unbalance_response(speed_range=w)``
``run_ucs(bearing_frequency_range=...)``             ``run_ucs(bearing_speed_range=...)``
``UCSResults.bearing_frequency_range``               ``UCSResults.bearing_speed_range``
TOML / JSON key ``frequency`` of saved elements      key ``speed`` (files with ``frequency`` still load)
===================================================  ==========================================================

Saved rotor files written by version 2 keep loading: their ``frequency`` tables become
excitation-frequency tables, which give identical results in every synchronous analysis. Run
``ross_2to3`` on the file (or re-save the rotor) before running analyses that decouple the
excitation frequency from the speed.

Other behavior changes of the coefficient rework:

- Coefficient attributes (``brg.kxx``, ...) are always plain (nested) lists, the canonical form
  the table is validated and serialized in; wrap them in ``np.array`` for array arithmetic.
- ``BearingElement`` gained the positional parameter ``speed`` right before ``frequency``; callers
  passing ``frequency`` positionally must switch to the keyword.
- The dimension error message reads ``Arguments (coefficients, speed and frequency) must have the
  same dimension``.
- ``SealElement`` persists ``seal_leakage`` on ``save()`` / ``load()``.
- ``SqueezeFilmDamper.save()`` and ``ThrustPad.save()`` write the solved coefficient table as a
  ``BearingElement`` section, as ``FluidFilmBearing`` does (``BearingElement.save_coefficient_table``);
  both classes could not load the files they wrote before. ``BearingElement.load`` builds the element
  with the class named in the file, so ``SqueezeFilmDamper.load(file)`` returns that table.
- Constant coefficients are returned exactly instead of through a two-point interpolator (which
  added round-off of the order of 1e-13 away from zero speed).
- 1-D tables were interpolated with a smoothing spline (``scipy.interpolate.UnivariateSpline`` with
  its default smoothing factor), which did not pass through the tabulated values: the deviation
  was negligible for large coefficients but reached tens of percent for small ones (e.g. damping of
  the order of 10 N*s/m). Tables are now interpolated exactly (PCHIP), so off-grid values change by
  about 1e-5 relative for typical tables and by more where the old smoothing was wrong, and
  extrapolation is linear from the end slope instead of the cubic tail.
- ``transfer_matrix`` (hence ``run_freq_response`` and the forced responses) evaluates
  frequency-tabulated elements (``MagneticBearingElement``, ``SqueezeFilmDamper``) at the
  excitation frequency instead of the rotor speed, including the ``free_free`` branch, which
  used to evaluate every coefficient at zero speed. Speed-tabulated elements are unchanged.
- ``MultiRotor`` scales the driven rotor speed by the gear ratio while the excitation frequency is
  global to the coupled system; only frequency-tabulated elements on the driven shaft see a
  different lookup.
- ``BearingElement.from_table`` and ``table_to_toml`` return the table axis as ``speed``.
- Time integration keeps evaluating the coefficients at the instantaneous speed (synchronous
  lookup); the whirl content of a transient is not resolved per frequency.

Coefficients change with this release
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. warning::

   Dynamic coefficients computed by ``PlainJournal`` and ``TiltingPad`` differ from version 2.3:
   the new engine uses a different turbulence model, cavitation treatment and energy equation, and
   its results are anchored to the literature cases above rather than to the previous regression
   values. Expect differences of a few percent in typical laminar cases and larger differences near
   the laminar-turbulent transition, under heavy loads, or whenever the old adiabatic model
   underpredicted film temperatures.

Removed classes, modules and parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

===================================================  ==========================================================
Removed                                              Replacement
===================================================  ==========================================================
``ross.bearings.fluid_flow`` (subpackage)            ``PlainJournal`` (TEHD) or ``CylindricalBearing`` (analytical)
``BearingFluidFlow``                                 ``PlainJournal`` or ``CylindricalBearing``
``ST_BearingElement.from_fluid_flow``                build coefficient arrays with ``PlainJournal`` and pass them to ``ST_BearingElement``
``PlainJournalResults`` / ``TiltingPadResults``      ``FluidFilmBearingResults`` (created automatically by the bearing)
``rotor_amb_example(...)``                           ``rotor_example_amb_general_controllers(...)`` (same ``controller_transfer_function`` argument); see also ``rotor_example_amb_simple()`` and ``rotor_example_amb_complex_controllers()``
``LabyrinthSeal(analz=...)``                         removed — leakage and dynamic coefficients are always computed
===================================================  ==========================================================

Moved modules
^^^^^^^^^^^^^

Top-level imports (``rs.GearElement``, ``rs.MultiRotor``, ``rs.Mesh``, ...) are unchanged; only
imports from the old module paths need updating:

===================================================  ==========================================================
Old module                                           New module
===================================================  ==========================================================
``ross.gear_element``                                ``ross.multi_rotor.gear_element``
``ross.multi_rotor`` (module)                        ``ross.multi_rotor.multi_rotor`` (inside the new ``ross.multi_rotor`` package)
``ross.bearings.magnetic.controllers``               ``ross.bearings.magnetic.amb_controllers``
===================================================  ==========================================================

``PlainJournal`` parameter changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``sommerfeld_type``, ``method`` and ``model_type`` were removed — the solver reports the
  Sommerfeld number directly, has a single perturbation route for the coefficients and is always
  thermo-hydro-dynamic.
- ``groove_factor`` was removed — groove mixing is modeled through the solver's hot-oil carryover
  factor (``hot_oil_lambda``).
- ``geometry`` was removed — use ``MultiLobeBearing`` / ``EllipticalBearing`` for non-circular
  bores.
- ``elements_circumferential`` / ``elements_axial`` became the optional ``total_ex_film`` /
  ``total_ez_film`` mesh overrides (the solver mesh is used when omitted).
- Plain numbers are no longer interpreted as degrees (``pad_arc_length``) or degrees Celsius
  (``reference_temperature``): ``pad_arc`` takes radians and ``oil_supply_temperature`` kelvin,
  or a pint quantity in any unit.
- ``operating_type`` uses the engine vocabulary (``"flooded"`` → ``"regular_flooded"``,
  ``"starvation"`` → ``"starved_condition_even"``).

``TiltingPad`` parameter changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``equilibrium_type="determine_eccentricity"`` is now ``"match_load"``;
  ``"match_eccentricity"`` keeps its meaning (the journal is held at the prescribed position).
- ``load=[fx, fy]`` became ``fxs_load`` / ``fys_load``, and the solver-iteration knobs
  ``solver_options``, ``initial_pads_angles``, ``inlet_temperature_tolerance``,
  ``max_inlet_iterations``, ``max_jtemp_iter``, ``jtemp_error``, ``max_relax_change`` and
  ``h_sump`` were removed — the solver owns its iteration strategy and convergence tolerances.
- ``journal_temperature`` is in kelvin (it defaulted to 25 °C as a plain number).
- New capabilities through keyword arguments: pivot flexibility (``deform_type``,
  ``pivot_type``, ``pivot_stiffness``), leading-edge-groove and spray-bar lubrication
  (``bearing_type``), starved and high-ambient-pressure operation (``operating_type``), and
  parallel solution of the speed table (``num_processes``).

Post-processing methods
^^^^^^^^^^^^^^^^^^^^^^^

=================================================  ==========================================================
Old method                                         New method
=================================================  ==========================================================
``plot_pressure_distribution(...)``                ``plot_pressure_2d()`` / ``plot_pressure_3d()``
``plot_thermal_pad_results(freq_index, pad)``      ``plot_film_temperature_3d(freq_index, pad_index)``
``plot_temperature_3d(...)``                       ``plot_film_temperature_3d(...)`` (deprecated alias kept with a warning)
``plot_film_average_temperature()``                ``plot_temperature_2d()``
``_print_single_frequency_results(...)``           ``show_results()``
``show_optimization_convergence(...)``             removed — the solver owns its convergence strategy
``plot_bearing_representation()``                  removed
``plot_babbitt_surface_temperature()``             removed — with ``thermal_type="full"`` the film
                                                   temperature already reflects pad conduction
``plot_solid_pad_results(...)``                    ``plot_pad_temperature_3d()``
=================================================  ==========================================================

``plot_results()``, ``show_results()``, ``show_coefficients_comparison()`` and
``show_execution_time()`` keep their names, and ``plot_film_thickness_2d()`` and
``plot_pad_temperature_3d()`` are new.

``run_clearance_analysis`` signature
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The v2.3 signature ``run_clearance_analysis(speed, node, unbalance_magnitude, unbalance_phase,
frequency=None, modes=None)`` is replaced by
``run_clearance_analysis(speed_range, minimum_allowable_speed, maximum_continuous_speed, probes,
mode=0, node=None, unbalance_magnitude=None, unbalance_phase=None, scale_factor_cap=None)``.
The result object no longer offers the ``speed_rpm`` / ``bearing_nodes`` / ``magnitudes`` /
``clearance`` / ``clearance_75`` keys; use the attributes listed in ``ClearanceResults`` and
``results.data()``. Amplitudes are stored in metres peak to peak and clearances are diametral.

Bug Fixes
~~~~~~~~~

Fix ``Orbit`` Modal Analysis with numba >= 0.67 / numpy 2.5
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

numpy 2.5 changed ``np.linalg.eig`` to always return complex arrays, and numba 0.67 follows that
typing, which broke the compilation of the jitted ``_init_orbit`` used by modal analysis. Since the
matrix involved is symmetric, ``Orbit`` now uses ``np.linalg.eigh``, which returns real arrays on
every numpy version, and the major-axis eigenvector sign is normalized so the major axis angle
always falls in the upper half plane (`#1342 <https://github.com/petrobras/ross/pull/1342>`_).

.. note::

   Behavior change: because of the deterministic sign normalization, the major/minor axis
   *angles* — and hence the phase returned by ``Orbit.calculate_amplitude("major")`` — can differ
   by π from previous releases for many orbits. All physical amplitudes (``major_axis``,
   ``minor_axis``, ``kappa``) are unchanged to machine precision.

Fix ``LabyrinthSeal`` Coefficients on Repeated Runs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``LabyrinthSeal`` produced incorrect coefficients when computing multiple frequencies with the same
instance, because the perturbation arrays were not cleared between frequency runs
(`#1313 <https://github.com/petrobras/ross/pull/1313>`_).

Fix ``seal_leakage`` Units and ``HybridSeal`` Mass-Flow Balance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``LabyrinthSeal`` now reports ``seal_leakage`` in kg/s, and the ``HybridSeal`` mass-flow balance
between the hole-pattern and labyrinth sections was corrected
(`#1329 <https://github.com/petrobras/ross/pull/1329>`_).

General Fixes
^^^^^^^^^^^^^

- Fixed ``Probe`` and ``np.matrix`` deprecation warnings raised during the test suite, a highly
  fragmented ``DataFrame`` issue, and element tag handling; ``concatenate_rotors`` was fixed and
  promoted to a classmethod (`#1315 <https://github.com/petrobras/ross/pull/1315>`_).
- ``run_ucs()`` now applies the ``@check_units`` decorator to its arguments
  (`#1308 <https://github.com/petrobras/ross/pull/1308>`_).

Contributors
~~~~~~~~~~~~

This release includes contributions from: @jguarato, @Raimundovpn, @murilloabs, @ArthurIasbeck,
@gsabinoo, @ViniciusTxc3, @kiracofe8, @raphaeltimbo
