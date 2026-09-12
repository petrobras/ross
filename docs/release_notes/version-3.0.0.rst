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
  (``scipy.interpolate.RegularGridInterpolator``, linear, extrapolating).

Each coefficient is wrapped in the new ``BearingCoefficient`` class (``brg.kxx_interpolated``),
whose ``kind`` attribute tells which axes it carries and which is evaluated as
``brg.kxx_interpolated(frequency, speed)``. The element matrices ``K``, ``C`` and ``M`` and the
rotor matrices ``Rotor.K``, ``Rotor.C``, ``Rotor.M`` and ``Rotor.A`` take an optional ``speed``
alongside ``frequency``; with ``speed=None`` the synchronous diagonal (``speed == frequency``) is
evaluated, which reproduces the previous numerics exactly. ``plot()`` and ``format_table()`` follow
the tabulated axes (one curve per speed against the frequency axis for 2-D tables), and 2-D tables
round-trip through ``save()`` / ``load()``.

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
excitation-frequency tables, which give identical results in every synchronous analysis. Re-save
the rotor (or edit the key to ``speed``) before running analyses that decouple the excitation
frequency from the speed.

Other behavior changes of the coefficient rework:

- Coefficient attributes (``brg.kxx``, ...) are always plain (nested) lists, the canonical form
  the table is validated and serialized in; wrap them in ``np.array`` for array arithmetic.
- ``BearingElement`` gained the positional parameter ``speed`` right before ``frequency``; callers
  passing ``frequency`` positionally must switch to the keyword.
- The dimension error message reads ``Arguments (coefficients, speed and frequency) must have the
  same dimension``.
- ``SealElement`` persists ``seal_leakage`` on ``save()`` / ``load()``.
- Constant coefficients are returned exactly instead of through a two-point interpolator (which
  added round-off of the order of 1e-13 away from zero speed).
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

- ``sommerfeld_type`` and ``method`` are deprecated and ignored — the solver reports the
  Sommerfeld number directly and has a single perturbation route for the coefficients.
- ``groove_factor`` is deprecated and ignored — groove mixing is modeled through the solver's
  hot-oil carryover factor (``hot_oil_lambda``).
- ``geometry="lobe"`` / ``"elliptical"`` are deprecated — use ``MultiLobeBearing`` /
  ``EllipticalBearing``.
- ``elements_circumferential`` / ``elements_axial`` are now optional (the solver mesh is used
  when omitted).
- A plain-number ``reference_temperature`` is still interpreted as degC (with a warning) — pass a
  pint quantity to be explicit; likewise a plain-number ``pad_arc_length`` is interpreted as
  degrees.
- ``operating_type`` names map onto the engine vocabulary (``"flooded"`` →
  ``"regular_flooded"``, ``"starvation"`` → ``"starved_condition_even"``); the engine names are
  also accepted.

``TiltingPad`` parameter changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``equilibrium_type="determine_eccentricity"`` is now ``"match_load"``;
  ``"match_eccentricity"`` keeps its meaning (the journal is held at the prescribed position).
- The solver-iteration knobs ``solver_options``, ``initial_pads_angles``,
  ``inlet_temperature_tolerance``, ``max_inlet_iterations``, ``max_jtemp_iter``, ``jtemp_error``,
  ``max_relax_change`` and ``h_sump`` are deprecated and ignored — the solver owns its iteration
  strategy and convergence tolerances.
- New capabilities through keyword arguments: pivot flexibility (``deform_type``,
  ``pivot_type``, ``pivot_stiffness``), leading-edge-groove and spray-bar lubrication
  (``bearing_type``), starved and high-ambient-pressure operation (``operating_type``), and
  parallel solution of the frequency table (``num_processes``).

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
