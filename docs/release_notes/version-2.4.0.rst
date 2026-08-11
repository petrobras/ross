Version 2.4.0
-------------

The following enhancements and bug fixes were implemented for this release:

Enhancements
~~~~~~~~~~~~

Fluid-Film TEHD Engine for Journal Bearings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The hydrodynamic journal-bearing classes were rebuilt on a shared thermo-elasto-hydro-dynamic (TEHD)
solver (the internal ``ross.bearings.fluid_film`` package). The engine solves the finite-element
Reynolds equation coupled with the film energy equation and, optionally, pad heat conduction,
pad/pivot elastic deformation and shell thermal growth. It models turbulence (Reichardt eddy
viscosity with a laminar-turbulent transition band), Swift–Stieber cavitation, groove hot-oil
carryover mixing, and Hertzian pivot-contact flexibility (ball-in-socket, button and rocker-back
pivots) — capabilities that were not available in the previous solvers.

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
regression suite at ``rtol=1e-8`` in ``ross/tests/test_fluid_film_solver.py``.

Documentation
^^^^^^^^^^^^^

The bearings tutorial (``tutorial_bearings_part_2``) was regenerated on the new engine, the
advanced-bearings cookbook recipe was rewritten, and the notebooks based on the removed
``fluid_flow`` subpackage (examples 7 and 9 and the four ``fluid_flow_*`` notebooks) were removed.

API Changes and Migration Guide
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Coefficients change with this release
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Dynamic coefficients computed by ``PlainJournal`` and ``TiltingPad`` differ from version 2.3:
the new engine uses a different turbulence model, cavitation treatment and energy equation, and
its results are anchored to the literature cases above rather than to the previous regression
values. Expect differences of a few percent in typical laminar cases and larger differences near
the laminar-turbulent transition, under heavy loads, or whenever the old adiabatic model
underpredicted film temperatures.

Removed classes and modules
^^^^^^^^^^^^^^^^^^^^^^^^^^^

===================================================  ==========================================================
Removed                                              Replacement
===================================================  ==========================================================
``ross.bearings.fluid_flow`` (subpackage)            ``PlainJournal`` (TEHD) or ``CylindricalBearing`` (analytical)
``BearingFluidFlow``                                 ``PlainJournal`` or ``CylindricalBearing``
``ST_BearingElement.from_fluid_flow``                build coefficient arrays with ``PlainJournal`` and pass them to ``ST_BearingElement``
``PlainJournalResults`` / ``TiltingPadResults``      ``FluidFilmBearingResults`` (created automatically by the bearing)
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
``plot_pad_temperature_3d()`` are new: the latter draws the pads as real geometry
colored by the solid pad conduction field (``thermal_type="full"`` only), the
through-pad counterpart of ``plot_film_temperature_3d()``.
