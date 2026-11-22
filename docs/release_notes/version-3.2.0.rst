Version 3.2.0
-------------

ROSS 3.2.0 reworks the fluid-film journal bearing solver behind ``FluidFilmBearing`` and the
``TiltingPad`` and ``PlainJournal`` classes built on it. Every solve now reports whether it
converged, and its work is bounded. The coefficients are evaluated at the converged state.
Tilting-pad coefficients are condensed exactly at any whirl frequency without a new solve. Speed
sweeps are meshed once, warm-started and can run in parallel. New, opt-in options add a Walther
viscosity law, lubricants built from an oil datasheet, a Taylor turbulence onset and a
mass-conserving (Elrod-Adams) cavitation model. The tilting-pad bearing is validated against
calculated tables of Someya (1989).

The release also replaces the bulk-flow formulation of ``HolePatternSeal`` and adds a real-gas
closure to it. Magnetic bearings get the correct sign of the negative stiffness in time-domain
analyses. ``MultiRotor`` models now save and load, and the analyses that failed on them now run.
Two new ``MotorElement`` constructors estimate the equivalent circuit from nameplate or catalog
data. The release also fixes the real-gas labyrinth seal, the ``hertz`` unit and point masses on
shaft nodes, and adds new tools to the graphical interface.

Results of existing models change for:

- fluid-film bearings;
- hole-pattern seals, and hybrid seals through them;
- real-gas labyrinth seals;
- rotors with magnetic bearings in time-domain analyses;
- unbalanced driven shafts of a ``MultiRotor``;
- ``compressor_example``;
- values given in ``hertz`` or ``kilohertz``.

Some signatures and saved files change too. Read *Results That Change for Existing Inputs* and
*Behavior Changes* in the fluid-film section, and *API Changes*, before upgrading. Unless stated
otherwise, the sizes quoted are those measured in the pull request that made the change, on its
test cases.

The following enhancements and bug fixes were implemented for this release:

Fluid-Film Journal Bearings
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Results That Change for Existing Inputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Fixed-geometry bearings change through items 1, 2, 3 and, in speed sweeps, 5; tilting-pad
bearings through all six. An isoviscous fixed-geometry bearing solved at one speed changes only
through item 3 and the "no data" temperature.

1. **Full thermal model on thin films**
   (`#1487 <https://github.com/petrobras/ross/pull/1487>`_). The full thermal model
   (``thermal_type="full"``, the default) decided whether an energy-mesh node lies on the
   film/pad interface with a tolerance of 1e-6 m. When the first film layer, of height
   ``h_min / total_ey_film``, was thinner than 1 µm, that layer was taken for the pad: it lost
   its viscous dissipation and got the pad-edge convection, so the peak pad temperature came out
   too low and the power loss too high, more so as the cross-film mesh was refined. The
   tolerance is now ``INTERFACE_TOL = 1e-8`` m. With the default ``total_ey_film=30``, every
   full-thermal bearing whose minimum film thickness is below 30 µm gets different results: a
   higher peak pad temperature and a lower power loss. On the two-axial-groove bearing of
   Fitzgerald and Neal (1992) at 2,000 to 8,000 rpm, refining the cross-film mesh across the
   1 µm threshold changed the peak pad temperature by 0.71 to 0.97 K and the power loss by 0.40 %
   to 0.88 % before the fix, and by at most 0.019 K and 0.09 % after it. Isoviscous and adiabatic
   results do not change.

2. **Solver constants in plain SI values**
   (`#1488 <https://github.com/petrobras/ross/pull/1488>`_). Constants written as unit
   conversions were replaced by rounded SI values:

   - The rigid-rotor stability threshold (``threshold_rpm``) uses standard gravity,
     9.80665 m/s², instead of 9.8044 m/s², so it rises by 0.0115 %. No other output depends on
     gravity.
   - The convergence tolerances were rounded, all temperature tolerances towards tighter values:
     shear stress 70 Pa (was 68.95 Pa), pad temperature 0.005 K (was 0.00556 K), journal
     temperature 0.05 K (was 0.0556 K), inlet temperature 0.25 K (was 0.278 K), sump temperature
     0.5 K (was 0.556 K), and the journal-temperature step limit 5 K (was 5.56 K). Results of
     thermal and deformation models move: on the stored reference cases by at most 3.3e-4 of
     the largest value of each output (in ``cxy``), and on a five-pad full-thermal tilting pad
     ``kxy`` moved by 1.5e4 N/m, with ``kxx`` = 4.8e8 N/m. Run with every tolerance ten times
     tighter, the new results are closer to that solution than the old ones. Isoviscous results
     do not move, apart from the "no data" temperature (see *Behavior Changes*).

3. **Coefficients at the converged state**
   (`#1492 <https://github.com/petrobras/ross/pull/1492>`_). The stiffness was taken from the
   Jacobian of the equilibrium search, at the Newton iterate before the converged one. Stiffness
   and damping are now evaluated at the state the other outputs report. On the 26 stored
   reference cases, the largest change over the largest coefficient of the case is 0.36 % in the
   stiffness of tilting pads and 0.28 % in the stiffness of fixed-geometry bearings; the damping
   of fixed-geometry bearings does not change, and that of tilting pads changes by up to 0.26 %.
   ``threshold_rpm`` moves by up to 0.10 % and ``whirl_frequency_ratio`` by up to 0.18 %. With a
   nonzero ``cavitation_pressure`` the cavitated film is now held at that pressure; it was set
   to 0 Pa in the diverging zone and on fully cavitated pads. With the default
   ``cavitation_pressure=0`` this changes nothing.

4. **Tilting-pad inertia about the pivot**
   (`#1493 <https://github.com/petrobras/ross/pull/1493>`_). The pad mass and inertia were those
   of a rectangular block about its centroid, :math:`m (t^2 + L^2) / 12`, which includes the
   axial length and uses the wrong axis. They are now those of the annular pad sector about the
   pivot, with the coupling between tilt and pivot motion of an offset pivot. Fixed-geometry
   results do not change. On the stored tilting-pad cases, at synchronous whirl, ``kxx`` moves by
   up to 0.31 %, ``kyy`` by 0.05 %, ``cxx`` by 0.13 % and ``cyy`` by 0.03 %. Above synchronous
   whirl the pad tilt resonance shifts and the change is much larger: on a five-pad isoviscous
   tilting pad, ``kxx`` changes by -0.30 % at 1X, -5.4 % at 2X, +55 % at 3X and +18 % at 5X, and
   ``kyy`` by -0.2 % at 2X, -0.7 % at 3X and -3.3 % at 5X. These cases have thick pads
   (thickness/radius 0.75), for which the new inertia is 3.8 times the old one. On
   ``tilting_pad_adiabatic_example`` (thickness/radius 0.25) the change is about 2e-4.

5. **Warm-started speed sweeps**
   (`#1494 <https://github.com/petrobras/ross/pull/1494>`_). A bearing solved at several speeds
   now starts each speed from the converged state of the previous one (``warm_start=True``, the
   default; see *New Options*). A warm-started speed converges to the same tolerances as before,
   from another starting point, so **the coefficient at one speed now depends, within the solver
   tolerances, on the other speeds of the sweep**. Measured against independent solves, the
   difference reaches 3.4e-3 relative (``cxx`` of a ten-speed adiabatic tilting pad); on the
   other sweeps measured it is between 8.1e-4 and 2.3e-3 in the largest direct term. With
   ``warm_start=False`` every speed is an independent solve and equals a bearing built for that
   speed alone, bit for bit. A bearing solved at one speed does not change. A speed that fails
   from the warm start is solved again from the cold start, and some speeds that did not converge
   from the cold start now converge (an isoviscous plain journal at 8,000 to 10,000 rpm).

6. **Tilt search of tilting pads**
   (`#1495 <https://github.com/petrobras/ross/pull/1495>`_). The pad tilts are found with
   Brent's method instead of bisection, to a tighter tolerance: a 1e-10 rad bracket
   (``TILT_ERROR``, was 1e-8 rad) or a pivot moment below 1e-9 of the pad's moment scale
   (``TILT_MOMENT_ERROR``). Every tilting-pad result moves slightly: on the stored cases the
   stiffness by at most 4.2e-5 and the damping by at most 2.3e-5 of the largest entry of each
   matrix, the tilts by at most 8.5e-9 rad and the journal position by at most 5.4e-6 of the
   bore clearance. Fixed-geometry results do not change.

Smaller changes of results:

- The attitude angle is ``arctan2(x, -y)``, continuous onto the axes; the previous formula
  returned 0 for a journal exactly on an axis. Off the axes the two agree. A Reynolds number
  exactly equal to ``re_laminar`` is now treated as laminar; before, it fell into neither regime
  (`#1490 <https://github.com/petrobras/ross/pull/1490>`_).
- The arc of an inter-pad groove that crosses 0 deg was computed by adding 360 to an angle in
  radians, so a 20 deg groove became a 354 rad one. This affected pad lists such as
  ``pivot_angle=[270, 60]``; the same bearing listed as ``[60, 270]`` was right
  (`#1491 <https://github.com/petrobras/ross/pull/1491>`_).
- The outputs of a case stopped before it converged are those of its last Newton iterate; before,
  a case stopped inside a thermal pass reported outputs of its initial guess
  (`#1491 <https://github.com/petrobras/ross/pull/1491>`_).
- Static coefficients (``frequency=0`` or ``excitation_ratio=0``) of tilting pads have a finite
  damping; it was NaN (`#1493 <https://github.com/petrobras/ross/pull/1493>`_).
- The new cavitation model (`#1499 <https://github.com/petrobras/ross/pull/1499>`_) leaves the
  default path unchanged: the stored reference cases do not move beyond 1e-12 relative. Its only
  change on the default path is that the rigid-rotor stability estimate no longer raises
  ``ZeroDivisionError`` for a singular damping matrix; it gives no threshold (reported as zero),
  as for the other degenerate cases.
- The closed-film stop of thermal cases (`#1500 <https://github.com/petrobras/ross/pull/1500>`_)
  changes no result of a case whose film stays open (see *Behavior Changes* for the cases
  whose film closes).

Behavior Changes
^^^^^^^^^^^^^^^^

- **Failed speeds are left out of the coefficient table.** A case is converged when every loop
  that ran met its tolerance; for the equilibrium this is the force balance
  :math:`|F_{hyd} + F_{ext}| / |F_{ext}| < 2 \times 10^{-3}`, recomputed from the returned
  state. With the new default ``on_failure="warn"``, a speed that did not converge is left out
  of the element's coefficient table, which then interpolates between the converged speeds, and
  a ``ConvergenceWarning`` (a ``RuntimeWarning``) names the speed, the loop and the reason. **A
  sweep with a failed speed therefore returns a shorter table.** The failed case stays in
  ``results`` and ``operating_results`` with ``converged=False``. ``on_failure="raise"`` raises
  ``FluidFilmConvergenceError`` instead; ``on_failure="keep"`` keeps every row, as before, and
  still warns. If no speed converges, every row is kept and the warning says that the table
  holds failed solves. In a (speed, frequency) table each speed has one equilibrium, so a failed
  speed fails all its frequencies (`#1491 <https://github.com/petrobras/ross/pull/1491>`_,
  `#1493 <https://github.com/petrobras/ross/pull/1493>`_).
- **Work budget.** Each case has a budget of Reynolds solves, ``max_reynolds_solves`` (default
  400,000 per pad, counting at least 5 pads), a cap of 4,000 per pad on one equilibrium solve,
  and an optional ``time_limit`` in seconds. The largest converging case measured needed about a
  quarter of the default budget. **A case that reaches the budget was stopped before it
  converged; it does not mean that it cannot converge.** It is reported with reason
  ``"budget"`` (or ``"time_limit"``) and handled by ``on_failure``; raise
  ``max_reynolds_solves`` to let it continue. A fully cavitated pad is not counted as a solve,
  and every mass-conserving film solve is counted
  (`#1491 <https://github.com/petrobras/ross/pull/1491>`_,
  `#1495 <https://github.com/petrobras/ross/pull/1495>`_,
  `#1499 <https://github.com/petrobras/ross/pull/1499>`_).
- **Cases that never returned now end.** A plain journal loaded far beyond its capacity
  (8e5 N), which looped forever when a Newton step moved the journal through the bore, ends
  with reason ``"contact"`` (`#1491 <https://github.com/petrobras/ross/pull/1491>`_).
- **Convergence summary.** ``results.convergence`` holds a summary per case, with one entry per
  loop. Every case has a ``cavitation`` entry, whose reason is ``"not_applicable"`` with the
  default cavitation model (`#1499 <https://github.com/petrobras/ross/pull/1499>`_). The
  ``tilt`` entry gives the residual of the worst pad relative to its tolerances (converged at 1)
  (`#1495 <https://github.com/petrobras/ross/pull/1495>`_).
- **Thermal cases stop on a closed film.** When an equilibrium pass of a thermal case
  (``"adiabatic"`` or ``"full"``) puts the journal through a pad (minimum film thickness at or
  below zero), the case ends there with reason ``"contact"``, without solving the energy equation
  on the closed film. Its outlet temperatures are NaN and its peak pad temperature is the
  "no data" value (NaN in ``operating_results``). Before, such cases kept iterating on
  the inverted film, diverged to a point that depended on the platform, or raised
  ``ZeroDivisionError`` (`#1500 <https://github.com/petrobras/ross/pull/1500>`_).
- **Input validation.** Inputs are checked before the first solve, and an input outside its
  valid range raises ``ValueError`` with a message that names the argument: non-positive or
  non-finite speeds (reverse rotation is not supported), negative frequencies or
  ``excitation_ratio``, non-positive geometry, density or flow, preload or offset out of range,
  overlapping pads or pads that cover more than 360 deg, and a pivot model without the data it
  needs. Most of these used to give ``ZeroDivisionError``, NaN coefficients or a solve that
  never returned. The checks run only when the coefficients are solved, not when ``kxx`` and
  the other coefficients are given directly (`#1490 <https://github.com/petrobras/ross/pull/1490>`_).
- **New warnings.** A ``UserWarning`` when a load is given with
  ``equilibrium_type="match_eccentricity"`` (the ``TiltingPad`` default), where the load is not
  used, and when nonzero ambient pressures are given with ``operating_type="regular_flooded"``,
  where they act only in the inter-pad grooves
  (`#1490 <https://github.com/petrobras/ross/pull/1490>`_).
- **"No data" temperature 0 K.** The temperature outputs of runs without a thermal model
  (``tpad_max``, ``temp_in_let``, ``temp_out_let``, ``temp_out_let_bulk``, ``temp_sump``, the
  journal and probe temperatures and ``y_max_t``) are 0.0 K instead of 255.37 K. Code that
  compares them with 255.37 K should compare with
  ``ross.bearings.fluid_film.driver.ZERO_TEMPERATURE_SENTINEL``. ``FluidFilmBearing.save`` keeps
  only the coefficient table, so no saved file holds the old value
  (`#1488 <https://github.com/petrobras/ross/pull/1488>`_).
- **Parallel sweeps raise when the worker processes cannot start.** ``num_processes`` is now an
  alias of the new ``n_jobs``; passing both with different values raises ``ValueError``. The
  worker processes are started with ``spawn``. When they stop before returning results, a
  ``RuntimeError`` is raised, chained to the worker error; there is no fallback to a serial
  solve. The usual cause is a script that calls the solve with ``n_jobs > 1`` outside an
  ``if __name__ == "__main__":`` guard. Such scripts already failed on Windows and macOS and on
  Linux with Python 3.14 and later. **On Linux with Python 3.13, an unguarded script with**
  ``num_processes > 1`` **used to run and now stops with this error:** put the call under the
  guard, or use ``n_jobs=1``. Jupyter notebooks need no guard
  (`#1494 <https://github.com/petrobras/ross/pull/1494>`_).
- **Warm-started speeds and** ``relax_pressure``. A warm-started speed takes full Newton steps;
  ``relax_pressure`` applies to cold starts and to the cold retry of a failed warm start
  (`#1494 <https://github.com/petrobras/ross/pull/1494>`_).
- **Tilting pads with zero preload no longer fail.** Before, the tilt search of an unloaded pad could end on a
  film of round-off thickness and raise ``ZeroDivisionError``. Trial films thinner than
  ``1e-9`` of the bore clearance are now rejected as closed. The matching of energy and
  Reynolds nodes in the full thermal model raises ``ValueError`` where it used to write out of
  bounds (`#1497 <https://github.com/petrobras/ross/pull/1497>`_).
- ``re_turbulent`` **default None.** The default of ``re_turbulent`` changed from 1000 to None,
  which resolves to 1000 with a numerical ``re_laminar``, so existing calls behave as before
  and the attribute still holds the integer 1000. With ``re_laminar="taylor"`` it resolves to
  twice the Taylor onset (see *New Options*)
  (`#1496 <https://github.com/petrobras/ross/pull/1496>`_).
- **Internal modules.** Code that imports from ``ross.bearings.fluid_film`` directly may need
  changes: ``coefficients.dynamic_reduction`` and ``dynamic_reduction_pivot`` were removed (the
  condensation is in the new ``padmatrices`` module), ``pressure.press`` returns a fourth item,
  the cavitated mask, and the perturbation functions of ``coefficients`` take that mask
  (`#1492 <https://github.com/petrobras/ross/pull/1492>`_,
  `#1493 <https://github.com/petrobras/ross/pull/1493>`_).
  ``_numba_kernels.viscosity_temp_jit`` and ``thd.temp_mixing_carryover`` moved to the new
  ``ross.bearings.film`` subpackage and still import from their old modules
  (`#1489 <https://github.com/petrobras/ross/pull/1489>`_). ``FluidFilmBearingResults`` files
  saved by 3.1 still load.

New Options
^^^^^^^^^^^

All of these are opt-in or add outputs; none changes a result unless it is used.

Convergence and operating results
  ``FluidFilmBearing`` takes ``on_failure="warn"|"raise"|"keep"``, ``max_reynolds_solves`` and
  ``time_limit`` (see *Behavior Changes*). ``results`` and ``operating_results`` are available
  on the bearing. ``FluidFilmBearingResults`` gains ``converged``, ``convergence``,
  ``whirl_frequency``, ``pad_forces``, ``bearing_data`` and ``operating_results``, a
  ``pandas.DataFrame`` with one row per case: speed, frequency, convergence and reason,
  iterations, eccentricity, attitude, minimum film thickness, peak pressure, power loss,
  Sommerfeld number, the maximum, Couette and Taylor Reynolds numbers, and the force residual.
  The new module ``ross.bearings.fluid_film.diagnostics`` holds the loop records, the budget
  and ``FluidFilmConvergenceError``
  (`#1491 <https://github.com/petrobras/ross/pull/1491>`_).

Pad matrices and coefficients at any whirl frequency
  ``results.coefficients(i, frequency)`` returns the stiffness and damping of speed ``i``
  condensed exactly at any whirl frequency (or an array of them) from one solve, and
  ``results.coefficients(i, frequency=0)`` the static limit. ``results.pad_matrices(i)``
  returns the stiffness, damping and mass matrices of every pad on journal, tilt and pivot
  coordinates (``ross.bearings.fluid_film.padmatrices.PadMatrices``), with ``condense``,
  ``condense_static``, ``impedance(s)`` at a complex frequency (the bearing at a damped rotor
  eigenvalue) and ``condense_limit``. ``condense_limit`` is the high-frequency asymptote of the
  condensation, approached only far above any rotor whirl: it is not a stiffness to use in a
  rotor model. A (speed, frequency) table solves one equilibrium per speed and condenses it at
  each frequency. The element's ``coefficients()`` still interpolates its table, as the rotor
  does (`#1493 <https://github.com/petrobras/ross/pull/1493>`_).

Speed sweeps
  ``warm_start=True`` (default) and ``chunk_size=5``: the sorted speeds are cut into chains of
  ``chunk_size`` consecutive speeds, and each speed after the first of its chain starts from the
  previous one. ``n_jobs`` solves chains in parallel worker processes, with results identical
  for every ``n_jobs``, bit for bit. The mesh and the speed-independent arrays are built once
  per sweep. ``results.convergence[i]["warm_start"]`` records how each speed was started. On the
  bearings measured, warm-started sweeps needed 40 % to 60 % fewer Reynolds solves
  (`#1494 <https://github.com/petrobras/ross/pull/1494>`_). With the faster inner loops of
  `#1495 <https://github.com/petrobras/ross/pull/1495>`_ (Brent tilt search, no solve for a
  fully cavitated pad, one factorisation for the perturbation solves, compiled post-processing
  and adiabatic loops), a ten-speed isoviscous tilting-pad sweep needs 1,013 Reynolds solves
  instead of 10,222 before these two changes, and a ten-speed full-thermal one 56,925 instead of
  771,962.
  ``ross.bearings.fluid_film.sweep.film_force_at`` gives the film force, pad forces and moments
  with the journal held at a position (`#1492 <https://github.com/petrobras/ross/pull/1492>`_).

Lubricants
  ``viscosity_model="exponential"`` (the two-point fit used so far, and the default) or
  ``"walther"`` (ASTM D341). The two laws pass through the same two table points and differ
  between and outside them: for ISOVG32 the exponential fit is 10.8 % to 17.7 % above the
  Walther law between 50 and 80 °C, 33 % below at 20 °C and 20 % below at 120 °C. ``lubricant``
  also accepts a ``ross.bearings.film.lubricant.Lubricant``, which keeps its own law.
  ``Lubricant.from_datasheet(nu40, nu100, rho15)`` builds one from the kinematic viscosities at
  40 and 100 °C and the density at 15 °C (Walther law by default), with the thermal expansion of
  the density in the form of API MPMS 11.1 and the specific heat and conductivity of Cragoe
  (1929), each of which can be overridden. ``Lubricant.to_dict()`` writes the ``lubricants_dict``
  format plus ``viscosity_model``
  (`#1496 <https://github.com/petrobras/ross/pull/1496>`_).

Taylor turbulence onset
  ``re_laminar="taylor"`` sets the laminar limit to Taylor's onset of vortex flow,
  :math:`Re_c = 41.2 \sqrt{R / C_b}`, and ``re_turbulent=None`` then resolves to
  :math:`2 Re_c`. Two choices of this implementation are not part of Taylor's criterion: the
  factor 2 for the fully turbulent limit, and the comparison with the local Reynolds number of
  the film (``re_max``) instead of one based on the mean clearance, which on preloaded pads
  brings the onset earlier. ``operating_results`` reports ``re_max``, ``re_cb`` and
  ``re_taylor`` (`#1496 <https://github.com/petrobras/ross/pull/1496>`_).

Mass-conserving cavitation
  ``cavitation_model="mass_conserving"`` (default ``"reynolds"``) solves each pad's film with
  the Elrod-Adams model, in pressure and film fraction, on the pad's Reynolds nodes, so the oil
  that crosses the cavitated zone as a striated film is conserved and the film ruptures and
  reforms where the mass balance requires. It supports ``thermal_type`` None or ``"adiabatic"``,
  rigid pads, ``"regular_flooded"`` operation, and ``"starved_condition_even"`` on
  fixed-geometry bearings; other combinations (full thermal model, pad deformation, starved
  tilting pads, the pressure dam, a 360 deg pad, other operating types) raise
  ``NotImplementedError`` before any solve. The pad flows come from the fluxes of the film
  solver, so a starved bearing drains exactly its supply. The stiffness is the derivative of the
  mass-conserving film force; the damping is approximate (see *Known Limitations*). The film
  fraction is in ``results.film_fraction_fields`` and ``plot_film_fraction_2d()``
  (`#1499 <https://github.com/petrobras/ross/pull/1499>`_). The solver is
  ``ross.bearings.film.cavitation`` (``CavitationGrid``, ``solve_elrod_adams``)
  (`#1498 <https://github.com/petrobras/ross/pull/1498>`_).

  Under starved supply the two models differ in kind: the default model moves the film onset into
  the pad, while the mass-conserving film keeps the onset at the leading edge and sets the inlet
  film fraction so that each pad draws the oil available to it. On the two-axial-groove example
  journal (isoviscous, carry-over 0.8) the direct stiffnesses of the mass-conserving film differ
  from the default model by -14 % (``kxx``) and +33 % (``kyy``) at 12 L/min, and by -32 % and
  +76 % at 8 L/min (from `#1499 <https://github.com/petrobras/ross/pull/1499>`_). The
  cross-coupled stiffness changes more: measured on main on the same case for these notes, ``kxy``
  rises by 238 % at 12 L/min and by 521 % at 8 L/min, where the default model's ``kxy`` is 0.30
  and 0.27 times its ``kxx``. Flooded, the two models differ by at most 0.76 % in any coefficient
  on the finest of the four meshes compared (160 x 80); see *Known Limitations* for the trend with
  the mesh.

Film core
  ``ross.bearings.film`` is a new internal subpackage with the parts of the solver that do not
  depend on the bearing geometry: lubricant laws, hot-oil mixing, turbulence constants,
  convergence records, pad condensation and the Elrod-Adams solver. It is not exported from
  ``ross`` or ``ross.bearings``, and ``import ross`` does not load it
  (`#1489 <https://github.com/petrobras/ross/pull/1489>`_).

Validation against Someya (1989)
  The tilting-pad bearing is compared with four calculated tables of Someya (Ed.),
  *Journal-Bearing Databook* (4 pads of 80 deg and 5 pads of 60 deg, load between pads,
  L/D 0.5, centred pivots, preload 0 and 1/2): the eccentricity within 0.02 and the synchronous
  direct stiffness and damping within 8 %, for eccentricities 0.2 to 0.8 at preload 1/2 and 0.3
  to 0.8 at preload 0. The largest gap on those rows is 7.8 %
  (`#1497 <https://github.com/petrobras/ross/pull/1497>`_).

Known Limitations
^^^^^^^^^^^^^^^^^

- Starved operation with the mass-conserving film is supported only on fixed-geometry bearings,
  and its results have not been compared with measurements or published calculations. Without
  hot-oil carry-over, several strongly starved cases end on contact or on the iteration limit
  and are reported as not converged; whether contact is the physical answer there was not
  checked (`#1499 <https://github.com/petrobras/ross/pull/1499>`_).
- Flooded, the gap between the mass-conserving and the default coefficients does not decrease
  steadily as the mesh is refined (on four meshes, ``kxy`` differs by -3.21 %, -1.39 %, +2.29 %
  and +0.76 %). The cause is not known
  (`#1499 <https://github.com/petrobras/ross/pull/1499>`_).
- The damping of the mass-conserving film is approximate: it leaves out the change of the film
  content with the squeeze velocity (`#1499 <https://github.com/petrobras/ross/pull/1499>`_).
- The density constant used by ``Lubricant.from_datasheet`` for the thermal expansion
  (0.6278 kg/(m³ K) for lubricating oils) has not been checked against API MPMS 11.1
  (`#1496 <https://github.com/petrobras/ross/pull/1496>`_).
- At zero preload and low eccentricity, the tilting-pad stiffness is below the Someya tables, by
  7 % to 47 % for eccentricities of 0.2 and below, while the damping agrees within about 5 %.
  The cause is not known (`#1497 <https://github.com/petrobras/ross/pull/1497>`_).
- The density, specific heat and thermal conductivity in the energy equation of the journal
  bearings are constant, evaluated at ``oil_supply_temperature`` for a temperature-dependent
  ``Lubricant`` (`#1496 <https://github.com/petrobras/ross/pull/1496>`_).
- At the default tolerances the stiffness and damping of thermal cases carry a convergence error
  of about 0.3 % to 0.5 %, measured against solves with ten times tighter tolerances
  (`#1488 <https://github.com/petrobras/ross/pull/1488>`_).

Enhancements
~~~~~~~~~~~~

Motor Equivalent Circuit from Nameplate and Catalog Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``MotorElement`` needs the per-phase T equivalent circuit (``stator_resistance``,
``rotor_resistance``, ``stator_reactance``, ``rotor_reactance`` and ``mutual_reactance``). Two
new constructors estimate it, with methods compared by Wengerkievicz et al. (2025).
Contributed by Jessica Guarato, based on code by Hélder de Paula
(`#1470 <https://github.com/petrobras/ross/pull/1470>`_).

- ``MotorElement.from_nameplate(n, power_rated, voltage_rated, speed_rated, efficiency,
  power_factor, Ip_motor, ..., method="kueck")`` uses rated data only. Its methods are
  ``"asdp"``, ``"kueck"`` and ``"lee"``.
- ``MotorElement.from_catalog(..., method="haque2")`` also takes the efficiency and power factor
  at 75 % and 50 % load and the starting and maximum torque ratios. Its methods are ``"haque1"``,
  ``"natarajan_misra"`` and ``"haque2"``.
- The functions behind them are exported from ``ross``: ``estimate_from_nameplate``,
  ``estimate_from_catalog``, ``NAMEPLATE_METHODS`` and ``CATALOG_METHODS``. They return an
  ``EstimationResult``, which holds the circuit, the core-loss resistance ``Rc`` when the method
  gives one, a convergence flag and warnings.
- Voltage and current are per phase of the star equivalent.
- For ASDP, Kueck, Lee and Haque 1, the stator share of the leakage reactance depends on
  ``motor_category``: 40 % for N, 30 % for H and 50 % for D.
- A method that does not converge issues a ``UserWarning``. A method that never reaches a
  physical circuit raises ``RuntimeError``.
- ``MotorElement`` has no core-loss branch, so it does not use ``Rc``.

``tutorial_motor`` has a new section 1.1, and the agent skill has a recipe for it.

Real-Gas Closure for ``HolePatternSeal``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``HolePatternSeal`` takes ``gas_model``, as ``LabyrinthSeal`` does
(`#1481 <https://github.com/petrobras/ross/pull/1481>`_):

- ``gas_model="ideal"``, the default, is a perfect gas with the Sutherland viscosity.
- ``gas_model="real"`` needs ``gas_composition``. The density, enthalpy, entropy and viscosity come
  from the equation of state set by ``ccp.config.EOS``. They are tabulated over pressure and
  temperature and fed to the same equations as the perfect gas. The table uses bicubic splines on
  41 x 41 nodes, on a grid sheared along the inlet isentrope. A table is never extrapolated: a flow
  that leaves it gets a wider table, for that speed only.
- An inlet that is not a single-phase gas raises ``ValueError`` before any other call to the
  equation of state. So does a table that reaches a liquid or two-phase state; the message names
  the state.
- The new attributes are ``gas_model``, ``gas`` and ``choked``, one flag per speed.

The closure is checked against the equation of state, against the ideal closure at low pressure,
against closed-form gases and against conservation laws. It has not been compared with dense-gas
seal measurements (see *Known Issues*). The equations it uses are the corrected formulation
described under *Bug Fixes*.

``MultiRotor`` Analyses and Concatenation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Analyses inherited from ``Rotor`` used to rebuild a ``MultiRotor`` as a plain ``Rotor`` and
failed. They now rebuild it with its two-rotor topology, so ``run_ucs``, ``run_level1``,
``run_static`` and ``summary`` run on a ``MultiRotor``. On 3.1.0 they raised. Contributed by
Murillo Alvaro Borges de Sousa (`#1426 <https://github.com/petrobras/ross/pull/1426>`_, issue
`#1425 <https://github.com/petrobras/ross/issues/1425>`_). The changes include:

- ``MultiRotor.summary()`` gives one row per shaft, and the UCS plot draws every mode instead of
  every second one.
- ``MultiRotor.run_critical_speed(speed_range=None, num_modes=12, rtol=0.005, shaft=0)`` gives the
  critical speeds of the shaft chosen by ``shaft``.
- ``MultiRotor.run_unbalance_response`` evaluates the response at the speed of the unbalanced
  shaft. It raises ``ValueError`` when the unbalances are on shafts with different speeds.
- The faults (``Rubbing``, ``Crack``, ``MisalignmentFlex`` and ``MisalignmentRigid``) use the
  speed and angle of the faulted shaft, through the new ``to_shaft_rotation(value, node)``.
- The harmonic balance converts each harmonic to the speed of its shaft and warns when the
  converted order is not an integer.
- ``MultiRotor.concatenate(*rotors, tag=None)``, also called by ``+`` and by ``Rotor.concatenate``,
  adds rotors at the start of the driving shaft or at the end of the driven shaft. Multi-rotors
  can be nested into trains of more than two shafts, whose gear ratios are multiplied along the
  train.
- ``MultiRotor.split_elements(elements, node_scale=1)`` separates elements into those of the
  driving rotor and those of the driven rotor.
- ``MultiRotor`` takes ``min_w``, ``max_w`` and ``rated_w``.
- ``run_level1`` with ``rated_w=None`` runs at speed 0 instead of raising, also on a plain
  ``Rotor``.

Graphical Interface
~~~~~~~~~~~~~~~~~~~

The graphical interface (``ross-interface``) gained the following, contributed by Leonardo
Cabral:

- **Plot window of transient analyses.** The ``time_response``, ``misalignment``, ``rubbing``
  and ``crack`` forms have ``window_start`` and ``window_end`` fields (in s) and a
  ``one_revolution`` field. These are passed to ``plot_1d``, ``plot_2d`` and ``plot_3d``. They are
  hidden for the DFFT, and changing them redraws the plot without integrating again
  (`#1468 <https://github.com/petrobras/ross/pull/1468>`_).
- **Mesh convergence options.** The discretization dialog has a strategy ("Uniform and graded",
  which is ``cheapest``, or "Uniform only", which is ``uniform``) and ``max_elements``. Both are
  passed to ``Rotor.run_mesh_convergence`` (`#1468 <https://github.com/petrobras/ross/pull/1468>`_).
- **Hybrid seal stages as fields.** The ``HybridSeal`` form shows one field per key of
  ``hole_pattern_parameters`` and ``labyrinth_parameters``. Each field has its unit and the help
  text of ROSS's docstring. Lengths are in mm, temperatures in K and viscosities in Pa·s. The
  dictionaries stay in the advanced fields for projects saved with them, and a field overrides
  the same key there (`#1469 <https://github.com/petrobras/ross/pull/1469>`_).
- **Standard views.** A button in the 3-D view, and the menu of the empty view, offer five views:
  the opening view, isometric, front (along the rotor axis), side and top. The camera flies to
  the view in half a second, a drag or a scroll hands control back, and with
  ``prefers-reduced-motion`` the camera jumps instead
  (`#1472 <https://github.com/petrobras/ross/pull/1472>`_).
- **Element forms**, after user tests (`#1467 <https://github.com/petrobras/ross/pull/1467>`_):

  - ``xj`` and ``yj`` of ``TiltingPad`` have help text. They are read as fractions of the radial
    clearance; before, they were mapped to mm, so a typed 0.3 reached ROSS as 0.0003.
  - ``equilibrium_type``, the shaft's ``shear_method_calc`` and every ``lubricant`` field are
    selects.
  - The ball, roller and magnetic bearing forms have units, in SI by default.
  - A list can be typed without brackets (``1000, 1500``). A single speed or frequency is passed
    as a one-element array where ROSS needs a list: ``CylindricalBearing`` ``speed`` and
    ``frequency``, and ``SqueezeFilmDamper`` ``frequency``.
  - The point-mass labels say that ``mx``, ``my`` and ``mz`` are masses.
  - A quantity that ``check_units`` does not convert is passed to ROSS as its SI magnitude. Before,
    a point mass with ``mz`` failed with "Cannot convert from 'kilogram' to 'dimensionless'".

- **MultiRotor halves** (`#1471 <https://github.com/petrobras/ross/pull/1471>`_):

  - The coupled node follows the gear when one half is split, meshed or has a shaft inserted.
    Before, the MultiRotor then refused to build with "Each rotor needs a GearElement in the
    coupled nodes!".
  - An element of either half is offered the materials of both halves. A material chosen from
    the other half is copied into the element's own half.

- **Rotor figure** (`#1473 <https://github.com/petrobras/ross/pull/1473>`_):

  - The figure's box grows with the vertical scale. At 5x the figure used to overflow it by
    548 px.
  - The 2D/3D buttons are visible before a rotor is built.
  - The empty figure has an "Add the first shaft" button.

Bug Fixes
~~~~~~~~~

Fix the Hole-Pattern Seal Formulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The bulk-flow formulation of ``HolePatternSeal`` was replaced by a corrected one
(`#1481 <https://github.com/petrobras/ross/pull/1481>`_, issue
`#1475 <https://github.com/petrobras/ross/issues/1475>`_):

- **What was wrong.** The defects listed in the issue are fixed, apart from two model terms (see
  *Known Issues*). Among them, the perturbation solution had no exit boundary condition and treated the cell depth as
  clearance, so a smooth seal got negative direct damping. A choked result depended on the
  iteration history, and the static stiffness was off by 4 % to 18 %. On the smooth seal of the
  issue (air from 70 to 35 bar, with ``frequency`` equal to ``speed`` at 10,200 to 20,200 rpm),
  ``cxx`` goes from -1.120e5 to -9.68e4 N·s/m before the fix to +3.158e4 to +3.338e4 N·s/m after
  it.
- **The new equations.** One set of equations covers a general equation of state: the base flow
  is solved for pressure, temperature and swirl, and the cells are storage compressed
  isentropically.
- **The outlet search** brackets the outlet and refines it with Brent's method.
- **Choked seals.** A choked seal returns the converged choke-limit state, flagged in
  ``choked``, and its leakage no longer depends on ``outlet_pressure``. Before, it returned the
  first unchoked trial, or NaN when the first guesses choked.

Compared with Childs and Wade (2004), the frequency trends are reproduced but the magnitudes are
not (see *Known Issues*). The coefficients of every hole-pattern model change, and so does what
they mean (see *API Changes*). A new theory page describes the model.

Fix the Sign of the AMB Negative Stiffness in Time-Domain Analyses
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The frequency-domain model of ``MagneticBearingElement`` already treated ``ks`` as a negative
stiffness (ISO 14839-1). The two time-domain paths did not, and they also left out the amplifier
and sensor gains. These paths are ``compute_amb_controller``, used by the Newmark integration of
``run_time_response``, and ``AmbTimeResponse``, which uses ``lsim``. Contributed by
@mariac-souza (`#1464 <https://github.com/petrobras/ross/pull/1464>`_).

- **The force.** Both paths applied ``+ks * x``. With ``ks < 0`` this is a restoring force, so the
  negative stiffness acted as a positive one. The force is now ``ki * i - ks * x``.
- **The gains.** ``k_sense`` scales the controller input and ``k_amp`` its output, so the current
  output of ``AmbTimeResponse`` is the coil current. Before, the time-domain paths did not apply
  them. ``k_amp`` is documented in A/V; it was documented in V/A.
- **The two example rotors.** With the sign fixed, both were found to be stable only because of
  the error:

  - ``rotor_example_amb_general_controllers``: ``Kp`` goes from 1000 to 3000. Levitation requires
    ``ki * k_amp * k_sense * Kp > |ks|``, that is ``Kp > 1000`` for this actuator.
  - ``rotor_example_amb_complex_controllers``: the proportional gains are applied once; they were
    applied twice. ``nw`` goes from 138 to 276 and ``ag`` from 470.3e-6 to 235.15e-6 m², which are
    the turns of one electromagnet and the area of one pole.

- ``MagneticBearingElement.axis_gain(gain, dof_index)`` is a new static method. It returns the
  gain of one control axis.
- ``tutorial_bearings_part_3`` gives the force law with ``Ks < 0`` and corrects the ``K_eq`` and
  ``C_eq`` formulas.

Frequency-domain results do not change. The sign error is in 3.1.0.

Fix the Real-Gas Swirl Term of ``LabyrinthSeal``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The fix is in `#1479 <https://github.com/petrobras/ross/pull/1479>`_:

- **The defect.** With ``gas_model="real"``, the swirl-velocity coefficient of the linearised
  cavity continuity equation was multiplied by :math:`p \kappa_T`, which equals 1 only for a
  perfect gas. It is now :math:`\rho A / R_s`, with the cavity density of the base flow.
- **The size, on a test seal.** For methane at 30 MPa and 350 K, :math:`p \kappa_T` is 0.76, and
  the old code gave ``kxy`` 190 % too high, ``kxx`` 16.8 % too high and ``cxx`` 31.8 % too high.
  For air at 0.3 MPa, ``kxy`` moves by 2 %.
- **What does not change.** The leakage and the base flow, and every result of
  ``gas_model="ideal"``, bit for bit.
- **New methods.** The gas backends provide ``IdealGas.cg2`` and ``RealGas.cg2`` in
  ``ross.seals.gas_model``.

Fix ``LabyrinthSeal`` Leakage Iterations That Never Ended
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The fix is in `#1480 <https://github.com/petrobras/ross/pull/1480>`_:

- **The defect.** The leakage bisection bracketed every tooth with the critical pressure ratio of
  the last tooth, computed from the state left by the previous iterate. Fluxes near the solution
  were then taken as choked, and the bisection could cycle forever. Both closures were affected:
  the real gas, and the perfect gas mostly at high cp/cv.
- **The fix.**

  - Each tooth is bracketed with its own critical ratio.
  - A near-choked march is accepted only when the outlet station is above the outlet pressure.
  - A bracket that closes at the choke onset of the last throttle gives the choked solution.
  - After ``MAX_LEAKAGE_ITERATIONS = 200`` bisection steps, the loop raises ``RuntimeError``.

- **Effect on a scan of 5,053 inputs.**

  - The 111 inputs that hung now converge, and so do the 31 that failed with a NaN.
  - No input that was solved before now fails.
  - Of the 4,629 inputs that both versions solve, 4,430 have a bit-identical leakage.

- **The leakage that moved.**

  - Choked seals that were already inside the choke band moved by at most 5.0e-4 in leakage and
    1.2 % in ``kxy``.
  - Fifteen results whose outlet station was below the outlet pressure now match it. One of them
    moves by 12 %, the others by at most 0.3 %.
  - Two results now choke, 0.68 % lower.

- **Cost.** The real-gas base flow is slower: for a 16-tooth air seal it went from 34 to 89 ms.

Fix Saving and Loading ``MultiRotor``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A saved ``MultiRotor`` could not be loaded. ``Rotor.load`` raised ``ValueError: 4 is not in list``,
``MultiRotor.load`` raised ``TypeError``, and results saved with a ``MultiRotor`` could not be
loaded either. The fix is in `#1426 <https://github.com/petrobras/ross/pull/1426>`_ (issue
`#1425 <https://github.com/petrobras/ross/issues/1425>`_):

- A ``MultiRotor`` file holds the constructor options and the driving and driven rotors, nested
  in the same file.
- Every rotor file now records its class (``rotor_type``), and ``Rotor.load`` returns an instance
  of that class. ``Results.load`` therefore returns results with a ``MultiRotor``.
- The backlash, square-wave stiffness, mesh stiffness, mesh damping and orientation are kept.

Fix ``PointMass`` on a Shaft Node
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A ``PointMass`` on a shaft node got its own block of three degrees of freedom, which no element
connected, so the mass matrix was singular. It now uses the translational degrees of freedom of
the shaft node. External bearing-support nodes get one block per node, shared by the point masses
on it. A point mass on a node that is neither a shaft node nor a connected support node raises
``ValueError``. Contributed by Murillo Alvaro Borges de Sousa
(`#1463 <https://github.com/petrobras/ross/pull/1463>`_, issue
`#1461 <https://github.com/petrobras/ross/issues/1461>`_).

Fix ``hertz`` and the Units Loaded After ``ccp``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There were two defects (`#1474 <https://github.com/petrobras/ross/pull/1474>`_):

- **Import order.** ``ross/units.py`` loaded ``new_units.txt`` only while pint's application
  registry was still lazy. ``ccp`` sets up that registry, so after ``import ccp`` and then
  ``import ross``, ``Q_(150, "Hz")`` converted to 1432.4 rpm instead of 9000, and ``pkpk`` and
  ``cv`` were undefined. ROSS now loads its definitions into the existing registry.
- **The** ``hertz`` **unit.** ``Hz`` was defined as revolutions per second, but ``hertz``,
  ``kHz`` and ``kilohertz`` stayed 1/s, so ``Q_(150, "hertz")`` also gave 1432.4 rpm. All of them
  are now revolutions per second. The unit still prints as ``Hz``.

Documentation and Agent Tooling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Documentation
^^^^^^^^^^^^^

- A new theory page, ``holepattern_seal``, covers the hole-pattern seal. It describes the
  model and both gas closures, the base-flow and first-order equations, the boundary relations
  including the choked exit, the coefficient convention, the comparison with Childs and Wade
  (2004), an ideal against real-gas comparison for methane, and the limitations. Also from the same
  pull request:

  - ``tutorial_seals`` has a rewritten hole-pattern section, which also fixes the stage order and
    the ``frequency`` rows.
  - The references list CoolProp (Bell et al., 2014) and REFPROP (Lemmon et al., 2018)
    (`#1481 <https://github.com/petrobras/ross/pull/1481>`_).

- ``tutorial_bearings_part_3`` gives the AMB force law with a negative ``Ks``, and the equivalent
  stiffness and damping formulas are corrected
  (`#1464 <https://github.com/petrobras/ross/pull/1464>`_).
- ``tutorial_motor`` has a section on estimating the equivalent circuit, and the references list
  seven new sources (`#1470 <https://github.com/petrobras/ross/pull/1470>`_).
- The new ``compressor_example`` model is used throughout the docs
  (`#1465 <https://github.com/petrobras/ross/pull/1465>`_):

  - ``tutorial_analyses_part_1``, ``tutorial_analyses_part_2`` and ``tutorial_modeling_part_1``
    were executed again with the new model.
  - ``docs/user_guide/shaft_si.xls`` and ``shaft_us.xls`` hold the new rotor.

- ``ross/interface/README_INTERFACE.md`` describes the views button of the 3-D view
  (`#1472 <https://github.com/petrobras/ross/pull/1472>`_).

Agent Tooling
^^^^^^^^^^^^^

- ``AGENTS.md`` states when changing a test tolerance is not weakening the test
  (`#1454 <https://github.com/petrobras/ross/pull/1454>`_):

  - A tolerance tighter than the computation can deliver is a wrong expectation. Examples are
    platform round-off, a solver's stopping tolerance, ill-conditioning, or the digits a reference
    publishes.
  - Such a tolerance is sized from its source, kept well below the checked effect, the same on
    every platform, and the test must still fail on its defect.
  - Results moved by new code are never a reason to widen a tolerance.
  - The PR template and the ``ross-pr-create`` and ``ross-pr-review`` skills accept a physical or
    numerical reason.

- The ``seals`` recipe of the agent skill covers the new hole-pattern model
  (`#1481 <https://github.com/petrobras/ross/pull/1481>`_), and the ``motor`` recipe covers
  circuit estimation (`#1470 <https://github.com/petrobras/ross/pull/1470>`_).

Contributing and Release Process
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- New issue forms: a bug report with a ``BUG:`` title prefix and required description, code
  example and versions. The issue chooser also offers blank issues and a link to GitHub
  Discussions, and the Q&A and Ideas discussion categories get their own forms. Contributed by
  Vinicius Teixeira (`#1462 <https://github.com/petrobras/ross/pull/1462>`_).
- ``main`` became the 3.2.0 development line after ``v3.1.0`` was tagged and
  ``maintenance/3.1.x`` created (`#1458 <https://github.com/petrobras/ross/pull/1458>`_).
- Two import tests read a marker line instead of the whole standard output. A notice that CoolProp
  prints when REFPROP cannot be loaded no longer fails them
  (`#1500 <https://github.com/petrobras/ross/pull/1500>`_).
- ``tools/data/c123701``, a rotor model used only by ``tools/generate_comparison_plots.py``, was
  removed, and the comparison data of the compressor is in ``tools/data/compressor``
  (`#1465 <https://github.com/petrobras/ross/pull/1465>`_).

API Changes
~~~~~~~~~~~

The changes of the fluid-film bearings are listed in their own section above.

Results that change for existing models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Hole-pattern seals** (`#1481 <https://github.com/petrobras/ross/pull/1481>`_). Every
  coefficient and the leakage change, through ``HybridSeal`` too, because the formulation is
  corrected (see *Bug Fixes*). The leakage of the test seal moves by -3.3e-4. What the
  coefficients mean changes as well:

  - ``kxx``, ``kxy``, ``kyx`` and ``kyy`` are now the stiffness at the whirl frequency, and
    ``mxx``, ``mxy``, ``myx`` and ``myy`` are 0. Before, the element stored the zero-frequency
    stiffness, with the frequency dependence in the mass terms. Those mass terms are large and
    negative in the corrected model, and the rotor multiplies them by the eigenvalue squared
    rather than by the whirl frequency squared.
  - A one-dimensional table holds the coefficients at ``excitation_ratio * speed``, so
    frequency-dependent analyses need ``frequency=``.
  - ``cxx`` and ``cxy`` at a zero whirl frequency are the zero-frequency limit; they were 0.
  - The mass terms of ``HybridSeal`` are now 0.

- **Magnetic bearings in time-domain analyses**
  (`#1464 <https://github.com/petrobras/ross/pull/1464>`_):

  - ``run_time_response`` and ``AmbTimeResponse`` results change for every rotor with magnetic
    bearings, and again for any model with ``k_amp`` or ``k_sense`` different from 1.
  - ``compute_amb_controller`` returns the force ``ki * i - ks * x``, and the current includes
    ``k_amp``.
  - The parameters of the two AMB example rotors change.
  - Frequency-domain results do not change.

- **Real-gas labyrinth seals.** The stiffness and damping of ``gas_model="real"`` change
  (`#1479 <https://github.com/petrobras/ross/pull/1479>`_). Some choked results and some results
  that did not match the outlet pressure change as well
  (`#1480 <https://github.com/petrobras/ross/pull/1480>`_). See *Bug Fixes* for sizes.
- **MultiRotor with an unbalanced driven shaft.** On the driven shaft, the unbalance force whirls
  with that shaft, at its speed, and faults and the harmonic balance use the speed of their shaft
  (`#1426 <https://github.com/petrobras/ross/pull/1426>`_).
- ``compressor_example()`` **is a different rotor**
  (`#1465 <https://github.com/petrobras/ross/pull/1465>`_):

  - **What stays.** It is a generic six-stage centrifugal compressor with the structure of the old
    model: 91 shaft elements, 7 disks, 2 bearings and 12 seals, at the same nodes. It is still
    stable at its operating speeds.
  - **What changes.** It is about 5 % longer and 2.5 % lighter. Its diameters, disk properties,
    materials, and bearing and seal coefficients are different. Its natural frequencies and
    critical speeds are 7 % to 10 % lower.
  - ``ross/data/shaft_si.xls`` and ``shaft_us.xls`` hold the new rotor.
  - The docstring no longer cites Timbó and Ritto (2019), whose results the model no longer
    reproduces.

- **Units.** ``hertz``, ``kHz`` and ``kilohertz`` are revolutions per second, like ``Hz``. A
  value given in them reads 2π times larger in rad/s than before
  (`#1474 <https://github.com/petrobras/ross/pull/1474>`_).
- **Point masses on shaft nodes.** They no longer add degrees of freedom, so ``Rotor.ndof`` is
  smaller for such rotors and the mass matrix is no longer singular
  (`#1463 <https://github.com/petrobras/ross/pull/1463>`_).

Changed signatures
^^^^^^^^^^^^^^^^^^

- ``MultiRotor.__init__`` (`#1426 <https://github.com/petrobras/ross/pull/1426>`_):

  - ``mesh_damping_ratio`` moved from last to right after ``backlash``. ``min_w``, ``max_w`` and
    ``rated_w`` were added before ``tag``.
  - The 3.1 order was ``..., backlash, orientation_angle, position, tag, mesh_damping_ratio``. The
    3.2 order is ``..., backlash, mesh_damping_ratio, orientation_angle, position, min_w, max_w,
    rated_w, tag``.
  - **A 3.1 call that passes** ``orientation_angle`` **or a later argument by position now binds
    it to a different parameter.** Pass those arguments by keyword.

- ``HolePatternSeal.__init__`` has ``gas_model="ideal"`` right after ``gas_composition``.
  ``molar_mass`` and every later parameter move one position to the right, so pass them by keyword
  (`#1481 <https://github.com/petrobras/ross/pull/1481>`_).
- ``Rotor.concatenate`` takes ``tag=None``
  (`#1426 <https://github.com/petrobras/ross/pull/1426>`_).

Changed defaults and behavior
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``HolePatternSeal`` (`#1481 <https://github.com/petrobras/ross/pull/1481>`_):

  - **Failed speeds in a table.** A table of 3 or more speeds with a failed speed raises
    ``ValueError`` listing the failed speeds. Before, scipy raised "``y`` must contain only finite
    values". With 1 or 2 speeds, a failed speed warns and gets NaN, as before.
  - **Speeds in parallel.** Speeds are solved in a process pool only above 4 speeds; the limit was
    2. Up to 4 speeds therefore need no ``if __name__ == "__main__":`` guard.
  - **Input checks.** An unknown ``gas_model``, ``"real"`` without ``gas_composition`` and
    ``nz < 2`` raise ``ValueError``. Missing manual gas properties raise ``TypeError`` with a
    message.
  - **Solver settings.** ``relaxation_factor`` has no effect. ``max_iterations`` is the maximum
    number of trial integrations of the outlet search, ``first_step_size`` sets the first trial
    outlet pressure, and ``tolerance`` is a fraction of the pressure drop. No default changed.

- ``LabyrinthSeal`` raises ``RuntimeError`` when its leakage iteration does not converge within
  ``ross.seals.labyrinth_seal.MAX_LEAKAGE_ITERATIONS`` steps; before, it could loop forever
  (`#1480 <https://github.com/petrobras/ross/pull/1480>`_).
- ``MultiRotor`` and ``Mesh`` (`#1426 <https://github.com/petrobras/ross/pull/1426>`_):

  - **Configuration dictionaries.** ``backlash`` and ``square_varying_stiffness`` must have
    exactly their documented keys, even when ``enable`` is False, or ``ValueError`` is raised. For
    example, ``backlash={"enable": False}`` now raises.
  - **Removed.** ``MultiRotor.check_speed`` and the public ``Rotor.parameters`` dictionary. A
    ``DiskElement`` that is not in a rotor no longer has ``n_l`` and ``n_r``.
  - **New public attributes.** ``coupled_nodes``, ``coupled_tags``, ``gear_mesh_stiffness``,
    ``square_varying_stiffness``, ``backlash``, ``mesh_damping_ratio``, ``orientation_angle`` and
    ``position``. ``MultiRotor.K_coupling`` is set again, next to ``coupling_matrix``, with the
    same values.
  - **Comparisons and summaries.** ``Rotor.__eq__`` ignores ``tag``. The "Starting point" and
    "Total length" of ``SummaryResults`` come from the shaft node positions, which changes the
    table of rotors that do not start at 0.

- A ``PointMass`` on a node that is neither a shaft node nor a connected support node raises
  ``ValueError`` (`#1463 <https://github.com/petrobras/ross/pull/1463>`_).

Saved files
^^^^^^^^^^^

- Rotor files record ``rotor_type`` (`#1426 <https://github.com/petrobras/ross/pull/1426>`_).
  Rotor files saved by 3.1 load in 3.2. Measured on main for these notes, a ``rotor_example()``
  file saved by 3.2 fails to load in 3.1 with ``ModuleNotFoundError``. Saved ``MultiRotor`` files
  did not load in 3.1 either.
- A real-gas ``HolePatternSeal`` loaded from saved coefficients has ``gas=None``, so no table is
  built on load. Solving it raises a ``RuntimeError`` that says how to rebuild it
  (`#1481 <https://github.com/petrobras/ross/pull/1481>`_).

Interface Project Files and Scripts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``xj`` and ``yj`` of ``TiltingPad`` are read as typed (fractions of the clearance); an ``xj_unit``
  saved in an existing project is ignored. Ball, roller and magnetic bearing rows saved without
  units keep their meaning, since they are read in SI. In exported scripts, a quantity that ROSS
  does not convert is written as ``Q_(value, unit).m_as(base)``
  (`#1467 <https://github.com/petrobras/ross/pull/1467>`_).
- ``HybridSeal`` rows saved with the stage dictionaries still load, with the dictionaries in the
  advanced fields (`#1469 <https://github.com/petrobras/ross/pull/1469>`_).

Known Issues
~~~~~~~~~~~~

The fluid-film bearings have their own *Known Limitations* above.

- ``HolePatternSeal`` (`#1481 <https://github.com/petrobras/ross/pull/1481>`_):

  - **Model terms that did not change.** The stator friction is still a quarter of a smooth Moody
    wall, and the Sutherland viscosity of the ideal closure is still used outside its fit range.
    Changing the friction changes every result again and needs a hole-pattern stator law with its
    own validation.
  - **Against Childs and Wade (2004).** The frequency trends are reproduced, but not the
    magnitudes. Against values read from the figures, ``C`` is 1.11 to 1.70 times the measured
    value and ``k`` 1.93 to 2.75 times. The leakage is 1.76 to 1.86 times Table 5. The
    effective-damping crossover of the 0.20 mm seal is at 186 Hz, against 95 Hz in the paper. Most
    of the excess comes from the stator friction.
  - **Real-gas closure.**

    - It has not been compared with dense-gas seal measurements.
    - For mixtures, the phase check samples the 41 pressure nodes of the table, so a narrow
      two-phase band between nodes can be missed.
    - The table floor is conservative.
    - Near a critical point the table loses accuracy: ethane's sound speed is off by up to 22 %
      there.
    - A ``HybridSeal`` with ``gas_model="real"`` rebuilds the table at every step of its
      interface-pressure search: 27 s against 6.7 s for air, and minutes for a natural gas.

  - **Cell compressibility.** Whether the cells are isentropic or isothermal is an open question;
    isothermal cells would give lower stiffness and damping.
  - **Near an exit choke.** Within about 1e-3 of the choked outlet station pressure, the unchoked
    flow switches to the choked exit row. At the edge of that band, ``kxx`` jumps by 1.8e-3 and
    ``cxy`` by 4.7e-3 on the 0.20 mm air seal.

- ``LabyrinthSeal`` (`#1479 <https://github.com/petrobras/ross/pull/1479>`_,
  `#1480 <https://github.com/petrobras/ross/pull/1480>`_):

  - **Perfect-gas forms in the real-gas closure.** Two remain in the linearised equations: the
    leakage derivatives, and the pressure derivative of the wall shear.
  - **Critical ratio of the perfect gas.** With kinetic-energy carry-over, the critical ratio of
    ``IdealGas`` is not the sonic point.
  - **Two-phase states.** The real-gas table is not checked for them.

- **Magnetic bearings** (`#1464 <https://github.com/petrobras/ross/pull/1464>`_):

  - Two-value ``k_amp`` and ``k_sense`` work in the time-domain paths only.
  - The sensor angle is fixed at 45 deg in ``AmbTimeResponse``, and defaults to 45 deg in the
    Newmark controller, instead of following ``sensors_axis_rotation``.
  - With a variable speed, the Newmark path counts the AMB forces twice.
  - ``run_ucs`` fails with a negative bearing stiffness.
  - The ISO 14839-3 zones in section 6 of ``tutorial_bearings_part_3`` are incomplete.

- **Motor circuit estimation** (`#1470 <https://github.com/petrobras/ross/pull/1470>`_). The
  estimates are consistent with the rated data, but they are not necessarily close to the true
  circuit. Against published reference circuits, the RMS parameter error is large for several
  motors, about 70 % for ASDP on the Chapman motor.
- **Motor coupling.** The open problems of ``Rotor.run_with_motor`` listed in the 3.1.0 notes are
  not addressed in this release.

Contributors
~~~~~~~~~~~~

This release includes contributions from: @jguarato, @murilloabs, @mariac-souza, @leonardocabr,
@ViniciusTxc3, @raphaeltimbo
