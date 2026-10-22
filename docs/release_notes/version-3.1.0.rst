Version 3.1.0
-------------

ROSS 3.1.0 adds a three-phase induction motor model that drives the rotor time response
(``MotorElement`` and ``Rotor.run_with_motor``), a mesh convergence analysis with shaft
refinement methods (``Rotor.run_mesh_convergence``, ``Rotor.refine`` and ``Rotor.refine_mesh``),
gear mesh damping for ``MultiRotor``, time windows in the time-response plots, a helper that
builds speed ranges from their end points, and a 3-D rotor view and new modeling tools in the
graphical interface.

Although this is a minor release, it breaks code written for 3.0. ``CoAxialRotor``,
``coaxrotor_example``, ``Rotor.convergence`` and ``ConvergenceResults`` were removed and
``MultiRotor.K_coupling`` was renamed ``coupling_matrix``, all without a deprecation period. The
default mesh damping of backlash models changed from 0.07 to 0, the new ``motor_element``
parameter of ``Rotor`` moves ``min_w`` and the following parameters one position to the right,
and the frequency-response plots read off-diagonal pairs as the response at ``out`` to a force
at ``inp``, the transpose of what they plotted in 3.0. ROSS 3.1 supports Python 3.13 to 3.15;
Python 3.12 is no longer supported. Read *API Changes* below before upgrading.

The following enhancements and bug fixes were implemented for this release:

Enhancements
~~~~~~~~~~~~

Three-Phase Induction Motor: ``MotorElement`` and ``run_with_motor``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The new ``ross.motors`` subpackage models a three-phase induction motor. Contributed by Jessica
Guarato and Bruno Zambolini Vicente (`#1296 <https://github.com/petrobras/ross/pull/1296>`_):

- ``MotorElement(n, power_rated, voltage_rated, speed_rated, stator_resistance,
  rotor_resistance, stator_reactance, rotor_reactance, mutual_reactance, Ip_motor, ...)`` is the
  motor at node ``n``; ``rs.motor_example()`` returns one. The motor can be simulated on its own
  with ``run_direct_on_line()`` (direct-on-line start, with optional grid harmonics and voltage
  unbalance), ``run_with_inverter_vf()`` (V/f inverter) or ``run_with_inverter_foc()``
  (field-oriented-control inverter). The time loops are fourth-order Runge-Kutta steps compiled
  with numba. The results are a ``MotorResponseResults``, with ``plot_torque()``,
  ``plot_speed()``, ``plot_phase_currents()``, ``plot_phase_voltages()`` and
  ``plot_line_voltages()``, and ``PhaseResults`` plots a phase signal and its DFFT. Both classes
  are defined in ``ross.motors.results`` and exported from ``ross``, like the other results
  classes (`#1445 <https://github.com/petrobras/ross/pull/1445>`_).
- A rotor carries at most one motor, given as ``Rotor(..., motor_element=motor)`` or added with
  ``rotor.add_elements([motor])``, which replaces an existing motor. The motor has zero mass,
  stiffness, damping and gyroscopic matrices, so it does not change the rotor matrices. It is
  listed in ``Rotor.df``, drawn by ``plot_rotor``, kept and renumbered by ``add_nodes``,
  ``refine`` and ``refine_mesh`` (`#1439 <https://github.com/petrobras/ross/pull/1439>`_) and
  saved and loaded with the rotor. ``Rotor.concatenate`` keeps the motor of the last rotor that
  has one. A ``MultiRotor`` takes the motor of its driving or its driven rotor; when both carry one,
  ``MultiRotor`` raises ``Warning`` (an exception).
- ``Rotor.run_with_motor(t, node, unbalance_magnitude, unbalance_phase, drive_mode, ...)``
  simulates the motor with ``drive_mode`` ``"DOL"``, ``"VFD_VF"`` or ``"VFD_FOC"``, applies the
  electric torque minus the load torque to the torsional degree of freedom of the motor node,
  computes the unbalance force from the motor speed history and integrates the rotor with the
  Newmark method, adding the torque-dependent stiffness ``ShaftElement.Ktq() * torque`` of the
  shaft elements at the motor node. With the default ``steady_state=True`` only the last third
  of the interval after ``load_torque_entrance_time`` is integrated. It returns a
  ``TimeResponseResults`` whose ``motor_results`` attribute holds the motor simulation.
- Also added: ``ShaftElement.Ktq()``, the torque stiffness matrix per unit torque, which
  ``ShaftElement.K()`` now uses for its ``torque`` term (the matrix values are unchanged);
  ``Ktq`` and ``torque`` keyword arguments of the time integration with a speed array;
  ``ross.units.format_unit()``, which formats a unit as an axis label (``"N*m"`` becomes
  ``"N⋅m"``); ``ross.utils.downsample_figure()`` and ``ross.utils.limit_data_range()``; and the
  unit ``cv`` (metric horsepower).
- The new tutorial ``tutorial_motor`` ("Tutorial - Three-Phase Induction Motor") is in the user
  guide.

Mesh Convergence and Shaft Refinement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``Rotor.run_mesh_convergence(rtol=1e-3, frequencies=6, strategy="cheapest", speed=0,
max_elements=1000)`` finds a shaft discretization whose lowest ``frequencies`` natural
frequencies are within ``rtol`` of a converged reference. The reference is found by halving the
element length until the frequencies change less than ``rtol / 10``. The search covers a uniform
family of meshes and, with ``strategy="cheapest"``, a graded family that gives each element a
fixed fraction of the local bending wavelength, so thin sections get shorter elements; it returns
the discretization with the fewest elements. The rotor is not modified. The method returns the
new ``MeshConvergenceResults``, which holds the refined rotor (``results.rotor``), the
subdivisions of each shaft interval, the natural frequencies and their errors. Its ``plot()``
draws the maximum error against the number of shaft elements, and it round-trips ``save()`` /
``load()``. It replaces ``Rotor.convergence``; see *API Changes*. The analysis has a recipe in the
agent skill (``mesh_convergence``). Contributed by Vinicius Vianna
(`#1420 <https://github.com/petrobras/ross/pull/1420>`_).

Two methods return a refined copy of the rotor:

- ``Rotor.refine(subdivisions)`` splits each shaft interval (an int for all intervals, or one int
  per interval) into equal elements built with ``ShaftElement.copy``. Tapered diameters are
  interpolated along the length, layered shafts are split layer by layer, couplings are not
  split, and bearing housings (``n_link``), disk ``n_l`` / ``n_r`` and the AMB ``sensor_node`` are
  renumbered (`#1420 <https://github.com/petrobras/ross/pull/1420>`_).
- ``Rotor.refine_mesh(max_ld=0.5, min_ld=0.1, parts=None)`` cuts every shaft element longer than
  ``max_ld`` times its larger outer diameter into as few equal parts as keep each part at or
  under that ratio (API RP 684, section 1.5.2.1), or into ``parts`` equal parts, through
  ``add_nodes``. Elements that share a node pair are cut together, couplings are not cut, and a
  ``UserWarning`` lists the elements shorter than ``min_ld`` diameters. As with ``add_nodes``, the
  tags of the shaft elements are not kept when any element is cut.
  ``MultiRotor.refine_mesh(max_ld=0.5, min_ld=0.1, parts=None)`` refines the driving and driven
  rotors separately. Contributed by Leonardo Cabral
  (`#1429 <https://github.com/petrobras/ross/pull/1429>`_); ``parts`` of ``MultiRotor.refine_mesh``
  was added in `#1439 <https://github.com/petrobras/ross/pull/1439>`_.

``refine``, ``refine_mesh`` and ``add_nodes`` number the new nodes the same way, including bearing
housings, disk ``n_l`` / ``n_r``, the AMB ``sensor_node`` and the motor
(`#1439 <https://github.com/petrobras/ross/pull/1439>`_).

``refine`` and ``run_mesh_convergence`` work on ``Rotor`` only: on a ``MultiRotor`` or another
subclass of ``Rotor`` they raise ``NotImplementedError``.

Gear Mesh Damping for ``MultiRotor``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``MultiRotor`` accepts ``mesh_damping_ratio`` (dimensionless, a float or a quantity such as
``Q_(7, "percent")``, default 0.0). The mesh damping coefficient
:math:`c_m = 2 \zeta \sqrt{k_m M_{eq}}` (Yi et al., 2019) multiplies the same coupling matrix as
the mesh stiffness. Without backlash it is added to ``MultiRotor.C()`` by the new
``MultiRotor.C_mesh()``, so modal, Campbell, frequency, forced and time responses include it; with
backlash the same ratio enters the nonlinear contact force. When the mesh stiffness varies in
time, :math:`c_m` follows the stiffness of the current step. The equivalent mass
:math:`M_{eq}` is the new ``Mesh.M_eq``, computed from the polar inertias and base radii of the
gears, so :math:`\zeta` is the damping ratio of the torsional mesh mode. The MultiRotor tutorial
describes the model. Contributed by Murillo Alvaro Borges de Sousa
(`#1427 <https://github.com/petrobras/ross/pull/1427>`_).

Time Windows in Time-Response Plots
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``TimeResponseResults.plot_1d``, ``plot_2d`` and ``plot_3d`` accept the keyword options
``t_initial`` and ``t_final``, which plot a window of the stored response without running the
simulation again, and ``one_revolution=True``, which plots one shaft revolution: the last one, or
the one that starts at ``t_initial`` or ends at ``t_final``. Contributed by Murillo Alvaro Borges
de Sousa (`#1416 <https://github.com/petrobras/ross/pull/1416>`_, issue
`#1415 <https://github.com/petrobras/ross/issues/1415>`_):

- Plain numbers are read in the ``time_units`` of ``plot_1d`` and in seconds by ``plot_2d`` and
  ``plot_3d``; quantities keep their units. A window outside the stored interval,
  ``t_initial >= t_final`` or a window with fewer than 10 samples
  (``ross.results.MIN_WINDOW_SAMPLES``) raise ``ValueError``; the message for a short window gives the time step that would be enough.
  With ``one_revolution=True`` this needs a time step of at most ``2 * pi / (9 * speed)`` s
  (`#1443 <https://github.com/petrobras/ross/pull/1443>`_). A plain window needs both
  ``t_initial`` and ``t_final``, and ``one_revolution=True`` takes at most one of them
  (``ValueError`` otherwise).
- The revolution length is computed from the speed the results now store
  (``TimeResponseResults.speed``); a speed that varies along the response is integrated.
  ``one_revolution=True`` raises ``ValueError`` when the results hold no speed.
- A windowed ``plot_1d`` sets the x-axis range to the window, and windowed ``plot_2d`` and
  ``plot_3d`` mark the initial and final points.
- ``TimeResponseResults``, ``AmbTimeResponseResults`` and ``BacklashResults`` take and store the
  run speed (``speed=None``), and ``data_time_response`` accepts ``t`` and ``yout`` arrays in place
  of the stored ones.

Speed Ranges from End Points
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``ross.utils.speed_range_from_endpoints(speed_start, speed_stop, num_points=None,
points_factor=20)`` returns an evenly spaced speed array in rad/s for ``run_campbell``,
``run_freq_response`` and ``run_unbalance_response``. Each end point can be given in its own
unit (``Q_(0, "Hz")``, ``Q_(3600, "RPM")``). Without ``num_points`` the count is
``ceil(points_factor * sqrt(span))``, with the span in rad/s, limited to [20, 2000], so a narrow
range gets a finer step than a wide one; about 4 is enough as ``points_factor`` for a Campbell
diagram. It raises ``ValueError`` when ``speed_stop <= speed_start`` or ``num_points < 2``. The
three ``run_*`` docstrings point to it; their signatures are unchanged. The function is in
``ross.utils`` and is not exported from the top-level ``ross`` namespace. Contributed by Vinicius
Vianna (`#1414 <https://github.com/petrobras/ross/pull/1414>`_).

Graphical Interface
~~~~~~~~~~~~~~~~~~~

The graphical interface (``ross-interface``) gained the following, contributed by Leonardo
Cabral:

- **3-D rotor view.** The modeling screen shows a 3-D view beside the 2-D ROSS figure, drawn
  from a scene the server builds from the assembled rotor. A right-button menu splits,
  discretizes, edits, copies, hides, adds and deletes elements; the view shows node numbers and
  dimensions, has a measuring tape, a legend that hides categories and an optional motor bench.
  three.js r186 is vendored under ``ross/interface/frontend/vendor/three/`` and loaded the first
  time the view opens (`#1429 <https://github.com/petrobras/ross/pull/1429>`_).
- **Discretization and filters.** Shafts can be discretized by length-to-diameter ratio
  (API RP 684), into a number of parts, or by ``Rotor.run_mesh_convergence`` (refused for rotors
  coupled by gears), and the element list can be filtered by material, model, node, tag or field
  value (`#1429 <https://github.com/petrobras/ross/pull/1429>`_).
- **Geometry and probes.** Disks and gears can be entered by their dimensions
  (``DiskElement.from_geometry``, ``GearElement.from_geometry``). A new probe category places
  radial and axial probes on the model; analysis probe tables and the frequency-response output
  can name these probes, and their tags reach the chart legends
  (`#1429 <https://github.com/petrobras/ross/pull/1429>`_).
- **Modeling tools.** Rotor concatenation (``Rotor.concatenate``) and shaft splitting
  (``Rotor.add_nodes``) have their own dialogs; Ctrl+Z / Ctrl+Y undo and redo; several elements
  can be selected and copied or deleted together
  (`#1419 <https://github.com/petrobras/ross/pull/1419>`_).
- **Rotor figure and layout.** The screen keeps the geometry of the figure ROSS returns and
  applies a declared vertical exaggeration; the layout has a minimum width instead of reflowing
  in narrow windows (`#1419 <https://github.com/petrobras/ross/pull/1419>`_).
- **Projects.** Project files are loaded as written, so a saved ``MultiRotor`` reopens as one
  instead of as an empty rotor, and files with old Portuguese key names still load. The
  MultiRotor dialog shows where each rotor's gears are, fills the coupled nodes when only one
  choice exists and builds the ``MultiRotor`` on the server before saving, so a ROSS error appears
  while the fields are still on screen (`#1419 <https://github.com/petrobras/ross/pull/1419>`_).
- **Materials.** Spaces in material names become underscores, because ``rs.Material`` refuses
  spaces, and an element that names a material the project does not have is refused with "This
  rotor has no material by that name" (HTTP 400); before, it silently got the first material
  (`#1419 <https://github.com/petrobras/ross/pull/1419>`_).
- **Analyses.** The frequency-response and unbalance forms have a *Steps* field (default 50;
  it was fixed at 50), and the Campbell, clearance, frequency-response and unbalance sweeps refuse
  fewer than 2 steps. The speed fields are number fields without bounds instead of sliders. An
  empty nodes field of the modal orbit plot draws every node, and an unknown node is refused with
  the list of the rotor's nodes. Exceptions that ROSS raises on purpose return HTTP 400 with their
  message alone; before, those other than ``ValueError`` were a 500 shown as "Unexpected error
  (<type>): <message>" (`#1419 <https://github.com/petrobras/ross/pull/1419>`_).

The Advanced section of the MultiRotor form has a *Mesh Damping Ratio [-]* field (default 0.0),
passed to ``rs.MultiRotor`` as ``mesh_damping_ratio``, written into exported scripts and kept in
the project file. An empty field, or a project saved without it, leaves the argument out, so the
library default applies. A negative ratio is refused with "The mesh damping ratio must be zero or
positive" (`#1444 <https://github.com/petrobras/ross/pull/1444>`_).

Bug Fixes
~~~~~~~~~

Fix Double Unit Conversion in ``plot_1d`` and ``plot_dfft``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``TimeResponseResults.plot_1d()`` and ``plot_dfft()`` converted to ``displacement_units`` the
response that ``data_time_response()`` had already converted, so any unit other than ``"m"``
scaled the plot by the conversion factor a second time (1e3 too large in ``"mm"``, 1e6 in
``"um"``). Plots in ``"m"`` are unchanged. Contributed by Gabriel Sabino
(`#1418 <https://github.com/petrobras/ross/pull/1418>`_, issue
`#1417 <https://github.com/petrobras/ross/issues/1417>`_).

Fix NaN AMB Time Response with scipy 1.18 and Later
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With scipy 1.18 and later, ``run_time_response`` on AMB rotors whose controllers are high-order
transfer functions (for example ``rotor_example_amb_complex_controllers()``) returned NaN for every
output: the companion-form realization of the controllers gave a closed-loop matrix with a 1-norm
around 1e26, and the matrix exponential inside ``scipy.signal.lsim`` overflowed.
``AmbTimeResponse.build_matrices`` (``ross.bearings.magnetic.amb_time_response``) now balances
``[[A_cl, B_cl], [C_cl, 0]]`` with ``scipy.linalg.matrix_balance``, which rescales only the state;
the new attribute ``state_scaling`` holds the factors (physical state = ``state_scaling * z``), the
initial condition is converted, and inputs and outputs keep their physical units. Contributed by
Jessica Guarato (`#1422 <https://github.com/petrobras/ross/pull/1422>`_, issue
`#1421 <https://github.com/petrobras/ross/issues/1421>`_).

Fix ``MultiRotor`` Rebuilds Dropping the Mesh Settings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``MultiRotor.add_nodes`` and ``add_elements`` rebuilt the model without its backlash settings and
rebuilt the square-wave stiffness from the mesh attributes. The rebuilt model now keeps the
``backlash`` and ``square_varying_stiffness`` dictionaries and the ``gear_mesh_stiffness`` given to
the constructor (`#1427 <https://github.com/petrobras/ross/pull/1427>`_).

Fix ``ModalResults.plot_orbit`` Without Nodes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``ModalResults.plot_orbit(mode)`` with the default ``nodes=None`` returned an empty figure. It now
plots the orbit of every node of the mode shape (`#1419 <https://github.com/petrobras/ross/pull/1419>`_).

Fix Frequency-Response Plots Reading the Transposed Entry
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``run_freq_response`` stores :math:`H = Z^{-1}`, so ``freq_resp[out, inp, i]`` is the response
at ``out`` to a unit force at ``inp``. ``FrequencyResponseResults.plot_magnitude``,
``plot_phase``, ``plot_polar_bode`` (and so ``plot``) and ``ST_Rotor.run_freq_response`` read
``freq_resp[inp, out]``. On a gyroscopic rotor at speed the cross x/y pairs had the phase off by
:math:`\pi`, and with cross-coupled bearings (``kxy != -kyx``) magnitude and phase were wrong on
every off-diagonal pair. They now read ``freq_resp[out, inp]``; the stored array is unchanged and
diagonal pairs plot as before. The probe outputs of the interface frequency-response analysis,
and the ``probe_output`` function written into exported scripts, projected the force axis to match
the old reading, which mirrored a radial probe at angle :math:`a` to :math:`-a`. They now project
the response axis (`#1441 <https://github.com/petrobras/ross/pull/1441>`_,
`#1449 <https://github.com/petrobras/ross/pull/1449>`_).

Fix ``add_nodes`` and ``refine_mesh`` Moving AMB Sensors and Disk Nodes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``Rotor.add_nodes``, and through it ``refine_mesh``, renumbered the shaft but left the
``sensor_node`` of magnetic bearings and the ``n_l`` / ``n_r`` of disks at the old numbers, so a
collocated AMB became non-collocated and AMB results were wrong. They now use the renumbering of
``Rotor.refine``. ``add_nodes`` also renumbers the motor's ``n_l`` / ``n_r`` with its ``n``, and
``refine`` keeps the motor, which it dropped
(`#1439 <https://github.com/petrobras/ross/pull/1439>`_).

Fix ``check_units`` with pint 0.26.2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

pint 0.26.2 parses a string magnitude as an expression and raises ``AssertionError`` for
``Q_("major", "rad")``. ``check_units`` relied on a ``TypeError`` there to return option strings
unchanged, so calls such as ``Probe(3, "major")`` and ``run_campbell(frequency_type="wd")``
failed. ``check_units`` now returns ``str`` arguments without converting them, and
``Probe.get_label`` checks for a string angle
(`#1438 <https://github.com/petrobras/ross/pull/1438>`_).

Fix the Mayes and Gasch Crack Models with pandas 3.1
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``Crack`` read the Papadopoulos compliance table with ``pd.eval`` on a whole column, which
pandas 3.1 refuses, so ``crack_model="Mayes"`` and ``"Gasch"`` failed. The table is now parsed
with ``json.loads``; the coefficients are unchanged
(`#1456 <https://github.com/petrobras/ross/pull/1456>`_).

Fix the Version Warning of ``compressor_example``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``rs.compressor_example()`` reads a TOML file shipped with ROSS and emitted the ``UserWarning``
"File was created with ROSS 3.0.0.dev0, but current version is ..." at every call once the minor
version changed. It now builds the rotor without the version check; ``Rotor.load`` warns for
other files as before (`#1440 <https://github.com/petrobras/ross/pull/1440>`_).

Documentation and Agent Tooling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Documentation
^^^^^^^^^^^^^

- Fourteen notebooks whose stored figures still showed the frame used before the right-handed
  frame of 3.0.0 were re-executed: the getting-started overview, the modeling, analyses and
  MultiRotor tutorials, and examples 5, 16, 22, 24, 28, 29, 30 and 31. Example 31 runs again
  (``frequencies=8`` moved from ``CampbellResults.plot`` to ``run_campbell``, and ``run_modal``
  asks for ``num_modes=16``).
  ``docs/user_guide/rotor2.toml`` was regenerated in the 3.x save format, and the README lost its
  emojis and has the codecov badge in Markdown and the corrected name of the Federal University
  of Uberlândia (`#1412 <https://github.com/petrobras/ross/pull/1412>`_).
- ``tutorial_analyses_part_2`` and the ``run_time_response`` docstring describe ``t_initial``,
  ``t_final`` and ``one_revolution`` (`#1416 <https://github.com/petrobras/ross/pull/1416>`_), and
  ``tutorial_multirotor`` describes the mesh damping model, with Visnadi (2022) added to the
  references (`#1427 <https://github.com/petrobras/ross/pull/1427>`_).
- The API reference lists ``Rotor.run_mesh_convergence``, ``Rotor.refine`` and
  ``MeshConvergenceResults`` instead of ``ConvergenceResults``
  (`#1420 <https://github.com/petrobras/ross/pull/1420>`_).
- ``CoAxialRotor`` and ``coaxrotor_example`` were dropped from the API reference, and example 25
  is now titled "Overhung Rotor on Flexible Supports"; it is Friswell's Example 6.8.1 and never
  used ``CoAxialRotor`` (`#1434 <https://github.com/petrobras/ross/pull/1434>`_).
- ``ross/interface/README_INTERFACE.md`` describes the 3-D view
  (`#1429 <https://github.com/petrobras/ross/pull/1429>`_).
- The API reference lists ``MotorElement``, ``motor_example``, ``Rotor.run_with_motor``,
  ``Rotor.run_amb_non_collocation``, ``Rotor.refine_mesh``, ``MotorResponseResults``,
  ``PhaseResults``, ``AmbNonCollocationResults``, ``SourceAC``, ``InverterVF``, ``InverterFOC``
  (in a new "Motor Drives" section) and ``ross.utils.speed_range_from_endpoints``; the entries
  ``rotor_amb_example`` and ``concatenate_rotor``, which name nothing in ``ross``, gave way to the
  three ``rotor_example_amb_*`` functions. The docstrings of ``run_with_motor``,
  ``run_direct_on_line``, ``run_harmonic_balance_response``, ``run_level1``,
  ``run_misalignment`` and ``MotorResponseResults.sample_at`` render without docutils errors. The
  ``MotorElement`` docstring and ``tutorial_motor`` state that the constructor's ``voltage_net``
  and ``frequency_net`` do not set the direct-on-line supply and that the ``steady_state=True``
  window starts from rest (`#1445 <https://github.com/petrobras/ross/pull/1445>`_).
- ``tutorial_stochastic``, ``tutorial_motor``, ``tutorial_bearings_part_3`` and
  ``tutorial_modeling_part_1`` run without errors or side effects: the stochastic unbalance
  response draws ``var_size`` samples and its sweeps start at 1 rad/s, the figures use the
  notebook renderer instead of ``renderer="svg"`` (which needs kaleido), and the material example
  saves and removes ``my_steel`` instead of overwriting the packaged Steel entry. ``example_2``,
  ``tutorial_units``, ``tutorial_bearings_part_1``, ``tutorial_bearings_part_2`` and
  ``tutorial_seals`` were executed again, and the tqdm ``IProgress not found`` stderr output was
  removed from 18 notebooks (`#1446 <https://github.com/petrobras/ross/pull/1446>`_). The stored
  version warning of ``compressor_example()`` was removed from ``tutorial_analyses_part_1`` and
  ``tutorial_analyses_part_2`` (`#1440 <https://github.com/petrobras/ross/pull/1440>`_).
- The ``run_time_response`` doctest and the one-revolution example of
  ``tutorial_analyses_part_2`` use a time step that puts about 50 samples in each revolution
  (`#1443 <https://github.com/petrobras/ross/pull/1443>`_).
- The installation page, ``AGENTS.md`` and ``ross/interface/README_INTERFACE.md`` give Python 3.13
  as the oldest supported version, and the installation page describes the ``downsample`` extra
  (`#1447 <https://github.com/petrobras/ross/pull/1447>`_).

Agent Tooling
^^^^^^^^^^^^^

- ``AGENTS.md`` is the only instruction file for coding agents; ``CLAUDE.md`` was removed and its
  content moved there. It holds the usage reference (analysis methods with their results classes,
  plots and recipes, example rotors and units) and the development rules: the five criteria a
  contribution must meet, scope, architecture, tests, writing, pull requests and reviewing,
  including that agents never post to GitHub without explicit approval. The new
  ``.github/PULL_REQUEST_TEMPLATE.md`` has the sections Summary, Scope, Design decisions for
  review, API changes, Evidence, Changes to existing tests or reference data, Documentation and
  Where to look, and ``CONTRIBUTING.md`` points to it and to ``AGENTS.md``
  (`#1432 <https://github.com/petrobras/ross/pull/1432>`_).
- Two agent skills for contributors and reviewers: ``ross-pr-create``
  (``.agents/skills/ross-pr-create/``) checks a change against ``AGENTS.md``, runs the checks and
  drafts the PR text from the template; ``ross-pr-review`` (``.agents/skills/ross-pr-review/``)
  reviews a pull request, runs correctness probes on its head and base and can write an HTML
  report, and posts nothing to GitHub until the person who asked approves. Both use the new
  ``tools/pr_facts.py`` (``--base``, ``--head``, ``--json``), which compares two revisions with git
  and ``ast`` only, without running the PR code. ``.claude/skills/`` links to both skills, CI runs
  ``pytest tools/tests``, and ruff also checks ``tools/`` and ``.agents/``. The ``ross`` package is
  unchanged (`#1433 <https://github.com/petrobras/ross/pull/1433>`_).
- The review report of ``ross-pr-review`` turns its page background dark in dark mode, and a pipe
  escaped as ``\|`` or written inside a code span stays in its table cell
  (`#1436 <https://github.com/petrobras/ross/pull/1436>`_).
- The agent skill has a ``motor`` recipe, following ``tutorial_motor``, and ``SKILL.md`` indexes
  every ``Rotor.run_*`` method, with ``help(...)`` rows for ``run_harmonic_balance_response``,
  ``run_amb_sensitivity`` and ``run_amb_non_collocation``, which have no recipe yet. The
  ``time_response`` recipe shows the plot windows, and the Campbell, unbalance and frequency
  response recipes use ``speed_range_from_endpoints``. ``AGENTS.md`` lists ``run_with_motor``,
  ``motor_example`` and ``ross/motors/`` (`#1445 <https://github.com/petrobras/ross/pull/1445>`_).
  The ``frequency_response`` recipe gives ``freq_resp[out, inp, :]`` as the plotted entry
  (`#1441 <https://github.com/petrobras/ross/pull/1441>`_).

Release Process
^^^^^^^^^^^^^^^

- The Release workflow (``publish-to-pypi.yml``) runs ``tools/check_release_version.py`` between
  the build and the upload. It stops the release unless the tag is ``v`` followed by the canonical
  version of every built wheel and sdist, so a tag pushed before the version bump no longer
  uploads a ``.dev0`` package; ``CONTRIBUTING.md`` describes the check
  (`#1448 <https://github.com/petrobras/ross/pull/1448>`_).
- CI tests Python 3.13, 3.14 and 3.15, and the release is built on 3.13
  (`#1447 <https://github.com/petrobras/ross/pull/1447>`_).
- Two tests no longer depend on the platform or on what other packages print, and the Lint
  step and the release-check test pass on Windows
  (`#1453 <https://github.com/petrobras/ross/pull/1453>`_,
  `#1455 <https://github.com/petrobras/ross/pull/1455>`_). 3.1.0rc1 was the release candidate
  (`#1451 <https://github.com/petrobras/ross/pull/1451>`_).

API Changes
~~~~~~~~~~~

Removed classes, methods and attributes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

None of these has a deprecated alias: the old name raises ``AttributeError`` (``ImportError`` for
``from ross import ...``).

===================================================  ==========================================================
Removed                                              Replacement
===================================================  ==========================================================
``CoAxialRotor``                                     none; stay on 3.0.x (where only part of the analyses worked, see #1430) until coaxial support returns
``coaxrotor_example()``                              none
``Rotor.convergence(n_eigval, err_max)``             ``Rotor.run_mesh_convergence(rtol, frequencies, ...)``
``MultiRotor.convergence`` (inherited)               none; ``run_mesh_convergence`` and ``refine`` raise ``NotImplementedError`` on ``MultiRotor``
``Rotor.error_arr``                                  ``MeshConvergenceResults.errors`` / ``.error``
``ConvergenceResults``                               ``MeshConvergenceResults``
``MultiRotor.K_coupling``                            ``MultiRotor.coupling_matrix`` (same values)
===================================================  ==========================================================

``CoAxialRotor`` was removed because about half of the analyses failed on a coaxial rotor in 3.0
(among them ``run_static``, ``run_ucs``, ``run_level1``, ``convergence`` and ``add_nodes``), both
shafts ran at one speed, and saved coaxial rotors could not be loaded
(`#1434 <https://github.com/petrobras/ross/pull/1434>`_, issue
`#1430 <https://github.com/petrobras/ross/issues/1430>`_). The rework in
`#1423 <https://github.com/petrobras/ross/pull/1423>`_ was closed, and coaxial support may return
in a later release. ``MultiRotor`` is unchanged.

``Rotor.convergence`` (`#1420 <https://github.com/petrobras/ross/pull/1420>`_) checked one
eigenvalue, doubled the number of elements at every step and replaced the rotor's contents in
place. ``run_mesh_convergence`` checks the lowest ``frequencies`` natural frequencies, so
``n_eigval=k`` is covered by ``frequencies=k + 1``. ``err_max`` (default 1e-2) bounded the change
between two successive doublings; ``rtol`` (default 1e-3) bounds the error against a converged
reference. On a ``MultiRotor``, ``convergence`` rebuilt the model as a plain ``Rotor`` without the
gear mesh. The refined rotor is ``results.rotor``, and the rotor the method is called on is not
modified. ``ConvergenceResults`` (``el_num``, ``eigv_arr``, ``error_arr``) gave way to
``MeshConvergenceResults`` (``elements``, ``wn``, ``wn_reference``, ``errors``, ``error``, ...).

``K_coupling`` was renamed because the matrix now multiplies both the mesh stiffness and the mesh
damping (`#1427 <https://github.com/petrobras/ross/pull/1427>`_).

Changed signatures
^^^^^^^^^^^^^^^^^^

- ``Rotor.__init__`` has ``motor_element=None`` as its fifth parameter, after
  ``point_mass_elements``. ``min_w``, ``max_w``, ``rated_w``, ``modal_damping_ratio``,
  ``default_damping_ratio``, ``alpha``, ``beta`` and ``tag`` each move one position to the right,
  so a call that passes ``min_w`` or a later argument by position now binds it to
  ``motor_element``. Pass those arguments by keyword
  (`#1296 <https://github.com/petrobras/ross/pull/1296>`_).
- ``ross.multi_rotor.mesh.Backlash.__init__`` has a new required parameter ``M_eq`` between
  ``damping_ratio`` and ``module``, which ``Mesh`` now passes. Only code that builds ``Backlash``
  directly is affected (`#1427 <https://github.com/petrobras/ross/pull/1427>`_).
- Parameters added at the end, so positional calls keep working: ``MultiRotor(...,
  mesh_damping_ratio=0.0)`` (`#1427 <https://github.com/petrobras/ross/pull/1427>`_);
  ``TimeResponseResults``, ``AmbTimeResponseResults`` and ``BacklashResults`` take ``speed=None``,
  and ``data_time_response`` takes ``t=None`` and ``yout=None``
  (`#1416 <https://github.com/petrobras/ross/pull/1416>`_).
- ``Shape.plot_orbit(nodes, fig=None)`` became ``Shape.plot_orbit(nodes=None, fig=None)``; with
  ``nodes=None`` it plots every node (`#1419 <https://github.com/petrobras/ross/pull/1419>`_).

Changed defaults and behavior
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Mesh damping of backlash models.** The default ``damping_ratio`` of ``Mesh`` changed from
  0.07 to 0.0, and ``MultiRotor`` now passes its ``mesh_damping_ratio`` (default 0.0) to it. In 3.0
  ``MultiRotor`` passed no ratio, so a backlash model built with default arguments used 0.07; in
  3.1 it has no mesh damping, and its time responses change (the transmission error and the mesh
  force). Pass ``mesh_damping_ratio=0.07`` to ``MultiRotor`` (or ``damping_ratio=0.07`` to
  ``Mesh``) to reproduce 3.0 results. Models without backlash had no mesh damping in 3.0 and have
  none with the default (`#1427 <https://github.com/petrobras/ross/pull/1427>`_). The
  interface's MultiRotor form has a field for the ratio
  (`#1444 <https://github.com/petrobras/ross/pull/1444>`_).
- ``MultiRotor.add_nodes`` and ``add_elements`` rebuild the model with the ``gear_mesh_stiffness``
  given to the constructor, so a value assigned to ``mesh.stiffness`` afterwards, or left there by
  a time response with ``update_mesh_stiffness=True``, is not carried over
  (`#1427 <https://github.com/petrobras/ross/pull/1427>`_).
- ``Rotor.disk_elements`` and ``Rotor.point_mass_elements`` are sorted by node number; before,
  they kept the order they were given in (`#1296 <https://github.com/petrobras/ross/pull/1296>`_).
- ``Rotor.add_elements`` accepts a ``MotorElement``, which replaces the rotor's motor
  (`#1296 <https://github.com/petrobras/ross/pull/1296>`_). Other objects that are not supported
  elements still raise ``ValueError("<obj> is not a valid element.")``, and
  ``MultiRotor.add_elements`` now raises the same ``ValueError`` for them instead of
  ``AttributeError`` (`#1442 <https://github.com/petrobras/ross/pull/1442>`_).
- **Frequency-response plots.** For ``inp != out``, ``FrequencyResponseResults.plot_magnitude``,
  ``plot_phase``, ``plot_polar_bode`` and ``plot`` plot ``freq_resp[out, inp]``, the response at
  ``out`` to a unit force at ``inp``, and ``ST_Rotor.run_freq_response`` stores those samples; in
  3.0 they used ``freq_resp[inp, out]``. Off-diagonal plots change wherever :math:`H` is not
  symmetric (gyroscopic rotors at speed, cross-coupled or anisotropic bearings). Scripts that
  swapped ``inp`` and ``out`` to get the response at ``out`` now get the transposed entry and must
  swap them back (see *Bug Fixes*, `#1441 <https://github.com/petrobras/ross/pull/1441>`_).
- Functions decorated with ``check_units`` return ``str`` arguments unchanged. A numeric string
  such as ``"3"`` given for a unit argument stays a string; pint converted it to the number 3
  before (`#1438 <https://github.com/petrobras/ross/pull/1438>`_).
- ``Rotor.add_nodes`` ignores repeated positions (3.0 raised ``ZeroDivisionError``) and no longer
  sorts the caller's ``new_nodes_pos`` list in place. For layered shafts, the tags of the shaft
  elements of the returned rotor follow the order of the pieces of each original element, as in
  ``refine``, instead of their position; geometry and numbering are unchanged
  (`#1439 <https://github.com/petrobras/ross/pull/1439>`_).
- ``ross.utils.downsample_figure`` keeps at most ``max(n_out, 4)`` points of each trace;
  ``n_out`` below 4 is raised to 4 (`#1447 <https://github.com/petrobras/ross/pull/1447>`_).
- ``BacklashResults.PARAMS``: the ``"units"`` key of every entry is now ``"unit"``, and the unit of
  ``contact_ratio`` is ``"dimensionless"`` instead of ``"--"``
  (`#1296 <https://github.com/petrobras/ross/pull/1296>`_).
- ``TimeResponseResults.plot_dfft(frequency_range=...)`` raises ``ValueError`` when the range
  contains no DFFT points; before, the trace was empty
  (`#1296 <https://github.com/petrobras/ross/pull/1296>`_).
- With ``displacement_units`` other than ``"m"``, ``plot_1d`` and ``plot_dfft`` no longer apply
  the unit conversion twice, so their values differ from 3.0 by the conversion factor (see *Bug
  Fixes*, `#1418 <https://github.com/petrobras/ross/pull/1418>`_).
- The progress messages "Running direct method" and "Running with model reduction: ..." of the
  time integration (``run_time_response``, ``run_crack``, ``run_rubbing``,
  ``run_misalignment``) end with a carriage return instead of a newline, so the next output
  overwrites them (`#1296 <https://github.com/petrobras/ross/pull/1296>`_).
- ``AmbTimeResponse.A_cl``, ``B_cl`` and ``C_cl`` are stored in the balanced coordinates after
  ``build_matrices``, in which the physical state is ``state_scaling * z``. ``D_cl`` and the
  outputs are unchanged
  (`#1422 <https://github.com/petrobras/ross/pull/1422>`_).

Saved files
^^^^^^^^^^^

- A rotor with a motor is saved with a ``MotorElement`` section, which ROSS 3.0 cannot read.
  Rotor files without a motor are unchanged (`#1296 <https://github.com/petrobras/ross/pull/1296>`_).
- ``TimeResponseResults``, ``AmbTimeResponseResults`` and ``BacklashResults`` files store the run
  speed. Files saved by 3.0 load in 3.1 with ``speed=None`` (so ``one_revolution=True`` is not
  available for them); files saved by 3.1 from ``run_time_response`` store the speed and fail to
  load in 3.0 with ``TypeError`` (`#1416 <https://github.com/petrobras/ross/pull/1416>`_).
- ``ConvergenceResults`` files can no longer be loaded; ``MeshConvergenceResults`` has its own
  format, which stores the refined rotor (`#1420 <https://github.com/petrobras/ross/pull/1420>`_).
- Coaxial rotor files did not load in 3.0 either, and still do not
  (`#1434 <https://github.com/petrobras/ross/pull/1434>`_).

Dependencies
^^^^^^^^^^^^

- **Python.** ``requires-python`` is ``>=3.13``: ROSS 3.1 supports Python 3.13, 3.14 and 3.15,
  and Python 3.12 is dropped, following the SPEC 0 support window. Stay on 3.0.x for Python 3.12
  (`#1447 <https://github.com/petrobras/ross/pull/1447>`_).
- **Floors.** ``plotly>=5.15`` (was 5.11) and ``pint>=0.24.4`` (was 0.18); the older versions do
  not work with the numpy 2.2 floor. The other minimum versions are those of 3.0.0
  (`#1447 <https://github.com/petrobras/ross/pull/1447>`_).
- **tsdownsample (optional).** ``downsample_figure`` thins the traces of the motor result plots
  (`#1296 <https://github.com/petrobras/ross/pull/1296>`_). It uses the MinMaxLTTB algorithm of
  ``tsdownsample`` when that package is installed, with the new extra
  ``pip install "ross-rotordynamics[downsample]"`` (``tsdownsample>=0.1.5``, which has no wheels
  for Python 3.15); otherwise it falls back to a min-max decimation in numpy that keeps the
  lowest and highest sample of each interval and the first and last samples. ``tsdownsample``
  is imported when ``downsample_figure`` is called, and the private module attribute
  ``ross.utils._downsampler`` was removed (`#1447 <https://github.com/petrobras/ross/pull/1447>`_).
- The interface vendors three.js r186 (MIT license, about 2.1 MB) inside the package, with no new
  Python dependency (`#1429 <https://github.com/petrobras/ross/pull/1429>`_).

Interface Project Files, Scripts and Routes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- The frontend no longer publishes its functions on ``window``, and buttons dispatch
  ``data-action`` attributes instead of inline HTML handlers; scripts that called those globals
  stop working (`#1419 <https://github.com/petrobras/ross/pull/1419>`_).
- A coupling with a blank ``n`` is numbered by its position in the list, and values such as
  ``1_0`` or ``0x10`` count as blank (`#1419 <https://github.com/petrobras/ross/pull/1419>`_).
- Project files have a ``probes`` category, disks and gears can have the element type
  ``Geometry``, and any element can carry a ``shape3d`` field used only by the 3-D view; files
  without them load with empty values (`#1429 <https://github.com/petrobras/ross/pull/1429>`_).
  Project files can carry ``multi_params.mesh_damping_ratio``; files without it load and
  build as before (`#1444 <https://github.com/petrobras/ross/pull/1444>`_).
- The ``probe_output`` function written into exported scripts returns arrays of shape
  ``(1, ndof, nfreq)`` instead of ``(ndof, 1, nfreq)``: the probe is output 0
  (`#1441 <https://github.com/petrobras/ross/pull/1441>`_).
- Exported Python scripts write step counts as integer literals and refuse non-numeric ones, use
  the *Steps* field for the unbalance sweep, write material names with underscores
  (`#1419 <https://github.com/petrobras/ross/pull/1419>`_), and build disks from a list of
  ``(constructor, kwargs)`` pairs so ``DiskElement.from_geometry`` can appear
  (`#1429 <https://github.com/petrobras/ross/pull/1429>`_).
- The ``/build_rotor`` route takes an optional ``figure`` field (default ``true``; ``false`` skips
  the 2-D figure) and returns the 3-D ``scene``. New routes: ``/api/rotor/concatenate`` and
  ``/api/rotor/split_shaft`` (`#1419 <https://github.com/petrobras/ross/pull/1419>`_),
  ``/api/rotor/mesh_shafts`` and ``/api/units/convert``
  (`#1429 <https://github.com/petrobras/ross/pull/1429>`_).

Known Issues
~~~~~~~~~~~~

The motor coupling of ``Rotor.run_with_motor``
(`#1296 <https://github.com/petrobras/ross/pull/1296>`_) has the following open problems in
3.1.0. The stand-alone motor analyses of ``MotorElement`` match the steady-state equivalent
circuit and are not affected by the first two.

- **Speed history.** The motor speed is integrated with ``Ip_motor + Ip_load`` only, not with the
  inertia of the rotor model, and the net torque is also applied to the torsional degree of
  freedom of the motor node. With the default ``Ip_load=0`` the start-up is too fast (18.8 times
  on the rotor of the tutorial), so run-ups through a critical speed are wrong. The torsional
  displacements drift without bound; the lateral response is not affected by the drift.
- **Default steady-state window.** With ``steady_state=True`` the integration starts from rest
  inside the window, and the free vibration has not decayed when it ends: unbalance amplitudes
  read 19 % to more than 100 % above the forced response, depending on the damping. Use
  ``steady_state=False`` and discard the start of the response.
- **Save and load.** ``MotorResponseResults`` and ``PhaseResults`` fail to load after ``save``,
  and the ``motor_results`` attribute of the results of ``run_with_motor`` is not saved.
- **Force array.** ``run_with_motor`` adds the unbalance force and the torque to the ``F`` array
  it is given, in place. Pass a copy when calling it more than once with the same array.
- **Supply settings.** The ``voltage_net`` and ``frequency_net`` given to the ``MotorElement``
  constructor are ignored by ``run_direct_on_line``; pass them to the ``run_*`` call.
  ``load_torque_entrance_time`` and ``viscosity_coeff`` accept only floats, not ``Q_`` values
  (``viscosity_coeff`` is a rotational friction coefficient in N·m·s/rad, not a viscosity), and
  ``stator_reactance`` and ``rotor_reactance`` are leakage reactances.
- **Motor on a** ``MultiRotor``. A motor on the driven rotor receives the speed of the driving
  rotor.

These will be addressed in a later release.

Contributors
~~~~~~~~~~~~

This release includes contributions from: @jguarato, @bgglzv, @viniciuserrav, @murilloabs,
@gsabinoo, @leonardocabr, @raphaeltimbo
