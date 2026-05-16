Version 2.3.0
-------------

The following enhancements and bug fixes were implemented for this release:

Enhancements
~~~~~~~~~~~~

Fully Coupled Oil Film–Pad Thermal Model for Tilting-Pad Bearings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Implemented a fully coupled thermal model for the ``TiltingPad`` class, introducing pad conduction solvers and film-pad coupling iteration.
The previous model treated the pad as adiabatic — no heat was exchanged between the oil film and the pad body. The new formulation resolves 
the temperature field across the pad cross-section and iteratively couples it with the Reynolds and energy equations (`#1278 <https://github.com/petrobras/ross/pull/1278>`_).

``BearingResults`` Pattern for Bearing Post-Processing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Introduced a standardized post-processing layer for ``TiltingPad``, ``ThrustPad``, ``PlainJournal``, and ``SqueezeFilmDamper`` in ``ross/bearings/``.
The new abstract base class ``BearingResults`` centralizes the result visualization and console output interface (``show_results()``, ``show_coefficients_comparison()``, ``show_execution_time()``, ``plot_*()``), 
eliminating method duplication across bearing types (`#1303 <https://github.com/petrobras/ross/pull/1303>`_).

Clearance Analysis
^^^^^^^^^^^^^^^^^^

Added ``run_clearance_analysis()`` to the ``Rotor`` class. Users can now sweep bearing clearances, inspect bearing vibration magnitudes,
and visualize the relationship between response amplitudes and clearance limits in a bar chart. The result object behaves both as a
mapping-like interface and as a plot-ready result container (`#1285 <https://github.com/petrobras/ross/pull/1285>`_).

Rotor ID Summary
^^^^^^^^^^^^^^^^

Added a rotor identification summary method that surfaces key rotor descriptors for reporting and traceability (`#1280 <https://github.com/petrobras/ross/pull/1280>`_).

Architecture Overview Documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Added an Architecture Overview page (``docs/getting_started/architecture.md``) with interactive d2-generated SVG diagrams covering the full ROSS workflow: element definition,
rotor assembly, ``run_*()`` analyses, and results visualization. All 13 diagrams are interlinked, allowing navigation from the overview down to detailed element and analysis
views (`#1275 <https://github.com/petrobras/ross/pull/1275>`_).

Line Shape Option for Magnitude Plots
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Exposed Plotly's ``line_shape`` interpolation option on ``FrequencyResponseResults.plot_magnitude``, ``ForcedResponseResults.plot_magnitude``, and their stochastic counterparts,
allowing users to smooth magnitude plots on demand. Default behavior (``linear``) is unchanged (`#1271 <https://github.com/petrobras/ross/pull/1271>`_).

Use ``BearingElement`` for Dummy Bearings in ``run_static()``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``run_static()`` previously instantiated dummy bearings using the original bearing's class, which failed for subclasses (such as ``rossxl.MaxBrg``) whose constructors require extra parameters. 
The dummy bearings now use ``BearingElement`` directly, since they only need to provide linear spring supports (`#1273 <https://github.com/petrobras/ross/pull/1273>`_).

``ruff format --check`` in CI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Extended the CI workflow to run ``ruff format --check`` alongside ``ruff check``, ensuring that all code merged into the repository is both lint-compliant and formatted consistently (`#1277 <https://github.com/petrobras/ross/pull/1277>`_).

New Tutorials
^^^^^^^^^^^^^

Added several tutorials to make ROSS easier to learn:

- **Bearing classes tutorial** — covers the use of all available bearing classes and their result visualization methods, highlighting similarities and differences across types (`#1286 <https://github.com/petrobras/ross/pull/1286>`_).
- **Seal models tutorial** — examples for ``LabyrinthSeal``, ``HolePatternSeal``, and ``HybridSeal``, replacing the previous Hybrid Seal Analysis tutorial (`#1284 <https://github.com/petrobras/ross/pull/1284>`_).
- **Faults tutorial** — practical examples of misalignment, rubbing, crack, and related fault analyses (`#1288 <https://github.com/petrobras/ross/pull/1288>`_).
- **Tutorials on ReadTheDocs** — added ``tutorial_bearings_part_1``–``3`` and reorganized the tutorial ordering on ReadTheDocs (`#1299 <https://github.com/petrobras/ross/pull/1299>`_).
- General documentation updates (`#1269 <https://github.com/petrobras/ross/pull/1269>`_).


Bug Fixes
~~~~~~~~~

Fix ``run_level1()`` Cross-Coupling Element Instantiation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``run_level1()`` previously created the cross-coupling element via ``bearings[0].__class__(...)``, which assumed all bearing subclasses shared the ``BearingElement`` constructor signature. 
The cross-coupling element is now instantiated as a ``BearingElement`` directly, consistent with the fix applied to ``run_static()`` (`#1302 <https://github.com/petrobras/ross/pull/1302>`_).

Fix Sporadic ``NaN`` in ``_init_orbit``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``test_kappa_rotor3`` was failing intermittently because ``la.eigvals`` could return slightly negative values for a symmetric positive semi-definite matrix due to floating-point rounding, 
producing ``NaN`` after ``sqrt``. Eigenvalues are now clipped to zero before the square root (`#1295 <https://github.com/petrobras/ross/pull/1295>`_).

Fix Mode Shape Discontinuity in Plots
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Addressed discontinuities in the ``Shape`` and ``Orbit`` classes that produced incorrect ``plot_2d()`` visualizations. Updated the ``_calculate()`` method in ``Shape``, 
corrected ``plot_2d()`` for the ``x``, ``y``, and ``major`` orientations, and updated the ``Rotor._index()`` sort so that only ``w_d > 0`` modes are at the top of the index (`#1289 <https://github.com/petrobras/ross/pull/1289>`_).

Fix Dynamic Coefficient Method for Plain Journal Bearings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Corrected the perturbation method used to compute the dynamic coefficients of plain journal bearings in the ``PlainJournal`` class (`#1287 <https://github.com/petrobras/ross/pull/1287>`_).

Fix ``GearElementTVMS`` Initialization from Geometry
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Fixed ``GearElementTVMS.from_geometry`` so that all geometric parameters are correctly propagated, resolving inconsistencies in gear element representation and downstream calculations (`#1274 <https://github.com/petrobras/ross/issues/1253>`_, `#1274 <https://github.com/petrobras/ross/pull/1274>`_).


Contributors
~~~~~~~~~~~~

This release includes contributions from: @gsabinoo, @ViniciusTxc3, @jguarato, @fernandarossi, @mariac-souza, @kiracofe8, @raphaeltimbo
