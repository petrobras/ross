Version 2.1.0
-------------

The following enhancements and bug fixes were implemented for this release:

Enhancements
~~~~~~~~~~~~

Harmonic Balance Method
^^^^^^^^^^^^^^^^^^^^^^^

Added ``run_harmonic_balance()`` method to the ``Rotor`` class for computing steady-state periodic responses under harmonic excitation using the Harmonic Balance Method (HBM). Features include:

- Multi-harmonic expansion for nonlinear and parametric excitation
- Support for crack-induced stiffness variation
- ``HarmonicBalanceResults`` class with ``plot_dfft()``, ``plot_deflected_shape()``, and probe response methods
- Tutorial with worked examples

.. code-block:: python

   import ross as rs
   import numpy as np

   rotor = rs.rotor_example()
   speed = 300
   dt = 1e-4
   t = np.arange(0, 5, dt)

   node = [3]
   unb_mag = [0.001]
   unb_phase = [0]

   F = rotor.unbalance_force_over_time(
       node=node, magnitude=unb_mag, phase=unb_phase, omega=speed, t=t
   )

   hb_results = rotor.run_harmonic_balance(
       speed=speed,
       F=F.T,
       dt=dt,
       num_harmonics=5,
   )


Squeeze Film Damper
^^^^^^^^^^^^^^^^^^^

Added ``SqueezeFilmDamper`` class as a subclass of ``BearingElement`` for modeling squeeze film dampers (SFDs). Supports:

- Short and long bearing approximations
- End seal and groove configurations
- Frequency-dependent stiffness and damping coefficients
- Integration with existing rotor assembly workflow


Hybrid Seal
^^^^^^^^^^^

Added ``HybridSeal`` class for modeling hybrid seals that combine labyrinth and hole-pattern seal stages in series configuration. Features:

- Coupled solution of labyrinth and hole-pattern stages
- Convergence criterion based on seal leakage matching
- Pressure distribution plotting across both stages
- Dynamic coefficients (mass, stiffness, damping) for the combined seal


Enhanced Magnetic Bearing with General Controllers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Upgraded ``MagneticBearingElement`` from a fixed PID controller to a generalized control system approach using the ``control`` library. Enhancements include:

- Support for arbitrary transfer function controllers (not limited to PID)
- Coordinate transformations with bearing sensor rotation
- Scaled AMB equivalent gains with ``k_amp`` and ``k_sense``
- Updated AMB sensitivity analysis (``run_amb_sensitivity``) with corrected sensor rotation handling
- New tutorial (tutorial_part_5) with auxiliary methods for defining and evaluating transfer functions


GearElement with Time-Varying Mesh Stiffness (TVMS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Extended gear modeling with time-varying mesh stiffness capabilities:

- ``GearElementTVMS`` and ``Mesh`` classes for computing stiffness as a function of angular position
- Spur gear geometry computation (involute curves, transition curves, contact ratio)
- Square profile approximation for variable stiffness
- Support for user-defined constant stiffness or geometry-based TVMS
- Integration with ``run_time_response()`` for variable stiffness simulations


Post-Processing Methods for THD Bearings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Added comprehensive post-processing and visualization methods across THD bearing classes (``TiltingPad``, ``ThrustPad``, ``PlainJournal``):

- ``show_results()`` — formatted display of field quantities at all speeds
- ``show_coefficients_comparison()`` — tabular comparison of dynamic coefficients
- ``plot_results()`` — field visualization (pressure, temperature, film thickness)
- Optimization convergence tracking with residual history plots
- Per-pad convergence visualization for tilting pad bearings
- Viridis colorscale standardization across all THD visualizations


TiltingPad Performance Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Significant performance improvements to the ``TiltingPad`` class:

- Numba JIT compilation for performance-critical numerical routines
- Sparse matrix solvers for Reynolds equation
- Vectorized operations replacing explicit loops
- Fixed ``determine_eccentricity`` to require load inputs and clarified docstring


MCP Server for AI-Assisted Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Added ``ross/mcp/`` package providing a Model Context Protocol (MCP) server for AI-assisted rotordynamics analysis. Includes 6 tools:

- ``create_example_rotor`` — load built-in example rotors
- ``load_rotor_from_file`` — load rotors from saved ``.json`` or ``.toml`` files
- ``describe_rotor`` — detailed text description of rotor components
- ``run_modal_analysis`` — eigenvalue analysis at a given speed
- ``run_campbell_diagram`` — Campbell diagram data over a speed range
- ``run_unbalance_response`` — unbalance response analysis

.. code-block:: bash

   pip install "ross-rotordynamics[mcp]"
   ross-mcp          # run via stdio transport
   python -m ross.mcp  # alternative


JSON Save/Load Support
^^^^^^^^^^^^^^^^^^^^^^

Added JSON as an alternative serialization format alongside TOML:

- ``Rotor.save()`` and ``Rotor.load()`` now support both ``.toml`` and ``.json`` file extensions
- Format-aware I/O helpers (``load_data``, ``dump_data``) detect format from the file extension
- JSON format enables easier integration with web applications and AI tools


Rotor File Version Traceability
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Added version tracking to saved rotor files:

- ``Rotor.save()`` now includes ``ross_version`` as a top-level key
- ``Rotor.load()`` compares saved version against current version and emits a warning if they differ
- Helps diagnose compatibility issues when loading files saved with different ROSS versions


Bearing/Seal Skip-Computation on Load
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Improved loading of rotors with complex bearing and seal elements:

- Override ``read_toml_data`` in ``BearingElement`` so saved coefficients are passed through to ``__init__``
- Subclass constructors (THD bearings, seals) skip expensive recomputation when pre-computed coefficients are available
- Significantly faster ``Rotor.load()`` for models with THD bearings or seals


Hover Information for Bearings and Seals
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Added interactive hover information to bearing and seal elements in rotor plots, showing element details on mouse hover — matching the existing shaft element hover functionality.


Bug Fixes
~~~~~~~~~

Fix add_nodes for conical shaft elements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Fixed incorrect behavior in ``Rotor.add_nodes`` when applied to conical (tapered) shaft elements. The method now correctly interpolates diameters and lengths when splitting tapered elements.


Fix transfer matrix for zero-speed FRF
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Corrected the transfer matrix formulation used in frequency response function (FRF) calculations at zero rotation speed, resolving significant discrepancies between numerical and experimental results.


Fix plot_rotor for coupling element
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Fixed an issue that occurred during rotor plotting when a coupling element was combined with bearings.


Fix units in format_table of bearings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Corrected coefficient units displayed in the ``.format_table()`` method of ``BearingElement`` subclasses.


Fix TiltingPad class example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Corrected the example in the ``TiltingPad`` class for using the ``determine_eccentricity`` equilibrium type, which now requires explicit load inputs.


Fix warning message formatting in TiltingPad
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Fixed formatting of warning messages during ``TiltingPad`` initialization.


Fix typos and documentation clarity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Fixed numerous typos and improved clarity across documentation files, Jupyter notebooks, and Python docstrings.


Documentation and Tutorials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Added MCP server tutorial to documentation
- Added hybrid seal tutorial
- Updated AMB sensitivity analysis tutorial (tutorial_part_5) with transfer function methods
- Added harmonic balance tutorial with worked examples
- Updated multi-rotor and gear element tutorials
- Fixed tutorial print output for ``ShaftElement``
- Updated README links


Contributors
~~~~~~~~~~~~

This release includes contributions from: @raphaeltimbo, @jguarato, @gsabinoo, @murilloabs, @luisotaviomc2002, @ArthurIasbeck, @Raimundovpn, @ViniciusTxc3, @tches-co
