Version 2.0.0
-------------

The following enhancements and bug fixes were implemented for this release:

Enhancements
~~~~~~~~~~~~

Model Reduction
^^^^^^^^^^^^^^^

Added comprehensive model reduction functionality to reduce the size of rotor models while preserving dynamic behavior. Two methods are available:

- **Guyan reduction**: Reduces model to specified degrees of freedom (DOFs)
- **Pseudo-modal reduction**: Uses modal transformation with specified number of modes

The ``ModelReduction`` class provides methods to reduce matrices and vectors, and revert reduced vectors back to the full model.

.. code-block:: python

   import ross as rs

    compressor = rs.compressor_example()

    dt = 1e-4
    t = np.arange(0, 10, dt)
    speed = 600

    node = [29]
    unb_mag = [0.003]
    unb_phase = [0]

    probe_nodes = [30]

    # Unbalance force
    F = compressor.unbalance_force_over_time(
        node=node, magnitude=unb_mag, phase=unb_phase, omega=speed, t=t
    )

    response = compressor.run_time_response(
        speed,
        F.T,
        t,
        method="newmark",
        model_reduction={
            "method": "guyan",
            "include_nodes": probe_nodes, # Make sure to include output nodes
            "dof_mapping": ["x", "y"],
        },
    )

Thermo-Hydro-Dynamic Tilting-Pad Bearings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Added two new bearing classes with full THD (Thermo-Hydro-Dynamic) analysis:

**TiltingPad**: Tilting-pad journal bearing with individual pad analysis, pivot mechanics, and thermal effects. Uses Reynolds equation and energy equation with finite difference methods, Lund's perturbation method for dynamic coefficients.

**ThrustPad**: Tilting-pad thrust bearing with similar THD capabilities for axial applications.

Both classes support multiple lubricant types, turbulence modeling, and provide accurate dynamic coefficients for stability analysis.


HolePatternSeal (Annular Seals)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Added ``HolePatternSeal`` class for modeling annular seals with holepattern. Uses bulk flow theory with:

- 1D compressible flow through annular clearance
- Perturbation analysis for dynamic coefficients
- Mass, stiffness, and damping matrices
- Leakage prediction
- Temperature-dependent viscosity using Sutherland's law

Previously named "honeycomb seal", this is a comprehensive implementation for pocket damper seals in turbomachinery.


LabyrinthSeal Enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Significant improvements to the ``LabyrinthSeal`` class:

- Integration with ccp for gas properties
- Support for custom gas mixtures via composition dictionary
- Improved numerical methods for pressure distribution
- Enhanced velocity field calculations with swirl effects
- Up to **10x performance improvement** through code optimization


MultiRotor Class
^^^^^^^^^^^^^^^^^

Added ``MultiRotor`` class for analyzing systems with multiple interconnected rotors, such as geared systems. Supports:

- Multiple rotors connected via coupling or gear elements
- Coordinated modal analysis across all rotors
- Campbell diagram for multi-rotor systems
- Frequency response analysis
- Proper handling of coupled dynamics


GearElement and Mesh Stiffness
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

New methodology for gear mesh coupling:

- ``GearElement`` class for connecting rotors via gear mesh
- Mesh stiffness calculation based on gear geometry
- Contact ratio computation
- Support for multi-rotor analysis with gears


Flexible Crack Models
^^^^^^^^^^^^^^^^^^^^^^

Added flexible crack model implementation (breathing and open crack models) for shaft crack analysis. Enables:

- Time-varying stiffness due to crack opening/closing
- Breathing crack behavior
- Open crack model for continuous stiffness reduction


Gravitational Force
^^^^^^^^^^^^^^^^^^^

Added ``gravitational_force()`` method to apply gravitational loads on the rotor system, useful for static deflection analysis and considering gravity in dynamic simulations.


Free-Free Analysis
^^^^^^^^^^^^^^^^^^

Added free-free boundary condition option in ``run_freq_response()`` for analyzing rotors without supports, useful for:

- Component mode synthesis
- Free-free modal analysis
- Validation against experimental modal testing


Torsional Model Conversion
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Added ``convert_6dof_to_torsional()`` function to convert a 6-DoF rotor model to a torsional-only model, enabling:

- Simplified torsional analysis
- Faster computation for torsional-dominated problems
- Separate lateral and torsional analysis


Orbit Animation
^^^^^^^^^^^^^^^

Added animation capability to orbit plots with ``animation=True`` parameter. Visualizes:

- Dynamic motion of the rotor orbit
- Time-evolution of displacements
- Easier identification of motion patterns


AMB Sensitivity Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^

Added ``run_amb_sensitivity()`` method for computing Active Magnetic Bearing (AMB) sensitivities according to ISO 14839-3 standard. This enables:

- Open-loop, closed-loop, and sensitivity transfer functions calculation
- Compliance with API STANDARD 617 stability assessment requirements
- Chirp disturbance signal injection for frequency response analysis
- Multiple AMB analysis with configurable sensor positions
- Time-domain and frequency-domain visualization

The method returns a ``SensitivityResults`` object with attributes for magnitude, phase, and maximum sensitivities, plus plotting methods for both frequency-domain (``plot()``) and time-domain (``plot_run_time_results()``) visualization.

.. code-block:: python

   rotor_amb = rs.rotor_amb_example()

   sensitivity_results = rotor_amb.run_amb_sensitivity(
       speed=0,
       t_max=5,
       dt=1e-4,
       disturbance_amplitude=10e-6,
       disturbance_min_frequency=0.01,
       disturbance_max_frequency=150,
       amb_tags=["Magnetic Bearing 0", "Magnetic Bearing 1"],
       sensors_theta=45,
   )

   # Access maximum sensitivity for a specific bearing
   max_sens = sensitivity_results.max_abs_sensitivities["Magnetic Bearing 0"]["x"]

   # Plot frequency response
   fig = sensitivity_results.plot()


ROSS GPT Integration
^^^^^^^^^^^^^^^^^^^^

Integrated ROSS GPT assistant into documentation for interactive help and guidance on using the library.


Performance Improvements
~~~~~~~~~~~~~~~~~~~~~~~~

Significant performance optimizations across multiple modules:

HolePatternSeal Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Optimized HolePatternSeal for **1.6x faster** computation through:

- Caching of repeated calculations
- Vectorized operations
- Reduced function call overhead


LabyrinthSeal Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^

Optimized LabyrinthSeal for **up to 10x faster** performance:

- Smart multiprocessing thresholds
- Matrix creation optimization
- Reduced memory allocations
- Efficient array operations


THD Bearing Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^

Optimized PlainJournal (THD Cylindrical) bearing calculations using:

- Numba JIT compilation for performance-critical sections
- Reduced redundant computations
- Improved numerical integration


General Optimizations
^^^^^^^^^^^^^^^^^^^^^

- Applied ``lru_cache`` to frequently-called methods across rotor assembly
- Optimized matrix building operations
- Reduced computation time in modal and response analyses
- Pre-computed base matrices in ``Rotor.__init__`` for faster repeated calculations


API Changes
~~~~~~~~~~~

Remove 4-DoF Model (Breaking Change)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Removed support for the 4-DoF model. All analyses now use the **6-DoF model only** (lateral, axial, and torsional). This change:

- Simplifies the codebase
- Ensures consistent behavior across all elements
- Improves maintainability
- Affects: ``ShaftElement``, ``DiskElement``, ``BearingElement``, ``PointMass``

**Migration**: Update any code that explicitly used ``n_dof=4`` to work with the 6-DoF model.


Folder Restructuring
^^^^^^^^^^^^^^^^^^^^

Renamed the ``fluid_flow`` folder to ``bearing`` for better organization and clarity. All bearing-related classes are now in the ``ross.bearing`` module.

**Migration**: Update imports from ``ross.fluid_flow`` to ``ross.bearing``.


Build System Update
^^^^^^^^^^^^^^^^^^^

Migrated from ``setup.py`` to ``pyproject.toml`` for modern Python packaging standards. This improves:

- Dependency management
- Build reproducibility
- PEP 517 compliance


CylindricalBearing Updates
^^^^^^^^^^^^^^^^^^^^^^^^^^

Enhanced ``CylindricalBearing`` class with:

- Improved parameter handling
- Better documentation with "when to use" guidance
- Support for frequency-dependent coefficients
- Oil flow properties calculation


Format Table Methods
^^^^^^^^^^^^^^^^^^^^

Added ``format_table()`` methods to results classes for displaying:

- Modal analysis results in formatted tables
- Bearing coefficients comparison
- Critical speeds with detailed information


Check Units Decorator
^^^^^^^^^^^^^^^^^^^^^

Added ``@check_units`` decorator to ``run_*`` methods for automatic unit validation and conversion, improving:

- Type safety with pint quantities
- Clear error messages for unit mismatches
- Consistent unit handling across methods


Pressure Distribution Plot
^^^^^^^^^^^^^^^^^^^^^^^^^^

Added pressure distribution plotting capability for THD bearings to visualize:

- 2D pressure field across bearing surface
- Pressure contours
- Location of maximum pressure


Documentation and Tutorials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Documentation Build Updates
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Updated documentation dependencies to latest versions:

- Sphinx 7.2.6 → 8.2.3
- myst-nb 1.1.0 → 1.3.0
- Updated ReadTheDocs Python version to 3.13

Enhanced API documentation with comprehensive theoretical foundations for bearing and seal classes, explaining the numerical methods used.


Tutorial Enhancements
^^^^^^^^^^^^^^^^^^^^^

Expanded tutorials with new examples:

- Friswell book examples added to tutorial
- MultiRotor usage examples
- CouplingElement demonstration
- Torsional analysis guide
- GearElement examples
- New user guide examples (User Guide 30)


Bug Fixes
~~~~~~~~~

Fix integrate_system for Variable Speed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Fixed ``NameError`` in ``integrate_system()`` when using variable speed with frequency-dependent bearing coefficients. The bug caused failures in time response analysis with bearings that have speed-dependent properties.


Fix Lund Perturbation Method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Corrected formulation in Lund's perturbation method for calculating dynamic coefficients in cylindrical bearings, ensuring accurate cross-coupled stiffness and damping terms.


Fix UCS with Rotor Supports
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Fixed bugs in ``run_ucs()`` (Unbalance Constant Speed) analysis when rotor has support elements, including:

- Correct equivalent stiffness calculation
- Proper DOF handling with supports
- Pseudo-modal reduction compatibility


Fix Modal Loading
^^^^^^^^^^^^^^^^^

Fixed error when loading ``ModalResults`` from saved files with NumPy version >= 2.0, ensuring backward compatibility.


Fix MultiRotor Plots
^^^^^^^^^^^^^^^^^^^^^

Fixed plotting issues for ``MultiRotor`` systems, including:

- Bearing element visualization
- Proper scaling and positioning
- Shape updates and reordering


Fix Campbell for Torsional Modes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Corrected Campbell diagram to properly classify and display torsional modes, preventing misidentification of mode types.


Fix Critical Speed Tracking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Improved critical speed identification in Campbell diagrams using MAC (Modal Assurance Criterion) to properly track modes across speed range.


Fix Coupling Element Save/Load
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Fixed ``save()`` and ``load()`` methods for ``CouplingElement`` to correctly persist and restore coupling properties.


Fix Misalignment Parameter
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Corrected the sign of the ``fib`` variable relationship in flexible coupling misalignment calculations.


Fix Crack Test Precision
^^^^^^^^^^^^^^^^^^^^^^^^^

Fixed numerical precision issues in crack test assertions to handle floating-point comparison properly.


Fix plot_deflected_shape Units
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Fixed unit handling in ``plot_deflected_shape()`` to properly accept speed as pint Quantity.


Fix Plotly Rendering
^^^^^^^^^^^^^^^^^^^^^

Fixed plotly rendering issues and dropped deprecated ``heatmapgl`` traces for compatibility with plotly v6.0.0.


Fix Mode Shape Updates
^^^^^^^^^^^^^^^^^^^^^^^

Fixed ``update_mode_3d`` issue and adapted mode shape plotting methods to work correctly with MultiRotor systems.


Testing and Compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~

Python 3.12 and 3.13 Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Updated test suite and codebase for compatibility with Python 3.12 and 3.13:

- Removed Python 3.9 testing (minimum version now 3.10)
- Added Python 3.12 to CI pipeline
- Fixed deprecation warnings
- Updated dependencies for new Python versions


Contributors
~~~~~~~~~~~~

This release includes contributions from: @raphaeltimbo, @jguarato, @gsabinoo, @ViniciusTxc3, @murilloabs, @Raimundovpn, @gNicchetti, @Emanuela-Carneiro, @vitorp0604, @stanley-washington, @CisneirosRaphael
