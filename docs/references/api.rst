.. _API:

.. currentmodule:: ross

API Reference
=============

Material
--------
.. autosummary::
    :toctree: generated/material

    Material

Elements
--------
.. autosummary::
    :toctree: generated/elements

    ShaftElement
    DiskElement
    BearingElement
    SealElement
    BallBearingElement
    RollerBearingElement
    CylindricalBearing
    FluidFilmBearing
    FixedGeometryBearing
    PlainJournal
    PartialArcBearing
    EllipticalBearing
    OffsetHalvesBearing
    MultiLobeBearing
    PressureDamBearing
    TiltingPad
    ThrustPad
    SqueezeFilmDamper
    LabyrinthSeal
    HolePatternSeal
    HybridSeal
    MagneticBearingElement
    GearElement
    GearElementTVMS
    Mesh
    CouplingElement
    Probe
    PointMass
    MotorElement

Rotor and Analysis
------------------
Rotor class and available methods for analysis (``run_*`` methods).

.. autosummary::
    :toctree: generated/rotor

    Rotor
    MultiRotor
    Rotor.run_modal
    Rotor.run_critical_speed
    Rotor.run_freq_response
    Rotor.run_forced_response
    Rotor.run_unbalance_response
    Rotor.run_campbell
    Rotor.run_time_response
    Rotor.run_level1
    Rotor.run_ucs
    Rotor.run_static
    Rotor.run_misalignment
    Rotor.run_rubbing
    Rotor.run_crack
    Rotor.run_harmonic_balance_response
    Rotor.run_amb_sensitivity
    Rotor.run_amb_non_collocation
    Rotor.run_clearance_analysis
    Rotor.run_mesh_convergence
    Rotor.run_with_motor
    Rotor.refine
    Rotor.refine_mesh
    Rotor.api617_unbalance

Example Rotors
--------------
Ready-to-use rotor models for tutorials, doctests, and quick experimentation.

.. autosummary::
    :toctree: generated/examples

    rotor_example
    compressor_example
    rotor_example_6dof
    rotor_example_with_damping
    rotor_example_amb_simple
    rotor_example_amb_complex_controllers
    rotor_example_amb_general_controllers
    motor_example

Results
-------
These are classes used to store results and to provide useful methods such as plotting.

.. autosummary::
    :toctree: generated/results

    Orbit
    Shape
    CriticalSpeedResults
    ModalResults
    CampbellResults
    FrequencyResponseResults
    ForcedResponseResults
    StaticResults
    SummaryResults
    MeshConvergenceResults
    TimeResponseResults
    Level1Results
    UCSResults
    HarmonicBalanceResults
    SensitivityResults
    AmbNonCollocationResults
    ClearanceResults
    MotorResponseResults
    PhaseResults

Bearing Results
---------------
Post-processing classes for thermo-hydro-dynamic bearing models.

.. autosummary::
    :toctree: generated/bearing_results

    BearingResults
    FluidFilmBearingResults
    ThrustPadResults
    SqueezeFilmDamperResults

Faults
------
Fault models for misalignment, rubbing, and crack analyses.

.. autosummary::
    :toctree: generated/faults

    MisalignmentFlex
    MisalignmentRigid
    Rubbing
    Crack

Model Reduction
---------------
Methods for reducing the size of rotor finite element models.

.. autosummary::
    :toctree: generated/model_reduction

    ModelReduction
    PseudoModal
    Guyan

Motor Drives
------------
Electrical sources that feed a ``MotorElement``.

.. currentmodule:: ross.motors.motor_drive

.. autosummary::
    :toctree: generated/motor_drives

    SourceAC
    InverterVF
    InverterFOC

Stochastic Analysis
-------------------
Stochastic elements, rotor assembly, and results for uncertainty quantification.

.. currentmodule:: ross.stochastic

.. autosummary::
    :toctree: generated/stochastic

    ST_Material
    ST_ShaftElement
    ST_DiskElement
    ST_BearingElement
    ST_PointMass
    ST_Rotor
    ST_CampbellResults
    ST_FrequencyResponseResults
    ST_ForcedResponseResults
    ST_TimeResponseResults

Utilities
---------
Unit conversion helpers, visualization utilities and speed ranges for sweep analyses.

.. currentmodule:: ross

.. autosummary::
    :toctree: generated/utilities

    Q_
    get_data_from_figure
    visualize_matrix
    lubricants_dict

.. currentmodule:: ross.utils

.. autosummary::
    :toctree: generated/utilities

    speed_range_from_endpoints
