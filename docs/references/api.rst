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
    PlainJournal
    BearingFluidFlow
    MagneticBearingElement
    GearElement
    CouplingElement
    Probe
    PointMass

Rotor and Results
-----------------
Rotor class and available methods for analysis (`run_` methods).

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
    ConvergenceResults
    TimeResponseResults
    Level1Results
    UCSResults