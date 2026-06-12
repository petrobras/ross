from ross.bearings.magnetic.time_response import TimeResponseAmb
from ross.bearings.magnetic.utils import rotor_amb_example_with_complex_controllers


def test_run_time_amb_response():
    """
    Test the time response simulation for a rotor with AMBs.

    Verifies that the simulation runs and populates the output data structures
    correctly in memory.
    """
    # Setup the rotor model
    rotor = rotor_amb_example_with_complex_controllers()
    sim = TimeResponseAmb(rotor)

    # Execute the simulation
    sim.run()

    # Assertions to verify simulation output in memory
    assert sim.y is not None, "Simulation output matrix 'y' should not be None"
    assert sim.t is not None, "Simulation time vector 't' should not be None"
    assert len(sim.t) > 0, "Time vector should not be empty"

    # The output matrix 'y' should have the same number of rows as the time vector
    assert sim.y.shape[0] == len(
        sim.t
    ), "Output matrix and time vector should have consistent length"
