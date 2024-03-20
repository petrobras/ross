from pathlib import Path
from tempfile import tempdir

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal

from ross import Q_
from ross.results import *
from ross.rotor_assembly import *


@pytest.fixture
def rotor1():
    return rotor_example()


def test_save_load_campbell(rotor1):
    speed = np.linspace(0, 1000, 51)
    response = rotor1.run_campbell(speed)

    file = Path(tempdir) / "campbell.toml"
    response.save(file)
    response2 = CampbellResults.load(file)

    assert response2.speed_range.all() == response.speed_range.all()
    assert response2.wd.all() == response.wd.all()
    assert response2.log_dec.all() == response.log_dec.all()
    assert response2.whirl_values.all() == response.whirl_values.all()


def test_save_load_criticalspeed(rotor1):
    response = rotor1.run_critical_speed()

    file = Path(tempdir) / "critical_speed.toml"
    response.save(file)
    response2 = CriticalSpeedResults.load(file)

    assert response2._wn.all() == response._wn.all()
    assert response2._wd.all() == response._wd.all()
    assert response2.log_dec.all() == response.log_dec.all()
    assert response2.damping_ratio.all() == response.damping_ratio.all()


def test_save_load_modal(rotor1):
    response = rotor1.run_modal(0)

    file = Path(tempdir) / "modal.toml"
    response.save(file)
    response2 = ModalResults.load(file)

    assert response2.speed == response.speed
    assert response2.evalues.all() == response.evalues.all()
    assert response2.evectors.all() == response.evectors.all()
    assert response2.wn.all() == response.wn.all()
    assert response2.wd.all() == response.wd.all()
    assert response2.damping_ratio.all() == response.damping_ratio.all()
    assert response2.log_dec.all() == response.log_dec.all()
    assert response2.ndof == response.ndof
    assert np.array(response2.nodes).all() == np.array(response.nodes).all()
    assert np.array(response2.nodes_pos).all() == np.array(response.nodes_pos).all()
    assert (
        np.array(response2.shaft_elements_length).all()
        == np.array(response.shaft_elements_length).all()
    )


def test_save_load_freqresponse(rotor1):
    speed = np.linspace(0, 1000, 11)
    response = rotor1.run_freq_response(speed_range=speed)

    file = Path(tempdir) / "frf.toml"
    response.save(file)
    response2 = FrequencyResponseResults.load(file)

    assert response2.freq_resp.all() == response.freq_resp.all()
    assert response2.speed_range.all() == response.speed_range.all()
    assert response2.velc_resp.all() == response.velc_resp.all()
    assert response2.accl_resp.all() == response.accl_resp.all()


def test_save_load_unbalance_response(rotor1):
    speed = np.linspace(0, 1000, 51)
    response = rotor1.run_unbalance_response(3, 0.01, 0.0, speed)

    file = Path(tempdir) / "unbalance.toml"
    response.save(file)
    response2 = ForcedResponseResults.load(file)

    assert response2.rotor == response.rotor
    assert response2.forced_resp.all() == response.forced_resp.all()
    assert response2.speed_range.all() == response.speed_range.all()
    assert response2.velc_resp.all() == response.velc_resp.all()
    assert response2.accl_resp.all() == response.accl_resp.all()
    assert response2.unbalance.all() == response.unbalance.all()


def test_save_load_static(rotor1):
    response = rotor1.run_static()

    file = Path(tempdir) / "static.toml"
    response.save(file)
    response2 = StaticResults.load(file)

    assert np.array(response2.deformation).all() == np.array(response.deformation).all()
    assert np.array(response2.Vx).all() == np.array(response.Vx).all()
    assert np.array(response2.Bm).all() == np.array(response.Bm).all()
    assert np.array(response2.w_shaft).all() == np.array(response.w_shaft).all()
    assert response2.disk_forces == response.disk_forces
    assert response2.bearing_forces == response.bearing_forces
    assert np.array(response2.nodes).all() == np.array(response.nodes).all()
    assert np.array(response2.nodes_pos).all() == np.array(response.nodes_pos).all()
    assert np.array(response2.Vx_axis).all() == np.array(response.Vx_axis).all()


def test_save_load_convergence(rotor1):
    response = rotor1.convergence()

    file = Path(tempdir) / "convergence.toml"
    response.save(file)
    response2 = ConvergenceResults.load(file)

    assert response2.el_num.all() == response.el_num.all()
    assert response2.eigv_arr.all() == response.eigv_arr.all()
    assert response2.error_arr.all() == response.error_arr.all()


def test_save_load_timeresponse(rotor1):
    speed = 500.0
    size = 1000
    node = 3
    t = np.linspace(0, 10, size)
    F = np.zeros((size, rotor1.ndof))
    F[:, 4 * node] = 10 * np.cos(2 * t)
    F[:, 4 * node + 1] = 10 * np.sin(2 * t)
    response = rotor1.run_time_response(speed, F, t)

    file = Path(tempdir) / "time.toml"
    response.save(file)
    response2 = TimeResponseResults.load(file)

    assert response2.t.all() == response.t.all()
    assert response2.yout.all() == response.yout.all()
    assert response2.xout.all() == response.xout.all()
    assert response2.rotor == response.rotor


def test_campbell_plot(rotor1):
    speed = np.linspace(0, 400, 101)
    camp = rotor1.run_campbell(speed)
    fig = camp.plot(
        harmonics=[1, 2],
        damping_parameter="damping_ratio",
        frequency_range=Q_((2000, 10000), "RPM"),
        damping_range=(-0.1, 100),
        frequency_units="RPM",
    )
    crit_array_x = np.array(
        [2590.2641754, 1306.51513941, 2868.14592367, 1420.76907353, 3264.81334336]
    )
    crit_array_y = np.array(
        [2590.2641754, 2613.03027882, 2868.14592367, 2841.53814705, 6529.62668672]
    )
    assert_allclose(fig.data[0]["x"], crit_array_x)
    assert_allclose(fig.data[0]["y"], crit_array_y)


def test_orbit():
    orb = Orbit(node=0, node_pos=0, ru_e=(1 + 1j), rv_e=(1 - 1j))
    assert_allclose(orb.minor_axis, np.sqrt(2))
    assert_allclose(orb.major_axis, np.sqrt(2))
    assert_allclose(orb.kappa, 1)
    assert orb.whirl == "Forward"

    orb = Orbit(node=0, node_pos=0, ru_e=(1 - 1j), rv_e=(1 + 1j))
    assert_allclose(orb.minor_axis, np.sqrt(2))
    assert_allclose(orb.major_axis, np.sqrt(2))
    assert_allclose(orb.kappa, -1)
    assert orb.whirl == "Backward"


def test_orbit_calculate_amplitude():
    # create orbit with major axis at 45deg
    orb = Orbit(node=0, node_pos=0, ru_e=(2 + 1j), rv_e=(2 - 1j))

    assert_allclose(orb.calculate_amplitude(Q_(0, "deg"))[0], 1.7949401413591568)
    assert_allclose(orb.calculate_amplitude(Q_(45, "deg"))[0], 2.8284271247461903)
    assert_allclose(
        orb.calculate_amplitude(Q_(135, "deg"))[0], 1.4142135623730947, rtol=1e-3
    )
    assert_allclose(orb.calculate_amplitude("minor")[0], 1.4142135623730947)
    assert_allclose(orb.calculate_amplitude("major")[0], 2.8284271247461903)


def test_probe_response(rotor1):
    speed = 500.0
    size = 50
    node = 3
    t = np.linspace(0, 10, size)
    F = np.zeros((size, rotor1.ndof))
    F[:, 4 * node] = 10 * np.cos(2 * t)
    F[:, 4 * node + 1] = 10 * np.sin(2 * t)
    response = rotor1.run_time_response(speed, F, t)

    probe1 = (3, 0)  # node 3, orientation 0° (X dir.)
    probe2 = (3, 90)  # node 3, orientation 90°(Y dir.)
    resp_prob1 = np.array(
        [
            0.00000000e00,
            4.07504756e-06,
            1.19778973e-05,
            1.68562228e-05,
            1.34097882e-05,
            -8.95773240e-08,
            -1.74067471e-05,
            -2.95209579e-05,
            -3.01855154e-05,
            -1.89205317e-05,
            -2.29132987e-06,
            1.13248561e-05,
            1.57680942e-05,
            1.12840083e-05,
            4.45184988e-06,
            1.44108502e-06,
            5.73232266e-06,
            1.47254770e-05,
            2.03602747e-05,
            1.71911791e-05,
            4.09506854e-06,
            -1.40651357e-05,
            -2.74353128e-05,
            -2.95896810e-05,
            -2.01103533e-05,
            -4.31640811e-06,
            8.88805222e-06,
            1.34063573e-05,
            9.36472795e-06,
            2.74500015e-06,
            2.47977244e-07,
            4.79900877e-06,
            1.43604897e-05,
            2.11877017e-05,
            1.84083031e-05,
            5.60872538e-06,
            -1.26209274e-05,
            -2.73822073e-05,
            -3.08770410e-05,
            -2.24365372e-05,
            -7.55159499e-06,
            5.39814673e-06,
            1.02831370e-05,
            6.85800700e-06,
            6.80814109e-07,
            -1.00412393e-06,
            4.62490597e-06,
            1.50749152e-05,
            2.32801701e-05,
            2.16932061e-05,
        ]
    )
    resp_prob2 = np.array(
        [
            0.00000000e00,
            4.13295078e-06,
            8.25529257e-06,
            1.28932310e-05,
            1.59791798e-05,
            1.56518494e-05,
            1.12427869e-05,
            2.99902717e-06,
            -6.75754376e-06,
            -1.53896798e-05,
            -2.03376588e-05,
            -2.10011006e-05,
            -1.86573716e-05,
            -1.41929210e-05,
            -9.53828958e-06,
            -5.02735549e-06,
            4.01893485e-07,
            6.18917879e-06,
            1.25133310e-05,
            1.78765280e-05,
            1.95444824e-05,
            1.71799501e-05,
            1.05712588e-05,
            1.50416503e-06,
            -6.69050872e-06,
            -1.22119438e-05,
            -1.38000853e-05,
            -1.23936175e-05,
            -9.33203262e-06,
            -5.43072370e-06,
            -1.97294057e-06,
            2.50027650e-06,
            8.33850112e-06,
            1.40760658e-05,
            1.95280317e-05,
            2.18363607e-05,
            1.90716366e-05,
            1.23657844e-05,
            2.63204407e-06,
            -6.98323587e-06,
            -1.38752052e-05,
            -1.74493461e-05,
            -1.75393132e-05,
            -1.60858779e-05,
            -1.38304769e-05,
            -1.07066814e-05,
            -6.86439888e-06,
            -8.67836788e-07,
            6.24067733e-06,
            1.26193829e-05,
        ]
    )
    data = response.data_probe_response(probe=[probe1, probe2], probe_units="degree")
    assert_allclose(data["probe_resp[0]"].to_numpy(), resp_prob1)
    assert_allclose(data["probe_resp[1]"].to_numpy(), resp_prob2)
