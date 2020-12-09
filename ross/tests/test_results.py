from pathlib import Path
from tempfile import tempdir

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal

from ross.rotor_assembly import *
from ross.results import *


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

    assert response2.wn.all() == response.wn.all()
    assert response2.wd.all() == response.wd.all()
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
    assert np.array(response2.shaft_elements_length).all() == np.array(response.shaft_elements_length).all()


def test_save_load_freqresponse(rotor1):
    speed = np.linspace(0, 1000, 51)
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
