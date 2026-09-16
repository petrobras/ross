import pytest
from numpy.testing import assert_allclose

from ross.bearings.squeeze_film_damper import SqueezeFilmDamper
from ross.units import Q_


@pytest.fixture
def squeeze_film_damper():
    bearing = SqueezeFilmDamper(
        n=0,
        frequency=Q_([18600], "rpm"),
        axial_length=Q_(0.9, "inches"),
        journal_diameter=Q_(5.1, "inches"),
        radial_clearance=Q_(0.003, "inches"),
        eccentricity_ratio=0.5,
        lubricant="ISOVG32",
        geometry="groove-end_seals",
        cavitation=True,
    )

    return bearing


def test_squeeze_film_damper(squeeze_film_damper):
    assert_allclose(squeeze_film_damper.kxx, 1.69362187e08, rtol=0.0001)
    assert_allclose(squeeze_film_damper.kyy, 1.69362187e08, rtol=0.0001)
    assert_allclose(squeeze_film_damper.cxx, 118283.83590277865, rtol=0.0001)
    assert_allclose(squeeze_film_damper.cyy, 118283.83590277865, rtol=0.0001)
    assert_allclose(squeeze_film_damper.p_max, 10248075.8971382, rtol=0.0001)
    assert_allclose(squeeze_film_damper.frequency[0], 1947.78744523, rtol=0.0001)


def test_save_writes_a_frequency_coefficient_table(squeeze_film_damper, tmp_path):
    import ross as rs
    from ross.utils import load_data

    file = tmp_path / "sfd.toml"
    squeeze_film_damper.save(file)
    data = load_data(file)
    assert list(data) == [f"BearingElement_{squeeze_film_damper.tag}"]
    section = data[f"BearingElement_{squeeze_film_damper.tag}"]
    assert "frequency" in section and "speed" not in section
    assert "journal_diameter" not in section

    for loader in (rs.BearingElement, SqueezeFilmDamper):
        loaded = loader.load(file)
        assert type(loaded) is rs.BearingElement
        assert_allclose(loaded.kxx, squeeze_film_damper.kxx)
        assert_allclose(loaded.cxx, squeeze_film_damper.cxx)
        assert_allclose(loaded.frequency, squeeze_film_damper.frequency)


def test_rotor_with_damper_round_trips(squeeze_film_damper, tmp_path):
    import ross as rs

    rotor = rs.rotor_example()
    squeeze_film_damper.n = 6
    rotor = rs.Rotor(
        rotor.shaft_elements,
        rotor.disk_elements,
        [rotor.bearing_elements[0], squeeze_film_damper],
    )
    file = tmp_path / "rotor.toml"
    rotor.save(file)
    loaded = rs.Rotor.load(file)
    assert_allclose(loaded.bearing_elements[1].cxx, squeeze_film_damper.cxx)
