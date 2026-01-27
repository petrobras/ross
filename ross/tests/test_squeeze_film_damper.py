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
    journal_radius=Q_(2.55, "inches"),
    radial_clearance=Q_(0.003, "inches"),
    eccentricity_ratio=0.5,
    lubricant = "ISOVG32",
    groove=True,
    end_seals=True,
    cav=True,
)

    return bearing


def test_squeeze_film_damper(squeeze_film_damper):
    assert_allclose(squeeze_film_damper.kxx, 1.69362187e+08, rtol=0.0001)
    assert_allclose(squeeze_film_damper.kyy, 1.69362187e+08, rtol=0.0001)
    assert_allclose(squeeze_film_damper.cxx, 118283.83590277865, rtol=0.0001)
    assert_allclose(squeeze_film_damper.cyy, 118283.83590277865, rtol=0.0001)
    assert_allclose(squeeze_film_damper.frequency[0], 1947.78744523, rtol=0.0001)

