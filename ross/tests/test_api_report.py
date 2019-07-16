import os

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_allclose, assert_equal

from ross.bearing_seal_element import *
from ross.disk_element import *
from ross.materials import steel
from ross.rotor_assembly import *
from ross.rotor_assembly import MAC_modes
from ross.api_report import *
from ross.shaft_element import *

test_dir = os.path.dirname(__file__)


@pytest.fixture
def rotor1():
    # Rotor with damping, 7 regions, 14 shaft elements, 3 disks, 2 bearings
    length = [0.1, 0.6, 0.3, 1.2, 0.4, 0.8, 0.2]
    outer_diam = [0.08, 0.12, 0.18, 0.25, 0.16, 0.13, 0.07]
    inner_diam = [0, 0, 0, 0, 0, 0, 0]
    disk_data = [
        DiskElement.from_geometry(n=2, material=steel, width=0.05, i_d=0.12, o_d=0.4),
        DiskElement.from_geometry(n=3, material=steel, width=0.05, i_d=0.18, o_d=0.5), 
        DiskElement.from_geometry(n=4, material=steel, width=0.05, i_d=0.16, o_d=0.4),
    ]
    brg_seal_data=[
        BearingElement(n=0, kxx=1e7, cxx=3e4, kyy=1e7, cyy=3e4, kxy=0, cxy=0, kyx=0, cyx=0),
        BearingElement(n=7, kxx=1e7, cxx=3e4, kyy=1e7, cyy=3e4, kxy=0, cxy=0, kyx=0, cyx=0),
    ]

    return Rotor.from_section(
                leng_data=length,
                o_ds_data=outer_diam,
                i_ds_data=inner_diam,
                disk_data=disk_data,
                brg_seal_data=brg_seal_data,
                w=0,
                nel_r=2
           )
