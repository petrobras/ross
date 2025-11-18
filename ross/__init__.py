__version__ = "2.0.0"
import sys
from plotly import io as _pio

import ross.plotly_theme

from .bearing_seal_element import *
from .faults import *
from .disk_element import *
from .gear_element import *
from .materials import *
from .point_mass import *
from .probe import *
from .results import *
from .rotor_assembly import *
from .multi_rotor import *
from .shaft_element import *
from .coupling_element import *
from .units import Q_
from .utils import get_data_from_figure, visualize_matrix
from ross.bearings.lubricants import lubricants_dict
from ross.bearings.plain_journal import *
from ross.bearings.thrust_pad import *
from ross.bearings.tilting_pad import *
from ross.model_reduction import *
from ross.seals.labyrinth_seal import *
from ross.seals.holepattern_seal import *
from ross.seals.hybrid_seal import *

_pio.templates.default = "ross"
if "ipykernel" in sys.modules:
    _pio.renderers.default = "notebook"
elif "google.colab" in sys.modules:
    _pio.renderers.default = "colab"
else:
    _pio.renderers.default = "browser"
