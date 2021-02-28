__version__ = "1.0.0rc2"
from plotly import io as _pio

import ross.plotly_theme

from .bearing_seal_element import *
from .defects import *
from .disk_element import *
from .materials import *
from .point_mass import *
from .results import *
from .rotor_assembly import *
from .shaft_element import *
from .utils import get_data_from_figure, visualize_matrix

_pio.templates.default = "ross"
