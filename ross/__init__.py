__version__ = "0.4.1"
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
from .utils import visualize_matrix, get_data_from_figure

_pio.templates.default = "ross"
