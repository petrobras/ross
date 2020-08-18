__version__ = "0.4.0"
from plotly import io as _pio

import ross.plotly_theme

from .bearing_seal_element import *
from .disk_element import *
from .materials import *
from .point_mass import *
from .rotor_assembly import *
from .shaft_element import *
from .utils import visualize_matrix

_pio.templates.default = "ross"
