from ross.units import Q_, check_units


class Probe:
    """Class of a probe.

    This class will create a probe object to be used in the rotor model.
    The probe is used to measure the response of the rotor at a specific
    location and orientation.

    Parameters
    ----------
    node : int
        Indicate the node where the probe is located.
    angle : float, pint.Quantity
        Probe orientation angle about the shaft (rad).
    tag : str, optional
        A tag to name the element.

    Example
    -------
    >>> import ross as rs
    >>> probe1 = rs.Probe(10, Q_(45, "degree"), "Probe Drive End - X")
    >>> probe1.node
    10
    >>> probe1.angle
    0.7853981633974483
    """

    @check_units
    def __init__(self, node, angle, tag=None):
        self.node = node
        self.angle = angle
        if tag is None:
            self.tag = f"Probe - Node {self.node}, Angle {self.angle}"
        else:
            self.tag = tag

    @property
    def info(self):
        return self.node, self.angle

    def __str__(self):
        return (
            f"Probe {self.tag}"
            f'\n{20*"-"}'
            f"\nNode location           : {self.node}"
            f"\nProbe orientation angle (rad) : {self.angle}"
        )
