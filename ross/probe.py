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
        Probe orientation angle about the shaft (rad) if the probe is in
        the radial direction. Default is None.
    direction : str, optional
        Measurement direction of the probe, which can be 'radial' or
        'axial'. Default is 'radial'.
    tag : str, optional
        A tag to name the element.

    Example
    -------
    >>> from ross import Probe
    >>> probe1 = Probe(10, Q_(45, "deg"), tag="Probe Drive End - X")
    >>> probe1.node
    10
    >>> probe1.angle
    0.7853981633974483
    >>> probe1.direction
    'radial'
    >>> probe2 = Probe(3, direction="axial", tag="Probe AX - Z")
    >>> probe2.node
    3
    >>> probe2.direction
    'axial'
    """

    @check_units
    def __init__(self, node, angle=None, direction="radial", tag=None):
        self.node = node
        self.angle = angle
        self.direction = direction
        self.tag = tag

        if self.direction not in ("radial", "axial"):
            raise ValueError("Direction must be 'radial' or 'axial'")

        if self.direction == "radial" and self.angle is None:
            raise ValueError("Angle must be provided when direction is 'radial'")
        elif self.direction == "axial":
            self.angle = None

    @property
    def info(self):
        return self.node, self.angle, self.direction

    def __str__(self):
        tag_str = "Probe" if self.tag is None else f"Probe {self.tag}"
        nod_str = f"\nNode location                 : {self.node}"
        dir_str = f"\nProbe direction               : {self.direction}"
        ang_str = (
            f"\nProbe orientation angle (rad) : {self.angle}"
            if self.direction == "radial"
            else ""
        )

        return f"{tag_str}\n{20 * '-'}{nod_str}{dir_str}{ang_str}"

    def get_label(self, id=None):
        label = "Probe" if id is None else f"Probe {id}"
        label += f" - Node {self.node} "

        if self.direction == "radial":
            try:
                angle = Q_(self.angle, "rad").to("deg").m
                label += f"({angle:.0f}°)"
            except TypeError:
                label += f"({self.angle})"
        else:
            label += "(axial direction)"

        return label
