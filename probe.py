class Probe():
    """Class of a probe.
    
    Parameters
    ----------
    node : int -> Indicate the node where the probe is located.
    angle : float, pint.Quantity -> Probe orientation angle about the shaft.
    tag : str, optional -> Probe tag to be add a DataFrame column title.
        
    Example
    -------
    >>> probe1 = Probe(10,Q_(45,"degree"),"V1")
    >>> probe1.info
    (10, pi/4)
    """

    @check_units
    def __init__(self, node, angle, tag=None):
        self.node = node
        self.angle = angle
        if tag is not None:
            self.tag = tag

    @property        
    def info(self):
        return (self.node, self.angle)

    def __str__(self):
        return (
            f"{self.tag}"
            f'\n{20*"-"}'
            f"\Node location           : {self.node}"
            f"\nProbe orientation angle (rad) : {self.angle}"
        )