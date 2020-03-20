from ross.stochastic.st_bearing_seal_element import ST_BearingElement
from ross.stochastic.st_disk_element import ST_DiskElement
from ross.stochastic.st_point_mass import ST_PointMass
from ross.stochastic.st_shaft_element import ST_ShaftElement

__all__ = ["ST_Rotor"]


class ST_Rotor(object):
    r"""A random rotor object.

    This class will create several rotors according to random elements passed
    to the arguments.
    The number of rotors to be created depends on the amount of random
    elements instantiated and theirs respective sizes.

    Parameters
    ----------
    shaft_elements : list
        List with the shaft elements
    disk_elements : list
        List with the disk elements
    bearing_elements : list
        List with the bearing elements
    point_mass_elements: list
        List with the point mass elements
    sparse : bool, optional
        If sparse, eigenvalues will be calculated with arpack.
        Default is True.
    n_eigen : int, optional
        Number of eigenvalues calculated by arpack.
        Default is 12.
    tag : str
        A tag for the rotor

    Attributes
    ----------

    Returns
    -------

    Examples
    --------
    """

    def __init__(
        self,
        shaft_elements,
        disk_elements=None,
        bearing_elements=None,
        point_mass_elements=None,
        sparse=True,
        n_eigen=12,
        min_w=None,
        max_w=None,
        rated_w=None,
        tag=None,
    ):

        it = iter(
            [elm for elm in shaft_elements if isinstance(elm, ST_ShaftElement)]
        )
        the_len = len(next(it))
        if not all(len(l) == the_len for l in it):
            raise ValueError(
                "not all random shaft elements lists have same length."
            )

        it = iter(
            [elm for elm in disk_elements if isinstance(elm, ST_DiskElement)]
        )
        the_len = len(next(it))
        if not all(len(l) == the_len for l in it):
            raise ValueError(
                "not all random disk elements lists have same length."
            )

        it = iter(
            [
                elm
                for elm in bearing_elements
                if isinstance(elm, ST_BearingElement)
            ]
        )
        the_len = len(next(it))
        if not all(len(l) == the_len for l in it):
            raise ValueError(
                "not all random bearing elements lists have same length."
            )

        it = iter(
            [
                elm
                for elm in point_mass_elements
                if isinstance(elm, ST_PointMass)
            ]
        )
        the_len = len(next(it))
        if not all(len(l) == the_len for l in it):
            raise ValueError(
                "not all random point mass lists have same length."
            )
