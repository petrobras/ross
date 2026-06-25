from ross import MagneticBearingElement


def get_ambs(rotor) -> list:
    """
    Get magnetic bearing elements from a rotor, ordered by node number.

    Parameters
    ----------
    rotor : ross.Rotor
        The rotor model.

    Returns
    -------
    ambs : list
        A list of MagneticBearingElement objects found in the rotor,
        sorted by the node (n) in which they are located.

    Examples
    --------
    >>> import ross as rs
    >>> rotor_amb = rs.rotor_example_amb_complex_controllers()
    >>> ambs = get_ambs(rotor_amb)
    >>> len(ambs)
    2
    """
    ambs = [
        brg for brg in rotor.bearing_elements if isinstance(brg, MagneticBearingElement)
    ]
    return sorted(ambs, key=lambda brg: brg.n)


def has_ambs(rotor):
    """
    Check if the rotor has magnetic bearing elements.

    Parameters
    ----------
    rotor : ross.Rotor
        The rotor model.

    Returns
    -------
    has_ambs : bool
        True if the rotor has magnetic bearing elements, False otherwise.

    Examples
    --------
    >>> import ross as rs
    >>> rotor_amb = rs.rotor_example_amb_complex_controllers()
    >>> has_ambs(rotor_amb)
    True
    """
    ambs = get_ambs(rotor)
    return len(ambs) > 0
