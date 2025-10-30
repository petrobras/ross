import warnings
import numpy as np
from numpy import linalg as la
from scipy.linalg import eigh


class ModelReduction:
    subclasses = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        ModelReduction.subclasses[cls.__name__.lower()] = cls

    def __new__(cls, method="guyan", **kwargs):
        if cls is ModelReduction:
            subcls = ModelReduction.subclasses.get(method.lower())

            if subcls is None:
                raise ValueError(f"Method {method} not exists in ModelReduction.")

            return super().__new__(subcls)

        else:
            return super().__new__(cls)


class PseudoModal(ModelReduction):
    """Pseudo-modal method.

    This method can be used to apply modal transformation to reduce model
    of the rotor system.

    Parameters
    ----------
    rotor: rs.Rotor
        The rotor object.
    speed : float
        Rotor speed.
    num_modes : int
        The number of eigenvectors to consider in the modal transformation
        with model reduction. Default is 24.

    Examples
    --------
    >>> import ross as rs
    >>> rotor = rs.rotor_example()
    >>> size = 10000
    >>> node = 3
    >>> speed = 500.0
    >>> t = np.linspace(0, 10, size)
    >>> F = np.zeros((size, rotor.ndof))
    >>> F[:, rotor.number_dof * node + 0] = 10 * np.cos(2 * t)
    >>> F[:, rotor.number_dof * node + 1] = 10 * np.sin(2 * t)
    >>> mr = ModelReduction(rotor=rotor, speed=speed, method="pseudomodal", num_modes=12)
    >>> F_modal = mr.reduce_vector(F.T).T
    >>> la.norm(F_modal) # doctest: +ELLIPSIS
    195.466...
    """

    def __init__(self, rotor, speed, num_modes=24, **kwargs):
        self.num_modes = num_modes
        self.bearings = [
            b for b in rotor.bearing_elements if b.n not in rotor.link_nodes
        ]
        self.M = rotor.M(speed)
        self.K = rotor.K(speed)
        self.transf_matrix = self.get_transformation_matrix(speed)

    def get_transformation_matrix(self, speed):
        """Build modal matrix

        Parameters
        ----------
        speed: np.ndarray
            Rotor speed

        Returns
        -------
        modal_matrix : np.ndarray
            Modal matrix for the pseudo-modal method.
        """
        M = self.M
        K_aux = self.K.copy()

        # Eliminate cross-coupled coefficients of bearing stiffness matrix
        for elm in self.bearings:
            dofs = list(elm.dof_global_index.values())
            elim_factor = 1 - np.eye(len(dofs))
            K_aux[np.ix_(dofs, dofs)] -= elm.K(speed) * elim_factor

        _, modal_matrix = eigh(K_aux, M)
        modal_matrix = modal_matrix[:, : self.num_modes]

        return modal_matrix

    def reduce_matrix(self, array):
        """Transform a square matrix from physical to modal space.

        Parameters
        ----------
        array: np.ndarray
            Square matrix to be transformed.

        Returns
        -------
        array_reduced : np.ndarray
            Reduced matrix.
        """
        return self.transf_matrix.T @ array @ self.transf_matrix

    def reduce_vector(self, array):
        """Transform a vector from physical to modal space.

        Parameters
        ----------
        array: np.ndarray
            Vector to be transformed.

        Returns
        -------
        array_reduced : np.ndarray
            Reduced vector.
        """
        return self.transf_matrix.T @ array

    def revert_vector(self, array_reduced):
        """Transform a vector from modal to physical space.

        Parameters
        ----------
        array_reduced: np.ndarray
            Reduced vector to be reverted.

        Returns
        -------
        array : np.ndarray
            Vector in physical space.
        """
        return self.transf_matrix @ array_reduced


class Guyan(ModelReduction):
    """Guyan reduction method.

    This method can be used to reduce model of the rotor system
    to a defined list of degrees of freedom (DOF).

    Parameters
    ----------
    rotor: rs.Rotor
        The rotor object.
    speed : float
        Rotor speed.
    include_nodes : list of int, optional
        List of the nodes to be included in the reduction.
    dof_mapping : list of str, optional
        List of the local DOFs to be considered in the reduction.
        Valid values are: 'x', 'y', 'z', 'alpha', 'beta', 'theta', corresponding to:
            - 'x' and 'y': lateral translations
            - 'z': axial translation
            - 'alpha': rotation about the x-axis
            - 'beta': rotation about the y-axis
            - 'theta': torsional rotation (about the z-axis)
        Default is ['x', 'y'].
    include_dofs : list of int, optional
        List of DOFs to be included in the reduction,
        e.g., DOFs with applied forces or probe locations.

    Examples
    --------
    >>> import ross as rs
    >>> rotor = rs.rotor_example()
    >>> size = 10000
    >>> node = 3
    >>> speed = 500.0
    >>> t = np.linspace(0, 10, size)
    >>> F = np.zeros((size, rotor.ndof))
    >>> dofx = rotor.number_dof * node + 0
    >>> dofy = rotor.number_dof * node + 1
    >>> F[:, dofx] = 10 * np.cos(2 * t)
    >>> F[:, dofy] = 10 * np.sin(2 * t)
    >>> mr = ModelReduction(
    ...     rotor=rotor,
    ...     speed=speed,
    ...     method="guyan",
    ...     include_dofs=[dofx, dofy]
    ... )
    >>> F_reduct = mr.reduce_vector(F.T).T
    >>> np.where(F_reduct[5000, :] != 0)[0] # doctest: +ELLIPSIS
    array([4, 5])
    """

    def __init__(
        self,
        rotor,
        speed,
        include_nodes=None,
        dof_mapping=None,
        include_dofs=None,
        **kwargs,
    ):
        self.rotor = rotor
        self.ndof = rotor.ndof
        self.number_dof = rotor.number_dof
        self.M = rotor.M(speed)
        self.K = rotor.K(speed)

        if include_nodes is None:
            include_nodes = []
        if dof_mapping is None:
            dof_mapping = ["x", "y"]
        if include_dofs is None:
            include_dofs = []

        dof_dict = {"x": 0, "y": 1, "z": 2, "alpha": 3, "beta": 4, "theta": 5}
        local_dofs = [dof_dict[dof] for dof in dof_mapping]

        self.selected_dofs, self.ignored_dofs = self._separate_dofs(
            include_nodes, local_dofs, include_dofs
        )

        self.reordering = self.selected_dofs + self.ignored_dofs
        self.transf_matrix = self.get_transformation_matrix()

    def _select_elem_dofs(self, local_dofs):
        """Select DOFs from rotor bearings and disks"""
        selected_dofs = []
        elements = self.rotor.bearing_elements + self.rotor.disk_elements

        for elm in elements:
            if elm.n in self.rotor.link_nodes:
                dofs = np.array(list(elm.dof_global_index.values()))
                local_dofs_l = list(filter(lambda dof: dof < 3, local_dofs))
                include_dofs = dofs[local_dofs_l]
                for dof in include_dofs:
                    selected_dofs.append(dof)
            else:
                for i in local_dofs:
                    selected_dofs.append(elm.n * self.number_dof + i)

        return selected_dofs

    def _separate_dofs(self, include_nodes=None, local_dofs=None, include_dofs=None):
        """Separate the selected DOFs from the ignored DOFs."""
        if include_nodes is None:
            include_nodes = []
        if include_dofs is None:
            include_dofs = []
        if not local_dofs:
            local_dofs = [0, 1]

        selected_dofs = set()
        selected_dofs.update(include_dofs)
        selected_dofs.update(self._select_elem_dofs(local_dofs))

        for n in include_nodes:
            dofs = n * self.number_dof + np.array(local_dofs)
            selected_dofs.update(dofs)

        ignored_dofs = sorted(set(range(self.ndof)) - selected_dofs)
        selected_dofs = sorted(selected_dofs)

        return selected_dofs, ignored_dofs

    def get_transformation_matrix(self):
        """Build transformation matrix

        Returns
        -------
        Tg : np.ndarray
            Transformation matrix for Guyan method.
        """
        K = self.K

        n_selected = len(self.selected_dofs)
        I = np.eye(n_selected)

        Kss = K[np.ix_(self.ignored_dofs, self.ignored_dofs)]
        Ksm = K[np.ix_(self.ignored_dofs, self.selected_dofs)]

        # Compute transformation matrix
        try:
            inv_Kss = la.inv(Kss)
        except np.linalg.LinAlgError as err:
            warnings.warn(
                f"{err} error. Using the pseudo-inverse to proceed.", UserWarning
            )
            inv_Kss = la.pinv(Kss)

        Tg = np.vstack((I, -inv_Kss @ Ksm))

        return Tg

    def _rearrange_matrix(self, matrix):
        """Rearrange matrix based on selected and ignored DOFs"""
        return np.block(
            [
                [
                    matrix[np.ix_(self.selected_dofs, self.selected_dofs)],
                    matrix[np.ix_(self.selected_dofs, self.ignored_dofs)],
                ],
                [
                    matrix[np.ix_(self.ignored_dofs, self.selected_dofs)],
                    matrix[np.ix_(self.ignored_dofs, self.ignored_dofs)],
                ],
            ]
        )

    def reduce_matrix(self, array):
        """Transform a square matrix from complete to reduced model.

        Parameters
        ----------
        array: np.ndarray
            Square matrix to be transformed.

        Returns
        -------
        array_reduced : np.ndarray
            Reduced matrix.
        """
        return self.transf_matrix.T @ self._rearrange_matrix(array) @ self.transf_matrix

    def reduce_vector(self, array):
        """Transform a vector from complete to reduced model.

        Parameters
        ----------
        array: np.ndarray
            Vector to be transformed.

        Returns
        -------
        array_reduced : np.ndarray
            Reduced vector.
        """
        if array.ndim == 1:
            array_reduced = self.transf_matrix.T @ array[self.reordering]
        else:
            array_reduced = self.transf_matrix.T @ array[self.reordering, :]

        return array_reduced

    def revert_vector(self, array_reduced):
        """Transform a vector from reduced to complete model.

        Parameters
        ----------
        array_reduced: np.ndarray
            Reduced vector to be reverted.

        Returns
        -------
        array : np.ndarray
            Vector of complete model.
        """
        array_transf = self.transf_matrix @ array_reduced
        array = np.zeros_like(array_transf)

        if array_transf.ndim == 1:
            array[self.reordering] = array_transf
        else:
            array[self.reordering, :] = array_transf

        return array
