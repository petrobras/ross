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
    def __init__(self, rotor, speed, num_modes=24, **kwargs):
        self.num_modes = num_modes
        self.bearings = [
            b for b in rotor.bearing_elements if b.n not in rotor.link_nodes
        ]
        self.M = rotor.M(speed)
        self.K = rotor.K(speed)
        self.transf_matrix = self.get_transformation_matrix(speed)

    def get_transformation_matrix(self, speed):
        M = self.M
        K_aux = self.K.copy()

        # Cancel cross-coupled coefficients of bearing stiffness matrix
        cancel_cross_coeffs = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])

        for elm in self.bearings:
            dofs = list(elm.dof_global_index.values())
            if elm.n_link is None:
                K_aux[np.ix_(dofs, dofs)] -= elm.K(speed) * cancel_cross_coeffs
            else:
                K_aux[np.ix_(dofs, dofs)] -= elm.K(speed) * np.tile(
                    cancel_cross_coeffs, (2, 2)
                )

        _, modal_matrix = eigh(K_aux, M)
        modal_matrix = modal_matrix[:, : self.num_modes]

        return modal_matrix

    def reduce_matrix(self, array):
        return self.transf_matrix.T @ array @ self.transf_matrix

    def reduce_vector(self, array):
        return self.transf_matrix.T @ array

    def revert_vector(self, array_reduced):
        return self.transf_matrix @ array_reduced


class Guyan(ModelReduction):
    def __init__(self, rotor, speed, ndof_limit, include_dofs=[], include_nodes=[]):
        self.ndof = rotor.ndof
        self.number_dof = rotor.number_dof
        self.M = rotor.M(speed)
        self.K = rotor.K(speed)

        self.ndof_limit = int(self.ndof * 0.15) if ndof_limit is None else ndof_limit

        self.selected_dofs, self.ignored_dofs = self.separate_dofs(
            include_dofs, include_nodes
        )
        self.reordering = self.selected_dofs + self.ignored_dofs
        self.transf_matrix = self.get_transformation_matrix()

    def separate_dofs(self, include_dofs=[], include_nodes=[]):
        # Sort DOFs by mass-stiffness ratio (M/K)
        with np.errstate(divide="ignore"):
            diag_K = np.diag(self.K)
            M_K = np.where(diag_K != 0, np.diag(self.M) / diag_K, 0)

        ordered_dofs = np.argsort(M_K)[::-1]

        selected_dofs = set()
        selected_dofs.update(include_dofs)

        for n in include_nodes:
            dofs = n * self.number_dof + np.arange(self.number_dof)
            selected_dofs.update(dofs)

        n = self.ndof_limit - len(selected_dofs)
        i = 0
        while n > 0:
            selected_dofs.update(ordered_dofs[i:n])
            i += n
            n = self.ndof_limit - len(selected_dofs)

        ignored_dofs = sorted(set(range(self.ndof)) - selected_dofs)
        selected_dofs = sorted(selected_dofs)

        return selected_dofs, ignored_dofs

    def get_transformation_matrix(self):
        K = self.K

        n_selected = len(self.selected_dofs)
        I = np.eye(n_selected)

        Kss = K[np.ix_(self.ignored_dofs, self.ignored_dofs)]
        Ksm = K[np.ix_(self.ignored_dofs, self.selected_dofs)]

        # Compute transformation matrix
        Tg = np.vstack((I, -la.pinv(Kss) @ Ksm))

        return Tg

    def rearrange_matrix(self, matrix):
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
        return self.transf_matrix.T @ self.rearrange_matrix(array) @ self.transf_matrix

    def reduce_vector(self, array):
        if array.ndim == 1:
            array_reduced = self.transf_matrix.T @ array[self.reordering]
        else:
            array_reduced = self.transf_matrix.T @ array[self.reordering, :]

        return array_reduced

    def revert_vector(self, array_reduced):
        array_transf = self.transf_matrix @ array_reduced
        array = np.zeros_like(array_transf)

        if array_transf.ndim == 1:
            array[self.reordering] = array_transf
        else:
            array[self.reordering, :] = array_transf

        return array
