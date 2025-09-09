import numpy as np
from numpy import linalg as la
from scipy.linalg import eigh


class ModelReduction:
    """
    Base class for model reduction methods.
    """

    def __init__(
        self,
        rotor,
        speed,
        include_dofs=[],
        include_nodes=[],
        method="guyan",
        limit_percent=0.15,
    ):
        """
        Initialize the model reduction with a given model.

        Parameters
        ----------
        model : ross.Rotor
            The rotor model to be reduced.
        """

        self.ndof = rotor.ndof
        self.number_dof = rotor.number_dof
        self.K = rotor.K(speed)
        self.M = rotor.M(speed)

        self.ignored_dofs = None
        self.selected_dofs = None
        self.reordering = None
        self.transf_matrix = None

        try:
            self.model_reduction_technique = getattr(self, method)
        except AttributeError:
            print(f"Method {method} not exists in ModelReduction.")

        print("Applied technique =", method)
        self.reduce_model(include_dofs, include_nodes, limit_percent)

        n_selected = len(self.selected_dofs)
        print(
            f"Number of selected DOFs = {n_selected} / {self.ndof} ({n_selected / self.ndof * 100:.2f}%)"
        )

    @staticmethod
    def select_nodes_based_rotor(rotor, include_nodes=[]):
        selected_dofs = set()

        elements = [
            *rotor.disk_elements,
            *rotor.bearing_elements,
            *rotor.point_mass_elements,
        ]

        for elm in elements:
            if elm.n not in rotor.nodes:
                continue

            dofs = list(elm.dof_global_index.values())
            selected_dofs.update(dofs)

        for n in include_nodes:
            dofs = n * rotor.number_dof + np.arange(rotor.number_dof)
            selected_dofs.update(dofs)

        ignored_dofs = sorted(set(range(rotor.ndof)) - selected_dofs)
        selected_dofs = sorted(selected_dofs)

        return selected_dofs, ignored_dofs

    def separate_dofs(self, include_dofs=[], include_nodes=[], limit_percent=0.15):
        # Sort DOFs by mass-stiffness ratio (M/K)
        with np.errstate(divide="ignore"):
            diag_K = np.diag(self.K)
            M_K = np.where(diag_K != 0, np.diag(self.M) / diag_K, 0)

        ordered_dofs = np.argsort(M_K)[::-1]

        limit = int(self.ndof * limit_percent)

        selected_dofs = set()
        selected_dofs.update(ordered_dofs[:limit])
        selected_dofs.update(include_dofs)

        for n in include_nodes:
            dofs = n * self.number_dof + np.arange(self.number_dof)
            selected_dofs.update(dofs)

        ignored_dofs = sorted(set(range(self.ndof)) - selected_dofs)
        selected_dofs = sorted(selected_dofs)

        return selected_dofs, ignored_dofs

    @staticmethod
    def rearrange_matrix(matrix, selected_dofs, ignored_dofs):
        return np.block(
            [
                [
                    matrix[np.ix_(selected_dofs, selected_dofs)],
                    matrix[np.ix_(selected_dofs, ignored_dofs)],
                ],
                [
                    matrix[np.ix_(ignored_dofs, selected_dofs)],
                    matrix[np.ix_(ignored_dofs, ignored_dofs)],
                ],
            ]
        )

    def reduce_matrix(self, array):
        return (
            self.transf_matrix.T
            @ self.rearrange_matrix(array, self.selected_dofs, self.ignored_dofs)
            @ self.transf_matrix
        )

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

    def increment_nodes(self, add_nodes=[]):
        num_dof = self.number_dof

        for n in add_nodes:
            dofs = range(n * num_dof + 0, n * num_dof + num_dof)
            self.slaves_dofs.extend(dofs)

            for dof in dofs:
                self.selected_dofs.remove(dof)

        self.reduce_model()

    def reduce_model(self, include_dofs=[], include_nodes=[], limit_percent=0.15):
        if self.selected_dofs is None:
            self.selected_dofs, self.ignored_dofs = self.separate_dofs(
                include_dofs, include_nodes, limit_percent
            )
        self.reordering = self.selected_dofs + self.ignored_dofs
        self.transf_matrix = self.model_reduction_technique(
            self.selected_dofs, self.ignored_dofs
        )

    def guyan(self, selected_dofs, ignored_dofs):
        """
        Standard Guyan Reduction method.
        """
        K = self.K

        n_selected = len(selected_dofs)
        I = np.eye(n_selected)

        Kss = K[np.ix_(ignored_dofs, ignored_dofs)]
        Ksm = K[np.ix_(ignored_dofs, selected_dofs)]

        # Compute transformation matrix
        Tg = np.vstack((I, -la.pinv(Kss) @ Ksm))

        return Tg

    def improved_guyan(self, selected_dofs, ignored_dofs):
        M = self.M
        K = self.K

        Tg = self.guyan(selected_dofs, ignored_dofs)

        Kss = K[np.ix_(ignored_dofs, ignored_dofs)]

        # Build flexibility matrix
        Kfi = np.zeros_like(K)
        if Kss.shape == (1, 1):
            Kfi[-1, -1] = Kss[0, 0]
        else:
            start = K.shape[0] - Kss.shape[0]
            Kfi[start:, start:] = Kss

        # Reduced mass and stiffness matrices via Guyan transformation
        Mrr = self.rearrange_matrix(M, selected_dofs, ignored_dofs)
        Krr = self.rearrange_matrix(K, selected_dofs, ignored_dofs)
        Mr = Tg.T @ Mrr @ Tg
        Kr = Tg.T @ Krr @ Tg

        # IRS transformation (Improved Reduced System)
        Tirs = Tg + la.pinv(Kfi) @ M @ Tg @ la.pinv(Mr) @ Kr

        return Tirs

    def serep(self, selected_dofs, ignored_dofs):
        M = self.M
        K = self.K

        Mr = self.rearrange_matrix(M, selected_dofs, ignored_dofs)
        Kr = self.rearrange_matrix(K, selected_dofs, ignored_dofs)

        n_selected = len(selected_dofs)
        I = np.eye(n_selected)

        _, Phi_r = eigh(Kr, Mr)
        Phi_mm = Phi_r[:n_selected, :n_selected]
        Phi_sm = Phi_r[n_selected:, :n_selected]

        Ts = np.vstack((I, Phi_sm @ la.pinv(Phi_mm)))

        return Ts
