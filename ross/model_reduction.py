import numpy as np


class ModelReduction:
    """
    Base class for model reduction methods.
    """

    def __init__(
        self, rotor, speed, include_nodes=[], method="guyan", limit_percent=0.5
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

        include_nodes.append(min(rotor.nodes))
        include_nodes.append(max(rotor.nodes))

        if method == "guyan":
            self.model_reduction_technique = self.guyan
        elif method == "guyan_melhorado":
            self.model_reduction_technique = self.guyan_melhorado
        else:
            raise ValueError(f"Pass a existing {method}.")

        self.ignored_dofs, self.selected_dofs = self.separate_dofs(
            include_nodes, limit_percent
        )
        self.reduce_model(self.ignored_dofs, self.selected_dofs)

        print("Applied technique =", method)

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

        ignored_dofs = set(range(rotor.ndof)) - selected_dofs

        return sorted(ignored_dofs), sorted(selected_dofs)

    def separate_dofs(self, include_nodes=[], limit_percent=0.5):
        # Sort DOFs by mass-stiffness ratio (M/K)
        M_K = np.diag(self.K) / np.diag(self.M)
        ordered_dofs = np.argsort(M_K)[::-1]

        limit = int(len(ordered_dofs) * limit_percent)

        selected_dofs = set()
        selected_dofs.update(ordered_dofs[:limit])

        for n in include_nodes:
            dofs = n * self.number_dof + np.arange(self.number_dof)
            selected_dofs.update(dofs)

        ignored_dofs = set(range(self.ndof)) - selected_dofs

        return sorted(ignored_dofs), sorted(selected_dofs)

    @staticmethod
    def rearrange_matrix(matrix, ignored_dofs, selected_dofs):
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

    def get_transformation_functions(self, transf_matrix, ignored_dofs, selected_dofs):
        reordering = selected_dofs + ignored_dofs

        inverse_order = np.zeros_like(reordering)
        for i, idx in enumerate(reordering):
            inverse_order[idx] = i

        # Verificar - talvez seja melhor separar em várias funções
        reduce_matrix = (
            lambda array: (
                transf_matrix.T
                @ self.rearrange_matrix(array, ignored_dofs, selected_dofs)
            )
            @ transf_matrix
        )

        def reduce_vector(array):
            if array.ndim == 1:
                array_reduced = transf_matrix.T @ array[reordering]
            else:
                array_reduced = transf_matrix.T @ array[reordering, :]
            return array_reduced

        def revert_vector(array):
            array_reordered = transf_matrix @ array

            array_full = np.zeros_like(array_reordered)
            if array_reordered.ndim == 1:
                array_full[reordering] = array_reordered
            else:
                array_full[reordering, :] = array_reordered
            return array_full

        return reduce_matrix, reduce_vector, revert_vector

    def increment_nodes(self, add_nodes=[]):
        num_dof = self.number_dof

        for n in add_nodes:
            dofs = range(n * num_dof + 0, n * num_dof + num_dof)
            self.slaves_dofs.extends(dofs)

            for dof in dofs:
                self.selected_dofs.remove(dof)

        self.reduce_model(self.ignored_dofs, self.selected_dofs)

    def reduce_model(self, ignored_dofs, selected_dofs):
        transf_matrix = self.model_reduction_technique(ignored_dofs, selected_dofs)

        functions = self.get_transformation_functions(
            transf_matrix, ignored_dofs, selected_dofs
        )

        reduce_matrix, reduce_vector, revert_vector = functions

        self.transf_matrix = transf_matrix
        self.reduce_matrix = reduce_matrix
        self.reduce_vector = reduce_vector
        self.revert_vector = revert_vector

    def optimize(self):
        n = 0
        aux_count = 0
        alldofs = np.arange(self.ndof)

        error = 1
        tol = 1e-6

        while error > tol:
            speed_dot = 0

            # # Select slave DOFs (less relevant ones)
            # ignored_dofs = np.sort(range(self.ndof))
            # ignored_dofs = ignored_dofs[~np.isin(ignored_dofs, inputs)]
            # ignored_dofs = ignored_dofs[~np.isin(ignored_dofs, outputs)]

            # if n != 0:
            #     ignored_dofs = ignored_dofs[~np.isin(ignored_dofs, self.indx[-n:])]

            # # Retained DOFs are the remaining ones
            # selected_dofs = np.sort(alldofs[~np.isin(alldofs, ignored_dofs)])

            # # Indices of inputs and outputs in the reduced model
            # inputs_r = np.where(np.isin(selected_dofs, inputs))[0]
            # outputs_r = np.where(np.isin(selected_dofs, outputs))[0]

            # transform_functions = self.model_reduction_technique(ignored_dofs, selected_dofs)

            # # Compute FRFs for both full and reduced models for comparison
            # FRF_Full, FRF_Red = Rm.compare_FRF_guyan(
            #     M, K, C, G, Kst, Mr, Kr, Cr, Gr, Kstr,
            #     frequency_range, inputs, outputs, speed,
            #     inputs_r, outputs_r, self.model_reduction_technique, speed_dot,n,fig_plot=True)

            # # Store relative error between FRFs
            # error = np.linalg.norm(FRF_Full - FRF_Red) / np.linalg.norm(FRF_Full)

            n += 3
            aux_count += 1

    def guyan(self, ignored_dofs, selected_dofs):
        """
        Standard Guyan Reduction method.
        """
        K = self.K

        # Compute transformation matrix
        Kia = (
            -np.linalg.pinv(K[np.ix_(ignored_dofs, ignored_dofs)])
            @ K[np.ix_(ignored_dofs, selected_dofs)]
        )
        Tg = np.block([[np.eye(Kia.shape[1])], [Kia]])

        return Tg

    def guyan_melhorado(self, ignored_dofs, selected_dofs):
        M = self.M
        K = self.K

        # Compute transformation matrix using inverse of slave stiffness matrix
        Kia = (
            -np.linalg.inv(K[np.ix_(ignored_dofs, ignored_dofs)])
            @ K[np.ix_(ignored_dofs, selected_dofs)]
        )
        Tg = np.block([[np.eye(Kia.shape[1])], [Kia]])

        n = K.shape[0]

        Kss = np.linalg.inv(K[np.ix_(ignored_dofs, ignored_dofs)])

        # Build flexibility matrix
        Kfi = np.zeros_like(K)
        if Kss.shape == (1, 1):
            Kfi[-1, -1] = Kss[0, 0]
        else:
            start = n - Kss.shape[0]
            Kfi[start:, start:] = Kss

        # Reduced mass and stiffness matrices via Guyan transformation
        Mrr = self.rearrange_matrix(M, ignored_dofs, selected_dofs)
        Krr = self.rearrange_matrix(K, ignored_dofs, selected_dofs)
        Mr = Tg.T @ Mrr @ Tg
        Kr = Tg.T @ Krr @ Tg

        # IRS transformation (Improved Reduced System)
        Tirs = Tg + Kfi @ M @ Tg @ np.linalg.inv(Mr) @ Kr

        return Tirs
