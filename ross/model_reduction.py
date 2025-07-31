import numpy as np


class ModelReduction:
    """
    Base class for model reduction methods.
    """

    def __init__(self, K, M, method="guyan"):
        """
        Initialize the model reduction with a given model.

        Parameters
        ----------
        model : ross.Rotor
            The rotor model to be reduced.
        """

        self.M = M
        self.K = K

        self.ndof = K.shape[0]

        M_K = np.diag(K) / np.diag(M)
        # Sort DOFs by mass-stiffness ratio (M/K)
        self.indx = np.argsort(M_K)[::-1]

        if method == "guyan":
            self.model_reduction_technique = self.guyan
        elif method == "guyan_melhorado":
            self.model_reduction_technique = self.guyan_melhorado
        else:
            raise ValueError(f"Pass a existing {method}.")

        self.run()

        return

    def run(self):
        n = 0
        aux_count = 0
        alldofs = np.arange(self.ndof)

        error = 1
        tol = 1e-6

        while error > tol:
            speed_dot = 0

            # # Select slave DOFs (less relevant ones)
            # slave_dofs = np.sort(range(self.ndof))
            # slave_dofs = slave_dofs[~np.isin(slave_dofs, inputs)]
            # slave_dofs = slave_dofs[~np.isin(slave_dofs, outputs)]

            # if n != 0:
            #     slave_dofs = slave_dofs[~np.isin(slave_dofs, self.indx[-n:])]

            # # Retained DOFs are the remaining ones
            # retained_dofs = np.sort(alldofs[~np.isin(alldofs, slave_dofs)])

            # # Indices of inputs and outputs in the reduced model
            # inputs_r = np.where(np.isin(retained_dofs, inputs))[0]
            # outputs_r = np.where(np.isin(retained_dofs, outputs))[0]

            # transform_functions = self.model_reduction_technique(slave_dofs, retained_dofs)

            # # Compute FRFs for both full and reduced models for comparison
            # FRF_Full, FRF_Red = Rm.compare_FRF_guyan(
            #     M, K, C, G, Kst, Mr, Kr, Cr, Gr, Kstr,
            #     frequency_range, inputs, outputs, speed,
            #     inputs_r, outputs_r, self.model_reduction_technique, speed_dot,n,fig_plot=True)

            # # Store relative error between FRFs
            # error = np.linalg.norm(FRF_Full - FRF_Red) / np.linalg.norm(FRF_Full)

            n += 3
            aux_count += 1

    def remove_dofs_from_matrix(matrix, slave_dofs, retained_dofs):
        return np.block(
            [
                [
                    matrix[np.ix_(retained_dofs, retained_dofs)],
                    matrix[np.ix_(retained_dofs, slave_dofs)],
                ],
                [
                    matrix[np.ix_(slave_dofs, retained_dofs)],
                    matrix[np.ix_(slave_dofs, slave_dofs)],
                ],
            ]
        )

    def get_transformation_functions(self, transformation_matrix):
        reduce_matrix = (
            lambda array: (
                transformation_matrix.T @ self.remove_dofs_from_matrix(array)
            )
            @ transformation_matrix
        )
        reduce_vector = (
            lambda array: transformation_matrix.T @ self.remove_dofs_from_matrix(array)
        )
        get_complete_vector = lambda array: transformation_matrix @ array

        return reduce_matrix, reduce_vector, get_complete_vector

    def guyan(self):
        """
        Standard Guyan Reduction method.
        """
        K = self.K
        slave_dofs = self.slave_dofs
        retained_dofs = self.retained_dofs

        # Compute transformation matrix
        Kia = (
            -np.linalg.pinv(K[np.ix_(slave_dofs, slave_dofs)])
            @ K[np.ix_(slave_dofs, retained_dofs)]
        )
        Tg = np.block([[np.eye(Kia.shape[1])], [Kia]])

        return self.get_transformation_functions(Tg)

    def guyan_melhorado(self, speed, slave_dofs, retained_dofs):
        M = self.M
        K = self.K
        slave_dofs = self.slave_dofs
        retained_dofs = self.retained_dofs

        # Compute transformation matrix using inverse of slave stiffness matrix
        Kia = (
            -np.linalg.inv(K[np.ix_(slave_dofs, slave_dofs)])
            @ K[np.ix_(slave_dofs, retained_dofs)]
        )
        Tg = np.block([[np.eye(Kia.shape[1])], [Kia]])

        n = K.shape[0]

        Kss = np.linalg.inv(K[np.ix_(slave_dofs, slave_dofs)])

        # Build flexibility matrix
        Kfi = np.zeros_like(K)
        if Kss.shape == (1, 1):
            Kfi[-1, -1] = Kss[0, 0]
        else:
            start = n - Kss.shape[0]
            Kfi[start:, start:] = Kss

        # Reduced mass and stiffness matrices via Guyan transformation
        Mrr = self.get_red_matrix(M, slave_dofs, retained_dofs)
        Krr = self.get_red_matrix(K, slave_dofs, retained_dofs)
        Mr = Tg.T @ Mrr @ Tg
        Kr = Tg.T @ Krr @ Tg

        # IRS transformation (Improved Reduced System)
        Tirs = Tg + Kfi @ M @ Tg @ np.linalg.inv(Mr) @ Kr

        return self.get_transformation_functions(Tirs)
