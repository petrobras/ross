import numpy as np
from ross import Rotor

from copy import deepcopy


class ModelReduction:
    """
    Base class for model reduction methods.
    """

    def __init__(self, rotor, important_nodes=[], method="guyan"):
        """
        Initialize the model reduction with a given model.

        Parameters
        ----------
        model : ross.Rotor
            The rotor model to be reduced.
        """

        self.ndof = rotor.ndof
        self.number_dof = rotor.number_dof
        self.K = rotor.K(0)
        self.M = rotor.M()

        self.orig_rotor = rotor

        M_K = np.diag(self.K) / np.diag(self.M)
        # Sort DOFs by mass-stiffness ratio (M/K)
        self.indx = np.argsort(M_K)[::-1]

        if method == "guyan":
            self.model_reduction_technique = self.guyan
        elif method == "guyan_melhorado":
            self.model_reduction_technique = self.guyan_melhorado
        else:
            raise ValueError(f"Pass a existing {method}.")

        self.slave_dofs, self.retained_dofs = self.select_nodes_based_rotor(
            rotor, important_nodes
        )
        self.reduced_rotor = self.reduce_model(self.slave_dofs, self.retained_dofs)

    @staticmethod
    def copy_rotor(rotor, tag=None):
        return Rotor(
            deepcopy(rotor.shaft_elements),
            disk_elements=deepcopy(rotor.disk_elements),
            bearing_elements=deepcopy(rotor.bearing_elements),
            point_mass_elements=deepcopy(rotor.point_mass_elements),
            min_w=rotor.min_w,
            max_w=rotor.max_w,
            rated_w=rotor.rated_w,
            tag=tag or rotor.tag,
        )

    @staticmethod
    def select_nodes_based_rotor(rotor, important_nodes=[]):
        ignored_dofs = []
        selected_dofs = []

        important_nodes.append(min(rotor.nodes))
        important_nodes.append(max(rotor.nodes))

        for elm in rotor.elements:
            if elm.n not in rotor.nodes:
                continue

            dofs = list(elm.dof_global_index.values())

            if elm in rotor.shaft_elements and elm.n not in important_nodes:
                ignored_dofs.extend(dofs)
            else:
                selected_dofs.extend(dofs)

        return sorted(set(ignored_dofs)), sorted(set(selected_dofs))

    @staticmethod
    def rearrange_matrix(matrix, slave_dofs, retained_dofs):
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

    def get_transformation_functions(self, transf_matrix, slave_dofs, retained_dofs):
        # Verificar - talvez seja melhor separar em várias funções
        reduce_matrix = (
            lambda array: (
                transf_matrix.T
                @ self.rearrange_matrix(array, slave_dofs, retained_dofs)
            )
            @ transf_matrix
        )

        reduce_vector = lambda array: transf_matrix.T @ self.rearrange_matrix(
            array, slave_dofs, retained_dofs
        )

        get_complete_vector = lambda array: transf_matrix @ array

        return reduce_matrix, reduce_vector, get_complete_vector

    def increment_nodes(self, add_nodes=[]):
        num_dof = self.number_dof

        for n in add_nodes:
            dofs = range(n * num_dof + 0, n * num_dof + num_dof)
            self.slaves_dofs.extends(dofs)

            for dof in dofs:
                self.retained_dofs.remove(dof)

        self.reduce_model(self.slave_dofs, self.retained_dofs)

    def reduce_model(self, slave_dofs, retained_dofs):
        transf_matrix = self.model_reduction_technique(slave_dofs, retained_dofs)
        functions = self.get_transformation_functions(
            transf_matrix, slave_dofs, retained_dofs
        )
        reduce_matrix = functions[0]

        rotor = self.orig_rotor
        reduced_rotor = self.copy_rotor(rotor)

        reduced_rotor.M = lambda frequency=None, synchronous=False: reduce_matrix(
            rotor.M(frequency=frequency, synchronous=synchronous)
        )
        reduced_rotor.K = lambda frequency: reduce_matrix(rotor.K(frequency))
        reduced_rotor.Ksdt = lambda: reduce_matrix(rotor.Ksdt())
        reduced_rotor.C = lambda frequency: reduce_matrix(rotor.C(frequency))
        reduced_rotor.G = lambda: reduce_matrix(rotor.G())

        return reduced_rotor

    def optimize(self):
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

    def guyan(self, slave_dofs, retained_dofs):
        """
        Standard Guyan Reduction method.
        """
        K = self.K

        # Compute transformation matrix
        Kia = (
            -np.linalg.pinv(K[np.ix_(slave_dofs, slave_dofs)])
            @ K[np.ix_(slave_dofs, retained_dofs)]
        )
        Tg = np.block([[np.eye(Kia.shape[1])], [Kia]])

        return Tg

    def guyan_melhorado(self, slave_dofs, retained_dofs):
        M = self.M
        K = self.K

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
        Mrr = self.rearrange_matrix(M, slave_dofs, retained_dofs)
        Krr = self.rearrange_matrix(K, slave_dofs, retained_dofs)
        Mr = Tg.T @ Mrr @ Tg
        Kr = Tg.T @ Krr @ Tg

        # IRS transformation (Improved Reduced System)
        Tirs = Tg + Kfi @ M @ Tg @ np.linalg.inv(Mr) @ Kr

        return Tirs
