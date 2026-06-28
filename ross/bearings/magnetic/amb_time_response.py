import time

import numpy as np
from scipy import signal
from scipy.linalg import eigh, block_diag
import control as ct
from copy import deepcopy

from ross import MagneticBearingElement


class AmbTimeResponse:
    """
    Class to simulate the time response of a rotor with Active Magnetic Bearings (AMB).

    This class handles the conversion of the rotor model to the modal domain,
    the integration of magnetic bearing controllers, the construction of the
    closed-loop state-space system, and the execution of the time-response
    simulation.

    Parameters
    ----------
    rotor : ross.Rotor
        The rotor model to be simulated.
    weight : bool, optional
        Whether to include gravitational forces in the simulation.
        Defaults to True.
    """

    def __init__(self, rotor, t, speed, F=None, **kwargs):
        """
        Initialize the AmbTimeResponse simulation.

        Parameters
        ----------
        rotor : ross.Rotor
            The rotor model to be simulated.
        t : array_like
            Time vector for the simulation in seconds.
        speed : float
            Rotor speed in rad/s.
        F : array_like, optional
            External forces applied to the rotor.

        Examples
        --------
        >>> from ross.bearings.magnetic.amb_models import rotor_example_amb_complex_controllers
        >>> rotor_amb = rotor_example_amb_complex_controllers()
        >>> t_ = np.linspace(0, 10, 10)
        >>> sim = AmbTimeResponse(rotor_amb, t=t_, speed=0)
        >>> sim.run() # doctest: +ELLIPSIS
        Simulation time: ...
        """

        # Simulation Parameters
        self.speed = speed  # Rotor speed in rad/s

        # Rotor Physical Model
        self.rotor = deepcopy(rotor)  # Copy of the rotor model
        self.weight = kwargs.get(
            "weight", True
        )  # Whether to include gravitational forces
        self.n_dof = None  # Number of rotor degrees of freedom
        self.n_x = None  # Dimension of physical state vector
        self.n_u = None  # Dimension of input vector (forces)
        self.W = None  # Gravitational force vector
        self.n_amb = 0  # Number of magnetic bearings
        self.n_x_c = None  # Dimension of controllers state vector
        self.t = t  # Time vector for simulation
        self.y = None  # Simulation output matrix for plotting
        self.F = F  # External forces applied to the rotor
        self.d_v = kwargs.get("disturbance", None)  # Disturbance vector

        # Conversion to modal domain
        self.num_modes = kwargs.get("num_modes", -1)  # Number of modes to consider
        self.x_0 = kwargs.get("x_0", 0)  # Initial state condition
        self.Phi = None  # Modal matrix (eigenvectors)
        self.M_m = None  # Modal mass matrix
        self.C_m = None  # Modal damping matrix
        self.K_m = None  # Modal stiffness matrix

        # Magnetic Bearings and Control (AMB)
        self.node = []  # Bearing application nodes
        self.k_i = []  # Current stiffness constants
        self.k_s = []  # Position stiffness constants (magnetic)
        self.a_c = None  # Controller A matrices
        self.b_c = None  # Controller B matrices
        self.c_c = None  # Controller C matrices
        self.d_c = None  # Controller D matrices
        self.A_cl = None  # Closed-loop system A matrix
        self.B_cl = None  # Closed-loop system B matrix
        self.C_cl = None  # Closed-loop system C matrix
        self.D_cl = None  # Closed-loop system D matrix

        # Current Physical State
        self.x = None  # Complete physical state vector
        self.x_disp = [0, 0]  # Instantaneous displacements in X
        self.y_disp = [0, 0]  # Instantaneous displacements in Y

    def setup_modal_domain(self):
        """
        Convert the physical model to the modal domain.

        Calculates eigenvalues and eigenvectors for the rotor's mass and
        stiffness matrices, projects the M, C, and K matrices into the modal
        domain, and prints the first natural frequencies.

        Examples
        --------
        >>> from ross.bearings.magnetic.amb_models import rotor_example_amb_complex_controllers
        >>> rotor = rotor_example_amb_complex_controllers()
        >>> t_ = np.linspace(0, 10, 10)
        >>> sim = AmbTimeResponse(rotor, t=t_, speed=0)
        >>> sim.setup_modal_domain() # doctest: +ELLIPSIS
        ... # doctest: +ELLIPSIS
        ...
        """
        modal_reduction = True

        M = self.rotor.M(self.speed)
        C = self.rotor.C(self.speed)
        K = self.rotor.K(self.speed)

        eigenvalues, eigenvectors = eigh(K, M)

        if self.num_modes == -1:
            modal_reduction = False
            self.num_modes = len(eigenvalues)

        selected_eigenvalues = eigenvalues[: self.num_modes]

        omega_rad_s = np.sqrt(np.abs(selected_eigenvalues))
        freq_hz = omega_rad_s / (2 * np.pi)
        speed_rpm = freq_hz * 60

        if modal_reduction:
            print("\n" + 120 * "=")
            print(f"- Frequencies of the first {self.num_modes} modes")
            for i in range(self.num_modes):
                print(f"Mode {i + 1}: {freq_hz[i]:.2f} Hz | {speed_rpm[i]:.2f} RPM")
            print(120 * "=")

        self.Phi = eigenvectors[:, : self.num_modes]
        self.M_m = self.Phi.T @ M @ self.Phi
        self.C_m = self.Phi.T @ C @ self.Phi
        self.K_m = self.Phi.T @ K @ self.Phi

    def get_weight_force(self):
        """
        Compute the static gravitational force vector for the rotor.

        Examples
        --------
        >>> from ross.bearings.magnetic.amb_models import rotor_example_amb_complex_controllers
        >>> rotor = rotor_example_amb_complex_controllers()
        >>> t_ = np.linspace(0, 10, 10)
        >>> sim = AmbTimeResponse(rotor, t=t_, speed=0)
        >>> sim.get_weight_force()
        """
        if self.weight:
            g = -9.81
        else:
            g = 0

        g_vec = np.zeros(self.rotor.ndof)
        g_vec[1 :: self.rotor.number_dof] = g
        self.W = self.rotor.M(0) @ g_vec

    def process_rotor(self):
        """
        Process the rotor model to identify magnetic bearings and extract controllers.

        Identifies all MagneticBearingElement instances, extracts their analog
        controllers, and prepares the data for the closed-loop matrix construction.
        Also removes the magnetic bearing elements from the rotor model.

        Examples
        --------
        >>> from ross.bearings.magnetic.amb_models import rotor_example_amb_complex_controllers
        >>> rotor = rotor_example_amb_complex_controllers()
        >>> t_ = np.linspace(0, 10, 10)
        >>> sim = AmbTimeResponse(rotor, t=t_, speed=0)
        >>> sim.process_rotor()
        """
        self.n_dof = self.rotor.ndof

        # Identifying magnetic bearings
        magnetic_bearings = [
            brg
            for brg in self.rotor.bearing_elements
            if isinstance(brg, MagneticBearingElement)
        ]
        self.n_amb = len(magnetic_bearings)

        # Initialize lists
        self.a_c, self.b_c, self.c_c, self.d_c = [], [], [], []
        self.k_i, self.k_s, self.node = [], [], []

        for amb in magnetic_bearings:
            # Controller
            C_s = ct.ss(amb.get_analog_controller())

            # Add twice for X/Y channels
            for _ in range(2):
                self.a_c.append(C_s.A)
                self.b_c.append(C_s.B)
                self.c_c.append(C_s.C)
                self.d_c.append(C_s.D)

            # Constants
            self.k_i.extend([amb.ki, amb.ki])
            self.k_s.extend([amb.ks, amb.ks])
            self.node.extend([amb.n, amb.n])

        # Removing AMBs from model
        self.rotor.bearing_elements = [
            brg
            for brg in self.rotor.bearing_elements
            if not isinstance(brg, MagneticBearingElement)
        ]

        self.n_dof = self.rotor.ndof
        self.n_x = 2 * self.n_dof
        self.n_u = self.n_dof

        self.x = np.zeros((self.n_x, 1))

        # Validations
        self.d_v = (
            self.d_v
            if self.d_v is not None
            else np.zeros((len(self.t), self.n_amb * 2))
        )

        if self.d_v.shape[0] != self.t.size or self.d_v.shape[1] != self.n_amb * 2:
            raise RuntimeError(
                "The disturbance vector d_v must have the same number of rows as the time vector t and twice the "
                "number of columns as the number of magnetic bearings."
            )

    def build_matrices(self):
        """
        Assemble the closed-loop state-space matrices.

        Constructs the A_cl, B_cl, C_cl, and D_cl matrices for the system
        in closed-loop with the magnetic bearing controllers, considering
        the bearing orientation and coupling effects.

        Examples
        --------
        >>> from ross.bearings.magnetic.amb_models import rotor_example_amb_complex_controllers
        >>> rotor = rotor_example_amb_complex_controllers()
        >>> t_ = np.linspace(0, 10, 10)
        >>> sim = AmbTimeResponse(rotor, t=t_, speed=0)
        >>> sim.process_rotor()
        >>> sim.setup_modal_domain() # doctest: +ELLIPSIS
        ...
        >>> sim.build_matrices()
        """
        theta = np.pi / 4  # Bearing orientation angle (45 degrees)
        n_controllers = 2 * self.n_amb

        K = self.rotor.K(self.speed)

        M_m = self.M_m
        C_m = self.C_m
        Phi = self.Phi

        phi = np.zeros((self.n_dof, n_controllers))
        for i in range(self.n_amb):
            node = self.node[2 * i]
            phi[node * 6 + 0, 2 * i] = 1  # AMB i - X
            phi[node * 6 + 1, 2 * i + 1] = 1  # AMB i - Y

        c_theta = np.cos(theta)
        s_theta = np.sin(theta)
        r = np.array([[c_theta, s_theta], [-s_theta, c_theta]])
        R = block_diag(*[r for _ in range(self.n_amb)])

        A_c = block_diag(*self.a_c)
        B_c = block_diag(*self.b_c)
        C_c = block_diag(*self.c_c)
        D_c = block_diag(*self.d_c)

        self.n_x_c = A_c.shape[0]

        K_x = np.diag(self.k_s)
        K_i = np.diag(self.k_i)

        m = self.num_modes
        N = self.n_dof
        n = self.n_x_c
        n_c = n_controllers

        zeros = lambda rows, columns: np.zeros((rows, columns))

        I = np.eye(self.num_modes)
        M_m_inv = np.linalg.inv(M_m)
        R_T = np.transpose(R)
        phi_t = np.transpose(phi)
        Phi_T = np.transpose(Phi)

        self.A_cl = np.block(
            [
                [zeros(m, m), I, zeros(m, n)],
                [
                    M_m_inv
                    @ Phi_T
                    @ (phi @ R_T @ (K_x - K_i @ D_c) @ R @ phi_t - K)
                    @ Phi,
                    -M_m_inv @ C_m,
                    M_m_inv @ Phi_T @ phi @ R_T @ K_i @ C_c,
                ],
                [-B_c @ R @ phi_t @ Phi, zeros(n, m), A_c],
            ]
        )

        self.B_cl = np.block(
            [
                [zeros(m, n_c), zeros(m, N)],
                [zeros(m, n_c), M_m_inv @ Phi_T],
                [-B_c @ R @ phi_t @ phi @ R_T, zeros(n, N)],
            ]
        )

        self.C_cl = np.block(
            [
                [Phi, zeros(N, m), zeros(N, n)],
                [phi_t @ Phi, zeros(n_c, m), zeros(n_c, n)],
                [R @ phi_t @ Phi, zeros(n_c, m), zeros(n_c, n)],
                [
                    R_T @ (K_x - K_i @ D_c) @ R @ phi_t @ Phi,
                    zeros(n_c, m),
                    R_T @ K_i @ C_c,
                ],
                [(K_x - K_i @ D_c) @ R @ phi_t @ Phi, zeros(n_c, m), K_i @ C_c],
                [-D_c @ R @ phi_t @ Phi, zeros(n_c, m), C_c],
            ]
        )
        self.D_cl = np.block(
            [
                [zeros(N, n_c), zeros(N, N)],
                [zeros(n_c, n_c), zeros(n_c, N)],
                [zeros(n_c, n_c), zeros(n_c, N)],
                [zeros(n_c, n_c), zeros(n_c, N)],
                [zeros(n_c, n_c), zeros(n_c, N)],
                [zeros(n_c, n_c), zeros(n_c, N)],
            ]
        )

    def run(self):
        """
        Run the time-response simulation.

        Executes the simulation using `scipy.signal.lsim` based on the
        constructed closed-loop state-space system, gravitational forces,
        and initial conditions, then saves the results.

        Returns
        -------
        t : array_like
            Time vector for the simulation.
        x : array_like
            Physical state vector of the rotor over time.
        results : list
            List containing displacements, velocities, and forces for the magnetic bearings.

        Examples
        --------
        >>> from ross.bearings.magnetic.amb_models import rotor_example_amb_complex_controllers
        >>> rotor = rotor_example_amb_complex_controllers()
        >>> t_ = np.linspace(0, 10, 10)
        >>> sim = AmbTimeResponse(rotor, t=t_, speed=0)
        >>> sim.run() # doctest: +ELLIPSIS
        ...
        Simulation time: ...
        """
        self.process_rotor()
        self.get_weight_force()
        self.setup_modal_domain()
        self.build_matrices()

        tic = time.time()

        sys = signal.lti(self.A_cl, self.B_cl, self.C_cl, self.D_cl)
        Phi_T = np.transpose(self.Phi)
        M = self.rotor.M(self.speed)

        F_e = np.zeros((len(self.t), self.n_dof))
        F_e += np.tile(self.W, (len(self.t), 1))

        if self.F is not None:
            F_e += self.F

        u = np.block([self.d_v, F_e])

        x_0 = np.zeros((self.n_dof, 1))
        x_0[1::6, 0] = self.x_0
        x_0 = Phi_T @ M @ x_0  # Transformation to modal domain

        d_x0 = np.zeros((self.n_dof, 1))
        d_x0 = Phi_T @ M @ d_x0

        x_c_0 = np.zeros((self.n_x_c, 1))

        z_0 = np.block([[x_0], [d_x0], [x_c_0]]).reshape(-1)

        self.t, self.y, _ = signal.lsim(sys, U=u, T=self.t, X0=z_0)
        print(f"Simulation time: {time.time() - tic:.2f} seconds")

        mma_dof = 2 * self.n_amb

        infos = []
        infos_size = [self.rotor.ndof, mma_dof, mma_dof, mma_dof, mma_dof, mma_dof]

        i_0 = 0
        for info_size in infos_size:
            i_f = i_0 + info_size
            infos.append(self.y[:, i_0:i_f])
            i_0 = i_f

        x = infos[0]
        x_amb = infos[1]
        v_amb = infos[2]
        F_x = infos[3]
        F_v = infos[4]
        I = infos[5]

        return self.t, x, [x_amb, v_amb, F_x, F_v, I]
