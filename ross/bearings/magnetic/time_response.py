import time

import numpy as np
from scipy import signal
from scipy.linalg import eigh, block_diag
import control as ct
from copy import deepcopy

from ross import MagneticBearingElement
from ross.bearings.magnetic.utils import rotor_amb_example_with_complex_controllers


class TimeResponseAmb:
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

    def __init__(self, rotor, weight=True):
        """
        Initialize the RunTimeAmbResponse simulation.

        Examples
        --------
        >>> rotor = rotor_amb_example_with_complex_controllers()
        >>> sim = TimeResponseAmb(rotor)
        """

        # Simulation Parameters
        self.t_sim = 10
        self.dt = 1e-5
        self.speed = 0

        # Rotor Physical Model
        self.rotor = deepcopy(rotor)
        self.weight = weight
        self.n_dof = None
        self.n_x = None  # Dimension of physical state vector
        self.n_u = None  # Dimension of input vector (forces)
        self.W = None
        self.n_amb = 0  # Number of magnetic bearings
        self.n_dof = None  # Number of rotor degrees of freedom
        self.n_x_c = None  # Dimension of controllers state vector
        self.t = np.arange(0, self.t_sim, self.dt)  # Time vector for simulation
        self.y = None  # Simulation output matrix (bearing displacements) for plotting

        # Conversion to modal domain
        self.num_modes = 10
        self.Phi = None
        self.M_m = None
        self.C_m = None
        self.K_m = None

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
        >>> rotor = rotor_amb_example_with_complex_controllers()
        >>> sim = TimeResponseAmb(rotor)
        >>> sim.setup_modal_domain()
        """
        M = self.rotor.M(self.speed)
        C = self.rotor.C(self.speed)
        K = self.rotor.K(self.speed)

        eigenvalues, eigenvectors = eigh(K, M)

        if self.num_modes == -1:
            self.num_modes = len(eigenvalues)

        selected_eigenvalues = eigenvalues[: self.num_modes]

        omega_rad_s = np.sqrt(np.abs(selected_eigenvalues))
        freq_hz = omega_rad_s / (2 * np.pi)
        speed_rpm = freq_hz * 60

        print("\n" + 120 * "=")
        print(f"- Frequencies of the first {self.num_modes} modes")
        for i in range(self.num_modes):
            print(f"Mode {i + 1}: {freq_hz[i]:.2f} Hz | {speed_rpm[i]:.2f} RPM")
        print(120 * "=")

        self.Phi = eigenvectors[:, : self.num_modes]
        self.M_m = self.Phi.T @ M @ self.Phi
        self.C_m = self.Phi.T @ C @ self.Phi
        self.K_m = self.Phi.T @ K @ self.Phi

    def get_weigth_force(self):
        """
        Compute the static gravitational force vector for the rotor.

        Examples
        --------
        >>> rotor = rotor_amb_example_with_complex_controllers()
        >>> sim = TimeResponseAmb(rotor)
        >>> sim.get_weigth_force()
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
        >>> rotor = rotor_amb_example_with_complex_controllers()
        >>> sim = TimeResponseAmb(rotor)
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

    def build_matrices(self):
        """
        Assemble the closed-loop state-space matrices.

        Constructs the A_cl, B_cl, C_cl, and D_cl matrices for the system
        in closed-loop with the magnetic bearing controllers, considering
        the bearing orientation and coupling effects.

        Examples
        --------
        >>> rotor = rotor_amb_example_with_complex_controllers()
        >>> sim = TimeResponseAmb(rotor)
        >>> sim.process_rotor()
        >>> sim.setup_modal_domain()
        >>> sim.build_matrices()
        """
        theta = np.pi / 4  # Bearing orientation angle (45 degrees)
        num_channels = 2 * self.n_amb

        K = self.rotor.K(self.speed)

        M_m = self.M_m
        C_m = self.C_m
        Phi = self.Phi

        phi = np.zeros((self.n_dof, num_channels))
        for i in range(self.n_amb):
            node = self.node[2 * i]
            phi[node * 6 + 1, 2 * i] = 1  # AMB i - X
            phi[node * 6 + 3, 2 * i + 1] = 1  # AMB i - Y

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

        # m -> num_modes
        # N -> n_dof
        # n -> n_x_c
        # nc -> num_channels
        zero_mm = np.zeros((self.num_modes, self.num_modes))
        zero_mn = np.zeros((self.num_modes, self.n_x_c))
        zero_mnc = np.zeros((self.num_modes, num_channels))
        zero_ncm = np.zeros((num_channels, self.num_modes))
        zero_ncn = np.zeros((num_channels, self.n_x_c))
        zero_nm = np.zeros((self.n_x_c, self.num_modes))
        zero_nc = np.zeros((num_channels, num_channels))
        zero_mN = np.zeros((self.num_modes, self.n_dof))
        zero_nN = np.zeros((self.n_x_c, self.n_dof))
        zero_ncN = np.zeros((num_channels, self.n_dof))
        I = np.eye(self.num_modes)
        M_m_inv = np.linalg.inv(M_m)
        R_T = np.transpose(R)
        phi_t = np.transpose(phi)
        Phi_T = np.transpose(Phi)

        self.A_cl = np.block(
            [
                [zero_mm, I, zero_mn],
                [
                    M_m_inv
                    @ Phi_T
                    @ (phi @ R_T @ (K_x - K_i @ D_c) @ R @ phi_t - K)
                    @ Phi,
                    -M_m_inv @ C_m,
                    M_m_inv @ Phi_T @ phi @ R_T @ K_i @ C_c,
                ],
                [-1e6 * B_c @ R @ phi_t @ Phi, zero_nm, A_c],
            ]
        )

        self.B_cl = np.block(
            [
                [zero_mnc, zero_mN],
                [zero_mnc, M_m_inv @ Phi_T],
                [-B_c @ R @ phi_t @ phi @ R_T, zero_nN],
            ]
        )

        self.C_cl = np.block(
            [
                [phi_t @ Phi, zero_ncm, zero_ncn],
                [R @ phi_t @ Phi, zero_ncm, zero_ncn],
                [R_T @ (K_x - K_i @ D_c) @ R @ phi_t @ Phi, zero_ncm, R_T @ K_i @ C_c],
                [(K_x - K_i @ D_c) @ R @ phi_t @ Phi, zero_ncm, K_i @ C_c],
                [-D_c @ R @ phi_t @ Phi, zero_ncm, C_c],
            ]
        )
        self.D_cl = np.block(
            [
                [zero_nc, zero_ncN],
                [zero_nc, zero_ncN],
                [zero_nc, zero_ncN],
                [zero_nc, zero_ncN],
                [zero_nc, zero_ncN],
            ]
        )

    def run(self):
        """
        Run the time-response simulation.

        Executes the simulation using `scipy.signal.lsim` based on the
        constructed closed-loop state-space system, gravitational forces,
        and initial conditions, then saves the results.

        Examples
        --------
        >>> rotor = rotor_amb_example_with_complex_controllers()
        >>> sim = TimeResponseAmb(rotor)
        >>> sim.run()
        """
        self.process_rotor()
        self.get_weigth_force()
        self.setup_modal_domain()
        self.build_matrices()

        tic = time.time()

        sys = signal.lti(self.A_cl, self.B_cl, self.C_cl, self.D_cl)
        Phi_T = np.transpose(self.Phi)
        M = self.rotor.M(self.speed)

        d_v = np.zeros((len(self.t), self.n_amb * 2))
        F_e = np.zeros((len(self.t), self.n_dof))
        F_e += np.tile(self.W, (len(self.t), 1))

        u = np.block([d_v, F_e])

        x_0 = np.zeros((self.n_dof, 1))
        x_0[1::6, 0] = -100e-6
        x_0 = Phi_T @ M @ x_0  # Transformation to modal domain

        d_x0 = np.zeros((self.n_dof, 1))
        d_x0 = Phi_T @ M @ d_x0

        x_c_0 = np.zeros((self.n_x_c, 1))

        z_0 = np.block([[x_0], [d_x0], [x_c_0]]).reshape(-1)

        _, self.y, _ = signal.lsim(sys, U=u, T=self.t, X0=z_0)

        print(f"Simulation time: {time.time() - tic:.2f} seconds")
