import warnings
import numpy as np
from scipy.fft import fft, fftfreq

from ross.units import Q_
from ross.results import HarmonicBalanceResults


class HarmonicBalance:
    def __init__(self, rotor, n_harmonics=None):
        """
        Harmonic Balance method for nonlinear rotor dynamic systems.

        This class implements the Harmonic Balance (HB) method to compute
        steady-state responses of rotor systems under periodic excitations.
        It supports gravitational loads, external harmonic forces, and
        potential crack-induced stiffness variations.

        Parameters
        ----------
        rotor : ross.Rotor
            Rotor object representing the system model.
        n_harmonics : int, optional
            Number of harmonics to include in the analysis.
            Default is 1.
        """
        self.rotor = rotor
        self.noh = n_harmonics if n_harmonics else 1

    def run(
        self,
        forces,
        speed,
        t,
        gravity=False,
        F_ext=None,
    ):
        """
        Solve the rotor system in the frequency domain using the
        Harmonic Balance (HB) method.

        Parameters
        ----------
        node : list, int
            Node indices where the harmonic forces are applied.
        magnitude : list, float
            Magnitudes of the applied harmonic forces [N].
            Interpretation depends on the excitation type:
                - For direct harmonic forces: force amplitudes [N].
                - For unbalance excitation: product ``m * e * speed**2`` [N],
                where ``m`` is the unbalance mass [kg], ``e`` is the eccentricity [m],
                and ``speed`` is the rotational speed [rad/s].
        phase : list, float
            Phase angles of the applied forces [rad].
        harmonic : list, int
            Harmonic order(s) of the excitation (1 for fundamental, 2 for second,
            etc.).
        speed : float
            Rotor rotational speed [rad/s].
        t : float
            Time array used for Fourier expansion [s].
        gravity : bool, optional
            If True, include gravitational forces. Default is False.
        F_ext : ndarray, optional
            External force array of shape (ndof, N), where N is the number of time
            samples.

        Returns
        -------
        Qt : ndarray
            Complex displacement vector in frequency domain.
        Qo : ndarray
            Static (mean) displacement vector.
        dQ : ndarray
            Harmonic displacement coefficients.
        dQ_s : ndarray
            Complex conjugate of harmonic coefficients.

        Notes
        -----
        This method constructs and solves the HB system:

        .. math::
            H Q = F

        where `H` is the harmonic balance matrix and `F` is the
        assembled force vector containing both static and harmonic components.
        """
        rotor = self.rotor

        accel = 0  # Assuming always constant speed
        crack = None  # Assuming there's no crack, crack model needs to be integrated
        freq = Q_(speed, "rad/s").to("Hz").m
        dt = t[1] - t[0]

        # Harmonic force
        Fh, Fh_s = self._harmonic_force(forces)

        # Weight
        W = rotor.gravitational_force() * int(gravity)

        # External force
        Fo, Fn, Fn_s = self._external_force(dt, freq, F_ext)

        Fn += Fh
        Fn_s += Fh_s
        F = self._assemble_forces(W, Fo, Fn, Fn_s)

        # Crack stiffness matrices
        Ko, Kn, Kn_s = self._crack_stiffness_matrices(dt, freq, crack)

        # Harmonic Balance Matrix
        H = self._build_harmonic_balance_matrix(
            speed,
            accel,
            rotor.M(speed),
            rotor.K(speed),
            rotor.Ksdt(),
            rotor.C(speed),
            rotor.G(),
            Ko,
            Kn,
            Kn_s,
        )

        Qt, Qo, dQ, dQ_s = self._solve_freq_domain(H, F)

        return HarmonicBalanceResults(rotor, speed, t, Qt, Qo, dQ, dQ_s, self.noh)

    def _harmonic_force(self, forces):
        """
        Construct the harmonic force components in the frequency domain.

        Parameters
        ----------
        node : list, int
            Node indices where harmonic loads are applied.
        magnitude : list, float
            Force magnitudes [N].
        phase : list, float
            Force phase angles [rad].
        harmonic : list, int
            Harmonic orders of each excitation.

        Returns
        -------
        F : ndarray, complex
            Harmonic force array of shape (ndof, noh).
        F_s : ndarray, complex
            Complex conjugate of the harmonic forces.
        """
        ndof = self.rotor.ndof
        number_dof = self.rotor.number_dof

        for f in forces:
            if f["harmonic"] > self.noh:
                self.noh = f["harmonic"]

        F = np.zeros((ndof, self.noh), dtype=np.complex128)

        for f in forces:
            n = f["node"]
            p = f["phase"]
            m = f["magnitude"]
            h = f["harmonic"]

            cos = np.cos(p)
            sin = np.sin(p)

            Fa = m * np.array([cos, sin])
            Fb = m * np.array([-sin, cos])

            dofs = [number_dof * n, number_dof * n + 1]
            F[dofs, h - 1] += Fa - 1j * Fb

        F_s = np.conjugate(F)

        return F, F_s

    def _unbalance_force(self, node, magnitude, phase, omega, alpha=0):
        """
        Compute unbalance forces in the frequency domain.

        Parameters
        ----------
        node : list, int
            Node indices with unbalance.
        magnitude : list, float
            Unbalance magnitudes [kg·m].
        phase : list, float
            Phase angles of unbalance [rad].
        omega : float
            Angular speed [rad/s].
        alpha : float, optional
            Angular acceleration [rad/s²], default is 0.

        Returns
        -------
        F : ndarray, complex
            Unbalance force vector.
        F_s : ndarray, complex
            Complex conjugate of the unbalance force.
        """
        ndof = self.rotor.ndof
        number_dof = self.rotor.number_dof

        F = np.zeros((ndof), dtype=np.complex128)

        for n, m, p in zip(node, magnitude, phase):
            cos = np.cos(p)
            sin = np.sin(p)

            Fa = m * omega**2 * np.array([cos, sin])
            Fa += m * alpha * np.array([-sin, cos])

            Fb = m * omega**2 * np.array([-sin, cos])
            Fb += m * alpha * np.array([cos, -sin])

            dofs = [number_dof * n, number_dof * n + 1]
            F[dofs] += Fa - 1j * Fb

        F_s = np.conjugate(F)

        return F, F_s

    def _external_force(self, dt, freq, F=None):
        """
        Compute Fourier expansion of external time-domain forces.

        Parameters
        ----------
        dt : float
            Time step [s].
        freq : float
            Fundamental excitation frequency [Hz].
        F : ndarray, optional
            External force time history of shape (ndof, N).

        Returns
        -------
        Fo : ndarray
            Static (mean) force component.
        Fn : ndarray
            Harmonic force coefficients.
        Fn_s : ndarray
            Complex conjugate of harmonic force coefficients.
        """
        ndof = self.rotor.ndof

        Fo = np.zeros(ndof, dtype=complex)
        Fn = np.zeros((ndof, self.noh), dtype=complex)
        Fn_s = np.zeros((ndof, self.noh), dtype=complex)

        if F is not None:
            dofs = list(set(np.where(F != 0)[0]))

            Fo_, Fn_ = self._Fourier_expansion(F[dofs, :], dt, freq, self.noh)

            Fo[dofs] += Fo_
            Fn[dofs, :] += Fn_
            Fn_s[dofs, :] += np.conjugate(Fn_)

        return Fo, Fn, Fn_s

    def _assemble_forces(self, W, Fo, Fn, Fn_s):
        """
        Assemble the total complex force vector for the HB system.

        Parameters
        ----------
        W : ndarray
            Static gravitational force vector.
        Fo : ndarray
            Static external force component.
        Fn : ndarray
            Harmonic force coefficients.
        Fn_s : ndarray
            Conjugate harmonic force coefficients.

        Returns
        -------
        F : ndarray, complex
            Combined force vector for harmonic balance analysis.
        """
        ndof = self.rotor.ndof

        F = np.zeros(((self.noh * 2 + 1) * ndof), dtype=complex)

        F[:ndof] = 4 * W + 2 * Fo

        for i in range(1, self.noh + 1):
            F[(2 * i - 1) * ndof : 2 * i * ndof] = 2 * Fn[:, i - 1]
            F[2 * i * ndof : (2 * i + 1) * ndof] = 2 * Fn_s[:, i - 1]

        return F

    def _crack_stiffness_matrices(self, dt, freq, crack=None):
        """
        Compute Fourier-expanded stiffness matrices for cracked shafts.

        Parameters
        ----------
        dt : float
            Time step [s].
        freq : float
            Rotational frequency [Hz].
        crack : object, optional
            Crack model object providing `dofs`, `crack_coeff`, and `_Kflex()`
            methods.

        Returns
        -------
        Ko : ndarray
            Static stiffness correction matrix.
        Kn : ndarray
            Harmonic stiffness correction matrices.
        Kn_s : ndarray
            Conjugate harmonic stiffness matrices.
        """
        ndof = self.rotor.ndof
        n_aux = 2 * self.noh

        Ko = np.zeros((ndof, ndof), dtype=complex)
        Kn = np.zeros((ndof, ndof, n_aux), dtype=complex)
        Kn_s = np.zeros((ndof, ndof, n_aux), dtype=complex)

        if crack:
            dof = crack.dofs
            crack_coeff = crack.crack_coeff

            Kco, Kcn, Kcn_s = self._Fourier_expansion(crack_coeff, dt, freq, n_aux)

            Ko[dof, dof] = crack._Kflex(Kco.reshape(-1, 1))[:, :, 0]
            Kn_ = crack._Kflex(Kcn)
            Kn_s_ = crack._Kflex(Kcn_s)

            for i in range(n_aux):
                Kn[dof, dof, i] = Kn_[:, :, i]
                Kn_s[dof, dof, i] = Kn_s_[:, :, i]

        return Ko, Kn, Kn_s

    def _build_harmonic_balance_matrix(
        self,
        speed,
        accel,
        M,
        K,
        Ksdt,
        C,
        G,
        Ko,
        Kn,
        Kn_s,
        Co=None,
        Cn=None,
        Cn_s=None,
    ):
        """
        Construct the Harmonic Balance matrix `H`.

        Parameters
        ----------
        speed : float
            Rotational speed [rad/s].
        accel : float
            Angular acceleration [rad/s²].
        M, K, Ksdt, C, G : ndarray
            Rotor mass, stiffness, stiffness time-derivative, damping, and
            gyroscopic matrices.
        Ko : ndarray
            Static crack stiffness correction.
        Kn, Kn_s : ndarray
            Harmonic stiffness correction and its conjugate.
        Co, Cn, Cn_s : ndarray, optional
            Damping correction matrices. Default: zeros.

        Returns
        -------
        H0 : ndarray, complex
            Full harmonic balance system matrix.
        """
        ndof = self.rotor.ndof
        noh = self.noh

        # Co, Cn, Cn_s are already considered in C matrix (bearing elements)
        Co = np.zeros((ndof, ndof), dtype=complex)
        Cn = np.zeros((ndof, ndof, 2 * noh), dtype=complex)
        Cn_s = np.zeros((ndof, ndof, 2 * noh), dtype=complex)

        # alpha and beta are already considered in C matrix (shaft elements)
        alpha = 0
        beta = 0

        size = ndof * (2 * noh + 1)
        H0 = np.zeros((size, size), dtype=complex)

        idx0 = slice(0, ndof)
        idx1 = slice(1 * ndof, 2 * ndof)
        idx2 = slice(2 * ndof, 3 * ndof)

        H0[idx0, idx0] = Ko + 2 * (K + Ksdt * accel)

        K_aux = 1j * beta * speed
        aux_C = 1j * speed
        aux1 = 1 * K_aux + 1

        H0[idx1, idx2] = np.conjugate(aux1) * Kn[:, :, 1] - 1 * aux_C * Cn[:, :, 1]
        H0[idx2, idx1] = aux1 * Kn_s[:, :, 1] + 1 * aux_C * Cn_s[:, :, 1]

        for n in range(1, noh + 1):
            idx3 = slice((2 * n - 1) * ndof, 2 * n * ndof)
            idx4 = slice(2 * n * ndof, (2 * n + 1) * ndof)

            aux2 = -2 * n**2 * speed**2 + 2 * n * aux_C * alpha
            aux3 = 2 * n * 1j * speed**2
            aux4 = n * aux_C
            aux5 = 2 * n * K_aux + 2
            aux6 = 1 * n * K_aux + 1
            H0[idx3, idx3] = (
                aux2 * M
                + aux3 * G
                + aux4 * (2 * C + Co)
                + aux5 * K
                + 2 * (Ksdt * accel)
                + aux6 * Ko
            )

            H0[idx4, idx4] = (
                np.conjugate(aux2) * M
                + np.conjugate(aux3) * G
                + np.conjugate(aux4) * (2 * C + Co)
                + np.conjugate(aux5) * K
                + 2 * (Ksdt * accel)
                + np.conjugate(aux6) * Ko
            )

            H0[idx3, idx0] = Kn[:, :, n - 1]
            H0[idx4, idx0] = Kn_s[:, :, n - 1]
            H0[idx0, idx3] = aux6 * Kn_s[:, :, n - 1] + aux4 * Cn_s[:, :, n - 1]
            H0[idx0, idx4] = (
                np.conjugate(aux6) * Kn[:, :, n - 1]
                + np.conjugate(aux4) * Cn[:, :, n - 1]
            )

            if n < noh:
                aux7 = 2 * noh - 2 * n + 2

                for k in range(2, aux7):
                    idx5 = slice((k - 1) * ndof, k * ndof)
                    idx6 = slice((k - 1 + 2 * n) * ndof, (k + 2 * n) * ndof)
                    aux8 = np.ceil((k + 2 * n - 1) / 2)

                    if np.mod(k, 2) == 1:
                        A_ = (-aux8 * K_aux + 1) * Kn[:, :, n - 1] - aux8 * aux_C * Cn[
                            :, :, n - 1
                        ]
                    else:
                        A_ = (aux8 * K_aux + 1) * Kn_s[
                            :, :, n - 1
                        ] + aux8 * aux_C * Cn_s[:, :, n - 1]

                    H0[idx5, idx6] = A_

                aux9 = 2 * n + 1
                for k in range(2 * noh + 1, aux9, -1):
                    idx7 = slice((k - 1) * ndof, k * ndof)
                    idx8 = slice((k - 1 - 2 * n) * ndof, (k - 2 * n) * ndof)
                    aux10 = np.ceil((k - 2 * n - 1) / 2)

                    if np.mod(k, 2) == 1:
                        B_ = (-aux10 * K_aux + 1) * Kn_s[
                            :, :, n - 1
                        ] - aux10 * aux_C * Cn_s[:, :, n - 1]
                    else:
                        B_ = (+aux10 * K_aux + 1) * Kn[
                            :, :, n - 1
                        ] + aux10 * aux_C * Cn[:, :, n - 1]

                    H0[idx7, idx8] = B_

        aux11 = 2 * noh - 1
        for n in range(2 * noh, 2, -1):
            aux12 = 2 * n - 1

            if aux12 > 2 * noh:
                aux12 = 2 * noh + 1

            if aux11 < 2:
                aux11 = 1

            aux13 = 2 * n + 1

            for k in range(aux12, aux11, -1):
                idx9 = slice((k - 1) * ndof, k * ndof)
                idx10 = slice((aux13 - k - 1) * ndof, (aux13 - k) * ndof)
                aux14 = np.ceil((aux13 - k - 1) / 2)

                if np.mod(k, 2) == 1:
                    C_ = (+aux14 * K_aux + 1) * Kn_s[
                        :, :, n - 1
                    ] + aux14 * aux_C * Cn_s[:, :, n - 1]
                else:
                    C_ = (-aux14 * K_aux + 1) * Kn[:, :, n - 1] - aux14 * aux_C * Cn[
                        :, :, n - 1
                    ]

                H0[idx9, idx10] = C_

            aux11 = aux11 - 2

        return H0

    def _solve_freq_domain(self, H, F):
        """
        Solve the Harmonic Balance system in frequency domain.

        Parameters
        ----------
        H : ndarray
            Harmonic Balance matrix.
        F : ndarray
            Force vector.

        Returns
        -------
        Qt : ndarray, complex
            Full frequency-domain displacement vector.
        Qo : ndarray, complex
            Static displacement component.
        Qn : ndarray, complex
            Harmonic displacement components.
        Qn_s : ndarray, complex
            Conjugate harmonic displacement components.

        Notes
        -----
        If the system is singular, the pseudo-inverse is used instead of direct solve.
        """
        ndof = self.rotor.ndof

        try:
            Qt = np.linalg.solve(H, F)
        except np.linalg.LinAlgError as err:
            warnings.warn(
                f"{err} error. Using the pseudo-inverse to proceed.", UserWarning
            )
            Qt = F @ np.linalg.pinv(H)

        Qo = np.real(Qt[:ndof])
        Qn = np.zeros((ndof, self.noh), dtype=complex)
        Qn_s = np.zeros((ndof, self.noh), dtype=complex)

        for i in range(1, self.noh + 1):
            Qn[:ndof, i - 1] = Qt[(2 * i - 1) * ndof : (2 * i) * ndof]
            Qn_s[:ndof, i - 1] = Qt[(2 * i) * ndof : (2 * i + 1) * ndof]

        return Qt, Qo, Qn, Qn_s

    @staticmethod
    def _reconstruct_time_domain(omega, t, Qo, dQ, n_harmonics):
        """
        Reconstruct the time-domain response from frequency-domain results.

        Parameters
        ----------
        omega : float
            Rotational speed [rad/s].
        t : array_like
            Time vector [s].
        Qo : ndarray
            Static displacement vector.
        dQ : ndarray
            Harmonic displacement coefficients.
        n_harmonics : int, optional
            Number of harmonics to include in the analysis.

        Returns
        -------
        y : ndarray
            Displacement response over time.
        ydot : ndarray
            Velocity response over time.
        y2dot : ndarray
            Acceleration response over time.
        """
        shape = (np.size(Qo), len(t))

        sum_y = np.zeros(shape)
        sum_ydot = np.zeros(shape)
        sum_y2dot = np.zeros(shape)

        for i in range(1, n_harmonics + 1):
            an = np.transpose(np.array([np.real(dQ[:, i - 1])]))
            bn = np.transpose(np.array([-np.imag(dQ[:, i - 1])]))

            cos = np.array([np.cos(i * omega * t)])
            sin = np.array([np.sin(i * omega * t)])

            sum_y += np.dot(an, cos) + np.dot(bn, sin)
            sum_ydot += i * omega * (np.dot(bn, cos) - np.dot(an, sin))
            sum_y2dot -= (i * omega) ** 2 * (np.dot(an, cos) + np.dot(bn, sin))

        y = Qo[:, np.newaxis] / 2 + sum_y
        ydot = sum_ydot
        y2dot = sum_y2dot

        return y, ydot, y2dot

    @staticmethod
    def _Fourier_expansion(F, dt, fo, size):
        """
        Perform Fourier expansion of a time-domain signal.

        Parameters
        ----------
        F : ndarray
            Input force array of shape (ndof, N).
        dt : float
            Time step [s].
        fo : float
            Fundamental frequency [Hz].
        size : int
            Number of harmonic components to compute.

        Returns
        -------
        Fo : ndarray
            Mean (static) component.
        Fn : ndarray
            Complex harmonic coefficients.

        Notes
        -----
        Uses FFT to compute coefficients corresponding to multiples of fundamental
        frequency.
        """
        row, N = F.shape
        b = N // 2

        X = fft(F)[:, :b]
        X *= 2 / N
        freqs = fftfreq(N, dt)[:b]

        Fo = np.real(X[:, 0])
        an = np.real(X)
        bn = -np.imag(X)

        Fn = np.zeros((row, size), dtype=complex)
        for n in range(1, size + 1):
            idx = np.argmin(np.abs(freqs - n * fo))
            Fn[:, n - 1] = an[:, idx] - 1j * bn[:, idx]

        return Fo, Fn
