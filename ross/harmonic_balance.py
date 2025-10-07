import warnings
import numpy as np
import numpy.linalg as la
from scipy.fft import fft, fftfreq
from collections.abc import Iterable

from ross.units import Q_, check_units
from ross.results import (
    ForcedResponseResults,
    FrequencyResponseResults,
    TimeResponseResults,
)


class HarmonicBalance:
    def __init__(self, rotor, n_harmonics=1):
        self.rotor = rotor
        self.ndof = rotor.ndof
        self.probe_force = None
        self.noh = n_harmonics

    @check_units
    def run(
        self,
        node,
        unb_magnitude,
        unb_phase,
        speed,
        dt,
        F_ext=None,
    ):
        rotor = self.rotor

        accel = 0
        has_gravity = 0
        has_crack = False

        # Forces
        W = rotor.gravitational_force() * has_gravity
        F_unb, F_unb_s = self._unbalance_force(node, unb_magnitude, unb_phase, speed)

        freq = Q_(speed, "rad/s").to("Hz").m
        Fo, Fn, Fn_s = self._external_force(dt, freq, F_ext)

        F = self._assemble_forces(W, F_unb, F_unb_s, Fo, Fn, Fn_s)

        # Crack stiffness matrices
        Ko, Kn, Kn_s = self._crack_stiffness_matrices(dt, freq, has_crack)

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

        return Qt, Qo, dQ, dQ_s

    def run_forced_response(
        self,
        force,
        speed_range,
        unbalance=None,
    ):
        ndof = self.rotor.ndof

        samples = 20
        omega_max = max(self.noh * np.max(speed_range), 1e-6)
        dt = 2 * np.pi / (omega_max * samples)

        forced_resp = np.zeros((ndof, len(speed_range)), dtype=complex)
        velc_resp = np.zeros((ndof, len(speed_range)), dtype=complex)
        accl_resp = np.zeros((ndof, len(speed_range)), dtype=complex)

        if unbalance is None:
            node = [0]
            unb_magnitude = [0]
            unb_phase = [0]
        else:
            node, unb_magnitude, unb_phase = unbalance

        for i, speed in enumerate(speed_range):
            _, Qo, dQ, _ = self.run(
                node=np.int_(node),
                unb_magnitude=unb_magnitude,
                unb_phase=unb_phase,
                speed=speed,
                dt=dt,
                F_ext=force[:, i] if force is not None else None,
            )

            forced_resp[:, i] = Qo / 2 + np.sum(dQ, axis=1)
            velc_resp[:, i] = 1j * speed * np.sum(dQ, axis=1)
            accl_resp[:, i] = -(speed**2) * np.sum(dQ, axis=1)

        forced_resp = ForcedResponseResults(
            rotor=self.rotor,
            forced_resp=forced_resp,
            velc_resp=velc_resp,
            accl_resp=accl_resp,
            speed_range=speed_range,
            unbalance=unbalance,
        )

        return forced_resp

    def run_unbalance_response(
        self,
        node,
        unb_magnitude,
        unb_phase,
        speed_range,
    ):
        unbalance = np.vstack((node, unb_magnitude, unb_phase))

        forced_resp = self.run_forced_response(
            force=None,
            speed_range=speed_range,
            unbalance=unbalance,
        )

        return forced_resp

    def run_time_response(
        self,
        speed,
        F,
        t,
    ):
        dt = t[1] - t[0]

        _, Qo, dQ, _ = self.run(
            node=[0],
            unb_magnitude=[0],
            unb_phase=[0],
            speed=speed,
            dt=dt,
            F_ext=F.T,
        )

        y, ydot, y2dot = self._reconstruct_time_domain(speed, t, Qo, dQ)
        time_resp = TimeResponseResults(self.rotor, t, y.T, [])

        return time_resp

    def run_unbalance_time_response(
        self,
        node,
        unb_magnitude,
        unb_phase,
        speed,
        t,
    ):
        dt = t[1] - t[0]

        _, Qo, dQ, _ = self.run(
            node=node,
            unb_magnitude=unb_magnitude,
            unb_phase=unb_phase,
            speed=speed,
            dt=dt,
        )

        y, ydot, y2dot = self._reconstruct_time_domain(speed, t, Qo, dQ)
        time_resp = TimeResponseResults(self.rotor, t, y.T, [])

        return time_resp

    def _unbalance_force(self, node, magnitude, phase, omega, alpha=0):
        ndof = self.rotor.ndof
        number_dof = self.rotor.number_dof

        F0 = np.zeros((ndof), dtype=np.complex128)

        for n, m, p in zip(node, magnitude, phase):
            cos = np.cos(p)
            sin = np.sin(p)

            an = m * omega**2 * np.array([cos, sin])
            an += m * alpha * np.array([-sin, cos])

            bn = m * omega**2 * np.array([-sin, cos])
            bn += m * alpha * np.array([cos, -sin])

            dofs = [number_dof * n, number_dof * n + 1]
            F0[dofs] += an - 1j * bn

        F0_s = np.conjugate(F0)

        return F0, F0_s

    def _unbalance_force_over_time(self, node, magnitude, phase, omega, t):
        ndof = self.rotor.ndof

        F0 = np.zeros((len(t), ndof))
        F0_s = np.zeros((len(t), ndof))

        if isinstance(omega, Iterable):
            alpha = np.gradient(omega, t)

            for i, w in enumerate(omega):
                F0[i, :], F0_s[i, :] = self._unbalance_force(
                    node, magnitude, phase, w, alpha[i]
                )

        else:
            F0[:, :], F0_s[:, :] = self._unbalance_force(
                node, magnitude, phase, omega, 0
            )

        return F0, F0_s

    def _external_force(self, dt, freq, F_ext=None):
        ndof = self.rotor.ndof

        Fo = np.zeros(ndof, dtype=complex)
        Fn = np.zeros((ndof, self.noh), dtype=complex)
        Fn_s = np.zeros((ndof, self.noh), dtype=complex)

        if F_ext is not None:
            dofs = list(set(np.where(F_ext != 0)[0]))

            Fo_, Fn_ = self._Fourier_expansion(F_ext[dofs, :], dt, freq, self.noh)

            Fo[dofs] += Fo_
            Fn[dofs, :] += Fn_
            Fn_s[dofs, :] += np.conjugate(Fn_)

        return Fo, Fn, Fn_s

    def _assemble_forces(self, W, F_unb, F_unb_s, Fo, Fn, Fn_s):
        ndof = self.rotor.ndof
        F0 = np.zeros(((self.noh * 2 + 1) * ndof), dtype=complex)

        F0[:ndof] = 4 * W + 2 * Fo
        F0[ndof : 2 * ndof] = 2 * (F_unb + Fn[:, 0])
        F0[2 * ndof : 3 * ndof] = 2 * (F_unb_s + Fn_s[:, 0])

        for i in range(2, self.noh + 1):
            F0[(2 * i - 1) * ndof : 2 * i * ndof] = 2 * Fn[:, i - 1]
            F0[2 * i * ndof : (2 * i + 1) * ndof] = 2 * Fn_s[:, i - 1]

        return F0

    def _crack_stiffness_matrices(self, dt, freq, has_crack=True):
        ndof = self.rotor.ndof
        n_aux = 2 * self.noh

        Ko = np.zeros((ndof, ndof), dtype=complex)
        Kn = np.zeros((ndof, ndof, n_aux), dtype=complex)
        Kn_s = np.zeros((ndof, ndof, n_aux), dtype=complex)

        if has_crack:
            idx = self.crack.dof_crack
            coeff_flex = self.crack.coeff_flex

            Kco, Kcn, Kcn_s = self._Fourier_expansion(coeff_flex, dt, freq, n_aux)
            Ko[idx, idx] = self.crack._Kflex(Kco.reshape(-1, 1))[:, :, 0]

            Kn_flex = self.crack._Kflex(Kcn)
            Kn_flex_s = self.crack._Kflex(Kcn_s)

            for i in range(n_aux):
                Kn[idx, idx, i] = Kn_flex[:, :, i]
                Kn_s[idx, idx, i] = Kn_flex_s[:, :, i]

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

    def _reconstruct_time_domain(self, omega, t, Qo, dQ, aux=None):
        shape = (np.size(Qo), len(t))

        sum_y = np.zeros(shape)
        sum_ydot = np.zeros(shape)
        sum_y2dot = np.zeros(shape)

        # --------------------------------------------------------------------------
        # ---------------------------------------------- Solution in the time domain
        # --------------------- yyHB=Qo/2+an*cos(n*Omega*time)+bn*sin(n*Omega*time)
        # ------ yyptHB=-n*Omega*an*sin(n*Omega*time)+n*Omega*bn*cos(n*Omega*time)
        # yy2ptHB=-n**2*Omega**2*an*cos(n*Omega*time)-n**2*Omega**2*bn*sin(n*Omega*time)
        # --------------------------------------------------------------------------
        # -- dQ=[Q1   Q1s]
        # -- Q1  = a1 -I*b1
        # -- Q1s = a1 +I*b1

        for i in range(1, self.noh + 1):
            an = np.transpose(np.array([np.real(dQ[:, i - 1])]))
            bn = np.transpose(np.array([-np.imag(dQ[:, i - 1])]))

            cos = np.array([np.cos(i * omega * t)])
            sin = np.array([np.sin(i * omega * t)])

            sum_y += np.dot(an, cos) + np.dot(bn, sin)

            if aux is None:
                sum_ydot += i * omega * (np.dot(bn, cos) - np.dot(an, sin))
                sum_y2dot -= (i * omega) ** 2 * (np.dot(an, cos) + np.dot(bn, sin))

        y = Qo[:, np.newaxis] / 2 + sum_y
        ydot = sum_ydot
        y2dot = sum_y2dot

        return y, ydot, y2dot

    @staticmethod
    def _Fourier_expansion(F, dt, fo, size):
        row, N = F.shape
        b = N // 2

        X = fft(F)[:, :b]
        X *= 2 / N
        freq = fftfreq(N, dt)[:b]

        Fo = np.real(X[:, 0])
        an = np.real(X)
        bn = -np.imag(X)

        Fn = np.zeros((row, size), dtype=complex)
        for n in range(1, size + 1):
            idx = np.argmin(np.abs(freq - n * fo))
            Fn[:, n - 1] = an[:, idx] - 1j * bn[:, idx]

        return Fo, Fn
