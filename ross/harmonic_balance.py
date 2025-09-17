import numpy as np
from scipy.fft import fft

from ross.units import Q_, check_units
from ross.results import (
    FrequencyResponseResults,
    TimeResponseResults,
)


class HarmonicBalance:
    def __init__(self, rotor):
        self.rotor = rotor

        self.probeForce = None

    @check_units
    def run(
        self,
        node,
        unb_magnitude,
        unb_phase,
        speed,
        t,
        n_harmonics=6,
        F_ext=None,
        points=None,
    ):
        rotor = self.rotor
        accel = 0

        self.noh = n_harmonics
        if points is None or points > len(t):
            points = int(len(t) / 2)

        self.points = points

        W = rotor.gravitational_force()
        F_unb, F_unb_s = self._unbalance_force(node, unb_magnitude, unb_phase, speed)

        if F_ext is None:
            F_ext = np.zeros((len(t), rotor.ndof))
        dt = t[1] - t[0]
        freq = Q_(speed, "rad/s").to("Hz").m
        Fo, Fn, Fn_s = self._external_force(F_ext, dt, freq)

        F = self._assemble_forces(W, F_unb, F_unb_s, Fo, Fn, Fn_s)

        H = self._build_harmonic_balance_matrix(
            speed,
            accel,
            rotor.M(speed),
            rotor.K(speed),
            rotor.Ksdt(),
            rotor.C(speed),
            rotor.G(),
        )

        _, Qo, dQ, dQ_s = self._solve_freq_response(H, F)

        y, ydot, y2dot = self._reconstruct_time_domain(speed, t, Qo, dQ)

        return TimeResponseResults(rotor, t, y.T, [])

    def _unbalance_force(self, node, magnitude, phase, omega):
        ndof = self.rotor.ndof
        number_dof = self.rotor.number_dof

        F0 = np.zeros((ndof), dtype=np.complex128)

        for n, m, p in zip(node, magnitude, phase):
            Fa = m * omega**2 * np.array([np.cos(p), np.sin(p)])
            Fb = m * omega**2 * np.array([np.sin(p), -np.cos(p)])

            dofs = [number_dof * n, number_dof * n + 1]
            F0[dofs] += Fa - 1j * Fb

        F0_s = np.conjugate(F0)

        return F0, F0_s

    def _external_force(self, F_ext, dt, freq):
        ndof = self.rotor.ndof
        number_dof = self.rotor.number_dof

        t_size = F_ext.shape[1]

        Fo = np.zeros(ndof, dtype=complex)
        F = np.zeros((ndof, self.noh), dtype=complex)
        F_s = np.zeros((ndof, self.noh), dtype=complex)

        if self.probeForce is not None:
            for i in range(len(self.probeForce)):
                idx = slice(
                    (self.probeForce[i]) * number_dof - 6,
                    (self.probeForce[i]) * number_dof,
                )

                Fnew = F_ext[idx, -(t_size - self.points) :]
                Fo, Fn = self._expand_Fourier(Fnew, dt, freq)

                Fo[idx] += Fo
                F[idx, :] += Fn
                F_s[idx, :] += np.conjugate(Fn)

        return Fo, F, F_s

    @check_units
    def _expand_Fourier(self, F, dt, fo):
        row, N = F.shape
        Fn = np.array(np.zeros((row, self.noh)), dtype=complex)

        b = int(np.floor(N / 2))
        df = 1 / (N * dt)

        A = fft(F)
        X = A[:, :b]
        X = X * 2 / N
        Fo = 1 * np.real(X[:, 0])
        an = 1 * np.real(X)
        bn = -1 * np.imag(X)

        for n in range(1, self.noh + 1):
            idx = int(n * fo / df)
            Fn[:, n - 1] = an[:, idx] - 1j * bn[:, idx]

        return Fo, Fn

    def _assemble_forces(self, W, F_unb, F_unb_s, Fo, Fn, Fn_s):
        ndof = self.rotor.ndof

        F0 = np.zeros(((self.noh * 2 + 1) * ndof), dtype=complex)
        # F0[0 : 3 * ndof] = [4 * W + 2 * Fo, 2 * F_unb + 2 * Fn[:, 0], 2 * F_unb_s + 2 * Fn_s[:, 0]]
        F0[0:ndof] = 4 * W + 2 * Fo
        F0[ndof : 2 * ndof] = 2 * F_unb + 2 * Fn[:, 0]
        F0[2 * ndof : 3 * ndof] = 2 * F_unb_s + 2 * Fn_s[:, 0]

        for i in range(2, self.noh + 1):
            F0[(2 * i - 1) * ndof : 2 * i * ndof] = 2 * Fn[:, i - 1]
            F0[2 * i * ndof : (2 * i + 1) * ndof] = 2 * Fn_s[:, i - 1]

        return F0

    def _build_harmonic_balance_matrix(
        self,
        speed,
        accel,
        M,
        K,
        Ksdt,
        C,
        G,
        Ko=None,
        Kn=None,
        Kn_s=None,
        Co=None,
        Cn=None,
        Cn_s=None,
    ):
        ndof = self.rotor.ndof
        alpha = self.rotor.shaft_elements[0].alpha
        beta = self.rotor.shaft_elements[0].beta

        if Ko is None:
            Ko = np.zeros((ndof, ndof), dtype=complex)
        if Co is None:
            Co = np.zeros((ndof, ndof), dtype=complex)
        if Kn is None:
            Kn = np.zeros((ndof, ndof, 2 * self.noh), dtype=complex)
        if Kn_s is None:
            Kn_s = np.zeros((ndof, ndof, 2 * self.noh), dtype=complex)
        if Cn is None:
            Cn = np.zeros((ndof, ndof, 2 * self.noh), dtype=complex)
        if Cn_s is None:
            Cn_s = np.zeros((ndof, ndof, 2 * self.noh), dtype=complex)

        size = ndof * (2 * self.noh + 1)
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

        for n in range(1, self.noh + 1):
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

            if n < self.noh:
                aux7 = 2 * self.noh - 2 * n + 2

                for k in range(2, aux7):
                    idx5 = slice((k - 1) * ndof, k * ndof)
                    idx6 = slice((k - 1 + 2 * n) * ndof, (k + 2 * n) * ndof)
                    aux8 = np.ceil((k + 2 * n - 1) / 2)
                    if np.mod(k, 2) == 1:
                        A = (-aux8 * K_aux + 1) * Kn[:, :, n - 1] - aux8 * aux_C * Cn[
                            :, :, n - 1
                        ]
                    else:
                        A = (aux8 * K_aux + 1) * Kn_s[
                            :, :, n - 1
                        ] + aux8 * aux_C * Cn_s[:, :, n - 1]

                    H0[idx5, idx6] = A

                aux9 = 2 * n + 1
                for k in range(2 * self.noh + 1, aux9, -1):
                    idx7 = slice((k - 1) * ndof, k * ndof)
                    idx8 = slice((k - 1 - 2 * n) * ndof, (k - 2 * n) * ndof)
                    aux10 = np.ceil((k - 2 * n - 1) / 2)
                    if np.mod(k, 2) == 1:
                        B = (-aux10 * K_aux + 1) * Kn_s[
                            :, :, n - 1
                        ] - aux10 * aux_C * Cn_s[:, :, n - 1]
                    else:
                        B = (+aux10 * K_aux + 1) * Kn[:, :, n - 1] + aux10 * aux_C * Cn[
                            :, :, n - 1
                        ]

                    H0[idx7, idx8] = B

        aux11 = 2 * self.noh - 1
        for n in range(2 * self.noh, 2, -1):
            aux12 = 2 * n - 1

            if aux12 > 2 * self.noh:
                aux12 = 2 * self.noh + 1

            if aux11 < 2:
                aux11 = 1
            aux13 = 2 * n + 1

            for k in range(aux12, aux11, -1):
                idx9 = slice((k - 1) * ndof, k * ndof)
                idx10 = slice((aux13 - k - 1) * ndof, (aux13 - k) * ndof)
                aux14 = np.ceil((aux13 - k - 1) / 2)
                if np.mod(k, 2) == 1:
                    CC = (+aux14 * K_aux + 1) * Kn_s[
                        :, :, n - 1
                    ] + aux14 * aux_C * Cn_s[:, :, n - 1]
                else:
                    CC = (-aux14 * K_aux + 1) * Kn[:, :, n - 1] - aux14 * aux_C * Cn[
                        :, :, n - 1
                    ]
                H0[idx9, idx10] = CC
            aux11 = aux11 - 2

        return H0

    def _solve_freq_response(self, H, F):
        ndof = self.rotor.ndof

        # Qt = np.linalg.solve(H, F)
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
