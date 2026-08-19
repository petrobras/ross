import numpy as np
from scipy.optimize import curve_fit
from warnings import warn

from ross import SealElement
from ross.seals.gas_model import extract_gas_properties
from ross.seals.solver_tools import solve_frequencies
from ross.units import check_units

__all__ = ["HolePatternSeal"]

MOODY_FRICTION_COEFFICIENT = 1.375e-3
MOODY_ROUGHNESS_SCALE = 1.0e4
MOODY_VISCOSITY_SCALE = 5.0e5
CHOKED_MZ2_LIMIT = 0.98


class HolePatternSolver:
    """Bulk-flow solver for a hole-pattern annular seal.

    This class owns the mutable per-run state of the bulk-flow solution (base
    state arrays, perturbation integrals) and computes the leakage and dynamic
    coefficients for one shaft speed at a time. The :class:`HolePatternSeal`
    element builds one solver at construction and maps it over the requested
    frequencies; the solver holds only plain data, so it can be pickled to
    worker processes for multi-frequency runs.

    Parameters
    ----------
    shaft_radius : float
        Shaft radius (m).
    radial_clearance : float
        Seal radial clearance (m).
    axial_length : float
        Axial length of the seal (m).
    relative_roughness : float
        Relative roughness E / D of the shaft, dimensionless.
    cell_depth : float
        Depth of a hole-pattern cell (m).
    inlet_pressure, outlet_pressure : float
        Boundary pressures (Pa).
    inlet_temperature : float
        Inlet temperature (deg K).
    preswirl : float
        Ratio of gas circumferential velocity to shaft surface velocity.
    entrance_loss_coefficient, exit_loss_coefficient : float
        Loss coefficients at the seal entrance and exit.
    excitation_ratio : float
        Ratio of the excitation (whirl) frequency to the rotational speed.
    nz : int
        Number of discretization points in the axial direction.
    max_iterations : int
        Maximum iterations for the base state calculation.
    tolerance : float
        Base state tolerance as a fraction of the pressure differential.
    first_step_size : float
        Initial step for the inlet pressure iteration.
    relaxation_factor : float
        Relaxation factor of the inlet pressure iteration.
    R : float
        Specific gas constant (J/(kg K)).
    gamma : float
        Ratio of specific heats.
    sutherland_b, sutherland_s : float
        Sutherland viscosity model coefficients.
    """

    def __init__(
        self,
        shaft_radius,
        radial_clearance,
        axial_length,
        relative_roughness,
        cell_depth,
        inlet_pressure,
        outlet_pressure,
        inlet_temperature,
        preswirl,
        entrance_loss_coefficient,
        exit_loss_coefficient,
        excitation_ratio,
        nz,
        max_iterations,
        tolerance,
        first_step_size,
        relaxation_factor,
        R,
        gamma,
        sutherland_b,
        sutherland_s,
    ):
        self._shaft_radius = shaft_radius
        self.radial_clearance = radial_clearance
        self.relative_roughness = relative_roughness
        self.cell_depth = cell_depth
        self.inlet_pressure = inlet_pressure
        self.outlet_pressure = outlet_pressure
        self.inlet_temperature = inlet_temperature
        self.preswirl = preswirl
        self.entrance_loss_coefficient = entrance_loss_coefficient
        self.exit_loss_coefficient = exit_loss_coefficient
        self.excitation_ratio = excitation_ratio
        self.nz = nz
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.first_step_size = first_step_size
        self.relaxation_factor = relaxation_factor
        self.R = R
        self.gamma = gamma
        self.sutherland_b = sutherland_b
        self.sutherland_s = sutherland_s

        self.dz = axial_length / float(nz)
        self.t = np.zeros(nz + 1)
        self.mz2 = np.zeros(nz + 1)
        self.mt = np.zeros(nz + 1)

        # Index maps and signs pairing the cosine/sine components of the four
        # simultaneous clearance perturbations (+X, -X, +Y, -Y).
        self.i_t = np.array([1, 0, 3, 2])
        self.i_th = np.array([2, 3, 0, 1])
        self.sgn_t = np.array([-1.0, 1.0, -1.0, 1.0])
        self.sgn_th = np.array([-1.0, -1.0, 1.0, 1.0])

    def solve(self, frequency):
        """Solve the seal at one shaft speed (rad/s).

        Returns a dict with the dynamic coefficients, the leakage and the
        axial pressure distribution. If the solve fails, the coefficients are
        returned as NaN and a warning explains the failure.
        """
        self.gamma1 = self.gamma - 1.0
        self.gamma12 = self.gamma1 / 2.0
        self.omega = frequency
        self.area = np.pi * 2.0 * self._shaft_radius * self.radial_clearance

        self._gamma_R = self.gamma * self.R
        self._radius_omega = self._shaft_radius * self.omega
        self._rough_factor = MOODY_ROUGHNESS_SCALE * self.relative_roughness
        self._mu_factor = MOODY_VISCOSITY_SCALE

        try:
            base_state_results = self._solve_base_state()
            if not base_state_results:
                raise RuntimeError("Error calculating leakage.")

            force_coeffs, p_base = self._solve_force_coefficients(base_state_results)

            pressure = np.insert(p_base, 0, self.inlet_pressure)
            pressure = np.insert(pressure, 1, base_state_results.get("p2", 0))
            pressure = np.append(pressure, base_state_results.get("p5", 0))

            attribute_coef = {
                "kxx": force_coeffs.get("K_dir", 0),
                "kyy": force_coeffs.get("K_dir", 0),
                "kxy": force_coeffs.get("k_cross", 0),
                "kyx": -force_coeffs.get("k_cross", 0),
                "cxx": force_coeffs.get("C_dir", 0),
                "cyy": force_coeffs.get("C_dir", 0),
                "cxy": force_coeffs.get("c_cross", 0),
                "cyx": -force_coeffs.get("c_cross", 0),
                "mxx": force_coeffs.get("M_dir", 0),
                "myy": force_coeffs.get("M_dir", 0),
                "mxy": force_coeffs.get("m_cross", 0),
                "myx": -force_coeffs.get("m_cross", 0),
                "seal_leakage": base_state_results.get("mdot", 0),
                "pressure": pressure,
            }
            return attribute_coef
        except Exception as e:
            warn(
                f"Could not calculate the hole-pattern seal at frequency "
                f"{frequency} rad/s; the coefficients for this frequency are "
                f"set to NaN. Original error: {e}"
            )
            failed = dict.fromkeys(
                [
                    "kxx",
                    "kyy",
                    "kxy",
                    "kyx",
                    "cxx",
                    "cyy",
                    "cxy",
                    "cyx",
                    "mxx",
                    "myy",
                    "mxy",
                    "myx",
                    "seal_leakage",
                ],
                np.nan,
            )
            failed["pressure"] = np.full(self.nz + 4, np.nan)
            return failed

    def _inlet_loss(self, p2):
        if p2 >= self.inlet_pressure:
            p2 = self.inlet_pressure * 0.9999
        if p2 <= 0:
            return 0, 0, 0, 0
        m2_sq_term = (self.inlet_pressure / p2) ** (self.gamma1 / self.gamma) - 1.0
        if m2_sq_term < 0:
            return 0, 0, 0, 0
        m2 = np.sqrt(m2_sq_term / self.gamma12)
        T2 = self.inlet_temperature * (p2 / self.inlet_pressure) ** (
            self.gamma1 / self.gamma
        )
        c2 = np.sqrt(self.gamma * self.R * T2)
        mdot = (p2 / (self.R * T2)) * self.area * (m2 * c2)
        mt2 = self.preswirl * (self._shaft_radius * self.omega) / c2
        p30_denom = (1.0 + self.gamma12 * m2**2) ** (self.gamma / self.gamma1)
        if p30_denom == 0:
            p30_denom = 1e-9
        p30 = self.inlet_pressure * (
            1.0
            - self.entrance_loss_coefficient * (self.gamma / 2.0) * m2**2 / p30_denom
        )
        m3 = m2
        for _ in range(30):
            m3_term = 1.0 + self.gamma12 * m3**2
            if m3_term <= 0:
                m3 = 0.99
                m3_term = 1.0 + self.gamma12 * m3**2
            if (self.area * p30) == 0:
                return 0, 0, 0, 0
            m3 = (
                mdot
                / (self.area * p30)
                * np.sqrt(self.R * self.inlet_temperature / self.gamma)
                * m3_term ** ((1.0 + self.gamma) / (2.0 * self.gamma1))
            )
        T3 = self.inlet_temperature / (1.0 + self.gamma12 * m3**2)
        mt3 = mt2 * m3 / m2 if abs(m2) > 1e-9 else 0.0
        m_sq3 = m3**2
        return mdot, m_sq3, T3, mt3

    def _form_rhs(self, mz2, T, mt):
        if T <= 0:
            T = 1e-9

        if mz2 <= 0:
            mz2 = 1e-9

        mz = np.sqrt(mz2)
        mt2 = mt**2
        c = np.sqrt(self._gamma_R * T)
        u = mz * c
        if u == 0:
            u = 1e-9
        w = mt * c
        rho = self.mdot / (self.area * u)
        Romega = self._radius_omega
        mr = Romega / c if c > 0 else 0
        u2 = u**2
        w_minus_Romega = w - Romega
        utot = np.sqrt(u2 + w**2) * 0.5
        utot_rotor = np.sqrt(u2 + w_minus_Romega**2)
        if utot == 0:
            utot = 1e-9
        if utot_rotor == 0:
            utot_rotor = 1e-9
        # Pre-compute T**1.5 and shared divisor
        T_15 = T**1.5
        mu = self.sutherland_b * T_15 / (self.sutherland_s + T)
        mu_factor_mu = self._mu_factor * mu
        fs_term = mu_factor_mu / (rho * self.radial_clearance * utot)
        fs = MOODY_FRICTION_COEFFICIENT * (1.0 + fs_term ** (1.0 / 3.0))
        fs_geom = (
            np.sqrt(1.0 + mt2 / mz2) / (4.0 * self.radial_clearance) * fs
            if self.radial_clearance > 0
            else 0
        )
        fr_term = self._rough_factor + mu_factor_mu / (
            rho * self.radial_clearance * utot_rotor
        )
        fr = MOODY_FRICTION_COEFFICIENT * (1.0 + fr_term ** (1.0 / 3.0))
        fr_geom = (
            np.sqrt(1.0 + (mt - mr) ** 2 / mz2) / self.radial_clearance * fr
            if self.radial_clearance > 0
            else 0
        )
        RH1 = -self.gamma * mz2 / (1.0 + self.gamma * mz2) * (fs_geom + fr_geom)
        RH2 = (
            -self.gamma12
            / (1.0 + self.gamma12 * (mz2 + mt2))
            * ((mt - mr) * mr)
            * fr_geom
        )
        RH3 = -(fs_geom * mt + (mt - mr) * fr_geom)
        if abs(mz2 - 1.0) < 1e-9:
            mz2 = 1.0 - 1e-9
        term_denom1, term_denom2 = (
            (1.0 + self.gamma12 * (mz2 + mt2)),
            (1.0 + self.gamma * mz2),
        )
        RHmz = (
            mz2
            * term_denom2
            * (1.0 + self.gamma12 * mz2)
            / (mz2 - 1.0)
            * (
                RH1 * (1.0 + self.gamma12 * mz2) / term_denom1
                - RH2
                + RH3 * self.gamma12 * mt / term_denom1
            )
        )
        RHt = (
            T * (RH1 + (1.0 - self.gamma * mz2) / term_denom2 * RHmz / mz2)
            if mz2 > 0
            else 0
        )
        RHmt = (RH3 - mt * RHt / T) / 2.0 if T > 0 else 0
        return RHmz, RHt, RHmt

    def _integrate_base_state(self):
        ichoke = 0
        for iz in range(1, self.nz + 1):
            iz1 = iz - 1
            RHmz_pred, RHt_pred, RHmt_pred = self._form_rhs(
                self.mz2[iz1], self.t[iz1], self.mt[iz1]
            )
            mztmp_pred, ttmp_pred, mttmp_pred = (
                self.mz2[iz1] + self.dz * RHmz_pred,
                self.t[iz1] + self.dz * RHt_pred,
                self.mt[iz1] + self.dz * RHmt_pred,
            )
            RHmz_corr, RHt_corr, RHmt_corr = self._form_rhs(
                mztmp_pred, ttmp_pred, mttmp_pred
            )
            self.mz2[iz] = self.mz2[iz1] + self.dz * (RHmz_pred + RHmz_corr) / 2.0
            self.t[iz] = self.t[iz1] + self.dz * (RHt_pred + RHt_corr) / 2.0
            self.mt[iz] = self.mt[iz1] + self.dz * (RHmt_pred + RHmt_corr) / 2.0
            if self.mz2[iz] > CHOKED_MZ2_LIMIT:
                ichoke = 1
                break
        return ichoke

    def _exit_loss(self, msquared4, T4):
        m4 = np.sqrt(msquared4)
        if m4 == 0:
            return self.outlet_pressure - 1, 0, 0
        p4 = np.sqrt(self.R * T4 / self.gamma) * self.mdot / (self.area * m4)
        p40_denom = 1.0 + self.gamma12 * msquared4
        p40 = p4 * p40_denom ** (self.gamma / self.gamma1)
        p50_denom = p40_denom ** (self.gamma / self.gamma1)
        if p50_denom == 0:
            p50_denom = 1e-9
        p50 = p40 * (
            1.0
            - self.exit_loss_coefficient * (self.gamma / 2.0) * msquared4 / p50_denom
        )
        m5 = m4
        for _ in range(30):
            m5_term = 1.0 + self.gamma12 * m5**2
            if m5_term <= 0:
                m5 = 0.99
                m5_term = 1.0 + self.gamma12 * m5**2
            if (self.area * p50) == 0:
                return self.outlet_pressure - 1, 0, 0
            m5 = (
                self.mdot
                / (self.area * p50)
                * np.sqrt(self.R * self.inlet_temperature / self.gamma)
                * m5_term ** ((1.0 + self.gamma) / (2.0 * self.gamma1))
            )
        T5 = self.inlet_temperature / (1.0 + self.gamma12 * m5**2)
        p5_denom = (1.0 + self.gamma12 * m5**2) ** (self.gamma / self.gamma1)
        if p5_denom == 0:
            p5_denom = 1e-9
        p5 = p50 / p5_denom
        return p5, m5, T5

    def _solve_base_state(self):
        iglobalchoke = 0
        p2_old = (1.0 - self.first_step_size) * self.inlet_pressure
        self.mdot, self.mz2[0], self.t[0], self.mt[0] = self._inlet_loss(p2_old)
        ichoke = self._integrate_base_state()
        if ichoke:
            return None
        p5, _, _ = self._exit_loss(self.mz2[self.nz], self.t[self.nz])
        delp_old = p5 - self.outlet_pressure
        p2 = (1.0 - 2.0 * self.first_step_size) * self.inlet_pressure
        self.mdot, self.mz2[0], self.t[0], self.mt[0] = self._inlet_loss(p2)
        ichoke = self._integrate_base_state()
        if ichoke:
            return None
        p5, _, _ = self._exit_loss(self.mz2[self.nz], self.t[self.nz])
        delp = p5 - self.outlet_pressure
        for itr in range(1, self.max_iterations + 1):
            if abs(delp - delp_old) < 1e-12:
                break
            temp_delp, temp_p = delp, p2
            p2 = (
                self.relaxation_factor
                * (delp * p2_old - delp_old * p2)
                / (delp - delp_old)
                + (1.0 - self.relaxation_factor) * p2
            )
            p2_old, delp_old = temp_p, temp_delp
            for _ in range(60):
                self.mdot, self.mz2[0], self.t[0], self.mt[0] = self._inlet_loss(p2)
                ichoke = self._integrate_base_state()
                if not ichoke:
                    p5, _, _ = self._exit_loss(self.mz2[self.nz], self.t[self.nz])
                    delp = p5 - self.outlet_pressure
                    break
                iglobalchoke = 1
                p2 = p2_old + 0.5 * (p2 - p2_old)
            else:
                return None
            if (
                abs(delp)
                < self.tolerance * (self.inlet_pressure - self.outlet_pressure)
                or iglobalchoke
            ):
                break

        return {
            "mdot": self.mdot,
            "t": self.t,
            "mz2": self.mz2,
            "mt": self.mt,
            "p2": p2,
            "p5": p5,
        }

    def _one_step_perturbed(
        self, h_pert, whirl_frequency, deep, base_old, base_new, pert_old, iz
    ):
        """Advance the perturbed state by one axial step (predictor-corrector).

        Returns the perturbed state at the next station and the wall shear
        contributions of the four perturbation directions.
        """
        dz = self.dz
        rad = self._shaft_radius
        g = self.gamma
        R = self.R
        relative_roughness = self.relative_roughness
        sutherland_b = self.sutherland_b
        sutherland_s = self.sutherland_s
        omg = self.omega
        w_omg = whirl_frequency
        cp = g * R / (g - 1.0)
        delta = MOODY_ROUGHNESS_SCALE * relative_roughness
        alpha = MOODY_FRICTION_COEFFICIENT
        mu0 = MOODY_VISCOSITY_SCALE
        b = base_old
        up, wp = (
            (base_new["u"] - base_old["u"]) / dz,
            (base_new["w"] - base_old["w"]) / dz,
        )
        rhop, Tp, pp = (
            (base_new["rho"] - base_old["rho"]) / dz,
            (base_new["T"] - base_old["T"]) / dz,
            (base_new["p"] - base_old["p"]) / dz,
        )
        us = (
            np.sqrt(b["u"] ** 2 + b["w"] ** 2)
            if (b["u"] ** 2 + b["w"] ** 2) > 0
            else 1e-9
        )
        w_rel = b["w"] - (omg * rad)
        ur = (
            np.sqrt(base_new["u"] ** 2 + w_rel**2)
            if (base_new["u"] ** 2 + w_rel**2) > 0
            else 1e-9
        )
        mu = (
            mu0 * sutherland_b * b["T"] ** 1.5 / (sutherland_s + b["T"])
            if b["T"] > 0
            else 0
        )
        mut = (
            mu0
            * (sutherland_b / 2.0)
            * np.sqrt(b["T"])
            * (1.5 * sutherland_s + b["T"])
            / (sutherland_s + b["T"]) ** 2
            if b["T"] > 0
            else 0
        )
        mus = mu * 2.0
        Res = (
            mus / (h_pert[0] * b["rho"] * us)
            if (h_pert[0] * b["rho"] * us) != 0
            else 1e12
        )
        Rer = (
            mu / (h_pert[0] * b["rho"] * ur)
            if (h_pert[0] * b["rho"] * ur) != 0
            else 1e12
        )
        den_res_23 = Res ** (2.0 / 3.0) if Res > 1e-9 else 1e12
        den_rer_23 = (delta + Rer) ** (2.0 / 3.0) if (delta + Rer) > 1e-9 else 1e12
        fs = alpha * (1.0 + Res ** (1.0 / 3.0))
        fr = alpha * (1.0 + (delta + Rer) ** (1.0 / 3.0))
        fx = b["rho"] * b["u"] * (us * fs / 4.0 + ur * fr) / 2.0
        fxmu = (
            b["rho"]
            * b["u"]
            / 2.0
            * (
                alpha / (12.0 * h_pert[0] * b["rho"] * den_res_23)
                + alpha / (3.0 * h_pert[0] * b["rho"] * den_rer_23)
            )
        )
        fxrho = (
            b["rho"]
            * b["u"]
            / 2.0
            * (
                -(alpha * mus) / (12.0 * h_pert[0] * b["rho"] ** 2 * den_res_23)
                - alpha * mu / (3.0 * h_pert[0] * b["rho"] ** 2 * den_rer_23)
            )
            + b["u"] * (us * fs / 4.0 + ur * fr) / 2.0
        )
        fxu = (
            b["rho"]
            * b["u"]
            / 2.0
            * (
                -(alpha * mus * b["u"])
                / (12.0 * h_pert[0] * b["rho"] * den_res_23 * us**2)
                + b["u"] * fs / (4.0 * us)
                - (alpha * mu * base_new["u"])
                / (3.0 * h_pert[0] * b["rho"] * den_rer_23 * ur**2)
                + b["u"] * fr / ur
            )
            + b["rho"] * (us * fs / 4.0 + ur * fr) / 2.0
        )
        fxw = (
            b["rho"]
            * b["u"]
            / 2.0
            * (
                -(alpha * mus * b["w"])
                / (12.0 * h_pert[0] * b["rho"] * den_res_23 * us**2)
                + b["w"] * fs / (4.0 * us)
                - (alpha * mu * w_rel)
                / (3.0 * h_pert[0] * b["rho"] * den_rer_23 * ur**2)
                + w_rel * fr / ur
            )
        )
        fxh = (
            b["rho"]
            * b["u"]
            / 2.0
            * (
                -(alpha * mus) / (12.0 * h_pert[0] ** 2 * b["rho"] * den_res_23)
                - alpha * mu / (3.0 * h_pert[0] ** 2 * b["rho"] * den_rer_23)
            )
        )
        ft = b["rho"] * (b["w"] * us * fs / 4.0 + w_rel * ur * fr) / 2.0
        ftmu = (
            b["rho"]
            / 2.0
            * (
                alpha * b["w"] / (12.0 * h_pert[0] * b["rho"] * den_res_23)
                + alpha * w_rel / (3.0 * h_pert[0] * b["rho"] * den_rer_23)
            )
        )
        ftrho = (
            b["rho"]
            / 2.0
            * (
                -(alpha * mus * b["w"])
                / (12.0 * h_pert[0] * b["rho"] ** 2 * den_res_23)
                - (alpha * mu * w_rel) / (3.0 * h_pert[0] * b["rho"] ** 2 * den_rer_23)
            )
            + (b["w"] * us * fs / 4.0 + w_rel * ur * fr) / 2.0
        )
        ftu = (
            b["rho"]
            / 2.0
            * (
                -(alpha * mus * b["u"] * b["w"])
                / (12.0 * h_pert[0] * b["rho"] * den_res_23 * us**2)
                + b["u"] * b["w"] * fs / (4.0 * us)
                - (alpha * mu * base_new["u"] * w_rel)
                / (3.0 * h_pert[0] * b["rho"] * den_rer_23 * ur**2)
                + b["u"] * w_rel * fr / ur
            )
        )
        ftw = (
            b["rho"]
            / 2.0
            * (
                -(alpha * mus * b["w"] ** 2)
                / (12.0 * h_pert[0] * b["rho"] * den_res_23 * us**2)
                + b["w"] ** 2 * fs / (4.0 * us)
                + us * fs / 4.0
                - (alpha * mu * w_rel**2)
                / (3.0 * h_pert[0] * b["rho"] * den_rer_23 * ur**2)
                + w_rel**2 * fr / ur
                + ur * fr
            )
        )
        fth = (
            b["rho"]
            / 2.0
            * (
                -(alpha * mus * b["w"])
                / (12.0 * h_pert[0] ** 2 * b["rho"] * den_res_23)
                - alpha * mu * w_rel / (3.0 * h_pert[0] ** 2 * b["rho"] * den_rer_23)
            )
        )
        derivs_store = {}
        results_pert = {}
        for step in ["predictor", "corrector"]:
            if step == "predictor":
                current_base = base_old
                p_old_pert = pert_old
            else:
                current_base = base_new
                p_old_pert = results_pert["predictor"]
            ht = (
                -h_pert[1:] + deep * p_old_pert["p"] / current_base["p"]
                if current_base["p"] != 0
                else -h_pert[1:]
            )
            cof, rhs = np.zeros((4, 4)), np.zeros((4, 4))
            cof[0, 0], cof[0, 1], cof[0, 2] = (
                current_base["u"] * (h_pert[0] + deep),
                current_base["rho"] * current_base["u"] * deep / current_base["T"]
                if current_base["T"] != 0
                else 0,
                current_base["rho"] * h_pert[0],
            )
            a1_c, a2_c, a3_c, a4_c, a5_c, a6_c = (
                current_base["w"] * h_pert[0] / rad if rad != 0 else 0,
                current_base["rho"] * h_pert[0] / rad if rad != 0 else 0,
                current_base["rho"] * current_base["w"] / rad if rad != 0 else 0,
                up * h_pert[0],
                rhop * h_pert[0],
                (base_new["rho"] * base_new["u"] - base_old["rho"] * base_old["u"])
                / dz,
            )
            a7_c, a8_c = (
                current_base["rho"]
                * current_base["u"]
                * deep
                * rhop
                / current_base["rho"] ** 2
                if current_base["rho"] != 0
                else 0,
                current_base["rho"]
                * current_base["u"]
                * deep
                * Tp
                / current_base["T"] ** 2
                if current_base["T"] != 0
                else 0,
            )
            for i in range(4):
                rhs[0, i] = (
                    -(
                        self.sgn_t[i]
                        * w_omg
                        * (
                            current_base["rho"] * ht[self.i_t[i]]
                            + h_pert[0] * p_old_pert["rho"][self.i_t[i]]
                        )
                        + self.sgn_th[i]
                        * (
                            a1_c * p_old_pert["rho"][self.i_th[i]]
                            + a2_c * p_old_pert["w"][self.i_th[i]]
                            + a3_c * ht[self.i_th[i]]
                        )
                        + a4_c * p_old_pert["rho"][i]
                        + a5_c * p_old_pert["u"][i]
                        + a6_c * ht[i]
                    )
                    + a7_c * p_old_pert["rho"][i]
                    + a8_c * p_old_pert["T"][i]
                )
            cof[1, 1], cof[1, 2], cof[1, 3] = (
                h_pert[0] * current_base["rho"] * current_base["u"] * cp,
                h_pert[0] * current_base["rho"] * current_base["u"] ** 2,
                h_pert[0] * current_base["rho"] * current_base["u"] * current_base["w"],
            )
            rhs[1, :] = 0.0
            cof[2, 0], cof[2, 1], cof[2, 2] = (
                h_pert[0] * R * current_base["T"],
                h_pert[0] * R * current_base["rho"],
                h_pert[0] * current_base["rho"] * current_base["u"],
            )
            a1_ax, a2_ax, a3_ax, a4_ax, a5_ax = (
                current_base["rho"] * h_pert[0],
                h_pert[0] * current_base["u"] * up,
                h_pert[0] * current_base["rho"] * up,
                current_base["rho"] * current_base["u"] * up,
                current_base["rho"] * current_base["w"] * h_pert[0] / rad
                if rad > 0
                else 0,
            )
            a6_ax, a7_ax = h_pert[0] * R * Tp, h_pert[0] * R * rhop
            for i in range(4):
                friction_axial_terms = (
                    fxmu * mut * p_old_pert["T"][i]
                    + fxrho * p_old_pert["rho"][i]
                    + fxu * p_old_pert["u"][i]
                    + fxw * p_old_pert["w"][i]
                    + fxh * ht[i]
                )
                rhs[2, i] = (
                    -(
                        self.sgn_t[i] * w_omg * a1_ax * p_old_pert["u"][self.i_t[i]]
                        + self.sgn_th[i] * a5_ax * p_old_pert["u"][self.i_th[i]]
                        + a2_ax * p_old_pert["rho"][i]
                        + a3_ax * p_old_pert["u"][i]
                        + a4_ax * ht[i]
                    )
                    - a6_ax * p_old_pert["rho"][i]
                    - a7_ax * p_old_pert["T"][i]
                    - pp * ht[i]
                    - friction_axial_terms
                )
            cof[3, 3] = h_pert[0] * current_base["rho"] * current_base["u"]
            a1_t, a2_t, a3_t, a4_t, a5_t = (
                current_base["rho"] * h_pert[0],
                h_pert[0] * current_base["u"] * wp,
                h_pert[0] * current_base["rho"] * wp,
                current_base["rho"] * current_base["u"] * wp,
                current_base["rho"] * current_base["w"] * h_pert[0] / rad
                if rad != 0
                else 0,
            )
            for i in range(4):
                friction_tang_terms = (
                    ftmu * mut * p_old_pert["T"][i]
                    + ftrho * p_old_pert["rho"][i]
                    + ftu * p_old_pert["u"][i]
                    + ftw * p_old_pert["w"][i]
                    + fth * ht[i]
                )
                rhs[3, i] = (
                    -(
                        self.sgn_t[i] * w_omg * a1_t * p_old_pert["w"][self.i_t[i]]
                        + self.sgn_th[i] * a5_t * p_old_pert["w"][self.i_th[i]]
                        + a2_t * p_old_pert["rho"][i]
                        + a3_t * p_old_pert["u"][i]
                        + a4_t * ht[i]
                    )
                    - (h_pert[0] * self.sgn_th[i] * p_old_pert["p"][self.i_th[i]])
                    - friction_tang_terms
                )
            try:
                derivs = np.linalg.solve(cof, rhs)
            except np.linalg.LinAlgError:
                warn(f"Singular matrix in step {step}, iz={iz}")
                derivs = np.zeros((4, 4))
            derivs_store[step] = derivs
            if step == "predictor":
                results_pert[step] = {
                    "rho": pert_old["rho"] + dz * derivs[0, :],
                    "T": pert_old["T"] + dz * derivs[1, :],
                    "u": pert_old["u"] + dz * derivs[2, :],
                    "w": pert_old["w"] + dz * derivs[3, :],
                }
                results_pert[step]["p"] = self.R * (
                    base_new["T"] * results_pert[step]["rho"]
                    + base_new["rho"] * results_pert[step]["T"]
                )
        derivs_pred = derivs_store["predictor"]
        derivs_corr = derivs_store["corrector"]
        pert_new = {
            "rho": pert_old["rho"] + dz * (derivs_pred[0, :] + derivs_corr[0, :]) / 2.0,
            "T": pert_old["T"] + dz * (derivs_pred[1, :] + derivs_corr[1, :]) / 2.0,
            "u": pert_old["u"] + dz * (derivs_pred[2, :] + derivs_corr[2, :]) / 2.0,
            "w": pert_old["w"] + dz * (derivs_pred[3, :] + derivs_corr[3, :]) / 2.0,
        }
        b_new = base_new
        us_new = (
            np.sqrt(b_new["u"] ** 2 + b_new["w"] ** 2)
            if (b_new["u"] ** 2 + b_new["w"] ** 2) > 0
            else 1e-9
        )
        w_rel_new = b_new["w"] - (omg * rad)
        ur_new = (
            np.sqrt(b_new["u"] ** 2 + w_rel_new**2)
            if (b_new["u"] ** 2 + w_rel_new**2) > 0
            else 1e-9
        )
        mu_new = (
            mu0 * sutherland_b * b_new["T"] ** 1.5 / (sutherland_s + b_new["T"])
            if b_new["T"] > 0
            else 0
        )
        mut_new = (
            mu0
            * (sutherland_b / 2.0)
            * np.sqrt(b_new["T"])
            * (1.5 * sutherland_s + b_new["T"])
            / (sutherland_s + b_new["T"]) ** 2
            if b_new["T"] > 0
            else 0
        )
        Rer_new = (
            mu_new / (h_pert[0] * b_new["rho"] * ur_new)
            if (h_pert[0] * b_new["rho"] * ur_new) != 0
            else 1e12
        )
        den_rer_23_new = (
            (delta + Rer_new) ** (2.0 / 3.0) if (delta + Rer_new) > 1e-9 else 1e12
        )
        ftmu_final = (
            b_new["rho"]
            * (alpha * w_rel_new / (3.0 * h_pert[0] * b_new["rho"] * den_rer_23_new))
            / 2.0
        )
        ftrho_final = (
            b_new["rho"]
            * (
                -alpha
                * mu_new
                * w_rel_new
                / (3.0 * h_pert[0] * b_new["rho"] ** 2 * den_rer_23_new)
            )
            / 2.0
            + w_rel_new * ur_new * fr / 2.0
        )
        ftu_final = (
            b_new["rho"]
            * (
                -alpha
                * mu_new
                * b_new["u"]
                * w_rel_new
                / (3.0 * h_pert[0] * b_new["rho"] * ur_new**2 * den_rer_23_new)
                + b_new["u"] * w_rel_new * fr / ur_new
            )
            / 2.0
        )
        ftw_final = (
            b_new["rho"]
            * (
                -alpha
                * mu_new
                * w_rel_new**2
                / (3.0 * h_pert[0] * b_new["rho"] * ur_new**2 * den_rer_23_new)
                + w_rel_new**2 * fr / ur_new
            )
            / 2.0
        )
        fth_final = (
            -b_new["rho"]
            * (
                alpha
                * mu_new
                * w_rel_new
                / (3.0 * h_pert[0] ** 2 * b_new["rho"] * den_rer_23_new)
            )
            / 2.0
        )
        p_new = self.R * (b_new["T"] * pert_new["rho"] + b_new["rho"] * pert_new["T"])
        ht_final = (
            -h_pert[1:] + deep * p_new / b_new["p"] if b_new["p"] != 0 else -h_pert[1:]
        )
        shear = (
            ftmu_final * mut_new * pert_new["T"]
            + ftrho_final * pert_new["rho"]
            + ftu_final * pert_new["u"]
            + ftw_final * pert_new["w"]
            + fth_final * ht_final
        )
        return pert_new, shear

    def _integrate_perturbation(
        self, whirl_frequency, deep, rho_base, t_base, u_base, w_base, p_base
    ):
        """Integrate the perturbation equations along the seal.

        Marches the four simultaneous clearance perturbations from inlet to
        outlet and accumulates the circumferential pressure and shear
        integrals (trapezoidal rule) that build the reaction forces. Returns
        the sine and cosine force components in the X and Y directions.
        """
        pi_radius = np.pi * self._shaft_radius
        pert = np.zeros((5, 4, self.nz + 1))
        h_pert = np.array([self.radial_clearance, 0.0, 0.0, 0.0, 1.0])
        fx_sin = fx_cos = fy_sin = fy_cos = 0.0
        shear_end = np.zeros(4)
        for iz in range(1, self.nz + 1):
            iz1 = iz - 1
            base_old = {
                "rho": rho_base[iz1],
                "T": t_base[iz1],
                "u": u_base[iz1],
                "w": w_base[iz1],
                "p": p_base[iz1],
            }
            base_new = {
                "rho": rho_base[iz],
                "T": t_base[iz],
                "u": u_base[iz],
                "w": w_base[iz],
                "p": p_base[iz],
            }
            pert_old = {
                "rho": pert[0, :, iz1],
                "T": pert[1, :, iz1],
                "u": pert[2, :, iz1],
                "w": pert[3, :, iz1],
                "p": pert[4, :, iz1],
            }
            pert_new, shear = self._one_step_perturbed(
                h_pert, whirl_frequency, deep, base_old, base_new, pert_old, iz
            )
            pert[0, :, iz], pert[1, :, iz], pert[2, :, iz], pert[3, :, iz] = (
                pert_new["rho"],
                pert_new["T"],
                pert_new["u"],
                pert_new["w"],
            )
            pert[4, :, iz] = self.R * (
                base_new["T"] * pert[0, :, iz] + base_new["rho"] * pert[1, :, iz]
            )
            fx_sin += pi_radius * (-shear[0] - pert[4, 2, iz])
            fx_cos += pi_radius * (-shear[1] - pert[4, 3, iz])
            fy_sin += pi_radius * (shear[2] - pert[4, 0, iz])
            fy_cos += pi_radius * (shear[3] - pert[4, 1, iz])
            if iz == self.nz:
                shear_end = shear
        fx_sin = (
            fx_sin - 0.5 * pi_radius * (-shear_end[0] - pert[4, 2, self.nz])
        ) * self.dz
        fx_cos = (
            fx_cos - 0.5 * pi_radius * (-shear_end[1] - pert[4, 3, self.nz])
        ) * self.dz
        fy_sin = (
            fy_sin - 0.5 * pi_radius * (shear_end[2] - pert[4, 0, self.nz])
        ) * self.dz
        fy_cos = (
            fy_cos - 0.5 * pi_radius * (shear_end[3] - pert[4, 1, self.nz])
        ) * self.dz
        return fx_sin, fx_cos, fy_sin, fy_cos

    def _solve_force_coefficients(self, base_state_results):
        """Extract stiffness, damping and mass coefficients.

        A static perturbation (zero whirl frequency) gives the stiffness; a
        second perturbation at the whirl frequency separates the damping (from
        the sine force components) and the mass (from the cosine components).
        Returns the coefficient dict and the base pressure distribution.
        """
        mdot = base_state_results["mdot"]
        t_base = base_state_results["t"]
        mz2_base = base_state_results["mz2"]
        mt_base = base_state_results["mt"]

        rho_base, u_base, w_base = [np.zeros(self.nz + 1) for _ in range(3)]
        for iz in range(self.nz + 1):
            term = self.gamma * self.R * t_base[iz]
            sqrt_term = np.sqrt(term) if term > 0 else 0
            u_base[iz] = np.sqrt(mz2_base[iz]) * sqrt_term if mz2_base[iz] > 0 else 0
            w_base[iz] = mt_base[iz] * sqrt_term
            rho_base[iz] = mdot / (self.area * u_base[iz]) if u_base[iz] > 1e-9 else 0
        p_base = rho_base * self.R * t_base[: self.nz + 1]

        deep = self.cell_depth / self.gamma

        _, fx_cos, _, fy_cos = self._integrate_perturbation(
            0.0, deep, rho_base, t_base, u_base, w_base, p_base
        )
        K_dir, k_cross = -fx_cos, fy_cos

        whirl_frequency = self.omega * self.excitation_ratio
        if abs(whirl_frequency) < 1e-9:
            force_coefficients = {
                "K_dir": K_dir,
                "k_cross": k_cross,
                "M_dir": 0,
                "m_cross": 0,
                "C_dir": 0,
                "c_cross": 0,
            }
            return force_coefficients, p_base

        fx_sin, fx_cos, fy_sin, fy_cos = self._integrate_perturbation(
            whirl_frequency, deep, rho_base, t_base, u_base, w_base, p_base
        )
        force_coefficients = {
            "K_dir": K_dir,
            "k_cross": k_cross,
            "M_dir": (K_dir + fx_cos) / whirl_frequency**2,
            "m_cross": (k_cross - fy_cos) / whirl_frequency**2,
            "C_dir": fx_sin / whirl_frequency,
            "c_cross": -fy_sin / whirl_frequency,
        }
        return force_coefficients, p_base


class HolePatternSeal(SealElement):
    """Hole-pattern annular seal - Bulk flow model with dynamic coefficients.

    This class provides a **comprehensive numerical model** for annular seals with
    hole (pocket) patterns using bulk flow theory. The model solves 1D compressible
    flow equations with perturbation analysis to calculate leakage and rotordynamic
    force coefficients.

    **Theoretical Approach:**

    The model solves the **1D bulk flow problem** using:

    1. **Base State Calculation** (equilibrium flow):
       - Compressible flow through annular clearance with hole patterns
       - Governing equations for axial Mach number, temperature, and tangential Mach number
       - Predictor-corrector integration (modified Euler method)
       - Friction effects from both stator and rotor surfaces
       - Inlet and exit loss modeling with adjustable coefficients
       - Reynolds number-dependent friction factors
       - Iterative solution to match outlet pressure using relaxation method

    2. **Leakage Calculation**:
       - Mass flow rate determined from pressure balance
       - Choke detection (critical Mach number checking)
       - Accounts for entrance losses, friction, and exit losses
       - Temperature-dependent viscosity using Sutherland's law

    3. **Perturbation Analysis** (for dynamic coefficients):
       - Small harmonic perturbations in clearance (4 directions: ±X, ±Y)
       - Linearized perturbation equations for density, temperature, velocities
       - 4×4 system of equations solved at each axial station
       - Predictor-corrector integration for perturbed variables
       - Accounts for:
         * Temporal inertia effects (mass matrix)
         * Fluid inertia and convection
         * Compressibility effects
         * Friction perturbations
         * Pressure gradient perturbations
         * Preswirl and rotation effects

    4. **Force Coefficients Extraction**:
       - Stiffness (K): From static displacement perturbations
       - Damping (C): From velocity perturbations at whirl frequency
       - Mass (M): From acceleration perturbations (inertia effects)
       - Direct and cross-coupled terms
       - Integrated over seal length using trapezoidal rule

    Parameters
    ----------
    n : int
        Node in which the bearing will be located.
    shaft_diameter : float, pint.Quantity
        Diameter of the shaft (m).
    radial_clearance : float, pint.Quantity
        Seal radial clearance (m).
    axial_length : float, pint.Quantity
        Axial length of the seal (m).
    relative_roughness : float
        Relative roughness E / D (roughness / diameter) of the shaft,
        dimensionless.
    cell_length : float, pint.Quantity
        Typical length of a cell in the axial direction (m).
    cell_width : float, pint.Quantity
        Typical length of a cell in the azimuthal direction (m).
    cell_depth : float, pint.Quantity
        Depth of a cell (m).
    inlet_pressure : float
        Inlet pressure (Pa).
    outlet_pressure : float
        Outlet pressure (Pa).
    inlet_temperature : float
        Inlet temperature (deg K).
    frequency : list, pint.Quantity
        Shaft rotational speeds (rad/s). The coefficients are evaluated at
        the whirl frequency ``excitation_ratio * frequency``.
    gas_composition : dict, optional
        Gas composition as a dictionary {component: molar_fraction}.
    molar_mass : float, pint.Quantity, optional
        Molecular mass (kg/kgmol). For Air: molar_mass=28.97 kg/kgmol. Required if gas_composition is None.
        Default is None.
    gamma : float, optional
        Gas constant gamma (Cp/Cv). For Air: gamma=1.4. Required if gas_composition is None.
        Default is None.
    sutherland_b : float, optional
        b coefficient for the Sutherland viscosity model. Required if gas_composition is None.
        Default is None.
    sutherland_s : float, optional
        s coefficient for the Sutherland viscosity model. Required if gas_composition is None.
        Default is None.
    preswirl : float, optional
        Ratio of the circumferential velocity of the gas to the surface velocity of the shaft.
        Default is 0.0.
    entrance_loss_coefficient : float, optional
        Entrance loss coefficient.
        Default is 0.1.
    exit_loss_coefficient : float, optional
        Exit loss coefficient.
        Default is 0.5.
    excitation_ratio : float, optional
        Ratio of the excitation (whirl) frequency to the rotational speed;
        1.0 means synchronous excitation.
        Default is 1.0.
    nz : int, optional
        Number of discretization points in the axial direction.
        Default is 80.
    max_iterations : int, optional
        Maximum number of iterations for basic state calculation.
        Default is 180.
    tolerance : float, optional
        Tolerance of the solution expressed as a percentage of the pressure differential across the seal.
        Default is 0.0001.
    first_step_size : float, optional
        Initial step for the solution method. It should not be more than 0.01.
        Default is 0.01.
    relaxation_factor : float, optional
        Relaxation factor. Should be smaller than 0.1.
        Default is 0.1.
    tag : str, optional
        A tag to name the element.
        Default is None.
    n_link : int, optional
        Node to which the bearing will connect. If None the bearing is
        connected to ground.
        Default is None.
    scale_factor : float, optional
        The scale factor is used to scale the bearing drawing.
        Default is 1.
    color : str, optional
        A color to be used when the element is represented.
        Default is "#77ACA2".

    Examples
    --------
    >>> from ross.seals.holepattern_seal import HolePatternSeal
    >>> from ross.units import Q_
    >>> holepattern = HolePatternSeal(
    ...     n=0,
    ...     shaft_diameter=0.145,
    ...     radial_clearance=0.0003,
    ...     axial_length=0.04699,
    ...     relative_roughness=0.0001,
    ...     cell_length=0.003175,
    ...     cell_width=0.003175,
    ...     cell_depth=0.0025,
    ...     inlet_pressure=689000.0,
    ...     outlet_pressure=94300.0,
    ...     inlet_temperature=322.0,
    ...     frequency=Q_([8000], "RPM"),
    ...     gas_composition={"Nitrogen": 0.79, "Oxygen": 0.21},
    ...     preswirl=0.8,
    ...     entrance_loss_coefficient=0.5,
    ...     exit_loss_coefficient=1.0,
    ...     nz=18
    ... )
    """

    _pressure_plot_label = "Hole Pattern Seal"

    @check_units
    def __init__(
        self,
        n,
        shaft_diameter,
        radial_clearance,
        axial_length,
        relative_roughness,
        cell_length,
        cell_width,
        cell_depth,
        inlet_pressure,
        outlet_pressure,
        inlet_temperature,
        frequency,
        gas_composition=None,
        molar_mass=None,
        gamma=None,
        sutherland_b=None,
        sutherland_s=None,
        preswirl=0.0,
        entrance_loss_coefficient=0.1,
        exit_loss_coefficient=0.5,
        excitation_ratio=1.0,
        nz=80,
        max_iterations=180,
        tolerance=0.0001,
        first_step_size=0.01,
        relaxation_factor=0.1,
        **kwargs,
    ):
        self.n = n
        self.shaft_diameter = shaft_diameter
        self._shaft_radius = shaft_diameter / 2
        self.radial_clearance = radial_clearance
        self.axial_length = axial_length
        self.relative_roughness = relative_roughness
        self.cell_length = cell_length
        self.cell_width = cell_width
        self.cell_depth = cell_depth
        self.inlet_pressure = inlet_pressure
        self.outlet_pressure = outlet_pressure
        self.inlet_temperature = inlet_temperature
        self.frequency = frequency
        self.gas_composition = gas_composition
        self.preswirl = preswirl
        self.entrance_loss_coefficient = entrance_loss_coefficient
        self.exit_loss_coefficient = exit_loss_coefficient
        self.excitation_ratio = excitation_ratio
        self.nz = nz
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.first_step_size = first_step_size
        self.relaxation_factor = relaxation_factor

        if gas_composition is not None:

            def sutherland_formula(T, b, S):
                return (b * T ** (3 / 2)) / (S + T)

            state, molar_mass, gamma, R = extract_gas_properties(
                gas_composition, inlet_pressure, inlet_temperature
            )

            x = []
            y = []
            for T in range(260, 400, 20):
                try:
                    state.update(p=state.p(), T=T)
                    x.append(T)
                    y.append(state.viscosity().m)
                except ValueError:
                    # Skip temperatures where the state update fails (e.g.
                    # HEOS convergence issues).
                    continue

            if len(x) < 3:
                raise RuntimeError(
                    f"Could not collect enough viscosity data points ({len(x)} "
                    "points) to fit Sutherland coefficients. Try providing "
                    "sutherland_b, sutherland_s, molar_mass, and gamma manually."
                )

            popt, _ = curve_fit(sutherland_formula, x, y)
            sutherland_b, sutherland_s = popt
        else:
            R = 8314.0 / molar_mass  # Universal gas constant over molar mass.

        self.R = R
        self.molar_mass = molar_mass
        self.gamma = gamma
        self.sutherland_b = sutherland_b
        self.sutherland_s = sutherland_s

        self.dz = axial_length / float(nz)
        self.z = np.zeros(nz + 4)
        self.z[0] = -self.dz
        self.z[1] = 0.0
        self.z[2:-1] = np.arange(nz + 1) * self.dz
        self.z[-1] = self.z[-2]

        self.solver = HolePatternSolver(
            shaft_radius=self._shaft_radius,
            radial_clearance=radial_clearance,
            axial_length=axial_length,
            relative_roughness=relative_roughness,
            cell_depth=cell_depth,
            inlet_pressure=inlet_pressure,
            outlet_pressure=outlet_pressure,
            inlet_temperature=inlet_temperature,
            preswirl=preswirl,
            entrance_loss_coefficient=entrance_loss_coefficient,
            exit_loss_coefficient=exit_loss_coefficient,
            excitation_ratio=excitation_ratio,
            nz=nz,
            max_iterations=max_iterations,
            tolerance=tolerance,
            first_step_size=first_step_size,
            relaxation_factor=relaxation_factor,
            R=R,
            gamma=gamma,
            sutherland_b=self.sutherland_b,
            sutherland_s=self.sutherland_s,
        )

        coefficients_dict = {}
        if kwargs.get("kxx") is None:
            results = solve_frequencies(
                self.solver.solve, frequency, parallel_threshold=2
            )

            self.p = [r["pressure"] for r in results]

            coefficients_dict = {
                c: [k[c] for k in results]
                for c in results[0].keys()
                if c not in ["pressure"]
            }

        super().__init__(
            self.n,
            frequency=frequency,
            **coefficients_dict,
            **kwargs,
        )
