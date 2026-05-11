# -*- coding: utf-8 -*-
"""
Created on Fri May  1 11:25:24 2026

@author: bruno

SourceAC - Ideal 3-phase AC source with harmonic distortion and voltage unbalance support.
"""
import plotly.graph_objects as go

import numpy as np


class SourceAC:
    """
    Ideal 3-phase AC source with support for:
      - Arbitrary harmonic orders and amplitudes (fHO, aHO)
      - Per-phase voltage magnitude unbalance (Vunb) [%]
      - Per-phase voltage angle deviation (Adev) [degrees]

    Parameters
    ----------
    voltage_net  : float  – RMS phase voltage [V]
    frequency_net  : float  – Fundamental frequency [Hz]
    theta0: float  – Initial phase angle [rad] (default 0)
    fHO   : list/array of int   – Harmonic orders  (e.g. [5, 7, 11])
    aHO   : list/array of float – Harmonic amplitudes as % of voltage_net (e.g. [10, 5, 2])
    Vunb  : list/array of float – Voltage magnitude unbalance per phase [%]
                                  Order: [phase_A, phase_B, phase_C]
                                  Positive → higher voltage; Negative → lower voltage.
    Adev  : list/array of float – Angle deviation per phase [degrees]
                                  Order: [phase_A, phase_B, phase_C]
    Behaviour
    ----------
    Configuring Harmonics and Unbalances: set fHO,aHO, Vunb,Adev --> _pending
    Enabling disturbances: call .harmonics("enable") / .unbalances("enable") -->_active
    Disabling disturbances: call .harmonics("disable") / .unbalances("disable"), keeping configuration

    """

    # Phase angle offsets for a balanced 3-phase system [rad]
    # Organization: [phase_A, phase_B, phase_C]
    _PHASE_OFFSET = [0.0, -2*np.pi/3, +2*np.pi/3]
    _PHASE_LABEL  = ['A', 'B', 'C']

    def __init__(self, voltage_net, frequency_net, theta0=0.0,
                 fHO=None, aHO=None,
                 Vunb=None, Adev=None):

        self.voltage_net   = voltage_net
        self.frequency_net   = frequency_net
        self.theta0 = theta0

        # --- Harmonics ---
        self._fHO_pending  = None   # staged (not yet enabled)
        self._aHO_pending  = None
        self._fHO_active   = None   # active configuration
        self._aHO_active   = None
        self._harmonics_on = False

        if fHO is not None:
            self._fHO_pending, self._aHO_pending = self._parse_harmonics(fHO, aHO)

        # --- Unbalances ---
        self._Vunb_pending   = None
        self._Adev_pending   = None
        self._Vunb_active    = None
        self._Adev_active    = None
        self._unbalances_on  = False

        if Vunb is not None or Adev is not None:
            self._Vunb_pending = self._parse_3vec(Vunb, 0.0, 'Vunb')
            self._Adev_pending = self._parse_3vec(Adev, 0.0, 'Adev')

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                  #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_harmonics(fHO, aHO):
        """Validate and align fHO / aHO vectors."""
        fHO = list(fHO)
        n   = len(fHO)
        # Adjust aHO vector to the fHO length
        if aHO is None:
            aHO = [0.0] * n
        else:
            aHO = list(aHO)
            if len(aHO) < n:
                aHO += [0.0] * (n - len(aHO)) # filling up to the fHO length
            else:
                aHO = aHO[:n]          # truncate to fHO length

        # Clamp amplitudes to [0, 100]
        aHO = [float(np.clip(a, 0.0, 100.0)) for a in aHO]
        # Truncate fHO to integer numbers
        fHO = [int(h) for h in fHO]
        return fHO, aHO

    @staticmethod
    def _parse_3vec(v, default, name):
        """Return a 3-element list. Fills missing entries with *default*."""
        """Addresses both unbalances and deviations."""
        if v is None:
            return [default, default, default]
        v = list(v)
        if len(v) < 3:
            v += [default] * (3 - len(v))
        elif len(v) > 3:
            v = v[:3]
        return [float(x) for x in v]

    def _build_phase_voltage(self, t, phase_id):
        """
        Compute instantaneous voltage for one phase at time *t*, applying 
         harmonics frequencies, unbalanced amplitudes and angle deviations
        phase_id : 0 → A, 1 → B, 2 → C
        """
        phi   = self._PHASE_OFFSET[phase_id]   # nominal phase shift
        w = 2 * np.pi * self.frequency_net

        # --- Unbalance factors for this phase ---
        if self._unbalances_on and self._Vunb_active is not None:
            vd_pct =  self._Vunb_active[phase_id]   # % magnitude deviation
            ad_deg =  self._Adev_active[phase_id]    # angle deviation [°]
        else:
            vd_pct = 0.0
            ad_deg = 0.0

        mag_factor   = (100.0 + vd_pct) / 100.0       # amplitude scale
        angle_offset = ad_deg * np.pi / 180.0          # extra phase [rad]

        # Phase A gets +Adev, Phase B neutral, Phase C gets –Adev
        # (mirrors the sign convention used in the original snippet)
        sign = [+1.0, 0.0, -1.0][phase_id]
        angle_offset *= sign

        # --- Fundamental ---
        a_harm =0.0
        for a_harm_id in self._aHO_active:
            a_harm +=a_harm_id
        
        harm_factor = ((100 - a_harm)/100)
        
        v = np.sqrt(2)*self.voltage_net * mag_factor * harm_factor * np.cos(w * t + self.theta0 + phi + angle_offset)

        # --- Harmonics ---
        if self._harmonics_on and self._fHO_active:
            for h, a in zip(self._fHO_active, self._aHO_active):
                v += (a / 100.0) *np.sqrt(2)* self.voltage_net * mag_factor * np.cos(
                    h * w * t + self.theta0 + phi + angle_offset )

        return v

    # ------------------------------------------------------------------ #
    #  __call__ – voltage waveform at time t                              #
    # ------------------------------------------------------------------ #
    def __call__(self, t):
        """Return instantaneous phase voltages (vas, vbs, vcs) at time *t*."""
        vas = self._build_phase_voltage(t, 0)
        vbs = self._build_phase_voltage(t, 1)
        vcs = self._build_phase_voltage(t, 2)
        return vas, vbs, vcs

    # ------------------------------------------------------------------ #
    #  harmonics()                                                         #
    # ------------------------------------------------------------------ #
    def harmonics(self, *args, fHO=None, aHO=None):
        """
        Configure, enable, disable, or report harmonic distortion.

        Usage
        -----
        harmonics()                    → print report of current config
        harmonics(fHO=[5,7], aHO=[10,5])
                                       → stage a new harmonic config
        harmonics("enable")            → activate staged config
        harmonics("disable")           → deactivate harmonics (config kept)
        """
        # ---- Single string command ----
        if len(args) == 1 and isinstance(args[0], str):
            cmd = args[0].strip().lower()
            if cmd == 'enable':
                if self._fHO_pending is not None:
                    self._fHO_active  = self._fHO_pending
                    self._aHO_active  = self._aHO_pending
                    self._harmonics_on = True
                    print("[harmonics] Harmonic distortion ENABLED.")
                else:
                    print("[harmonics] Nothing staged – call harmonics(fHO=..., aHO=...) first.")
                return
            elif cmd == 'disable':
                self._harmonics_on = False
                self._fHO_active   = None
                self._aHO_active   = None
                print("[harmonics] Harmonic distortion DISABLED.")
                return
            else:
                print(f"[harmonics] Unknown command '{args[0]}'. Use 'enable' or 'disable'.")
                return

        # ---- Stage new configuration ----
        if fHO is not None or aHO is not None:
            if fHO is None:
                print("[harmonics] fHO is required when staging a new harmonic config.")
                return
            self._fHO_pending, self._aHO_pending = self._parse_harmonics(fHO, aHO)
            print("[harmonics] New harmonic config staged. Call harmonics('enable') to activate.")
            return

        # ---- Report ----
        self._print_harmonics_report()

    def _print_harmonics_report(self):
        """Print a formatted harmonic distortion report."""
        print("=" * 45)
        print("  Harmonic Distortion Report")
        print("=" * 45)
        status = "ENABLED" if self._harmonics_on else "DISABLED"
        print(f"  Status : {status}")
        print(f"  {'Order':<10} {'Amplitude':>12}")
        print(f"  {'-'*10} {'-'*12}")
        print(f"  {'Fundamental':<10} {'100.00 %':>12}")

        if self._harmonics_on and self._fHO_active:
            for h, a in zip(self._fHO_active, self._aHO_active):
                suffix = {1:'st', 2:'nd', 3:'rd'}.get(h % 10 if h % 100 not in (11,12,13) else 0, 'th')
                label  = f"{h}{suffix}. harmonic"
                print(f"  {label:<10} {a:>11.2f} %")
        elif self._fHO_pending:
            print("  (staged – not yet enabled)")
            for h, a in zip(self._fHO_pending, self._aHO_pending):
                suffix = {1:'st', 2:'nd', 3:'rd'}.get(h % 10 if h % 100 not in (11,12,13) else 0, 'th')
                label  = f"{h}{suffix}. harmonic"
                print(f"  {label:<10} {a:>11.2f} %  [staged]")
        else:
            print("  (no harmonics configured)")
        print("=" * 45)

    # ------------------------------------------------------------------ #
    #  unbalances()                                                        #
    # ------------------------------------------------------------------ #
    def unbalances(self, *args, Vunb=None, Adev=None):
        """
        Configure, enable, disable, or report voltage unbalances.

        Usage
        -----
        unbalances()                           → print report
        unbalances(Vunb=[5,-3,0], Adev=[2,0,-2])
                                               → stage new unbalance config
        unbalances("enable")                   → activate staged config
        unbalances("disable")                  → deactivate unbalances
        """
        # ---- Single string command ----
        if len(args) == 1 and isinstance(args[0], str):
            cmd = args[0].strip().lower()
            if cmd == 'enable':
                if self._Vunb_pending is not None or self._Adev_pending is not None:
                    self._Vunb_active   = self._Vunb_pending if self._Vunb_pending is not None else [0.0]*3
                    self._Adev_active   = self._Adev_pending if self._Adev_pending is not None else [0.0]*3
                    self._unbalances_on = True
                    print("[unbalances] Voltage unbalances ENABLED.")
                else:
                    print("[unbalances] Nothing staged – call unbalances(Vunb=..., Adev=...) first.")
                return
            elif cmd == 'disable':
                self._unbalances_on = False
                self._Vunb_active   = None
                self._Adev_active   = None
                print("[unbalances] Voltage unbalances DISABLED.")
                return
            else:
                print(f"[unbalances] Unknown command '{args[0]}'. Use 'enable' or 'disable'.")
                return

        # ---- Stage new configuration ----
        if Vunb is not None or Adev is not None:
            self._Vunb_pending = self._parse_3vec(Vunb, 0.0, 'Vunb')
            self._Adev_pending = self._parse_3vec(Adev, 0.0, 'Adev')
            print("[unbalances] New unbalance config staged. Call unbalances('enable') to activate.")
            return

        # ---- Report ----
        self._print_unbalances_report()

    def _print_unbalances_report(self):
        """Print a formatted voltage unbalance report."""
        print("=" * 45)
        print("  Voltage Unbalance Report")
        print("=" * 45)
        status = "ENABLED" if self._unbalances_on else "DISABLED"
        print(f"  Status : {status}")

        Vunb = self._Vunb_active if self._unbalances_on and self._Vunb_active else (self._Vunb_pending or [0.0]*3)
        Adev = self._Adev_active if self._unbalances_on and self._Adev_active else (self._Adev_pending or [0.0]*3)
        staged = not self._unbalances_on and (self._Vunb_pending or self._Adev_pending)

        print()
        for i, ph in enumerate(self._PHASE_LABEL):
            tag = "  [staged]" if staged else ""
            print(f"  Voltage unbalance on phase {ph}: {Vunb[i]:>7.2f} %{tag}")
        print()
        for i, ph in enumerate(self._PHASE_LABEL):
            tag = "  [staged]" if staged else ""
            print(f"  Angle deviation on phase {ph}:   {Adev[i]:>7.2f} °{tag}")
        print("=" * 45)

    # ------------------------------------------------------------------ #
    #  __repr__                                                            #
    # ------------------------------------------------------------------ #
    def __repr__(self):
        return (
            f"SourceAC(voltage_net={self.voltage_net} V, frequency_net={self.frequency_net} Hz, "
            f"theta0={self.theta0:.4f} rad, "
            f"harmonics={'ON' if self._harmonics_on else 'OFF'}, "
            f"unbalances={'ON' if self._unbalances_on else 'OFF'})"
        )


    def plot(self, results):
        """Plot the simulation Voltage results.
        
        Parameters
        ----------
        results : dict
            Dictionary returned by the 'run' method containing lists of results.
        
        Returns
        -------
        fig_voltages: tuple of plotly.graph_objects.Figure
            
        """
    
        # Figure 1: 3-Phase Tensions
        fig_voltages = go.Figure()
        fig_voltages.add_trace(
            go.Scatter(x=results['time'], y=results['Vas'], name="Va (V)", line=dict(color='blue'))
        )
        fig_voltages.add_trace(
            go.Scatter(x=results['time'], y=results['Vbs'], name="Vb (V)", line=dict(color='black'))
        )
        fig_voltages.add_trace(
            go.Scatter(x=results['time'], y=results['Vcs'], name="Vc (V)", line=dict(color='red'))
        )
        fig_voltages.update_layout(
            title="Motor operation: 3-phase Source Voltage",
            xaxis_title="Time (s)",
            yaxis_title="Source Voltage (V)"
        )
        return fig_voltages

    def run(self, time_vector):
        """Run the simulation for a series of time steps.
    
        Parameters
        ----------
        time_vector : array_like
            Array of time steps.
    
        Returns
        -------
        results : dict
            A dictionary containing lists of results for the entire simulation:
            - time, Vas, Vbs, Vcs.
        """
        results = {
            'time': [], 'Vas': [], 'Vbs': [], 'Vcs': []
    
        }
    
        # Ensure inputs are iterable/arrays
        time_vector = np.array(time_vector)
    
        for i, t in enumerate(time_vector):
    
            # Run single phase voltade
            vas = self._build_phase_voltage(t, 0)
            vbs = self._build_phase_voltage(t, 1)
            vcs = self._build_phase_voltage(t, 2)
    
            step_result={
                'time': t,
                'Vas': vas,
                'Vbs': vbs,
                'Vcs': vcs} 
    
            # Append results
            for key in results:
                results[key].append(step_result[key])
    
        return results


# ============================================================= #
#  Quick usage example (run as script)                          #
# ============================================================= #

    def sourceAC_example():
        #if __name__ == '__main__':
        # --- Create source ---
        src = SourceAC(voltage_net=220.0, frequency_net=60.0)
        print(src)
        print()
    
        # --- Configure harmonics ---
        src.harmonics(fHO=[5, 7, 11, 13], aHO=[10.0, 5.0, 3.0])   # aHO shorter → 13th = 0%
        src.harmonics('enable')
        src.harmonics()                   # print report
    
        print()
    
        # --- Configure unbalances ---
        src.unbalances(Vunb=[5.0, 0.0, -3.0], Adev=[2.0, 0.0, -2.0])
        src.unbalances('enable')
        src.unbalances()                  # print report
    
        print()
    
        # --- Compute a few voltages ---
        t_samples = np.linspace(0, 1/60, 5)
        resp = src.run(t_samples)
        fig_V = src.plot(resp)
    
        print()
    
        # --- Disable harmonics ---
        src.harmonics('disable')
        src.harmonics()
        
        fig_V.write_html('plot.html', auto_open=True)
        
        return src, fig_V