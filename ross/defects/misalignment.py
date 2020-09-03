"""Misalignment module.

This module defines the Defect classes which will be used to represent problems
such as misalignments of various types on the shaft coupling, rubbing effects and
cracks on the shaft. There are a number of options, for the formulation of 6 DoFs
(degrees of freedom).
"""
from abc import ABC, abstractmethod

import numpy as np
import scipy as sp
import scipy.integrate
import scipy.linalg
import time
from ross.units import Q_

import plotly.graph_objects as go

from .abs_defect import Defect

__all__ = [
    "MisalignmentFlexParallel",
    "MisalignmentFlexAngular",
    "MisalignmentFlexCombined",
    "MisalignmentRigid",
    # "Rubbing",
    # "CrackGasch",
    # "CrackMayes",
]


class MisalignmentFlex(Defect, ABC):
    """A flexible coupling with misalignment of some kind.
    
    Calculates the dynamic reaction force of hexangular flexible coupling 
    induced by 6DOF's rotor parallel and angular misalignment.

    Parameters
    ----------
    dt : float
        Time step
    tI : float
        Initial time
    tF : float
        Final time
    kd : float
        Radial stiffness of flexible coupling
    ks : float
        Bending stiffness of flexible coupling
    eCOUPx : float
        Parallel misalignment offset between driving rotor and driven rotor along X direction
    eCOUPy : float
        Parallel misalignment offset between driving rotor and driven rotor along Y direction
    misalignment_angle : float
        Angle of the angular misaligned 
    TD : float
        Driving torque
    TL : float
        Driven torque
    n1 : float
        Node where the misalignment is ocurring
    speed : float
        Operational speed of the machine

    Returns
    -------
    A force to be applied on the shaft.

    References
    ----------
    .. [1] 'Xia, Y., Pang, J., Yang, L., Zhao, Q., & Yang, X. (2019). Study on vibration response 
    and orbits of misaligned rigid rotors connected by hexangular flexible coupling. Applied 
    Acoustics, 155, 286-296..

    Examples
    --------
    AQUI AINDA TEM QUE SER ATUALIZADO, ABAIXO SEGUE SOMENTE UM EXEMPLO PARA A "SHAFT ELEMENT"
    >>> from ross.materials import steel
    >>> Timoshenko_Element = ShaftElement(
    ...                         material=steel, L=0.5, idl=0.05, odl=0.1,
    ...                         rotary_inertia=True,
    ...                         shear_effects=True)
    >>> Timoshenko_Element.phi
    0.1571268472906404
    """

    def __init__(
        self, dt, tI, tF, kd, ks, eCOUPx, eCOUPy, misalignment_angle, TD, TL, n1, speed,
    ):
        self.dt = dt
        self.tI = tI
        self.tF = tF
        self.kd = kd
        self.ks = ks
        self.eCOUPx = eCOUPx
        self.eCOUPy = eCOUPy
        self.misalignment_angle = misalignment_angle
        self.TD = TD
        self.TL = TL
        self.n1 = n1
        self.n2 = n1 + 1
        self.speed = speed

        self.speedI = speed
        self.speedF = speed

    def _run(self, rotor, mis_type=str):
        """Calculates the shaft angular position and the misalignment amount at X / Y directions.

        Parameters
        ----------
        radius : float
                Radius of shaft in node of misalignment
        ndof : float
                Total number of degrees of freedom

        Returns
        -------
        
                
        """
        self.rotor = rotor
        self.radius = rotor.elements[self.n1].odl
        self.ndof = rotor.ndof
        self.mis_type = mis_type
        self.iteration = 0

        self.Cte = (
            self.ks * self.radius * np.sqrt(2 - 2 * np.cos(self.misalignment_angle))
        )

        warI = self.speedI * np.pi / 30
        warF = self.speedF * np.pi / 30

        # parameters for the time integration
        self.lambdat = 0.00001
        Faxial = 0
        TorqueI = 0
        TorqueF = 0

        # pre-processing of auxilary variuables for the time integration
        self.sA = (
            warI * np.exp(-self.lambdat * self.tF)
            - warF * np.exp(-self.lambdat * self.tI)
        ) / (np.exp(-self.lambdat * self.tF) - np.exp(-self.lambdat * self.tI))
        self.sB = (warF - warI) / (
            np.exp(-self.lambdat * self.tF) - np.exp(-self.lambdat * self.tI)
        )

        # sAT = (
        #     TorqueI * np.exp(-lambdat * self.tF) - TorqueF * np.exp(-lambdat * self.tI)
        # ) / (np.exp(-lambdat * self.tF) - np.exp(-lambdat * self.tI))
        # sBT = (TorqueF - TorqueI) / (
        #     np.exp(-lambdat * self.tF) - np.exp(-lambdat * self.tI)
        # )

        # SpeedV = sA + sB * np.exp(-lambdat * self.t)
        # TorqueV = sAT + sBT * np.exp(-lambdat * self.t)
        # AccelV = -lambdat * sB * np.exp(-lambdat * self.t)

        # Determining the modal matrix
        self.K = self.rotor.K(self.speedI * np.pi / 30)
        self.C = self.rotor.C(self.speedI * np.pi / 30)
        self.G = self.rotor.G()
        self.M = self.rotor.M()
        self.Kst = self.rotor.Kst()

        _, ModMat = scipy.linalg.eigh(self.K, self.M, type=1, turbo=False,)
        ModMat = ModMat[:, :12]
        self.ModMat = ModMat

        # Modal transformations
        self.Mmodal = ((ModMat.T).dot(self.M)).dot(ModMat)
        self.Cmodal = ((ModMat.T).dot(self.C)).dot(ModMat)
        self.Gmodal = ((ModMat.T).dot(self.G)).dot(ModMat)
        self.Kmodal = ((ModMat.T).dot(self.K)).dot(ModMat)
        self.Kstmodal = ((ModMat.T).dot(self.Kst)).dot(ModMat)

        y0 = np.zeros(24)
        self.dt = 0.0001
        t_eval = np.arange(self.dt, self.tF, self.dt)
        self.inv_Mmodal = np.linalg.pinv(self.Mmodal)
        t1 = time.time()

        x = scipy.integrate.solve_ivp(
            self._equation_of_movement,
            (self.tI, self.tF),
            y0,
            method="Radau",
            t_eval=t_eval,
            # dense_output=True,
            # atol=1e-03,
            # rtol=0.1,
        )
        t2 = time.time()
        print(f"Time spent: {t2-t1} s")

        self.displacement = x.y[:12, :]
        self.velocity = x.y[12:, :]
        self.time_vector = x.t
        self.response = self.ModMat.dot(self.displacement)

    def _equation_of_movement(self, T, Y):
        self.iteration += 1
        if self.iteration % 10000 == 0:
            print(f"iteration: {self.iteration} \n time: {T}")

        positions = Y[:12]
        velocity = Y[12:]  # velocity ign space state

        self.angular_position = (
            self.sA * T
            - (self.sB / self.lambdat) * np.exp(-self.lambdat * T)
            + (self.sB / self.lambdat)
        )

        if self.mis_type is None or self.mis_type == "parallel":
            ft = self._parallel()
        elif self.mis_type == "angular":
            ft = self._angular()
        elif self.mis_type == "combined":
            ft = self._combined()
        else:
            raise Exception("Check the misalignment type!")

        ftmodal = (self.ModMat.T).dot(ft)

        # Omega = self.speedI * np.pi / 30
        Omega = self.sA + self.sB * np.exp(-self.lambdat * T)
        AccelV = -self.lambdat * self.sB * np.exp(-self.lambdat * T)

        # proper equation of movement to be integrated in time
        new_V_dot = (
            ftmodal
            - ((self.Cmodal + self.Gmodal * Omega)).dot(velocity)
            - ((self.Kmodal + self.Kstmodal * AccelV).dot(positions))
        ).dot(self.inv_Mmodal)

        new_X_dot = velocity

        new_Y = np.zeros(24)
        new_Y[:12] = new_X_dot
        new_Y[12:] = new_V_dot

        return new_Y

    def _parallel(self):
        """Reaction forces of parallel misalignment
        
        Returns
        -------
        F_mis_p(12,n) : numpy.ndarray
            Excitation force caused by the parallel misalignment for a 6DOFs system with 'n' values of angular position  
        """

        F_mis_p = np.zeros(self.ndof)

        fib = np.arctan(self.eCOUPy / self.eCOUPx)

        self.mi_y = (
            (
                np.sqrt(
                    self.radius ** 2
                    + self.eCOUPx ** 2
                    + self.eCOUPy ** 2
                    + 2
                    * self.radius
                    * np.sqrt(self.eCOUPx ** 2 + self.eCOUPy ** 2)
                    * np.sin(fib + self.angular_position)
                )
                - self.radius
            )
            * np.cos(self.angular_position)
            + (
                np.sqrt(
                    self.radius ** 2
                    + self.eCOUPx ** 2
                    + self.eCOUPy ** 2
                    + 2
                    * self.radius
                    * np.sqrt(self.eCOUPx ** 2 + self.eCOUPy ** 2)
                    * np.cos(np.pi / 6 + fib + self.angular_position)
                )
                - self.radius
            )
            * np.cos(2 * np.pi / 3 + self.angular_position)
            + (
                self.radius
                - np.sqrt(
                    self.radius ** 2
                    + self.eCOUPx ** 2
                    + self.eCOUPy ** 2
                    - 2
                    * self.radius
                    * np.sqrt(self.eCOUPx ** 2 + self.eCOUPy ** 2)
                    * np.sin(np.pi / 3 + fib + self.angular_position)
                )
            )
            * np.cos(4 * np.pi / 3 + self.angular_position)
        )

        self.mi_x = (
            (
                np.sqrt(
                    self.radius ** 2
                    + self.eCOUPx ** 2
                    + self.eCOUPy ** 2
                    + 2
                    * self.radius
                    * np.sqrt(self.eCOUPx ** 2 + self.eCOUPy ** 2)
                    * np.sin(fib + self.angular_position)
                )
                - self.radius
            )
            * np.sin(self.angular_position)
            + (
                np.sqrt(
                    self.radius ** 2
                    + self.eCOUPx ** 2
                    + self.eCOUPy ** 2
                    + 2
                    * self.radius
                    * np.sqrt(self.eCOUPx ** 2 + self.eCOUPy ** 2)
                    * np.cos(np.pi / 6 + fib + self.angular_position)
                )
                - self.radius
            )
            * np.sin(2 * np.pi / 3 + self.angular_position)
            + (
                self.radius
                - np.sqrt(
                    self.radius ** 2
                    + self.eCOUPx ** 2
                    + self.eCOUPy ** 2
                    - 2
                    * self.radius
                    * np.sqrt(self.eCOUPx ** 2 + self.eCOUPy ** 2)
                    * np.sin(np.pi / 3 + fib + self.angular_position)
                )
            )
            * np.sin(4 * np.pi / 3 + self.angular_position)
        )

        Fpy = self.kd * self.mi_y

        Fpx = self.kd * self.mi_x

        F_mis_p[0 + 6 * self.n1] = -Fpx
        F_mis_p[1 + 6 * self.n1] = Fpy
        F_mis_p[5 + 6 * self.n1] = self.TD
        F_mis_p[0 + 6 * self.n2] = Fpx
        F_mis_p[1 + 6 * self.n2] = -Fpy
        F_mis_p[5 + 6 * self.n2] = self.TL

        return F_mis_p

    def _angular(self):
        """Reaction forces of angular misalignment
        
        Returns
        -------
        F_mis_a(12,n) : numpy.ndarray
            Excitation force caused by the angular misalignment for a 6DOFs system with 'n' values of angular position 
        """
        F_mis_a = np.zeros(self.ndof)

        Fay = (
            np.abs(
                self.Cte
                * np.sin(self.angular_position)
                * np.sin(self.misalignment_angle)
            )
            * np.sin(self.angular_position + np.pi)
            + np.abs(
                self.Cte
                * np.sin(self.angular_position + 2 * np.pi / 3)
                * np.sin(self.misalignment_angle)
            )
            * np.sin(self.angular_position + np.pi + 2 * np.pi / 3)
            + np.abs(
                self.Cte
                * np.sin(self.angular_position + 4 * np.pi / 3)
                * np.sin(self.misalignment_angle)
            )
            * np.sin(self.angular_position + np.pi + 4 * np.pi / 3)
        )

        Fax = (
            np.abs(
                self.Cte
                * np.sin(self.angular_position)
                * np.sin(self.misalignment_angle)
            )
            * np.cos(self.angular_position + np.pi)
            + np.abs(
                self.Cte
                * np.sin(self.angular_position + 2 * np.pi / 3)
                * np.sin(self.misalignment_angle)
            )
            * np.cos(self.angular_position + np.pi + 2 * np.pi / 3)
            + np.abs(
                self.Cte
                * np.sin(self.angular_position + 4 * np.pi / 3)
                * np.sin(self.misalignment_angle)
            )
            * np.cos(self.angular_position + np.pi + 4 * np.pi / 3)
        )

        F_mis_a[0 + 6 * self.n1] = -Fax
        F_mis_a[1 + 6 * self.n1] = Fay
        F_mis_a[5 + 6 * self.n1] = self.TD
        F_mis_a[0 + 6 * self.n2] = Fax
        F_mis_a[1 + 6 * self.n2] = -Fay
        F_mis_a[5 + 6 * self.n2] = self.TL

        return F_mis_a

    def _combined(self):
        """Reaction forces of combined (parallel and angular) misalignment
        
        Returns
        -------
        F_misalign(12,n) : numpy.ndarray
            Excitation force caused by the combined misalignment for a 6DOFs system with 'n' values of angular position 
        """
        F_misalign = self._parallel() + self._angular()
        return F_misalign

    def _plot_time_response(
        self,
        probe,
        probe_units="rad",
        displacement_units="m",
        time_units="s",
        fig=None,
        **kwargs,
    ):
        """
        """
        if fig is None:
            fig = go.Figure()

        for i, p in enumerate(probe):
            dofx = p[0] * self.rotor.number_dof
            dofy = p[0] * self.rotor.number_dof + 1
            angle = Q_(p[1], probe_units).to("rad").m

            # fmt: off
            operator = np.array(
                [[np.cos(angle), - np.sin(angle)],
                 [np.cos(angle), + np.sin(angle)]]
            )

            _probe_resp = operator @ np.vstack((self.response[dofx,:], self.response[dofy,:]))
            probe_resp = (
                _probe_resp[0] * np.cos(angle) ** 2  +
                _probe_resp[1] * np.sin(angle) ** 2
            )
            # fmt: on

            probe_resp = Q_(probe_resp, "m").to(displacement_units).m

            fig.add_trace(
                go.Scatter(
                    x=Q_(self.time_vector, "s").to(time_units).m,
                    y=Q_(probe_resp, "m").to(displacement_units).m,
                    mode="lines",
                    name=f"Probe {i + 1}",
                    legendgroup=f"Probe {i + 1}",
                    showlegend=True,
                    hovertemplate=f"Time ({time_units}): %{{x:.2f}}<br>Amplitude ({displacement_units}): %{{y:.2e}}",
                )
            )

        fig.update_xaxes(title_text=f"Time ({time_units})")
        fig.update_yaxes(title_text=f"Amplitude ({displacement_units})")
        fig.update_layout(**kwargs)

        return fig

    def _plot_dfft(
        self, probe, probe_units="rad", fig=None, log=False, **kwargs,
    ):
        """
        """
        if fig is None:
            fig = go.Figure()

        for i, p in enumerate(probe):
            dofx = p[0] * self.rotor.number_dof
            dofy = p[0] * self.rotor.number_dof + 1
            angle = Q_(p[1], probe_units).to("rad").m

            # fmt: off
            operator = np.array(
                [[np.cos(angle), - np.sin(angle)],
                 [np.cos(angle), + np.sin(angle)]]
            )
            row, cols = self.response.shape
            _probe_resp = operator @ np.vstack((self.response[dofx,int(2*cols/3):], self.response[dofy,int(2*cols/3):]))
            probe_resp = (
                _probe_resp[0] * np.cos(angle) ** 2  +
                _probe_resp[1] * np.sin(angle) ** 2
            )
            # fmt: on

            amp, freq = self._dfft(probe_resp, self.dt)

            fig.add_trace(
                go.Scatter(
                    x=freq,
                    y=amp,
                    mode="lines",
                    name=f"Probe {i + 1}",
                    legendgroup=f"Probe {i + 1}",
                    showlegend=True,
                    hovertemplate=f"Frequency (Hz): %{{x:.2f}}<br>Amplitude (m): %{{y:.2e}}",
                )
            )

        fig.update_xaxes(title_text=f"Frequency (Hz)")
        fig.update_yaxes(title_text=f"Amplitude (m)")
        fig.update_layout(**kwargs)

        if log:
            fig.update_layout(yaxis_type="log")

        return fig

    def _dfft(self, x, dt):
        b = np.floor(len(x) / 2)
        c = len(x)
        df = 1 / (c * dt)

        x_amp = sp.fft(x)[: int(b)]
        x_amp = x_amp * 2 / c
        x_phase = np.angle(x_amp)
        x_amp = np.abs(x_amp)

        freq = np.arange(0, df * b, df)
        freq = freq[: int(b)]  # Frequency vector

        return x_amp, freq


class MisalignmentFlexParallel(MisalignmentFlex):
    def run(self, rotor, mis_type="parallel"):
        """[summary]

        Returns
        -------
        [type]
            [description]
        """
        return self._run(rotor, mis_type)

    def plot_time_response(
        self,
        probe,
        probe_units="rad",
        displacement_units="m",
        time_units="s",
        fig=None,
        **kwargs,
    ):
        return self._plot_time_response(
            probe,
            probe_units=probe_units,
            displacement_units=displacement_units,
            time_units=time_units,
            fig=fig,
            **kwargs,
        )

    def plot_dfft(self, probe, probe_units="rad", fig=None, log=False, **kwargs):
        return self._plot_dfft(
            probe, probe_units=probe_units, fig=fig, log=log, **kwargs
        )


class MisalignmentFlexAngular(MisalignmentFlex):
    def run(self, rotor, mis_type="angular"):
        return self._run(rotor, mis_type)

    def plot_time_response(
        self,
        probe,
        probe_units="rad",
        displacement_units="m",
        time_units="s",
        fig=None,
        **kwargs,
    ):
        return self._plot_time_response(
            probe,
            probe_units=probe_units,
            displacement_units=displacement_units,
            time_units=time_units,
            fig=fig,
            **kwargs,
        )

    def plot_dfft(self, probe, probe_units="rad", fig=None, log=False, **kwargs):
        return self._plot_dfft(
            probe, probe_units=probe_units, fig=fig, log=log, **kwargs
        )


class MisalignmentFlexCombined(MisalignmentFlex):
    def run(self, rotor, mis_type="combined"):
        return self._run(rotor, mis_type)

    def plot_time_response(
        self,
        probe,
        probe_units="rad",
        displacement_units="m",
        time_units="s",
        fig=None,
        **kwargs,
    ):
        return self._plot_time_response(
            probe,
            probe_units=probe_units,
            displacement_units=displacement_units,
            time_units=time_units,
            fig=fig,
            **kwargs,
        )

    def plot_dfft(self, probe, probe_units="rad", fig=None, log=False, **kwargs):
        return self._plot_dfft(
            probe, probe_units=probe_units, fig=fig, log=log, **kwargs
        )


class MisalignmentRigid(Defect, ABC):
    """A rigid coupling with parallel misalignment.
    
    Calculates the dynamic reaction force of hexangular rigid coupling 
    induced by 6DOF's rotor parallel misalignment.

    Parameters
    ----------
    dt : float
        Time step
    tI : float
        Initial time
    tF : float
        Final time
    kd : float
        Radial stiffness of flexible coupling
    ks : float
        Bending stiffness of flexible coupling
    eCOUPx : float
        Parallel misalignment offset between driving rotor and driven rotor along X direction
    eCOUPy : float
        Parallel misalignment offset between driving rotor and driven rotor along Y direction
    misalignment_angle : float
        Angle of the angular misaligned 
    TD : float
        Driving torque
    TL : float
        Driven torque
    n1 : float
        Node where the misalignment is ocurring
    speed : float
        Operational speed of the machine

    Returns
    -------
    A force to be applied on the shaft.

    References
    ----------
    .. [1] 'Xia, Y., Pang, J., Yang, L., Zhao, Q., & Yang, X. (2019). Study on vibration response 
    and orbits of misaligned rigid rotors connected by hexangular flexible coupling. Applied 
    Acoustics, 155, 286-296..

    Examples
    --------
    AQUI AINDA TEM QUE SER ATUALIZADO, ABAIXO SEGUE SOMENTE UM EXEMPLO PARA A "SHAFT ELEMENT"
    >>> from ross.materials import steel
    >>> Timoshenko_Element = ShaftElement(
    ...                         material=steel, L=0.5, idl=0.05, odl=0.1,
    ...                         rotary_inertia=True,
    ...                         shear_effects=True)
    >>> Timoshenko_Element.phi
    0.1571268472906404
    """

    def __init__(
        self, tI, tF, Kcoup_auxI, Kcoup_auxF, kCOUP, eCOUP, TD, TL, n1, speed,
    ):
        self.tI = tI
        self.tF = tF
        self.Kcoup_auxI = Kcoup_auxI
        self.Kcoup_auxF = Kcoup_auxF
        self.kCOUP = kCOUP
        self.eCOUP = eCOUP
        self.TD = TD
        self.TL = TL
        self.n1 = n1
        self.n2 = n1 + 1
        self.speedI = speed
        self.speedF = speed
        self.DoF = np.arange((self.n1 * 6), (self.n2 * 6 + 6))

    def run(self, rotor):
        """Calculates the shaft angular position and the misalignment amount at X / Y directions.

        Parameters
        ----------
        rotor : object
                All properties from the rotor model

        Returns
        -------
        The integrated variables.
                
        """

        self.rotor = rotor
        self.ndof = rotor.ndof
        self.iteration = 0

        warI = self.speedI * np.pi / 30
        warF = self.speedF * np.pi / 30

        self.lambdat = 0.00001
        # Faxial = 0
        # TorqueI = 0
        # TorqueF = 0

        # pre-processing of auxilary variuables for the time integration
        self.sA = (
            warI * np.exp(-self.lambdat * self.tF)
            - warF * np.exp(-self.lambdat * self.tI)
        ) / (np.exp(-self.lambdat * self.tF) - np.exp(-self.lambdat * self.tI))
        self.sB = (warF - warI) / (
            np.exp(-self.lambdat * self.tF) - np.exp(-self.lambdat * self.tI)
        )

        # THIS LINES ARE USED FOR RUN-UP EVALUATIONS, CURRENTLY DISABLED.
        # sAT = (
        #     TorqueI * np.exp(-lambdat * self.tF) - TorqueF * np.exp(-lambdat * self.tI)
        # ) / (np.exp(-lambdat * self.tF) - np.exp(-lambdat * self.tI))
        # sBT = (TorqueF - TorqueI) / (
        #     np.exp(-lambdat * self.tF) - np.exp(-lambdat * self.tI)
        # )

        # self.SpeedV = sA + sB * np.exp(-lambdat * t)
        # self.TorqueV = sAT + sBT * np.exp(-lambdat * t)
        # self.AccelV = -lambdat * sB * np.exp(-lambdat * t)

        # Determining the modal matrix
        self.K = self.rotor.K(self.speedI * np.pi / 30)
        self.C = self.rotor.C(self.speedI * np.pi / 30)
        self.G = self.rotor.G()
        self.M = self.rotor.M()
        self.Kst = self.rotor.Kst()

        _, ModMat = scipy.linalg.eigh(self.K, self.M, type=1, turbo=False,)
        ModMat = ModMat[:, :12]
        self.ModMat = ModMat

        # Modal transformations
        self.Mmodal = ((ModMat.T).dot(self.M)).dot(ModMat)
        self.Cmodal = ((ModMat.T).dot(self.C)).dot(ModMat)
        self.Gmodal = ((ModMat.T).dot(self.G)).dot(ModMat)
        self.Kmodal = ((ModMat.T).dot(self.K)).dot(ModMat)
        self.Kstmodal = ((ModMat.T).dot(self.Kst)).dot(ModMat)

        self.angANG = -np.pi / 180

        FFmis = np.zeros(self.ndof)

        y0 = np.zeros(24)
        self.dt = 0.0001
        t_eval = np.arange(self.dt, self.tF, self.dt)
        self.inv_Mmodal = np.linalg.pinv(self.Mmodal)
        t1 = time.time()

        x = scipy.integrate.solve_ivp(
            self._equation_of_movement,
            (self.tI, self.tF),
            y0,
            method="Radau",
            t_eval=t_eval,
            # dense_output=True,
            # atol=1e-03,
            # rtol=0.1,
        )
        t2 = time.time()
        print(f"Time spent: {t2-t1} s")

        self.displacement = x.y[:12, :]
        self.velocity = x.y[12:, :]
        self.time_vector = x.t
        self.response = self.ModMat.dot(self.displacement)

    def _equation_of_movement(self, T, Y):
        self.iteration += 1
        if self.iteration % 10000 == 0:
            print(f"iteration: {self.iteration} \n time: {T}")

        positions = Y[:12]
        velocity = Y[12:]  # velocity ign space state

        kcoup_auxt = self.K[5 + 6 * self.n2, 5 + 6 * self.n2] / (
            self.K[5 + 6 * self.n1, 5 + 6 * self.n1]
            + self.K[5 + 6 * self.n2, 5 + 6 * self.n2]
        )

        angular_position = (
            self.sA * T
            - (self.sB / self.lambdat) * np.exp(-self.lambdat * T)
            + (self.sB / self.lambdat)
        )

        self.angANG = (
            self.Kcoup_auxI * angular_position
            + self.Kcoup_auxF * angular_position
            + self.kCOUP
            * kcoup_auxt
            * self.eCOUP
            * (
                -positions[0 + 6 * self.n1] * np.sin(self.angANG)
                + positions[0 + 6 * self.n2] * np.sin(self.angANG)
                - positions[1 + 6 * self.n1] * np.cos(self.angANG)
                + positions[1 + 6 * self.n2] * np.cos(self.angANG)
            )
        )
        positionsFis = self.ModMat.dot(positions)
        Fmis, ft = self._parallel(positionsFis, self.angANG)
        ftmodal = (self.ModMat.T).dot(ft)

        # Omega = self.speedI * np.pi / 30
        Omega = self.sA + self.sB * np.exp(-self.lambdat * T)
        AccelV = -self.lambdat * self.sB * np.exp(-self.lambdat * T)

        # proper equation of movement to be integrated in time
        new_V_dot = (
            ftmodal
            - ((self.Cmodal + self.Gmodal * Omega)).dot(velocity)
            - ((self.Kmodal + self.Kstmodal * AccelV).dot(positions))
        ).dot(self.inv_Mmodal)

        new_X_dot = velocity

        new_Y = np.zeros(24)
        new_Y[:12] = new_X_dot
        new_Y[12:] = new_V_dot

        return new_Y

    def _parallel(self, positions, fir):

        k0 = self.kCOUP
        delta1 = self.eCOUP

        betam = 0

        k_misalignbeta1 = np.array(
            [
                -k0 * self.Kcoup_auxI * delta1 * np.sin(betam - fir),
                -k0 * self.Kcoup_auxI * delta1 * np.cos(betam - fir),
                0,
                0,
                0,
                0,
                +k0 * self.Kcoup_auxF * delta1 * np.sin(betam - fir),
                k0 * self.Kcoup_auxF * delta1 * np.cos(betam - fir),
                0,
                0,
                0,
                0,
            ]
        )

        K_mis_matrix = np.zeros((12, 12))
        K_mis_matrix[5, :] = k_misalignbeta1
        K_mis_matrix[11, :] = -k_misalignbeta1

        Force_kkmis = K_mis_matrix.dot(positions[self.DoF])

        F_misalign = np.array(
            [
                -(-k0 * delta1 * np.cos(betam - fir) + k0 * delta1),
                -k0 * delta1 * np.sin(betam - fir),
                0,
                0,
                0,
                self.TD - self.TL,
                -(k0 * delta1 * np.cos(betam - fir) - k0 * delta1),
                k0 * delta1 * np.sin(betam - fir),
                0,
                0,
                0,
                -(self.TD - self.TL),
            ]
        )

        Fmis = Force_kkmis + F_misalign
        FFmis = np.zeros(self.ndof)
        FFmis[self.DoF] = Fmis

        return Fmis, FFmis

    def plot_time_response(
        self,
        probe,
        probe_units="rad",
        displacement_units="m",
        time_units="s",
        fig=None,
        **kwargs,
    ):
        """
        """
        if fig is None:
            fig = go.Figure()

        for i, p in enumerate(probe):
            dofx = p[0] * self.rotor.number_dof
            dofy = p[0] * self.rotor.number_dof + 1
            angle = Q_(p[1], probe_units).to("rad").m

            # fmt: off
            operator = np.array(
                [[np.cos(angle), - np.sin(angle)],
                 [np.cos(angle), + np.sin(angle)]]
            )

            _probe_resp = operator @ np.vstack((self.response[dofx,:], self.response[dofy,:]))
            probe_resp = (
                _probe_resp[0] * np.cos(angle) ** 2  +
                _probe_resp[1] * np.sin(angle) ** 2
            )
            # fmt: on

            probe_resp = Q_(probe_resp, "m").to(displacement_units).m

            fig.add_trace(
                go.Scatter(
                    x=Q_(self.time_vector, "s").to(time_units).m,
                    y=Q_(probe_resp, "m").to(displacement_units).m,
                    mode="lines",
                    name=f"Probe {i + 1}",
                    legendgroup=f"Probe {i + 1}",
                    showlegend=True,
                    hovertemplate=f"Time ({time_units}): %{{x:.2f}}<br>Amplitude ({displacement_units}): %{{y:.2e}}",
                )
            )

        fig.update_xaxes(title_text=f"Time ({time_units})")
        fig.update_yaxes(title_text=f"Amplitude ({displacement_units})")
        fig.update_layout(**kwargs)

        return fig

    def plot_dfft(
        self, probe, probe_units="rad", fig=None, log=False, **kwargs,
    ):
        """
        """
        if fig is None:
            fig = go.Figure()

        for i, p in enumerate(probe):
            dofx = p[0] * self.rotor.number_dof
            dofy = p[0] * self.rotor.number_dof + 1
            angle = Q_(p[1], probe_units).to("rad").m

            # fmt: off
            operator = np.array(
                [[np.cos(angle), - np.sin(angle)],
                 [np.cos(angle), + np.sin(angle)]]
            )
            row, cols = self.response.shape
            _probe_resp = operator @ np.vstack((self.response[dofx,int(2*cols/3):], self.response[dofy,int(2*cols/3):]))
            probe_resp = (
                _probe_resp[0] * np.cos(angle) ** 2  +
                _probe_resp[1] * np.sin(angle) ** 2
            )
            # fmt: on

            amp, freq = self._dfft(probe_resp, self.dt)

            fig.add_trace(
                go.Scatter(
                    x=freq,
                    y=amp,
                    mode="lines",
                    name=f"Probe {i + 1}",
                    legendgroup=f"Probe {i + 1}",
                    showlegend=True,
                    hovertemplate=f"Frequency (Hz): %{{x:.2f}}<br>Amplitude (m): %{{y:.2e}}",
                )
            )

        fig.update_xaxes(title_text=f"Frequency (Hz)")
        fig.update_yaxes(title_text=f"Amplitude (m)")
        fig.update_layout(**kwargs)

        if log:
            fig.update_layout(yaxis_type="log")

        return fig

    def _dfft(self, x, dt):
        b = np.floor(len(x) / 2)
        c = len(x)
        df = 1 / (c * dt)

        x_amp = sp.fft(x)[: int(b)]
        x_amp = x_amp * 2 / c
        x_phase = np.angle(x_amp)
        x_amp = np.abs(x_amp)

        freq = np.arange(0, df * b, df)
        freq = freq[: int(b)]  # Frequency vector

        return x_amp, freq

