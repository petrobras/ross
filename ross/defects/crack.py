"""Cracks module.

This module defines the Defect classes for cracks on the shaft. There 
are a number of options, for the formulation of 6 DoFs (degrees of freedom).
"""
from abc import ABC, abstractmethod

import ross
import numpy as np
import scipy as sp
import scipy.integrate
import scipy.linalg
import time
from ross.units import Q_

import plotly.graph_objects as go

from .abs_defect import Defect
from .integrate_solver import Integrator

__all__ = [
    "Crack",
]


class Crack:
    """A Gasch and Mayes crack models for a shaft.
    
    Calculates the dynamic forces of a crack on a given shaft element.

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

    def __init__(self, dt, tI, tF, cd, n_crack, speed):

        self.dt = dt
        self.tI = tI
        self.tF = tF
        self.cd = cd
        self.n_crack = n_crack
        self.speed = speed

    def run(self, rotor):
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
        self.radius = rotor.elements[self.n_crack].odl / 2
        self.ndof = rotor.ndof
        self.iteration = 0

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

        x = Integrator(0, y0, self.tF, self.dt, self._equation_of_movement)
        x = x.rk4()
        t2 = time.time()
        print(f"spend time: {t2-t1} s")

        self.displacement = x[:12, :]
        self.velocity = x[12:, :]
        self.time_vector = t_eval
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

        ft = self._force()
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

    def _crack(self):
        """Reaction forces of cracked element
        
        Returns
        -------
        F_mis_p(12,n) : numpy.ndarray
            Excitation force caused by the parallel misalignment for a 6DOFs system with 'n' values of angular position  
        """

        dof_crack = np.arange((self.n_crack * 6), (self.n_crack * 6 + 6))

