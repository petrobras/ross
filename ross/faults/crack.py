from abc import ABC
from pathlib import Path

import numpy as np
import pandas as pd

import ross as rs
from ross.units import Q_, check_units

__all__ = ["Crack"]


class Crack(ABC):
    """Model a crack based on Linear Fracture Mechanics on a given shaft element
    of a rotor system.

    Contains transversal crack models :cite:`gasch1993survey` and :cite:`mayes1984analysis`.
    The reference coordenate system is:
        - x-axis and y-axis in the sensors' planes;
        - z-axis throught the shaft center.

    Parameters
    ----------
    rotor : ross.Rotor
        Rotor object.
    n : float
        Number of shaft element where crack is located.
    depth_ratio : float
        Crack depth ratio related to the diameter of the crack container element.
        A depth value of 0.1 is equal to 10%, 0.2 equal to 20%, and so on.
        This parameter is restricted to up to 50% within the implemented approach,
        as discussed in :cite `papadopoulos2004some`.
    crack_model : string, optional
        Name of the chosen crack model. The avaible types are: "Mayes" and "Gasch".
        Default is "Mayes".

    Returns
    -------
    A crack object.

    Attributes
    ----------
    shaft_elem : ross.ShaftElement
        A 6 degrees of freedom shaft element object where crack is located.
    K_elem : np.ndarray
        Stiffness matrix of the shaft element without crack.
    Ko : np.ndarray
        Stiffness of the shaft with the crack closed (equivalent to the shaft without crack).
    Kc : np.ndarray
        Stiffness of the shaft including compliance coefficients according to the crack depth.
    forces : np.ndarray
        Force matrix due to crack. Each row corresponds to a dof and each column to a time.

    References
    ----------
    .. bibliography::
        :filter: docname in docnames

    Examples
    --------
    >>> rotor = rs.rotor_example_with_damping()
    >>> fault = Crack(rotor, n=18, depth_ratio=0.2, crack_model="Gasch")
    >>> fault.shaft_elem
    ShaftElement(L=0.03, idl=0.0, idr=0.0, odl=0.019,  odr=0.019, material='Steel', n=18)
    """

    @check_units
    def __init__(
        self,
        rotor,
        n,
        depth_ratio,
        crack_model="Mayes",
        # cross_discret=None,
    ):
        self.rotor = rotor
        # self.cross_discret = cross_discret

        if depth_ratio <= 0.5:
            self.depth_ratio = depth_ratio
        else:
            raise ValueError(
                """
                The implemented approach is based on Linear Fracture Mechanics.
                For cracks deeper than 50% of diameter, this approach has a singularity and cannot be used.
                This is discussed in Papadopoulos (2004).
                """
            )
        
        self.crack_model = crack_model

        if crack_model is None or crack_model == "Mayes":
            self._crack_model = self.mayes
        elif crack_model == "Gasch":
            self._crack_model = self.gasch
        elif crack_model == "Flex Open": 
            self._crack_model = self.flex
        elif crack_model == "Flex Breathing":
            self._crack_model = self.flex
            # if cross_discret is None:
            #     raise ValueError("The 'cross_discret' argument must be provided for the flex breathing model.")
        else:
            raise Exception("Check the crack model!")

        # Shaft element with crack
        self.shaft_elem = [elm for elm in rotor.shaft_elements if elm.n == n][0]

        self.dofs = list(self.shaft_elem.dof_global_index.values())

        self.K_elem = self.shaft_elem.K()

        if self._crack_model == self.mayes or self._crack_model == self.gasch:

            dir_path = Path(__file__).parents[0] / "data/PAPADOPOULOS.csv"
            self.coefficient_data = pd.read_csv(dir_path)

            L = self.shaft_elem.L
            E = self.shaft_elem.material.E
            Ie = self.shaft_elem.Ie
            phi = self.shaft_elem.phi

            co1 = L**3 * (1 + phi / 4) / 3
            co2 = L**2 / 2
            co3 = L

            # fmt: off
            Co = np.array([
                [co1,   0,     0, co2],
                [  0,  co1, -co2,   0],
                [  0, -co2,  co3,   0],
                [co2,    0,    0, co3],
            ]) / (E * Ie)
            # fmt: on

            if self.depth_ratio == 0:
                Cc = Co
            else:
                c44 = self._get_coefficient("c44")
                c55 = self._get_coefficient("c55")
                c45 = self._get_coefficient("c45")

                Cc = Co + np.array(
                    [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, c55, c45], [0, 0, c45, c44]]
                )

            self.Ko = np.linalg.pinv(Co)
            self.Kc = np.linalg.pinv(Cc)

    def _get_coefficient(self, coeff):
        """Get terms of the compliance matrix.

        Parameters
        -----------
        coeff : string
            Name of the coefficient according to the corresponding direction.

        Returns
        -------
        c : np.ndarray
            Compliance coefficient according to the crack depth.
        """

        Poisson = self.shaft_elem.material.Poisson
        E = self.shaft_elem.material.E
        radius = self.shaft_elem.odl / 2

        c = np.array(pd.eval(self.coefficient_data[coeff]))
        ind = np.where(c[:, 1] >= self.depth_ratio * 2)[0]

        c = c[ind[0], 0] * (1 - Poisson**2) / (E * radius**3)

        return c

    def compute_crack_stiffness(self, ap):
        """Compute stiffness matrix of the shaft element with crack in inertial coordinates.

        Parameters
        ----------
        ap : float
            Angular position of the element.

        Returns
        -------
        K : np.ndarray
            Stiffness matrix of the cracked element.
        """
        if self._crack_model == self.mayes or self._crack_model == self.gasch:
            L = self.shaft_elem.L

            Kmodel = self._crack_model(ap)

            Toxy = np.array([[-1, 0], [-L, -1], [1, 0], [0, 1]])
            kxy = np.array([[Kmodel[0, 0], self.Ko[0, 3]], [self.Ko[3, 0], self.Ko[3, 3]]])
            Koxy = Toxy @ kxy @ Toxy.T

            Toyz = np.array([[-1, 0], [L, -1], [1, 0], [0, 1]])
            kyz = np.array([[Kmodel[1, 1], self.Ko[1, 2]], [self.Ko[2, 1], self.Ko[2, 2]]])
            Koyz = Toyz @ kyz @ Toyz.T

            # fmt: off
            K = np.array([
                [Koxy[0,0],         0,   0,         0, Koxy[0,1],   0, Koxy[0,2],         0,   0,         0, Koxy[0,3],   0],
                [        0, Koyz[0,0],   0, Koyz[0,1],         0,   0,         0, Koyz[0,2],   0, Koyz[0,3],         0,   0],
                [        0,         0,   0,         0,         0,   0,         0,         0,   0,         0,         0,   0],
                [        0, Koyz[1,0],   0, Koyz[1,1],         0,   0,         0, Koyz[1,2],   0, Koyz[1,3],         0,   0],
                [Koxy[1,0],         0,   0,         0, Koxy[1,1],   0, Koxy[1,2],         0,   0,         0, Koxy[1,3],   0],
                [        0,         0,   0,         0,         0,   0,         0,         0,   0,         0,         0,   0],
                [Koxy[2,0],         0,   0,         0, Koxy[2,1],   0, Koxy[2,2],         0,   0,         0, Koxy[2,3],   0],
                [        0, Koyz[2,0],   0, Koyz[2,1],         0,   0,         0, Koyz[2,2],   0, Koyz[2,3],         0,   0],
                [        0,         0,   0,         0,         0,   0,         0,         0,   0,         0,         0,   0],
                [        0, Koyz[3,0],   0, Koyz[3,1],         0,   0,         0, Koyz[3,2],   0, Koyz[3,3],         0,   0],
                [Koxy[3,0],         0,   0,         0, Koxy[3,1],   0, Koxy[3,2],         0,   0,         0, Koxy[3,3],   0],
                [        0,         0,   0,         0,         0,   0,         0,         0,   0,         0,         0,   0]
            ])
            # fmt: on

        if self._crack_model == self.flex:
            K = self._crack_model(ap)

        return K

    def gasch(self, ap):
        """Stiffness matrix of the shaft element with crack in rotating coordinates
        according to the breathing model of Gasch.

        Paramenters
        -----------
        ap : float
            Angular position of the shaft.

        Returns
        -------
        K : np.ndarray
            Stiffness matrix of the cracked element.
        """

        # Gasch
        ko = self.Ko[0, 0]
        kcx = self.Kc[0, 0]
        kcz = self.Kc[1, 1]

        kme = (ko + kcx) / 2
        kmn = (ko + kcz) / 2
        kde = (ko - kcx) / 2
        kdn = (ko - kcz) / 2

        size = 18
        cosine_sum = np.sum(
            [(-1) ** i * np.cos((2 * i + 1) * ap) / (2 * i + 1) for i in range(size)]
        )

        ke = kme + (4 / np.pi) * kde * cosine_sum
        kn = kmn + (4 / np.pi) * kdn * cosine_sum

        T_matrix = np.array(
            [
                [np.cos(ap), np.sin(ap)],
                [-np.sin(ap), np.cos(ap)],
            ]
        )

        K = T_matrix.T @ np.array([[ke, 0], [0, kn]]) @ T_matrix

        return K

    def mayes(self, ap):
        """Stiffness matrix of the shaft element with crack in rotating coordinates
        according to the breathing model of Mayes.

        Paramenters
        -----------
        ap : float
            Angular position of the shaft.

        Returns
        -------
        K : np.ndarray
            Stiffness matrix of the cracked element.
        """

        # Mayes
        ko = self.Ko[0, 0]
        kcx = self.Kc[0, 0]
        kcz = self.Kc[1, 1]

        ke = 0.5 * (ko + kcx) + 0.5 * (ko - kcx) * np.cos(ap)
        kn = 0.5 * (ko + kcz) + 0.5 * (ko - kcz) * np.cos(ap)

        T_matrix = np.array(
            [
                [np.cos(ap), np.sin(ap)],
                [-np.sin(ap), np.cos(ap)],
            ]
        )

        K = T_matrix.T @ np.array([[ke, 0], [0, kn]]) @ T_matrix

        return K
    
    def flex(self, ap):
        """Stiffness matrix of the shaft element with crack in rotating coordinates
        according to the open model of Flex.

        Paramenters
        -----------
        ap : float
            Angular position of the shaft.

        Returns
        -------
        K : np.ndarray
            Stiffness matrix of the cracked element.
        """


        # Flex
        G_s = self.shaft_elem.material.G_s
        Poisson = self.shaft_elem.material.Poisson
        E = self.shaft_elem.material.E
        radius = self.shaft_elem.odl / 2
        depth_ratio = self.depth_ratio
        Lce = (
            -33.3333 * depth_ratio ** 5
            + 77.5641 * depth_ratio ** 4
            - 49.5542 * depth_ratio ** 3
            + 9.0485 * depth_ratio ** 2
            + 1.2415 * depth_ratio
            - 0.0024
        ) * 2* radius 

        if self.crack_model == "Flex Open":

            mi = 2 * depth_ratio 
            gama = (mi * (2 - mi))**(1/2) 

            Ix = np.pi * (radius**4)/8 + (radius**4)/4 * ((1 - mi) * (2 * mi**2 - 4*mi + 1) * gama + np.arcsin(1 - mi))
            Iy = np.pi * (radius**4)/4 - (radius**4)/12 * ((1-mi) * (2*mi**2 - 4*mi - 3) * gama + 3*np.arcsin(gama))

            AT = radius**2 * (np.pi - np.arccos(1 - mi) + (1 - mi) * (mi * (2 - mi))**(1/2)) 
            ee = 2 * radius**3/(3*AT)*(mi*(2 - mi))**(3/2)

            Ixb = Ix - AT * ee**2
            Iyb = Iy

            kcc = (6 * (1 + Poisson)) / (7 + 6 * Poisson)
            IXX = (Ixb + Iyb) / 2 + (Ixb - Iyb) / 2 * np.cos(2 * ap)
            IYY = (Ixb + Iyb) / 2 - (Ixb - Iyb) /2 * np.cos(2 * ap)
            IXY = -(Ixb - Iyb) / 2 * np.sin(2 * ap)       

        if self.crack_model == "Flex Breathing":

            cross_discret = 20

            JJ = (np.pi / 4) * radius**4

            # Discretização da área trincada 
            step = radius / cross_discret

            rotation_indexes = [3, 4, 5, 9, 10, 11] 
            rot_resp = self.disp_resp[rotation_indexes]
            
            Tx1 = rot_resp[0]
            Ty1 = rot_resp[1]
            Tx2 = rot_resp[2]
            Ty2 = rot_resp[3]

            # Momento gerado no elemento trincado 
            Mdx = E * JJ * (Tx2 - Tx1) / Lce
            Mpx = 0
            Mdy = E * JJ * (Ty2 - Ty1) / Lce
            Mpy = 0

            MTx = Mdx + Mpx
            MTy = Mdy + Mpy

            # Discretization of the cracked cross-section e identificação dos pontos que pertencem à área trincada
            CCi          = np.zeros((2 * cross_discret + 1, 2 * cross_discret + 1), dtype=complex)
            Ci           = np.zeros_like(CCi, dtype=complex)
            CC           = np.zeros_like(CCi, dtype=complex)
            C            = np.zeros_like(CCi, dtype=complex)
            pointsCRACKi = np.zeros_like(CCi, dtype=complex)
            Ai           = np.zeros((2 * cross_discret, 2 * cross_discret), dtype=complex)
            A            = np.zeros_like(Ai, dtype=complex)
            Rareai       = np.zeros_like(Ai, dtype=complex)
            Rarea        = np.zeros_like(Ai, dtype=complex)
            XYi          = np.zeros_like(Ai, dtype=complex)
            IXX          = 1j*0
            IXX          = JJ
            IYY          = 1j*0
            IYY          = JJ
            IXY          = 1j*0
            XY           = np.zeros_like(Ai, dtype=complex)

            # Determinação de CCi
            x = np.linspace(-radius, radius, (2*cross_discret)+1)
            y = np.linspace(-radius, radius, (2*cross_discret)+1)
            X, Y = np.meshgrid(x, y)
            Z = X + 1j * Y  # pontos complexos

            absZ = np.abs(Z)
            angleZ = np.angle(Z)
            CCi = absZ * np.exp(1j * angleZ)

            mask_CCi = absZ <= radius
            Ci = np.where(mask_CCi, CCi, 0)
            mask_depth = np.imag(CCi) > (radius - depth_ratio * radius)
            # mask_depth = np.imag(CCi) > (radius - depth_ratio)
            pointsCRACKi = np.where(mask_depth, CCi, 500000)

            # Rotation of the cracked area de acordo com a nova posição angular do rotor
            pointsCRACK = np.absolute(pointsCRACKi) * np.exp(1j * (np.angle(pointsCRACKi) + ap))

            # Determinação de CC e C
            absZ = np.abs(Z)
            angleZ = np.angle(Z)
            CC = absZ * np.exp(1j * (angleZ + ap))
            C = CC.copy()

            # Identificação de campo de tensões, área e momentos de inércia do elemento trincado
            stepAV = 1
            si = 0
            erroTHETA = 1000
            angANTERIOR = 1j*0
            ang = 1j*0
            K = 1j * np.zeros((2*6, 2*6))

            while erroTHETA > 1e-5:
                Tensao = (-(MTx * IYY + MTy * IXY) / (IXX * IYY - IXY**2)) * np.imag(XY) + ((MTy * IXX + MTx * IXY) / (IXX * IYY - IXY**2)) * np.real(XY)

                original_shape = (2 * cross_discret, 2 * cross_discret)

                if np.isscalar(Tensao):
                    Tensao = np.full(original_shape, Tensao)
                
                AT = 1j*0

                step_half = step / 2
                exp_step_angle = np.exp(1j * np.angle(step_half + step_half * 1j))
                abs_step_angle = np.absolute(step_half + step_half * 1j)

                n, m = C.shape  #  (2*cross_discret + 1, 2*cross_discret + 1)
                n1, m1 = n - 1, m - 1  # limites

                # Submatrizes deslocadas para o cálculo de A
                C_center = C[:n1, :m1]
                C_right  = C[:n1, 1:]
                C_down   = C[1:, :m1]

                # Cálculo preliminar de A
                A_pre = np.abs((C_center - C_right) * (C_center - C_down))
                Rarea = C_center + abs_step_angle * exp_step_angle
                Rareai = Rarea.copy()
                A = A_pre.copy()

                # Condições
                mask_radius = np.abs(Rareai) >= radius
                A[mask_radius] = 0
                Rarea[mask_radius] = 0
                mask_crack = (np.abs(pointsCRACK[:n1, :m1] - C_center) < 1e-8) & (np.abs(Tensao[:n1, :m1]) >= 0)
                A[mask_crack] = 0

                AT = np.sum(A)
                    
                # Determinação do CG
                Cgg = 1j*0

                for bb in range(A.shape[0]):
                    for aa in range(A.shape[1]):
                        Cgg = Cgg + Rarea[bb,aa]*A[bb,aa]

                CG = Cgg / AT
                CG = sum(Rarea.flatten() * A.flatten()) / AT

                IXX = 1j*0
                IYY = 1j*0
                IXY = 1j*0
                XY = 0*XY

                n, m = C.shape
                n1, m1 = n - 1, m - 1

                # Submatrizes deslocadas
                C0 = C[:n1, :m1]          # C[bb, aa]
                C1 = C[:n1, 1:]           # C[bb, aa + 1]
                C2 = C[1:, :m1]           # C[bb + 1, aa]
                A_ = A[:n1, :m1]
                Rarea_ = Rarea[:n1, :m1]
                Rareai_ = Rareai[:n1, :m1]
                pointsCRACK_ = pointsCRACK[:n1, :m1]
                Tensao_ = Tensao[:n1, :m1]

                # Inicializa XY
                XY = np.zeros_like(Rarea_, dtype=complex)
                XY_mask = np.abs(Rareai_) < radius
                XY[XY_mask] = Rarea_[XY_mask] - CG

                dx = C0 - C1
                dy = C0 - C2
                dx3 = np.abs(dx)**3
                dy3 = np.abs(dy)**3

                Ixx = (np.abs(dx * dy3) / 12.0) + A_ * np.abs(np.imag(XY))**2
                Iyy = (np.abs(dx3 * dy) / 12.0) + A_ * np.abs(np.real(XY))**2
                Ixy = A_ * np.imag(XY) * np.real(XY)

                # Condições
                mask_crack = (np.abs(pointsCRACK_ - C0) < 1e-8) & (np.abs(Tensao_) > 0)
                mask_out = np.abs(Rareai_) >= radius
                Ixx[mask_out | mask_crack] = 0
                Iyy[mask_out | mask_crack] = 0
                Ixy[mask_out | mask_crack] = 0

                XY[mask_out] = 0  

                IXX = np.sum(Ixx)
                IYY = np.sum(Iyy)
                IXY = np.sum(Ixy)

                ang = np.arctan(-IXY / ((IXX - IYY)) / 2) / 2 


                if stepAV >= 2:
                    erroTHETA = np.absolute(np.absolute(angANTERIOR) - np.absolute(ang))

                XY1 = XY
                si += 1

                if si == 50:
                    break

                angANTERIOR = ang
                stepAV += 1

        mmm = 0
        kcc = (6 * (1 + Poisson) * (1+mmm**2)**2) / ((7 + 6 * Poisson) * (1+mmm**2)**2 + (20 + 12 * Poisson)*mmm**2)

        fiiY = (12 * E * IXX / (G_s * kcc * AT * Lce**2))
        fiiX = (12 * E * IYY / (G_s * kcc * AT * Lce**2))
        fiiXY = (12 * E * IYY / (G_s * kcc * AT * Lce**2)) 

        aF = 12 * IYY * E / ((1 + fiiX) * Lce**3)
        bF = 12 * IXX * E / ((1 + fiiY) * Lce**3)
        cF = 6 * IYY * E / ((1 + fiiX) * Lce**2)
        dF = 6 * IXX * E / ((1 + fiiY) * Lce**2)
        eF = (4 + fiiX) * IYY * E / ((1 + fiiX) * Lce)
        fF = (2 - fiiX) * IYY * E / ((1 + fiiX) * Lce)
        gF = (2 - fiiY) * IXX * E / ((1 + fiiY) * Lce)
        hF = (4 + fiiY) * IXX * E / ((1 + fiiY) * Lce)
        pF = 12 * IXY * E / ((1 + fiiXY) * Lce**3)
        qF = 6 * IXY * E / ((1 + fiiXY) * Lce**2)
        rF = (4 + fiiXY) * IXY * E / ((1 + fiiXY) * Lce)
        sF = (2 - fiiXY) * IXY * E / ((1 + fiiXY) * Lce)      
        tF = 0
        uF = 0       
        wF = 0
        kF = 0
        mF = 0
        nF = 0
        iF = 0
        jF = 0

        K = np.array([    
            [ bF, -pF,  kF,	-qF, -dF,  nF, -bF,	 pF, -kF, -qF, -dF,	-mF],
            [-pF,  aF,	wF,	 cF,  qF,  mF,	pF,	-aF, -wF,  cF,  qF, -mF],
            [ kF,  tF,	wF,	 iF,   0,  jF, -kF,	-tF, -wF, -iF,   0,	-jF],
            [-qF,  cF,	iF,	 eF,  rF,   0,	qF,	-cF, -iF,  fF,  sF,	  0],
            [-dF,  qF,	jF,	 rF,  hF,   0,	dF,	-qF, -jF,  sF,	gF,	  0],
            [ nF,   0,	mF,	  0,  uF,   0, -nF,	  0, -mF,	0, -uF,	  0],
            [-bF,  pF, -kF,	 qF,  dF, -nF,	bF,	-pF,  kF,  qF,  dF,	 nF],
            [ pF, -aF, -wF,	-cF, -qF, -mF, -pF,	 aF,  wF, -cF, -qF,	 mF],
            [-kF, -tF, -wF,	-iF,   0, -jF,	 0,	 tF,  wF,  mF,   0,	 nF], 
            [-qF,  cF, -iF,	 fF,  sF,   0,	qF,	-cF,  iF,  eF,	rF,   0],
            [-dF,  qF, -jF,  sF,  gF,   0,	dF,	-qF,   0,  rF,	hF,	  0],
            [-mF,	0, -mF,   0, -uF,   0,	 0,	  0,  mF,	0,	uF,	  0] 
                ])
            
        return K
        
    def _get_force_in_time(self, step, disp_resp, ang_pos):
        """Calculate the dynamic force related on given time step.

        Paramenters
        -----------
        step : int
            Current time step index.
        disp_resp : np.ndarray
            Displacement response of the system at the current time step.
        ang_pos : float
            Angular position of the shaft at the current time step.

        Returns
        -------
        F : np.ndarray
            Force matrix related to the open crack in the current time step `t[step]`.
        """

        self.disp_resp = disp_resp

        K_crack = self.compute_crack_stiffness(ang_pos)

        F = np.zeros(self.rotor.ndof)
        F[self.dofs] = (self.K_elem - K_crack) @ disp_resp[self.dofs]
        self.forces[:, step] = F

        return F

    def run(self, node, unb_magnitude, unb_phase, speed, t, **kwargs):
        """Run analysis for the system with crack given an unbalance force.

        System time response is simulated considering weight force.

        Parameters
        ----------
        node : list, int
            Node where the unbalance is applied.
        unb_magnitude : list, float
            Unbalance magnitude (kg.m).
        unb_phase : list, float
            Unbalance phase (rad).
        speed : float or array_like, pint.Quantity
            Rotor speed.
        t : array
            Time array.
        **kwargs : optional
            Additional keyword arguments can be passed to define the parameters
            of the Newmark method if it is used (e.g. gamma, beta, tol, ...).
            See `ross.utils.newmark` for more details.
            Other keyword arguments can also be passed to be used in numerical
            integration (e.g. num_modes).
            See `Rotor.integrate_system` for more details.

        Returns
        -------
        results : ross.TimeResponseResults
            For more information on attributes and methods available see:
            :py:class:`ross.TimeResponseResults`
        """

        rotor = self.rotor

        # Unbalance force
        F, ang_pos, _, _ = rotor._unbalance_force_over_time(
            node, unb_magnitude, unb_phase, speed, t
        )

        # Weight force
        g = np.zeros(rotor.ndof)
        g[1::6] = -9.81
        M = rotor.M()

        for i in range(len(t)):
            F[:, i] += M @ g

        self.forces = np.zeros((rotor.ndof, len(t)))

        force_crack = lambda step, **state: self._get_force_in_time(
            step, state.get("disp_resp"), ang_pos[step]
        )

        results = rotor.run_time_response(
            speed=speed,
            F=F.T,
            t=t,
            method="newmark",
            add_to_RHS=force_crack,
            **kwargs,
        )

        return results


def crack_example():
    """Create an example to evaluate the influence of transverse cracks in a
    rotating shaft.

    This function returns time response results of a transversal crack fault.
    The purpose is to make available a simple example so that a doctest can be
    written using it.

    Returns
    -------
    results : ross.TimeResponseResults
        Results for a shaft with crack.

    Examples
    --------
    >>> from ross.faults.crack import crack_example
    >>> from ross.probe import Probe
    >>> results = crack_example()
    Running direct method
    >>> probe1 = Probe(14, 0)
    >>> probe2 = Probe(22, 0)
    >>> fig = results.plot_1d([probe1, probe2])
    """

    rotor = rs.rotor_example_with_damping()

    n1 = rotor.disk_elements[0].n
    n2 = rotor.disk_elements[1].n

    results = rotor.run_crack(
        n=18,
        depth_ratio=0.2,
        node=[n1, n2],
        unbalance_magnitude=[5e-4, 0],
        unbalance_phase=[-np.pi / 2, 0],
        crack_model="Mayes",
        speed=Q_(1200, "RPM"),
        t=np.arange(0, 0.5, 0.0001),
    )

    return results