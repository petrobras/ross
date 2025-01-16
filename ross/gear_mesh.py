import numpy as np
import scipy as sp
import pandas as pd
import plotly.graph_objects as go
import logging
import traceback

class Gear:
    """ 
    A Gear element for evaluating gear teeth stiffness parameters.

    All physical parameters are expected in the S.I and angulars in radians. 

    Parameters
    ---------
    module:
        Module of the gear.
    n_tooth:
        Number of teeth.
    pressure_angle:
        The pressure angle of the gear.
    width:
        The width of the gear.
    hub_bore_radius:
        Hub bore radius of the gear.
    young:
        Young modulus of the gear material.
    poisson:
        The poisson value of the gear material.
    disc_points:
        Discretization points for plotting the gear geometry teeth.
    addendum_coeff:
        Addendum coefficient of the gear, it is normalized by the module.
    clearance_coeff:
        Clearance coefficient of the gear, it is normalized by the module.
    
    Example:
    -------
    >>> pinion = Gear(8 / np.pi * 25.4e-3, 19, 20 * np.pi/180, 16e-3,15e-3/2, 200e9, 0.3, 500, 1, 0.25)
    """

    def __init__(self,  module: int | float, n_tooth: int | float, pressure_angle: int | float, width: int | float, hub_bore_radius: int |float,
        young: int | float, poisson: int | float, disc_points: int, addendum_coeff: int | float=1., clearance_coeff: int | float=0.25, ):
        
        try:
            if module <= 0 or n_tooth <= 0 or addendum_coeff <= 0 or clearance_coeff < 0 or disc_points <= 0: 
                raise ValueError(f"Gear parameters must be positive.")
            
            self.m = module # [m]
            self.N = n_tooth # [1]
            self.alpha = pressure_angle # [rad]
            self.L = width # [m]
            self.r_shaft = hub_bore_radius # [m]
            self.E = young # [Pa]
            self.v = poisson # [1] 
            self.ha_ = addendum_coeff # [1]
            self.c_ = clearance_coeff # [1]
            
            self.G: float = self.E/2/(1+self.v)

            self._gear_geometry_constants()
            self.plot_tooth_geometry()

            self.stiffness = self.gear_stiffness()

        except (TypeError, ValueError) as e:  # Catch specific errors
            error = f'{traceback.format_exc()}'
            logging.error(error)
        except Exception as e:  # Catch-all for other exceptions
            error = f'{traceback.format_exc()}'
            logging.critical(f"Unexpected error: {error}")

    def _gear_geometry_constants(self) -> None:
        """
        Initialize geometry constants derived from Gear parameters.
        """
        self.r_b: float = 1 / 2 * self.m * self.N * np.cos(self.alpha) # radii of base circle [m] MAOK
        self.r_p: float = self.r_b / np.cos(self.alpha)                # radii of pitch circle [m] MAOK
        self.r_a: float = self.r_p + self.ha_ * self.m                 # radii of addendum circle [m] 
        self.r_c: float = np.sqrt( np.square(self.r_b * np.tan(self.alpha)  - self.ha_ * self.m /  np.sin(self.alpha) ) + np.square(self.r_b) ) # radii of the involute starting point [m] MAOK
        self.r_f: float = 1 / 2 * self.m * self.N - (self.ha_ + self.c_) * self.m   # radii of root circle [m] MAOK
        self.r_rho: float = self.c_ * self.m / (1 - np.sin(self.alpha) )            # radii of cutter tip round corner [m] MAOK
        self.r_rho_: float = self.r_rho / self.m

        self.alpha_a: float = np.arccos(self.r_b / self.r_a) # pressure angle when the contact point is on the addendum circle [rad] MAOK
        self.alpha_c: float = np.arccos(self.r_b / self.r_c) # pressure angle when the contact point is on the C point [rad] MAOK
        self.theta_f: float = 1 / self.N * ( np.pi / 2 + 2 * np.tan(self.alpha) * (self.ha_ - self.r_rho_) + 2 * self.r_rho_ / np.cos(self.alpha)) # The angle between the tooth center-line and de junction with the root circle [radians]

        self.theta_b: float = np.pi / (2*self.N) + self.involute(self.alpha)

        self.a1: float = (self.ha_ + self.c_) * self.m - self.r_rho # MAOK
        self.b1: float = np.pi * self.m / 4 + self.ha_ * self.m * np.tan(self.alpha) + self.r_rho * np.cos(self.alpha) # MAOK
        
        self.tau_c: float = self.to_tau(self.alpha_c)
        self.tau_a: float = self.to_tau(self.alpha_a)


    @staticmethod
    def involute(angle: float) -> float:
        """Involute function

        Calculates the involute function for a given angle.

        The involute function is used to describe the contact region of the gear profile.

        :math:`inv(angle) = tan(angle) - angle`
        
        This functions computes the value of the involute for an input angle in radians.

        Parameters
        ----------
        angle : float
            Input angle in radians.

        Returns 
        ---------
        float
            The inv(angle)
        
        Notes
        --------
        - Ensure that the angle is in radians before passing it to this function.

        Examples
        --------
        >>> Gear.involute(20 / 180 * np.pi)
        0.014904383867336446

        >>> Gear.involute(15 / 180 * np.pi)
        0.006149804631973288
        """

        return np.tan(angle) - angle
    
    def transition_curve(self, gamma) -> tuple[float, float, float, float]:
        """
        Transition curve

        Compute the geometric transition curve profile.

        Based on:
        Ma, H., Song, R., Pang, X., & Wen, B. (2014). Time-varying mesh stiffness calculation of cracked spur gears. 
        Engineering Failure Analysis, 44, 179–194. https://doi.org/10.1016/j.engfailanal.2014.05.018

        Parameters
        ----------
        gamma : float
            Input angles in radians, within [alpha, pi/2].

        Returns 
        -------
        tuple
            -`x_1` : float
                x-coordinates of the transition curve.
            -`y_1`: float
                y-coordinates of the transition curve.
            -`A_y1` : float
                Area values based on x_1.
            -`I_y1` : float
                Moment of inertia values based on x_1.

        Examples
        --------

        >>> self.transition_curve(45 * np.pi / 180)
        (0.002281501426336625, 0.05272239401873227, 9.1260057053465e-05, 1.5834376622229213e-10)

        """
        if not (self.alpha <= gamma <= np.pi/2):
            raise ValueError(f"alpha must be within the range [{self.alpha}, {np.pi/2}]. Current alpha: {gamma}")
        
        phi = lambda gamma: (self.a1/np.tan(gamma) + self.b1)/self.r_p
        phi = phi(gamma)

        x_1 = lambda gamma: self.r_p * np.sin( phi ) - ( self.a1 / np.sin(gamma) + self.r_rho ) * np.cos(gamma - phi)
        x_1 = x_1(gamma)
        
        y_1 = lambda gamma: self.r_p * np.cos(phi) - (self.a1/np.sin(gamma) + self.r_rho) * np.sin(gamma-phi)
        y_1 = y_1(gamma)

        A_y1 = 2 * x_1 * self.L
        I_y1 = 2/3*np.power(x_1,3)*self.L

        return x_1, y_1, A_y1, I_y1
    
    def involute_curve(self, tau) -> tuple[np.ndarray[float]]:
        """
        Compute the geometric involute curve profile.
            
        Parameters
        ----------
        tau : float
            Input angles in radians, within [###CORRECT HERE###].

        Returns 
        -------
            tuple
                -`x_2` : float
                    x-coordinates of the involute curve.
                -`y_2` : float
                    y-coordinates of the involute curve.
                -`A_y2` : float
                    Area values based on x_2.
                -`I_y2` : float
                    Moment of inertia values based on x_2.
        
        Example
        --------
        >>> self.involute_curve(0.3405551128775112)
        (0.0014447795342141163, 0.055344111158185265, 5.7791181368564653e-05, 4.02108709529994e-11)

        References
        ---------
        From Ma, H., Song, R., Pang, X., & Wen, B. (2014). Time-varying mesh stiffness calculation of cracked spur gears. 
        Engineering Failure Analysis, 44, 179–194. https://doi.org/10.1016/j.engfailanal.2014.05.018

        """

        x_2 = self.r_b * ( (tau + self.theta_b) * np.cos(tau) - np.sin(tau) )
        y_2 = self.r_b * ( (tau + self.theta_b) * np.sin(tau) + np.cos(tau) )
        A_y2 = 2 * x_2 * self.L
        I_y2 = 2/3 * np.power(x_2, 3) * self.L

        return x_2, y_2, A_y2, I_y2

    def to_tau(self, alpha_i: float) -> float:
        """
        Transforms the alpha angle, used to build the involute profile, into the integration variable tau.

        :math:`tau(alpha_i) = alpha_i - self.theta_b + self.involute(alpha_i)`
        
        Parameters
        ----------
        alpha_i : float
            An angle within the involute profile.

        Returns 
        ---------
        float
            tau_i
    
        Examples
        ---------
        >>> self.to_tau(31 * np.pi / 180)
        0.5573963019457713

        References
        --------
        Ma, H., Pang, X., Song, R., & Yang, J. (2014). 基于改进能量法的直齿轮时变啮合刚度计算 
        [Time-varying mesh stiffness calculation of spur gears based on improved energy method].
        Journal of Northeastern University (Natural Science), 35(6), 863–867. https://doi.org/10.3969/j.issn.1005-3026.2014.06.023
        """
        return alpha_i - self.theta_b + self.involute(alpha_i)

    def plot_tooth_geometry(self) -> None:
        """
        Plot the geometry of the tooth profile.
        """

        # Generate the transition geometry
        transition = np.linspace(np.pi/2, self.alpha, 200)
        transition_vectorized = np.vectorize(self.transition_curve)
        x_1, y_1, _, _ = transition_vectorized(transition)

        # Generate the involute geometry
        involute = np.linspace(self.alpha_c, self.alpha_a, 200)
        tau_vectorize = np.vectorize(self.to_tau)
        tau = tau_vectorize(involute)

        involute_vectorize = np.vectorize(self.involute_curve)
        x_2, y_2, _, _ = involute_vectorize(tau)

        # Create the plot
        fig = go.Figure()

        # Add the transition curve
        fig.add_trace(go.Scatter(x=y_1[-1::-1], y=x_1[-1::-1], mode='lines', name='Transition Curve'))

        # Add the involute curve
        fig.add_trace(go.Scatter(x=y_2, y=x_2, mode='lines', name='Involute Curve'))

        # Update layout with gridlines
        fig.update_layout(
            title='Tooth Profile Geometry',
            xaxis_title='Y-axis [m]',
            yaxis_title='X-axis [m]',
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True),
        )

        # Show the plot
        fig.show()

    def gear_stiffness(self):
        """
        Returns
        -------
        GearStiffness object declared with the self Gear object.
        """
        return GearStiffness(self)

class GearStiffness:
    """
    A class which evaluates the stiffness of a single gear.

    Parameters:
    --------
    gear: Gear object

    """

    def __init__(self, gear: Gear):
        self.gear = gear 

    def diff_tau(self, tau_i) -> float:
        """
        Method for evaluating the stiffness commonly found in the integrative functions of the involute region on a specified angle tau_i.

        Parameters
        ----------
        tau_i : float
            Operational tau_i angle.

        Returns
        ----------
        float
            The value of diff_tau(tau_i)
        """
        return self.gear.r_b * (tau_i + self.gear.theta_b) * np.cos(tau_i)

    def diff_gamma(self, gamma) -> float:
        """
        Method used in evaluating the stiffness commonly found in the integrative functions of the transition region.

        Parameters
        ----------
        gamma : float
            Value of the gamma angle used to describre the profile.

        Returns
        ----------
        float
            The value of diff_gamma(gamma)
        """
        gear = self.gear
        a1 = gear.a1
        b1 = gear.b1
        r_p = gear.r_p
        r_rho = gear.r_rho

        term_1 = ( a1 * np.sin( (a1/np.tan(gamma) + b1) / r_p ) * (1 + np.square( np.tan(gamma) ) )) / np.square( np.tan(gamma) )
        term_2 =  a1 * np.cos(gamma) / np.square(np.sin(gamma)) * np.sin(gamma - (a1/np.tan(gamma) + b1)/r_p)
        term_3 =   - (a1 / np.sin(gamma) + r_rho) * np.cos(gamma - (a1/np.tan(gamma) + b1)/r_p) * (1 + a1 * (1 + np.square(np.tan(gamma)))/r_p/np.square(np.tan(gamma)))

        return term_1 + term_2 + term_3


    def compute_stiffness(self, tau_op) -> tuple[float, float, float, float]:
        """
        Computes the stiffness in the direction of the applied force on the gear, according to the involute profile. 
        It evaluates each of them separetly, and those values are returned as 1/stiffness for an approach in accordance with Ma, H. et. al. (2014).
        
        Parameters
        ----------
        tau_op : float
            The tau operational angle, e.g. the angle formed by the normal of the contact involute curves and the x axis. 
        
        Returns
        ----------
        tuple of float
            A tuple containing the computed stiffness components as 1/stiffness values. The elements represent:
            - ka : float
                Stiffness related to axial stresses.
            - kb : float
                Stiffness related to bending stresses.
            - kf : float 
                Stiffness related to body of the gear.
            - ks : float
                Stiffness related to shear stresses.
        """
        
        kf = self._kf(tau_op)
        ka = self._ka(tau_op)
        kb = self._kb(tau_op)
        ks = self._ks(tau_op)

        if (np.isnan(kf)):
            return ka, kb, ks
        
        return  ka, kb, kf, ks

    def _gear_body_polynominal(self) -> pd.DataFrame | str:
        """
        This method uses the approach described by Sainsot et al. (2004) to calculate the stiffness factor (kf) 
        contributing to tooth deflections. If the parameters fall outside the experimental range used to derive 
        the analytical formula, the method returns `'oo'` to indicate an infinite stiffness approximation.

        Returns
        ---------
        float
            The calculated stiffness factor (kf) for the gear base. 
        
        str
            Return 'oo' if it doesn't match the criteria for the experimental range where this method was built.
        """

        gear = self.gear
        h = gear.r_f / gear.r_shaft

        poly = pd.DataFrame()
        poly['var'] = ['L'          , 'M',            'P'           , 'Q']
        poly['A_i'] = [-5.574e-5    , 60.111e-5     , -50.952e-5    , -6.2042e-5]
        poly['B_i'] = [-1.9986e-3   , 28.100e-3     , 185.50e-3     , 9.0889e-3]
        poly['C_i'] = [-2.3015e-4   , -83.431e-4    , 0.0538e-4     , -4.0964e-4]
        poly['D_i'] = [4.7702e-3    , -9.9256e-3    , 53.300e-3     , 7.8297e-3]
        poly['E_i'] = [0.0271       ,  0.1624       , 0.2895        , -0.1472]
        poly['F_i'] = [6.8045       , 0.9086        , 0.9236        , 0.6904]

        calculate_x_i = lambda row: (
            row['A_i'] / (gear.theta_f ** 2)
            + row['B_i'] * h ** 2
            + row['C_i'] * h / gear.theta_f
            + row['D_i'] / gear.theta_f
            + row['E_i'] * h
            + row['F_i']
        ) # OK

        poly['X_i'] = poly.apply(lambda row: calculate_x_i(row), axis=1)
        
        limits = {
            'L': (6.82, 6.94),
            'M': (1.08, 3.29),
            'P': (2.56, 13.47),
            'Q': (0.141, 0.62)
        }

        for index, row in poly.iterrows():
            var_name = row['var']
            X_i = row['X_i']
            lower_limit, upper_limit = limits[var_name]

            # if (not lower_limit <= X_i <= upper_limit) :#or (not 1.4 <= h <= 7) or (not 0.01 <= gear.theta_f <= 0.12):
            #     # for the stiffness on the base of the tooth to match the model, it has to match those criteria above. If not, kf -> oo.
            #     global contador
            #     contador+=1 
            #     return 'oo'


        return poly.loc[:,['var','X_i']]
    
    def _kf(self, tau_op) -> float:
        """
        Sainsot, P., Velex, P., & Duverger, O. (2004). Contribution of gear body to tooth deflections - A new bidimensional analytical 
        formula. Journal of Mechanical Design, 126(4), 748–752. https://doi.org/10.1115/1.1758252

        Calculate the stiffness contribution from the gear base, given a point on the involute curve.

        Parameters
        ---------
        tau_op (float): 
            The operational pressure angle (tau) in radians. This angle is used to determine the stiffness characteristics 
            based on the gear geometry and material properties.

        Returns
        ---------
        float: 
            The calculated 1/kf for the gear base.
    
        """
        gear = self.gear

        # obtain a dataframe of polynomials coefficients
        poly = self._gear_body_polynominal()

        # Extrapolating the range of interpolation described by Sainsot et. al. (2014).
        # if type(poly) == str:
        #     return 0

        L_poly, M_poly, P_poly, Q_poly =  poly['X_i']

        _, y, _, _ = gear.involute_curve(tau_op)

        Sf = 2 * gear.r_f * gear.theta_f
        u = y - gear.r_f

        kf = (
            ( np.cos(tau_op)**2 / (gear.E * gear.L) )
            * ( 
                L_poly * ( u /Sf)**2 
                + M_poly * u / Sf 
                + P_poly * (1 + Q_poly * np.tan(tau_op)**2 )
            ) 
        ) # OK, conferi com a fórmula (só n sei se o beta é o mesmo ...)

        return kf

    def _ks(self, tau_op) -> float:
        """
        Calculate the stiffness contribution from the gear resistance from shear stresses, given the tau operational angle.

        Parameters
        ----------
        tau_op
            Operational tau angle.
        
        Returns
        ---------
        float
            The shear stiffness in the form of 1/ks.
        """        
        
        gear = self.gear
        f_transiction = lambda gamma: (
            1.2*np.cos(tau_op)**2 
            / (gear.G * gear.transition_curve(gamma)[2]) 
            * self.diff_gamma(gamma)
        ) # OK

        k_transiction, error1 = sp.integrate.quad(f_transiction, np.pi/2, gear.alpha) # OK

        f_involute = lambda tau: (
            1.2 * (
                np.cos(tau_op)**2
                / (gear.G*gear.involute_curve(tau)[2])
                * self.diff_tau(tau)
            )
        ) # OK

        k_involute, error2 = sp.integrate.quad(f_involute, gear.to_tau(gear.alpha_c), tau_op) # OK

        return (k_transiction + k_involute)

    def _kb(self, tau_op) -> float:
        """
        Calculate the stiffness contribution from the gear resistance from bending stresses, given the tau operational angle.

        Parameters
        ----------
        tau_op
            Operational tau angle.
        
        Returns
        ---------
        float
            The bending stiffness in the form of 1/kb.
        """        
             
        gear = self.gear

        f_transiction = lambda gamma: (
            (
                np.cos(tau_op) 
                * (gear.involute_curve(tau_op)[1] - gear.transition_curve(gamma)[1])
                - gear.involute_curve(tau_op)[0] * np.sin(tau_op)
            )**2 
            / (gear.E * gear.transition_curve(gamma)[3])
            * self.diff_gamma(gamma)
        ) # OK

        k_transiction, error1 = sp.integrate.quad(f_transiction, np.pi/2, gear.alpha)

        f_involute = lambda tau: (
            (
                np.cos(tau_op) 
                * (gear.involute_curve(tau_op)[1] - gear.involute_curve(tau)[1])
                - gear.involute_curve(tau_op)[0] * np.sin(tau_op)
            )**2
            / (gear.E * gear.involute_curve(tau)[3])
            * self.diff_tau(tau)
        ) # OK

        k_involute, error2 = sp.integrate.quad(f_involute, gear.to_tau(gear.alpha_c), tau_op)

        return (k_transiction + k_involute)

    def _ka(self, tau_op: float) -> float:
        """
        Calculate the stiffness contribution from the gear resistance from axial stresses, given the tau operational angle.

        Parameters
        ----------
        tau_op
            Operational tau angle.
        
        Returns
        ---------
        float
            The axial stiffness in the form of 1/ka.
        """        
        gear = self.gear

        f_transiction = lambda gamma: (
            np.sin(tau_op)**2
            / (gear.E * gear.transition_curve(gamma)[2])
            * self.diff_gamma(gamma)
        ) # OK
        k_transiction, error1 = sp.integrate.quad(f_transiction, np.pi/2, gear.alpha)

        f_involute = lambda tau: (
            np.sin(tau_op)**2 
            / (gear.E * gear.involute_curve(tau)[2])
            * self.diff_tau(tau)
        ) # OK

        k_involute, error2 = sp.integrate.quad(f_involute, gear.to_tau(gear.alpha_c), tau_op)

        return (k_transiction + k_involute)

    @staticmethod
    def kh(gear1: Gear, gear2: Gear) -> float:
        """
        Evaluates the contact hertzian stiffness considering that both elasticity modulus are equal.

        Parameters
        ----------
        gear1
            Gear object.
        
        gear2
            Gear object.

        Returns
        ----------
        float
            The hertzian contact stiffness.

        Notes
        ----------
        - It returns kh, not 1/kh.
        """

        return np.pi * gear1.E * gear1.L / 4 / (1 - gear1.v**2)

class Mesh:
    """
    Represents the meshing behavior between two gears, typically a pinion and a crown gear, 
    including stiffness and contact ratio calculations.

    Parameters:
    -----------
    pinion : Gear
        The pinion gear object used in the gear pair (driver).
    gear : Gear
        The crown gear object used in the gear pair (driven).
    pinion_w : float
        The rotational speed [rad/sec] of the pinion gear in radians per second.
        
    Attributes:
    -----------
    gear : Gear
        The gear wheel object, which contains information about the geometry and properties of the wheel gear.
    pinion : Gear
        The pinion gear object, which contains information about the geometry and properties of the pinion gear.
    tm : float
        The meshing period, calculated based on the rotational speed and the number of teeth of the pinion.
    kh : float
        Hertzian stiffness of 2 tooth in contact (same Elasticity Modulus).
    k_mesh : float
        The equivalent stiffness of the gear mesh, combining the stiffness of the pinion and crown.
    cr : float
        The contact ratio, representing the average number of tooth in contact during meshing.

    """

    def __init__(self, pinion: Gear, gear: Gear, pinion_w: float):
        self.pinion = pinion
        self.gear = gear

        eta = gear.N/pinion.N # Gear ratio 

        self.pinion_w = pinion_w # pinion speed [rad/sec] 
        self.gear_w = - pinion_w / eta # gear speed [rad/sec]
        self.tm = 2 * np.pi / (self.pinion_w * self.pinion.N) # Gearmesh period [seconds/engagement]
        self.kh = GearStiffness.kh(pinion, gear)
        self.cr = self.contact_ratio(self.pinion, self.gear)
        self.ctm = self.cr * self.tm # [seconds/tooth] how much time each tooth remains in contact

    @staticmethod
    def contact_ratio(pinion: Gear, gear: Gear) -> float:
        """
        Parameters:
        ---------
        pinion : Gear
            The pinion object.
        gear : Gear
            The wheel object.
        
        Returns
        -------
        CR : float
            The contact ratio of the gear pair.

        Example
        -------
        >>> Mesh.contact_ratio(pinion, Gear)
        1.7939883590132295

        Reference
        ----------
        Understanding the contact ratio for spur gears with some comments on ways to read a textbook.
        Retrieved from: https://www.myweb.ttu.edu/amosedal/articles.html
        """

        pb = np.pi * 2 * pinion.r_b / pinion.N # base pitch
        C = pinion.r_p + gear.r_p # center distance (not the operating one)

        lc = ( # length of contact (not the operating one) # OK
            np.sqrt(pinion.r_a**2 - pinion.r_b**2) 
            + np.sqrt(gear.r_a**2 - gear.r_b**2) 
            - C * np.sin(pinion.alpha)
        )

        CR = lc / pb # contact ratio
    
        return CR


    def time_equivalent_stiffness(self, pinion: Gear, gear: Gear, t: float) -> float:
        """
        Parameters
        ---------
        pinion : Gear
            The pinion object.
        gear: Gear
            The gear object.
        t: float
            The time of meshing [0, self.tm]
        
        Returns
        -------
        float
            The equivalent stiffness of the contact based on the time [0, self.tm] of mesh.
        
        Example
        --------
        >>> self.time_equivalent_stiffness()
        167970095.70859054
        """

        # Angular displacements 
        alpha_pinion = t * self.pinion_w + pinion.alpha_c # angle variation of the pinion [rad]
        alpha_gear   = t * self.gear_w   + gear.alpha_a   # angle variation of the gear   [rad]
        
        # Tau displacementes
        d_tau_pinion = self.pinion.to_tau(alpha_pinion) # angle variation of the pinion in tau [rad]
        d_tau_gear   = self.gear.to_tau(alpha_gear)     # angle variation of the pinion in tau [rad]

        # Contact stiffness according to tau angles
        ka_1, kb_1, kf_1, ks_1 = pinion.stiffness.compute_stiffness(d_tau_pinion)
        ka_2, kb_2, kf_2, ks_2 = gear.stiffness.compute_stiffness(d_tau_gear)

        # Evaluating the equivalate meshing stiffness. 
        k_t = 1 / (1/self.kh + ka_1 + kb_1 + kf_1 + ks_1 + ka_2 + kb_2 + kf_2 + ks_2)

        return k_t, d_tau_pinion, d_tau_gear
    
    def mesh(self, pinion, gear, t):
        """
        Calculate the time-varying meshing stiffness of a gear pair.

        This method computes the equivalent stiffness of a gear mesh at a given time `t`, taking into
        account the periodic nature of the meshing process and the contact ratio (`cr`) of the gear pair.
        The computation considers whether one or two pairs of teeth are in contact during the meshing cycle.

        Parameters
        ----------
        pinion : Gear
            The pinion gear object containing properties needed for stiffness calculation.
        gear : Gear
            The gear object containing properties needed for stiffness calculation.
        t : float
            Time instant for which the meshing stiffness is calculated.

        Returns
        -------
        tuple
            - float: The total equivalent meshing stiffness at time `t`.
            - float: The meshing stiffness of the first tooth pair in contact (returns `np.nan` if not applicable).
            - float: The meshing stiffness of the second tooth pair in contact.

        Notes
        -----
        - The calculation considers the periodic nature of meshing and the contact ratio (cr) of the gear pair.
        - The stiffness contribution varies depending on whether one or two pairs of teeth are in contact.
        """

        tm = self.tm
        t = t - t // tm * tm
        
        if t <= (self.cr-1) * tm:
            k_mesh1, d_tau_pinion1, d_tau_gear1 = self.time_equivalent_stiffness(pinion, gear, t)
            k_mesh0,  d_tau_pinion0, d_tau_gear0 = self.time_equivalent_stiffness(pinion, gear, self.tm + t)

            return k_mesh0 + k_mesh1, k_mesh0, k_mesh1
        
        elif t > (self.cr-1) * tm:
            k_mesh1, d_tau_pinion1, d_tau_gear1 = self.time_equivalent_stiffness(pinion, gear, t)

            return k_mesh1, np.nan, k_mesh1       

def example() -> None:
    pinion = Gear(
        module=2e-3, 
        n_tooth=55, 
        pressure_angle=20 * np.pi/180, 
        width=20e-3, 
        hub_bore_radius=17.5e-3, 
        young=212e9, 
        poisson=0.29, 
        disc_points=1e3, 
        addendum_coeff=1, 
        clearance_coeff=0.25
    )  # MA 3
    
    crown = Gear(
        module=2e-3, 
        n_tooth=75, 
        pressure_angle=20 * np.pi/180, 
        width=20e-3, 
        hub_bore_radius=17.5e-3, 
        young=212e9, 
        poisson=0.29, 
        disc_points=1e3, 
        addendum_coeff=1, 
        clearance_coeff=0.25
    )  # MA 3

    pinion_w = 11 * 2 * np.pi 
    meshing = Mesh(pinion, crown, pinion_w)

    n_tm = 1

    time_range = np.linspace(0, n_tm * meshing.tm, int(200))
    angle_range = time_range * pinion_w

    stiffness = np.zeros(np.shape(time_range))
    k0_stiffness = np.zeros(np.shape(time_range))
    k1_stiffness = np.zeros(np.shape(time_range))

    for i, time in enumerate(time_range):
        stiffness[i], k0_stiffness[i], k1_stiffness[i] = meshing.mesh(pinion, crown, time)

    # Calculate limits and yticks
    x_lim = n_tm * meshing.tm * pinion_w * 180 / np.pi
    yticks = np.arange(3.8e8, int(4.4e8), int(0.1e8))

    # Create figure
    fig = go.Figure()

    # Add the main plot lines
    fig.add_trace(go.Scatter(
        x=angle_range * 180 / np.pi,
        y=stiffness,
        mode='lines',
        line=dict(color='blue', width=1),
        name='Stiffness'
    ))

    fig.add_trace(go.Scatter(
        x=angle_range * 180 / np.pi,
        y=k1_stiffness,
        mode='lines',
        line=dict(color='red', dash='solid'),
        name='K1 Stiffness'
    ))

    fig.add_trace(go.Scatter(
        x=angle_range * 180 / np.pi,
        y=k0_stiffness,
        mode='lines',
        line=dict(color='black', dash='dot'),
        name='K0 Stiffness'
    ))

    # Update layout
    fig.update_layout(
        title='Stiffness x Angular Displacement',
        xaxis=dict(
            title='Angular Displacement [deg]',
            range=[0, x_lim],
        ),
        yaxis=dict(
            title='Stiffness [N/m]',
            autorange=True,
            tickformat=".1e",  # Use scientific notation for y-axis labels
        ),
        showlegend=True
    )

    fig.show()

if __name__ == '__main__':
    example()
