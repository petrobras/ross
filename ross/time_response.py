import numpy as np
from numpy import linalg as la
from scipy.integrate import cumulative_trapezoid as integrate
from numba import jitclass, float64, njit


spec = [
    ("speed", float64[:]),
    ("t", float64[:]),
    ("M", float64[:, :]),
    ("C", float64[:, :]),
    ("K", float64[:, :]),
    ("G", float64[:, :]),
    ("Ksdt", float64[:, :]),
    ("F", float64[:, :]),
    ("update_coefficients", bool),
]


@jitclass(spec)
class TimeResponse:
    def __init__(self, speed, t, M, C, K, G, Ksdt, F):
        self.speed = speed
        self.t = t
        self.M = M
        self.C = C
        self.K = K
        self.G = G
        self.Ksdt = Ksdt
        self.F = F
        self.force_functions = []

        self.accel = np.gradient(speed, t)
        self.displ = integrate(speed, t, initial=0)

        self.speed_is_variable = np.std(speed) == 0
        self.speed_mean = np.mean(speed)

    def add_force_function(self, force_functions):
        if isinstance(force_functions, list):
            self.force_functions.extend(force_functions)
        else:
            self.force_functions.append(force_functions)

    def _total_forces(self, step, **current_state):

        summed_forces = self.F[step]

        for force in self.force_functions:
            summed_forces += force(step, **current_state)

        return summed_forces

    def system_with_constant_speed(self, step, **current_state):
        return (
            self.M,
            self.C + self.G * self.speed_mean,
            self.K,
            self._total_forces(step, **current_state),
        )

    def system_with_variable_speed(self, step, **current_state):
        return (
            self.M,
            self.C + self.G * self.speed[step],
            self.K + self.Ksdt * self.accel[step],
            self._total_forces(step, **current_state),
        )

    # def system_with_variable_coefficients(self, step, **current_state):
    #     C = self.C(self.speed[step])
    #     K = self.K(self.speed[step])
    #     return (
    #         self.M,
    #         C + self.G * self.speed[step],
    #         K + self.Ksdt * self.accel[step],
    #         self._total_forces(step, **current_state),
    #     )

    def run(self, **kwargs):

        if self.speed_is_variable:
            system = self.system_with_variable_speed
        else:
            system = self.system_with_constant_speed

        size = self.F.shape[1]
        yout = newmark(system, self.t, size, **kwargs)

        return self.t, yout


@njit
def newmark(func, t, y_size, newmark_type="simple", **options):
    """Transient solution of the dynamic behavior of the system.

    Perform numerical integration using the Newmark method with Newton-Raphson
    iterations of the generic equation of motion:
    M * y'' + C * y' + K * y = RHS(t, y)

    Parameters
    ----------
    func : callable
        A function that calculates the system matrices and right-hand side (RHS) vector at each
        time step. It should take at least one argument `(step, dt=None, y=None, ydot=None, y2dot=None)`
        and return a tuple `(M, C, K, RHS)`, where `step` is a scalar int related to the current time
        step, `dt` is the current time step in seconds, `y` is a ndarray of current state of the system,
        `ydot` and `y2dot` are its first and second time derivatives. `M`, `C`, `K` are ndarrays with
        `np.shape(M) = (y_size, y_size)` and `RHS` is a ndarray with `len(RHS) = y_size`.
    t : array_like
        Time array.
    y_size : int
        Size of the state vector.
    **options
        Options passed for controlling the integration parameters. All options available are
        listed below.
    gamma : float, optional
        Parameter of the integration algorithm related to the velocity interpolation equation.
        Default is 0.5.
    beta : float, optional
        Parameter of the integration algorithm related to the displacement interpolation equation.
        Default is 0.25.
    tol : float, optional
        Convergence tolerance for the Newton-Raphson iterations. Default is 1e-6.
    progress_interval : float, optional
        Time interval at which progress is printed. Default is to not show progress.

    Returns
    -------
    yout : ndarray
        System response. It is an array containing the state variables at each time step of `t` with
        `np.shape(yout) = (len(t), y_size)`

    References
    ----------
    Newmark, N. M. (1959). A method of computation for structural dynamics.
    Journal of the Engineering Mechanics Division, 85(3), 67-94.

    Examples
    --------
    >>> import ross as rs
    >>> rotor = rs.rotor_example()
    >>> size = 10000
    >>> node = 3
    >>> speed = 500.0
    >>> accel = 0.0
    >>> t = np.linspace(0, 10, size)
    >>> F = np.zeros((size, rotor.ndof))
    >>> F[:, rotor.number_dof * node + 0] = 10 * np.cos(2 * t)
    >>> F[:, rotor.number_dof * node + 1] = 10 * np.sin(2 * t)
    >>> M = rotor.M(speed)
    >>> C1 = rotor.C(speed)
    >>> K1 = rotor.K(speed)
    >>> C2 = rotor.G()
    >>> K2 = rotor.Ksdt()
    >>> rotor_system = lambda i, **state: (M, C1 + C2 * speed, K1 + K2 * accel, F[i, :])
    >>> yout = newmark(rotor_system, t, rotor.ndof)
    >>> yout[:, rotor.number_dof * node + 1] # doctest: +ELLIPSIS
    array([0.00000000e+00, 8.49140057e-09, 4.34296767e-08, ...,
           1.16148468e-05, 1.16492353e-05, 1.16859622e-05])
    """

    gamma = options.get("gamma", 0.5)
    beta = options.get("beta", 0.25)
    epsilon = options.get("epsilon", 1e-8)
    tol = options.get("tol", 1e-6)
    progress_interval = options.get("progress_interval", t[-1] + 1)
    args = options.get("args", [])

    n_steps = len(t)
    ny = y_size

    if newmark_type == "robust":
        print("Using robust integration")
        yout = _converge_robust_newmark(
            func, args, ny, n_steps, t, progress_interval, gamma, beta, epsilon, tol
        )
    else:
        print("Using simple integration")
        yout = _converge_simple_newmark(
            func, args, ny, n_steps, t, progress_interval, gamma, beta, tol
        )

    return yout


@njit
def _converge_simple_newmark(
    func, args, ny, n_steps, t, progress_interval, gamma, beta, tol
):
    y0 = np.zeros(ny)
    ydot0 = np.zeros(ny)
    y2dot0 = np.zeros(ny)

    yout = np.zeros((n_steps, ny))
    yout[0, :] = y0

    for step in range(1, n_steps):
        aux = round(t[step] / progress_interval, 9)
        if aux - int(aux) == 0:
            print("Time: ", t[step], " seconds")

        dt = t[step] - t[step - 1]

        M, C, K, RHS = func(
            step,
            time_step=dt,
            disp_resp=y0,
            velc_resp=ydot0,
            accl_resp=y2dot0,
            args=args,
        )

        y2dot = np.zeros(ny)
        ydot = ydot0 + y2dot0 * (1.0 - gamma) * dt
        y = y0 + ydot0 * dt + y2dot0 * (0.5 - beta) * (dt**2)

        res = RHS - (M @ y2dot + C @ ydot + K @ y)
        nr_iter = 0

        while la.norm(res) >= tol:
            nr_iter += 1
            if nr_iter > 50:
                raise RuntimeError(
                    """Newton-Raphson algorithm diverged. Maximum number of iterations reached. 
                    Try decreasing the time step or using robust integration."""
                )

            J = M + C * gamma * dt + K * beta * (dt**2)
            dy2dot = la.solve(J, res)

            y2dot += dy2dot
            ydot += dy2dot * gamma * dt
            y += dy2dot * beta * (dt**2)

            M, C, K, RHS = func(
                step,
                time_step=dt,
                disp_resp=y,
                velc_resp=ydot,
                accl_resp=y2dot,
                args=args,
            )
            res = RHS - (M @ y2dot + C @ ydot + K @ y)

        y0 = y
        ydot0 = ydot
        y2dot0 = y2dot

        yout[step, :] = y0

    return yout


@njit
def _converge_robust_newmark(
    func, args, ny, n_steps, t, progress_interval, gamma, beta, epsilon, tol
):
    y0 = np.zeros(ny)
    ydot0 = np.zeros(ny)
    y2dot0 = np.zeros(ny)

    yout = np.zeros((n_steps, ny))
    yout[0, :] = y0

    for step in range(1, n_steps):
        aux = round(t[step] / progress_interval, 9)
        if aux - int(aux) == 0:
            print("Time: ", t[step], " seconds")

        t_curr = t[step - 1]
        t_target = t[step]
        dt = t_target - t_curr

        dt_min = dt * 1e-4
        dt_max = dt

        while t_curr < t_target:
            y2dot = np.zeros(ny)
            ydot = ydot0 + y2dot0 * (1.0 - gamma) * dt
            y = y0 + ydot0 * dt + y2dot0 * (0.5 - beta) * (dt**2)

            M, C, K, RHS = func(
                step,
                time_step=dt,
                disp_resp=y,
                velc_resp=ydot,
                accl_resp=y2dot,
                args=args,
            )
            res = RHS - (M @ y2dot + C @ ydot + K @ y)
            nr_iter = 0

            active_dofs = np.where(RHS != 0)[0]

            while la.norm(res) >= tol:
                nr_iter += 1
                converged = True

                if nr_iter > 15:
                    converged = False
                    break

                J = M + C * gamma * dt + K * beta * (dt**2)

                # update jacobian with perturbation
                F_base = RHS

                for i in active_dofs:
                    y_i, ydot_i, y2dot_i = y[i], ydot[i], y2dot[i]

                    y2dot[i] += epsilon
                    ydot[i] += epsilon * gamma * dt
                    y[i] += epsilon * beta * (dt**2)

                    _, _, _, F_pert = func(
                        step,
                        time_step=dt,
                        disp_resp=y,
                        velc_resp=ydot,
                        accl_resp=y2dot,
                        args=args,
                    )

                    J[:, i] -= (F_pert - F_base) / epsilon

                    y[i], ydot[i], y2dot[i] = y_i, ydot_i, y2dot_i

                dy2dot = la.solve(J, res)

                y2dot += dy2dot
                ydot += dy2dot * gamma * dt
                y += dy2dot * beta * (dt**2)

                M, C, K, RHS = func(
                    step,
                    time_step=dt,
                    disp_resp=y,
                    velc_resp=ydot,
                    accl_resp=y2dot,
                    args=args,
                )
                res = RHS - (M @ y2dot + C @ ydot + K @ y)

            if converged:
                y0 = y
                ydot0 = ydot
                y2dot0 = y2dot
                t_curr += dt

                if nr_iter <= 5:
                    dt = min(dt * 2.0, dt_max)
                elif nr_iter >= 10:
                    dt = max(dt * 0.5, dt_min)
            else:
                dt *= 0.25
                if dt < dt_min:
                    raise RuntimeError(
                        f"Time step dropped below minimum threshold ({dt_min}) without convergence."
                    )
                if t_curr + dt > t_target:
                    dt = t_target - t_curr

        yout[step, :] = y0

    return yout
