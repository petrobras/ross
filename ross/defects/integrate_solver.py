"""Time integration module.

This module defines the time integration routines to evaluate time responses from
the rotors. These simulate the transient evolution of the dynamic behavior of the
rotors defined by the user, and return the time signals for rotor displacement.

"""

import numpy as np


class Integrator:

    """A series of Runge-Kutta time integration algorithms.

    Calculates the time response for the rotors input to the routine.

    Parameters
    ----------
    x0 : float
        Initial time
    y0 : float
        Initial condition for the integration
    x : float
        Iteration time
    h : float
        Step height
    func : object
        Function to be integrated in time

    Returns
    -------
    The rotor transient response.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Runge–Kutta_methods ..

    """

    def __init__(self, x0, y0, x, h, func, print_progress=False):

        self.x0 = x0
        self.y0 = y0
        self.x = x
        self.h = h
        self.func = func
        self.print_progress = print_progress

    def rk4(self):
        # Runge-Kutta 4th order (RK4)

        # Count number of iterations using step size or
        # step height h
        n = int((self.x - self.x0) / self.h)

        # Iterate for number of iterations
        y = self.y0
        result = np.zeros((24, n + 1))
        result[:, 0] = self.y0

        # 4th-order Runge-Kutta

        for i in range(1, n + 1):
            if i % 10000 == 0 and self.print_progress:
                print(f"Iteration: {i} \n Time: {self.x0}")

            "Apply Runge Kutta Formulas to find next value of y"
            k1 = self.h * self.func(self.x0, y, i)
            k2 = self.h * self.func(self.x0 + 0.5 * self.h, y + 0.5 * k1, i)
            k3 = self.h * self.func(self.x0 + 0.5 * self.h, y + 0.5 * k2, i)
            k4 = self.h * self.func(self.x0 + self.h, y + k3, i)

            # Update next value of y
            y = y + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            result[:, i] = np.copy(y)

            # Update next value of x
            self.x0 = self.x0 + self.h

        return result

    def rk45(self):
        # Runge-Kutta-Fehlberg (RKF45)

        # Count number of iterations using step size or
        # step height h
        n = int((self.x - self.x0) / self.h)

        # Iterate for number of iterations
        y = self.y0
        result = np.zeros((24, n + 1))
        result[:, 0] = self.y0

        for i in range(1, n + 1):
            if i % 10000 == 0 and self.print_progress:
                print(f"Iteration: {i} \n Time: {self.x0}")

            "Apply Runge Kutta Formulas to find next value of y"
            k1 = 1 * self.func(self.x0, y, i)
            yp2 = y + k1 * (self.h / 5)
            k2 = 1 * self.func(self.x0 + (self.h / 5), yp2, i)
            yp3 = y + k1 * (3 * self.h / 40) + k2 * (9 * self.h / 40)
            k3 = 1 * self.func(self.x0 + (3 * self.h / 10), yp3, i)
            yp4 = (
                y
                + k1 * (3 * self.h / 10)
                - k2 * (9 * self.h / 10)
                + k3 * (6 * self.h / 5)
            )
            k4 = 1 * self.func(self.x0 + (3 * self.h / 5), yp4, i)
            yp5 = (
                y
                - k1 * (11 * self.h / 54)
                + k2 * (5 * self.h / 2)
                - k3 * (70 * self.h / 27)
                + k4 * (35 * self.h / 27)
            )
            k5 = 1 * self.func(self.x0 + self.h, yp5, i)
            yp6 = (
                y
                + k1 * (1631 * self.h / 55296)
                + k2 * (175 * self.h / 512)
                + k3 * (575 * self.h / 13824)
                + k4 * (44275 * self.h / 110592)
                + k5 * (253 * self.h / 4096)
            )
            k6 = 1 * self.func(self.x0 + (7 * self.h / 8), yp6, i)

            # Update next value of y
            y = y + self.h * (
                37 * k1 / 378 + 250 * k3 / 621 + 125 * k4 / 594 + 512 * k6 / 1771
            )
            result[:, i] = np.copy(y)

            # Update next value of x
            self.x0 = self.x0 + self.h

        return result
