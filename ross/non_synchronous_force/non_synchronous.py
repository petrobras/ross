# Class to aplicate the non synchronous force in the rotor

# import ross as rs
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def non_synchronous_force(
    F, m, a, k, omega, s, time_non_sync_params, n_frequencies_plot=None
):

    if type(omega) == int:

        omega_ns = np.array([omega]).reshape((1, 1))

    elif type(omega) == list:

        omega_ns = np.array(omega).reshape((1, len(omega)))

    else:

        omega_ns = omega.reshape((1, len(omega)))

    # Variables:

    mx = 14.29

    mz = 14.29

    ax = 2.871

    az = -2.871

    kx = 1.195 * (10**6)

    kz = 1.195 * (10**6)

    # Equations:

    time_assync = np.arange(
        time_non_sync_params[0],
        time_non_sync_params[1] + time_non_sync_params[2],
        time_non_sync_params[2],
    )

    Amplitude_x = []

    Amplitude_z = []

    qx = []

    qz = []

    if n_frequencies_plot:

        if type(n_frequencies_plot) == int:

            n_frequencies_plot = np.array([n_frequencies_plot]).reshape((1, 1))

        elif type(n_frequencies_plot) == list:

            n_frequencies_plot = np.array(n_frequencies_plot).reshape(
                (1, len(n_frequencies_plot))
            )

        count = len(n_frequencies_plot[0])

        for ii in range(len(n_frequencies_plot[0])):

            exist = np.argwhere(omega_ns[0] == n_frequencies_plot[0, ii])

            if len(exist) != 0:

                count = count - 1

        idx_1 = 0

        aux_omega = np.zeros((1, len(omega_ns[0]) + count))

        index = []

        index.append(np.argwhere(omega_ns[0] < n_frequencies_plot[0, 0]))

        if len(index) != 0:

            aux_omega[0, idx_1 : len(index[0])] = omega_ns[0, index[0]]

            aux_omega[0, len(index[0])] = n_frequencies_plot[0, 0]

            idx_1 = len(index[0]) + 1

        for ii in range(1, len(n_frequencies_plot[0])):

            index.append(np.argwhere(omega_ns[0] < n_frequencies_plot[0, ii]))

            if len(index[ii]) != 0:

                idx_2 = np.argwhere(index[ii] != index[ii - 1])

                aux_omega[0, idx_1 : idx_1 + len(idx_2[:, 0])] = omega_ns[
                    0, index[ii][idx_2[:, 0]].T
                ]

                aux_omega[0, idx_1 + len(idx_2[:, 0])] = n_frequencies_plot[0, ii]

                idx_1 = idx_1 + len(idx_2[:, 0]) + 1

        index.append(np.argwhere(omega_ns[0] > n_frequencies_plot[0, -1]))

        if len(index[-1]) != 0:

            aux_omega[0, idx_1:] = omega_ns[0, index[-1]]

        omega_ns = aux_omega

    for ii in range(len(omega_ns[0])):

        Amplitude_x.append(
            (
                F
                * (
                    mz * s**2 * omega_ns[0, ii] ** 2
                    + ax * s * omega_ns[0, ii] ** 2
                    - kz
                )
            )
            / (
                ax * az * s**2 * omega_ns[0, ii] ** 4
                - kx * kz
                + kx * mz * s**2 * omega_ns[0, ii] ** 2
                + kz * mx * s**2 * omega_ns[0, ii] ** 2
                - mx * mz * s**4 * omega_ns[0, ii] ** 4
            )
        )

        Amplitude_z.append(
            (
                F
                * (
                    mx * s**2 * omega_ns[0, ii] ** 2
                    + az * s * omega_ns[0, ii] ** 2
                    - kx
                )
            )
            / (
                ax * az * s**2 * omega_ns[0, ii] ** 4
                - kx * kz
                + kx * mz * s**2 * omega_ns[0, ii] ** 2
                + kz * mx * s**2 * omega_ns[0, ii] ** 2
                - mx * mz * s**4 * omega_ns[0, ii] ** 4
            )
        )

        aux_x = []

        aux_z = []

        for jj in time_assync:

            aux_x.append(Amplitude_x[-1] * np.sin(s * omega_ns[0, ii] * jj))

            aux_z.append(Amplitude_z[-1] * np.cos(s * omega_ns[0, ii] * jj))

        qx.append(aux_x)

        qz.append(aux_z)

    # Plotting:

    # Amplitudes and Orbit:

    if len(n_frequencies_plot) != 0:

        for ii in range(len(n_frequencies_plot[0])):

            idx = int(np.argwhere(omega_ns[0, :] == n_frequencies_plot[0, ii]))

            title_string_amp_x = (
                "Amplitude on X axis for non synchronous force in "
                + str(n_frequencies_plot[0, ii])
                + " rpm"
            )

            title_string_amp_z = (
                "Amplitude on Z axis for non synchronous force in "
                + str(n_frequencies_plot[0, ii])
                + " rpm"
            )

            title_string_orbit = (
                "Orbits for non synchronous force in "
                + str(n_frequencies_plot[0, ii])
                + " rpm"
            )

            fig = make_subplots(
                rows=3,
                cols=1,
                subplot_titles=(
                    title_string_amp_x,
                    title_string_amp_z,
                    title_string_orbit,
                ),
            )

            fig.add_trace(
                go.Scatter(x=time_assync, y=np.array(qx[idx][:])), row=1, col=1
            )

            fig.add_trace(
                go.Scatter(x=time_assync, y=np.array(qz[idx][:])), row=2, col=1
            )

            fig.add_trace(go.Scatter(x=qx[idx], y=qz[idx]), row=3, col=1)

            # Update xaxis properties
            fig.update_xaxes(title_text="time [s]", row=1, col=1)
            fig.update_xaxes(title_text="time [s]", row=2, col=1)
            fig.update_xaxes(title_text="Amplitude in x axis [m]", row=3, col=1)

            # Update yaxis properties
            fig.update_yaxes(title_text="Amplitude in x axis [m]", row=1, col=1)
            fig.update_yaxes(title_text="Amplitude in z axis [m]", row=2, col=1)
            fig.update_yaxes(title_text="Amplitude in z axis [m]", row=3, col=1)

            fig.show()

            fig = []

    return Amplitude_x, Amplitude_z, qx, qz


if __name__ == "__main__":

    F = 1
    m = 1
    a = 1
    k = 1
    w = np.array([1, 10, 15, 20])  # w = 500 / w = [500, 1200] / w = np.arange(0,1501,1)
    s = 0.5
    time_assync_params = [0, 180, 0.1]
    n_frequencies_plot = [5, 19]

    non_synchronous_force(F, m, a, k, w, s, time_assync_params, n_frequencies_plot)
