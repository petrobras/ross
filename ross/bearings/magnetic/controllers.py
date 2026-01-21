import control as ct
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

s = ct.TransferFunction.s


def pid(k_p, k_i, k_d, n_f=10000):
    return k_p + k_i / s + k_d * s * (1 / (1 + (1 / n_f) * s))


def lqg(A, B, C, Q_lqr, R_lqr, Q_kalman, R_kalman):
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    C = np.array(C, dtype=float)

    n = A.shape[0]

    K, _, _ = ct.lqr(A, B, Q_lqr, R_lqr)

    G = np.eye(n)
    L, _, _ = ct.lqe(A, G, C, Q_kalman, R_kalman)

    A_c = A - B @ K - L @ C
    B_c = L
    C_c = -K
    D_c = np.zeros((K.shape[0], C.shape[0]))

    sys_ss = ct.ss(A_c, B_c, C_c, D_c)
    return ct.ss2tf(sys_ss)


def lead_lag(tau, alpha, k=1.0):
    num = [k * tau, k]
    den = [alpha * tau, 1.0]

    return ct.tf(num, den)


def second_order(b2, b1, b0, a1, a0):
    num = [b2, b1, b0]
    den = [1.0, a1, a0]
    return ct.tf(num, den)


def low_pass_filter(w_c, k=1.0):
    num = [k * w_c]
    den = [1.0, w_c]
    return ct.tf(num, den)


def notch_filter(w_n, zeta_z, zeta_p, k=1.0):
    num = [1.0, 2.0 * zeta_z * w_n, w_n**2]
    den = [1.0, 2.0 * zeta_p * w_n, w_n**2]
    return k * ct.tf(num, den)


def combine(*args):
    result_controller = 1
    for tf in args:
        result_controller *= tf

    return result_controller


def plot_frequency_response(*systems, **kwargs):
    w_min = kwargs.get("w_min", 1e-2)
    w_max = kwargs.get("w_max", 1e3)
    n_points = kwargs.get("n_points", 1000)
    title = kwargs.get("title", "Frequency Response")
    legends: None | list = kwargs.get("legends", None)

    tableau_colors = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#7f7f7f",  # gray
        "#bcbd22",  # olive
        "#17becf",  # cyan
    ]

    if legends is None:
        legends = [f"System {i+1}" for i in range(len(systems))]
    elif len(legends) != len(systems):
        raise ValueError("O número de legendas deve ser igual ao número de sistemas.")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)

    w = np.logspace(np.log10(w_min), np.log10(w_max), n_points)

    for idx, (system, legend) in enumerate(zip(systems, legends)):
        mag, phase, _ = ct.frequency_response(system, w)
        mag_db = 20 * np.log10(mag)
        phase_deg = phase * 180 / np.pi

        color = tableau_colors[idx % len(tableau_colors)]

        fig.add_trace(
            go.Scatter(
                x=w,
                y=mag_db,
                mode="lines",
                name=legend,
                line=dict(color=color),
                legendgroup=legend,
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=w,
                y=phase_deg,
                mode="lines",
                name=legend,
                line=dict(color=color),
                legendgroup=legend,
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        fig.update_xaxes(type="log", row=1, col=1)
        fig.update_xaxes(type="log", title="Frequency (rad/s)", row=2, col=1)

        fig.update_yaxes(title="Magnitude (dB)", row=1, col=1)
        fig.update_yaxes(title="Phase (°)", row=2, col=1)

        fig.update_layout(title=title)

    fig.show()
