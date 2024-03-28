import re

import numpy as np
import pandas as pd
from numpy import linalg as la
from plotly import graph_objects as go
from scipy.linalg import eigh


class DataNotFoundError(Exception):
    """
    An exception indicating that the data could not be found in the file.
    """

    pass


def read_table_file(file, element, sheet_name=0, n=0, sheet_type="Model"):
    """Instantiate one or more element objects using inputs from an Excel table.

    Parameters
    ----------
    file: str
        Path to the file containing the shaft parameters.
    element: str
        Specify the type of element to be instantiated: bearing, shaft or disk.
    sheet_name: int or str, optional
        Position of the sheet in the file (starting from 0) or its name. If none is
        passed, it is assumed to be the first sheet in the file.
    n: int
        Exclusive for bearing elements, since this parameter is given outside the table
        file.
    sheet_type: str
        Exclusive for shaft elements, as they have a Model table in which more
        information can be passed, such as the material parameters.

    Returns
    -------
    A dictionary of parameters.

    Examples
    --------
    >>> import os
    >>> file_path = os.path.dirname(os.path.realpath(__file__)) + '/tests/data/shaft_si.xls'
    >>> read_table_file(file_path, "shaft", sheet_type="Model", sheet_name="Model") # doctest: +ELLIPSIS
    {'L': [0.03...
    """
    df = pd.read_excel(file, header=None, sheet_name=sheet_name)

    # Assign specific values to variables
    parameter_columns = {}
    optional_parameter_columns = {}
    header_key_word = ""
    default_dictionary = {}
    parameters = {}
    if element == "bearing":
        header_key_word = "kxx"
        parameter_columns["kxx"] = ["kxx"]
        parameter_columns["cxx"] = ["cxx"]
        optional_parameter_columns["kyy"] = ["kyy"]
        optional_parameter_columns["kxy"] = ["kxy"]
        optional_parameter_columns["kyx"] = ["kyx"]
        optional_parameter_columns["cyy"] = ["cyy"]
        optional_parameter_columns["cxy"] = ["cxy"]
        optional_parameter_columns["cyx"] = ["cyx"]
        optional_parameter_columns["frequency"] = ["frequency", "speed"]
        default_dictionary["kyy"] = None
        default_dictionary["kxy"] = 0
        default_dictionary["kyx"] = 0
        default_dictionary["cyy"] = None
        default_dictionary["cxy"] = 0
        default_dictionary["cyx"] = 0
        default_dictionary["frequency"] = None
    elif element == "shaft":
        if sheet_type == "Model":
            header_key_word = "od_left"
        else:
            header_key_word = "material"
        parameter_columns["L"] = ["length"]
        parameter_columns["idl"] = ["i_d_l", "idl", "id_left"]
        parameter_columns["odl"] = ["o_d_l", "odl", "od_left"]
        parameter_columns["idr"] = ["i_d_r", "idr", "id_right"]
        parameter_columns["odr"] = ["o_d_r", "odr", "od_right"]
        parameter_columns["material"] = ["material", "matnum"]
        optional_parameter_columns["n"] = ["n", "elemnum"]
        optional_parameter_columns["axial_force"] = [
            "axial_force",
            "axial force",
            "axial",
        ]
        optional_parameter_columns["torque"] = ["torque"]
        optional_parameter_columns["shear_effects"] = ["shear_effects", "shear effects"]
        optional_parameter_columns["rotary_inertia"] = [
            "rotary_inertia",
            "rotary inertia",
        ]
        optional_parameter_columns["gyroscopic"] = ["gyroscopic"]
        optional_parameter_columns["shear_method_calc"] = [
            "shear_method_calc",
            "shear method calc",
        ]
        default_dictionary["n"] = None
        default_dictionary["axial_force"] = 0
        default_dictionary["torque"] = 0
        default_dictionary["shear_effects"] = True
        default_dictionary["rotary_inertia"] = True
        default_dictionary["gyroscopic"] = True
        default_dictionary["shear_method_calc"] = "cowper"
    elif element == "disk":
        header_key_word = "ip"
        parameter_columns["n"] = ["unnamed: 0", "n"]
        parameter_columns["m"] = ["m", "mass"]
        parameter_columns["Id"] = ["it", "id"]
        parameter_columns["Ip"] = ["ip"]

    # Find table header and define if conversion is needed
    header_index = -1
    header_found = False
    convert_to_metric = False
    convert_to_rad_per_sec = False
    for index, row in df.iterrows():
        for i in range(0, row.size):
            if isinstance(row[i], str):
                if not header_found:
                    if row[i].lower() == header_key_word:
                        header_index = index
                        header_found = True
                if (
                    "inches" in row[i].lower()
                    or "lbm" in row[i].lower()
                    or "lb" in row[i].lower()
                ):
                    convert_to_metric = True
                if "rpm" in row[i].lower():
                    convert_to_rad_per_sec = True
                if header_found and convert_to_metric and convert_to_rad_per_sec:
                    break
        if header_found and convert_to_metric:
            break
    if not header_found:
        raise ValueError(
            "Could not find the header. Make sure the table has a header "
            "containing the names of the columns. In the case of a " + element + ", "
            "there should be a column named " + header_key_word + "."
        )

    # Get specific data from the file
    new_materials = {}
    if element == "shaft" and sheet_type == "Model":
        material_header_index = -1
        material_header_found = False
        material_header_key_word = "matno"
        for index, row in df.iterrows():
            for i in range(0, row.size):
                if isinstance(row[i], str):
                    if row[i].lower() == material_header_key_word:
                        material_header_index = index
                        material_header_found = True
                        break
            if material_header_found:
                break
        if not material_header_found:
            raise ValueError(
                "Could not find the header for the materials. Make sure the table has a header "
                "with the parameters for the materials that will be used. There should be a column "
                "named " + material_header_key_word + "."
            )
        df_material = pd.read_excel(
            file, header=material_header_index, sheet_name=sheet_name
        )
        material_name = []
        material_rho = []
        material_e = []
        material_g_s = []
        material_color = []
        for index, row in df_material.iterrows():
            if not pd.isna(row["matno"]):
                material_name.append(int(row["matno"]))
                material_rho.append(row["rhoa"])
                material_e.append(row["ea"])
                material_g_s.append(row["ga"])
                if "color" in df_material.columns:
                    if pd.isna(row["color"]):
                        material_color.append("#525252")
                    else:
                        material_color.append(row["color"])
                else:
                    material_color.append("#525252")

            else:
                break
        if convert_to_metric:
            for i in range(0, len(material_name)):
                material_rho[i] = material_rho[i] * 27679.904
                material_e[i] = material_e[i] * 6894.757
                material_g_s[i] = material_g_s[i] * 6894.757
        new_materials["matno"] = material_name
        new_materials["rhoa"] = material_rho
        new_materials["ea"] = material_e
        new_materials["ga"] = material_g_s
        new_materials["color"] = material_color

    df = pd.read_excel(file, header=header_index, sheet_name=sheet_name)
    df.columns = df.columns.str.lower()

    # Find and isolate data rows
    first_data_row_found = False
    last_data_row_found = False
    indexes_to_drop = []
    for index, row in df.iterrows():
        if (
            not first_data_row_found
            and (
                isinstance(row[header_key_word], int)
                or isinstance(row[header_key_word], float)
            )
            and not pd.isna(row[header_key_word])
        ):
            first_data_row_found = True
        elif not first_data_row_found:
            indexes_to_drop.append(index)
        elif first_data_row_found and not last_data_row_found:
            if (
                isinstance(row[header_key_word], int)
                or isinstance(row[header_key_word], float)
            ) and not pd.isna(row[header_key_word]):
                continue
            else:
                last_data_row_found = True
                indexes_to_drop.append(index)
        elif last_data_row_found:
            indexes_to_drop.append(index)
    if not first_data_row_found:
        raise DataNotFoundError(
            "Could not find the data. Make sure you have at least one row containing "
            "data below the header."
        )
    if len(indexes_to_drop) > 0:
        df = df.drop(indexes_to_drop)

    # Build parameters list
    if element == "bearing":
        parameters["n"] = n
    for key, value in parameter_columns.items():
        for name in value:
            try:
                parameters[key] = df[name].tolist()
                break
            except KeyError:
                if name == value[-1]:
                    raise ValueError(
                        "Could not find a column with one of these names: " + str(value)
                    )
                continue
    for key, value in optional_parameter_columns.items():
        for name in value:
            try:
                parameters[key] = df[name].tolist()
                break
            except KeyError:
                if name == value[-1]:
                    parameters[key] = [default_dictionary[key]] * df.shape[0]
                else:
                    continue
    if element == "shaft":
        if sheet_type == "Model":
            new_material = parameters["material"]
            for i in range(0, df.shape[0]):
                new_material[i] = "shaft_mat_" + str(int(new_material[i]))
            parameters["material"] = new_material

    # change xltrc index to python index (0 based)
    if element in ("shaft", "disk"):
        new_n = parameters["n"]
        for i in range(0, df.shape[0]):
            new_n[i] -= 1
        parameters["n"] = new_n

    if convert_to_metric:
        for i in range(0, df.shape[0]):
            if element == "bearing":
                parameters["kxx"][i] = parameters["kxx"][i] * 175.126_836_986_4
                parameters["cxx"][i] = parameters["cxx"][i] * 175.126_836_986_4
                parameters["kyy"][i] = parameters["kyy"][i] * 175.126_836_986_4
                parameters["kxy"][i] = parameters["kxy"][i] * 175.126_836_986_4
                parameters["kyx"][i] = parameters["kyx"][i] * 175.126_836_986_4
                parameters["cyy"][i] = parameters["cyy"][i] * 175.126_836_986_4
                parameters["cxy"][i] = parameters["cxy"][i] * 175.126_836_986_4
                parameters["cyx"][i] = parameters["cyx"][i] * 175.126_836_986_4
            if element == "shaft":
                parameters["L"][i] = parameters["L"][i] * 0.0254
                parameters["idl"][i] = parameters["idl"][i] * 0.0254
                parameters["odl"][i] = parameters["odl"][i] * 0.0254
                parameters["idr"][i] = parameters["idr"][i] * 0.0254
                parameters["odr"][i] = parameters["odr"][i] * 0.0254
                parameters["axial_force"][i] = (
                    parameters["axial_force"][i] * 4.448_221_615_255
                )
            elif element == "disk":
                parameters["m"][i] = parameters["m"][i] * 0.453_592_37
                parameters["Id"][i] = parameters["Id"][i] * 0.000_292_639_7
                parameters["Ip"][i] = parameters["Ip"][i] * 0.000_292_639_7
    if convert_to_rad_per_sec:
        for i in range(0, df.shape[0]):
            if element == "bearing":
                parameters["frequency"][i] = (
                    parameters["frequency"][i] * 0.104_719_755_119_7
                )
    parameters.update(new_materials)
    return parameters


def visualize_matrix(rotor, matrix, frequency=None, **kwargs):
    """Visualize global matrix.

    This function gives some visualization of a given matrix, displaying
    values on a heatmap.

    Parameters
    ----------
    rotor: rs.Rotor
        The rotor object.
    matrix: str
        String for the desired matrix.
    frequency: float, optional
        Excitation frequency. Defaults to rotor speed.
    kwargs : optional
        Additional key word arguments can be passed to change the plot layout only
        (e.g. coloraixs=dict(colorscale="Rainbow"), width=1000, height=800, ...).
        *See Plotly Python Figure Reference for more information.

    Returns
    -------
    fig : Plotly graph_objects.Figure()
        The figure object with the plot.

    Examples
    --------
    >>> import ross as rs
    >>> rotor = rs.rotor_example()

    Visualizing Mass Matrix:
    >>> fig = rs.visualize_matrix(rotor, "M", frequency=100)

    Visualizing Stiffness Matrix:
    >>> fig = rs.visualize_matrix(rotor, "K", frequency=100)

    Visualizing Gyroscopic Matrix:
    >>> fig = rs.visualize_matrix(rotor, "G")
    """
    A = np.zeros((rotor.ndof, rotor.ndof))
    # E will store element's names and contributions to the global matrix
    E = np.zeros((rotor.ndof, rotor.ndof), dtype=object)

    M, N = E.shape
    for i in range(M):
        for j in range(N):
            E[i, j] = []

    for elm in rotor.elements:
        g_dofs = list(elm.dof_global_index.values())
        l_dofs = elm.dof_local_index()
        try:
            elm_matrix = getattr(elm, matrix)(frequency)
        except TypeError:
            elm_matrix = getattr(elm, matrix)()

        A[np.ix_(g_dofs, g_dofs)] += elm_matrix

        for l0, g0 in zip(l_dofs, g_dofs):
            for l1, g1 in zip(l_dofs, g_dofs):
                if elm_matrix[l0, l1] != 0:
                    E[g0, g1].append(
                        "<br>"
                        + elm.__class__.__name__
                        + f"(n={elm.n})"
                        + f": {elm_matrix[l0, l1]:.2E}"
                    )

    # list for dofs -> ['x0', 'y0', 'alpha0', 'beta0', 'x1'...]
    dof_list = [0 for i in range(rotor.ndof)]

    for elm in rotor.elements:
        for k, v in elm.dof_global_index.items():
            dof_list[v] = k

    data = {"row": [], "col": [], "value": [], "pos_value": [], "elements": []}
    for i, dof_row in enumerate(dof_list):
        for j, dof_col in enumerate(dof_list):
            data["row"].append(i)
            data["col"].append(j)
            data["value"].append(A[i, j])
            data["pos_value"].append(abs(A[i, j]))
            data["elements"].append(E[i, j])

    dof_string_list = []
    sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    for d in dof_list:
        d = d.replace("alpha", "α")
        d = d.replace("beta", "β")
        d = d.replace("_", "")
        d = d.translate(sub)
        dof_string_list.append(d)

    x_axis = dof_string_list
    y_axis = dof_string_list[::-1]
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=x_axis,
            y=y_axis,
            z=A[::-1],
            customdata=np.array(data["elements"], dtype=object).reshape(A.shape)[::-1],
            coloraxis="coloraxis",
            hovertemplate=(
                "<b>Value: %{z:.3e}<b><br>" + "<b>Elements:<b><br> %{customdata}"
            ),
            name="<b>Matrix {}</b>".format(matrix),
        )
    )

    fig.update_layout(
        coloraxis=dict(
            cmin=np.amin(A),
            cmax=np.amax(A),
            colorscale="Rainbow",
            colorbar=dict(
                title=dict(text="<b>Value</b>", side="top"), exponentformat="power"
            ),
        ),
        **kwargs,
    )

    return fig


def convert(name):
    """Converts a CamelCase str to a underscore_case str

    Parameters
    ----------
    file: str
        Path to the file containing the shaft parameters.

    Returns
    -------
    underscore_case string

    Examples
    --------
    >>> convert('CamelCase')
    'camel_case'
    """

    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def intersection(x1, y1, x2, y2):
    """
    Intersection code from https://github.com/sukhbinder/intersection
    MIT License

    Copyright (c) 2017 Sukhbinder Singh

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

    INTERSECTIONS Intersections of curves.
       Computes the (x,y) locations where two curves intersect.  The curves
       can be broken with NaNs or have vertical segments.

    Usage:
    x, y = intersection(x1, y1, x2, y2)

    Example:
    a, b = 1, 2
    phi = np.linspace(3, 10, 100)
    x1 = a*phi - b*np.sin(phi)
    y1 = a - b*np.cos(phi)

    x2=phi
    y2=np.sin(phi)+2
    x,y=intersection(x1,y1,x2,y2)

    plt.plot(x1,y1,c='r')
    plt.plot(x2,y2,c='g')
    plt.plot(x,y,'*k')
    plt.show()

    """

    def _rect_inter_inner(x1, x2):
        n1 = x1.shape[0] - 1
        n2 = x2.shape[0] - 1
        X1 = np.c_[x1[:-1], x1[1:]]
        X2 = np.c_[x2[:-1], x2[1:]]
        S1 = np.tile(X1.min(axis=1), (n2, 1)).T
        S2 = np.tile(X2.max(axis=1), (n1, 1))
        S3 = np.tile(X1.max(axis=1), (n2, 1)).T
        S4 = np.tile(X2.min(axis=1), (n1, 1))
        return S1, S2, S3, S4

    def _rectangle_intersection_(x1, y1, x2, y2):
        S1, S2, S3, S4 = _rect_inter_inner(x1, x2)
        S5, S6, S7, S8 = _rect_inter_inner(y1, y2)

        C1 = np.less_equal(S1, S2)
        C2 = np.greater_equal(S3, S4)
        C3 = np.less_equal(S5, S6)
        C4 = np.greater_equal(S7, S8)

        ii, jj = np.nonzero(C1 & C2 & C3 & C4)
        return ii, jj

    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)

    ii, jj = _rectangle_intersection_(x1, y1, x2, y2)
    n = len(ii)

    dxy1 = np.diff(np.c_[x1, y1], axis=0)
    dxy2 = np.diff(np.c_[x2, y2], axis=0)

    T = np.zeros((4, n))
    AA = np.zeros((4, 4, n))
    AA[0:2, 2, :] = -1
    AA[2:4, 3, :] = -1
    AA[0::2, 0, :] = dxy1[ii, :].T
    AA[1::2, 1, :] = dxy2[jj, :].T

    BB = np.zeros((4, n))
    BB[0, :] = -x1[ii].ravel()
    BB[1, :] = -x2[jj].ravel()
    BB[2, :] = -y1[ii].ravel()
    BB[3, :] = -y2[jj].ravel()

    for i in range(n):
        try:
            T[:, i] = np.linalg.solve(AA[:, :, i], BB[:, i])
        except:
            T[:, i] = np.Inf

    in_range = (T[0, :] >= 0) & (T[1, :] >= 0) & (T[0, :] <= 1) & (T[1, :] <= 1)

    xy0 = T[2:, in_range]
    xy0 = xy0.T
    return xy0[:, 0], xy0[:, 1]


def get_data_from_figure(fig):
    """Get data from a Plotly ccatter plot (XY) and convert to a pd.DataFrame.

    This function takes a go.Figure() object from Plotly and converts the data to a
    DataFrame from Pandas. It works only for Scatter Plots (XY) and does no support
    subplots or polar plots.

    This function is suitable to the following analyzes:
        - run_freq_response()
        - run_unbalance_response()
        - run_time_response()

    Parameters
    ----------
    fig : go.Figure
        A Plotly figure generated by go.Figure() command.

    Returns
    -------
    df : pd.DataFrame
        A DataFrame with the fig.data.

    Examples
    --------
    >>> import ross as rs
    >>> rotor = rs.rotor_example()

    Get post-processed data from an unbalance response
    >>> resp = rotor.run_unbalance_response(
    ...     3, 0.001, 0.0, np.linspace(0, 1000, 101)
    ... )
    >>> fig = resp.plot_magnitude(probe=[(3, 0.0, "probe1"), (3, np.pi/2, "probe2")])
    >>> df = rs.get_data_from_figure(fig)

    Use the probe tag to navigate through pandas data
    Index 0 for frequency array
    >>> df["probe1"][0] # doctest: +ELLIPSIS
    array([   0.,   10.,   20.,   30.,...

    Index 1 for amplitude array
    >>> df["probe1"][1] # doctest: +ELLIPSIS
    array([0.00000000e+00,...

    Or use "iloc" to obtain the desired array from pandas
    >>> df.iloc[1, 0] # doctest: +ELLIPSIS
    array([0.00000000e+00, 1.6057...
    """
    dict_data = {data["name"]: {} for data in fig.data}

    xaxis_label = fig.layout.xaxis.title.text
    yaxis_label = fig.layout.yaxis.title.text

    for i, data in enumerate(fig.data):
        dict_data[data["name"]][xaxis_label] = data.x
        dict_data[data["name"]][yaxis_label] = data.y

    df = pd.DataFrame(dict_data)

    return df


def newmark(fun, t, y_size, **options):
    """Transient solution of the dynamic behavior of the system.

    Perform numerical integration using the Newmark method with Newton-Raphson
    iterations of the generic equation of motion:
    M * y'' + C * y' + K * y = RHS(t, y)

    Parameters
    ----------
    fun : callable
        A function that calculates the system matrices and right-hand side (RHS) vector at each
        time step. It should take one argument `(step)` and return a tuple `(M, C, K, RHS)`,
        where `step` is a scalar int related to the current time step, `M`, `C`, `K` are ndarrays
        with `np.shape(M) = (y_size, y_size)` and `RHS` is a ndarray with `len(RHS) = y_size`.
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
        Time interval at which progress information is printed. Default is 1e6 seconds.

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
    >>> F[:, rotor.number_dof * node] = 10 * np.cos(2 * t)
    >>> F[:, rotor.number_dof * node + 1] = 10 * np.sin(2 * t)
    >>> M = rotor.M(speed)
    >>> C1 = rotor.C(speed)
    >>> K1 = rotor.K(speed)
    >>> C2 = rotor.G()
    >>> K2 = rotor.Ksdt()
    >>> rotor_system = lambda i, u, v: (M, C1 + C2 * speed, K1 + K2 * accel, F[i, :])
    >>> yout = newmark(rotor_system, t, rotor.ndof)
    >>> dof = 13
    >>> yout[:, dof] # doctest: +ELLIPSIS
    array([0.0000000e+00, 8.4914005e-09, 4.3429676e-08, ...
    """

    gamma = options.get("gamma", 0.5)
    beta = options.get("beta", 0.25)
    tol = options.get("tol", 1e-6)
    progress_interval = options.get("progress_interval", 1e6)

    n_steps = len(t)
    ny = y_size

    y0 = np.zeros(ny)
    ydot0 = np.zeros(ny)
    y2dot0 = np.zeros(ny)

    yout = np.full((n_steps, ny), 1e-38, dtype=np.float32)
    yout[0, :] = y0

    for step in range(1, n_steps):
        aux = round(t[step] / progress_interval, 9)
        if aux - int(aux) == 0:
            print(f"Time step: {t[step]:.6f} s")

        dt = t[step] - t[step - 1]

        M, C, K, RHS = fun(step, y0, ydot0)

        y2dot = np.zeros(ny)
        ydot = ydot0 + y2dot0 * (1 - gamma) * dt
        y = y0 + ydot0 * dt + y2dot0 * (0.5 - beta) * (dt**2)

        res = RHS - (K @ y + C @ ydot) - M @ y2dot
        nr_iter = 0

        while la.norm(res) >= tol:
            nr_iter += 1
            if nr_iter > 1e5:
                raise Warning(
                    "The Newton-Raphson algorithm is taking a long time to converge."
                )

            dy2dot = la.solve(M + C * gamma * dt + K * beta * (dt**2), res)

            y2dot += dy2dot
            ydot += dy2dot * gamma * dt
            y += dy2dot * beta * (dt**2)

            res = RHS - (K @ y + C @ ydot) - M @ y2dot

        y0 = y
        ydot0 = ydot
        y2dot0 = y2dot

        yout[step, :] = y

    return yout


def apply_pseudo_modal(rotor, speed, num_modes):
    """Pseudo-modal method.

    This method can be used to apply modal transformation to reduce model
    of the rotor system.

    Parameters
    ----------
    rotor: rs.Rotor
        The rotor object.
    speed : float
        Rotor speed.
    num_modes : int
        The number of eigenvectors to consider in the modal transformation
        with model reduction.

    Returns
    -------
    matrix_to_modal : callable
        Function to transform a square matrix from physical to modal space.
    vector_to_modal : callable
        Function to transform a vector from physical to modal space.
    vector_from_modal : callable
        Function to transform a vector from modal to physical space.

    Examples
    --------
    >>> import ross as rs
    >>> rotor = rs.rotor_example()
    >>> size = 10000
    >>> node = 3
    >>> speed = 500.0
    >>> t = np.linspace(0, 10, size)
    >>> F = np.zeros((size, rotor.ndof))
    >>> F[:, rotor.number_dof * node] = 10 * np.cos(2 * t)
    >>> F[:, rotor.number_dof * node + 1] = 10 * np.sin(2 * t)
    >>> get_array = apply_pseudo_modal(rotor, speed, num_modes=12)
    >>> F_modal = get_array[1](F.T).T
    >>> round(la.norm(F_modal), 5)
    226.92798
    """

    M = rotor.M(speed)
    K_aux = rotor.K(speed)

    # Remove cross-coupled coefficients of bearing stiffness matrix
    rmv_cross_coeffs = [[0, 1], [1, 0]]
    if rotor.number_dof == 6:
        rmv_cross_coeffs = [[0, 1, 0], [1, 0, 0], [0, 0, 0]]

    for elm in rotor.bearing_elements:
        dofs = list(elm.dof_global_index.values())
        K_aux[np.ix_(dofs, dofs)] -= elm.K(speed) * rmv_cross_coeffs

    _, modal_matrix = eigh(K_aux, M)
    modal_matrix = modal_matrix[:, :num_modes]

    matrix_to_modal = lambda array: (modal_matrix.T @ array) @ modal_matrix
    vector_to_modal = lambda array: modal_matrix.T @ array
    vector_from_modal = lambda array: modal_matrix @ array

    return matrix_to_modal, vector_to_modal, vector_from_modal


def assemble_C_K_matrices(elements, C0, K0, speed=None):
    """Assemble damping and stiffness matrices considering
    specified elements a rotor.

    Parameters
    ----------
    elements : array_like
        List of elements of the rotor.
    C0 : ndarray
        Initial damping matrix.
    K0 : ndarray
        Initial stiffness matrix.
    speed : float, optional
        Rotor speed.
        If `elements` contain bearing elements, the speed must be provided.

    Returns
    -------
    C0 : ndarray
        Initial damping matrix incorporating the specified elements.
    K0 : ndarray
        Initial stiffness matrix incorporating the specified elements.

    Examples
    --------
    >>> import ross as rs
    >>> rotor = rs.rotor_example()
    >>> elements_without_bearing = [
    ...     *rotor.shaft_elements,
    ...     *rotor.disk_elements,
    ...     *rotor.point_mass_elements
    ... ]
    >>> C0 = np.zeros((rotor.ndof, rotor.ndof))
    >>> K0 = np.zeros((rotor.ndof, rotor.ndof))
    >>> C1, K1 = assemble_C_K_matrices(elements_without_bearing, C0, K0)
    >>> C, K = assemble_C_K_matrices(rotor.bearing_elements, C1, K1, speed=0)
    >>> C[:4, :4]
    array([[0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.]])
    >>> np.round(K[:4, :4]/1e6)
    array([[47.,  0.,  0.,  6.],
           [ 0., 46., -6.,  0.],
           [ 0., -6.,  1.,  0.],
           [ 6.,  0.,  0.,  1.]])
    """

    if speed is not None:
        for elm in elements:
            dofs = list(elm.dof_global_index.values())
            C0[np.ix_(dofs, dofs)] += elm.C(speed)
            K0[np.ix_(dofs, dofs)] += elm.K(speed)

    else:
        for elm in elements:
            dofs = list(elm.dof_global_index.values())
            C0[np.ix_(dofs, dofs)] += elm.C()
            K0[np.ix_(dofs, dofs)] += elm.K()

    return C0, K0


def integrate_rotor_system(rotor, speed, F, t, **kwargs):
    """Time integration for a rotor system.

    This method returns the time response for a rotor given a force, time and
    speed based on time integration with the Newmark method.

    Parameters
    ----------
    speed : float or array_like
        Rotor speed.
    F : ndarray
        Force array (needs to have the same length as time array).
    t : ndarray
        Time array.
    **kwargs : optional
        Additional keyword arguments can be passed to define the parameters
        of the Newmark method if it is used (e.g. gamma, beta, tol, ...).
        See `newmark` for more details. Other optional arguments are listed
        below.
    num_modes : int, optional
        If `num_modes` is passed as argument, the pseudo-modal method is applied reducing
        the model to the chosen number of modes.
    add_to_RHS : callable, optional
        An optional function that computes and returns an additional array to be added to
        the right-hand side of the equation of motion. This function should take arguments
        corresponding to the current state of the rotor system, including the time step
        number, displacements, and velocities. It should return an array of the same length
        as the degrees of freedom of the rotor system (`rotor.ndof`). This function allows
        for the incorporation of supplementary terms or external effects in the rotor system
        dynamics beyond the specified force input during the time integration process.

    Returns
    -------
    t : ndarray
        Time values for the output.
    yout : ndarray
        System response.

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
    >>> F[:, rotor.number_dof * node] = 10 * np.cos(2 * t)
    >>> F[:, rotor.number_dof * node + 1] = 10 * np.sin(2 * t)
    >>> t, yout = integrate_rotor_system(rotor, speed, F, t)
    Running direct method
    >>> dof = 13
    >>> yout[:, dof] # doctest: +ELLIPSIS
    array([0.0000000e+00, 8.4914005e-09, 4.3429676e-08, ...
    """

    try:
        speed_is_array = len(set(speed)) > 1
        speed_ref = np.mean(speed)

    except:
        speed_is_array = False
        speed_ref = speed

    num_modes = kwargs.get("num_modes")

    if num_modes and num_modes > 0:
        size = num_modes
        print("Running pseudo-modal method, number of modes =", size)
        get_array = apply_pseudo_modal(rotor, speed_ref, size)

    else:
        size = rotor.ndof
        print("Running direct method")
        return_array = lambda array: array
        get_array = [return_array for j in range(3)]

    add_to_RHS = kwargs.get("add_to_RHS")

    if add_to_RHS is None:
        ext_force = lambda i, u, v: 0

    else:
        ext_force = lambda i, u, v: get_array[1](
            add_to_RHS(i, get_array[2](u), get_array[2](v))
        )

    M = get_array[0](rotor.M())
    C2 = get_array[0](rotor.G())
    K2 = get_array[0](rotor.Ksdt())
    F = get_array[1](F.T).T

    if speed_is_array:
        accel = np.gradient(speed, t)

        freq_is_none = False
        for elm in rotor.bearing_elements:
            freq_is_none = (elm.frequency is None) or freq_is_none

        if freq_is_none:
            C1 = get_array[0](rotor.C(speed_ref))
            K1 = get_array[0](rotor.K(speed_ref))

            rotor_system = lambda step, disp_resp, velc_resp: (
                M,
                C1 + C2 * speed[step],
                K1 + K2 * accel[step],
                F[step, :] + ext_force(step, disp_resp, velc_resp),
            )

        else:
            elements_without_bearing = [
                *rotor.shaft_elements,
                *rotor.disk_elements,
                *rotor.point_mass_elements,
            ]

            C0, K0 = assemble_C_K_matrices(
                elements_without_bearing,
                np.zeros((rotor.ndof, rotor.ndof)),
                np.zeros((rotor.ndof, rotor.ndof)),
            )

            def rotor_system(step, disp_resp, velc_resp):
                C, K = assemble_C_K_matrices(
                    rotor.bearing_elements, np.copy(C0), np.copy(K0), speed[step]
                )

                C1 = get_array[0](C)
                K1 = get_array[0](K)

                return (
                    M,
                    C1 + C2 * speed[step],
                    K1 + K2 * accel[step],
                    F[step, :] + ext_force(step, disp_resp, velc_resp),
                )

    else:
        C1 = get_array[0](rotor.C(speed_ref))
        K1 = get_array[0](rotor.K(speed_ref))

        rotor_system = lambda step, disp_resp, velc_resp: (
            M,
            C1 + C2 * speed,
            K1,
            F[step, :] + ext_force(step, disp_resp, velc_resp),
        )

    response = newmark(rotor_system, t, size, **kwargs)
    yout = get_array[2](response.T).T
    return t, yout
