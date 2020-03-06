import re

import numpy as np
import pandas as pd
from bokeh.models import LogColorMapper
from bokeh.palettes import Viridis256
from bokeh.plotting import figure


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
        Position of the sheet in the file (starting from 0) or its name. If none is passed, it is
        assumed to be the first sheet in the file.
    n: int
        Exclusive for bearing elements, since this parameter is given outside the table file.
    sheet_type: str
        Exclusive for shaft elements, as they have a Model table in which more information can be passed,
        such as the material parameters.
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
        for index, row in df_material.iterrows():
            if not pd.isna(row["matno"]):
                material_name.append(int(row["matno"]))
                material_rho.append(row["rhoa"])
                material_e.append(row["ea"])
                material_g_s.append(row["ga"])
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


def visualize_matrix(rotor, matrix=None, frequency=None):
    """Visualize global matrix.

    This function gives some visualization of a given matrix, displaying
    values on a heatmap.

    Parameters
    ----------
    rotor: rs.Rotor
    
    matrix: str
        String for the desired matrix.
    frequency: float, optional
        Excitation frequency. Defaults to rotor speed.
    
    Returns
    -------
    fig: bokeh.figure
    """
    if frequency is None:
        frequency = rotor.w

    A = np.zeros((rotor.ndof, rotor.ndof))
    # E will store element's names and contributions to the global matrix
    E = np.zeros((rotor.ndof, rotor.ndof), dtype=np.object)

    M, N = E.shape
    for i in range(M):
        for j in range(N):
            E[i, j] = []

    for elm in rotor.elements:
        g_dofs = elm.dof_global_index()
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
                        "\n"
                        + elm.__class__.__name__
                        + f"(n={elm.n})"
                        + f": {elm_matrix[l0, l1]:.2E}"
                    )

    # list for dofs -> ['x0', 'y0', 'alpha0', 'beta0', 'x1'...]
    dof_list = [0 for i in range(rotor.ndof)]

    for elm in rotor.elements:
        for k, v in elm.dof_global_index()._asdict().items():
            dof_list[v] = k

    data = {"row": [], "col": [], "value": [], "pos_value": [], "elements": []}
    for i, dof_row in enumerate(dof_list):
        for j, dof_col in enumerate(dof_list):
            data["row"].append(i)
            data["col"].append(j)
            data["value"].append(A[i, j])
            data["pos_value"].append(abs(A[i, j]))
            data["elements"].append(E[i, j])

    TOOLTIPS = """
        <div style="width:150px;">
            <span style="font-size: 10px;">Value: @value</span>
            <br>
            <span style="font-size: 10px;">Elements: </span>
            <br>
            <span style="font-size: 10px;">@elements</span>
        </div>
    """
    fig = figure(
        tools="hover",
        tooltips=TOOLTIPS,
        x_range=(-0.5, len(A) - 0.5),
        y_range=(len(A) - 0.5, -0.5),
        x_axis_location="above",
    )
    fig.plot_width = 500
    fig.plot_height = 500
    fig.grid.grid_line_color = None
    fig.axis.axis_line_color = None
    fig.axis.major_tick_line_color = None
    fig.axis.major_label_text_font_size = "7pt"
    fig.axis.major_label_standoff = 0
    fig.xaxis.ticker = [i for i in range(len(dof_list))]
    fig.yaxis.ticker = [i for i in range(len(dof_list))]

    dof_string_list = []
    sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    for d in dof_list:
        d = d.replace("alpha", "α")
        d = d.replace("beta", "β")
        d = d.replace("_", "")
        d = d.translate(sub)
        dof_string_list.append(d)

    fig.xaxis.major_label_overrides = {k: v for k, v in enumerate(dof_string_list)}
    fig.yaxis.major_label_overrides = {k: v for k, v in enumerate(dof_string_list)}

    mapper = LogColorMapper(palette=Viridis256, low=0, high=A.max())

    fig.rect(
        "row",
        "col",
        0.95,
        0.95,
        source=data,
        fill_color={"field": "pos_value", "transform": mapper},
        line_color=None,
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
