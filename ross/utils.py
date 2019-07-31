import pandas as pd
import xlrd
import sys
from ross.materials import Material


def read_table_file(file, element, sheet_name=0, n=0, sheet_type="Model"):
    """Instantiate one or more element objects using inputs from an Excel table, csv, or similar.
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
    A matrix of parameters.
    """
    is_csv = False
    try:
        df = pd.read_excel(file, header=None, sheet_name=sheet_name)
    except FileNotFoundError:
        sys.exit(file + " not found.")
    except xlrd.biffh.XLRDError:
        df = pd.read_csv(file, header=None)
        is_csv = True

    # Assign specific values to variables
    parameter_columns = []
    optional_parameter_columns = []
    header_key_word = ""
    default_list = []
    if element == "bearing":
        header_key_word = "kxx"
        parameter_columns.append(["kxx", "Kxx", "KXX"])
        parameter_columns.append(["cxx", "Cxx", "CXX"])
        optional_parameter_columns.append(["kyy", "Kyy", "KYY"])
        optional_parameter_columns.append(["kxy", "Kxy", "KXY"])
        optional_parameter_columns.append(["kyx", "Kyx", "KYX"])
        optional_parameter_columns.append(["cyy", "Cyy", "CYY"])
        optional_parameter_columns.append(["cxy", "Cxy", "CXY"])
        optional_parameter_columns.append(["cyx", "Cyx", "CYX"])
        optional_parameter_columns.append(["w", "W", "speed", "Speed", "SPEED"])
        default_list.append(None)
        default_list.append(0)
        default_list.append(0)
        default_list.append(None)
        default_list.append(0)
        default_list.append(0)
        default_list.append(None)
    elif element == "shaft":
        if sheet_type == "Model":
            header_key_word = "od_left"
        else:
            header_key_word = "material"
        parameter_columns.append(["length", "Length", "LENGTH"])
        parameter_columns.append(["i_d", "I_D", "id", "ID", "id_left", "id_Left", "ID_LEFT"])
        parameter_columns.append(["o_d", "O_D", "od", "OD", "od_left", "od_Left", "OD_LEFT"])
        parameter_columns.append(["material", "Material", "MATERIAL",
                                  "matnum"])
        optional_parameter_columns.append(["n", "N", "elemnum"])
        optional_parameter_columns.append(["axial_force", "axial force", "Axial Force", "AXIAL FORCE",
                                           "axial", "Axial", "AXIAL"])
        optional_parameter_columns.append(["torque", "Torque", "TORQUE"])
        optional_parameter_columns.append(["shear_effects", "shear effects", "Shear Effects", "SHEAR EFFECTS"])
        optional_parameter_columns.append(["rotary_inertia", "rotary inertia", "Rotary Inertia", "ROTARY INERTIA"])
        optional_parameter_columns.append(["gyroscopic", "Gyroscopic", "GYROSCOPIC"])
        optional_parameter_columns.append(["shear_method_calc", "shear method calc", "Shear Method Calc",
                                           "SHEAR METHOD CALC"])
        default_list.append(None)
        default_list.append(0)
        default_list.append(0)
        default_list.append(True)
        default_list.append(True)
        default_list.append(True)
        default_list.append("cowper")
    elif element == "disk":
        header_key_word = "ip"
        parameter_columns.append(["Unnamed: 0", "n", "N"])
        parameter_columns.append(["m", "M", "mass", "Mass", "MASS"])
        parameter_columns.append(["ip", "Ip", "IP"])
        parameter_columns.append(["it", "It", "IT", "id", "Id", "ID"])

    # Find table header
    header_index = -1
    header_found = False
    for index, row in df.iterrows():
        for i in range(0, row.size):
            if isinstance(row[i], str):
                if row[i].lower() == header_key_word:
                    header_index = index
                    header_found = True
                    break
        if header_found:
            break
    if not header_found:
        sys.exit("Could not find the header. Make sure the table has a header "
                 "containing the names of the columns. In the case of a " + element + ", "
                 "there should be a column named " + header_key_word + ".")

    if not is_csv:
        df = pd.read_excel(file, header=header_index, sheet_name=sheet_name)
        df_unit = pd.read_excel(file, header=header_index, nrows=2, sheet_name=sheet_name)
    else:
        df = pd.read_csv(file, header=header_index)
        df_unit = pd.read_csv(file, header=header_index, nrows=2)

    # Define if it needs to be converted
    convert_to_metric = False
    for index, row in df_unit.iterrows():
        for i in range(0, row.size):
            if isinstance(row[i], str):
                if 'inches' in row[i].lower() or 'lbm' in row[i].lower():
                    convert_to_metric = True

    # Get specific data from the file
    if element == "shaft" and sheet_type == "Model":
        df_material = pd.read_excel(file, header=3, sheet_name=sheet_name)
        material_name = []
        material_rho = []
        material_e = []
        material_g_s = []
        new_materials = {}
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
        for i in range(0, len(material_name)):
            new_material = Material(
                name="shaft_mat_" + str(material_name[i]),
                rho=material_rho[i],
                E=material_e[i],
                G_s=material_g_s[i],
            )
            new_materials["shaft_mat_" + str(material_name[i])] = new_material

    # Find and isolate data rows
    first_data_row_found = False
    indexes_to_drop = []
    for index, row in df.iterrows():
        if not first_data_row_found and (isinstance(row[0], int) or isinstance(row[0], float)):
            first_data_row_found = True
        elif not first_data_row_found:
            indexes_to_drop.append(index)
        elif first_data_row_found:
            if isinstance(row[0], int) or isinstance(row[0], float):
                continue
            else:
                indexes_to_drop.append(index)
    if not first_data_row_found:
        sys.exit("Could not find the data. Make sure you have at least one row containing "
                 "data below the header.")
    if len(indexes_to_drop) > 0:
        df = df.drop(indexes_to_drop)

    # Build parameters list
    parameters = []
    if element == "bearing":
        parameters.append(n)
    for name_group in parameter_columns:
        for name in name_group:
            try:
                parameter = df[name].tolist()
                parameters.append(parameter)
                break
            except KeyError:
                if name == name_group[-1]:
                    sys.exit("Could not find a column with one of these names: " + str(name_group))
                continue
    for i in range(0, len(optional_parameter_columns)):
        for name in optional_parameter_columns[i]:
            try:
                parameter = df[name].tolist()
                parameters.append(parameter)
                break
            except KeyError:
                if name == optional_parameter_columns[i][-1]:
                    parameters.append([default_list[i]]*df.shape[0])
                else:
                    continue
    if element == "shaft" and sheet_type == "Model":
        for i in range(0, len(parameters[3])):
            parameters[3][i] = "shaft_mat_" + str(parameters[3][i])
    return parameters







