import pandas as pd
import xlrd
import sys
from ross.materials import Material


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
    header_key_word = ''
    default_dictionary = {}
    parameters = {}
    if element == 'bearing':
        header_key_word = 'kxx'
        parameter_columns['kxx'] = ['kxx']
        parameter_columns['cxx'] = ['cxx']
        optional_parameter_columns['kyy'] = ['kyy']
        optional_parameter_columns['kxy'] = ['kxy']
        optional_parameter_columns['kyx'] = ['kyx']
        optional_parameter_columns['cyy'] = ['cyy']
        optional_parameter_columns['cxy'] = ['cxy']
        optional_parameter_columns['cyx'] = ['cyx']
        optional_parameter_columns['w'] = ['w', 'speed']
        default_dictionary['kyy'] = None
        default_dictionary['kxy'] = 0
        default_dictionary['kyx'] = 0
        default_dictionary['cyy'] = None
        default_dictionary['cxy'] = 0
        default_dictionary['cyx'] = 0
        default_dictionary['w'] = None
    elif element == 'shaft':
        if sheet_type == 'Model':
            header_key_word = 'od_left'
        else:
            header_key_word = 'material'
        parameter_columns['L'] = ['length']
        parameter_columns['i_d'] = ['i_d', 'id', 'id_left']
        parameter_columns['o_d'] = ['o_d', 'od', 'od_left']
        parameter_columns['material'] = ['material', 'matnum']
        optional_parameter_columns['n'] = ['n', 'elemnum']
        optional_parameter_columns['axial_force'] = ['axial_force', 'axial force', 'axial']
        optional_parameter_columns['torque'] = ['torque']
        optional_parameter_columns['shear_effects'] = ['shear_effects', 'shear effects']
        optional_parameter_columns['rotary_inertia'] = ['rotary_inertia', 'rotary inertia']
        optional_parameter_columns['gyroscopic'] = ['gyroscopic']
        optional_parameter_columns['shear_method_calc'] = ['shear_method_calc', 'shear method calc']
        default_dictionary['n'] = None
        default_dictionary['axial_force'] = 0
        default_dictionary['torque'] = 0
        default_dictionary['shear_effects'] = True
        default_dictionary['rotary_inertia'] = True
        default_dictionary['gyroscopic'] = True
        default_dictionary['shear_method_calc'] = 'cowper'
    elif element == 'disk':
        header_key_word = 'ip'
        parameter_columns['n'] = ['unnamed: 0', 'n']
        parameter_columns['m'] = ['m', 'mass']
        parameter_columns['Id'] = ['it', 'id']
        parameter_columns['Ip'] = ['ip']

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
                if 'inches' in row[i].lower() or 'lbm' in row[i].lower():
                    convert_to_metric = True
                if 'rpm' in row[i].lower():
                    convert_to_rad_per_sec = True
                if header_found and convert_to_metric and convert_to_rad_per_sec:
                    break
        if header_found and convert_to_metric:
            break
    if not header_found:
        raise ValueError("Could not find the header. Make sure the table has a header "
                         "containing the names of the columns. In the case of a " + element + ", "
                         "there should be a column named " + header_key_word + ".")

    # Get specific data from the file
    new_materials = {}
    if element == 'shaft' and sheet_type == 'Model':
        material_header_index = -1
        material_header_found = False
        material_header_key_word = 'matno'
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
            raise ValueError("Could not find the header for the materials. Make sure the table has a header "
                             "with the parameters for the materials that will be used. There should be a column "
                             "named " + material_header_key_word + ".")
        df_material = pd.read_excel(file, header=material_header_index, sheet_name=sheet_name)
        material_name = []
        material_rho = []
        material_e = []
        material_g_s = []
        for index, row in df_material.iterrows():
            if not pd.isna(row['matno']):
                material_name.append(int(row['matno']))
                material_rho.append(row['rhoa'])
                material_e.append(row['ea'])
                material_g_s.append(row['ga'])
            else:
                break
        if convert_to_metric:
            for i in range(0, len(material_name)):
                material_rho[i] = material_rho[i] * 27679.904
                material_e[i] = material_e[i] * 6894.757
                material_g_s[i] = material_g_s[i] * 6894.757
        for i in range(0, len(material_name)):
            new_material = Material(
                name='shaft_mat_' + str(material_name[i]),
                rho=material_rho[i],
                E=material_e[i],
                G_s=material_g_s[i],
            )
            new_materials['shaft_mat_' + str(material_name[i])] = new_material

    df = pd.read_excel(file, header=header_index, sheet_name=sheet_name)
    df.columns = df.columns.str.lower()

    # Find and isolate data rows
    first_data_row_found = False
    last_data_row_found = False
    indexes_to_drop = []
    for index, row in df.iterrows():
        if not first_data_row_found \
                and (isinstance(row[header_key_word], int) or isinstance(row[header_key_word], float)) \
                and not pd.isna(row[header_key_word]):
            first_data_row_found = True
        elif not first_data_row_found:
            indexes_to_drop.append(index)
        elif first_data_row_found and not last_data_row_found:
            if (isinstance(row[header_key_word], int) or isinstance(row[header_key_word], float))\
                    and not pd.isna(row[header_key_word]):
                continue
            else:
                last_data_row_found = True
                indexes_to_drop.append(index)
        elif last_data_row_found:
            indexes_to_drop.append(index)
    if not first_data_row_found:
        raise DataNotFoundError("Could not find the data. Make sure you have at least one row containing "
                                "data below the header.")
    if len(indexes_to_drop) > 0:
        df = df.drop(indexes_to_drop)

    # Build parameters list
    if element == 'bearing':
        parameters['n'] = n
    for key, value in parameter_columns.items():
        for name in value:
            try:
                parameters[key] = df[name].tolist()
                break
            except KeyError:
                if name == value[-1]:
                    raise ValueError("Could not find a column with one of these names: " + str(value))
                continue
    for key, value in optional_parameter_columns.items():
        for name in value:
            try:
                parameters[key] = df[name].tolist()
                break
            except KeyError:
                if name == value[-1]:
                    parameters[key] = [default_dictionary[key]]*df.shape[0]
                else:
                    continue
    if element == 'shaft':
        new_n = parameters['n']
        for i in range(0, df.shape[0]):
            new_n[i] -= 1
        parameters['n'] = new_n
        if sheet_type == 'Model':
            new_material = parameters['material']
            for i in range(0, df.shape[0]):
                new_material[i] = 'shaft_mat_' + str(int(new_material[i]))
            parameters['material'] = new_material
    if convert_to_metric:
        for i in range(0, df.shape[0]):
            if element == 'bearing':
                parameters['kxx'][i] = parameters['kxx'][i] * 175.1268369864
                parameters['cxx'][i] = parameters['cxx'][i] * 175.1268369864
                parameters['kyy'][i] = parameters['kyy'][i] * 175.1268369864
                parameters['kxy'][i] = parameters['kxy'][i] * 175.1268369864
                parameters['kyx'][i] = parameters['kyx'][i] * 175.1268369864
                parameters['cyy'][i] = parameters['cyy'][i] * 175.1268369864
                parameters['cxy'][i] = parameters['cxy'][i] * 175.1268369864
                parameters['cyx'][i] = parameters['cyx'][i] * 175.1268369864
            if element == 'shaft':
                parameters['L'][i] = parameters['L'][i] * 0.0254
                parameters['i_d'][i] = parameters['i_d'][i] * 0.0254
                parameters['o_d'][i] = parameters['o_d'][i] * 0.0254
                parameters['axial_force'][i] = parameters['axial_force'][i] * 4.448221615255
            elif element == 'disk':
                parameters['m'][i] = parameters['m'][i] * 0.45359237
                parameters['Id'][i] = parameters['Id'][i] * 0.0002926397
                parameters['Ip'][i] = parameters['Ip'][i] * 0.0002926397
    if convert_to_rad_per_sec:
        for i in range(0, df.shape[0]):
            if element == 'bearing':
                parameters['w'][i] = parameters['w'][i] * 0.1047197551197
    parameters.update(new_materials)
    return parameters







