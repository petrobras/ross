import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import ross as rs
import numpy as np
import hashlib
import json
import base64
import struct
import math
import ast
import toml
from ross.units import Q_

app = Flask(__name__)
CORS(app) 

# Master Units Dictionary

UNITS_MAPPING = {
    'Material': {'rho': 'kg/m**3', 'E': 'N/m**2', 'G_s': 'N/m**2'},
    'ShaftElement': {'L': 'mm', 'idl': 'mm', 'odl': 'mm', 'idr': 'mm', 'odr': 'mm'},
    'DiskElement': {'m': 'kg', 'Id': 'kg*m**2', 'Ip': 'kg*m**2'},
    'GearElement': {'m': 'kg', 'Id': 'kg*m**2', 'Ip': 'kg*m**2', 'base_diameter': 'mm', 'pitch_diameter': 'mm', 'pr_angle': 'deg', 'helix_angle': 'deg', 'bore_diameter': 'mm'},
    'GearElementTVMS': {'width': 'mm', 'bore_diameter': 'mm', 'module': 'mm', 'pr_angle': 'deg', 'helix_angle': 'deg'},
    'CouplingElement': {'m_l': 'kg', 'm_r': 'kg', 'Ip_l': 'kg*m**2', 'Ip_r': 'kg*m**2', 'Id_l': 'kg*m**2', 'Id_r': 'kg*m**2', 'o_d': 'mm', 'L': 'mm'},
    'BearingElement': {'kxx': 'N/m', 'kxy': 'N/m', 'kyx': 'N/m', 'kyy': 'N/m', 'kzz': 'N/m', 'cxx': 'N*s/m', 'cxy': 'N*s/m', 'cyx': 'N*s/m', 'cyy': 'N*s/m', 'czz': 'N*s/m', 'mxx': 'kg', 'mxy': 'kg', 'myx': 'kg', 'myy': 'kg', 'mzz': 'kg', 'frequency': 'RPM'},
    'BallBearingElement': {},
    'RollerBearingElement': {},
    'MagneticBearingElement': {},
    'CylindricalBearing': {'speed': 'RPM', 'weight': 'N', 'bearing_length': 'mm', 'journal_diameter': 'mm', 'radial_clearance': 'mm', 'oil_viscosity': 'Pa*s'},
    'PlainJournal': {'axial_length': 'mm', 'journal_radius': 'mm', 'radial_clearance': 'mm', 'pad_arc_length': 'deg', 'frequency': 'RPM', 'fxs_load': 'N', 'fys_load': 'N', 'reference_temperature': 'degC', 'oil_flow_v': 'l/min', 'oil_supply_pressure': 'Pa'},
    'SqueezeFilmDamper': {'frequency': 'RPM', 'axial_length': 'mm', 'journal_radius': 'mm', 'radial_clearance': 'mm'},
    'ThrustPad': {'pad_inner_radius': 'mm', 'pad_outer_radius': 'mm', 'pad_pivot_radius': 'mm', 'pad_arc_length': 'deg', 'angular_pivot_position': 'deg', 'oil_supply_temperature': 'degC', 'frequency': 'RPM', 'radial_inclination_angle': 'rad', 'circumferential_inclination_angle': 'rad', 'initial_film_thickness': 'mm', 'axial_load': 'N'},
    'TiltingPad': { 'journal_diameter': 'mm', 'pad_thickness': 'mm', 'pad_arc': 'deg', 'pad_axial_length': 'mm', 'oil_supply_temperature': 'degC', 'radial_clearance': 'mm', 'pivot_angle': 'deg', 'frequency': 'RPM', 'attitude_angle': 'deg', 'xj': 'mm', 'yj': 'mm', 'initial_pads_angles': 'deg'},
    'SealElement': {'kxx': 'N/m', 'kxy': 'N/m', 'kyx': 'N/m', 'kyy': 'N/m', 'kzz': 'N/m', 'cxx': 'N*s/m', 'cxy': 'N*s/m', 'cyx': 'N*s/m', 'cyy': 'N*s/m', 'czz': 'N*s/m', 'mxx': 'kg', 'mxy': 'kg', 'myx': 'kg', 'myy': 'kg', 'mzz': 'kg', 'frequency': 'RPM'},
    'HolePatternSeal': {'shaft_radius': 'mm', 'radial_clearance': 'mm', 'length': 'mm', 'cell_length': 'mm', 'cell_width': 'mm', 'cell_depth': 'mm', 'inlet_pressure': 'Pa', 'outlet_pressure': 'Pa', 'inlet_temperature': 'degC', 'frequency': 'RPM'},
    'LabyrinthSeal': {'shaft_radius': 'mm', 'radial_clearance': 'mm', 'pitch': 'mm', 'tooth_height': 'mm', 'tooth_width': 'mm', 'inlet_pressure': 'Pa', 'outlet_pressure': 'Pa', 'inlet_temperature': 'degC', 'frequency': 'RPM'},
    'HybridSeal': {'shaft_radius': 'mm', 'inlet_pressure': 'Pa', 'outlet_pressure': 'Pa', 'inlet_temperature': 'degC', 'frequency': 'RPM'},
    'PointMass': {'m': 'kg', 'mx': 'kg', 'my': 'kg', 'mz': 'kg'}
}

# List of parameters that do not alter the physics of the analysis, only the plotting
PLOT_KEYS = {
    'plot_type', 'plot_idx', 'frequency_units', 'speed_units',
    'damping_parameter', 'animation', 'stiffness_units',
    'amplitude_units', 'phase_units', 'line_shape',
    'orientation', 'length_units', 'nodes', 'probe_units',
    'displacement_units', 'time_units', 'rotor_length_units',
    'deformation_units', 'force_units', 'moment_units', 'harmonics'
}

# Useful Functions

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)): return int(obj)
        if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)): return float(obj)
        if isinstance(obj, (np.complex_, np.complex64, np.complex128, complex)): return {'real': obj.real, 'imag': obj.imag}
        return super().default(obj)

def decode_bdata(obj):
    if isinstance(obj, dict):
        if 'bdata' in obj and 'dtype' in obj:
            try:
                bdata = obj['bdata']; dtype = obj['dtype']
                decoded = base64.b64decode(bdata)
                if dtype == 'f8': return list(struct.unpack(f'<{len(decoded)//8}d', decoded))
                elif dtype == 'f4': return list(struct.unpack(f'<{len(decoded)//4}f', decoded))
            except Exception: pass
        return {k: decode_bdata(v) for k, v in obj.items()}
    elif isinstance(obj, list): return [decode_bdata(v) for v in obj]
    return obj

def remove_nans(obj):
    if isinstance(obj, dict): return {k: remove_nans(v) for k, v in obj.items()}
    elif isinstance(obj, list): return [remove_nans(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj): return None
    return obj

def get_int(dictionary, key, default=0):
    val = dictionary.get(key)
    if val is None or str(val).strip() == "":
        return default
    return int(float(val))

def safe_math_eval(expr):
    safe_dict = {
        'pi': math.pi, 'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
        'sqrt': math.sqrt, 'exp': math.exp, 'log': math.log, 'np': np
    }
    try:
        return float(eval(expr, {"__builtins__": None}, safe_dict))
    except Exception:
        raise ValueError(f"It was not possible to evaluate the mathematical expression: {expr}")

def extract_kwargs(d, mat_dict, element_type, ignore_keys=['element_type', 'n']):
    kwargs = {}
    int_keys = ['n_pad', 'n_theta', 'n_radial', 'n_teeth', 'n_rollers', 'n_balls', 'nx', 'nz', 'nr_pad', 'max_inlet_iterations', 'max_jtemp_iter', 'max_iterations', 'elements_circumferential', 'elements_axial', 'n_link', 'n_l', 'n_r']    
    
    unit_map = UNITS_MAPPING.get(element_type, {})
    
    for k, v in d.items():
        if k in ignore_keys or k.endswith('_unit'): continue
        if v is None: continue
        
        if k == 'material':
            mat_name = str(v).strip().lower()
            if mat_name == "": continue 
            
            kwargs[k] = mat_dict.get(mat_name, list(mat_dict.values())[0]) if mat_dict else rs.materials.steel
            continue
            
        if isinstance(v, str):
            v_strip = v.strip()
            
            if v_strip == "": 
                continue 
                
            if v_strip.startswith('[') or v_strip.startswith('{'):
                try:
                    val_parsed = ast.literal_eval(v_strip)
                    if isinstance(val_parsed, list): 
                        val_parsed = [float(x) for x in val_parsed]
                        unit = d.get(f"{k}_unit", unit_map.get(k))
                        if unit:
                            kwargs[k] = Q_(np.array(val_parsed), unit).to_base_units()
                        else:
                            kwargs[k] = np.array(val_parsed)
                    else:
                        kwargs[k] = val_parsed
                    continue
                except: pass
                
            if v_strip.lower() == "true": kwargs[k] = True; continue
            if v_strip.lower() == "false": kwargs[k] = False; continue
            
            try:
                val_num = float(v_strip)
            except ValueError:
                try:
                    val_num = safe_math_eval(v_strip)
                except ValueError:
                    kwargs[k] = v_strip
                    continue

            if k in int_keys:
                val_num = int(val_num)
            
            unit = d.get(f"{k}_unit", unit_map.get(k))
            if unit:
                kwargs[k] = Q_(val_num, unit).to_base_units()
            else:
                kwargs[k] = val_num
            continue
        else:
            kwargs[k] = v
            
    return kwargs

# Memory Cache

ROSS_CACHE = {}
ANALYSIS_CACHE = {}

def get_analysis_hash(data):
    hash_data = {k: v for k, v in data.items() if k != 'params'}
    hash_data['analysis_type'] = data.get('analysis_type')    
    params = data.get('params', {})
    hash_params = {k: v for k, v in params.items() if k not in PLOT_KEYS}
    hash_data['params'] = hash_params    
    dict_str = json.dumps(hash_data, sort_keys=True)

    return hashlib.md5(dict_str.encode('utf-8')).hexdigest()

def get_eff_nodes(arr):
    eff = []
    auto_n = 0
    
    for el in arr:
        n_str = str(el.get('n', '')).strip()
        is_explicit = False
        
        if n_str != "":
            try:
                explicit_val = int(float(n_str))
                eff.append(explicit_val)
                is_explicit = True
            except ValueError:
                pass
                
        if not is_explicit:
            eff.append(auto_n)
            auto_n += 1
            
    return eff

# Build Rotor

def build_rotor_from_ui(data):
    global ROSS_CACHE
    current_hashes = set()
    
    mat_ui_props = {str(m.get('name', 'MaterialCustom')).strip().lower(): m for m in data.get('materials', [])}
    
    created_materials = {}
    for mat in data.get('materials', []):
        name = str(mat.get('name', 'MaterialCustom')).strip()
        kwargs = extract_kwargs(mat, {}, 'Material', ['name', 'element_type'])
        if 'poisson' in kwargs: kwargs['Poisson'] = kwargs.pop('poisson')
        created_materials[name.lower()] = rs.Material(name=name, **kwargs)
        
    def instantiate_with_cache(category, el_dict, n_val, builder_func, auto_tag):
        hash_data = {k: v for k, v in el_dict.items() if str(v).strip() != ""}
        hash_data['__cat'] = category
        hash_data['__n_calc'] = n_val 
        hash_data['__tag'] = el_dict.get('tag', auto_tag) 
        
        if 'material' in hash_data:
            m_name = str(hash_data['material']).strip().lower()
            if m_name in mat_ui_props:
                hash_data['__mat_props'] = {k: v for k, v in mat_ui_props[m_name].items() if str(v).strip() != ""}
        
        dict_str = json.dumps(hash_data, sort_keys=True)
        h = hashlib.md5(dict_str.encode('utf-8')).hexdigest()
        current_hashes.add(h) 
        
        if h in ROSS_CACHE:
            return ROSS_CACHE[h] 
        else:
            obj = builder_func() 
            ROSS_CACHE[h] = obj  
            return obj

    # Shafts
    ross_shafts = []
    shafts_eff = get_eff_nodes(data.get('shafts', []))
    for i, shaft in enumerate(data.get('shafts', [])):
        n_val = shafts_eff[i]
        auto_tag = f"shaft_{i}"
        def build_shaft():
            kwargs = extract_kwargs(shaft, created_materials, 'ShaftElement', ['element_type', 'n'])
            if 'tag' not in kwargs: kwargs['tag'] = auto_tag
            odl_mag = kwargs['odl'].m if hasattr(kwargs.get('odl'), 'm') else kwargs.get('odl', 0)
            if odl_mag <= 0.0: raise ValueError("Invalid outer diameter for Shaft!")
            return rs.ShaftElement(n=n_val, **kwargs)
        ross_shafts.append(instantiate_with_cache('shaft', shaft, n_val, build_shaft, auto_tag))
        
    if not ross_shafts: raise ValueError("Add at least one Shaft!")

    # Disks
    ross_disks = []
    disks_eff = get_eff_nodes(data.get('disks', []))
    for i, d in enumerate(data.get('disks', [])):
        n_val = disks_eff[i]
        auto_tag = f"disk_{i}"
        def build_disk():
            kwargs = extract_kwargs(d, created_materials, 'DiskElement', ['n', 'element_type'])
            if 'tag' not in kwargs: kwargs['tag'] = auto_tag 
            return rs.DiskElement(n=n_val, **kwargs)
        ross_disks.append(instantiate_with_cache('disk', d, n_val, build_disk, auto_tag))
        
    # Gears
    ross_gears = []
    gears_eff = get_eff_nodes(data.get('gears', []))
    for i, g in enumerate(data.get('gears', [])):
        n_val = gears_eff[i]
        type_val = g.get('element_type', 'BASIC')
        auto_tag = f"gear_{i}_{type_val}"
        def build_gear():
            element_class = 'GearElementTVMS' if type_val == 'TVMS' else 'GearElement'
            kwargs = extract_kwargs(g, created_materials, element_class, ['n', 'element_type'])
            if 'tag' not in kwargs: kwargs['tag'] = auto_tag
            if type_val == 'BASIC' and 'pitch_diameter' not in kwargs and 'base_diameter' not in kwargs:
                raise ValueError("For gears, provide either Pitch Diameter or Base Diameter.")
            ross_class = getattr(rs, element_class, rs.GearElement)
            return ross_class(n=n_val, **kwargs)
        ross_gears.append(instantiate_with_cache('gear', g, n_val, build_gear, auto_tag))

    # Bearings
    ross_bearings = []
    bearings_eff = get_eff_nodes(data.get('bearings', []))
    for i, m in enumerate(data.get('bearings', [])):
        n_val = bearings_eff[i]
        type_val = m.get('element_type', 'BASIC')
        auto_tag = f"bearing_{i}_{type_val}"
        def build_bearing():
            type_map = {
                'BASIC': 'BearingElement', 'BallBearing': 'BallBearingElement', 'RollerBearing': 'RollerBearingElement',
                'MagneticBearing': 'MagneticBearingElement', 'Cylindrical': 'CylindricalBearing', 
                'PlainJournal': 'PlainJournal', 'SqueezeFilm': 'SqueezeFilmDamper', 
                'ThrustPad': 'ThrustPad', 'TiltingPad': 'TiltingPad'
            }
            element_class = type_map.get(type_val, 'BearingElement')
            kwargs = extract_kwargs(m, created_materials, element_class, ['n', 'element_type'])
            if 'tag' not in kwargs: kwargs['tag'] = auto_tag
            ross_class = getattr(rs, element_class, rs.BearingElement)
            return ross_class(n=n_val, **kwargs)
        ross_bearings.append(instantiate_with_cache('bearing', m, n_val, build_bearing, auto_tag))

    # Seals
    ross_seals = []
    seals_eff = get_eff_nodes(data.get('seals', []))
    for i, s in enumerate(data.get('seals', [])):
        n_val = seals_eff[i]
        type_val = s.get('element_type', 'BASIC')
        auto_tag = f"seal_{i}_{type_val}"
        def build_seal():
            type_map = {
                'BASIC': 'SealElement', 'HolePattern': 'HolePatternSeal', 'Labyrinth': 'LabyrinthSeal', 'Hybrid': 'HybridSeal'
            }
            element_class = type_map.get(type_val, 'SealElement')
            kwargs = extract_kwargs(s, created_materials, element_class, ['n', 'element_type'])
            if 'tag' not in kwargs: kwargs['tag'] = auto_tag
            ross_class = getattr(rs, element_class, rs.SealElement)
            return ross_class(n=n_val, **kwargs)
        ross_seals.append(instantiate_with_cache('seal', s, n_val, build_seal, auto_tag))

    # Couplings
    ross_couplings = []
    for i, c in enumerate(data.get('couplings', [])):
        n_val_str = str(c.get('n', '')).strip()
        n_val = int(float(n_val_str)) if n_val_str else i
        auto_tag = f"coupling_{i}"
        def build_coupling():
            kwargs = extract_kwargs(c, created_materials, 'CouplingElement', ignore_keys=['element_type'])
            
            if 'n' in kwargs and str(kwargs['n']).strip() != "":
                kwargs['n'] = int(float(kwargs['n']))
            else:
                kwargs['n'] = n_val 
                
            if 'n_link' in kwargs:
                kwargs['n_link'] = int(float(kwargs['n_link']))
            
            if 'tag' not in kwargs: kwargs['tag'] = auto_tag 
            return rs.CouplingElement(**kwargs)
            
        ross_couplings.append(instantiate_with_cache('coupling', c, n_val, build_coupling, auto_tag))
        
    # Point Masses
    ross_pointmasses = []
    pointmass_eff = get_eff_nodes(data.get('pointmasses', []))
    for i, p in enumerate(data.get('pointmasses', [])):
        n_val = pointmass_eff[i]
        auto_tag = f"pointmass_{i}"
        def build_pointmass():
            kwargs = extract_kwargs(p, created_materials, 'PointMass', ['n', 'element_type'])
            if 'tag' not in kwargs: kwargs['tag'] = auto_tag
            return rs.PointMass(n=n_val, **kwargs)
        ross_pointmasses.append(instantiate_with_cache('pointmass', p, n_val, build_pointmass, auto_tag))

    # Rotor Assembly

    keys_to_remove = [h for h in ROSS_CACHE if h not in current_hashes]
    for h in keys_to_remove:
        del ROSS_CACHE[h]

    all_nodes = set()
    all_elements = (ross_shafts + ross_couplings + ross_bearings + 
                    ross_seals + ross_disks + ross_gears + ross_pointmasses)
    
    for elm in all_elements:
        if hasattr(elm, 'nodes'):
            if isinstance(elm.nodes, int):
                all_nodes.add(elm.nodes)
            else:
                all_nodes.update(elm.nodes)
        else:
            if hasattr(elm, 'n_l'): all_nodes.add(elm.n_l)
            if hasattr(elm, 'n_r'): all_nodes.add(elm.n_r)
            if hasattr(elm, 'n'): all_nodes.add(elm.n)
    
    all_nodes.discard(None)
    
    if all_nodes:
        max_n = max(all_nodes)
            
    if all_nodes:
        max_n = max(all_nodes)
        if max_n >= len(all_nodes):
            raise ValueError(f"Invalid node topology. The largest assigned node is {max_n}, but the rotor only has {len(all_nodes)} contiguous nodes. ROSS requires uninterrupted numbering (ex: 0, 1, 2, 3).")

    ross_shafts.sort(key=lambda x: x.n)
    ross_disks.sort(key=lambda x: x.n)
    ross_gears.sort(key=lambda x: x.n)
    ross_bearings.sort(key=lambda x: x.n)
    ross_seals.sort(key=lambda x: x.n)
    ross_pointmasses.sort(key=lambda x: x.n)

    return rs.Rotor(shaft_elements=ross_shafts + ross_couplings, 
                    disk_elements=ross_disks + ross_gears, 
                    bearing_elements=ross_bearings + ross_seals,
                    point_mass_elements=ross_pointmasses)

# Interaction with Javascript

@app.route('/build_rotor', methods=['POST'])
def build_rotor():
    try:
        rotor = build_rotor_from_ui(request.json)
        fig_dict = remove_nans(decode_bdata(rotor.plot_rotor().to_dict()))
        mass = float(rotor.m)
        ip = float(rotor.Ip)
        return jsonify({"status": "success", "plot_json": json.dumps(fig_dict, cls=NumpyEncoder), "mass": mass, "ip": ip})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/run_analysis', methods=['POST'])
def run_analysis():
    data = request.json
    analysis_type = data.get('analysis_type')
    params = data.get('params', {}) 
    
    # Helper functions to format string inputs
    def get_bool(key, default=False): return str(params.get(key, default)).lower() == 'true'
    def get_list(key):
        val = params.get(key, '')
        if not val: return None
        try: return ast.literal_eval(val)
        except: return None
        
    def get_kwargs(keys):
        return {k: params[k] for k in keys if k in params and str(params[k]).strip() != ""}

    try:
        rotor = build_rotor_from_ui(data)
        
        conversion_type = data.get('conversion_type', '')
        if conversion_type == '4dof':
            rotor = rs.utils.convert_6dof_to_4dof(rotor)
        elif conversion_type == 'torsional':
            rotor = rs.utils.convert_6dof_to_torsional(rotor)

        dofs_per_node = rotor.ndof // len(rotor.nodes)
        
        a_hash = get_analysis_hash(data)
        
        if len(ANALYSIS_CACHE) > 15:
            ANALYSIS_CACHE.pop(next(iter(ANALYSIS_CACHE)))
            
        fig = None
        
        if analysis_type == 'campbell':
            s_min = float(params.get('speed_min', 0.0))
            s_max = float(params.get('speed_max', 400.0))
            s_steps = int(float(params.get('speed_steps', 50)))
            plot_type = params.get('plot_type', 'Default')
            
            if plot_type == 'Mode Shape' and s_steps > 15:
                s_steps = 15
                
            speed_rads = np.linspace(s_min, s_max, s_steps)
            ana_kwargs = {'frequencies': int(float(params.get('frequencies', 6))), 
                          'frequency_type': params.get('frequency_type', 'wd'), 
                          'torsional_analysis': get_bool('torsional_analysis')}
            
            if a_hash in ANALYSIS_CACHE:
                camp = ANALYSIS_CACHE[a_hash]
            else:
                camp = rotor.run_campbell(speed_rads, **ana_kwargs)
                ANALYSIS_CACHE[a_hash] = camp
            
            plot_kwargs = get_kwargs(['frequency_units', 'speed_units', 'damping_parameter'])
            if get_list('harmonics'): plot_kwargs['harmonics'] = get_list('harmonics')
            
            if plot_type == 'Mode Shape':
                import threading
                import webbrowser
                import sys
                import re
                
                plot_kwargs['animation'] = get_bool('animation')
                
                if not hasattr(sys.stdout, '_is_dash_catcher'):
                    class DashURLCatcher:
                        def __init__(self, orig):
                            self.orig = orig
                            self._is_dash_catcher = True
                        def write(self, msg):
                            self.orig.write(msg)
                            if "Dash is running on" in msg or "Running on http" in msg:
                                match = re.search(r'(http://[0-9\.]+:\d+)', msg)
                                if match:
                                    url = match.group(1)
                                    if "5001" not in url: 
                                        webbrowser.open(url)
                        def flush(self):
                            self.orig.flush()
                    
                    sys.stdout = DashURLCatcher(sys.stdout)

                def run_dash_thread(camp_obj, p_kw):
                    try:
                        camp_obj.plot_with_mode_shape(**p_kw)
                    except Exception as e:
                        print(f"Dash Campbell Error: {e}")
                        
                t = threading.Thread(target=run_dash_thread, args=(camp, plot_kwargs))
                t.daemon = True
                t.start()
                
                return jsonify({"status": "info", "message": "Interactive Campbell Diagram launched!<br><br>A new Dash server is running and should open automatically in a new browser tab."})
            else:
                fig = camp.plot(**plot_kwargs)
            
        elif analysis_type == 'ucs':
            k_min = float(params.get('k_min', 4))
            k_max = float(params.get('k_max', 10))
            num_modes = int(float(params.get('num_modes', 4)))
            
            ana_kwargs = {'synchronous': get_bool('synchronous')}
            if get_list('bearing_frequency_range'): ana_kwargs['bearing_frequency_range'] = get_list('bearing_frequency_range')
            
            if a_hash in ANALYSIS_CACHE:
                ucs_res = ANALYSIS_CACHE[a_hash]
            else:
                ucs_res = rotor.run_ucs(stiffness_range=(k_min, k_max), num=50, num_modes=num_modes, **ana_kwargs)
                ANALYSIS_CACHE[a_hash] = ucs_res

            plot_kwargs = get_kwargs(['stiffness_units', 'frequency_units'])
            fig = ucs_res.plot(**plot_kwargs)
            
        elif analysis_type == 'freq_response':
            s_min = float(params.get('speed_min', 0.0))
            s_max = float(params.get('speed_max', 400.0))
            speed_rads = np.linspace(s_min, s_max, 50)
            
            ana_kwargs = {'free_free': get_bool('free_free')}
            if get_list('modes'): ana_kwargs['modes'] = get_list('modes')
            
            if a_hash in ANALYSIS_CACHE:
                response = ANALYSIS_CACHE[a_hash]
            else:
                response = rotor.run_freq_response(speed_rads, **ana_kwargs)
                ANALYSIS_CACHE[a_hash] = response

            plot_type = params.get('plot_type', 'Default')
            plot_method = {
                'Default': 'plot', 'Magnitude': 'plot_magnitude', 
                'Phase': 'plot_phase', 'Polar Bode': 'plot_polar_bode'
            }.get(plot_type, 'plot')

            plot_kwargs = get_kwargs(['frequency_units', 'amplitude_units'])
            if plot_type in ['Default', 'Phase', 'Polar Bode']: plot_kwargs.update(get_kwargs(['phase_units']))
            if plot_type == 'Magnitude': plot_kwargs.update(get_kwargs(['line_shape']))

            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            
            inps = params.get('inps', [{'node': 0, 'dof': 0}])
            outs = params.get('outs', [{'node': 0, 'dof': 0}])
            max_len = max(len(inps), len(outs))
            inps_padded = inps + [inps[-1]] * (max_len - len(inps)) if inps else [{'node':0,'dof':0}]*max_len
            outs_padded = outs + [outs[-1]] * (max_len - len(outs)) if outs else [{'node':0,'dof':0}]*max_len
            
            for i in range(max_len):
                inp, out = inps_padded[i], outs_padded[i]
                g_inp = inp['node'] * dofs_per_node + inp['dof']
                g_out = out['node'] * dofs_per_node + out['dof']
                
                fig_i = getattr(response, plot_method)(inp=g_inp, out=g_out, **plot_kwargs)
                current_color = colors[i % len(colors)]
                
                for j, trace in enumerate(fig_i.data):
                    trace.name = f"In(N{inp['node']} D{inp['dof']}) | Out(N{out['node']} D{out['dof']})"
                    trace.legendgroup = f"group_{i}"
                    trace.showlegend = (j == 0) 
                    if hasattr(trace, 'line') and trace.line is not None: trace.line.color = current_color
                    
                if fig is None: fig = fig_i 
                else: fig.add_traces(fig_i.data)
                
        elif analysis_type == 'modes':
            plot_type = params.get('plot_type', '2D')
            idx = int(float(params.get('plot_idx', 0)))
            
            ana_kwargs = {'sparse': get_bool('sparse', True), 'synchronous': get_bool('synchronous')}
            
            if a_hash in ANALYSIS_CACHE:
                modal_res = ANALYSIS_CACHE[a_hash]
            else:
                modal_res = rotor.run_modal(speed=float(params.get('speed', 0.0)), num_modes=int(float(params.get('num_modes', 12))), **ana_kwargs)
                ANALYSIS_CACHE[a_hash] = modal_res
            
            if plot_type == '3D':
                plot_kwargs = get_kwargs(['frequency_type', 'length_units', 'phase_units', 'frequency_units', 'damping_parameter'])
                plot_kwargs['animation'] = get_bool('animation')
                fig = modal_res.plot_mode_3d(idx, **plot_kwargs)
            elif plot_type == 'Orbit':
                plot_kwargs = {}
                if get_list('nodes'): plot_kwargs['nodes'] = get_list('nodes')
                fig = modal_res.plot_orbit(idx, **plot_kwargs)
            else:
                plot_kwargs = get_kwargs(['orientation', 'frequency_type', 'frequency_units', 'damping_parameter'])
                fig = modal_res.plot_mode_2d(idx, **plot_kwargs)
            
        elif analysis_type == 'unbalance':
            s_min = float(params.get('speed_min', 0.0))
            s_max = float(params.get('speed_max', 400.0))
            speed_rads = np.linspace(s_min, s_max, 50)
            
            unbalances = params.get('unbalances', [{'node': 0, 'mag': 0.01, 'phase': 0.0}])
            nodes = [min(int(u.get('node', 0)), len(rotor.nodes) - 1) for u in unbalances]
            mags = [float(u.get('mag', 0.01)) for u in unbalances]
            phases = [float(u.get('phase', 0.0)) for u in unbalances]
            
            ana_kwargs = {}
            if get_list('modes'): ana_kwargs['modes'] = get_list('modes')

            if a_hash in ANALYSIS_CACHE:
                response = ANALYSIS_CACHE[a_hash]
            else:
                response = rotor.run_unbalance_response(
                    node=nodes, unbalance_magnitude=mags, unbalance_phase=phases, frequency=speed_rads, **ana_kwargs
                )
                ANALYSIS_CACHE[a_hash] = response

            probes = params.get('probes', [{'node': 0, 'angle': 0.0}])
            probe_objects = [rs.Probe(int(p['node']), float(p.get('angle', 0.0))) for p in probes]
            
            plot_type = params.get('plot_type', 'Default')
            plot_method = {
                'Default': 'plot', 'Magnitude': 'plot_magnitude', 
                'Phase': 'plot_phase', 'Bode': 'plot_bode', 'Polar Bode': 'plot_polar_bode'
            }.get(plot_type, 'plot')
            
            plot_kwargs = get_kwargs(['probe_units', 'frequency_units', 'amplitude_units'])
            if plot_type in ['Default', 'Phase', 'Bode', 'Polar Bode']: plot_kwargs.update(get_kwargs(['phase_units']))
            if plot_type == 'Magnitude': plot_kwargs.update(get_kwargs(['line_shape']))

            if a_hash in ANALYSIS_CACHE:
                response = ANALYSIS_CACHE[a_hash]
            else:
                response = rotor.run_unbalance_response(
                    node=nodes, unbalance_magnitude=mags, unbalance_phase=phases, frequency=speed_rads, **ana_kwargs
                )
                ANALYSIS_CACHE[a_hash] = response

            fig = getattr(response, plot_method)(probe=probe_objects, **plot_kwargs)
            
        elif analysis_type in ['time_response', 'misalignment', 'rubbing', 'crack']:
            
            if analysis_type == 'time_response':
                speed = float(params.get('speed', 100))
                t_max = float(params.get('t_max', 1.0))
                steps = int(float(params.get('steps', 1000)))
                t_arr = np.linspace(0, t_max, steps)
                F = np.zeros((len(t_arr), rotor.ndof))
                forces = params.get('forces', [])
                safe_dict = {'t': t_arr, 'np': np, 'speed': speed, 'sin': np.sin, 'cos': np.cos, 'pi': np.pi}

                for f in forces:
                    n_f, dof_f = min(int(f.get('node', 0)), len(rotor.nodes) - 1), int(f.get('dof', 0))
                    g_dof = n_f * dofs_per_node + dof_f
                    try: F[:, g_dof] += eval(str(f.get('func', '0')), {"__builtins__": None}, safe_dict)
                    except Exception: pass

                ana_kwargs = {'method': params.get('method', 'default')}
                
                if a_hash in ANALYSIS_CACHE: response = ANALYSIS_CACHE[a_hash]
                else:
                    response = rotor.run_time_response(speed, F, t_arr, **ana_kwargs)
                    ANALYSIS_CACHE[a_hash] = response

            elif analysis_type == 'misalignment':
                speed = float(params.get('speed', 125.66))
                t_arr = np.linspace(float(params.get('t_initial', 0.0)), float(params.get('t_final', 0.5)), int(float(params.get('t_steps', 5000))))
                
                unbalances = params.get('unbalances', [{'node': 7, 'mag': 5e-4, 'phase': -1.57}])
                nodes = [int(u.get('node', 0)) for u in unbalances]
                mags = [float(u.get('mag', 0.0)) for u in unbalances]
                phases = [float(u.get('phase', 0.0)) for u in unbalances]
                
                coupling = params.get('coupling', 'flex')
                ana_kwargs = {'coupling': coupling}
                
                if 'n' in params and str(params['n']).strip(): ana_kwargs['n'] = int(float(params['n']))
                if 'input_torque' in params and str(params['input_torque']).strip(): ana_kwargs['input_torque'] = float(params['input_torque'])
                if 'load_torque' in params and str(params['load_torque']).strip(): ana_kwargs['load_torque'] = float(params['load_torque'])
                
                if coupling == 'flex':
                    if 'mis_type' in params: ana_kwargs['mis_type'] = params['mis_type']
                    if 'mis_distance_x' in params and str(params['mis_distance_x']).strip(): ana_kwargs['mis_distance_x'] = float(params['mis_distance_x'])
                    if 'mis_distance_y' in params and str(params['mis_distance_y']).strip(): ana_kwargs['mis_distance_y'] = float(params['mis_distance_y'])
                    if 'mis_angle' in params and str(params['mis_angle']).strip(): ana_kwargs['mis_angle'] = float(params['mis_angle'])
                    if 'radial_stiffness' in params and str(params['radial_stiffness']).strip(): ana_kwargs['radial_stiffness'] = float(params['radial_stiffness'])
                    if 'bending_stiffness' in params and str(params['bending_stiffness']).strip(): ana_kwargs['bending_stiffness'] = float(params['bending_stiffness'])
                else:
                    if 'mis_distance' in params and str(params['mis_distance']).strip(): ana_kwargs['mis_distance'] = float(params['mis_distance'])
                    
                if a_hash in ANALYSIS_CACHE: response = ANALYSIS_CACHE[a_hash]
                else:
                    response = rotor.run_misalignment(node=nodes, unbalance_magnitude=mags, unbalance_phase=phases, speed=speed, t=t_arr, **ana_kwargs)
                    ANALYSIS_CACHE[a_hash] = response

            elif analysis_type == 'rubbing':
                speed = float(params.get('speed', 125.66))
                t_arr = np.linspace(float(params.get('t_initial', 0.0)), float(params.get('t_final', 0.5)), int(float(params.get('t_steps', 5000))))
                
                unbalances = params.get('unbalances', [{'node': 7, 'mag': 5e-4, 'phase': -1.57}])
                nodes = [int(u.get('node', 0)) for u in unbalances]
                mags = [float(u.get('mag', 0.0)) for u in unbalances]
                phases = [float(u.get('phase', 0.0)) for u in unbalances]
                
                n_val = int(float(params.get('n', 0)))
                distance = float(params.get('distance', 0))
                c_stiff = float(params.get('contact_stiffness', 0))
                c_damp = float(params.get('contact_damping', 0))
                f_coeff = float(params.get('friction_coeff', 0))
                torque = get_bool('torque', False)
                
                if a_hash in ANALYSIS_CACHE: response = ANALYSIS_CACHE[a_hash]
                else:
                    response = rotor.run_rubbing(n=n_val, distance=distance, contact_stiffness=c_stiff, contact_damping=c_damp, friction_coeff=f_coeff, node=nodes, unbalance_magnitude=mags, unbalance_phase=phases, speed=speed, t=t_arr, torque=torque)
                    ANALYSIS_CACHE[a_hash] = response

            elif analysis_type == 'crack':
                speed = float(params.get('speed', 125.66))
                t_arr = np.linspace(float(params.get('t_initial', 0.0)), float(params.get('t_final', 0.5)), int(float(params.get('t_steps', 5000))))
                
                unbalances = params.get('unbalances', [{'node': 7, 'mag': 5e-4, 'phase': -1.57}])
                nodes = [int(u.get('node', 0)) for u in unbalances]
                mags = [float(u.get('mag', 0.0)) for u in unbalances]
                phases = [float(u.get('phase', 0.0)) for u in unbalances]
                
                n_val = int(float(params.get('n', 0)))
                depth_ratio = float(params.get('depth_ratio', 0))
                crack_model = params.get('crack_model', 'Mayes')
                
                ana_kwargs = {'crack_model': crack_model}
                if 'cross_divisions' in params and str(params['cross_divisions']).strip() != '':
                    ana_kwargs['cross_divisions'] = int(float(params['cross_divisions']))
                    
                if a_hash in ANALYSIS_CACHE: response = ANALYSIS_CACHE[a_hash]
                else:
                    response = rotor.run_crack(n=n_val, depth_ratio=depth_ratio, node=nodes, unbalance_magnitude=mags, unbalance_phase=phases, speed=speed, t=t_arr, **ana_kwargs)
                    ANALYSIS_CACHE[a_hash] = response

            probes = params.get('probes', [{'node': 0, 'angle': 0.0}])
            probe_objects = [rs.Probe(int(p['node']), float(p.get('angle', 0.0))) for p in probes]
            node_2d = probes[0]['node'] if probes else 0

            plot_type = params.get('plot_type', '1D')

            if plot_type == 'Frequency (DFFT)':
                plot_kwargs = get_kwargs(['probe_units', 'displacement_units', 'frequency_units'])
                fig = response.plot_dfft(probe=probe_objects, **plot_kwargs)
            elif plot_type == '2D':
                plot_kwargs = get_kwargs(['displacement_units'])
                fig = response.plot_2d(node=node_2d, **plot_kwargs)
            elif plot_type == '3D':
                plot_kwargs = get_kwargs(['displacement_units', 'rotor_length_units'])
                fig = response.plot_3d(**plot_kwargs)
            else:
                plot_kwargs = get_kwargs(['probe_units', 'displacement_units', 'time_units'])
                fig = response.plot_1d(probe=probe_objects, **plot_kwargs)

        elif analysis_type == 'static':
            plot_type = params.get('plot_type', 'Free Body Diagram')
            
            if a_hash in ANALYSIS_CACHE:
                static_res = ANALYSIS_CACHE[a_hash]
            else:
                static_res = rotor.run_static()
                ANALYSIS_CACHE[a_hash] = static_res
            
            if plot_type == 'Deformation':
                plot_kwargs = get_kwargs(['deformation_units', 'rotor_length_units'])
                fig = static_res.plot_deformation(**plot_kwargs)
            elif plot_type == 'Shearing Force':
                plot_kwargs = get_kwargs(['force_units', 'rotor_length_units'])
                fig = static_res.plot_shearing_force(**plot_kwargs)
            elif plot_type == 'Bending Moment':
                plot_kwargs = get_kwargs(['moment_units', 'rotor_length_units'])
                fig = static_res.plot_bending_moment(**plot_kwargs)
            else:
                plot_kwargs = get_kwargs(['force_units', 'rotor_length_units'])
                fig = static_res.plot_free_body_diagram(**plot_kwargs)

        elif analysis_type == 'harmonic_balance':
            speed = float(params.get('speed', 200))
            t_ini = float(params.get('t_initial', 0.0))
            t_fin = float(params.get('t_final', 0.5))
            t_steps = int(float(params.get('t_steps', 1001)))
            t_hb = np.linspace(t_ini, t_fin, t_steps)
            
            hb_node = int(float(params.get('hb_node', 0)))
            hb_mags = get_list('hb_magnitudes') or [2000.0]
            hb_phases = get_list('hb_phases') or [0.0]
            hb_harmonics = get_list('hb_harmonics') or [1]
            
            h_forces = [{
                'node': hb_node,
                'magnitudes': hb_mags,
                'phases': hb_phases,
                'harmonics': hb_harmonics
            }]
            
            ana_kwargs = {
                'gravity': get_bool('gravity', False),
                'n_harmonics': int(float(params.get('n_harmonics', 1)))
            }
            
            if a_hash in ANALYSIS_CACHE:
                hb_res = ANALYSIS_CACHE[a_hash]
            else:
                hb_res = rotor.run_harmonic_balance_response(speed, t_hb, h_forces, **ana_kwargs)
                ANALYSIS_CACHE[a_hash] = hb_res
                
            probes = params.get('probes', [{'node': 0, 'angle': 0.0}])
            probe_objects = [rs.Probe(int(p['node']), float(p.get('angle', 0.0))) for p in probes]
            
            plot_kwargs = get_kwargs(['amplitude_units', 'frequency_units'])
            fig = hb_res.plot(probe=probe_objects, **plot_kwargs)

        elif analysis_type == 'clearance':
            speed = float(params.get('speed', 600))
            node = int(float(params.get('node', 0)))
            unb_mag = get_list('unbalance_magnitude') or [0.05]
            unb_phase = get_list('unbalance_phase') or [0.0]
            
            ana_kwargs = {}
            if get_list('frequency'): ana_kwargs['frequency'] = get_list('frequency')
            if get_list('modes'): ana_kwargs['modes'] = get_list('modes')
            
            if a_hash in ANALYSIS_CACHE:
                clr_res = ANALYSIS_CACHE[a_hash]
            else:
                clr_res = rotor.run_clearance_analysis(speed, node, unb_mag, unb_phase, **ana_kwargs)
                ANALYSIS_CACHE[a_hash] = clr_res
                
            fig = clr_res.plot()
                
        else: raise ValueError("Analysis not implemented yet.")
        
        layout_update = dict(margin=dict(l=60, r=60, t=50, b=100))
        if analysis_type == 'campbell':
            layout_update['legend'] = dict(yanchor="top", y=-0.25)
            
        fig.update_layout(**layout_update)
        
        return jsonify({"status": "success", "plot_json": json.dumps(remove_nans(decode_bdata(fig.to_dict())), cls=NumpyEncoder)})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/shutdown', methods=['POST'])
def shutdown():
    os._exit(0)
    return jsonify({"status": "success"})

@app.route('/load_ross_file', methods=['POST'])
def load_ross_file():
    try:
        content = request.json.get('content', '')
        
        try:
            ross_data = toml.loads(content)
        except:
            ross_data = json.loads(content)

        project = {
            'materials': [], 'shafts': [], 'disks': [], 'gears': [],
            'couplings': [], 'seals': [], 'bearings': [], 'pointmasses': []
        }
        seen_materials = set()

        class_map = {
            'ShaftElement': ('shafts', 'BASIC'),
            'DiskElement': ('disks', 'BASIC'),
            'GearElement': ('gears', 'BASIC'),
            'GearElementTVMS': ('gears', 'TVMS'),
            'BearingElement': ('bearings', 'BASIC'),
            'BallBearingElement': ('bearings', 'BallBearing'),
            'RollerBearingElement': ('bearings', 'RollerBearing'),
            'MagneticBearingElement': ('bearings', 'MagneticBearing'),
            'CylindricalBearing': ('bearings', 'Cylindrical'),
            'PlainJournal': ('bearings', 'PlainJournal'),
            'SqueezeFilmDamper': ('bearings', 'SqueezeFilm'),
            'ThrustPad': ('bearings', 'ThrustPad'),
            'TiltingPad': ('bearings', 'TiltingPad'),
            'SealElement': ('seals', 'BASIC'),
            'HolePatternSeal': ('seals', 'HolePattern'),
            'LabyrinthSeal': ('seals', 'Labyrinth'),
            'HybridSeal': ('seals', 'Hybrid'),
            'CouplingElement': ('couplings', 'BASIC'),
            'PointMass': ('pointmasses', 'BASIC')
        }

        for key, val in ross_data.items():
            if key == 'parameters' or not isinstance(val, dict): 
                continue
            
            class_name = key.split('_')[0]
            
            if class_name in class_map:
                tab, type_val = class_map[class_name]
                item = {'element_type': type_val}
                
                for k, v in val.items():
                    if k == 'material' and isinstance(v, dict):
                        mat_name = v.get('name', 'CustomMaterial')
                        item['material'] = mat_name
                        
                        if mat_name not in seen_materials:
                            seen_materials.add(mat_name)
                            mat_obj = {'element_type': 'BASIC', 'name': mat_name}
                            unit_map_mat = UNITS_MAPPING.get('Material', {})
                            
                            for mk, mv in v.items():
                                if mk != 'name': 
                                    if mk in unit_map_mat and mv is not None:
                                        try:
                                            tgt_unit = unit_map_mat[mk]
                                            base_u = Q_(1, tgt_unit).to_base_units().units
                                            if isinstance(mv, list):
                                                mv = [float(Q_(float(x), base_u).to(tgt_unit).m) for x in mv]
                                            else:
                                                mv = float(Q_(float(mv), base_u).to(tgt_unit).m)
                                        except Exception: pass
                                        
                                    mat_obj[mk] = str(mv)
                            project['materials'].append(mat_obj)
                            
                    else:
                        unit_map = UNITS_MAPPING.get(class_name, {})
                        
                        if k in unit_map and v is not None:
                            try:
                                tgt_unit = unit_map[k]
                                base_u = Q_(1, tgt_unit).to_base_units().units
                                if isinstance(v, list):
                                    v = [float(Q_(float(x), base_u).to(tgt_unit).m) for x in v]
                                else:
                                    v = float(Q_(float(v), base_u).to(tgt_unit).m)
                            except Exception:
                                pass
                                
                        if isinstance(v, list): item[k] = str(v)
                        else: item[k] = str(v)
                        
                project[tab].append(item)

        return jsonify({"status": "success", "projectData": project})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    import threading
    import webbrowser
    import sys
    import time
    def open_interface():
        time.sleep(2.0) 
        if getattr(sys, "frozen", False):
            base_folder = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
        else:
            base_folder = os.path.dirname(os.path.abspath(__file__))
        webbrowser.open('file://' + os.path.join(base_folder, 'frontend', 'index.html'))
    threading.Thread(target=open_interface).start()
    app.run(port=5001, use_reloader=False)