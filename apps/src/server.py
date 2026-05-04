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

try:
    from ross.units import Q_
except ImportError:
    try:
        from ross import Q_
    except ImportError:
        import pint
        ureg = pint.UnitRegistry()
        Q_ = ureg.Quantity

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

def extract_kwargs(d, mat_dict, element_type, ignore_keys=['element_type', 'n']):
    kwargs = {}
    int_keys = ['n_pad', 'n_theta', 'n_radial', 'n_teeth', 'n_rollers', 'n_balls', 'nx', 'nz', 'nr_pad', 'max_inlet_iterations', 'max_jtemp_iter', 'max_iterations', 'elements_circumferential', 'elements_axial', 'n_link', 'n_l', 'n_r']    
    
    unit_map = UNITS_MAPPING.get(element_type, {})
    
    for k, v in d.items():
        if k in ignore_keys: continue
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
                        if k in unit_map:
                            kwargs[k] = Q_(np.array(val_parsed), unit_map[k]).to_base_units()
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
                if k in int_keys:
                    val_num = int(val_num)
                
                if k in unit_map:
                    kwargs[k] = Q_(val_num, unit_map[k]).to_base_units()
                else:
                    kwargs[k] = val_num
                continue
            except ValueError:
                kwargs[k] = v_strip
                continue
        else:
            kwargs[k] = v
            
    return kwargs

# Memory Cache

ROSS_CACHE = {}

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
    try:
        rotor = build_rotor_from_ui(data)
        dofs_per_node = rotor.ndof // len(rotor.nodes)
        
        if analysis_type == 'campbell':
            s_min = float(params.get('speed_min', 0.0))
            s_max = float(params.get('speed_max', 400.0))
            s_steps = int(float(params.get('speed_steps', 50)))
            speed_rads = np.linspace(s_min, s_max, s_steps)
            fig = rotor.run_campbell(speed_rads).plot()
            
        elif analysis_type == 'ucs':
            k_min = float(params.get('k_min', 4))
            k_max = float(params.get('k_max', 10))
            num_modes = int(float(params.get('num_modes', 4)))
            fig = rotor.run_ucs(stiffness_range=(k_min, k_max), num=50, num_modes=num_modes).plot()
            
        elif analysis_type == 'freq_response':
            s_min = float(params.get('speed_min', 0.0))
            s_max = float(params.get('speed_max', 400.0))
            speed_rads = np.linspace(s_min, s_max, 50)
            
            inps = params.get('inps', [{'node': 0, 'dof': 0}])
            outs = params.get('outs', [{'node': 0, 'dof': 0}])
            
            response = rotor.run_freq_response(speed_rads)
            fig = None
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            
            max_len = max(len(inps), len(outs))
            inps_padded = inps + [inps[-1]] * (max_len - len(inps)) if inps else [{'node':0,'dof':0}]*max_len
            outs_padded = outs + [outs[-1]] * (max_len - len(outs)) if outs else [{'node':0,'dof':0}]*max_len
            
            for i in range(max_len):
                inp = inps_padded[i]
                out = outs_padded[i]
                g_inp = inp['node'] * dofs_per_node + inp['dof']
                g_out = out['node'] * dofs_per_node + out['dof']
                
                fig_i = response.plot(inp=g_inp, out=g_out)
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
            modal_res = rotor.run_modal(speed=float(params.get('speed', 0.0)), num_modes=int(float(params.get('num_modes', 12))))
            
            if plot_type == '3D':
                fig = modal_res.plot_mode_3d(int(float(params.get('plot_idx', 0))))
            else:
                fig = modal_res.plot_mode_2d(int(float(params.get('plot_idx', 0))))
            
        elif analysis_type == 'unbalance':
            s_min = float(params.get('speed_min', 0.0))
            s_max = float(params.get('speed_max', 400.0))
            speed_rads = np.linspace(s_min, s_max, 50)
            
            unbalances = params.get('unbalances', [{'node': 0, 'mag': 0.01, 'phase': 0.0}])
            nodes = [int(u.get('node', 0)) for u in unbalances]
            mags = [float(u.get('mag', 0.01)) for u in unbalances]
            phases = [float(u.get('phase', 0.0)) for u in unbalances]
            
            max_node = len(rotor.nodes) - 1
            nodes = [min(n, max_node) for n in nodes]
            
            probes = params.get('probes', [{'node': 0, 'dof': 0}])
            probe_tuples = [(int(p['node']), int(p['dof'])) for p in probes]
            
            fig = rotor.run_unbalance_response(
                node=nodes, 
                unbalance_magnitude=mags, 
                unbalance_phase=phases, 
                frequency=speed_rads
            ).plot(probe=probe_tuples) 
            
        elif analysis_type == 'time_response':
            speed = float(params.get('speed', 100))
            t_max = float(params.get('t_max', 1.0))
            steps = int(float(params.get('steps', 1000)))
            plot_type = params.get('plot_type', '1D')

            t = np.linspace(0, t_max, steps)
            F = np.zeros((len(t), rotor.ndof))
            
            forces = params.get('forces', [])
            
            safe_dict = {
                't': t, 'np': np, 'speed': speed, 
                'sin': np.sin, 'cos': np.cos, 'pi': np.pi
            }

            for f in forces:
                n = int(f.get('node', 0))
                dof = int(f.get('dof', 0))
                func_str = str(f.get('func', '0'))

                if n >= len(rotor.nodes):
                    n = len(rotor.nodes) - 1

                g_dof = n * dofs_per_node + dof
                
                try:
                    F_val = eval(func_str, {"__builtins__": None}, safe_dict)
                    F[:, g_dof] += F_val
                except Exception as e:
                    print(f"Error evaluating force '{func_str}': {e}")

            response = rotor.run_time_response(speed, F, t)
            
            probes = params.get('probes', [{'node': 0, 'dof': 0}])
            probe_tuples = [(p['node'], p['dof']) for p in probes]

            node_2d = probes[0]['node'] if probes else 0

            if plot_type == 'Frequency (DFFT)':
                fig = response.plot_dfft(probe=probe_tuples)
            elif plot_type == '2D':
                fig = response.plot_2d(node=node_2d)
            elif plot_type == '3D':
                fig = response.plot_3d()
            else:
                fig = response.plot_1d(probe=probe_tuples)

        elif analysis_type == 'static':
            plot_type = params.get('plot_type', 'Free Body Diagram')
            static_res = rotor.run_static()
            
            if plot_type == 'Deformation':
                fig = static_res.plot_deformation()
            elif plot_type == 'Shearing Force':
                fig = static_res.plot_shearing_force()
            elif plot_type == 'Bending Moment':
                fig = static_res.plot_bending_moment()
            else:
                fig = static_res.plot_free_body_diagram()
                
        else: raise ValueError("Analysis not implemented yet.")
            
        fig.update_layout(margin=dict(l=60, r=60, t=50, b=50))
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
                            for mk, mv in v.items():
                                if mk != 'name': 
                                    mat_obj[mk] = str(mv)
                            project['materials'].append(mat_obj)
                    else:
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
        if getattr(sys, 'frozen', False): base_folder = os.path.dirname(sys.executable)
        else: base_folder = os.path.dirname(os.path.abspath(__file__))
        webbrowser.open('file://' + os.path.join(base_folder, 'index.html'))
    threading.Thread(target=open_interface).start()
    app.run(port=5001, use_reloader=False)