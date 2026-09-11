# -*- coding: utf-8 -*-
"""Generate the Python script that reproduces, outside the interface, what the
screen shows.

This generator used to live in frontend/app.js, assembled from the DOM. That
forced the JavaScript to keep its own copies of three things the backend
already knows -- node numbering, the unit map and the ROSS class names -- and
the copies drifted apart silently: the chart came out of one path and the
exported script out of another. Here the script is born from the same sources
that build the rotor.

The port is literal by decision: the old function and the new one are compared
output against output in the tests. The intentional differences are listed
in DEVIATIONS.
"""

import re
from decimal import Decimal

from .element_registry import ross_class_name
from .node_resolver import effective_nodes
import textwrap

from .schema import unit_map_by_class

# Deliberate differences from the old app.js generator:
#   1. material names go into the script as an escaped literal. Before, a
#      material named O'Brien produced a .py file that would not even open.
#   2. values coming from a <select> are escaped too. For any normal value
#      the output is byte for byte the same.
#   3. an explicitly null field counts as absent. Before it became the word
#      `null` in the middle of the script -- a NameError on first run.
DEVIATIONS = ("material_name_escaping", "select_value_escaping", "null_as_missing")

_DECIMAL = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
_RADIX = re.compile(r"^(?:0[xX][0-9a-fA-F]+|0[oO][0-7]+|0[bB][01]+)$")


def _js_is_nan(text):
    """Mirror JavaScript's isNaN(text), which converts before testing.

    Number() trims whitespace and accepts '', hexadecimal and Infinity.
    Python's float() accepts 'nan', 'inf' and '1_0', which JavaScript refuses.
    Without this function, '1_0' would become a number here and a string in the
    old generator.
    """
    rendered = text.strip()
    if rendered == "":
        return False
    if rendered in ("Infinity", "+Infinity", "-Infinity"):
        return False
    if _RADIX.match(rendered):
        return False
    return not _DECIMAL.match(rendered)


def _js_number(value):
    """Format a float the way JavaScript's String(number) would.

    Both use the shortest digits that round-trip to the same float, but they
    switch to exponential notation at different points: Python from 1e-5 and
    1e16, JavaScript only below 1e-6 and from 1e21 up. And the exponent comes
    out without a leading zero: 1e-7, not 1e-07.
    """
    if value != value:
        return "NaN"
    if value == float("inf"):
        return "Infinity"
    if value == float("-inf"):
        return "-Infinity"
    if value == int(value) and abs(value) < 1e21:
        return str(int(value))

    number = Decimal(repr(float(value)))
    if -7 < number.adjusted() < 21:
        return format(number, "f")

    sign, digits, exponent = number.normalize().as_tuple()
    body = "".join(str(d) for d in digits)
    power = len(body) - 1 + exponent
    mantissa = body[0] + ("." + body[1:] if len(body) > 1 else "")
    return "%s%se%s%d" % (
        "-" if sign else "",
        mantissa,
        "+" if power >= 0 else "-",
        abs(power),
    )


def _js_str(value):
    """Convert the way JavaScript's `${value}` interpolation would."""
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "null"
    if isinstance(value, float):
        return _js_number(value)
    if isinstance(value, (list, tuple)):
        return ",".join("" if item is None else _js_str(item) for item in value)
    if isinstance(value, dict):
        return "[object Object]"
    return str(value)


def _js_truthy(value):
    """Mirror JavaScript's `if (value)`: '0' is truthy, 0 is not."""
    if value is None or value is False:
        return False
    if isinstance(value, str):
        return value != ""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == value and value != 0
    return True


def _or(value, default):
    """The equivalent of JavaScript's `value || fallback`."""
    return value if _js_truthy(value) else default


def _py_string(value):
    """Escape text into a valid Python string literal."""
    return "'" + _js_str(value).replace("\\", "\\\\").replace("'", "\\'") + "'"


def _format_kwargs(obj, exclude_keys=(), class_name=""):
    """Build `key=value, ...` from what the form returned.

    The unit map comes from the schema, inherited units already resolved: a
    BallBearingElement's cxx is declared on BearingElement.
    """
    items = []
    unit_map = unit_map_by_class().get(class_name or "", {})
    for key, val in obj.items():
        if key in exclude_keys or key.endswith("_unit"):
            continue
        if val is None or (isinstance(val, str) and val == ""):
            continue

        unit = _or(obj.get(key + "_unit"), unit_map.get(key))
        is_text = isinstance(val, str)
        rendered = val.strip() if is_text else val
        # '(' is on the list: without it, initial_position=(0.1, -0.1) became a
        # string and ROSS got text where it expects a tuple.
        is_literal = is_text and rendered[:1] in ("[", "{", "(")

        if is_text and _js_is_nan(val) and not is_literal:
            lowered = rendered.lower()
            if lowered == "true":
                items.append("%s=True" % key)
            elif lowered == "false":
                items.append("%s=False" % key)
            else:
                items.append("%s=%s" % (key, _py_string(val)))
        else:
            final = _js_str(val)
            if is_text and rendered.startswith("[") and _js_truthy(unit):
                final = "np.array(%s)" % _js_str(val)
            if _js_truthy(unit):
                items.append("%s=Q_(%s, '%s')" % (key, final, unit))
            else:
                items.append("%s=%s" % (key, final))
    return ", ".join(items)


def _with_node_arg(element, args, effective_node):
    """Prefix n= when the element has no explicit node.

    This used to test args.includes('n='), which matched any key ending in 'n'
    -- cavitation=, gas_composition=, orientation= -- and the element came out
    without a node.
    """
    raw = element.get("n")
    has_explicit_node = raw is not None and str(raw).strip() != ""
    if has_explicit_node:
        return args
    return "n=%s%s" % (effective_node, (", " + args) if args else "")


def _material_expression(element, suffix):
    """Translate the form's material choice into a Python expression."""
    chosen = element.get("material")
    if not chosen or chosen == "Default (Steel)":
        return "rs.materials.steel"
    return "materials_dict%s.get(%s, default_mat%s)" % (
        suffix,
        _py_string(str(chosen).lower()),
        suffix,
    )


def _build_rotor_block(r_data, suffix):
    """Build the block that assembles a complete rotor."""
    r_data = r_data or {}
    py = "\n# --- Rotor Component %s ---\n" % suffix.upper()

    py += "materials_dict%s = {}\n" % suffix
    for material in r_data.get("materials") or []:
        copy_of = dict(material)
        if "poisson" in copy_of:
            copy_of["Poisson"] = copy_of.pop("poisson")
        args = _format_kwargs(copy_of, ["name", "element_type"], "Material")
        name = _or(copy_of.get("name"), "MaterialCustom")
        py += "materials_dict%s[%s] = rs.Material(name=%s, %s)\n" % (
            suffix,
            _py_string(str(name).lower()),
            _py_string(name),
            args,
        )
    py += (
        "default_mat{s} = list(materials_dict{s}.values())[0] "
        "if materials_dict{s} else rs.materials.steel\n"
    ).format(s=suffix)

    # Shafts
    py += "shafts_data%s = [\n" % suffix
    nodes = effective_nodes(r_data.get("shafts") or [])
    for position, shaft in enumerate(r_data.get("shafts") or []):
        args = _format_kwargs(shaft, ["material", "element_type"], "ShaftElement")
        args = _with_node_arg(shaft, args, nodes[position])
        py += "    dict(%s, material=%s),\n" % (
            args,
            _material_expression(shaft, suffix),
        )
    py += "]\nshafts{s} = [rs.ShaftElement(**kwargs) for kwargs in shafts_data{s}]\n".format(
        s=suffix
    )

    # Disks
    py += "disks_data%s = [\n" % suffix
    nodes = effective_nodes(r_data.get("disks") or [])
    for position, disk in enumerate(r_data.get("disks") or []):
        args = _format_kwargs(disk, ["element_type"], "DiskElement")
        args = _with_node_arg(disk, args, nodes[position])
        py += "    dict(%s),\n" % args
    py += (
        "]\ndisks{s} = [rs.DiskElement(**kwargs) for kwargs in disks_data{s}]\n".format(
            s=suffix
        )
    )

    # Gears
    py += "gears_data%s = [\n" % suffix
    nodes = effective_nodes(r_data.get("gears") or [])
    for position, gear in enumerate(r_data.get("gears") or []):
        klass = ross_class_name("gears", gear.get("element_type"))
        args = _format_kwargs(gear, ["element_type", "material"], klass)
        args = _with_node_arg(gear, args, nodes[position])
        material = _material_expression(gear, suffix)
        full_args = (
            ("%s, material=%s" % (args, material))
            if args
            else ("material=%s" % material)
        )
        py += "    (rs.%s, dict(%s)),\n" % (klass, full_args)
    py += "]\ngears{s} = [cls(**kwargs) for cls, kwargs in gears_data{s}]\n".format(
        s=suffix
    )

    # Bearings
    py += "bearings_data%s = [\n" % suffix
    nodes = effective_nodes(r_data.get("bearings") or [])
    for position, bearing in enumerate(r_data.get("bearings") or []):
        klass = ross_class_name("bearings", bearing.get("element_type"))
        args = _format_kwargs(bearing, ["element_type"], klass)
        args = _with_node_arg(bearing, args, nodes[position])
        py += "    (rs.%s, dict(%s)),\n" % (klass, args)
    py += (
        "]\nbearings{s} = [cls(**kwargs) for cls, kwargs in bearings_data{s}]\n".format(
            s=suffix
        )
    )

    # Seals
    py += "seals_data%s = [\n" % suffix
    nodes = effective_nodes(r_data.get("seals") or [])
    for position, seal in enumerate(r_data.get("seals") or []):
        klass = ross_class_name("seals", seal.get("element_type"))
        args = _format_kwargs(seal, ["element_type"], klass)
        args = _with_node_arg(seal, args, nodes[position])
        py += "    (rs.%s, dict(%s)),\n" % (klass, args)
    py += "]\nseals{s} = [cls(**kwargs) for cls, kwargs in seals_data{s}]\n".format(
        s=suffix
    )

    # Couplings
    py += "couplings_data%s = [\n" % suffix
    for coupling in r_data.get("couplings") or []:
        py += "    dict(%s),\n" % _format_kwargs(
            coupling, ["element_type"], "CouplingElement"
        )
    py += (
        "]\ncouplings{s} = [rs.CouplingElement(**kwargs) "
        "for kwargs in couplings_data{s}]\n"
    ).format(s=suffix)

    # PointMasses
    py += "point_masses_data%s = [\n" % suffix
    nodes = effective_nodes(r_data.get("pointmasses") or [])
    for position, mass in enumerate(r_data.get("pointmasses") or []):
        args = _format_kwargs(mass, ["element_type"], "PointMass")
        args = _with_node_arg(mass, args, nodes[position])
        py += "    dict(%s),\n" % args
    py += (
        "]\npoint_masses{s} = [rs.PointMass(**kwargs) "
        "for kwargs in point_masses_data{s}]\n"
    ).format(s=suffix)

    py += (
        "\nrotor{s} = rs.Rotor(\n"
        "    shaft_elements=shafts{s} + couplings{s},\n"
        "    disk_elements=disks{s} + gears{s},\n"
        "    bearing_elements=bearings{s} + seals{s},\n"
        "    point_mass_elements=point_masses{s}\n)\n"
    ).format(s=suffix)
    return py


def _multi_rotor_block(multi_params):
    """Build the coupling between the two rotors of a MultiRotor."""
    p = multi_params or {}
    nodes = _or(p.get("coupled_nodes"), "0, 0")
    args = "rotor_driving, rotor_driven, coupled_nodes=(%s), position=%s" % (
        _js_str(nodes),
        _py_string(_or(p.get("position"), "above")),
    )

    if _js_truthy(p.get("gear_mesh_stiffness")):
        args += ", gear_mesh_stiffness=%s" % _js_str(p["gear_mesh_stiffness"])
    if p.get("update_mesh_stiffness") == "true":
        args += ", update_mesh_stiffness=True"

    svs = p.get("square_varying_stiffness")
    if _js_truthy(svs):
        args += ', square_varying_stiffness={"enable": %s, "amplitude_ratio": %s}' % (
            "True" if _js_truthy(svs.get("enable")) else "False",
            _js_str(svs.get("amplitude_ratio")),
        )

    backlash = p.get("backlash")
    if _js_truthy(backlash):
        args += (
            ', backlash={"enable": %s, "initial_value": %s, "error_amp": %s, '
            '"smooth_operator": %s, "sigma": %s}'
        ) % (
            "True" if _js_truthy(backlash.get("enable")) else "False",
            _js_str(backlash.get("initial_value")),
            _js_str(backlash.get("error_amp")),
            "True" if _js_truthy(backlash.get("smooth_operator")) else "False",
            _js_str(backlash.get("sigma")),
        )

    if _js_truthy(p.get("orientation_angle")):
        args += ", orientation_angle=%s" % _js_str(p["orientation_angle"])

    return "\n# MultiRotor Coupling\nrotor = rs.MultiRotor(%s)\n" % args


def _py_val(params, key, target_unit=None):
    """Read a dashboard field already converted to the unit ROSS expects."""
    literal = params.get(key)
    if literal is None or literal == "":
        return "0"
    unit = params.get(key + "_unit")
    if _js_truthy(unit) and target_unit:
        return "float(Q_(%s, '%s').to('%s').m)" % (_js_str(literal), unit, target_unit)
    return _js_str(literal)


def _probes_expression(params):
    probes = params.get("probes") or []
    if not probes:
        return "rs.Probe(0, 0)"
    return ", ".join(
        "rs.Probe(%s, %s)" % (_js_str(s.get("node")), _js_str(s.get("angle")))
        for s in probes
    )


def _unbalance_columns(params, mag_default="0.01", phase_default="0.0"):
    """Return (nodes, magnitudes, phases) as text, with the generator's defaults.

    The defaults are arguments because the two callers disagree on them, exactly
    as the runners do: the unbalance response starts at 0.01 kg.m and the
    clearance analysis at 0.05. Hard-coding one of them here would make the
    exported script differ from what the interface just computed.
    """
    unbalances = params.get("unbalances") or []
    if not unbalances:
        return "0", mag_default, phase_default

    def joined(kwarg):
        return ", ".join(_js_str(u.get(kwarg)) for u in unbalances)

    return joined("node"), joined("mag"), joined("phase")


def _flag(params, key, default="False"):
    """Read a boolean <select>. Anything but 'True'/'False' gives the default."""
    expected = "True" if default == "False" else "False"
    return expected if params.get(key) == expected else default


def _opt_args(params, keys):
    """Optional plotting arguments, in the order the old generator put them.

    `keys` is a list of (name, 'str'|'raw'). The argument only appears when the
    field has a value -- that is how ROSS gets to use its own default.
    """
    args = []
    for name, shape in keys:
        literal = params.get(name)
        if not _js_truthy(literal):
            continue
        args.append(
            "%s=%s"
            % (name, _py_string(literal) if shape == "str" else _js_str(literal))
        )
    return args


def _analysis_block(position, analysis):
    kind = analysis.get("type")
    p = analysis.get("params") or {}
    py = "\n# Analysis %d: %s\n" % (position + 1, str(kind).upper())

    if kind == "campbell":
        py += "speed_rads = np.linspace(%s, %s, %s)\n" % (
            _py_val(p, "speed_min", "rad/s"),
            _py_val(p, "speed_max", "rad/s"),
            _js_str(_or(p.get("speed_steps"), 50)),
        )
        py += (
            "camp_%d = rotor.run_campbell(speed_rads, frequencies=%s, frequency_type=%s, torsional_analysis=%s)\n"
            % (
                position,
                _js_str(_or(p.get("frequencies"), 6)),
                _py_string(_or(p.get("frequency_type"), "wd")),
                _flag(p, "torsional_analysis"),
            )
        )

        args = _opt_args(
            p,
            [
                ("frequency_units", "str"),
                ("speed_units", "str"),
                ("damping_parameter", "str"),
                ("harmonics", "raw"),
            ],
        )
        if p.get("plot_type") == "Mode Shape":
            if _js_truthy(p.get("animation")):
                args.append("animation=%s" % _flag(p, "animation"))
            py += "camp_%d.plot_with_mode_shape(%s).show()\n" % (
                position,
                ", ".join(args),
            )
        else:
            py += "camp_%d.plot(%s).show()\n" % (position, ", ".join(args))

    elif kind == "ucs":
        # Two fields on screen, one argument in the script -- and the script is
        # the copy that LEAVES the program. When the field came back as a pair,
        # this block still read the old single `bearing_frequency_range` and
        # would have quietly stopped emitting the argument: the interface would
        # compute one thing and the exported script another, with nobody told.
        # The clearance correction paid for that lesson once already.
        # Half a range is not handled here because it cannot arrive: the runner
        # refuses one end without the other, so no analysis of this version can
        # be saved holding it, and no analysis of an older version has these
        # fields at all. Emitting nothing is the honest reading of a pair that
        # is not there.
        low, high = p.get("bearing_freq_min"), p.get("bearing_freq_max")
        range_arg = (
            ", bearing_frequency_range=[%s, %s]" % (_js_str(low), _js_str(high))
            if str(low).strip() not in ("", "None")
            and str(high).strip() not in ("", "None")
            else ""
        )
        py += (
            "ucs_%d = rotor.run_ucs(stiffness_range=(%s, %s), num=50, num_modes=%s, synchronous=%s%s)\n"
            % (
                position,
                _js_str(p.get("k_min")),
                _js_str(p.get("k_max")),
                _js_str(p.get("num_modes")),
                _flag(p, "synchronous"),
                range_arg,
            )
        )
        args = _opt_args(p, [("stiffness_units", "str"), ("frequency_units", "str")])
        py += "ucs_%d.plot(%s).show()\n" % (position, ", ".join(args))

    elif kind == "freq_response":
        py += "speed_rads = np.linspace(%s, %s, %s)\n" % (
            _py_val(p, "speed_min", "rad/s"),
            _py_val(p, "speed_max", "rad/s"),
            _js_str(_or(p.get("speed_steps"), 50)),
        )
        modes = ", modes=%s" % _js_str(p["modes"]) if _js_truthy(p.get("modes")) else ""
        py += "freq_%d = rotor.run_freq_response(speed_rads%s, free_free=%s)\n" % (
            position,
            modes,
            _flag(p, "free_free"),
        )
        py += "dofs_per_node = rotor.number_dof\n"

        method = {
            "Magnitude": "plot_magnitude",
            "Phase": "plot_phase",
            "Polar Bode": "plot_polar_bode",
        }.get(p.get("plot_type"), "plot")

        args = _opt_args(p, [("frequency_units", "str"), ("amplitude_units", "str")])
        if p.get("plot_type") in ("Default", "Phase", "Polar Bode"):
            args += _opt_args(p, [("phase_units", "str")])
        if p.get("plot_type") == "Magnitude":
            args += _opt_args(p, [("line_shape", "str")])

        entries = p.get("inps") or [{"node": 0, "dof": 0}]
        outputs = p.get("outs") or [{"node": 0, "dof": 0}]
        py += "fig_freq_%d = None\n" % position
        py += "colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']\n"
        for j in range(max(len(entries), len(outputs))):
            entry = entries[min(j, len(entries) - 1)]
            output = outputs[min(j, len(outputs) - 1)]
            py += "g_inp = %s * dofs_per_node + %s\n" % (
                _js_str(entry.get("node")),
                _js_str(entry.get("dof")),
            )
            py += "g_out = %s * dofs_per_node + %s\n" % (
                _js_str(output.get("node")),
                _js_str(output.get("dof")),
            )
            py += "fig_temp = freq_%d.%s(inp=g_inp, out=g_out, %s)\n" % (
                position,
                method,
                ", ".join(args),
            )
            py += "for k, trace in enumerate(fig_temp.data):\n"
            py += '    trace.name = f"In(N%s D%s) | Out(N%s D%s)"\n' % (
                _js_str(entry.get("node")),
                _js_str(entry.get("dof")),
                _js_str(output.get("node")),
                _js_str(output.get("dof")),
            )
            py += '    trace.legendgroup = f"group_%d"\n' % j
            py += "    trace.showlegend = (k == 0)\n"
            py += (
                "    if hasattr(trace, 'line') and trace.line is not None: "
                "trace.line.color = colors[%d %% len(colors)]\n" % j
            )
            py += "if fig_freq_%d is None: fig_freq_%d = fig_temp\n" % (
                position,
                position,
            )
            py += "else: fig_freq_%d.add_traces(fig_temp.data)\n" % position
        py += "fig_freq_%d.show()\n" % position

    elif kind == "modes":
        py += (
            "modal_%d = rotor.run_modal(speed=%s, num_modes=%s, sparse=%s, synchronous=%s)\n"
            % (
                position,
                _py_val(p, "speed", "rad/s"),
                _js_str(p.get("num_modes")),
                _flag(p, "sparse", default="True"),
                _flag(p, "synchronous"),
            )
        )

        if p.get("plot_type") == "3D":
            args = _opt_args(
                p,
                [
                    ("frequency_type", "str"),
                    ("length_units", "str"),
                    ("phase_units", "str"),
                    ("frequency_units", "str"),
                    ("damping_parameter", "str"),
                ],
            )
            if _js_truthy(p.get("animation")):
                args.append("animation=%s" % _flag(p, "animation"))
            py += "modal_%d.plot_mode_3d(%s, %s).show()\n" % (
                position,
                _js_str(p.get("plot_idx")),
                ", ".join(args),
            )
        elif p.get("plot_type") == "Orbit":
            nodes = (
                "nodes=%s" % _js_str(p["nodes"]) if _js_truthy(p.get("nodes")) else ""
            )
            py += "modal_%d.plot_orbit(%s, %s).show()\n" % (
                position,
                _js_str(p.get("plot_idx")),
                nodes,
            )
        else:
            args = _opt_args(
                p,
                [
                    ("orientation", "str"),
                    ("frequency_type", "str"),
                    ("frequency_units", "str"),
                    ("damping_parameter", "str"),
                ],
            )
            py += "modal_%d.plot_mode_2d(%s, %s).show()\n" % (
                position,
                _js_str(p.get("plot_idx")),
                ", ".join(args),
            )

    elif kind == "unbalance":
        py += "speed_rads = np.linspace(%s, %s, 50)\n" % (
            _py_val(p, "speed_min", "rad/s"),
            _py_val(p, "speed_max", "rad/s"),
        )
        nodes, mags, phases = _unbalance_columns(p)
        modes = ", modes=%s" % _js_str(p["modes"]) if _js_truthy(p.get("modes")) else ""
        py += (
            "unb_%d = rotor.run_unbalance_response(node=[%s], unbalance_magnitude=[%s], "
            "unbalance_phase=[%s], frequency=speed_rads%s)\n"
        ) % (position, nodes, mags, phases, modes)

        method = {
            "Magnitude": "plot_magnitude",
            "Phase": "plot_phase",
            "Bode": "plot_bode",
            "Polar Bode": "plot_polar_bode",
        }.get(p.get("plot_type"), "plot")
        args = _opt_args(
            p,
            [
                ("probe_units", "str"),
                ("frequency_units", "str"),
                ("amplitude_units", "str"),
            ],
        )
        if p.get("plot_type") in ("Default", "Phase", "Bode", "Polar Bode"):
            args += _opt_args(p, [("phase_units", "str")])
        if p.get("plot_type") == "Magnitude":
            args += _opt_args(p, [("line_shape", "str")])
        py += "unb_%d.%s(probe=[%s], %s).show()\n" % (
            position,
            method,
            _probes_expression(p),
            ", ".join(args),
        )

    elif kind in ("time_response", "misalignment", "rubbing", "crack"):
        py += _transient_block(position, kind, p)

    elif kind == "static":
        py += "static_%d = rotor.run_static()\n" % position
        args = _opt_args(p, [("rotor_length_units", "str")])
        chart = p.get("plot_type")
        if chart == "Deformation":
            args += _opt_args(p, [("deformation_units", "str")])
            py += "static_%d.plot_deformation(%s).show()\n" % (
                position,
                ", ".join(args),
            )
        elif chart == "Shearing Force":
            args += _opt_args(p, [("force_units", "str")])
            py += "static_%d.plot_shearing_force(%s).show()\n" % (
                position,
                ", ".join(args),
            )
        elif chart == "Bending Moment":
            args += _opt_args(p, [("moment_units", "str")])
            py += "static_%d.plot_bending_moment(%s).show()\n" % (
                position,
                ", ".join(args),
            )
        else:
            args += _opt_args(p, [("force_units", "str")])
            py += "static_%d.plot_free_body_diagram(%s).show()\n" % (
                position,
                ", ".join(args),
            )

    elif kind == "harmonic_balance":
        py += "t_hb = np.linspace(%s, %s, %s)\n" % (
            _js_str(_or(p.get("t_initial"), 0)),
            _js_str(_or(p.get("t_final"), 0.5)),
            _js_str(_or(p.get("t_steps"), 1001)),
        )
        py += "harmonic_forces = [{\n"
        py += "    'node': %s,\n" % _js_str(_or(p.get("hb_node"), 0))
        py += "    'magnitudes': %s,\n" % _js_str(_or(p.get("hb_magnitudes"), "[2000]"))
        py += "    'phases': %s,\n" % _js_str(_or(p.get("hb_phases"), "[0]"))
        py += "    'harmonics': %s\n" % _js_str(_or(p.get("hb_harmonics"), "[1]"))
        py += "}]\n"
        py += (
            "hb_%d = rotor.run_harmonic_balance_response(speed=%s, t=t_hb, "
            "harmonic_forces=harmonic_forces, gravity=%s, n_harmonics=%s)\n"
        ) % (
            position,
            _py_val(p, "speed", "rad/s"),
            _flag(p, "gravity"),
            _js_str(_or(p.get("n_harmonics"), 1)),
        )
        args = _opt_args(p, [("amplitude_units", "str"), ("frequency_units", "str")])
        py += "hb_%d.plot(probe=[%s], %s).show()\n" % (
            position,
            _probes_expression(p),
            ", ".join(args),
        )

    elif kind == "clearance":
        # The exported script had the same defect as the runner, and this is the
        # copy that leaves the program: it wrote `node=1` beside
        # `unbalance_magnitude=[0.05]`, so whoever ran it on numpy 2.5 got the
        # `TypeError` on their own machine, with no interface to blame. The
        # three columns now come from the same table, like the unbalance
        # response's block above.
        nodes, mags, phases = _unbalance_columns(p, "0.05", "0.0")
        extras = []
        if _js_truthy(p.get("frequency")):
            extras.append("frequency=%s" % _js_str(p["frequency"]))
        if _js_truthy(p.get("modes")):
            extras.append("modes=%s" % _js_str(p["modes"]))
        py += (
            "clearance_%d = rotor.run_clearance_analysis(speed=%s, node=[%s], "
            "unbalance_magnitude=[%s], unbalance_phase=[%s]%s)\n"
        ) % (
            position,
            _py_val(p, "speed", "rad/s"),
            nodes,
            mags,
            phases,
            (", " + ", ".join(extras)) if extras else "",
        )
        py += "clearance_%d.plot().show()\n" % position

    return py


def _transient_block(position, kind, p):
    """Time-domain analyses: forced response and the three ROSS faults."""
    py = ""
    if kind == "time_response":
        py += "speed = %s\n" % _py_val(p, "speed", "rad/s")
        py += "t = np.linspace(0, %s, %s)\n" % (
            _js_str(_or(p.get("t_max"), 1.0)),
            _js_str(_or(p.get("steps"), 1000)),
        )
        py += "dofs_per_node = rotor.number_dof\n"
        py += "F_%d = np.zeros((len(t), rotor.ndof))\n" % position
        for force in p.get("forces") or []:
            py += "n_force = min(%s, len(rotor.nodes) - 1)\n" % _js_str(
                force.get("node")
            )
            py += "g_dof = n_force * dofs_per_node + %s\n" % _js_str(force.get("dof"))
            # func is an expression written by the user: it goes in as typed.
            py += "F_%d[:, g_dof] += %s\n" % (position, _js_str(force.get("func")))
        py += "resp_%d = rotor.run_time_response(speed, F_%d, t, method=%s)\n" % (
            position,
            position,
            _py_string(_or(p.get("method"), "default")),
        )
    else:
        py += "t_sim = np.linspace(%s, %s, %s)\n" % (
            _js_str(_or(p.get("t_initial"), 0)),
            _js_str(_or(p.get("t_final"), 0.5)),
            _js_str(_or(p.get("t_steps"), 5000)),
        )
        nodes, mags, phases = _unbalance_columns(p)
        common = (
            "node=[%s], unbalance_magnitude=[%s], unbalance_phase=[%s], speed=%s, t=t_sim"
            % (nodes, mags, phases, _py_val(p, "speed", "rad/s"))
        )

        if kind == "misalignment":
            kw = ["coupling=%s" % _py_string(_or(p.get("coupling"), "flex"))]
            if p.get("n") is not None and p.get("n") != "":
                kw.append("n=%s" % _js_str(p["n"]))
            for name in ("input_torque", "load_torque"):
                if _js_truthy(p.get(name)):
                    kw.append("%s=%s" % (name, _js_str(p[name])))
            if p.get("coupling") == "flex":
                if _js_truthy(p.get("mis_type")):
                    kw.append("mis_type=%s" % _py_string(p["mis_type"]))
                for name in (
                    "mis_distance_x",
                    "mis_distance_y",
                    "mis_angle",
                    "radial_stiffness",
                    "bending_stiffness",
                ):
                    if _js_truthy(p.get(name)):
                        kw.append("%s=%s" % (name, _js_str(p[name])))
            elif _js_truthy(p.get("mis_distance")):
                kw.append("mis_distance=%s" % _js_str(p["mis_distance"]))
            py += "resp_%d = rotor.run_misalignment(%s, %s)\n" % (
                position,
                common,
                ", ".join(kw),
            )

        elif kind == "rubbing":
            py += (
                "resp_%d = rotor.run_rubbing(n=%s, distance=%s, contact_stiffness=%s, "
                "contact_damping=%s, friction_coeff=%s, %s, torque=%s)\n"
            ) % (
                position,
                _js_str(_or(p.get("n"), 0)),
                _js_str(_or(p.get("distance"), 0)),
                _js_str(_or(p.get("contact_stiffness"), 0)),
                _js_str(_or(p.get("contact_damping"), 0)),
                _js_str(_or(p.get("friction_coeff"), 0)),
                common,
                _flag(p, "torque"),
            )

        elif kind == "crack":
            extra = (
                ", cross_divisions=%s" % _js_str(p["cross_divisions"])
                if _js_truthy(p.get("cross_divisions"))
                else ""
            )
            py += (
                "resp_%d = rotor.run_crack(n=%s, depth_ratio=%s, %s, "
                "crack_model=%s%s)\n"
            ) % (
                position,
                _js_str(_or(p.get("n"), 0)),
                _js_str(_or(p.get("depth_ratio"), 0)),
                common,
                _py_string(_or(p.get("crack_model"), "Mayes")),
                extra,
            )

    probes = _probes_expression(p)
    first_node = p["probes"][0].get("node") if (p.get("probes") or []) else 0
    args = _opt_args(p, [("displacement_units", "str")])
    chart = p.get("plot_type")

    if chart == "Frequency (DFFT)":
        args += _opt_args(p, [("probe_units", "str"), ("frequency_units", "str")])
        py += "resp_%d.plot_dfft(probe=[%s], %s).show()\n" % (
            position,
            probes,
            ", ".join(args),
        )
    elif chart == "2D":
        py += "resp_%d.plot_2d(node=%s, %s).show()\n" % (
            position,
            _js_str(first_node),
            ", ".join(args),
        )
    elif chart == "3D":
        args += _opt_args(p, [("rotor_length_units", "str")])
        py += "resp_%d.plot_3d(%s).show()\n" % (position, ", ".join(args))
    else:
        args += _opt_args(p, [("probe_units", "str"), ("time_units", "str")])
        py += "resp_%d.plot_1d(probe=[%s], %s).show()\n" % (
            position,
            probes,
            ", ".join(args),
        )

    return py


def _compatibility_warning(analyses, conversion_type):
    """The warning header, when the script carries a combination the screen refuses.

    The interface blocks the combination before computing, but the exported
    script is the user's code: refusing to export would be deciding for them.
    What is owed is that they know, in the file, what the interface knew --
    otherwise the script fails in their hands with ROSS's raw error, or worse,
    runs and returns full-model numbers with the converted rotor right above.

    It only appears when there is something to warn about: a constant header
    would change every script and say nothing in most of them.
    """
    from domain.compatibility import MODEL_NAMES, reason

    found_items = []
    for analysis in analyses or ():
        kind = (analysis or {}).get("type")
        why = reason(kind, conversion_type) if kind else None
        if why and not any(kind == t for t, _ in found_items):
            found_items.append((kind, why))
    if not found_items:
        return ""

    model = MODEL_NAMES.get(conversion_type, conversion_type)
    lines = [
        "# " + "=" * 42,
        "# WARNING",
        "# " + "=" * 42,
        "# The interface does not run the analyses below on the %s model," % model,
        "# and this script does. Read why before trusting the numbers:",
        "#",
    ]
    for kind, why in found_items:
        lines.append("#   %s" % kind)
        for line in textwrap.wrap(why, 72):
            lines.append("#     %s" % line)
        lines.append("#")
    return "\n".join(lines) + "\n\n"


def build_script(project, analyses=(), conversion_type=""):
    """Build the complete Python script: modelling, conversion and analyses.

    `project` is the interface's projectData; `analyses` is the list of open
    cards, each with its analysis type and the parameters its dashboard used.
    """
    project = project or {}
    py = _compatibility_warning(analyses, conversion_type)
    py += "import ross as rs\nimport numpy as np\nfrom ross.units import Q_\n"
    py += "\n# ==========================================\n# Modeling \n# ==========================================\n"

    if project.get("isMultiRotor"):
        py += _build_rotor_block(project.get("driving_rotor"), "_driving")
        py += _build_rotor_block(project.get("driven_rotor"), "_driven")
        py += _multi_rotor_block(project.get("multi_params"))
    else:
        py += _build_rotor_block(project, "")
        if conversion_type == "4dof":
            py += 'rotor = rs.utils.convert_6dof_to_4dof(rotor)\nprint("Rotor converted to 4 DoF!")\n'
        elif conversion_type == "torsional":
            py += 'rotor = rs.utils.convert_6dof_to_torsional(rotor)\nprint("Rotor converted to Torsional!")\n'

    py += 'print("Rotor Successfully Modeled!")\nrotor.plot_rotor().show()\n\n'

    complete = [a for a in (analyses or []) if a and a.get("type") and a.get("params")]
    if complete:
        py += "# ==========================================\n"
        py += "# Analysis \n"
        py += "# ==========================================\n"
        for position, analysis in enumerate(complete):
            py += _analysis_block(position, analysis)
    return py
