"""Minimal MCP server exposing core ROSS rotordynamics tools.

Usage (stdio transport):
    python -m ross.mcp

Configure in .mcp.json for Claude Code auto-discovery.
"""

from mcp.server.fastmcp import FastMCP
import ross as rs
import numpy as np
import json

mcp = FastMCP("ROSS Rotordynamics")

_rotors: dict[str, rs.Rotor] = {}

_EXAMPLE_VARIANTS = {
    "default": rs.rotor_example,
    "compressor": rs.compressor_example,
    "6dof": rs.rotor_example_6dof,
    "damped": rs.rotor_example_with_damping,
}


def _rotor_summary(name: str, rotor: rs.Rotor) -> str:
    """Return a short text summary of a rotor."""
    return (
        f"Rotor '{name}': "
        f"{len(rotor.shaft_elements)} shaft elements, "
        f"{len(rotor.disk_elements)} disks, "
        f"{len(rotor.bearing_elements)} bearings, "
        f"{len(rotor.nodes)} nodes, "
        f"{rotor.ndof} DOFs, "
        f"total length = {rotor.L:.4f} m"
    )


@mcp.tool()
def load_rotor_from_file(name: str, file_path: str) -> str:
    """Load a rotor from a .json or .toml file into server state.

    Parameters
    ----------
    name : str
        Key to store the rotor under (used to reference it in other tools).
    file_path : str
        Absolute path to the .json or .toml file saved by ROSS.
    """
    from pathlib import Path

    path = Path(file_path)
    if not path.exists():
        return f"File not found: {file_path}"
    if path.suffix.lower() not in (".json", ".toml"):
        return f"Unsupported file format '{path.suffix}'. Use .json or .toml."
    try:
        rotor = rs.Rotor.load(file_path)
    except Exception as e:
        return f"Error loading rotor from '{file_path}': {e}"
    _rotors[name] = rotor
    return _rotor_summary(name, rotor)


@mcp.tool()
def create_example_rotor(name: str, variant: str = "default") -> str:
    """Load a pre-built ROSS example rotor into server state.

    Parameters
    ----------
    name : str
        Key to store the rotor under (used to reference it in other tools).
    variant : str
        One of "default", "compressor", "6dof", "damped".
    """
    if variant not in _EXAMPLE_VARIANTS:
        return f"Unknown variant '{variant}'. Choose from: {list(_EXAMPLE_VARIANTS)}"
    rotor = _EXAMPLE_VARIANTS[variant]()
    _rotors[name] = rotor
    return _rotor_summary(name, rotor)


@mcp.tool()
def describe_rotor(name: str) -> str:
    """Return a detailed text description of a stored rotor.

    Includes shaft elements (length, OD, ID), disks (node, mass, inertia),
    bearings (node, stiffness, damping), total mass, and DOF count.
    """
    if name not in _rotors:
        return f"No rotor named '{name}'. Available: {list(_rotors)}"
    rotor = _rotors[name]

    lines = [_rotor_summary(name, rotor), ""]

    # Shaft elements
    lines.append("Shaft elements:")
    for se in rotor.shaft_elements:
        lines.append(
            f"  nodes {se.n_l}-{se.n_r}: L={se.L:.4f} m, "
            f"OD={se.o_d:.4f} m, ID={se.i_d:.4f} m"
        )

    # Disk elements
    lines.append("\nDisk elements:")
    for de in rotor.disk_elements:
        lines.append(
            f"  node {de.n}: m={de.m:.4f} kg, "
            f"Id={de.Id:.6f} kg.m², Ip={de.Ip:.6f} kg.m²"
        )

    # Bearing elements
    lines.append("\nBearing elements:")
    for be in rotor.bearing_elements:
        kxx_val = be.kxx[0] if hasattr(be.kxx, "__len__") else be.kxx
        cxx_val = be.cxx[0] if hasattr(be.cxx, "__len__") else be.cxx
        lines.append(f"  node {be.n}: kxx={kxx_val:.2e} N/m, cxx={cxx_val:.2e} N.s/m")

    # Mass
    lines.append(f"\nTotal mass: {rotor.m:.4f} kg")
    lines.append(f"  Shaft mass: {rotor.m_shaft:.4f} kg")
    lines.append(f"  Disk mass: {rotor.m_disks:.4f} kg")

    return "\n".join(lines)


@mcp.tool()
def run_modal_analysis(name: str, speed: float, num_modes: int = 12) -> str:
    """Run eigenvalue analysis on a stored rotor at a given speed.

    Parameters
    ----------
    name : str
        Key of a previously stored rotor.
    speed : float
        Rotor speed in rad/s.
    num_modes : int
        Number of modes to compute (default 12).

    Returns a JSON object with natural frequencies, damped frequencies,
    damping ratios, and logarithmic decrements.
    """
    if name not in _rotors:
        return f"No rotor named '{name}'. Available: {list(_rotors)}"
    rotor = _rotors[name]
    modal = rotor.run_modal(speed=speed, num_modes=num_modes)
    result = {
        "speed_rad_s": speed,
        "wn_rad_s": modal.wn.tolist(),
        "wd_rad_s": modal.wd.tolist(),
        "damping_ratio": modal.damping_ratio.tolist(),
        "log_dec": modal.log_dec.tolist(),
    }
    return json.dumps(result, indent=2)


@mcp.tool()
def run_campbell_diagram(
    name: str,
    speed_max: float,
    num_speeds: int = 50,
    num_frequencies: int = 6,
) -> str:
    """Generate Campbell diagram data for a stored rotor.

    Computes damped natural frequencies over a speed range from 0 to speed_max.

    Parameters
    ----------
    name : str
        Key of a previously stored rotor.
    speed_max : float
        Maximum speed in rad/s.
    num_speeds : int
        Number of speed points (default 50).
    num_frequencies : int
        Number of frequencies to track (default 6).

    Returns a JSON object with speed range, tracked frequencies, and log decrements.
    """
    if name not in _rotors:
        return f"No rotor named '{name}'. Available: {list(_rotors)}"
    rotor = _rotors[name]
    speed_range = np.linspace(0, speed_max, num_speeds)
    campbell = rotor.run_campbell(speed_range, frequencies=num_frequencies)
    result = {
        "speed_range_rad_s": campbell.speed_range.tolist(),
        "wd_rad_s": campbell.wd.tolist(),
        "log_dec": campbell.log_dec.tolist(),
        "damping_ratio": campbell.damping_ratio.tolist(),
    }
    return json.dumps(result, indent=2)


@mcp.tool()
def run_unbalance_response(
    name: str,
    node: int,
    unbalance_magnitude: float,
    unbalance_phase: float,
    speed_max: float,
    num_speeds: int = 100,
) -> str:
    """Run unbalance response analysis on a stored rotor.

    Parameters
    ----------
    name : str
        Key of a previously stored rotor.
    node : int
        Node where unbalance is applied.
    unbalance_magnitude : float
        Unbalance magnitude in kg.m.
    unbalance_phase : float
        Unbalance phase in rad.
    speed_max : float
        Maximum frequency in rad/s.
    num_speeds : int
        Number of frequency points (default 100).

    Returns a JSON object with the frequency array and magnitude of
    displacement response at the unbalance node.
    """
    if name not in _rotors:
        return f"No rotor named '{name}'. Available: {list(_rotors)}"
    rotor = _rotors[name]
    frequency_range = np.linspace(0, speed_max, num_speeds)
    resp = rotor.run_unbalance_response(
        node=node,
        unbalance_magnitude=unbalance_magnitude,
        unbalance_phase=unbalance_phase,
        frequency=frequency_range,
    )
    # Extract displacement magnitude at the unbalance node's x-direction DOF
    dof_index = node * rotor.number_dof
    magnitude = np.abs(resp.forced_resp[dof_index])
    result = {
        "frequency_rad_s": resp.speed_range.tolist(),
        "displacement_magnitude_m": magnitude.tolist(),
        "node": node,
        "dof_index": dof_index,
    }
    return json.dumps(result, indent=2)


def main():
    """Entry point for the ``ross-mcp`` console script."""
    mcp.run(transport="stdio")
