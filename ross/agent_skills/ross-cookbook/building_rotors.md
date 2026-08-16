# Building a Rotor from Scratch

Source: `docs/user_guide/tutorial_part_1_1.ipynb`

## Material

```python
import ross as rs

steel = rs.Material(name="steel", rho=7810, E=211e9, G_s=81.2e9)
# rho: density (kg/m³), E: Young's modulus (Pa), G_s: shear modulus (Pa)
```

Pre-built materials: `rs.Material.use_material("Steel1020")`. List available with `rs.Material.available_materials()`.

## Shaft Elements

```python
# Uniform cross-section
shaft0 = rs.ShaftElement(L=0.05, idl=0, odl=0.05, material=steel)

# Tapered element (different left/right diameters)
shaft1 = rs.ShaftElement(L=0.05, idl=0, odl=0.05, idr=0.01, odr=0.06, material=steel)

# Build a shaft from multiple segments
shaft = [rs.ShaftElement(L=0.05, idl=0, odl=0.05, material=steel) for _ in range(6)]
```

Parameters: `L` (length, m), `idl`/`idr` (inner diameter left/right, m), `odl`/`odr` (outer diameter left/right, m).

## Node Numbering

Nodes are numbered sequentially from left (node 0) to right. Each shaft element connects two adjacent nodes. With 6 shaft elements: nodes 0–6 (7 nodes total).

```
node:  0    1    2    3    4    5    6
       |====|====|====|====|====|====|
shaft:    0    1    2    3    4    5
```

## Disk Elements

```python
# From geometry (automatically computes mass and inertia)
disk0 = rs.DiskElement.from_geometry(n=2, material=steel, width=0.07, i_d=0.05, o_d=0.28)

# From explicit properties
disk1 = rs.DiskElement(n=4, m=32.59, Id=0.178, Ip=0.329)
# m: mass (kg), Id: diametral inertia (kg·m²), Ip: polar inertia (kg·m²)
```

`n` is the node number where the disk is attached.

## Bearing Elements

```python
# Simple isotropic bearing
brg0 = rs.BearingElement(n=0, kxx=1e6, cxx=0)

# Anisotropic bearing
brg1 = rs.BearingElement(n=6, kxx=1e6, kyy=0.8e6, cxx=100, cyy=80)

# Cross-coupled coefficients
brg2 = rs.BearingElement(n=0, kxx=1e6, kyy=1e6, kxy=0.5e5, kyx=-0.5e5, cxx=100, cyy=100)
```

Parameters: `n` (node), `kxx`/`kyy` (direct stiffness, N/m), `kxy`/`kyx` (cross-coupled stiffness), `cxx`/`cyy` (direct damping, N·s/m), `cxy`/`cyx` (cross-coupled damping). See [bearings_advanced.md](bearings_advanced.md) for speed-dependent coefficients.

## Rotor Assembly

```python
rotor = rs.Rotor(shaft, [disk0, disk1], [brg0, brg1])

# Key properties
rotor.nodes          # list of node numbers
rotor.ndof           # total degrees of freedom
rotor.m              # total mass (kg)
rotor.L              # total length (m)
rotor.number_dof     # DOFs per node (4 for standard, 6 for 6-DOF)
```

DOF ordering per node (standard 4-DOF): `[x, y, θx, θy]`. DOF index for node `n`, direction `d`: `rotor.number_dof * n + d` where `d` is 0=x, 1=y, 2=θx, 3=θy.

## Save / Load

```python
rotor.save("my_rotor.toml")
rotor = rs.Rotor.load("my_rotor.toml")
```
