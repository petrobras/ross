import pytest
import ross as rs
import copy


def test_concatenate_rotor():
    # 1. SETUP: Create dummy elements and rotors for the test
    steel = rs.Material(name="Steel", rho=7810, E=211e9, G_s=81.2e9)

    # --- Rotor 1 ---
    # Original nodes: 0, 1, 2
    shaft1_r1 = rs.ShaftElement(L=0.25, idl=0.0, odl=0.05, material=steel, n=0)
    shaft2_r1 = rs.ShaftElement(L=0.25, idl=0.0, odl=0.05, material=steel, n=1)
    disk_r1 = rs.DiskElement(n=1, m=10, Id=0.2, Ip=0.1)
    brg_r1 = rs.BearingElement(n=0, kxx=1e6, cxx=0)

    rotor1 = rs.Rotor(
        shaft_elements=[shaft1_r1, shaft2_r1],
        disk_elements=[disk_r1],
        bearing_elements=[brg_r1],
    )

    # --- Rotor 2 ---
    # Original nodes: 0, 1
    shaft1_r2 = rs.ShaftElement(L=0.5, idl=0.0, odl=0.05, material=steel, n=0)
    disk_r2 = rs.DiskElement(n=1, m=5, Id=0.1, Ip=0.05)

    rotor2 = rs.Rotor(shaft_elements=[shaft1_r2], disk_elements=[disk_r2])

    # 2. ACTION: Execute the function we want to test
    rotor_concat = rs.concatenate_rotor([rotor1, rotor2])

    # 3. ASSERTIONS: Check if the result is as expected

    # Checking the total number of elements
    assert len(rotor_concat.shaft_elements) == 3
    assert len(rotor_concat.disk_elements) == 2
    assert len(rotor_concat.bearing_elements) == 1
    assert len(rotor_concat.point_mass_elements) == 0

    # Checking node continuity (Node Offset)
    # Rotor 1 ends at node 2. Therefore, Rotor 2 must start at node 2.
    assert rotor_concat.shaft_elements[0].n_l == 0  # Start of Rotor 1
    assert rotor_concat.shaft_elements[1].n_r == 2  # End of Rotor 1

    assert rotor_concat.shaft_elements[2].n_l == 2  # Start of Rotor 2 (formerly n=0)
    assert rotor_concat.shaft_elements[2].n_r == 3  # End of Rotor 2 (formerly n=1)

    # Checking if the disk nodes followed the offset
    assert rotor_concat.disk_elements[0].n == 1  # Disk 1 (originally at node 1)
    assert (
        rotor_concat.disk_elements[1].n == 3
    )  # Disk 2 (originally 1, +2 from the offset)

    # Checking if tags were applied correctly
    assert rotor_concat.shaft_elements[0].tag == "shaft_r0_0"
    assert rotor_concat.shaft_elements[2].tag == "shaft_r1_0"
    assert rotor_concat.disk_elements[1].tag == "disk_r1_0"
    assert rotor_concat.bearing_elements[0].tag == "bearing_r0_0"
