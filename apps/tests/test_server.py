import os
import sys
import pytest
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import *

@pytest.fixture
def client():
    """Configura o ambiente de teste do Flask."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

# Basic Construction and Materials Test

def test_build_rotor_minimum_valid(client):
    """Tests whether the rotor is successfully created using the default material and shaft (Steel)."""
    payload = {
        "materials": [{"name": "Steel", "rho": "7800", "E": "211e9", "G_s": "81.2e9"}],
        "shafts": [{"L": "500", "odl": "100", "idl": "0", "material": "Steel", "n": "0"}]
    }
    response = client.post('/build_rotor', json=payload)
    
    assert response.status_code == 200
    assert response.json['status'] == 'success'
    assert 'plot_json' in response.json
    assert response.json['mass'] > 0

# Test of Mass and Inertia Elements

def test_elements_disk_gear_defaults(client):
    """Tests the injection of Disk and Gear defaults."""
    payload = {
        "materials": [{"name": "Steel", "rho": "7800", "E": "211e9", "G_s": "81.2e9"}],
        "shafts": [{"L": "500", "odl": "100", "idl": "0", "material": "Steel", "n": "0"}],
        "disks": [{"m": "32", "Id": "0.2", "Ip": "0.3", "n": "0"}],
        "gears": [{"m": "4.67", "Id": "0.015", "Ip": "0.030", "n_teeth": "26", "pitch_diameter": "187", "pr_angle": "22.5", "helix_angle": "0", "n": "1"}],
    }
    response = client.post('/build_rotor', json=payload)
    
    assert response.status_code == 200
    assert response.json['status'] == 'success'
    assert response.json['mass'] > 38.67

# Bearing Element Tests

def test_bearing_elements_defaults(client):
    """Tests whether the backend can instantiate complex bearings using the UI's array and string defaults."""
    payload = {
        "materials": [{"name": "Steel", "rho": "7800", "E": "211e9", "G_s": "81.2e9"}],
        "shafts": [{"L": "500", "odl": "100", "idl": "0", "material": "Steel", "n": "0"}],
        "bearings": [
            # Bearing BASIC
            {"element_type": "BASIC", "kxx": "1e6", "kyy": "0.8e6", "cxx": "2e2", "cyy": "1.5e2", "n": "0"},
            # Cylindrical Bearing
            {"element_type": "Cylindrical", "speed": "[1500]", "weight": "525", "bearing_length": "30", "journal_diameter": "10", "radial_clearance": "0.1", "oil_viscosity": "0.1", "n": "1"}
        ]
    }
    response = client.post('/build_rotor', json=payload)
    
    assert response.status_code == 200
    assert response.json['status'] == 'success'

# Seal Element Tests

def test_seal_elements_defaults(client):
    """Tests labels that use nested JSON dictionaries in the default (e.g., gas_composition)."""
    payload = {
        "materials": [{"name": "Steel", "rho": "7800", "E": "211e9", "G_s": "81.2e9"}],
        "shafts": [{"L": "500", "odl": "100", "idl": "0", "material": "Steel", "n": "0"}],
        "seals": [
            # Labyrinth Seal
            {
                "element_type": "Labyrinth", 
                "shaft_radius": "72.5", "radial_clearance": "0.3", "n_teeth": "16", 
                "pitch": "3.175", "tooth_height": "3.175", "tooth_width": "0.1524", 
                "seal_type": "inter", "inlet_pressure": "308000", "outlet_pressure": "94300", 
                "inlet_temperature": "10", "frequency": "[8000]", "preswirl": "0.98", 
                "gas_composition": '{"Nitrogen": 0.79, "Oxygen": 0.21}', "n": "0"
            }
        ]
    }
    response = client.post('/build_rotor', json=payload)
    
    assert response.status_code == 200
    assert response.json['status'] == 'success'

# Error Handling Tests

def test_shaftless_rotor_failure(client):
    """Check if the server locks properly (Error 400) if the user does not submit axes."""
    payload = {
        "disks": [{"m": "32", "Id": "0.2", "Ip": "0.3", "n": "0"}]
    }
    response = client.post('/build_rotor', json=payload)
    
    assert response.status_code == 400
    assert response.json['status'] == 'error'
    assert "Add at least one Shaft" in response.json['message']