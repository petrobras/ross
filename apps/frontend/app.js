let projectData = { materials: [], shafts: [], disks: [], gears: [], couplings: [], seals: [], bearings: [], pointmasses: [] };
let currentTab = null;
let editingIndex = -1; 
let currentSubType = 'BASIC';

// Engineering defaults library

const DefaultExamples = {
    materials_BASIC: { name: "Steel", rho: "7800", E: "211e9", G_s: "81.2e9" },
    shafts_BASIC: { L: "500", odl: "100", idl: "0", material: "Steel" },
    disks_BASIC: { m: "32", Id: "0.2", Ip: "0.3" },
    gears_BASIC: { m: "4.67", Id: "0.015", Ip: "0.030", n_teeth: "26", pitch_diameter: "187", pr_angle: "22.5", helix_angle: "0" },
    gears_TVMS: { material: "Steel", width: "20", bore_diameter: "70", module: "2", n_teeth: "62", pr_angle: "20" },
    couplings_BASIC: { m_l: "37.8875", m_r: "37.8875", Ip_l: "1.0985", Ip_r: "1.0985", kr_z: "3.04256e6" },
    pointmasses_BASIC: { m: "2" },
    bearings_BASIC: { kxx: "1e6", kyy: "0.8e6", cxx: "2e2", cyy: "1.5e2" },
    bearings_BallBearing: { n_balls: "8", d_balls: "0.03", fs: "500", alpha: "0.523598" },
    bearings_RollerBearing: { n_rollers: "8", l_rollers: "0.03", fs: "500", alpha: "0.523598" },
    bearings_MagneticBearing: { g0: "1e-3", i0: "1", ag: "1e-4", nw: "200", kp_pid: "1", ki_pid: "0", kd_pid: "1", alpha: "0.392699", k_amp: "1", k_sense: "1" },
    bearings_Cylindrical: { speed: "[1500]", weight: "525", bearing_length: "30", journal_diameter: "10", radial_clearance: "0.1", oil_viscosity: "0.1" },
    bearings_PlainJournal: { axial_length: "263.144", journal_radius: "20", radial_clearance: "1.95e-1", n_pad: "2", pad_arc_length: "176", preload: "0", geometry: "circular", frequency: "[900]", fxs_load: "0", fys_load: "-112814.91", lubricant: "ISOVG32", reference_temperature: "50", groove_factor: "[0.52, 0.48]", elements_circumferential: "11", elements_axial: "3", sommerfeld_type: "2", initial_guess: "[0.1, -0.1]", method: "perturbation", operating_type: "flooded", oil_supply_pressure: "0", oil_flow_v: "37.86" },
    bearings_SqueezeFilm: { frequency: "[18600]", axial_length: "22.86", journal_radius: "64.77", radial_clearance: "7.62e-2", eccentricity_ratio: "0.5", lubricant: "ISOVG32", geometry: "groove", cavitation: "True" },
    bearings_ThrustPad: { pad_inner_radius: "1150", pad_outer_radius: "1725", pad_pivot_radius: "1442.5", pad_arc_length: "26", angular_pivot_position: "15", oil_supply_temperature: "40", lubricant: "ISOVG68", n_pad: "12", n_theta: "10", n_radial: "10", frequency: "[90]", equilibrium_position_mode: "calculate", radial_inclination_angle: "-2.75e-04", circumferential_inclination_angle: "-1.70e-05", initial_film_thickness: "0.2", axial_load: "13.32e6" },
    bearings_TiltingPad: { journal_diameter: "101.6", pre_load: "[0.5, 0.5, 0.5, 0.5, 0.5]", pad_thickness: "12.7", pad_arc: "[60, 60, 60, 60, 60]", offset: "[0.5, 0.5, 0.5, 0.5, 0.5]", pad_axial_length: "[50.8, 50.8, 50.8, 50.8, 50.8]", lubricant: "ISOVG32", oil_supply_temperature: "40", radial_clearance: "74.9e-3", pivot_angle: "[18, 90, 162, 234, 306]", frequency: "[3000]", equilibrium_type: "match_eccentricity", eccentricity: "0.35", attitude_angle: "287.5", load: "[884, -2670]" },
    seals_BASIC: { kxx: "1e6", cxx: "2e2", kyy: "0.8e6", cyy: "1.5e2" },
    seals_HolePattern: { shaft_radius: "72.5", radial_clearance: "0.3", length: "46.99", roughness: "0.0001", cell_length: "3.175", cell_width: "3.175", cell_depth: "2.5", inlet_pressure: "689000", outlet_pressure: "94300", inlet_temperature: "48.85", frequency: "[8000]", gas_composition: '{"Nitrogen": 0.79, "Oxygen": 0.21}', preswirl: "0.8", entr_coef: "0.5", exit_coef: "1.0", nz: "18" },
    seals_Labyrinth: { shaft_radius: "72.5", radial_clearance: "0.3", n_teeth: "16", pitch: "3.175", tooth_height: "3.175", tooth_width: "0.1524", seal_type: "inter", inlet_pressure: "308000", outlet_pressure: "94300", inlet_temperature: "10", frequency: "[8000]", preswirl: "0.98", gas_composition: '{"Nitrogen": 0.79, "Oxygen": 0.21}' },
    seals_Hybrid: { shaft_radius: "25", inlet_pressure: "500000", outlet_pressure: "100000", inlet_temperature: "26.85", frequency: "[2000]", gas_composition: '{"Nitrogen": 0.7812, "Oxygen": 0.2096, "Argon": 0.0092}', hole_pattern_parameters: '{"radial_clearance": 0.0003, "length": 0.04, "roughness": 0.0001, "cell_length": 0.003, "cell_width": 0.003, "cell_depth": 0.002, "preswirl": 0.8, "entr_coef": 0.5, "exit_coef": 1.0}', labyrinth_parameters: '{"radial_clearance": 0.00025, "n_teeth": 10, "pitch": 0.003, "tooth_height": 0.003, "tooth_width": 0.00015, "seal_type": "inter", "preswirl": 0.9, "tz": [300.0, 299.5], "muz": [1.85e-05, 1.84e-05]}' }
};

// Fill in the values ​​with the default

function fillDefault() {    
    let searchType = currentSubType === 'LIST' ? 'BASIC' : currentSubType;
    const defaultData = DefaultExamples[`${currentTab}_${searchType}`];    
    if (!defaultData) return alert("No default parameters available for this class.");
    for (let key in defaultData) {
        let input = document.getElementById(`inp-${key}`);
        if (input) input.value = defaultData[key];
    }
    const advBtn = document.getElementById('form-fields').querySelector('.btn-advanced');
    if (advBtn) {
        const advDiv = advBtn.nextElementSibling;
        if (advDiv && (advDiv.style.display === 'none' || advDiv.style.display === '')) toggleAdvanced(advBtn);
    }
}

// Templates

const FormTemplates = {
    materials: {
        BASIC: `
        <div class="input-group"><label>Material Name</label><input type="text" id="inp-name" placeholder="e.g., Steel"></div>
        <div class="input-group"><label>Density [kg/m³]</label><input type="text" id="inp-rho"></div>
        <div class="input-group"><label>Elastic Modulus [Pa] (Optional)</label><input type="text" id="inp-E"></div>
        <div class="input-group"><label>Shear Modulus [Pa] (Optional)</label><input type="text" id="inp-G_s"></div>
        <div class="input-group"><label>Poisson's Ratio (Optional)</label><input type="text" id="inp-poisson"></div>
        <button type="button" class="btn-advanced" onclick="toggleAdvanced(this)">Advanced <i class="fas fa-chevron-down"></i></button>
        <div class="advanced-fields" style="display: none; margin-top: 10px; border-top: 1px dashed #ccc; padding-top: 10px;">
            <div class="input-group"><label>Specific Heat [J/(kg.K)]</label><input type="text" id="inp-specific_heat"></div>
            <div class="input-group"><label>Thermal Conductivity [W/(m.K)]</label><input type="text" id="inp-thermal_conductivity"></div>
            <div class="input-group"><label>Hex Color</label><input type="text" id="inp-color"></div>
        </div>`
    },
    shafts: {
        BASIC: `
        <div class="input-group"><label>Node # (Optional)</label><input type="text" id="inp-n" placeholder="Auto-calculated if empty"></div>
        <div class="input-group"><label>Length [mm]</label><input type="text" id="inp-L"></div>
        <div class="input-group"><label>Left Outer Diam. [mm]</label><input type="text" id="inp-odl"></div>
        <div class="input-group"><label>Left Inner Diam. [mm]</label><input type="text" id="inp-idl"></div>
        <div class="input-group"><label>Material Name</label><select id="inp-material"></select></div>
        <button type="button" class="btn-advanced" onclick="toggleAdvanced(this)">Advanced <i class="fas fa-chevron-down"></i></button>
        <div class="advanced-fields" style="display: none; margin-top: 10px; border-top: 1px dashed #ccc; padding-top: 10px;">
            <div class="input-group"><label>Right Outer Diam. [mm]</label><input type="text" id="inp-odr"></div>
            <div class="input-group"><label>Right Inner Diam. [mm]</label><input type="text" id="inp-idr"></div>
            <div class="input-group"><label>Axial Force [N]</label><input type="text" id="inp-axial_force"></div>
            <div class="input-group"><label>Torque [N.m]</label><input type="text" id="inp-torque"></div>
            <div class="input-group"><label>Shear Effects</label><input type="text" id="inp-shear_effects"></div>
            <div class="input-group"><label>Rotary Inertia</label><input type="text" id="inp-rotary_inertia"></div>
            <div class="input-group"><label>Gyroscopic Effect</label><input type="text" id="inp-gyroscopic"></div>
            <div class="input-group"><label>Shear Calc Method</label><input type="text" id="inp-shear_method_calc"></div>
            <div class="input-group"><label>Alpha</label><input type="text" id="inp-alpha"></div>
            <div class="input-group"><label>Beta</label><input type="text" id="inp-beta"></div>
            <div class="input-group"><label>Tag</label><input type="text" id="inp-tag"></div>
        </div>`
    },
    disks: {
        BASIC: `
        <div class="input-group"><label>Node # (Optional)</label><input type="text" id="inp-n"></div>
        <div class="input-group"><label>Mass [kg]</label><input type="text" id="inp-m"></div>
        <div class="input-group"><label>Polar Inertia [kg.m²]</label><input type="text" id="inp-Ip"></div>
        <div class="input-group"><label>Diametral Inertia [kg.m²]</label><input type="text" id="inp-Id"></div>
        <button type="button" class="btn-advanced" onclick="toggleAdvanced(this)">Advanced <i class="fas fa-chevron-down"></i></button>
        <div class="advanced-fields" style="display: none; margin-top: 10px; border-top: 1px dashed #ccc; padding-top: 10px;">
            <div class="input-group"><label>Scale Factor</label><input type="text" id="inp-scale_factor"></div>
            <div class="input-group"><label>Tag</label><input type="text" id="inp-tag"></div>
            <div class="input-group"><label>Hex Color</label><input type="text" id="inp-color"></div>
        </div>`
    },
    gears: {
        BASIC: `
        <div class="input-group"><label>Node # (Optional)</label><input type="text" id="inp-n"></div>
        <div class="input-group"><label>Mass [kg]</label><input type="text" id="inp-m"></div>
        <div class="input-group"><label>Polar Inertia [kg.m²]</label><input type="text" id="inp-Ip"></div>
        <div class="input-group"><label>Diametral Inertia [kg.m²]</label><input type="text" id="inp-Id"></div>
        <div class="input-group"><label>Number of Teeth</label><input type="text" id="inp-n_teeth"></div>
        <div class="input-group"><label>Pitch Diameter [mm] (Optional)</label><input type="text" id="inp-pitch_diameter"></div>
        <div class="input-group"><label>Base Diameter [mm] (Optional)</label><input type="text" id="inp-base_diameter"></div>
        <button type="button" class="btn-advanced" onclick="toggleAdvanced(this)">Advanced <i class="fas fa-chevron-down"></i></button>
        <div class="advanced-fields" style="display: none; margin-top: 10px; border-top: 1px dashed #ccc; padding-top: 10px;">
            <div class="input-group"><label>Pressure Angle [deg]</label><input type="text" id="inp-pr_angle"></div>
            <div class="input-group"><label>Helix Angle [deg]</label><input type="text" id="inp-helix_angle"></div>
            <div class="input-group"><label>Bore Diameter [mm]</label><input type="text" id="inp-bore_diameter"></div>
            <div class="input-group"><label>Material Name</label><select id="inp-material"></select></div>
            <div class="input-group"><label>Scale Factor</label><input type="text" id="inp-scale_factor"></div>
            <div class="input-group"><label>Tag</label><input type="text" id="inp-tag"></div>
            <div class="input-group"><label>Hex Color</label><input type="text" id="inp-color"></div>
        </div>`,
        TVMS: `
        <div class="input-group"><label>Node # (Optional)</label><input type="text" id="inp-n"></div>
        <div class="input-group"><label>Material Name</label><select id="inp-material"></select></div>
        <div class="input-group"><label>Tooth Width [mm]</label><input type="text" id="inp-width"></div>
        <div class="input-group"><label>Bore Diameter [mm]</label><input type="text" id="inp-bore_diameter"></div>
        <div class="input-group"><label>Module [mm]</label><input type="text" id="inp-module"></div>
        <div class="input-group"><label>Number of Teeth</label><input type="text" id="inp-n_teeth"></div>
        <button type="button" class="btn-advanced" onclick="toggleAdvanced(this)">Advanced <i class="fas fa-chevron-down"></i></button>
        <div class="advanced-fields" style="display: none; margin-top: 10px; border-top: 1px dashed #ccc; padding-top: 10px;">
            <div class="input-group"><label>Pressure Angle [deg]</label><input type="text" id="inp-pr_angle"></div>
            <div class="input-group"><label>Helix Angle [deg]</label><input type="text" id="inp-helix_angle"></div>
            <div class="input-group"><label>Addendum Coefficient</label><input type="text" id="inp-addendum_coeff"></div>
            <div class="input-group"><label>Tip Clearance Coefficient</label><input type="text" id="inp-tip_clearance_coeff"></div>
            <div class="input-group"><label>Scale Factor</label><input type="text" id="inp-scale_factor"></div>
            <div class="input-group"><label>Tag</label><input type="text" id="inp-tag"></div>
            <div class="input-group"><label>Hex Color</label><input type="text" id="inp-color"></div>
        </div>`
    },
    bearings: {
        BASIC: `
        <div class="input-group"><label>Node # (Optional)</label><input type="text" id="inp-n"></div>
        <div style="display:flex;gap:5px;">
            <div class="input-group"><label>kxx [N/m]</label><input type="text" id="inp-kxx"></div>
            <div class="input-group"><label>cxx [N.s/m]</label><input type="text" id="inp-cxx"></div>
        </div>
        <button type="button" class="btn-advanced" onclick="toggleAdvanced(this)">Advanced <i class="fas fa-chevron-down"></i></button>
        <div class="advanced-fields" style="display: none; margin-top: 10px; border-top: 1px dashed #ccc; padding-top: 10px;">
            <div style="display:flex;gap:5px;">
                <div class="input-group"><label>kxy [N/m]</label><input type="text" id="inp-kxy"></div>
                <div class="input-group"><label>kyx [N/m]</label><input type="text" id="inp-kyx"></div>
            </div>
            <div style="display:flex;gap:5px;">
                <div class="input-group"><label>kyy [N/m]</label><input type="text" id="inp-kyy"></div>
                <div class="input-group"><label>cxy [N.s/m]</label><input type="text" id="inp-cxy"></div>
            </div>
            <div style="display:flex;gap:5px;">
                <div class="input-group"><label>cyx [N.s/m]</label><input type="text" id="inp-cyx"></div>
                <div class="input-group"><label>cyy [N.s/m]</label><input type="text" id="inp-cyy"></div>
            </div>
            <div class="input-group"><label>Link Node #</label><input type="text" id="inp-n_link"></div>
            <div class="input-group"><label>Frequency [RPM]</label><input type="text" id="inp-frequency"></div>
            <div style="display:flex;gap:5px;">
                <div class="input-group"><label>mxx [kg]</label><input type="text" id="inp-mxx"></div>
                <div class="input-group"><label>mxy [kg]</label><input type="text" id="inp-mxy"></div>
            </div>
            <div style="display:flex;gap:5px;">
                <div class="input-group"><label>myx [kg]</label><input type="text" id="inp-myx"></div>
                <div class="input-group"><label>myy [kg]</label><input type="text" id="inp-myy"></div>
            </div>
            <div style="display:flex;gap:5px;">
                <div class="input-group"><label>kzz [N/m]</label><input type="text" id="inp-kzz"></div>
                <div class="input-group"><label>czz [N.s/m]</label><input type="text" id="inp-czz"></div>
            </div>
            <div class="input-group"><label>mzz [kg]</label><input type="text" id="inp-mzz"></div>
            <div class="input-group"><label>Scale Factor</label><input type="text" id="inp-scale_factor"></div>
            <div class="input-group"><label>Tag</label><input type="text" id="inp-tag"></div>
            <div class="input-group"><label>Hex Color</label><input type="text" id="inp-color"></div>
        </div>`,
        BallBearing: `
        <div class="input-group"><label>Node # (Optional)</label><input type="text" id="inp-n"></div>
        <div class="input-group"><label>Number of Balls</label><input type="text" id="inp-n_balls"></div>
        <div class="input-group"><label>Ball Diameter [m]</label><input type="text" id="inp-d_balls"></div>
        <div class="input-group"><label>Static Force [N]</label><input type="text" id="inp-fs"></div>
        <div class="input-group"><label>Contact Angle [rad]</label><input type="text" id="inp-alpha"></div>
        <button type="button" class="btn-advanced" onclick="toggleAdvanced(this)">Advanced <i class="fas fa-chevron-down"></i></button>
        <div class="advanced-fields" style="display: none; margin-top: 10px; border-top: 1px dashed #ccc; padding-top: 10px;">
            <div class="input-group"><label>cxx [N.s/m]</label><input type="text" id="inp-cxx"></div>
            <div class="input-group"><label>cyy [N.s/m]</label><input type="text" id="inp-cyy"></div>
            <div class="input-group"><label>Link Node #</label><input type="text" id="inp-n_link"></div>
            <div class="input-group"><label>Scale Factor</label><input type="text" id="inp-scale_factor"></div>
            <div class="input-group"><label>Tag</label><input type="text" id="inp-tag"></div>
            <div class="input-group"><label>Hex Color</label><input type="text" id="inp-color"></div>
        </div>`,
        RollerBearing: `
        <div class="input-group"><label>Node # (Optional)</label><input type="text" id="inp-n"></div>
        <div class="input-group"><label>Number of Rollers</label><input type="text" id="inp-n_rollers"></div>
        <div class="input-group"><label>Roller Length [m]</label><input type="text" id="inp-l_rollers"></div>
        <div class="input-group"><label>Static Force [N]</label><input type="text" id="inp-fs"></div>
        <div class="input-group"><label>Contact Angle [rad]</label><input type="text" id="inp-alpha"></div>
        <button type="button" class="btn-advanced" onclick="toggleAdvanced(this)">Advanced <i class="fas fa-chevron-down"></i></button>
        <div class="advanced-fields" style="display: none; margin-top: 10px; border-top: 1px dashed #ccc; padding-top: 10px;">
            <div class="input-group"><label>cxx [N.s/m]</label><input type="text" id="inp-cxx"></div>
            <div class="input-group"><label>cyy [N.s/m]</label><input type="text" id="inp-cyy"></div>
            <div class="input-group"><label>Link Node #</label><input type="text" id="inp-n_link"></div>
            <div class="input-group"><label>Scale Factor</label><input type="text" id="inp-scale_factor"></div>
            <div class="input-group"><label>Tag</label><input type="text" id="inp-tag"></div>
            <div class="input-group"><label>Hex Color</label><input type="text" id="inp-color"></div>
        </div>`,
        MagneticBearing: `
        <div class="input-group"><label>Node # (Optional)</label><input type="text" id="inp-n"></div>
        <div class="input-group"><label>Nominal Air Gap [m]</label><input type="text" id="inp-g0"></div>
        <div class="input-group"><label>Bias Current [A]</label><input type="text" id="inp-i0"></div>
        <div class="input-group"><label>Effective Pole Area [m²]</label><input type="text" id="inp-ag"></div>
        <div class="input-group"><label>Turns per Coil</label><input type="text" id="inp-nw"></div>
        <div class="input-group"><label>PID P Gain</label><input type="text" id="inp-kp_pid"></div>
        <div class="input-group"><label>PID I Gain</label><input type="text" id="inp-ki_pid"></div>
        <div class="input-group"><label>PID D Gain</label><input type="text" id="inp-kd_pid"></div>
        <button type="button" class="btn-advanced" onclick="toggleAdvanced(this)">Advanced <i class="fas fa-chevron-down"></i></button>
        <div class="advanced-fields" style="display: none; margin-top: 10px; border-top: 1px dashed #ccc; padding-top: 10px;">
            <div class="input-group"><label>Frequency [rad/s]</label><input type="text" id="inp-frequency"></div>
            <div class="input-group"><label>Pole Angular Pos. [rad]</label><input type="text" id="inp-alpha"></div>
            <div class="input-group"><label>Amp Gain</label><input type="text" id="inp-k_amp"></div>
            <div class="input-group"><label>Sensor Gain</label><input type="text" id="inp-k_sense"></div>
            <div class="input-group"><label>Cutoff Freq.</label><input type="text" id="inp-n_f"></div>
            <div class="input-group"><label>Sensor Rotation Axis</label><input type="text" id="inp-sensors_axis_rotation"></div>
            <div class="input-group"><label>Link Node #</label><input type="text" id="inp-n_link"></div>
            <div class="input-group"><label>Scale Factor</label><input type="text" id="inp-scale_factor"></div>
            <div class="input-group"><label>Tag</label><input type="text" id="inp-tag"></div>
            <div class="input-group"><label>Hex Color</label><input type="text" id="inp-color"></div>
        </div>`,
        Cylindrical: `
        <div class="input-group"><label>Node # (Optional)</label><input type="text" id="inp-n"></div>
        <div class="input-group"><label>Frequencies [RPM] (e.g. [100, 200])</label><input type="text" id="inp-speed"></div>
        <div class="input-group"><label>Gravity Load [N]</label><input type="text" id="inp-weight"></div>
        <div class="input-group"><label>Bearing Length [mm]</label><input type="text" id="inp-bearing_length"></div>
        <div class="input-group"><label>Journal Diameter [mm]</label><input type="text" id="inp-journal_diameter"></div>
        <div class="input-group"><label>Radial Clearance [mm]</label><input type="text" id="inp-radial_clearance"></div>
        <div class="input-group"><label>Oil Viscosity [Pa.s]</label><input type="text" id="inp-oil_viscosity"></div>
        <button type="button" class="btn-advanced" onclick="toggleAdvanced(this)">Advanced <i class="fas fa-chevron-down"></i></button>
        <div class="advanced-fields" style="display: none; margin-top: 10px; border-top: 1px dashed #ccc; padding-top: 10px;">
            <div class="input-group"><label>Scale Factor</label><input type="text" id="inp-scale_factor"></div>
            <div class="input-group"><label>Tag</label><input type="text" id="inp-tag"></div>
            <div class="input-group"><label>Hex Color</label><input type="text" id="inp-color"></div>
        </div>`,
        PlainJournal: `
        <div class="input-group"><label>Node # (Optional)</label><input type="text" id="inp-n"></div>
        <div class="input-group"><label>Axial Length [mm]</label><input type="text" id="inp-axial_length"></div>
        <div class="input-group"><label>Journal Radius [mm]</label><input type="text" id="inp-journal_radius"></div>
        <div class="input-group"><label>Radial Clearance [mm]</label><input type="text" id="inp-radial_clearance"></div>
        <div class="input-group"><label>Number of Pads</label><input type="text" id="inp-n_pad"></div>
        <div class="input-group"><label>Pad Arc Length [deg]</label><input type="text" id="inp-pad_arc_length"></div>
        <div class="input-group"><label>Preload</label><input type="text" id="inp-preload"></div>
        <div class="input-group"><label>Geometry (e.g. circular, lobe)</label><input type="text" id="inp-geometry"></div>
        <div class="input-group"><label>Frequencies [RPM]</label><input type="text" id="inp-frequency"></div>
        <div class="input-group"><label>Load X [N]</label><input type="text" id="inp-fxs_load"></div>
        <div class="input-group"><label>Load Y [N]</label><input type="text" id="inp-fys_load"></div>
        <div class="input-group"><label>Lubricant</label><input type="text" id="inp-lubricant"></div>
        <div class="input-group"><label>Reference Temp [°C]</label><input type="text" id="inp-reference_temperature"></div>
        <div class="input-group"><label>Groove Factor (array)</label><input type="text" id="inp-groove_factor"></div>
        <div class="input-group"><label>Circumferential Elements</label><input type="text" id="inp-elements_circumferential"></div>
        <div class="input-group"><label>Axial Elements</label><input type="text" id="inp-elements_axial"></div>
        <button type="button" class="btn-advanced" onclick="toggleAdvanced(this)">Advanced <i class="fas fa-chevron-down"></i></button>
        <div class="advanced-fields" style="display: none; margin-top: 10px; border-top: 1px dashed #ccc; padding-top: 10px;">
            <div class="input-group"><label>Sommerfeld type</label><input type="text" id="inp-sommerfeld_type"></div>
            <div class="input-group"><label>Initial guess</label><input type="text" id="inp-initial_guess"></div>
            <div class="input-group"><label>Method</label><input type="text" id="inp-method"></div>
            <div class="input-group"><label>Model type</label><input type="text" id="inp-model_type"></div>
            <div class="input-group"><label>Operating type</label><input type="text" id="inp-operating_type"></div>
            <div class="input-group"><label>Oil flow [l/min]</label><input type="text" id="inp-oil_flow_v"></div>
            <div class="input-group"><label>Oil supply pressure [Pa]</label><input type="text" id="inp-oil_supply_pressure"></div>
            <div class="input-group"><label>Reyn</label><input type="text" id="inp-Reyn"></div>
            <div class="input-group"><label>Delta turb</label><input type="text" id="inp-delta_turb"></div>
            <div class="input-group"><label>Scale Factor</label><input type="text" id="inp-scale_factor"></div>
            <div class="input-group"><label>Tag</label><input type="text" id="inp-tag"></div>
            <div class="input-group"><label>Hex Color</label><input type="text" id="inp-color"></div>
        </div>`,
        SqueezeFilm: `
        <div class="input-group"><label>Node # (Optional)</label><input type="text" id="inp-n"></div>
        <div class="input-group"><label>Frequencies [RPM]</label><input type="text" id="inp-frequency"></div>
        <div class="input-group"><label>Axial Length [mm]</label><input type="text" id="inp-axial_length"></div>
        <div class="input-group"><label>Journal Radius [mm]</label><input type="text" id="inp-journal_radius"></div>
        <div class="input-group"><label>Radial Clearance [mm]</label><input type="text" id="inp-radial_clearance"></div>
        <div class="input-group"><label>Eccentricity Ratio</label><input type="text" id="inp-eccentricity_ratio"></div>
        <div class="input-group"><label>Lubricant</label><input type="text" id="inp-lubricant"></div>
        <button type="button" class="btn-advanced" onclick="toggleAdvanced(this)">Advanced <i class="fas fa-chevron-down"></i></button>
        <div class="advanced-fields" style="display: none; margin-top: 10px; border-top: 1px dashed #ccc; padding-top: 10px;">
            <div class="input-group"><label>Geometry (e.g. groove)</label><input type="text" id="inp-geometry"></div>
            <div class="input-group"><label>Cavitation (True/False)</label><input type="text" id="inp-cavitation"></div>
            <div class="input-group"><label>Scale Factor</label><input type="text" id="inp-scale_factor"></div>
            <div class="input-group"><label>Tag</label><input type="text" id="inp-tag"></div>
            <div class="input-group"><label>Hex Color</label><input type="text" id="inp-color"></div>
        </div>`,
        ThrustPad: `
        <div class="input-group"><label>Node # (Optional)</label><input type="text" id="inp-n"></div>
        <div class="input-group"><label>Pad Inner Radius [mm]</label><input type="text" id="inp-pad_inner_radius"></div>
        <div class="input-group"><label>Pad Outer Radius [mm]</label><input type="text" id="inp-pad_outer_radius"></div>
        <div class="input-group"><label>Pad Pivot Radius [mm]</label><input type="text" id="inp-pad_pivot_radius"></div>
        <div class="input-group"><label>Pad Arc Length [deg]</label><input type="text" id="inp-pad_arc_length"></div>
        <div class="input-group"><label>Angular Pivot Position [deg]</label><input type="text" id="inp-angular_pivot_position"></div>
        <div class="input-group"><label>Oil Supply Temp [°C]</label><input type="text" id="inp-oil_supply_temperature"></div>
        <div class="input-group"><label>Lubricant</label><input type="text" id="inp-lubricant"></div>
        <div class="input-group"><label>Number of Pads</label><input type="text" id="inp-n_pad"></div>
        <div class="input-group"><label>Theta Mesh (n_theta)</label><input type="text" id="inp-n_theta"></div>
        <div class="input-group"><label>Radial Mesh (n_radial)</label><input type="text" id="inp-n_radial"></div>
        <div class="input-group"><label>Frequencies [RPM]</label><input type="text" id="inp-frequency"></div>
        <div class="input-group"><label>Equilibrium Mode</label><input type="text" id="inp-equilibrium_position_mode"></div>
        <div class="input-group"><label>Radial Inclination [rad]</label><input type="text" id="inp-radial_inclination_angle"></div>
        <div class="input-group"><label>Circumferential Inclination [rad]</label><input type="text" id="inp-circumferential_inclination_angle"></div>
        <div class="input-group"><label>Initial Film Thickness [mm]</label><input type="text" id="inp-initial_film_thickness"></div>
        <button type="button" class="btn-advanced" onclick="toggleAdvanced(this)">Advanced <i class="fas fa-chevron-down"></i></button>
        <div class="advanced-fields" style="display: none; margin-top: 10px; border-top: 1px dashed #ccc; padding-top: 10px;">
            <div class="input-group"><label>Model type</label><input type="text" id="inp-model_type"></div>
            <div class="input-group"><label>Force/Moment Tolerance</label><input type="text" id="inp-tolerance_force_moment"></div>
            <div class="input-group"><label>Initial Residual</label><input type="text" id="inp-residual_force_moment"></div>
            <div class="input-group"><label>Axial Load [N]</label><input type="text" id="inp-axial_load"></div>
            <div class="input-group"><label>Scale Factor</label><input type="text" id="inp-scale_factor"></div>
            <div class="input-group"><label>Tag</label><input type="text" id="inp-tag"></div>
            <div class="input-group"><label>Hex Color</label><input type="text" id="inp-color"></div>
        </div>`,
        TiltingPad: `
        <div class="input-group"><label>Node # (Optional)</label><input type="text" id="inp-n"></div>
        <div class="input-group"><label>Journal Diameter [mm]</label><input type="text" id="inp-journal_diameter"></div>
        <div class="input-group"><label>Preload [array]</label><input type="text" id="inp-pre_load"></div>
        <div class="input-group"><label>Pad Thickness [mm]</label><input type="text" id="inp-pad_thickness"></div>
        <div class="input-group"><label>Pad Arcs [array deg]</label><input type="text" id="inp-pad_arc"></div>
        <div class="input-group"><label>Offsets [array]</label><input type="text" id="inp-offset"></div>
        <div class="input-group"><label>Axial Length [array mm]</label><input type="text" id="inp-pad_axial_length"></div>
        <div class="input-group"><label>Lubricant</label><input type="text" id="inp-lubricant"></div>
        <div class="input-group"><label>Oil Supply Temp [°C]</label><input type="text" id="inp-oil_supply_temperature"></div>
        <div class="input-group"><label>Radial Clearance [mm]</label><input type="text" id="inp-radial_clearance"></div>
        <div class="input-group"><label>Pivot Angles [array deg]</label><input type="text" id="inp-pivot_angle"></div>
        <div class="input-group"><label>Frequencies [RPM]</label><input type="text" id="inp-frequency"></div>
        <button type="button" class="btn-advanced" onclick="toggleAdvanced(this)">Advanced <i class="fas fa-chevron-down"></i></button>
        <div class="advanced-fields" style="display: none; margin-top: 10px; border-top: 1px dashed #ccc; padding-top: 10px;">
            <div class="input-group"><label>Circumferential Vol.</label><input type="text" id="inp-nx"></div>
            <div class="input-group"><label>Axial Vol.</label><input type="text" id="inp-nz"></div>
            <div class="input-group"><label>Thermal Radial Nodes</label><input type="text" id="inp-nr_pad"></div>
            <div class="input-group"><label>Journal temperature [°C]</label><input type="text" id="inp-journal_temperature"></div>
            <div class="input-group"><label>Hot oil carry over</label><input type="text" id="inp-hot_oil_carry_over"></div>
            <div class="input-group"><label>Inlet temp tolerance</label><input type="text" id="inp-inlet_temperature_tolerance"></div>
            <div class="input-group"><label>Max inlet iterations</label><input type="text" id="inp-max_inlet_iterations"></div>
            <div class="input-group"><label>h sump [W/(m²·K)]</label><input type="text" id="inp-h_sump"></div>
            <div class="input-group"><label>k pad [W/(m·K)]</label><input type="text" id="inp-k_pad"></div>
            <div class="input-group"><label>h edge [W/(m²·K)]</label><input type="text" id="inp-h_edge"></div>
            <div class="input-group"><label>Max jtemp iter</label><input type="text" id="inp-max_jtemp_iter"></div>
            <div class="input-group"><label>jtemp error</label><input type="text" id="inp-jtemp_error"></div>
            <div class="input-group"><label>Relax t</label><input type="text" id="inp-relax_t"></div>
            <div class="input-group"><label>Max relax change</label><input type="text" id="inp-max_relax_change"></div>
            <div class="input-group"><label>Link Node #</label><input type="text" id="inp-n_link"></div>
            <div class="input-group"><label>Journal Pos X [mm]</label><input type="text" id="inp-xj"></div>
            <div class="input-group"><label>Journal Pos Y [mm]</label><input type="text" id="inp-yj"></div>
            <div class="input-group"><label>Equilibrium Type</label><input type="text" id="inp-equilibrium_type"></div>
            <div class="input-group"><label>Thermal Type</label><input type="text" id="inp-thermal_type"></div>
            <div class="input-group"><label>Eccentricity</label><input type="text" id="inp-eccentricity"></div>
            <div class="input-group"><label>Attitude Angle [deg]</label><input type="text" id="inp-attitude_angle"></div>
            <div class="input-group"><label>Ext. Loads [fx, fy]</label><input type="text" id="inp-load"></div>
            <div class="input-group"><label>Initial Pad Angles [deg]</label><input type="text" id="inp-initial_pads_angles"></div>
            <div class="input-group"><label>Solver Options {dict}</label><input type="text" id="inp-solver_options"></div>
            <div class="input-group"><label>Scale Factor</label><input type="text" id="inp-scale_factor"></div>
            <div class="input-group"><label>Tag</label><input type="text" id="inp-tag"></div>
            <div class="input-group"><label>Hex Color</label><input type="text" id="inp-color"></div>
        </div>`
    },
    seals: {
        BASIC: `
        <div class="input-group"><label>Node # (Optional)</label><input type="text" id="inp-n"></div>
        <div style="display:flex;gap:5px;">
            <div class="input-group"><label>kxx [N/m]</label><input type="text" id="inp-kxx"></div>
            <div class="input-group"><label>cxx [N.s/m]</label><input type="text" id="inp-cxx"></div>
        </div>
        <button type="button" class="btn-advanced" onclick="toggleAdvanced(this)">Advanced <i class="fas fa-chevron-down"></i></button>
        <div class="advanced-fields" style="display: none; margin-top: 10px; border-top: 1px dashed #ccc; padding-top: 10px;">
            <div class="input-group"><label>Seal Leakage</label><input type="text" id="inp-seal_leakage"></div>
            <div style="display:flex;gap:5px;">
                <div class="input-group"><label>kxy [N/m]</label><input type="text" id="inp-kxy"></div>
                <div class="input-group"><label>kyx [N/m]</label><input type="text" id="inp-kyx"></div>
            </div>
            <div style="display:flex;gap:5px;">
                <div class="input-group"><label>kyy [N/m]</label><input type="text" id="inp-kyy"></div>
                <div class="input-group"><label>cxy [N.s/m]</label><input type="text" id="inp-cxy"></div>
            </div>
            <div style="display:flex;gap:5px;">
                <div class="input-group"><label>cyx [N.s/m]</label><input type="text" id="inp-cyx"></div>
                <div class="input-group"><label>cyy [N.s/m]</label><input type="text" id="inp-cyy"></div>
            </div>
            <div class="input-group"><label>Link Node #</label><input type="text" id="inp-n_link"></div>
            <div class="input-group"><label>Frequency [RPM]</label><input type="text" id="inp-frequency"></div>
            <div style="display:flex;gap:5px;">
                <div class="input-group"><label>mxx [kg]</label><input type="text" id="inp-mxx"></div>
                <div class="input-group"><label>mxy [kg]</label><input type="text" id="inp-mxy"></div>
            </div>
            <div style="display:flex;gap:5px;">
                <div class="input-group"><label>myx [kg]</label><input type="text" id="inp-myx"></div>
                <div class="input-group"><label>myy [kg]</label><input type="text" id="inp-myy"></div>
            </div>
            <div style="display:flex;gap:5px;">
                <div class="input-group"><label>kzz [N/m]</label><input type="text" id="inp-kzz"></div>
                <div class="input-group"><label>czz [N.s/m]</label><input type="text" id="inp-czz"></div>
            </div>
            <div class="input-group"><label>mzz [kg]</label><input type="text" id="inp-mzz"></div>
            <div class="input-group"><label>Scale Factor</label><input type="text" id="inp-scale_factor"></div>
            <div class="input-group"><label>Tag</label><input type="text" id="inp-tag"></div>
            <div class="input-group"><label>Hex Color</label><input type="text" id="inp-color"></div>
        </div>`,
        HolePattern: `
        <div class="input-group"><label>Node # (Optional)</label><input type="text" id="inp-n"></div>
        <div class="input-group"><label>Shaft Radius [mm]</label><input type="text" id="inp-shaft_radius"></div>
        <div class="input-group"><label>Radial Clearance [mm]</label><input type="text" id="inp-radial_clearance"></div>
        <div class="input-group"><label>Length [mm]</label><input type="text" id="inp-length"></div>
        <div class="input-group"><label>Roughness (E/D)</label><input type="text" id="inp-roughness"></div>
        <div class="input-group"><label>Cell Length [mm]</label><input type="text" id="inp-cell_length"></div>
        <div class="input-group"><label>Cell Width [mm]</label><input type="text" id="inp-cell_width"></div>
        <div class="input-group"><label>Cell Depth [mm]</label><input type="text" id="inp-cell_depth"></div>
        <div class="input-group"><label>Inlet Pressure [Pa]</label><input type="text" id="inp-inlet_pressure"></div>
        <div class="input-group"><label>Outlet Pressure [Pa]</label><input type="text" id="inp-outlet_pressure"></div>
        <div class="input-group"><label>Inlet Temp [°C]</label><input type="text" id="inp-inlet_temperature"></div>
        <div class="input-group"><label>Frequencies [RPM]</label><input type="text" id="inp-frequency"></div>
        <div class="input-group"><label>Gas Composition {dict}</label><input type="text" id="inp-gas_composition"></div>
        <button type="button" class="btn-advanced" onclick="toggleAdvanced(this)">Advanced <i class="fas fa-chevron-down"></i></button>
        <div class="advanced-fields" style="display: none; margin-top: 10px; border-top: 1px dashed #ccc; padding-top: 10px;">
            <div class="input-group"><label>Molar Mass</label><input type="text" id="inp-molar"></div>
            <div class="input-group"><label>Gamma (Cp/Cv)</label><input type="text" id="inp-gamma"></div>
            <div class="input-group"><label>b_suther</label><input type="text" id="inp-b_suther"></div>
            <div class="input-group"><label>s_suther</label><input type="text" id="inp-s_suther"></div>
            <div class="input-group"><label>Preswirl</label><input type="text" id="inp-preswirl"></div>
            <div class="input-group"><label>Entrance Coef</label><input type="text" id="inp-entr_coef"></div>
            <div class="input-group"><label>Exit Coef</label><input type="text" id="inp-exit_coef"></div>
            <div class="input-group"><label>Whirl Ratio</label><input type="text" id="inp-whirl_ratio"></div>
            <div class="input-group"><label>Axial Discretization nz</label><input type="text" id="inp-nz"></div>
            <div class="input-group"><label>Max Iterations</label><input type="text" id="inp-max_iterations"></div>
            <div class="input-group"><label>Tolerance</label><input type="text" id="inp-tolerance"></div>
            <div class="input-group"><label>First Step Size</label><input type="text" id="inp-first_step_size"></div>
            <div class="input-group"><label>Relaxation Factor</label><input type="text" id="inp-rlx_factor"></div>
            <div class="input-group"><label>Link Node #</label><input type="text" id="inp-n_link"></div>
            <div class="input-group"><label>Scale Factor</label><input type="text" id="inp-scale_factor"></div>
            <div class="input-group"><label>Tag</label><input type="text" id="inp-tag"></div>
            <div class="input-group"><label>Hex Color</label><input type="text" id="inp-color"></div>
        </div>`,
        Labyrinth: `
        <div class="input-group"><label>Node # (Optional)</label><input type="text" id="inp-n"></div>
        <div class="input-group"><label>Shaft Radius [mm]</label><input type="text" id="inp-shaft_radius"></div>
        <div class="input-group"><label>Radial Clearance [mm]</label><input type="text" id="inp-radial_clearance"></div>
        <div class="input-group"><label>Number of Teeth (<=30)</label><input type="text" id="inp-n_teeth"></div>
        <div class="input-group"><label>Seal Pitch [mm]</label><input type="text" id="inp-pitch"></div>
        <div class="input-group"><label>Tooth Height [mm]</label><input type="text" id="inp-tooth_height"></div>
        <div class="input-group"><label>Tooth Width [mm]</label><input type="text" id="inp-tooth_width"></div>
        <div class="input-group"><label>Seal Type</label><select id="inp-seal_type"><option value="inter">INTER</option><option value="rotor">ROTOR</option><option value="stator">STATOR</option></select></div>
        <div class="input-group"><label>Inlet Pressure [Pa]</label><input type="text" id="inp-inlet_pressure"></div>
        <div class="input-group"><label>Outlet Pressure [Pa]</label><input type="text" id="inp-outlet_pressure"></div>
        <div class="input-group"><label>Inlet Temp [°C]</label><input type="text" id="inp-inlet_temperature"></div>
        <div class="input-group"><label>Frequency [RPM]</label><input type="text" id="inp-frequency"></div>
        <div class="input-group"><label>Preswirl</label><input type="text" id="inp-preswirl"></div>
        <div class="input-group"><label>Gas Composition {dict}</label><input type="text" id="inp-gas_composition"></div>
        <button type="button" class="btn-advanced" onclick="toggleAdvanced(this)">Advanced <i class="fas fa-chevron-down"></i></button>
        <div class="advanced-fields" style="display: none; margin-top: 10px; border-top: 1px dashed #ccc; padding-top: 10px;">
            <div class="input-group"><label>Molar Mass</label><input type="text" id="inp-molar"></div>
            <div class="input-group"><label>Gamma (Cp/Cv)</label><input type="text" id="inp-gamma"></div>
            <div class="input-group"><label>Temperatures tz [array]</label><input type="text" id="inp-tz"></div>
            <div class="input-group"><label>Viscosities muz [array]</label><input type="text" id="inp-muz"></div>
            <div class="input-group"><label>Analysis (FULL/LEAKAGE)</label><input type="text" id="inp-analz"></div>
            <div class="input-group"><label>Print Params (nprt)</label><input type="text" id="inp-nprt"></div>
            <div class="input-group"><label>iopt1 (Tangential Moment)</label><input type="text" id="inp-iopt1"></div>
            <div class="input-group"><label>Link Node #</label><input type="text" id="inp-n_link"></div>
            <div class="input-group"><label>Scale Factor</label><input type="text" id="inp-scale_factor"></div>
            <div class="input-group"><label>Tag</label><input type="text" id="inp-tag"></div>
            <div class="input-group"><label>Hex Color</label><input type="text" id="inp-color"></div>
        </div>`,
        Hybrid: `
        <div class="input-group"><label>Node # (Optional)</label><input type="text" id="inp-n"></div>
        <div class="input-group"><label>Shaft Radius [mm]</label><input type="text" id="inp-shaft_radius"></div>
        <div class="input-group"><label>Inlet Pressure [Pa]</label><input type="text" id="inp-inlet_pressure"></div>
        <div class="input-group"><label>Outlet Pressure [Pa]</label><input type="text" id="inp-outlet_pressure"></div>
        <div class="input-group"><label>Inlet Temp [°C]</label><input type="text" id="inp-inlet_temperature"></div>
        <div class="input-group"><label>Frequencies [RPM]</label><input type="text" id="inp-frequency"></div>
        <div class="input-group"><label>Hole-Pattern Params {dict}</label><input type="text" id="inp-hole_pattern_parameters"></div>
        <div class="input-group"><label>Labyrinth Params {dict}</label><input type="text" id="inp-labyrinth_parameters"></div>
        <div class="input-group"><label>Gas Composition {dict}</label><input type="text" id="inp-gas_composition"></div>
        <button type="button" class="btn-advanced" onclick="toggleAdvanced(this)">Advanced <i class="fas fa-chevron-down"></i></button>
        <div class="advanced-fields" style="display: none; margin-top: 10px; border-top: 1px dashed #ccc; padding-top: 10px;">
            <div class="input-group"><label>Molar Mass</label><input type="text" id="inp-molar"></div>
            <div class="input-group"><label>Gamma (Cp/Cv)</label><input type="text" id="inp-gamma"></div>
            <div class="input-group"><label>Tolerance</label><input type="text" id="inp-tolerance"></div>
            <div class="input-group"><label>Max Iterations</label><input type="text" id="inp-max_iterations"></div>
            <div class="input-group"><label>Scale Factor</label><input type="text" id="inp-scale_factor"></div>
            <div class="input-group"><label>Hex Color</label><input type="text" id="inp-color"></div>
        </div>`
    },
    couplings: {
        BASIC: `
        <div class="input-group"><label>Node # (Optional)</label><input type="text" id="inp-n"></div>
        <div style="display:flex;gap:5px;">
            <div class="input-group"><label>m_l [kg]</label><input type="text" id="inp-m_l"></div>
            <div class="input-group"><label>m_r [kg]</label><input type="text" id="inp-m_r"></div>
        </div>
        <div style="display:flex;gap:5px;">
            <div class="input-group"><label>Ip_l [kg.m²]</label><input type="text" id="inp-Ip_l"></div>
            <div class="input-group"><label>Ip_r [kg.m²]</label><input type="text" id="inp-Ip_r"></div>
        </div>
        <button type="button" class="btn-advanced" onclick="toggleAdvanced(this)">Advanced <i class="fas fa-chevron-down"></i></button>
        <div class="advanced-fields" style="display: none; margin-top: 10px; border-top: 1px dashed #ccc; padding-top: 10px;">
            <div class="input-group"><label>Link Node #</label><input type="text" id="inp-n_link"></div>
            <div style="display:flex;gap:5px;">
                <div class="input-group"><label>Id_l [kg.m²]</label><input type="text" id="inp-Id_l"></div>
                <div class="input-group"><label>Id_r [kg.m²]</label><input type="text" id="inp-Id_r"></div>
            </div>
            <div style="display:flex;gap:5px;">
                <div class="input-group"><label>kt_x</label><input type="text" id="inp-kt_x"></div>
                <div class="input-group"><label>kt_y</label><input type="text" id="inp-kt_y"></div>
                <div class="input-group"><label>kt_z</label><input type="text" id="inp-kt_z"></div>
            </div>
            <div style="display:flex;gap:5px;">
                <div class="input-group"><label>ct_x</label><input type="text" id="inp-ct_x"></div>
                <div class="input-group"><label>ct_y</label><input type="text" id="inp-ct_y"></div>
                <div class="input-group"><label>ct_z</label><input type="text" id="inp-ct_z"></div>
            </div>
            <div style="display:flex;gap:5px;">
                <div class="input-group"><label>kr_x</label><input type="text" id="inp-kr_x"></div>
                <div class="input-group"><label>kr_y</label><input type="text" id="inp-kr_y"></div>
                <div class="input-group"><label>kr_z</label><input type="text" id="inp-kr_z"></div>
            </div>
            <div style="display:flex;gap:5px;">
                <div class="input-group"><label>cr_x</label><input type="text" id="inp-cr_x"></div>
                <div class="input-group"><label>cr_y</label><input type="text" id="inp-cr_y"></div>
                <div class="input-group"><label>cr_z</label><input type="text" id="inp-cr_z"></div>
            </div>
            <div class="input-group"><label>Outer Diameter [mm]</label><input type="text" id="inp-o_d"></div>
            <div class="input-group"><label>Length [mm]</label><input type="text" id="inp-L"></div>
            <div class="input-group"><label>Scale Factor</label><input type="text" id="inp-scale_factor"></div>
            <div class="input-group"><label>Tag</label><input type="text" id="inp-tag"></div>
            <div class="input-group"><label>Hex Color</label><input type="text" id="inp-color"></div>
        </div>`
    },
    pointmasses: {
        BASIC: `
        <div class="input-group"><label>Node # (Optional)</label><input type="text" id="inp-n"></div>
        <div class="input-group"><label>Mass [kg]</label><input type="text" id="inp-m"></div>
        <button type="button" class="btn-advanced" onclick="toggleAdvanced(this)">Advanced <i class="fas fa-chevron-down"></i></button>
        <div class="advanced-fields" style="display: none; margin-top: 10px; border-top: 1px dashed #ccc; padding-top: 10px;">
            <div class="input-group"><label>mx [kg]</label><input type="text" id="inp-mx"></div>
            <div class="input-group"><label>my [kg]</label><input type="text" id="inp-my"></div>
            <div class="input-group"><label>mz [kg]</label><input type="text" id="inp-mz"></div>
            <div class="input-group"><label>Scale Factor</label><input type="text" id="inp-scale_factor"></div>
            <div class="input-group"><label>Tag</label><input type="text" id="inp-tag"></div>
            <div class="input-group"><label>Hex Color</label><input type="text" id="inp-color"></div>
        </div>`
    }
};

// Automatic form generator 'LIST'

const listBanner = `<div style="background:#e8f4fd; border-left:4px solid #2980b9; padding:10px; margin-bottom:15px; font-size:12px; color:#2c3e50;"><i class="fas fa-layer-group"></i> <b>BATCH CREATION (LIST):</b> Enter comma-separated values (e.g. <code>0.1, 0.2, 0.3</code>). The system will generate multiple individual elements at once. Single values will be automatically broadcasted (repeated) for all elements.</div>`;

for (let key in FormTemplates) {
    if (FormTemplates[key]['BASIC']) {
        FormTemplates[key]['LIST'] = listBanner + FormTemplates[key]['BASIC'].replace(/id="inp-n"/g, 'id="inp-n" placeholder="e.g. 0, 1, 2"');
    }
}

// Function for the 'Advanced' button

function toggleAdvanced(btn) {
    const div = btn.nextElementSibling;
    if(div.style.display === 'none' || div.style.display === '') {
        div.style.display = 'block';
        btn.innerHTML = 'Hide Advanced <i class="fas fa-chevron-up"></i>';
    } else {
        div.style.display = 'none';
        btn.innerHTML = 'Advanced <i class="fas fa-chevron-down"></i>';
    }
}

// Function to switch between windows without leaving others active

function switchScreen(screenId) {
    document.querySelectorAll('.screen').forEach(t => t.classList.remove('active'));
    document.getElementById(screenId).classList.add('active');
    if(screenId === 'screen-modeling' && currentTab) openTab(currentTab);
}

// 'New Rotor' Window

window.startNewRotor = function() {    
    projectData = { materials: [], shafts: [], disks: [], gears: [], couplings: [], seals: [], bearings: [], pointmasses: [] };
    currentTab = null;
    editingIndex = -1;    
    document.getElementById('element-list').innerHTML = '';
    document.getElementById('empty-message').style.display = 'block';
    document.getElementById('list-area').style.display = 'none';
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.getElementById('plot-rotor').innerHTML = '<p style="color: #888; text-align: center; margin-top: 50%;">Add at least 1 Shaft to visualize the Rotor.</p>';
    document.getElementById('analysis-list').innerHTML = '<p style="color: #888; text-align: center; margin-top: 20%;">Generated dashboards will appear here.</p>';
    switchScreen('screen-modeling');
}

// Function to hide the sidebar of the modeling

function toggleSidebar() {
    const sidebar = document.querySelector('.sidebar');
    sidebar.classList.toggle('collapsed');
    setTimeout(() => window.dispatchEvent(new Event('resize')), 300);
}

// Function to hide the sidebar of the analysis

function toggleAnalysisSidebar() {
    const sidebar = document.querySelector('.analysis-controls-panel');
    sidebar.classList.toggle('collapsed');    
    setTimeout(() => window.dispatchEvent(new Event('resize')), 300);
}

// Sidebar function

function openTab(category) {
    currentTab = category;
    document.getElementById('empty-message').style.display = 'none';
    document.getElementById('list-area').style.display = 'block';
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    event.currentTarget.classList.add('active');
    document.getElementById('tab-title').innerHTML = `<span>${category}</span> <button class="btn-help-section" onclick="openSectionHelp('${category}')" title="Help about ${category}"><i class="fas fa-question-circle"></i></button>`;
    closeForm();
    renderList();    
    if (window.innerWidth <= 768) {
        const sidebar = document.querySelector('.sidebar');
        if (!sidebar.classList.contains('collapsed')) {
            sidebar.classList.add('collapsed');
            setTimeout(() => window.dispatchEvent(new Event('resize')), 300);
        }
    }
}

// Logic for recurrent node calculation

const getEffectiveNodes = (arr) => {
    let eff = [];
    let autoNodeCounter = 0;    
    for(let item of arr) {
        if (item.n !== undefined && item.n !== null && item.n !== "") {
            let explicitVal = parseInt(item.n);
            if(isNaN(explicitVal)) {
                eff.push(autoNodeCounter);
                autoNodeCounter++;
            } else {
                eff.push(explicitVal);
            }
        } else {
            eff.push(autoNodeCounter);
            autoNodeCounter++;
        }
    }
    return eff;
};

// Function for pivot table

function renderList() {
    const container = document.getElementById('element-list');
    container.innerHTML = '';
    const currentArray = projectData[currentTab];
    const effNodes = getEffectiveNodes(currentArray);
    currentArray.forEach((item, index) => {
        let titleStr = "";        
        if (currentTab === 'materials') {
            titleStr = `MATERIAL #${index + 1}`;
            if (item.name) titleStr += ` - ${item.name}`;
        } else if (currentTab === 'couplings') {
            titleStr = `COUPLING #${index + 1}`;
            if (item.element_type && item.element_type !== 'BASIC') titleStr += ` [${item.element_type}]`;
        } else {
            let singularName = (currentTab === 'pointmasses') ? 'POINTMASS' : currentTab.slice(0, -1).toUpperCase();
            let effectiveN = effNodes[index];
            titleStr = `${singularName} #${index + 1} (Node ${effectiveN})`;
            if (item.element_type && item.element_type !== 'BASIC') titleStr += ` [${item.element_type}]`;
        }
        const div = document.createElement('div');
        div.className = 'list-item';
        div.innerHTML = `
            <div style="display:flex; align-items:center; flex:1; overflow:hidden;">
                <i class="fas fa-grip-vertical item-drag"></i>
                <span class="item-text">${titleStr}</span>
            </div>
            <div class="item-actions">
                <button class="btn-action edit" onclick="editItem(${index})" title="Edit"><i class="fas fa-pen"></i></button>
                <button class="btn-action copy" onclick="copyItem(${index})" title="Copy"><i class="fas fa-copy"></i></button>
                <button class="btn-action delete" onclick="deleteItem(${index})" title="Delete"><i class="fas fa-trash"></i></button>
            </div>
        `;
        container.appendChild(div);
    });
    new Sortable(container, {
        handle: '.item-drag',
        animation: 150,
        onEnd: function (evt) {
            const oldIdx = evt.oldIndex;
            const newIdx = evt.newIndex;
            if(oldIdx === newIdx) return;
            const item = projectData[currentTab][oldIdx];
            const isExplicit = (item.n !== undefined && item.n !== null && item.n !== "");
            if (!isExplicit) {
                if (oldIdx < newIdx) {
                    for (let i = oldIdx + 1; i <= newIdx; i++) {
                        let el = projectData[currentTab][i];
                        if (el.n !== undefined && el.n !== null && el.n !== "") {
                            el.n = String(parseInt(el.n) - 1);
                        }
                    }
                } else {
                    for (let i = newIdx; i < oldIdx; i++) {
                        let el = projectData[currentTab][i];
                        if (el.n !== undefined && el.n !== null && el.n !== "") {
                            el.n = String(parseInt(el.n) + 1);
                        }
                    }
                }
            }
            projectData[currentTab].splice(oldIdx, 1);
            projectData[currentTab].splice(newIdx, 0, item);            
            renderList();
            buildRotorLive();
        }
    });
}

// Function to open the form

function openForm(isNew = true) {
    if (isNew) { editingIndex = -1; currentSubType = 'BASIC'; }
    const subTypes = Object.keys(FormTemplates[currentTab]);    
    if (isNew && subTypes.length > 1) {
        let html = '<h4 class="subtype-header">Select Model</h4><div class="subtype-grid">';
        subTypes.forEach(type => { html += `<button class="btn-subtype" onclick="selectSubType('${type}')">${type}</button>`; });
        html += '</div><button class="btn-cancel" style="width:100%; margin-top:15px;" onclick="closeForm()">Cancel</button>';
        document.getElementById('form-fields').innerHTML = html;
        document.querySelector('.form-actions').style.display = 'none';
    } else {
        selectSubType(isNew ? 'BASIC' : projectData[currentTab][editingIndex].element_type || 'BASIC');
    }    
    document.getElementById('btn-add-item').style.display = 'none';    
    const formBox = document.getElementById('insertion-form');
    formBox.style.display = 'block';    
    if (isNew) {
        document.getElementById('list-area').appendChild(formBox);
    } else {
        const listItems = document.getElementById('element-list').children;
        if (listItems[editingIndex]) {
            listItems[editingIndex].insertAdjacentElement('afterend', formBox);
        }
    }    
    if(!document.getElementById('btn-default-form')) {
        document.querySelector('.form-actions').insertAdjacentHTML('afterbegin', `<button type="button" id="btn-default-form" class="btn-default" onclick="fillDefault()"><i class="fas fa-magic"></i> Default</button>`);
    }
}

// Function for the 'Advanced' button

function selectSubType(type) {
    currentSubType = type;
    document.getElementById('form-fields').innerHTML = injectUnits(FormTemplates[currentTab][type]);
    document.querySelector('.form-actions').style.display = 'flex';
    const matSelects = document.getElementById('form-fields').querySelectorAll('select#inp-material');
    matSelects.forEach(sel => {
        sel.innerHTML = '';
        if (projectData.materials.length === 0) {
            sel.innerHTML = '<option value="">-- No Materials Created --</option>';
        } else {
            projectData.materials.forEach(m => {
                let mName = m.name || 'MaterialCustom';
                sel.innerHTML += `<option value="${mName}">${mName}</option>`;
            });
        }
    });
    if (editingIndex >= 0) {
        const item = projectData[currentTab][editingIndex];
        const inputs = document.getElementById('form-fields').querySelectorAll('input, select');
        let hasAdvancedVal = false;
        inputs.forEach(inp => {
            const key = inp.id.replace('inp-', '');
            if (item[key] !== undefined && item[key] !== null && item[key] !== '') {
                inp.value = item[key];
                if (inp.closest('.advanced-fields')) hasAdvancedVal = true;
            }
        });
        if (hasAdvancedVal) {
            const advBtn = document.getElementById('form-fields').querySelector('.btn-advanced');
            if(advBtn) toggleAdvanced(advBtn);
        }
    }
}

// Function to close the form

function closeForm() { 
    const formBox = document.getElementById('insertion-form');
    formBox.style.display = 'none'; 
    document.getElementById('list-area').appendChild(formBox);
    document.getElementById('btn-add-item').style.display = 'block'; 
    editingIndex = -1; 
}

// Element editing function

function editItem(index) { editingIndex = index; openForm(false); }

// Copy function for element

function copyItem(index) { 
    const original = projectData[currentTab][index];
    const copiedItem = JSON.parse(JSON.stringify(original));    
    if (copiedItem.tag) {
        let baseTag = copiedItem.tag.replace(/_\d+$/, '');
        let counter = 1;
        let newTag = `${baseTag}_${counter}`;
        const tagInUse = (cTag) => projectData[currentTab].some(item => item.tag === cTag);
        while (tagInUse(newTag)) {
            counter++;
            newTag = `${baseTag}_${counter}`;
        }
        copiedItem.tag = newTag;
    }
    const isExplicit = (original.n !== undefined && original.n !== null && original.n !== "");
    if (!isExplicit) {
        copiedItem.n = ""; 
        for (let i = index + 1; i < projectData[currentTab].length; i++) {
            let el = projectData[currentTab][i];
            if (el.n !== undefined && el.n !== null && el.n !== "") {
                el.n = String(parseInt(el.n) + 1);
            }
        }
    }    
    projectData[currentTab].splice(index + 1, 0, copiedItem); 
    renderList(); 
    buildRotorLive(); 
}

// Delete function for the element

function deleteItem(index) {
    projectData[currentTab].splice(index, 1);    
    if (editingIndex === index) {
        closeForm();
    } 
    else if (editingIndex > index) {
        editingIndex--;
    }    
    renderList();
    buildRotorLive();
}

// Function to save the element

function saveItem() {
    const inputs = document.getElementById('form-fields').querySelectorAll('input, select');    
    if (currentSubType === 'LIST') {
        let parsedData = {};
        let maxLen = 0;        
        inputs.forEach(inp => {
            const key = inp.id.replace('inp-', '');
            let value = inp.value.trim();
            if (value !== '') {
                let arr = value.split(/,(?![^\[]*\])/).map(v => v.trim());
                parsedData[key] = arr;
                if (arr.length > maxLen) maxLen = arr.length;
            }
        });        
        if (maxLen === 0) {
            closeForm();
            return;
        }        
        for (let i = 0; i < maxLen; i++) {
            let newObj = { element_type: 'BASIC' };
            for (let key in parsedData) {
                let arr = parsedData[key];
                newObj[key] = arr[i] !== undefined ? arr[i] : arr[arr.length - 1];
            }
            if (newObj.tag) {
                newObj.tag = newObj.tag + "_" + (i + 1);
            }
            projectData[currentTab].push(newObj);
        }        
    } else {
        let newObject = { element_type: currentSubType };        
        inputs.forEach(inp => {
            const key = inp.id.replace('inp-', '');
            let value = inp.value.trim();
            if (value !== '') {
                newObject[key] = value; 
            }
        });
        if (editingIndex >= 0) projectData[currentTab][editingIndex] = newObject;
        else projectData[currentTab].push(newObject);
    }    
    closeForm();
    renderList();
    buildRotorLive();
}

let rotorUpdateActive = false;
let debounceTimer = null;

// Real-time rotor construction functions

function buildRotorLive() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
        _fetchRotorLive();
    }, 600); 
}

async function _fetchRotorLive() {
    const plotContainer = document.getElementById('plot-rotor');
    const infoContainer = document.getElementById('rotor-info');
    if (projectData.shafts.length === 0) {
        plotContainer.innerHTML = '<p style="color: #888; text-align: center; margin-top: 50%;">Add at least 1 Shaft to visualize the Rotor.</p>';
        if(infoContainer) infoContainer.style.opacity = '0';
        return;
    }    
    rotorUpdateActive = true;
    plotContainer.style.opacity = '0.4';
    plotContainer.style.pointerEvents = 'none';
    let loadingTimer = setTimeout(() => {
        if(rotorUpdateActive) {
            plotContainer.style.opacity = '1';
            plotContainer.innerHTML = `
                <div style="display:flex; flex-direction:column; justify-content:center; align-items:center; height:100%; min-height:400px; color: var(--text-main);">
                    <i class="fas fa-microchip fa-spin fa-3x" style="margin-bottom:15px; color: var(--accent-primary);"></i>
                    <h3 style="margin:0;">Computing Element...</h3>
                    <p style="color: var(--text-muted); text-align:center; padding:0 20px;">Using memory cache for optimization.</p>
                </div>`;
        }
    }, 500);
    try {
        const response = await fetch('http://127.0.0.1:5001/build_rotor', {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(projectData) 
        });
        const data = await response.json();        
        rotorUpdateActive = false;
        clearTimeout(loadingTimer);
        plotContainer.style.opacity = '1';
        plotContainer.style.pointerEvents = 'auto';        
        if(data.status === "success") {
            plotContainer.innerHTML = ""; 
            const fig = JSON.parse(data.plot_json);
            Plotly.newPlot('plot-rotor', fig.data, fig.layout, {responsive: true});            
            if(infoContainer) {
                document.getElementById('info-mass').innerText = data.mass.toFixed(4);
                document.getElementById('info-ip').innerText = data.ip.toFixed(4);
                infoContainer.style.opacity = '1';
            }
        } else {
            plotContainer.innerHTML = `<div style="padding:20px; color:var(--accent-danger); text-align:center;"><i class="fas fa-exclamation-triangle fa-2x"></i><br><b>Modeling Error:</b><br>${data.message}</div>`;
            if(infoContainer) infoContainer.style.opacity = '0';
        }
    } catch (e) { 
        rotorUpdateActive = false; 
        clearTimeout(loadingTimer); 
        plotContainer.style.opacity = '1';
        plotContainer.innerHTML = "<p style='color:var(--accent-danger); text-align:center; margin-top:50%;'>Server connection error.</p>"; 
        if(infoContainer) infoContainer.style.opacity = '0';
    }
}

// Function to save the rotor

function saveRotor(event) {
    if (event) event.preventDefault();
    const blob = new Blob([JSON.stringify(projectData, null, 2)], {type: "application/json"});
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = "my_rotor.json";
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
}

// Function to load the rotor

async function loadRotor(event) {
    const file = event.target.files[0]; 
    if (!file) return;    
    const reader = new FileReader();    
    reader.onload = async function(e) {
        const content = e.target.result;
        let isInterfaceJSON = false;
        currentTab = null;
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.getElementById('empty-message').style.display = 'block';
        document.getElementById('list-area').style.display = 'none';
        document.getElementById('analysis-list').innerHTML = '<p style="color: #888; text-align: center; margin-top: 20%;">Generated dashboards will appear here.</p>';
        try {
            const loaded = JSON.parse(content);
            if (loaded.shafts !== undefined || loaded.eixos !== undefined) {
                isInterfaceJSON = true;
                projectData.materials = loaded.materials || loaded.materiais || [];
                projectData.shafts = loaded.shafts || loaded.eixos || [];
                projectData.disks = loaded.disks || loaded.discos || [];
                projectData.gears = loaded.gears || loaded.engrenagens || [];
                projectData.couplings = loaded.couplings || loaded.acoplamentos || [];
                projectData.seals = loaded.seals || loaded.selos || [];
                projectData.bearings = loaded.bearings || loaded.mancais || [];
                projectData.pointmasses = loaded.pointmasses || [];                
                switchScreen('screen-modeling');
                buildRotorLive();
                alert("Interface Rotor loaded successfully!");
            }
        } catch (err) { }
        if (!isInterfaceJSON) {
            try {
                const response = await fetch('http://127.0.0.1:5001/load_ross_file', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ content: content })
                });
                const data = await response.json();
                if (data.status === 'success') {
                    projectData.materials = data.projectData.materials || [];
                    projectData.shafts = data.projectData.shafts || [];
                    projectData.disks = data.projectData.disks || [];
                    projectData.gears = data.projectData.gears || [];
                    projectData.couplings = data.projectData.couplings || [];
                    projectData.seals = data.projectData.seals || [];
                    projectData.bearings = data.projectData.bearings || [];
                    projectData.pointmasses = data.projectData.pointmasses || [];                    
                    switchScreen('screen-modeling');
                    buildRotorLive();
                    alert("Native ROSS file loaded and converted successfully!");
                } else {
                    alert("Error converting native ROSS file:\n" + data.message);
                }
            } catch (err) {
                alert("Server connection error loading ROSS file.");
            }
        }
    };
    reader.readAsText(file); 
    event.target.value = '';
}

// Function to update analysis parameters

function updateAnalysisParameters() {
    const type = document.getElementById('analysis-type').value;
    const container = document.getElementById('analysis-parameters');
    container.innerHTML = ''; 
    if(!type) return;
    const mkNum = (id, txt, val) => `<div><label>${txt}</label><input type="number" id="${id}" value="${val}" step="any"></div>`;
    const mkTxt = (id, txt, val) => `<div><label>${txt}</label><input type="text" id="${id}" value="${val}"></div>`;
    const baseVel = mkNum('param-smin','Start Speed', '0') + mkNum('param-smax','End Speed', '400') + mkNum('param-steps','Steps', '50');
    let html = '';
    if (type === 'campbell') html = baseVel;
    else if (type === 'ucs') html = mkNum('param-kmin','Min K', '1e4') + mkNum('param-kmax','Max K', '1e10') + mkNum('param-num-modes','Nº Modes', '4');
    else if (type === 'freq_response') html = baseVel + mkTxt('param-inps','Input Nodes', '0') + mkTxt('param-inp-dofs','Input DoFs', '0') + mkTxt('param-outs','Output Nodes', '0') + mkTxt('param-out-dofs','Output DoFs', '0');
    else if (type === 'modes') html = mkNum('param-speed','Shaft Speed', '0') + mkNum('param-num-modes','Nº Modes', '12') + mkNum('param-plot-idx','Mode Index', '0');
    else if (type === 'unbalance') html = baseVel + mkNum('param-node','Unbalance Node', '1') + mkNum('param-mag','Magnitude', '0.01') + mkNum('param-phase','Phase', '0.0') + mkTxt('param-probes','Probes (Nodes)', '1') + mkTxt('param-probe-dofs','Probes (DoFs)', '0');
    container.innerHTML = html;
}

const getNum = (id, def) => document.getElementById(id) && document.getElementById(id).value !== "" ? parseFloat(document.getElementById(id).value) : def;
const getTxt = (id, def) => document.getElementById(id) && document.getElementById(id).value.trim() !== "" ? document.getElementById(id).value.trim() : def;

// Analysis dashboard logic

const AnalysisDashboards = {
    campbell: [
        { id: 'speed_min', label: 'Start Speed', type: 'range', min: 0, max: 1000, step: 10, val: 0, default_unit: 'rad/s' },
        { id: 'speed_max', label: 'End Speed', type: 'range', min: 100, max: 4000, step: 10, val: 400, default_unit: 'rad/s' },
        { id: 'speed_steps', label: 'Steps', type: 'range', min: 10, max: 200, step: 1, val: 50 },
        { id: 'plot_type', label: 'Plot Type', type: 'select', options: ['Default', 'Mode Shape'], val: 'Default' },
        
        { id: 'frequencies', label: 'Nº Frequencies', type: 'number', val: 6, adv: 'analysis' },
        { id: 'frequency_type', label: 'Freq. Type', type: 'select', options: ['wd', 'wn'], val: 'wd', adv: 'analysis' },
        { id: 'torsional_analysis', label: 'Torsional', type: 'select', options: ['False', 'True'], val: 'False', adv: 'analysis' },
        
        { id: 'harmonics', label: 'Harmonics [list]', type: 'text', val: '[1]', adv: 'plot' },
        { id: 'frequency_units', label: 'Freq. Units', type: 'select', options: ['RPM', 'rad/s'], val: 'RPM', adv: 'plot' },
        { id: 'speed_units', label: 'Speed Units', type: 'select', options: ['RPM', 'rad/s'], val: 'RPM', adv: 'plot' },
        { id: 'damping_parameter', label: 'Damping Param', type: 'select', options: ['log_dec', 'damping_ratio'], val: 'log_dec', adv: 'plot' },
        { id: 'animation', label: 'Animation', type: 'select', options: ['False', 'True'], val: 'False', adv: 'plot', deps: ['Mode Shape'] }
    ],
    ucs: [
        { id: 'k_min', label: 'Min Stiffness (10^x N/m)', type: 'number', val: 4 },
        { id: 'k_max', label: 'Max Stiffness (10^x N/m)', type: 'number', val: 10 },
        { id: 'num_modes', label: 'Nº Modes', type: 'number', val: 4 },
        
        { id: 'bearing_frequency_range', label: 'Bearing Freq Range [list]', type: 'text', val: '', adv: 'analysis' },
        { id: 'synchronous', label: 'Synchronous', type: 'select', options: ['False', 'True'], val: 'False', adv: 'analysis' },
        
        { id: 'stiffness_units', label: 'Stiffness Units', type: 'text', val: 'N/m', adv: 'plot' },
        { id: 'frequency_units', label: 'Freq. Units', type: 'select', options: ['RPM', 'Hz', 'rad/s'], val: 'rad/s', adv: 'plot' }
    ],
    freq_response: [
        { id: 'speed_min', label: 'Start Speed', type: 'range', min: 0, max: 1000, step: 10, val: 0, default_unit: 'rad/s' },
        { id: 'speed_max', label: 'End Speed', type: 'range', min: 100, max: 4000, step: 10, val: 400, default_unit: 'rad/s' },
        { id: 'plot_type', label: 'Plot Type', type: 'select', options: ['Default', 'Magnitude', 'Phase', 'Polar Bode'], val: 'Default' },
        { id: 'inps', label: 'Input Probes', type: 'probe_list', val: [{node: 0, dof: 0}] },
        { id: 'outs', label: 'Output Probes', type: 'probe_list', val: [{node: 0, dof: 0}] },
        
        { id: 'modes', label: 'Modes [list]', type: 'text', val: '', adv: 'analysis' },
        { id: 'free_free', label: 'Free Free', type: 'select', options: ['False', 'True'], val: 'False', adv: 'analysis' },
        
        { id: 'frequency_units', label: 'Freq. Units', type: 'select', options: ['RPM', 'Hz', 'rad/s'], val: 'rad/s', adv: 'plot' },
        { id: 'amplitude_units', label: 'Amplitude Units', type: 'select', options: ['m/N', 'mm/N', 'microm/N', 'm/(s*N)', 'm/(s**2*N)'], val: 'm/N', adv: 'plot' },
        { id: 'phase_units', label: 'Phase Units', type: 'select', options: ['rad', 'deg'], val: 'rad', adv: 'plot', deps: ['Default', 'Phase', 'Polar Bode'] },
        { id: 'line_shape', label: 'Line Shape', type: 'select', options: ['linear', 'log'], val: 'linear', adv: 'plot', deps: ['Magnitude'] }
    ],
    modes: [
        { id: 'speed', label: 'Shaft Speed', type: 'range', min: 0, max: 2000, step: 10, val: 0, default_unit: 'rad/s' },
        { id: 'num_modes', label: 'Nº Modes', type: 'number', val: 12 },
        { id: 'plot_type', label: 'Plot Type', type: 'select', options: ['2D', '3D', 'Orbit'], val: '2D' },
        { id: 'plot_idx', label: 'Mode Index', type: 'number', val: 0 },
        
        { id: 'sparse', label: 'Sparse', type: 'select', options: ['True', 'False'], val: 'True', adv: 'analysis' },
        { id: 'synchronous', label: 'Synchronous', type: 'select', options: ['False', 'True'], val: 'False', adv: 'analysis' },
        
        { id: 'orientation', label: 'Orientation', type: 'select', options: ['major', 'x', 'y'], val: 'major', adv: 'plot', deps: ['2D'] },
        { id: 'frequency_type', label: 'Freq Type', type: 'select', options: ['wd', 'wn'], val: 'wd', adv: 'plot', deps: ['2D', '3D'] },
        { id: 'frequency_units', label: 'Freq. Units', type: 'select', options: ['RPM', 'Hz', 'rad/s'], val: 'rad/s', adv: 'plot', deps: ['2D', '3D'] },
        { id: 'damping_parameter', label: 'Damping Param', type: 'select', options: ['log_dec', 'damping_ratio'], val: 'log_dec', adv: 'plot', deps: ['2D', '3D'] },
        { id: 'length_units', label: 'Length Units', type: 'select', options: ['m', 'mm'], val: 'm', adv: 'plot', deps: ['3D'] },
        { id: 'phase_units', label: 'Phase Units', type: 'select', options: ['rad', 'deg'], val: 'rad', adv: 'plot', deps: ['3D'] },
        { id: 'animation', label: 'Animation', type: 'select', options: ['False', 'True'], val: 'False', adv: 'plot', deps: ['3D'] },
        { id: 'nodes', label: 'Nodes [list]', type: 'text', val: '', adv: 'plot', deps: ['Orbit'] }
    ],
    unbalance: [
        { id: 'speed_min', label: 'Start Speed', type: 'range', min: 0, max: 1000, step: 10, val: 0, default_unit: 'rad/s' },
        { id: 'speed_max', label: 'End Speed', type: 'range', min: 100, max: 4000, step: 10, val: 400, default_unit: 'rad/s' },
        { id: 'plot_type', label: 'Plot Type', type: 'select', options: ['Default', 'Magnitude', 'Phase', 'Bode', 'Polar Bode'], val: 'Default' },
        { id: 'unbalances', label: 'Unbalance Excitations', type: 'unbalance_list', val: [{node: 0, mag: 0.01, phase: 0}] },
        { id: 'probes', label: 'Measurement Probes', type: 'angle_probe_list', val: [{node: 0, angle: 0}] },
        
        { id: 'modes', label: 'Modes [list]', type: 'text', val: '', adv: 'analysis' },
        { id: 'probe_units', label: 'Probe Units', type: 'select', options: ['rad', 'deg'], val: 'rad', adv: 'plot' },
        { id: 'frequency_units', label: 'Freq. Units', type: 'select', options: ['RPM', 'Hz', 'rad/s'], val: 'rad/s', adv: 'plot' },
        { id: 'amplitude_units', label: 'Amplitude Units', type: 'select', options: ['m', 'mm', 'microm'], val: 'm', adv: 'plot' },
        { id: 'phase_units', label: 'Phase Units', type: 'select', options: ['rad', 'deg'], val: 'rad', adv: 'plot', deps: ['Default', 'Phase', 'Bode', 'Polar Bode'] },
        { id: 'line_shape', label: 'Line Shape', type: 'select', options: ['linear', 'log'], val: 'linear', adv: 'plot', deps: ['Magnitude'] }
    ],
    time_response: [
        { id: 'speed', label: 'Rot. Speed', type: 'range', min: 0, max: 2000, step: 10, val: 100, default_unit: 'rad/s' },
        { id: 't_max', label: 'Max Time (s)', type: 'number', min: 0.1, max: 10, step: 0.1, val: 1 },
        { id: 'steps', label: 'Time Steps', type: 'number', min: 100, max: 10000, step: 100, val: 1000 },
        { id: 'plot_type', label: 'Plot Type', type: 'select', options: ['1D', '2D', '3D', 'Frequency (DFFT)'], val: '1D' },
        { id: 'forces', label: 'Applied Forces F(t)', type: 'force_list', val: [{node: 0, dof: 0, func: "1000 * np.cos(speed * t)"}] },
        { id: 'probes', label: 'Measurement Probes', type: 'angle_probe_list', val: [{node: 0, angle: 0}] },
        
        { id: 'method', label: 'Integration Method', type: 'select', options: ['default', 'newmark'], val: 'default', adv: 'analysis' },
        { id: 'probe_units', label: 'Probe Units', type: 'select', options: ['rad', 'deg'], val: 'rad', adv: 'plot', deps: ['1D', 'Frequency (DFFT)'] },
        { id: 'displacement_units', label: 'Displacement Units', type: 'select', options: ['m', 'mm', 'microm'], val: 'm', adv: 'plot' },
        { id: 'time_units', label: 'Time Units', type: 'select', options: ['s', 'min'], val: 's', adv: 'plot', deps: ['1D'] },
        { id: 'rotor_length_units', label: 'Rotor Len Units', type: 'select', options: ['m', 'mm'], val: 'm', adv: 'plot', deps: ['3D'] },
        { id: 'frequency_units', label: 'Freq. Units', type: 'select', options: ['RPM', 'Hz', 'rad/s'], val: 'Hz', adv: 'plot', deps: ['Frequency (DFFT)'] }
    ],
    static: [
        { id: 'plot_type', label: 'Plot Type', type: 'select', options: ['Free Body Diagram', 'Deformation', 'Shearing Force', 'Bending Moment'], val: 'Free Body Diagram' },
        
        { id: 'deformation_units', label: 'Deformation Units', type: 'select', options: ['m', 'mm', 'microm'], val: 'm', adv: 'plot', deps: ['Deformation'] },
        { id: 'rotor_length_units', label: 'Rotor Len Units', type: 'select', options: ['m', 'mm'], val: 'm', adv: 'plot' },
        { id: 'force_units', label: 'Force Units', type: 'text', val: 'N', adv: 'plot', deps: ['Free Body Diagram', 'Shearing Force'] },
        { id: 'moment_units', label: 'Moment Units', type: 'text', val: 'N*m', adv: 'plot', deps: ['Bending Moment'] }
    ],
    harmonic_balance: [
        { id: 'speed', label: 'Speed', type: 'range', min: 0, max: 2000, step: 10, val: 200, default_unit: 'rad/s' },
        { id: 't_initial', label: 'Initial Time (s)', type: 'number', val: 0 },
        { id: 't_final', label: 'Final Time (s)', type: 'number', val: 0.5 },
        { id: 't_steps', label: 'Time Steps', type: 'number', val: 1001 },
        
        { id: 'hb_node', label: 'Force Node', type: 'number', val: 0 },
        { id: 'hb_magnitudes', label: 'Magnitudes [list]', type: 'text', val: '[2000]' },
        { id: 'hb_phases', label: 'Phases [list]', type: 'text', val: '[0]' },
        { id: 'hb_harmonics', label: 'Harmonics [list]', type: 'text', val: '[1]' },
        
        { id: 'probes', label: 'Measurement Probes', type: 'angle_probe_list', val: [{node: 0, angle: 0}] },
        
        { id: 'gravity', label: 'Gravity', type: 'select', options: ['False', 'True'], val: 'False', adv: 'analysis' },
        { id: 'n_harmonics', label: 'Nº Harmonics', type: 'number', val: 1, adv: 'analysis' },
        
        { id: 'amplitude_units', label: 'Amplitude Units', type: 'select', options: ['m', 'mm', 'microm'], val: 'm', adv: 'plot' },
        { id: 'frequency_units', label: 'Freq. Units', type: 'select', options: ['Hz', 'RPM', 'rad/s'], val: 'Hz', adv: 'plot' }
    ],
    clearance: [
        { id: 'speed', label: 'Speed', type: 'range', min: 0, max: 2000, step: 10, val: 600, default_unit: 'rad/s' },
        { id: 'node', label: 'Node', type: 'number', val: 0 },
        { id: 'unbalance_magnitude', label: 'Unb. Mag [list]', type: 'text', val: '[0.05]' },
        { id: 'unbalance_phase', label: 'Unb. Phase [list]', type: 'text', val: '[0]' },
        
        { id: 'frequency', label: 'Frequency [list]', type: 'text', val: '[600]', adv: 'analysis' },
        { id: 'modes', label: 'Modes [list]', type: 'text', val: '', adv: 'analysis' }
    ],
    misalignment: [
        { id: 'speed', label: 'Speed', type: 'range', min: 0, max: 2000, step: 10, val: 125.66, default_unit: 'rad/s' },
        { id: 't_initial', label: 'Initial Time (s)', type: 'number', val: 0 },
        { id: 't_final', label: 'Final Time (s)', type: 'number', val: 0.5 },
        { id: 't_steps', label: 'Time Steps', type: 'number', val: 5000 },
        { id: 'unbalances', label: 'Unbalance Excitations', type: 'unbalance_list', val: [{node: 7, mag: 5e-4, phase: -1.57}] },
        { id: 'coupling', label: 'Coupling', type: 'select', options: ['flex', 'rigid'], val: 'flex' },
        
        { id: 'n', label: 'Shaft Element (n)', type: 'number', val: 0, adv: 'analysis' },
        { id: 'mis_type', label: 'Mis. Type', type: 'select', options: ['parallel', 'angular', 'combined'], val: 'parallel', adv: 'analysis', deps: ['flex'] },
        { id: 'mis_distance_x', label: 'Mis. Dist. X (m)', type: 'text', val: '0', adv: 'analysis', deps: ['flex'] },
        { id: 'mis_distance_y', label: 'Mis. Dist. Y (m)', type: 'text', val: '2e-4', adv: 'analysis', deps: ['flex'] },
        { id: 'mis_angle', label: 'Mis. Angle (rad)', type: 'text', val: '0', adv: 'analysis', deps: ['flex'] },
        { id: 'radial_stiffness', label: 'Radial Stiff.', type: 'text', val: '1e6', adv: 'analysis', deps: ['flex'] },
        { id: 'bending_stiffness', label: 'Bending Stiff.', type: 'text', val: '1e6', adv: 'analysis', deps: ['flex'] },
        { id: 'mis_distance', label: 'Mis. Distance (m)', type: 'text', val: '2e-4', adv: 'analysis', deps: ['rigid'] },
        { id: 'input_torque', label: 'Input Torque', type: 'text', val: '0', adv: 'analysis' },
        { id: 'load_torque', label: 'Load Torque', type: 'text', val: '0', adv: 'analysis' },
        
        { id: 'plot_type', label: 'Plot Type', type: 'select', options: ['1D', '2D', '3D', 'Frequency (DFFT)'], val: '1D' },
        { id: 'probes', label: 'Measurement Probes', type: 'angle_probe_list', val: [{node: 7, angle: 0}] },
        { id: 'probe_units', label: 'Probe Units', type: 'select', options: ['rad', 'deg'], val: 'rad', adv: 'plot', deps: ['1D', 'Frequency (DFFT)'] },
        { id: 'displacement_units', label: 'Displacement Units', type: 'select', options: ['m', 'mm', 'microm'], val: 'm', adv: 'plot' },
        { id: 'time_units', label: 'Time Units', type: 'select', options: ['s', 'min'], val: 's', adv: 'plot', deps: ['1D'] },
        { id: 'rotor_length_units', label: 'Rotor Len Units', type: 'select', options: ['m', 'mm'], val: 'm', adv: 'plot', deps: ['3D'] },
        { id: 'frequency_units', label: 'Freq. Units', type: 'select', options: ['RPM', 'Hz', 'rad/s'], val: 'Hz', adv: 'plot', deps: ['Frequency (DFFT)'] }
    ],
    rubbing: [
        { id: 'speed', label: 'Speed', type: 'range', min: 0, max: 2000, step: 10, val: 125.66, default_unit: 'rad/s' },
        { id: 't_initial', label: 'Initial Time (s)', type: 'number', val: 0 },
        { id: 't_final', label: 'Final Time (s)', type: 'number', val: 0.5 },
        { id: 't_steps', label: 'Time Steps', type: 'number', val: 5000 },
        { id: 'unbalances', label: 'Unbalance Excitations', type: 'unbalance_list', val: [{node: 7, mag: 5e-4, phase: -1.57}] },
        
        { id: 'n', label: 'Shaft Element (n)', type: 'number', val: 12, adv: 'analysis' },
        { id: 'distance', label: 'Distance (m)', type: 'text', val: '7.95e-5', adv: 'analysis' },
        { id: 'contact_stiffness', label: 'Contact Stiff.', type: 'text', val: '1.1e6', adv: 'analysis' },
        { id: 'contact_damping', label: 'Contact Damp.', type: 'text', val: '40', adv: 'analysis' },
        { id: 'friction_coeff', label: 'Friction Coeff.', type: 'text', val: '0.3', adv: 'analysis' },
        { id: 'torque', label: 'Torque', type: 'select', options: ['False', 'True'], val: 'False', adv: 'analysis' },
        
        { id: 'plot_type', label: 'Plot Type', type: 'select', options: ['1D', '2D', '3D', 'Frequency (DFFT)'], val: '1D' },
        { id: 'probes', label: 'Measurement Probes', type: 'angle_probe_list', val: [{node: 7, angle: 0}] },
        { id: 'probe_units', label: 'Probe Units', type: 'select', options: ['rad', 'deg'], val: 'rad', adv: 'plot', deps: ['1D', 'Frequency (DFFT)'] },
        { id: 'displacement_units', label: 'Displacement Units', type: 'select', options: ['m', 'mm', 'microm'], val: 'm', adv: 'plot' },
        { id: 'time_units', label: 'Time Units', type: 'select', options: ['s', 'min'], val: 's', adv: 'plot', deps: ['1D'] },
        { id: 'rotor_length_units', label: 'Rotor Len Units', type: 'select', options: ['m', 'mm'], val: 'm', adv: 'plot', deps: ['3D'] },
        { id: 'frequency_units', label: 'Freq. Units', type: 'select', options: ['RPM', 'Hz', 'rad/s'], val: 'Hz', adv: 'plot', deps: ['Frequency (DFFT)'] }
    ],
    crack: [
        { id: 'speed', label: 'Speed', type: 'range', min: 0, max: 2000, step: 10, val: 125.66, default_unit: 'rad/s' },
        { id: 't_initial', label: 'Initial Time (s)', type: 'number', val: 0 },
        { id: 't_final', label: 'Final Time (s)', type: 'number', val: 0.5 },
        { id: 't_steps', label: 'Time Steps', type: 'number', val: 5000 },
        { id: 'unbalances', label: 'Unbalance Excitations', type: 'unbalance_list', val: [{node: 7, mag: 5e-4, phase: -1.57}] },
        { id: 'crack_model', label: 'Crack Model', type: 'select', options: ['Mayes', 'Gasch', 'Flex Open', 'Flex Breathing'], val: 'Mayes' },
        
        { id: 'n', label: 'Shaft Element (n)', type: 'number', val: 18, adv: 'analysis' },
        { id: 'depth_ratio', label: 'Depth Ratio', type: 'text', val: '0.2', adv: 'analysis' },
        { id: 'cross_divisions', label: 'Cross Divisions', type: 'text', val: '', adv: 'analysis' },
        
        { id: 'plot_type', label: 'Plot Type', type: 'select', options: ['1D', '2D', '3D', 'Frequency (DFFT)'], val: '1D' },
        { id: 'probes', label: 'Measurement Probes', type: 'angle_probe_list', val: [{node: 7, angle: 0}] },
        { id: 'probe_units', label: 'Probe Units', type: 'select', options: ['rad', 'deg'], val: 'rad', adv: 'plot', deps: ['1D', 'Frequency (DFFT)'] },
        { id: 'displacement_units', label: 'Displacement Units', type: 'select', options: ['m', 'mm', 'microm'], val: 'm', adv: 'plot' },
        { id: 'time_units', label: 'Time Units', type: 'select', options: ['s', 'min'], val: 's', adv: 'plot', deps: ['1D'] },
        { id: 'rotor_length_units', label: 'Rotor Len Units', type: 'select', options: ['m', 'mm'], val: 'm', adv: 'plot', deps: ['3D'] },
        { id: 'frequency_units', label: 'Freq. Units', type: 'select', options: ['RPM', 'Hz', 'rad/s'], val: 'Hz', adv: 'plot', deps: ['Frequency (DFFT)'] }
    ]
};

// Global probe builders

window.generateProbeRowHTML = function(uniqueId, id, type, node=0, dof=0) {
    return `
    <div class="probe-row" style="align-items:center;">
        <span style="font-size:11px; color:var(--text-muted);">Node:</span> 
        <input type="number" class="probe-node" value="${node}" min="0">
        <span style="font-size:11px; color:var(--text-muted); margin-left:8px;">DoF:</span> 
        <select class="probe-dof">
            <option value="0" ${dof==0?'selected':''}>x</option>
            <option value="1" ${dof==1?'selected':''}>y</option>
            <option value="2" ${dof==2?'selected':''}>z</option>
            <option value="3" ${dof==3?'selected':''}>α</option>
            <option value="4" ${dof==4?'selected':''}>β</option>
            <option value="5" ${dof==5?'selected':''}>γ</option>
        </select>
        <button type="button" class="btn-remove-probe" style="margin-left:auto;" onclick="this.parentElement.remove();"><i class="fas fa-times"></i></button>
    </div>`;
}

// Function to add a probe

window.addProbeRow = function(uniqueId, id, type) {
    const container = document.getElementById(`probe-container-${id}-${uniqueId}`);
    container.insertAdjacentHTML('beforeend', window.generateProbeRowHTML(uniqueId, id, type));
}

// Force generators

window.generateForceRowHTML = function(uniqueId, id, type, node=0, dof=0, func="1000 * np.cos(speed * t)") {
    return `
    <div class="probe-row" style="flex-direction:column; align-items:stretch; gap:8px;">
        <div style="display:flex; gap:6px; align-items:center;">
            <span style="font-size:11px; color:var(--text-muted);">Node:</span> 
            <input type="number" class="force-node" value="${node}" min="0">
            <span style="font-size:11px; color:var(--text-muted); margin-left:8px;">DoF:</span> 
            <select class="force-dof">
                <option value="0" ${dof==0?'selected':''}>x</option>
                <option value="1" ${dof==1?'selected':''}>y</option>
                <option value="2" ${dof==2?'selected':''}>z</option>
                <option value="3" ${dof==3?'selected':''}>α</option>
                <option value="4" ${dof==4?'selected':''}>β</option>
                <option value="5" ${dof==5?'selected':''}>γ</option>
            </select>
            <button type="button" class="btn-remove-probe" style="margin-left:auto;" onclick="this.parentElement.parentElement.remove();"><i class="fas fa-times"></i></button>
        </div>
        <div style="display:flex; gap:6px; align-items:center;">
            <span style="font-size:11px; color:var(--text-muted);">F(t):</span> 
            <input type="text" class="force-func" value="${func}">
        </div>
    </div>`;
}

// Function to add a force

window.addForceRow = function(uniqueId, id, type) {
    const container = document.getElementById(`force-container-${id}-${uniqueId}`);
    container.insertAdjacentHTML('beforeend', window.generateForceRowHTML(uniqueId, id, type));
}

// Unbalance generators

window.generateUnbalanceRowHTML = function(uniqueId, id, type, node=0, mag=0.01, phase=0) {
    return `
    <div class="probe-row" style="flex-direction:column; align-items:stretch; gap:8px;">
        <div style="display:flex; gap:6px; align-items:center;">
            <span style="font-size:11px; color:var(--text-muted);">Node:</span> 
            <input type="number" class="unb-node" value="${node}" min="0">
            <button type="button" class="btn-remove-probe" style="margin-left:auto;" onclick="this.parentElement.parentElement.remove();"><i class="fas fa-times"></i></button>
        </div>
        <div style="display:flex; gap:6px; align-items:center;">
            <span style="font-size:11px; color:var(--text-muted);">Mag(kg.m):</span> 
            <input type="number" class="unb-mag" value="${mag}" step="0.001" min="0">
            <span style="font-size:11px; color:var(--text-muted); margin-left:8px;">Ph(rad):</span> 
            <input type="number" class="unb-phase" value="${phase}" step="0.01">
        </div>
    </div>`;
}

// Function to add a unbalance

window.addUnbalanceRow = function(uniqueId, id, type) {
    const container = document.getElementById(`unb-container-${id}-${uniqueId}`);
    container.insertAdjacentHTML('beforeend', window.generateUnbalanceRowHTML(uniqueId, id, type));
}

// Angle Probe generators (Node + Angle)
window.generateAngleProbeRowHTML = function(uniqueId, id, type, node=0, angle=0) {
    return `
    <div class="probe-row" style="align-items:center;">
        <span style="font-size:11px; color:var(--text-muted);">Node:</span> 
        <input type="number" class="probe-node" value="${node}" min="0">
        <span style="font-size:11px; color:var(--text-muted); margin-left:8px;">Angle(rad):</span> 
        <input type="number" class="probe-angle" value="${angle}" step="0.01">
        <button type="button" class="btn-remove-probe" style="margin-left:auto;" onclick="this.parentElement.remove();"><i class="fas fa-times"></i></button>
    </div>`;
}

// Function to add a probe

window.addAngleProbeRow = function(uniqueId, id, type) {
    const container = document.getElementById(`angle-probe-container-${id}-${uniqueId}`);
    container.insertAdjacentHTML('beforeend', window.generateAngleProbeRowHTML(uniqueId, id, type));
}

// Function that evaluates when the analysis parameters have changed

const cardTimers = {};
function triggerCardUpdate(uniqueId, type) {
    clearTimeout(cardTimers[uniqueId]);
    cardTimers[uniqueId] = setTimeout(() => {
        runCardAnalysis(uniqueId, type);
    }, 500); 
}

// Auxiliary functions for advanced panels

window.toggleDashAdv = function(btn) {
    const container = btn.nextElementSibling;
    if (container.style.display === 'grid') {
        container.style.display = 'none';
        btn.innerHTML = btn.dataset.textOriginal + ' <i class="fas fa-chevron-down"></i>';
    } else {
        container.style.display = 'grid';
        btn.innerHTML = 'Hide ' + btn.dataset.textOriginal + ' <i class="fas fa-chevron-up"></i>';
    }
};

window.checkDeps = function(uniqueId) {
    const depsItems = document.querySelectorAll(`.dash-dep-${uniqueId}`);
    depsItems.forEach(item => {
        const allowed = item.dataset.deps.split(',').map(a => a.trim());
        
        const selects = document.querySelectorAll(`select[id$="-${uniqueId}"]`);
        const activeVals = Array.from(selects).map(s => s.value);

        if (allowed.some(v => activeVals.includes(v))) item.style.display = 'flex';
        else item.style.display = 'none';
    });
};

// Function to build the analysis dashboard

function buildDashboardHTML(uniqueId, type) {
    const config = AnalysisDashboards[type];
    if(!config) return '';
    
    let htmlStandard = '';
    let htmlAdvAnalysis = '';
    let htmlAdvPlot = '';
    
    config.forEach(item => {
        let html = '';
        const depsAttr = item.deps ? `data-deps="${item.deps.join(',')}" class="dash-control-group dash-dep-${uniqueId}"` : `class="dash-control-group"`;
        
        if (item.type === 'probe_list' || item.type === 'force_list' || item.type === 'unbalance_list' || item.type === 'angle_probe_list') {
            let btnFunc = 'addProbeRow'; let contId = 'probe';
            if (item.type === 'force_list') { btnFunc = 'addForceRow'; contId = 'force'; }
            else if (item.type === 'unbalance_list') { btnFunc = 'addUnbalanceRow'; contId = 'unb'; }
            else if (item.type === 'angle_probe_list') { btnFunc = 'addAngleProbeRow'; contId = 'angle-probe'; }
            
            html += `<div ${depsAttr} style="flex-direction: column; align-items: stretch; grid-column: 1 / -1;">
                <div style="display:flex; justify-content:space-between; align-items: center; width:100%; margin-bottom:8px;">
                    <label style="margin:0;">${item.label}</label>
                    <button type="button" class="btn-add-probe" onclick="${btnFunc}('${uniqueId}', '${item.id}', '${type}')"><i class="fas fa-plus"></i></button>
                </div>
                <div class="probe-list-container" id="${contId}-container-${item.id}-${uniqueId}">`;
            
            if (item.type === 'probe_list') item.val.forEach(v => { html += window.generateProbeRowHTML(uniqueId, item.id, type, v.node, v.dof); });
            else if (item.type === 'force_list') item.val.forEach(v => { html += window.generateForceRowHTML(uniqueId, item.id, type, v.node, v.dof, v.func); });
            else if (item.type === 'angle_probe_list') item.val.forEach(v => { html += window.generateAngleProbeRowHTML(uniqueId, item.id, type, v.node, v.angle); });
            else item.val.forEach(v => { html += window.generateUnbalanceRowHTML(uniqueId, item.id, type, v.node, v.mag, v.phase); });
            
            html += `</div></div>`;
        } else {
            const changeEvent = (item.id === 'plot_type' || item.id === 'coupling') ? `onchange="checkDeps('${uniqueId}')"` : ``;            
            html += `<div ${depsAttr} style="flex-direction: column; align-items: stretch; gap: 6px;">`;
            html += `<label style="margin: 0; min-width: auto;">${item.label}</label>`;
            
            if (item.type === 'range') {
                html += `<div style="display: flex; align-items: center; gap: 10px; width: 100%;">`;                
                html += `<input type="range" id="range-${item.id}-${uniqueId}" min="${item.min}" max="${item.max}" step="${item.step}" value="${item.val}" oninput="document.getElementById('num-${item.id}-${uniqueId}').value = this.value;" style="flex: 1; margin: 0;">`;
                html += `<div class="unified-input" style="width: 140px; flex-shrink: 0;">`;
                html += `<input type="number" id="num-${item.id}-${uniqueId}" value="${item.val}" oninput="document.getElementById('range-${item.id}-${uniqueId}').value = this.value;">`;
                
                if (item.default_unit && UNIT_ALTERNATIVES[item.default_unit]) {
                    html += `<select id="unit-${item.id}-${uniqueId}" class="unified-unit">`;
                    UNIT_ALTERNATIVES[item.default_unit].forEach(u => {
                        html += `<option value="${u}">${u}</option>`;
                    });
                    html += `</select>`;
                }
                html += `</div></div>`;

            } else if (item.type === 'select') {
                html += `<select id="input-${item.id}-${uniqueId}" style="width: 100%;" ${changeEvent}>`;
                item.options.forEach(opt => { html += `<option value="${opt}" ${opt===item.val?'selected':''}>${opt}</option>`; });
                html += `</select>`;

            } else {
                html += `<div class="unified-input">`;
                html += `<input type="${item.type}" id="input-${item.id}-${uniqueId}" value="${item.val}">`;
                
                if (item.default_unit && UNIT_ALTERNATIVES[item.default_unit]) {
                    html += `<select id="unit-${item.id}-${uniqueId}" class="unified-unit">`;
                    UNIT_ALTERNATIVES[item.default_unit].forEach(u => {
                        html += `<option value="${u}">${u}</option>`;
                    });
                    html += `</select>`;
                }
                html += `</div>`;
            }
            html += `</div>`;
        }
        
        if (item.adv === 'analysis') htmlAdvAnalysis += html;
        else if (item.adv === 'plot') htmlAdvPlot += html;
        else htmlStandard += html;
    });

    let finalHtml = `<div class="light-dashboard-controls">${htmlStandard}`;
    
    if (htmlAdvAnalysis) {
        finalHtml += `
        <div style="grid-column: 1 / -1;">
            <button type="button" class="btn-adv-dash" data-text-original="Advanced Analysis" onclick="toggleDashAdv(this)">Advanced Analysis <i class="fas fa-chevron-down"></i></button>
            <div class="adv-dash-container">${htmlAdvAnalysis}</div>
        </div>`;
    }
    if (htmlAdvPlot) {
        finalHtml += `
        <div style="grid-column: 1 / -1;">
            <button type="button" class="btn-adv-dash" data-text-original="Advanced Plot" onclick="toggleDashAdv(this)">Advanced Plot <i class="fas fa-chevron-down"></i></button>
            <div class="adv-dash-container" id="adv-plot-${uniqueId}">${htmlAdvPlot}</div>
        </div>`;
    }
    
    finalHtml += '</div>';
    
    setTimeout(() => { checkDeps(uniqueId); }, 100);
    
    return finalHtml;
}

// Function to hide the analysis

function toggleAnalysis(uniqueId) {
    const body = document.getElementById(`body-${uniqueId}`);
    const icon = document.getElementById(`icon-${uniqueId}`);
    if (body.style.display === 'none') { body.style.display = 'block'; icon.innerHTML = '<i class="fas fa-chevron-down"></i>'; }
    else { body.style.display = 'none'; icon.innerHTML = '<i class="fas fa-chevron-right"></i>'; }
}

// Function to delete the analysis

function deleteAnalysis(event, cardId) {
    event.stopPropagation();
    if(confirm("Are you sure you want to delete this dashboard?")) document.getElementById(cardId).remove();
}

// Function to add the analysis

async function addAnalysis(event) {
    if(event) event.preventDefault();
    const type = document.getElementById('analysis-type').value;
    if(!type) return alert("Select an analysis.");
    
    const conversionNode = document.getElementById('rotor-conversion-type');
    const conversionType = conversionNode ? conversionNode.value : '';

    let conversionBadge = '';
    if (conversionType === '4dof') {
        conversionBadge = `<span class="badge-conversion" title="Model reduced to 4 Degrees of Freedom per node">4 DoF</span>`;
    } else if (conversionType === 'torsional') {
        conversionBadge = `<span class="badge-conversion badge-torsional" title="Model reduced to Torsional DoF only">Torsional</span>`;
    }
    
    const uniqueId = Date.now() + Math.random().toString().slice(2,8);
    const plotId = 'plot-' + uniqueId;
    const cardId = 'card-' + uniqueId;
    const list = document.getElementById('analysis-list');
    if(list.innerHTML.includes('Generated dashboards will appear')) list.innerHTML = '';
    const typeNames = {
        'campbell': 'Campbell Diagram', 'ucs': 'UCS Diagram',
        'freq_response': 'Frequency Response', 'modes': 'Modal Analysis',
        'unbalance': 'Unbalance Response', 'time_response': 'Time Response',
        'static': 'Static Analysis', 'harmonic_balance': 'Harmonic Balance',
        'clearance': 'Clearance Analysis', 'misalignment': 'Misalignment Response',
        'rubbing': 'Rubbing Response', 'crack': 'Crack Response'
    };
    const title = typeNames[type] || type.toUpperCase();
    const controlsHTML = buildDashboardHTML(uniqueId, type);
    
    list.insertAdjacentHTML('afterbegin', `
        <div class="analysis-card" id="${cardId}">
            <div class="analysis-header" onclick="toggleAnalysis('${uniqueId}')">
                <span class="analysis-title"><i class="fas fa-chart-line"></i> ${title} ${conversionBadge}</span>
                <div class="analysis-actions">
                    <button class="btn-update-analysis" onclick="event.stopPropagation(); runCardAnalysis('${uniqueId}', '${type}')"><i class="fas fa-sync-alt"></i> Update</button>
                    <button class="btn-help-analysis" onclick="openAnalysisCardHelp(event, '${type}')"><i class="fas fa-question-circle"></i> Help</button>
                    <button class="btn-delete-analysis" onclick="deleteAnalysis(event, '${cardId}')"><i class="fas fa-trash"></i> Delete</button>
                    <span id="icon-${uniqueId}"><i class="fas fa-chevron-down"></i></span>
                </div>
            </div>
            <div class="analysis-body" id="body-${uniqueId}" style="padding:0; background:var(--bg-workspace); position: relative;">
                <div id="${plotId}" style="min-height: 400px; width: 100%; overflow: hidden; position:relative;"></div>
                ${controlsHTML}
            </div>
        </div>
    `);
    runCardAnalysis(uniqueId, type);
}

// Function to run the analysis

async function runCardAnalysis(uniqueId, type) {
    const plotId = 'plot-' + uniqueId;
    const div = document.getElementById(plotId);
    if(!div) return;
    const config = AnalysisDashboards[type];
    let p = {};
    config.forEach(item => {
        if (item.type === 'probe_list') {
            const container = document.getElementById(`probe-container-${item.id}-${uniqueId}`);
            const probes = [];
            container.querySelectorAll('.probe-row').forEach(row => {
                probes.push({
                    node: parseInt(row.querySelector('.probe-node').value) || 0,
                    dof: parseInt(row.querySelector('.probe-dof').value) || 0
                });
            });
            p[item.id] = probes;
        } else if (item.type === 'force_list') {
            const container = document.getElementById(`force-container-${item.id}-${uniqueId}`);
            const forces = [];
            container.querySelectorAll('.probe-row').forEach(row => {
                forces.push({
                    node: parseInt(row.querySelector('.force-node').value) || 0,
                    dof: parseInt(row.querySelector('.force-dof').value) || 0,
                    func: row.querySelector('.force-func').value || "0"
                });
            });
            p[item.id] = forces;
        } else if (item.type === 'unbalance_list') {
            const container = document.getElementById(`unb-container-${item.id}-${uniqueId}`);
            const unbList = [];
            container.querySelectorAll('.probe-row').forEach(row => {
                unbList.push({
                    node: parseInt(row.querySelector('.unb-node').value) || 0,
                    mag: parseFloat(row.querySelector('.unb-mag').value) || 0,
                    phase: parseFloat(row.querySelector('.unb-phase').value) || 0
                });
            });
            p[item.id] = unbList;
        } else if (item.type === 'angle_probe_list') {
            const container = document.getElementById(`angle-probe-container-${item.id}-${uniqueId}`);
            const angleList = [];
            container.querySelectorAll('.probe-row').forEach(row => {
                angleList.push({
                    node: parseInt(row.querySelector('.probe-node').value) || 0,
                    angle: parseFloat(row.querySelector('.probe-angle').value) || 0
                });
            });
            p[item.id] = angleList;
        } else if (item.type === 'range') {
            p[item.id] = document.getElementById(`num-${item.id}-${uniqueId}`).value;
            const unitEl = document.getElementById(`unit-${item.id}-${uniqueId}`);
            if (unitEl) p[item.id + '_unit'] = unitEl.value;
        } else if (item.type === 'number') {
            p[item.id] = document.getElementById(`input-${item.id}-${uniqueId}`).value;
            const unitEl = document.getElementById(`unit-${item.id}-${uniqueId}`);
            if (unitEl) p[item.id + '_unit'] = unitEl.value;
        } else {
            p[item.id] = document.getElementById(`input-${item.id}-${uniqueId}`).value;
        }
    });
    div.style.opacity = '0.4';
    const loadingIndicatorId = `loading-${uniqueId}`;
    let loader = document.getElementById(loadingIndicatorId);
    if(!loader) {
        loader = document.createElement('div'); 
        loader.id = loadingIndicatorId;        
        loader.innerHTML = `
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 8px;">
                <i class="fas fa-sync fa-spin fa-2x" style="color: var(--accent-primary);"></i>
                <span style="font-weight: 600; font-size: 13px;">Updating...</span>
            </div>`;
        loader.style.position = 'absolute'; 
        loader.style.top = '200px';
        loader.style.left = '50%';
        loader.style.transform = 'translate(-50%, -50%)'; 
        loader.style.zIndex = '10'; 
        loader.style.color = 'var(--text-main)';
        loader.style.background = 'var(--bg-card)';
        loader.style.padding = '15px 25px';
        loader.style.borderRadius = 'var(--radius-ui)';
        loader.style.boxShadow = 'var(--shadow-card)';
        loader.style.border = '1px solid var(--border-color)';        
        div.parentElement.appendChild(loader);
    } else {
        loader.style.display = 'block';
    }
    const conversionNode = document.getElementById('rotor-conversion-type');
    const conversionType = conversionNode ? conversionNode.value : '';
    const payload = Object.assign({ analysis_type: type, params: p, conversion_type: conversionType }, projectData);
    try {
        const resp = await fetch('http://127.0.0.1:5001/run_analysis', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload) });
        const data = await resp.json();
        div.style.opacity = '1';
        if(loader) loader.style.display = 'none';
        if(data.status === "success") {
            div.innerHTML = ""; div.rossType = type; div.rossParams = Object.assign({}, p);
            const fig = JSON.parse(data.plot_json);
            if(fig.frames) div.rossFrames = fig.frames; 
            fig.layout.autosize = true;
            Plotly.newPlot(plotId, {
                data: fig.data, 
                layout: fig.layout, 
                frames: fig.frames || [], 
                config: {responsive: true}
            });
        } else if(data.status === "info") {
            div.rossType = type; div.rossParams = Object.assign({}, p);
            div.innerHTML = `<div style="color:var(--accent-primary); text-align:center; padding: 40px 20px;"><i class="fas fa-external-link-alt fa-3x" style="margin-bottom:15px;"></i><br><span style="font-size: 15px; font-weight: 600;">${data.message}</span></div>`;
        } else {
            div.innerHTML = `<div style="color:red; text-align:center; padding: 20px;"><i class="fas fa-exclamation-triangle fa-2x"></i><br><b>Error:</b> ${data.message}</div>`;
        }
    } catch(e) {
        div.style.opacity = '1'; if(loader) loader.style.display = 'none';
        div.innerHTML = "<p style='color:red; text-align:center;'>Server Connection Error.</p>"; 
    }
}

// Function to load the analysis

function loadAnalysis(event) {
    const file = event.target.files[0]; if (!file) return;
    const reader = new FileReader();
    reader.onload = e => {
        try {
            const loaded = JSON.parse(e.target.result);
            const container = document.getElementById('analysis-list');
            if(container.innerHTML.includes('Generated dashboards will appear')) container.innerHTML = '';            
            loaded.reverse().forEach(an => {
                const uniqueId = Date.now() + Math.random().toString().slice(2,8);
                const nid = 'plot-' + uniqueId;
                const cardId = 'card-' + uniqueId;
                let controlsHTML = '';
                if(an.type && an.params) {
                    const config = AnalysisDashboards[an.type];
                    if(config) {
                        config.forEach(item => { if(an.params[item.id] !== undefined) item.val = an.params[item.id]; });
                        controlsHTML = buildDashboardHTML(uniqueId, an.type);
                    }
                }                
                let typeVal = an.type || 'campbell';                
                container.insertAdjacentHTML('afterbegin', `
                    <div class="analysis-card" id="${cardId}">
                        <div class="analysis-header" onclick="toggleAnalysis('${uniqueId}')">
                            <span class="analysis-title">${an.title} (Loaded)</span>
                            <div class="analysis-actions">
                                <button class="btn-update-analysis" onclick="event.stopPropagation(); runCardAnalysis('${uniqueId}', '${typeVal}')"><i class="fas fa-sync-alt"></i> Update</button>
                                <button class="btn-help-analysis" onclick="openAnalysisCardHelp(event, '${typeVal}')"><i class="fas fa-question-circle"></i> Help</button>
                                <button class="btn-delete-analysis" onclick="deleteAnalysis(event, '${cardId}')"><i class="fas fa-trash"></i> Delete</button>
                                <span id="icon-${uniqueId}"><i class="fas fa-chevron-down"></i></span>
                            </div>
                        </div>
                        <div class="analysis-body" id="body-${uniqueId}" style="padding:0; background:#f8f9fa; position: relative;">
                            <div id="${nid}" style="min-height: 400px; display: block; width: 100%; overflow: hidden; position:relative;"></div>
                            ${controlsHTML}
                        </div>
                    </div>
                `);                
                const divNode = document.getElementById(nid);
                if (an.type && an.params) { divNode.rossType = an.type; divNode.rossParams = an.params; }
                an.layout.autosize = true; 
                if (an.frames) divNode.rossFrames = an.frames;
                Plotly.newPlot(nid, {
                    data: an.data, 
                    layout: an.layout, 
                    frames: an.frames || [], 
                    config: {responsive: true}
                }).then(() => window.dispatchEvent(new Event('resize')));
            });
        } catch(err) { alert("Error reading JSON."); }
    };
    reader.readAsText(file); event.target.value = '';
}

// Function to load the analysis directly

function loadAnalysisDirect(event) { switchScreen('screen-analysis'); loadAnalysis(event); }

// Function to save the analysis

function saveAnalysis(event) {
    if (event) event.preventDefault();
    const cards = document.querySelectorAll('.analysis-card');
    const saved = [];
    cards.forEach(c => {
        const p = c.querySelector('div[id^="plot-"]');
        const titleEl = c.querySelector('.analysis-title');
        const titleStr = titleEl ? titleEl.innerText.trim() : "Analysis";
        
        if(p && p.rossType) {
            saved.push({ 
                title: titleStr, 
                data: p.data || [], 
                layout: p.layout || {}, 
                frames: p.rossFrames || [], 
                type: p.rossType, 
                params: p.rossParams 
            });
        }
    });
    if(saved.length===0) return alert("No analysis to save!");
    const blob = new Blob([JSON.stringify(saved)], {type: "application/json"});
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = "analyses.json";
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
}

// Function to exit the application

async function exitApplication() {
    if(confirm("Do you really want to exit ROSS Interface? The background server will be shut down.")) {
        try { await fetch('http://127.0.0.1:5001/shutdown', {method: 'POST'}); } catch(e) {} 
        window.close();
        document.body.innerHTML = "<h2 style='text-align:center; margin-top:20%; color:#2c3e50;'><i class='fas fa-power-off'></i> Server Shutdown. You may close the browser.</h2>";
    }
}

// Units mapping for python generation

const UNITS_MAPPING = {
    'Material': {'rho': 'kg/m**3', 'E': 'N/m**2', 'G_s': 'N/m**2'},
    'ShaftElement': {'L': 'mm', 'idl': 'mm', 'odl': 'mm', 'idr': 'mm', 'odr': 'mm'},
    'DiskElement': {'m': 'kg', 'Id': 'kg*m**2', 'Ip': 'kg*m**2'},
    'GearElement': {'m': 'kg', 'Id': 'kg*m**2', 'Ip': 'kg*m**2', 'base_diameter': 'mm', 'pitch_diameter': 'mm', 'pr_angle': 'deg', 'helix_angle': 'deg', 'bore_diameter': 'mm'},
    'GearElementTVMS': {'width': 'mm', 'bore_diameter': 'mm', 'module': 'mm', 'pr_angle': 'deg', 'helix_angle': 'deg'},
    'CouplingElement': {'m_l': 'kg', 'm_r': 'kg', 'Ip_l': 'kg*m**2', 'Ip_r': 'kg*m**2', 'Id_l': 'kg*m**2', 'Id_r': 'kg*m**2', 'o_d': 'mm', 'L': 'mm'},
    'BearingElement': {'kxx': 'N/m', 'kxy': 'N/m', 'kyx': 'N/m', 'kyy': 'N/m', 'kzz': 'N/m', 'cxx': 'N*s/m', 'cxy': 'N*s/m', 'cyx': 'N*s/m', 'cyy': 'N*s/m', 'czz': 'N*s/m', 'mxx': 'kg', 'mxy': 'kg', 'myx': 'kg', 'myy': 'kg', 'mzz': 'kg', 'frequency': 'RPM'},
    'BallBearingElement': {}, 'RollerBearingElement': {}, 'MagneticBearingElement': {},
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
};

// Options for different types of units

const UNIT_ALTERNATIVES = {
    'kg/m**3': ['kg/m**3', 'g/cm**3', 'lb/in**3'],
    'N/m**2': ['N/m**2', 'Pa', 'MPa', 'GPa', 'psi', 'bar'],
    'Pa': ['Pa', 'MPa', 'bar', 'psi'],
    'm': ['m', 'mm', 'cm', 'in'],
    'mm': ['mm', 'm', 'cm', 'in'],
    'kg': ['kg', 'g', 'lb'],
    'kg*m**2': ['kg*m**2', 'lb*in**2', 'g*cm**2'],
    'deg': ['deg', 'rad'],
    'rad': ['rad', 'deg'],
    'N/m': ['N/m', 'N/mm', 'lbf/in'],
    'N*s/m': ['N*s/m', 'lbf*s/in'],
    'RPM': ['RPM', 'rad/s', 'Hz'],
    'N': ['N', 'lbf', 'kN'],
    'Pa*s': ['Pa*s', 'cP'],
    'degC': ['degC', 'kelvin', 'degF'],
    'l/min': ['l/min', 'm**3/s'],
    'rad/s': ['rad/s', 'RPM', 'Hz']
};

// Function to enter the unit type

function injectUnits(htmlString) {
    let tempDiv = document.createElement('div');
    tempDiv.innerHTML = htmlString;
    
    let inputs = tempDiv.querySelectorAll('input[id^="inp-"]');
    inputs.forEach(inp => {
        let key = inp.id.replace('inp-', '');
        
        let defaultUnit = null;
        for (let cls in UNITS_MAPPING) {
            if (UNITS_MAPPING[cls][key]) {
                defaultUnit = UNITS_MAPPING[cls][key];
                break;
            }
        }

        if (defaultUnit && UNIT_ALTERNATIVES[defaultUnit]) {
            let sel = document.createElement('select');
            sel.id = `inp-${key}_unit`;
            sel.className = 'unit-select';
            
            UNIT_ALTERNATIVES[defaultUnit].forEach(u => {
                let opt = document.createElement('option');
                opt.value = u;
                opt.innerText = u;
                if (u === defaultUnit) opt.selected = true;
                sel.appendChild(opt);
            });
            
            let wrapper = document.createElement('div');
            wrapper.className = 'input-unit-wrapper';
            inp.parentNode.insertBefore(wrapper, inp);
            wrapper.appendChild(inp);
            wrapper.appendChild(sel);
            
            let label = wrapper.previousElementSibling;
            if (label && label.tagName === 'LABEL') {
                label.innerText = label.innerText.replace(/\s*\[.*?\]\s*/g, ' ').trim();
            }
        }
    });
    return tempDiv.innerHTML;
}

// Format the kwargs

function formatKwargs(obj, excludeKeys=[], className='') {
    let items = [];
    let unitMap = UNITS_MAPPING[className] || {};
    for (let key in obj) {
        if (excludeKeys.includes(key) || key.endsWith('_unit')) continue;        
        let val = obj[key];        
        if (val !== undefined && val !== null && val !== "") {
            let unit = obj[key + '_unit'] || unitMap[key];            
            if (typeof val === 'string' && isNaN(val) && !val.trim().startsWith('[') && !val.trim().startsWith('{')) {
                if (val.toLowerCase() === 'true') items.push(`${key}=True`);
                else if (val.toLowerCase() === 'false') items.push(`${key}=False`);
                else items.push(`${key}='${val}'`);
            } else {
                let finalVal = val;                
                if (typeof val === 'string' && val.trim().startsWith('[')) {
                    if (unit) finalVal = `np.array(${val})`;
                }                
                if (unit) {
                    items.push(`${key}=Q_(${finalVal}, '${unit}')`);
                } else {
                    items.push(`${key}=${finalVal}`);
                }
            }
        }
    }
    return items.join(', ');
}

const ClassMap = {
    'BASIC_gears': 'GearElement', 'TVMS': 'GearElementTVMS',
    'BASIC_bearings': 'BearingElement', 'BallBearing': 'BallBearingElement', 'RollerBearing': 'RollerBearingElement',
    'MagneticBearing': 'MagneticBearingElement', 'Cylindrical': 'CylindricalBearing', 'PlainJournal': 'PlainJournal',
    'SqueezeFilm': 'SqueezeFilmDamper', 'ThrustPad': 'ThrustPad', 'TiltingPad': 'TiltingPad',
    'BASIC_seals': 'SealElement', 'HolePattern': 'HolePatternSeal', 'Labyrinth': 'LabyrinthSeal', 'Hybrid': 'HybridSeal'
};

// Function to generate the Python file

function generatePythonFile() {
    let py = `import ross as rs\nimport numpy as np\nfrom ross.units import Q_\n`;    
    py += `\n# ==========================================\n`;
    py += `# Modeling \n`;
    py += `# ==========================================\n`;
    py += `\n# Materials \n`;
    py += `materials_dict = {}\n`;
    projectData.materials.forEach(m => {
        let matCopy = Object.assign({}, m); 
        if (matCopy.poisson !== undefined) { matCopy.Poisson = matCopy.poisson; delete matCopy.poisson; }
        let args = formatKwargs(matCopy, ['name', 'element_type'], 'Material');
        let name = matCopy.name || 'MaterialCustom';
        py += `materials_dict['${name.toLowerCase()}'] = rs.Material(name='${name}', ${args})\n`;
    });
    py += `default_mat = list(materials_dict.values())[0] if materials_dict else rs.materials.steel\n`;
    py += `\n# Shafts \n`;
    py += `shafts_data = [\n`;
    let shaftNodes = getEffectiveNodes(projectData.shafts);
    projectData.shafts.forEach((s, index) => {
        let args = formatKwargs(s, ['material', 'element_type'], 'ShaftElement');
        let effN = shaftNodes[index];
        if (!args.includes('n=')) args = `n=${effN}` + (args.length > 0 ? `, ${args}` : ``);
        let mat = s.material ? `materials_dict.get('${s.material.toLowerCase()}', default_mat)` : `default_mat`;        
        py += `    dict(${args}, material=${mat}),\n`;
    });
    py += `]\n`;
    py += `shafts = [rs.ShaftElement(**kwargs) for kwargs in shafts_data]\n`;
    
    py += `\n# Disks \n`;
    py += `disks_data = [\n`;
    let diskNodes = getEffectiveNodes(projectData.disks);
    projectData.disks.forEach((d, index) => { 
        let args = formatKwargs(d, ['element_type'], 'DiskElement');
        let effN = diskNodes[index];
        if (!args.includes('n=')) args = `n=${effN}` + (args.length > 0 ? `, ${args}` : ``);
        py += `    dict(${args}),\n`; 
    });
    py += `]\n`;
    py += `disks = [rs.DiskElement(**kwargs) for kwargs in disks_data]\n`;
    
    py += `\n# Gears \n`;
    py += `gears_data = [\n`;
    let gearNodes = getEffectiveNodes(projectData.gears);
    projectData.gears.forEach((g, index) => { 
        let cls = ClassMap[g.element_type] || ClassMap['BASIC_gears'];
        let args = formatKwargs(g, ['element_type', 'material'], cls);
        let effN = gearNodes[index];
        if (!args.includes('n=')) args = `n=${effN}` + (args.length > 0 ? `, ${args}` : ``);
        
        let mat = g.material ? `materials_dict.get('${g.material.toLowerCase()}', default_mat)` : `default_mat`;
        let argStr = args.length > 0 ? `${args}, material=${mat}` : `material=${mat}`;
        
        py += `    (rs.${cls}, dict(${argStr})),\n`; 
    });
    py += `]\n`;
    py += `gears = [cls(**kwargs) for cls, kwargs in gears_data]\n`;

    py += `\n# Bearings \n`;
    py += `bearings_data = [\n`;
    let bearingNodes = getEffectiveNodes(projectData.bearings);
    projectData.bearings.forEach((b, index) => { 
        let cls = ClassMap[b.element_type] || ClassMap['BASIC_bearings'];
        let args = formatKwargs(b, ['element_type'], cls);
        let effN = bearingNodes[index];
        if (!args.includes('n=')) args = `n=${effN}` + (args.length > 0 ? `, ${args}` : ``);
        py += `    (rs.${cls}, dict(${args})),\n`; 
    });
    py += `]\n`;
    py += `bearings = [cls(**kwargs) for cls, kwargs in bearings_data]\n`;

    py += `\n# Seals \n`;
    py += `seals_data = [\n`;
    let sealNodes = getEffectiveNodes(projectData.seals);
    projectData.seals.forEach((s, index) => { 
        let cls = ClassMap[s.element_type] || ClassMap['BASIC_seals'];
        let args = formatKwargs(s, ['element_type'], cls);
        let effN = sealNodes[index];
        if (!args.includes('n=')) args = `n=${effN}` + (args.length > 0 ? `, ${args}` : ``);
        py += `    (rs.${cls}, dict(${args})),\n`; 
    });
    py += `]\n`;
    py += `seals = [cls(**kwargs) for cls, kwargs in seals_data]\n`;

    py += `\n# Couplings \n`;
    py += `couplings_data = [\n`;
    projectData.couplings.forEach(c => { 
        py += `    dict(${formatKwargs(c, ['element_type'], 'CouplingElement')}),\n`; 
    });
    py += `]\n`;
    py += `couplings = [rs.CouplingElement(**kwargs) for kwargs in couplings_data]\n`;

    py += `\n# Point Masses \n`;
    py += `point_masses_data = [\n`;
    let pmNodes = getEffectiveNodes(projectData.pointmasses);
    projectData.pointmasses.forEach((p, index) => { 
        let args = formatKwargs(p, ['element_type'], 'PointMass');
        let effN = pmNodes[index];
        if (!args.includes('n=')) args = `n=${effN}` + (args.length > 0 ? `, ${args}` : ``);
        py += `    dict(${args}),\n`; 
    });
    py += `]\n`;
    py += `point_masses = [rs.PointMass(**kwargs) for kwargs in point_masses_data]\n`;

    py += `\n# Rotor Assembly \n`;
    py += `rotor = rs.Rotor(\n    shaft_elements=shafts + couplings,\n    disk_elements=disks + gears,\n    bearing_elements=bearings + seals,\n    point_mass_elements=point_masses\n)\n`;

    const conversionNode = document.getElementById('rotor-conversion-type');
    const conversionType = conversionNode ? conversionNode.value : '';
    
    if (conversionType === '4dof') {
        py += `rotor = rs.utils.convert_6dof_to_4dof(rotor)\n`;
        py += `print("Rotor converted to 4 DoF!")\n`;
    } else if (conversionType === 'torsional') {
        py += `rotor = rs.utils.convert_6dof_to_torsional(rotor)\n`;
        py += `print("Rotor converted to Torsional!")\n`;
    }

    py += `print("Rotor Modeled Successfully!")\nrotor.plot_rotor().show()\n\n`;

    const cards = document.querySelectorAll('.analysis-card');
    const activeAnalyses = [];
    cards.forEach(c => {
        const p = c.querySelector('div[id^="plot-"]');
        if(p && p.rossType && p.rossParams) activeAnalyses.push({ type: p.rossType, params: p.rossParams });
    });

    if(activeAnalyses.length > 0) {
        py += `# ==========================================\n`;
        py += `# Analysis \n`;
        py += `# ==========================================\n`;
        activeAnalyses.forEach((a, i) => {
            let p = a.params;            
            const getPyVal = (key, targetUnit) => {
                let val = p[key];
                if (val === undefined || val === '') return '0';
                let unit = p[key + '_unit'];
                if (unit && targetUnit) {
                    return `float(Q_(${val}, '${unit}').to('${targetUnit}').m)`;
                }
                return val; 
            };

            py += `\n# Analysis ${i+1}: ${a.type.toUpperCase()}\n`;
            
            if (a.type === 'campbell') {
                py += `speed_rads = np.linspace(${getPyVal('speed_min', 'rad/s')}, ${getPyVal('speed_max', 'rad/s')}, ${p.speed_steps || 50})\n`;
                py += `camp_${i} = rotor.run_campbell(speed_rads, frequencies=${p.frequencies || 6}, frequency_type='${p.frequency_type || 'wd'}', torsional_analysis=${p.torsional_analysis === 'True' ? 'True' : 'False'})\n`;
                
                let pArgs = [];
                if(p.frequency_units) pArgs.push(`frequency_units='${p.frequency_units}'`);
                if(p.speed_units) pArgs.push(`speed_units='${p.speed_units}'`);
                if(p.damping_parameter) pArgs.push(`damping_parameter='${p.damping_parameter}'`);
                if(p.harmonics) pArgs.push(`harmonics=${p.harmonics}`);
                
                if (p.plot_type === 'Mode Shape') {
                    if(p.animation) pArgs.push(`animation=${p.animation === 'True' ? 'True' : 'False'}`);
                    py += `camp_${i}.plot_with_mode_shape(${pArgs.join(', ')}).show()\n`;
                } else {
                    py += `camp_${i}.plot(${pArgs.join(', ')}).show()\n`;
                }
                
            } else if (a.type === 'ucs') {
                let brgFreq = p.bearing_frequency_range ? `, bearing_frequency_range=${p.bearing_frequency_range}` : '';
                py += `ucs_${i} = rotor.run_ucs(stiffness_range=(${p.k_min}, ${p.k_max}), num=50, num_modes=${p.num_modes}, synchronous=${p.synchronous === 'True' ? 'True' : 'False'}${brgFreq})\n`;
                
                let pArgs = [];
                if(p.stiffness_units) pArgs.push(`stiffness_units='${p.stiffness_units}'`);
                if(p.frequency_units) pArgs.push(`frequency_units='${p.frequency_units}'`);
                py += `ucs_${i}.plot(${pArgs.join(', ')}).show()\n`;
                
            } else if (a.type === 'freq_response') {
                py += `speed_rads = np.linspace(${getPyVal('speed_min', 'rad/s')}, ${getPyVal('speed_max', 'rad/s')}, ${p.speed_steps || 50})\n`;
                let modesArg = p.modes ? `, modes=${p.modes}` : '';
                py += `freq_${i} = rotor.run_freq_response(speed_rads${modesArg}, free_free=${p.free_free === 'True' ? 'True' : 'False'})\n`;
                py += `dofs_per_node = rotor.ndof // len(rotor.nodes)\n`;
                
                let pMethod = 'plot';
                if (p.plot_type === 'Magnitude') pMethod = 'plot_magnitude';
                else if (p.plot_type === 'Phase') pMethod = 'plot_phase';
                else if (p.plot_type === 'Polar Bode') pMethod = 'plot_polar_bode';
                
                let pArgs = [];
                if(p.frequency_units) pArgs.push(`frequency_units='${p.frequency_units}'`);
                if(p.amplitude_units) pArgs.push(`amplitude_units='${p.amplitude_units}'`);
                if(['Default', 'Phase', 'Polar Bode'].includes(p.plot_type) && p.phase_units) pArgs.push(`phase_units='${p.phase_units}'`);
                if(p.plot_type === 'Magnitude' && p.line_shape) pArgs.push(`line_shape='${p.line_shape}'`);
                
                const inps = p.inps && p.inps.length > 0 ? p.inps : [{node:0, dof:0}];
                const outs = p.outs && p.outs.length > 0 ? p.outs : [{node:0, dof:0}];
                const max_len = Math.max(inps.length, outs.length);
                
                py += `fig_freq_${i} = None\n`;
                py += `colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']\n`;
                for(let j=0; j<max_len; j++) {
                    const inp = inps[Math.min(j, inps.length-1)];
                    const out = outs[Math.min(j, outs.length-1)];
                    py += `g_inp = ${inp.node} * dofs_per_node + ${inp.dof}\n`;
                    py += `g_out = ${out.node} * dofs_per_node + ${out.dof}\n`;
                    py += `fig_temp = freq_${i}.${pMethod}(inp=g_inp, out=g_out, ${pArgs.join(', ')})\n`;
                    py += `for k, trace in enumerate(fig_temp.data):\n`;
                    py += `    trace.name = f"In(N${inp.node} D${inp.dof}) | Out(N${out.node} D${out.dof})"\n`;
                    py += `    trace.legendgroup = f"group_${j}"\n`;
                    py += `    trace.showlegend = (k == 0)\n`;
                    py += `    if hasattr(trace, 'line') and trace.line is not None: trace.line.color = colors[${j} % len(colors)]\n`;
                    py += `if fig_freq_${i} is None: fig_freq_${i} = fig_temp\n`;
                    py += `else: fig_freq_${i}.add_traces(fig_temp.data)\n`;
                }
                py += `fig_freq_${i}.show()\n`;
                
            } else if (a.type === 'modes') {
                py += `modal_${i} = rotor.run_modal(speed=${getPyVal('speed', 'rad/s')}, num_modes=${p.num_modes}, sparse=${p.sparse === 'False' ? 'False' : 'True'}, synchronous=${p.synchronous === 'True' ? 'True' : 'False'})\n`;
                
                if (p.plot_type === '3D') {
                    let pArgs = [];
                    if(p.frequency_type) pArgs.push(`frequency_type='${p.frequency_type}'`);
                    if(p.length_units) pArgs.push(`length_units='${p.length_units}'`);
                    if(p.phase_units) pArgs.push(`phase_units='${p.phase_units}'`);
                    if(p.frequency_units) pArgs.push(`frequency_units='${p.frequency_units}'`);
                    if(p.damping_parameter) pArgs.push(`damping_parameter='${p.damping_parameter}'`);
                    if(p.animation) pArgs.push(`animation=${p.animation === 'True' ? 'True' : 'False'}`);
                    py += `modal_${i}.plot_mode_3d(${p.plot_idx}, ${pArgs.join(', ')}).show()\n`;
                } else if (p.plot_type === 'Orbit') {
                    let nodesArg = p.nodes ? `nodes=${p.nodes}` : '';
                    py += `modal_${i}.plot_orbit(${p.plot_idx}, ${nodesArg}).show()\n`;
                } else {
                    let pArgs = [];
                    if(p.orientation) pArgs.push(`orientation='${p.orientation}'`);
                    if(p.frequency_type) pArgs.push(`frequency_type='${p.frequency_type}'`);
                    if(p.frequency_units) pArgs.push(`frequency_units='${p.frequency_units}'`);
                    if(p.damping_parameter) pArgs.push(`damping_parameter='${p.damping_parameter}'`);
                    py += `modal_${i}.plot_mode_2d(${p.plot_idx}, ${pArgs.join(', ')}).show()\n`;
                }
                
            } else if (a.type === 'unbalance') {
                py += `speed_rads = np.linspace(${getPyVal('speed_min', 'rad/s')}, ${getPyVal('speed_max', 'rad/s')}, 50)\n`;
                const nodes = p.unbalances && p.unbalances.length > 0 ? p.unbalances.map(u => u.node).join(', ') : '0';
                const mags = p.unbalances && p.unbalances.length > 0 ? p.unbalances.map(u => u.mag).join(', ') : '0.01';
                const phases = p.unbalances && p.unbalances.length > 0 ? p.unbalances.map(u => u.phase).join(', ') : '0.0';
                let modesArg = p.modes ? `, modes=${p.modes}` : '';
                
                py += `unb_${i} = rotor.run_unbalance_response(node=[${nodes}], unbalance_magnitude=[${mags}], unbalance_phase=[${phases}], frequency=speed_rads${modesArg})\n`;
                
                let pMethod = 'plot';
                if (p.plot_type === 'Magnitude') pMethod = 'plot_magnitude';
                else if (p.plot_type === 'Phase') pMethod = 'plot_phase';
                else if (p.plot_type === 'Bode') pMethod = 'plot_bode';
                else if (p.plot_type === 'Polar Bode') pMethod = 'plot_polar_bode';
                
                let pArgs = [];
                if(p.probe_units) pArgs.push(`probe_units='${p.probe_units}'`);
                if(p.frequency_units) pArgs.push(`frequency_units='${p.frequency_units}'`);
                if(p.amplitude_units) pArgs.push(`amplitude_units='${p.amplitude_units}'`);
                if(['Default', 'Phase', 'Bode', 'Polar Bode'].includes(p.plot_type) && p.phase_units) pArgs.push(`phase_units='${p.phase_units}'`);
                if(p.plot_type === 'Magnitude' && p.line_shape) pArgs.push(`line_shape='${p.line_shape}'`);
                
                const probesStr = p.probes && p.probes.length > 0 ? p.probes.map(pr => `rs.Probe(${pr.node}, ${pr.angle})`).join(', ') : 'rs.Probe(0, 0)';
                py += `unb_${i}.${pMethod}(probe=[${probesStr}], ${pArgs.join(', ')}).show()\n`;
                
            } else if (['time_response', 'misalignment', 'rubbing', 'crack'].includes(a.type)) {
                
                if (a.type === 'time_response') {
                    py += `speed = ${getPyVal('speed', 'rad/s')}\n`;
                    py += `t = np.linspace(0, ${p.t_max || 1.0}, ${p.steps || 1000})\n`;
                    py += `dofs_per_node = rotor.ndof // len(rotor.nodes)\n`;
                    py += `F_${i} = np.zeros((len(t), rotor.ndof))\n`; 
                    if (p.forces && p.forces.length > 0) {
                        p.forces.forEach(f => {
                            py += `n_force = min(${f.node}, len(rotor.nodes) - 1)\n`;
                            py += `g_dof = n_force * dofs_per_node + ${f.dof}\n`;
                            py += `F_${i}[:, g_dof] += ${f.func}\n`;
                        });
                    }
                    py += `resp_${i} = rotor.run_time_response(speed, F_${i}, t, method='${p.method || 'default'}')\n`;
                } 
                else {
                    py += `t_sim = np.linspace(${p.t_initial || 0}, ${p.t_final || 0.5}, ${p.t_steps || 5000})\n`;
                    const nodes = p.unbalances && p.unbalances.length > 0 ? p.unbalances.map(u => u.node).join(', ') : '0';
                    const mags = p.unbalances && p.unbalances.length > 0 ? p.unbalances.map(u => u.mag).join(', ') : '0.01';
                    const phases = p.unbalances && p.unbalances.length > 0 ? p.unbalances.map(u => u.phase).join(', ') : '0.0';
                    
                    if (a.type === 'misalignment') {
                        let kw = [];
                        kw.push(`coupling='${p.coupling || 'flex'}'`);
                        if (p.n !== undefined && p.n !== '') kw.push(`n=${p.n}`);
                        if (p.input_torque) kw.push(`input_torque=${p.input_torque}`);
                        if (p.load_torque) kw.push(`load_torque=${p.load_torque}`);
                        if (p.coupling === 'flex') {
                            if (p.mis_type) kw.push(`mis_type='${p.mis_type}'`);
                            if (p.mis_distance_x) kw.push(`mis_distance_x=${p.mis_distance_x}`);
                            if (p.mis_distance_y) kw.push(`mis_distance_y=${p.mis_distance_y}`);
                            if (p.mis_angle) kw.push(`mis_angle=${p.mis_angle}`);
                            if (p.radial_stiffness) kw.push(`radial_stiffness=${p.radial_stiffness}`);
                            if (p.bending_stiffness) kw.push(`bending_stiffness=${p.bending_stiffness}`);
                        } else {
                            if (p.mis_distance) kw.push(`mis_distance=${p.mis_distance}`);
                        }
                        py += `resp_${i} = rotor.run_misalignment(node=[${nodes}], unbalance_magnitude=[${mags}], unbalance_phase=[${phases}], speed=${getPyVal('speed', 'rad/s')}, t=t_sim, ${kw.join(', ')})\n`;
                    }
                    else if (a.type === 'rubbing') {
                        py += `resp_${i} = rotor.run_rubbing(n=${p.n || 0}, distance=${p.distance || 0}, contact_stiffness=${p.contact_stiffness || 0}, contact_damping=${p.contact_damping || 0}, friction_coeff=${p.friction_coeff || 0}, node=[${nodes}], unbalance_magnitude=[${mags}], unbalance_phase=[${phases}], speed=${getPyVal('speed', 'rad/s')}, t=t_sim, torque=${p.torque === 'True' ? 'True' : 'False'})\n`;
                    }
                    else if (a.type === 'crack') {
                        let kw = [];
                        if (p.cross_divisions) kw.push(`cross_divisions=${p.cross_divisions}`);
                        py += `resp_${i} = rotor.run_crack(n=${p.n || 0}, depth_ratio=${p.depth_ratio || 0}, node=[${nodes}], unbalance_magnitude=[${mags}], unbalance_phase=[${phases}], speed=${getPyVal('speed', 'rad/s')}, t=t_sim, crack_model='${p.crack_model || 'Mayes'}'${kw.length > 0 ? ', ' + kw.join(', ') : ''})\n`;
                    }
                }
                
                const probesStr = p.probes && p.probes.length > 0 ? p.probes.map(pr => `rs.Probe(${pr.node}, ${pr.angle})`).join(', ') : 'rs.Probe(0, 0)';
                const firstNode = p.probes && p.probes.length > 0 ? p.probes[0].node : 0;
                
                let pArgs = [];
                if(p.displacement_units) pArgs.push(`displacement_units='${p.displacement_units}'`);
                
                if (p.plot_type === 'Frequency (DFFT)') {
                    if(p.probe_units) pArgs.push(`probe_units='${p.probe_units}'`);
                    if(p.frequency_units) pArgs.push(`frequency_units='${p.frequency_units}'`);
                    py += `resp_${i}.plot_dfft(probe=[${probesStr}], ${pArgs.join(', ')}).show()\n`;
                } else if (p.plot_type === '2D') {
                    py += `resp_${i}.plot_2d(node=${firstNode}, ${pArgs.join(', ')}).show()\n`;
                } else if (p.plot_type === '3D') {
                    if(p.rotor_length_units) pArgs.push(`rotor_length_units='${p.rotor_length_units}'`);
                    py += `resp_${i}.plot_3d(${pArgs.join(', ')}).show()\n`;
                } else {
                    if(p.probe_units) pArgs.push(`probe_units='${p.probe_units}'`);
                    if(p.time_units) pArgs.push(`time_units='${p.time_units}'`);
                    py += `resp_${i}.plot_1d(probe=[${probesStr}], ${pArgs.join(', ')}).show()\n`;
                }
                
            } else if (a.type === 'static') {
                py += `static_${i} = rotor.run_static()\n`;
                
                let pArgs = [];
                if(p.rotor_length_units) pArgs.push(`rotor_length_units='${p.rotor_length_units}'`);
                
                if (p.plot_type === 'Deformation') {
                    if(p.deformation_units) pArgs.push(`deformation_units='${p.deformation_units}'`);
                    py += `static_${i}.plot_deformation(${pArgs.join(', ')}).show()\n`;
                } else if (p.plot_type === 'Shearing Force') {
                    if(p.force_units) pArgs.push(`force_units='${p.force_units}'`);
                    py += `static_${i}.plot_shearing_force(${pArgs.join(', ')}).show()\n`;
                } else if (p.plot_type === 'Bending Moment') {
                    if(p.moment_units) pArgs.push(`moment_units='${p.moment_units}'`);
                    py += `static_${i}.plot_bending_moment(${pArgs.join(', ')}).show()\n`;
                } else {
                    if(p.force_units) pArgs.push(`force_units='${p.force_units}'`);
                    py += `static_${i}.plot_free_body_diagram(${pArgs.join(', ')}).show()\n`;
                }
            
            } else if (a.type === 'harmonic_balance') {
                py += `t_hb = np.linspace(${p.t_initial || 0}, ${p.t_final || 0.5}, ${p.t_steps || 1001})\n`;
                
                py += `harmonic_forces = [{\n`;
                py += `    'node': ${p.hb_node || 0},\n`;
                py += `    'magnitudes': ${p.hb_magnitudes || '[2000]'},\n`;
                py += `    'phases': ${p.hb_phases || '[0]'},\n`;
                py += `    'harmonics': ${p.hb_harmonics || '[1]'}\n`;
                py += `}]\n`;
                
                py += `hb_${i} = rotor.run_harmonic_balance_response(speed=${getPyVal('speed', 'rad/s')}, t=t_hb, harmonic_forces=harmonic_forces, gravity=${p.gravity === 'True' ? 'True' : 'False'}, n_harmonics=${p.n_harmonics || 1})\n`;
                
                const probesStr = p.probes && p.probes.length > 0 ? p.probes.map(pr => `rs.Probe(${pr.node}, ${pr.angle})`).join(', ') : 'rs.Probe(0, 0)';
                let pArgs = [];
                if(p.amplitude_units) pArgs.push(`amplitude_units='${p.amplitude_units}'`);
                if(p.frequency_units) pArgs.push(`frequency_units='${p.frequency_units}'`);
                
                py += `hb_${i}.plot(probe=[${probesStr}], ${pArgs.join(', ')}).show()\n`;
                
            } else if (a.type === 'clearance') {
                py += `unb_mag = ${p.unbalance_magnitude || '[0.05]'}\n`;
                py += `unb_phase = ${p.unbalance_phase || '[0]'}\n`;
                let kwargs = [];
                if(p.frequency) kwargs.push(`frequency=${p.frequency}`);
                if(p.modes) kwargs.push(`modes=${p.modes}`);
                
                py += `clearance_${i} = rotor.run_clearance_analysis(speed=${getPyVal('speed', 'rad/s')}, node=${p.node}, unbalance_magnitude=unb_mag, unbalance_phase=unb_phase${kwargs.length > 0 ? ', ' + kwargs.join(', ') : ''})\n`;
                py += `clearance_${i}.plot().show()\n`;
            }
        });
    }
    
    const blob = new Blob([py], {type: "text/x-python"});
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = "my_ross_script.py";
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
}

// Help Modal System

const HelpContent = {
    general: {
        title: "<i class='fas fa-book'></i> Interface Guide",
        body: `
            <h4>Sidebar Navigation</h4>
            <p>Use the buttons on the left sidebar to switch between different modeling categories (Materials, Shafts, Disks, etc.). Use the <i class="fas fa-bars"></i> button in the topbar to hide/show the sidebar.</p>
            <h4>Adding Elements</h4>
            <p>Click the <b>+ (Add)</b> button at the bottom of the list to create a new element. You can choose different models (like BASIC, TVMS for gears, etc.) if available for that specific component.</p>
            <h4>BASIC vs LIST Creation</h4>
            <p><b>BASIC:</b> Creates a single element.<br><b>LIST:</b> Allows batch creation. Enter comma-separated values (e.g., <code>0.1, 0.2, 0.3</code>) to generate multiple elements at once. Single values will be automatically copied for all elements in the batch.</p>
            <h4>Managing the List</h4>
            <ul>
                <li><b>Confirm:</b> Click <i class="fas fa-check"></i> (green button) to save the element.</li>
                <li><b>Cancel:</b> Click <i class="fas fa-times"></i> to discard changes.</li>
                <li><b>Edit:</b> Click <i class="fas fa-pen"></i> to modify an existing element.</li>
                <li><b>Copy:</b> Click <i class="fas fa-copy"></i> to duplicate an element exactly below it.</li>
                <li><b>Delete:</b> Click <i class="fas fa-trash"></i> to remove an element permanently.</li>
            </ul>
            <h4>Export & Save Actions</h4>
            <ul>
                <li><b><i class="fab fa-python"></i> Generate Python:</b> Exports your entire rotor model and active analyses into a ready-to-run Python script.</li>
                <li><b><i class="fas fa-save"></i> Save Rotor:</b> Saves your current rotor configuration as a .json file so you can load it later without losing progress.</li>
            </ul>
        `
    },
    analysis: {
        title: "<i class='fas fa-chart-line'></i> Analysis Guide",
        body: `
            <h4>Choosing an Analysis</h4>
            <p>Use the dropdown menu on the left sidebar to select the type of analysis you want to perform (e.g., Campbell Diagram, Unbalance Response). Click <b>⚙️ Add Dashboard Card</b> to generate it.</p>
            <h4>Using Dashboards</h4>
            <p>Each dashboard is an independent card containing a plot and its specific control parameters. <b>Manual Update System:</b> To prevent lag while setting up multiple parameters, the plot does not update automatically. Whenever you change values (like speed, node, or stiffness), you must click the <b><i class="fas fa-sync-alt"></i> Update</b> button at the top of the card to run the simulation and refresh the graph.</p>
            <h4>Managing Dashboards</h4>
            <ul>
                <li><b>Minimize/Expand:</b> Click the <i class="fas fa-chevron-down"></i> icon on the right side of the card header (or click the header itself) to hide or show the dashboard content.</li>
                <li><b>Delete:</b> Click the <i class="fas fa-trash"></i> <b>Delete</b> button to permanently remove that specific analysis card.</li>
            </ul>
            <h4>Export & Save Actions</h4>
            <ul>
                <li><b><i class="fab fa-python"></i> Generate Python:</b> Exports your rotor model and all currently active analysis cards into a Python script.</li>
                <li><b><i class="fas fa-save"></i> Save Analysis:</b> Saves all your current dashboard cards (and their parameters) as a .json file.</li>
                <li><b><i class="fas fa-folder-open"></i> Load:</b> Loads previously saved analysis dashboards from a .json file so you don't have to configure them again.</li>
            </ul>
        `
    },
    materials: {
        title: "<i class='fas fa-cube'></i> Materials Help",
        body: "<p>Define the physical properties of the materials used in your rotor.</p><h4>Key Parameters:</h4><ul><li><b>Density (rho):</b> Material density [kg/m³].</li><li><b>Elastic Modulus (E):</b> Young's Modulus [Pa].</li><li><b>Shear Modulus (G_s):</b> Modulus of rigidity [Pa].</li><li><b>Poisson's Ratio:</b> Transverse strain ratio.</li></ul><p><i>Note: You can assign a custom hex color for visualization in the advanced options.</i></p>"
    },
    shafts: {
        title: "<i class='fas fa-grip-lines'></i> Shafts Help",
        body: "<p>Shaft elements connect nodes and provide stiffness, mass, and gyroscopic effects to the rotor.</p><h4>Key Parameters:</h4><ul><li><b>Length (L):</b> Axial length of the element [mm].</li><li><b>Outer/Inner Diameters (odl, idl, odr, idr):</b> Diameters at the left (l) and right (r) sides [mm].</li><li><b>Material:</b> Link to a previously defined material name.</li></ul>"
    },
    disks: {
        title: "<i class='fas fa-compact-disc'></i> Disks Help",
        body: "<p>Disks represent concentrated masses on the rotor, like impellers, couplings hubs, or turbine blades.</p><h4>Key Parameters:</h4><ul><li><b>Mass (m):</b> Disk mass [kg].</li><li><b>Polar Inertia (Ip):</b> Inertia around the rotational axis [kg.m²].</li><li><b>Diametral Inertia (Id):</b> Inertia around the transverse axis [kg.m²].</li></ul>"
    },
    gears: {
        title: "<i class='fas fa-cog'></i> Gears Help",
        body: "<p>Gears act as disks but can also include meshing stiffness effects. The <b>TVMS</b> model accounts for Time-Varying Mesh Stiffness.</p><h4>Key Parameters:</h4><ul><li><b>Number of Teeth (n_teeth):</b> Gear teeth count.</li><li><b>Pitch/Base Diameter:</b> Gear sizing [mm].</li><li><b>Pressure & Helix Angles:</b> Meshing geometry [deg].</li></ul>"
    },
    couplings: {
        title: "<i class='fas fa-link'></i> Couplings Help",
        body: "<p>Couplings connect two different shaft sections or rotors. They add mass, inertia, and can transmit forces based on defined stiffness and damping.</p><h4>Key Parameters:</h4><ul><li><b>m_l / m_r:</b> Mass assigned to the left/right node [kg].</li><li><b>Ip_l / Ip_r:</b> Polar inertia assigned to the left/right node [kg.m²].</li><li><b>Stiffness/Damping:</b> Advanced parameters to define connection flexibility.</li></ul>"
    },
    bearings: {
        title: "<i class='fas fa-circle-notch'></i> Bearings Help",
        body: "<p>Bearings support the rotor and dictate its dynamic behavior. You can use a <b>BASIC</b> spring-damper model or select specific geometries like Cylindrical, Tilting Pad, etc.</p><h4>Key Parameters (BASIC):</h4><ul><li><b>kxx, kyy, kxy, kyx:</b> Direct and cross-coupled stiffness coefficients [N/m].</li><li><b>cxx, cyy, cxy, cyx:</b> Direct and cross-coupled damping coefficients [N.s/m].</li></ul>"
    },
    seals: {
        title: "<i class='fas fa-ring'></i> Seals Help",
        body: "<p>Seals prevent fluid leakage but introduce cross-coupled forces that can cause instability (like the Lomakin effect). Models include Labyrinth, Hole-Pattern, and Hybrid.</p><h4>Key Parameters:</h4><ul><li><b>Clearance / Radius:</b> Seal geometry [mm].</li><li><b>Inlet/Outlet Pressures:</b> Operating conditions [Pa].</li><li><b>Fluid Properties:</b> Advanced tabs usually contain gas composition and temperatures.</li></ul>"
    },
    pointmasses: {
        title: "<i class='fas fa-dot-circle'></i> Point Masses Help",
        body: "<p>A simple concentrated mass attached to a single node, useful for modeling unbalanced weights or small components.</p><h4>Key Parameters:</h4><ul><li><b>Mass (m):</b> Total mass [kg].</li><li><b>mx, my, mz:</b> Advanced options for asymmetric mass properties.</li></ul>"
    },
    campbell: {
        title: "<i class='fas fa-chart-line'></i> Campbell Diagram Help",
        body: `
            <p>A Campbell Diagram displays the system's natural frequencies as a function of the rotor's rotational speed. It is essential for tracking gyroscopic effects and predicting critical speeds.</p>
            <h4>Key Parameters:</h4>
            <ul>
                <li><b>Start/End Speed:</b> Rotational speed range boundaries for the analysis sweep [rad/s].</li>
                <li><b>Steps:</b> Number of evaluation intervals. Higher values generate smoother curves but slightly increase computation time.</li>
            </ul>
        `
    },
    ucs: {
        title: "<i class='fas fa-map-marked-alt'></i> UCS Diagram Help",
        body: `
            <p>The Undamped Critical Speed (UCS) Map plots the rotor's natural frequencies against a broad logarithmic spectrum of support stiffness. It helps engineers identify optimal stiffness targets for bearings configuration.</p>
            <h4>Key Parameters:</h4>
            <ul>
                <li><b>Min/Max Stiffness:</b> Bearing stiffness boundaries defined exponentially (10^x N/m). For example, 4 equals 10^4 N/m and 10 equals 10^10 N/m.</li>
                <li><b>Nº Modes:</b> Total number of operational vibration modes solved and displayed on the plot grid.</li>
            </ul>
        `
    },
    freq_response: {
        title: "<i class='fas fa-wave-square'></i> Frequency Response Help",
        body: `
            <p>Calculates the steady-state harmonic response of the assembly across a frequency range, plotting both amplitude and phase changes triggered by direct force excitations.</p>
            <h4>Key Parameters:</h4>
            <ul>
                <li><b>Start/End Speed:</b> Frequency sweep interval boundaries [rad/s].</li>
                <li><b>Input Probes:</b> Node positions and specific degrees of freedom (DoF) where external harmonic forces are applied.</li>
                <li><b>Output Probes:</b> Node positions and specific degrees of freedom (DoF) monitored by displacement sensors to render the output response.</li>
            </ul>
        `
    },
    modes: {
        title: "<i class='fas fa-project-diagram'></i> Vibration Modes Help",
        body: `
            <p>Performs a modal eigenvalue analysis at a static rotor speed to extract and display the specific deflected shape (mode shape) of the shaft structure.</p>
            <h4>Key Parameters:</h4>
            <ul>
                <li><b>Shaft Speed:</b> Constant operational speed at which the gyroscopic and stiffness matrices are evaluated [rad/s].</li>
                <li><b>Nº Modes:</b> Total number of modal points to resolve.</li>
                <li><b>Mode Index:</b> The exact index of the mode shape to render on screen (0 represents the 1st natural mode, 1 the 2nd mode, etc.).</li>
                <li><b>Plot Type:</b> Toggles visualization layout format between 2D or 3D isometric views.</li>
            </ul>
        `
    },
    unbalance: {
        title: "<i class='fas fa-balance-scale-left'></i> Unbalance Response Help",
        body: `
            <p>Simulates the synchronous dynamic behavior of the rotor under forces generated by physical mass eccentricities distributed along the rotor profile.</p>
            <h4>Key Parameters:</h4>
            <ul>
                <li><b>Start/End Speed:</b> Rotational speed scanning interval [rad/s].</li>
                <li><b>Unbalance Excitations:</b> Target node containing the unbalance property, specifying its magnitude [kg.m] and spatial phase offset [rad].</li>
                <li><b>Measurement Probes:</b> Target nodes and DoFs where virtual probes measure amplitude outputs.</li>
            </ul>
        `
    },
    time_response: {
        title: "<i class='fas fa-hourglass-half'></i> Time Response Help",
        body: `
            <p>Integrates the system equations of motion step-by-step over a timeline, letting you test generic, transient, or custom non-harmonic forces.</p>
            <h4>Key Parameters:</h4>
            <ul>
                <li><b>Rot. Speed:</b> Constant rotational speed during the physical simulation window [rad/s].</li>
                <li><b>Max Time / Time Steps:</b> Total duration [s] and discretization grid density of the simulation timeline.</li>
                <li><b>Applied Forces F(t):</b> Algebraic formula establishing forces. You can write equations using 't' (time), 'speed', or numpy functions (e.g., <code>1000 * np.cos(speed * t)</code>).</li>
                <li><b>Measurement Probes:</b> Target nodes and DoFs mapped by virtual probes.</li>
                <li><b>Plot Type:</b> Choose between 1D Time histories, 2D Trajectories/Orbits, 3D orbits, or DFFT spectral signatures.</li>
            </ul>
        `
    },
    static: {
        title: "<i class='fas fa-compress-arrows-alt'></i> Static Analysis Help",
        body: `
            <p>Evaluates structural static deflections, internal shear stresses, and bending moments due to dead-weight (gravity) and fixed static boundary conditions.</p>
            <h4>Key Parameters:</h4>
            <ul>
                <li><b>Plot Type:</b> Selects the diagram layer representation format: Free Body Diagram, Static Deformation curve, Shearing Force profile, or Bending Moment distribution.</li>
            </ul>
        `
    }
};

// Functions for opening modals
function openGeneralHelp() {
    document.getElementById('help-modal-title').innerHTML = HelpContent.general.title;
    document.getElementById('help-modal-body').innerHTML = HelpContent.general.body;
    document.getElementById('help-modal-overlay').style.display = 'flex';
}

function openAnalysisHelp() {
    document.getElementById('help-modal-title').innerHTML = HelpContent.analysis.title;
    document.getElementById('help-modal-body').innerHTML = HelpContent.analysis.body;
    document.getElementById('help-modal-overlay').style.display = 'flex';
}

function openSectionHelp(category) {
    const data = HelpContent[category] || { title: category + " Help", body: "Documentation available soon." };
    document.getElementById('help-modal-title').innerHTML = data.title;
    document.getElementById('help-modal-body').innerHTML = data.body;
    document.getElementById('help-modal-overlay').style.display = 'flex';
}

function closeHelpModal() {
    document.getElementById('help-modal-overlay').style.display = 'none';
}

// Function to intercept the click on the card and trigger contextual help
window.openAnalysisCardHelp = function(event, type) {
    event.stopPropagation();
    openSectionHelp(type);
};

// Close the modal by clicking outside the white box
document.getElementById('help-modal-overlay').addEventListener('click', function(e) {
    if (e.target === this) {
        closeHelpModal();
    }
});