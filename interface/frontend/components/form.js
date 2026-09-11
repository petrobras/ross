// An element form, built from the schema. Until Phase 1 this was ~450 lines of
// literal HTML in FormTemplates, with copies of UNITS_MAPPING alongside it.
import { openCustomAlert, openCustomPrompt } from './modals.js';
import { escapeHtml } from '../core/dom.js';
import { state } from '../core/state.js';
import { t } from '../core/i18n.js';
import { schemaFor } from '../core/schema.js';
function sectionHeaderHTML(title) {
    return '<div style="grid-column: 1 / -1; margin-top: 10px; border-bottom: 1px solid var(--border-color);">' +
        '<b style="font-size:12px; color:var(--text-muted);">' + escapeHtml(title).toUpperCase() + '</b></div>';
}

function unitSelectHTML(field) {
    const optionsHtml = field.unit_options.map(unit =>
        '<option value="' + escapeHtml(unit) + '"' +
        (unit === field.unit ? ' selected' : '') + '>' + escapeHtml(unit) + '</option>').join('');
    return '<select id="inp-' + field.name + '_unit" class="unit-select" data-prev="' +
        escapeHtml(field.unit) + '" onchange="handleUnitChange(this)">' +
        optionsHtml + '<option value="Others">' + escapeHtml(t('others')) + '</option></select>';
}

function buildFieldHTML(field) {
    let controlHtml;
    if (field.control === 'material_ref') {
        // filled in by selectSubType with the materials of the project
        controlHtml = '<select id="inp-' + field.name + '"></select>';
    } else if (field.options && field.options.length) {
        const optionsHtml = field.options.map(option =>
            '<option value="' + escapeHtml(option) + '">' +
            escapeHtml(String(option).toUpperCase()) + '</option>').join('');
        controlHtml = '<select id="inp-' + field.name + '">' + optionsHtml + '</select>';
    } else {
        // is_dict replaces the "{dict}" that used to sit glued to the old label
        const placeholderText = field.placeholder || (field.is_dict ? '{"key": value}' : null);
        const placeholder = placeholderText ? ' placeholder="' + escapeHtml(placeholderText) + '"' : '';
        controlHtml = '<input type="text" id="inp-' + field.name + '"' + placeholder + '>';
    }

    if (field.unit && field.unit_options && field.unit_options.length) {
        controlHtml = '<div class="input-unit-wrapper">' + controlHtml + unitSelectHTML(field) + '</div>';
    }

    // the help text comes from the docstring of the ROSS class itself
    const tooltip = field.help ? ' title="' + escapeHtml(field.help) + '"' : '';
    const label = escapeHtml(field.label) + (field.optional ? ' (' + t('optional') + ')' : '');
    return '<div class="input-group"><label' + tooltip + '>' + label + '</label>' + controlHtml + '</div>';
}

function batchBannerHTML() {
    return '<div style="background:#e8f4fd; border-left:4px solid #2980b9; padding:10px; ' +
        'margin-bottom:15px; font-size:12px; color:#2c3e50;">' +
        '<i class="fas fa-layer-group"></i> <b>' + t('batchTitle') + '</b> ' + t('batchBody') + '</div>';
}

export function buildFormHTML(category, subtype) {
    if (subtype === 'LIST') {
        return batchBannerHTML() +
            buildFormHTML(category, 'BASIC').replace(
                /id="inp-n"/g, 'id="inp-n" placeholder="e.g. 0, 1, 2"');
    }

    const definition = schemaFor(category, subtype);
    if (!definition) return '<p class="empty-message">' + t('unavailable') + '</p>';

    const mainOnes = [];
    const advancedOnes = [];
    let currentSection = null;

    definition.fields.forEach(field => {
        const destination = field.group === 'advanced' ? advancedOnes : mainOnes;
        if (field.group !== 'advanced' && field.section && field.section !== currentSection) {
            currentSection = field.section;
            destination.push(sectionHeaderHTML(field.section));
        }
        destination.push(buildFieldHTML(field));
    });

    let html = mainOnes.join('');
    if (advancedOnes.length) {
        html += '<button type="button" class="btn-advanced" onclick="toggleAdvanced(this)">' +
            t('advanced') + ' <i class="fas fa-chevron-down"></i></button>' +
            '<div class="advanced-fields" style="display: none; margin-top: 10px; ' +
            'border-top: 1px dashed #ccc; padding-top: 10px;">' + advancedOnes.join('') + '</div>';
    }
    return html;
}

// Typed values survive a language change: the form is redrawn with the new
// labels, not reopened from scratch.
export function capturedFormValues() {
    const values = {};
    document.querySelectorAll('#form-fields input, #form-fields select')
        .forEach(field => { values[field.id] = field.value; });
    const advanced = document.querySelector('#form-fields .advanced-fields');
    return { values, advancedOpen: !!advanced && advanced.style.display === 'block' };
}

export function restoreFormValues(stored) {
    if (!stored) return;
    Object.keys(stored.values).forEach(id => {
        const field = document.getElementById(id);
        if (!field) return;
        field.value = stored.values[id];
        // the unit selector keeps the previous choice for handleUnitChange
        if (id.endsWith('_unit')) field.dataset.prev = stored.values[id];
    });
    if (stored.advancedOpen) {
        const button = document.querySelector('#form-fields .btn-advanced');
        if (button) toggleAdvanced(button);
    }
}

// Engineering defaults library

const DefaultExamples = {
    materials_BASIC: { name: "Steel", rho: "7800", E: "211e9", G_s: "81.2e9" },
    shafts_BASIC: { L: "500", odl: "100", idl: "0", material: "Default (Steel)" },
    disks_BASIC: { m: "32", Id: "0.2", Ip: "0.3" },
    gears_BASIC: { m: "4.67", Id: "0.015", Ip: "0.030", n_teeth: "26", pitch_diameter: "187", pr_angle: "22.5", helix_angle: "0" },
    gears_TVMS: { material: "Default (Steel)", width: "20", bore_diameter: "70", module: "2", n_teeth: "62", pr_angle: "20" },
    couplings_BASIC: { m_l: "37.8875", m_r: "37.8875", Ip_l: "1.0985", Ip_r: "1.0985", kr_z: "3.04256e6" },
    pointmasses_BASIC: { m: "2" },
    bearings_BASIC: { kxx: "1e6", kyy: "0.8e6", cxx: "2e2", cyy: "1.5e2" },
    bearings_BallBearing: { n_balls: "8", d_balls: "0.03", fs: "500", alpha: "0.523598" },
    bearings_RollerBearing: { n_rollers: "8", l_rollers: "0.03", fs: "500", alpha: "0.523598" },
    bearings_MagneticBearing: { g0: "1e-3", i0: "1", ag: "1e-4", nw: "200", kp_pid: "1", ki_pid: "0", kd_pid: "1", alpha: "0.392699", k_amp: "1", k_sense: "1" },
    bearings_Cylindrical: { speed: "[1500]", weight: "525", bearing_length: "30", journal_diameter: "10", radial_clearance: "0.1", oil_viscosity: "0.1" },
    bearings_PlainJournal: { pad_axial_length: "30", journal_diameter: "100", radial_clearance: "0.1", n_pads: "1", pad_arc: "360", preload: "0.0", oil_supply_temperature: "40", frequency: "[90]", fxs_load: "0", fys_load: "1000", lubricant: "ISOVG32", initial_position: "(0.1, -0.1)", oil_supply_pressure: "0", oil_flow_v: "", pad_thickness: "20" },
    bearings_SqueezeFilm: { frequency: "[18600]", axial_length: "22.86", journal_diameter: "129.54", radial_clearance: "7.62e-2", eccentricity_ratio: "0.5", lubricant: "ISOVG32", geometry: "groove", cavitation: "True" },
    bearings_ThrustPad: { pad_inner_radius: "1150", pad_outer_radius: "1725", pad_pivot_radius: "1442.5", pad_arc: "26", pivot_angle: "15", oil_supply_temperature: "40", lubricant: "ISOVG68", n_pads: "12", n_theta: "10", n_radial: "10", frequency: "[90]", equilibrium_position_mode: "calculate", radial_inclination_angle: "-2.75e-04", circumferential_inclination_angle: "-1.70e-05", initial_film_thickness: "0.2", axial_load: "13.32e6" },
    bearings_TiltingPad: { journal_diameter: "100", preload: "0.5", pad_thickness: "20", pad_arc: "60", offset: "0.5", pad_axial_length: "30", lubricant: "ISOVG32", oil_supply_temperature: "40", radial_clearance: "0.1", pivot_angle: "0", frequency: "[90]", total_ex_film: "30", total_ez_film: "30", total_ey_pad: "16", xj: "", yj: "", equilibrium_type: "match_eccentricity", eccentricity: "0.3", attitude_angle: "4.71238", fxs_load: "0", fys_load: "1000", thermal_type: "full", pad_conductivity: "116.0", edges_convection: "1500.0", relax_temperature: "0.5", journal_temperature: "", oil_flow_v: "" },
    seals_BASIC: { kxx: "1e6", cxx: "2e2", kyy: "0.8e6", cyy: "1.5e2" },
    seals_HolePattern: { shaft_diameter: "145", radial_clearance: "0.3", axial_length: "46.99", relative_roughness: "0.0001", cell_length: "3.175", cell_width: "3.175", cell_depth: "2.5", inlet_pressure: "689000", outlet_pressure: "94300", inlet_temperature: "48.85", frequency: "[8000]", gas_composition: '{"Nitrogen": 0.79, "Oxygen": 0.21}', preswirl: "0.8", entrance_loss_coefficient: "0.5", exit_loss_coefficient: "1.0", nz: "18" },
    seals_Labyrinth: { shaft_diameter: "145", radial_clearance: "0.3", n_teeth: "16", pitch: "3.175", tooth_height: "3.175", tooth_width: "0.1524", seal_type: "inter", inlet_pressure: "308000", outlet_pressure: "94300", inlet_temperature: "10", frequency: "[8000]", preswirl: "0.98", gas_composition: '{"Nitrogen": 0.79, "Oxygen": 0.21}' },
    seals_Hybrid: { shaft_diameter: "50", inlet_pressure: "500000", outlet_pressure: "100000", inlet_temperature: "26.85", frequency: "[2000]", gas_composition: '{"Nitrogen": 0.7812, "Oxygen": 0.2096, "Argon": 0.0092}', hole_pattern_parameters: '{"radial_clearance": 0.0003, "axial_length": 0.04, "relative_roughness": 0.0001, "cell_length": 0.003, "cell_width": 0.003, "cell_depth": 0.002, "preswirl": 0.8, "entrance_loss_coefficient": 0.5, "exit_loss_coefficient": 1.0}', labyrinth_parameters: '{"radial_clearance": 0.00025, "n_teeth": 10, "pitch": 0.003, "tooth_height": 0.003, "tooth_width": 0.00015, "seal_type": "inter", "preswirl": 0.9, "reference_temperatures": [300.0, 299.5], "reference_viscosities": [1.85e-05, 1.84e-05]}' }
};

// Fill in the values ​​with the default

export async function fillDefault() {    
    let searchType = state.currentSubType === 'LIST' ? 'BASIC' : state.currentSubType;
    const defaultData = DefaultExamples[`${state.currentTab}_${searchType}`];    
    if (!defaultData) return await openCustomAlert(t('noDefaults'));
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

// Function for the 'Advanced' button

export function toggleAdvanced(btn) {
    const div = btn.nextElementSibling;
    if(div.style.display === 'none' || div.style.display === '') {
        div.style.display = 'block';
        btn.innerHTML = t('hideAdvanced') + ' <i class="fas fa-chevron-up"></i>';
    } else {
        div.style.display = 'none';
        btn.innerHTML = t('advanced') + ' <i class="fas fa-chevron-down"></i>';
    }
}

// --- Custom Units Manager ---

export const handleUnitChange = async function(sel) {
    if (sel.value === 'Others') {
        let custom = await openCustomPrompt("Enter custom unit string (e.g., 'lbf/in', 'Hz', 'lb*ft**2'):", "");
        if (custom && custom.trim() !== '') {
            custom = custom.trim();
            let exists = Array.from(sel.options).find(o => o.value === custom);
            if (!exists) {
                let newOpt = document.createElement('option');
                newOpt.value = custom;
                newOpt.innerText = custom;
                sel.insertBefore(newOpt, sel.querySelector('option[value="Others"]'));
            }
            sel.value = custom;
            sel.dataset.prev = custom;
        } else {
            sel.value = sel.dataset.prev || sel.options[0].value;
        }
    } else {
        sel.dataset.prev = sel.value;
    }
};
