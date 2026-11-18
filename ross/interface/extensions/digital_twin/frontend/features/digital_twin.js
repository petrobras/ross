// Digital Twin, steps 1 to 3: equipment, process data, rotor model.
// The server keeps the session; this file keeps only what is on screen.
//
// For now the equipment are the rotors of the user's Rotor Hub: step 1 lists
// them, step 2 (process data) is hidden and step 3 builds the chosen rotor.
// The pre-established models of the Digital Twin app, with their PI data,
// are kept behind USE_PRE_ESTABLISHED; they answer only on the Petrobras
// network, and outside it the screen says so and its controls stay disabled.
import { openCustomAlert } from '/components/modals.js';
import { projectForServer } from '/core/api.js';
import { escapeHtml } from '/core/dom.js';
import { onLanguageChanged, t } from '/core/i18n.js';
import { state as interfaceState } from '/core/state.js';
import { openRotorHub } from '/features/hub.js';
import { onScreenShown } from '/features/screens.js';
import { setStatus } from '../components/status.js';
import { apiJson } from '../core/api.js';
import { useModel } from './dt_analyses.js';
import { useMeasurement } from './measurement.js';
import { openPanel } from './panels.js';

const state = {
    platforms: [],
    sessionId: null,
    sessionName: null,
    table: null,
    piTime: null,
    settings: [],
    hasModel: false,
    build: 0,
};

const USE_PRE_ESTABLISHED = false;

const CONTROLS = ['dt-platform', 'dt-equipment', 'dt-rotation', 'dt-source', 'dt-load'];

export function startDigitalTwin() {
    if (!document.getElementById('dt-platform')) return;
    document.getElementById('dt-date').value = new Date().toISOString().slice(0, 10);
    document.getElementById('dt-platform').addEventListener('change', fillEquipment);
    document.getElementById('dt-equipment').addEventListener('change', onEquipmentChange);
    document.getElementById('dt-rotation').addEventListener('change', invalidate);
    document.getElementById('dt-source').addEventListener('change', invalidate);
    document.getElementById('dt-load').addEventListener('click', loadData);
    document.getElementById('dt-create').addEventListener('click', createModel);
    document.getElementById('dt-to-hub').addEventListener('click', openInHub);
    onLanguageChanged(redrawTexts);
    if (USE_PRE_ESTABLISHED) {
        checkNetwork();
    } else {
        startUserRotors();
    }
}

function startUserRotors() {
    ['dt-platform-group', 'dt-tab-process', 'dt-settings', 'dt-to-hub'].forEach(id => {
        document.getElementById(id).style.display = 'none';
    });
    const label = document.getElementById('dt-equipment-label');
    label.dataset.i18n = 'dtUserRotor';
    label.textContent = t('dtUserRotor');
    const hint = document.querySelector('[data-panel="model"] [data-i18n="dtModelHint"]');
    if (hint) {
        hint.dataset.i18n = 'dtUserModelHint';
        hint.textContent = t('dtUserModelHint');
    }
    onScreenShown('screen-dt', fillUserRotors);
}

// The rotors are read each time the screen opens: they may have been created,
// renamed or edited in the Hub since. A rotor open in the Hub is chosen first.
// The model is built again each time, so step 4 always has the nodes of the
// rotor as it is now; the planes already chosen are kept while their nodes
// exist.
function fillUserRotors() {
    const select = document.getElementById('dt-equipment');
    const chosen = select.value;
    const rotors = interfaceState.rotorLibrary;
    select.innerHTML = rotors.map((rotor, index) =>
        `<option value="${index}"${rotor.isMultiRotor ? ' disabled' : ''}>${escapeHtml(userRotorName(rotor, index))}</option>`
    ).join('');
    const active = interfaceState.activeRotorIndex;
    if (active >= 0 && rotors[active] && !rotors[active].isMultiRotor) {
        select.value = String(active);
    } else if (chosen !== '' && rotors[Number(chosen)] && !rotors[Number(chosen)].isMultiRotor) {
        select.value = chosen;
    } else {
        const first = rotors.findIndex(rotor => !rotor.isMultiRotor);
        if (first >= 0) select.value = String(first);
    }
    if (select.value !== chosen) clearModel();
    describeUserRotor();
    if (currentUserRotor()) createUserModel(false);
}

function userRotorName(rotor, index) {
    return rotor.name || `Rotor ${index + 1}`;
}

function currentUserRotor() {
    const value = document.getElementById('dt-equipment').value;
    if (value === '') return null;
    const rotor = interfaceState.rotorLibrary[Number(value)];
    return rotor && !rotor.isMultiRotor ? rotor : null;
}

function describeUserRotor() {
    const rotor = currentUserRotor();
    const note = document.getElementById('dt-equipment-note');
    note.textContent = rotor ? `${t('dtUserRotorHint')} ${t('dtRotationHint')}` : t('dtNoUserRotors');
    note.style.color = rotor ? '' : 'var(--accent-danger)';
    document.getElementById('dt-create').disabled = !rotor;
}

function onUserRotorChange() {
    clearModel();
    describeUserRotor();
    if (currentUserRotor()) createUserModel(false);
}

// `openBalancing`: the "Create model" button also moves on to step 4; a build
// started by choosing the rotor leaves the user on the step they are in.
async function createUserModel(openBalancing = true) {
    const rotor = currentUserRotor();
    if (!rotor) {
        openCustomAlert(t('dtNoUserRotors'));
        return;
    }
    const build = ++state.build;
    const button = document.getElementById('dt-create');
    button.disabled = true;
    clearModel();
    setStatus('dt-status', t('dtCreating'), 'busy');
    try {
        const session = await ensureSession();
        const index = Number(document.getElementById('dt-equipment').value);
        const body = await apiJson('/api/dt/user-model', {
            session,
            project: projectForServer(rotor),
            name: userRotorName(rotor, index),
            rotation: document.getElementById('dt-rotation').value,
        });
        if (build !== state.build) return;
        showModel(body);
        setStatus('dt-status', `${t('dtCreated')} ${body.session.model}.toml`, 'ok');
        if (openBalancing) openPanel('dt', 'balancing');
    } catch (error) {
        if (build !== state.build) return;
        setStatus('dt-status', '');
        openCustomAlert(error.message);
    } finally {
        if (build === state.build) describeUserRotor();
    }
}


async function checkNetwork() {
    let body;
    try {
        body = await apiJson('/api/dt/network');
    } catch (error) {
        body = { available: false, message: error.message };
    }
    const note = document.getElementById('dt-network-note');
    note.textContent = body.message || '';
    note.style.display = body.available ? 'none' : '';
    CONTROLS.forEach(id => { document.getElementById(id).disabled = !body.available; });
    if (body.available) loadEquipment();
}

async function loadEquipment() {
    try {
        const body = await apiJson('/api/dt/equipment');
        state.platforms = body.platforms;
    } catch (error) {
        openCustomAlert(error.message);
        return;
    }
    fillPlatforms();
    const select = document.getElementById('dt-platform');
    const firstUsable = state.platforms.find(platform => platform.equipment.some(item => item.available));
    if (firstUsable) select.value = firstUsable.name;
    fillEquipment();
}

function fillPlatforms() {
    const select = document.getElementById('dt-platform');
    const chosen = select.value;
    select.innerHTML = state.platforms.map(platform => {
        const usable = platform.equipment.some(item => item.available);
        return `<option value="${escapeHtml(platform.name)}">${escapeHtml(platform.name)}` +
            `${usable ? '' : ' ' + escapeHtml(t('dtUnavailable'))}</option>`;
    }).join('');
    if (chosen) select.value = chosen;
}

function fillEquipment() {
    const platform = state.platforms.find(item => item.name === document.getElementById('dt-platform').value);
    const select = document.getElementById('dt-equipment');
    const items = platform ? platform.equipment : [];
    select.innerHTML = equipmentOptions(items);
    const firstUsable = items.find(item => item.available);
    if (firstUsable) select.value = firstUsable.tag;
    onEquipmentChange();
}

function equipmentOptions(items) {
    return items.map(item =>
        `<option value="${escapeHtml(item.tag)}"${item.available ? '' : ' disabled'}>` +
        `${escapeHtml(item.tag)}${item.available ? '' : ' ' + escapeHtml(t('dtUnavailable'))}</option>`
    ).join('');
}

function currentEquipment() {
    const tag = document.getElementById('dt-equipment').value;
    return state.platforms.flatMap(platform => platform.equipment).find(entry => entry.tag === tag);
}

function describeEquipment() {
    const item = currentEquipment();
    const note = document.getElementById('dt-equipment-note');
    const load = document.getElementById('dt-load');
    if (!item || !item.available) {
        note.textContent = item ? item.reason : t('dtNoEquipment');
        note.style.color = 'var(--accent-danger)';
        load.disabled = true;
    } else {
        note.textContent = `${t('dtModel')}: ${item.model}. ${t('dtRotationHint')}`;
        note.style.color = '';
        load.disabled = false;
    }
}

function onEquipmentChange() {
    if (!USE_PRE_ESTABLISHED) {
        onUserRotorChange();
        return;
    }
    describeEquipment();
    invalidate();
    loadSettings();
}

// Step 3's additional settings: the quantities the model declares
// (RADIAL_BEARING_CLEARANCE, OIL_SUPPLY_TEMPERATURE, ...), with its defaults.
async function loadSettings() {
    const item = currentEquipment();
    state.settings = [];
    renderSettings();
    if (!item || !item.available) return;
    try {
        const body = await apiJson('/api/dt/model-settings?equipment=' + encodeURIComponent(item.tag));
        if (currentEquipment() !== item) return;
        state.settings = body.settings;
    } catch (error) {
        openCustomAlert(error.message);
    }
    renderSettings();
}

function renderSettings() {
    const box = document.getElementById('dt-settings');
    if (!box) return;
    if (!state.settings.length) {
        box.innerHTML = currentEquipment() ? `<p class="panel-hint">${escapeHtml(t('dtNoSettings'))}</p>` : '';
        return;
    }
    box.innerHTML = `<h4>${escapeHtml(t('dtSettings'))}</h4>` + state.settings.map((item, index) => `
        <div class="input-group">
            <label for="dt-setting-${index}">${escapeHtml(settingLabel(item.name))}</label>
            <div style="display: flex; gap: 6px;">
                <input id="dt-setting-${index}" type="number" step="any" value="${escapeHtml(String(item.value))}">
                <input id="dt-setting-unit-${index}" type="text" value="${escapeHtml(item.unit)}" style="max-width: 90px;">
            </div>
        </div>`).join('');
}

function settingLabel(name) {
    return name.toLowerCase().replace(/_/g, ' ').replace(/^./, letter => letter.toUpperCase());
}

function chosenSettings() {
    const chosen = {};
    state.settings.forEach((item, index) => {
        chosen[item.name] = {
            value: Number(document.getElementById(`dt-setting-${index}`).value),
            unit: document.getElementById(`dt-setting-unit-${index}`).value.trim(),
        };
    });
    return chosen;
}

// A new equipment, rotation or time makes the loaded data and the model stale.
function invalidate() {
    if (!USE_PRE_ESTABLISHED) {
        onUserRotorChange();
        return;
    }
    document.getElementById('dt-create').disabled = true;
    state.table = null;
    document.getElementById('dt-data-card').style.display = 'none';
    clearModel();
}

function clearModel() {
    useModel(null, null);
    useMeasurement(null, null);
    state.hasModel = false;
    document.getElementById('dt-to-hub').disabled = true;
    document.getElementById('dt-rotor-info').style.opacity = 0;
    document.getElementById('dt-plot').innerHTML =
        '<div class="analysis-no-chart" style="height: 100%;"><i class="fas fa-cogs"></i>' +
        `<p data-i18n="dtRotorWillAppear">${escapeHtml(t('dtRotorWillAppear'))}</p></div>`;
    document.getElementById('dt-components-card').style.display = 'none';
    setStatus('dt-status', '');
}

async function ensureSession() {
    if (state.sessionId) return state.sessionId;
    const body = await apiJson('/api/dt/session', {});
    state.sessionId = body.id;
    state.sessionName = body.name;
    document.getElementById('dt-info-session').textContent = body.name;
    return state.sessionId;
}

async function loadData() {
    const date = document.getElementById('dt-date').value;
    let time = document.getElementById('dt-time').value || '00:00:00';
    if (time.length === 5) time += ':00';
    if (!date) {
        openCustomAlert(t('dtPickDate'));
        return;
    }
    const button = document.getElementById('dt-load');
    button.disabled = true;
    invalidate();
    setStatus('dt-status', t('dtLoading'), 'busy');
    try {
        const session = await ensureSession();
        const body = await apiJson('/api/dt/process-data', {
            session,
            equipment: document.getElementById('dt-equipment').value,
            pi_time: date.replace(/-/g, '/') + ' ' + time,
            rotation: document.getElementById('dt-rotation').value,
            source: document.getElementById('dt-source').value,
        });
        state.table = body.table;
        state.piTime = body.session.pi_time;
        renderTable();
        document.getElementById('dt-create').disabled = false;
        setStatus('dt-status', t('dtLoaded'), 'ok');
        showMessages(body.messages);
        openPanel('dt', 'model');
    } catch (error) {
        setStatus('dt-status', '');
        openCustomAlert(error.message);
    } finally {
        describeEquipment();
    }
}

async function createModel() {
    if (!USE_PRE_ESTABLISHED) {
        createUserModel();
        return;
    }
    if (!state.table) {
        openCustomAlert(t('dtNeedData'));
        return;
    }
    const button = document.getElementById('dt-create');
    button.disabled = true;
    clearModel();
    setStatus('dt-status', t('dtCreating'), 'busy');
    try {
        const body = await apiJson('/api/dt/model', { session: state.sessionId, settings: chosenSettings() });
        showModel(body);
        showMessages(body.messages);
        if (body.nan_warning) {
            setStatus('dt-status', '');
            openCustomAlert(t('dtNanWarning'));
        } else {
            setStatus('dt-status', `${t('dtCreated')} ${body.session.model}.toml`, 'ok');
        }
    } catch (error) {
        setStatus('dt-status', '');
        openCustomAlert(error.message);
    } finally {
        button.disabled = false;
    }
}

function showModel(body) {
    const rotor = body.rotor;
    document.getElementById('dt-info-equipment').textContent = body.session.equipment;
    document.getElementById('dt-info-nodes').textContent = rotor.nodes;
    document.getElementById('dt-info-mass').textContent = Number(rotor.mass).toFixed(3);
    document.getElementById('dt-info-disks').textContent = rotor.disks;
    document.getElementById('dt-info-bearings').textContent = rotor.bearings;
    document.getElementById('dt-rotor-info').style.opacity = 1;

    const plot = document.getElementById('dt-plot');
    plot.innerHTML = '';
    if (window.Plotly) {
        window.Plotly.newPlot(plot, body.figure.data, body.figure.layout, { responsive: true, displaylogo: false });
    }
    document.getElementById('dt-bearings').textContent = body.components.bearings || '—';
    document.getElementById('dt-labyrinths').textContent = body.components.labyrinths || '—';
    document.getElementById('dt-components-card').style.display = '';
    useModel(state.sessionId, rotor);
    useMeasurement(state.sessionId, body.measurement);
    state.hasModel = true;
    document.getElementById('dt-to-hub').disabled = false;
    document.getElementById('dt-components-card').style.display = body.components.bearings ? '' : 'none';
}

// What the model reported through st.warning / st.info, which the Streamlit
// app showed on the page.
function showMessages(messages) {
    const warnings = (messages || []).filter(item => item.level === 'warning' || item.level === 'error');
    if (!warnings.length) return;
    openCustomAlert(t('dtModelMessages') + '\n' + warnings.map(item => '• ' + item.text).join('\n'));
}

// The rotor goes to the Rotor Hub as an ordinary project: from there it is
// edited and analysed with the interface's own screens.
async function openInHub() {
    if (!state.hasModel) {
        openCustomAlert(t('dtNeedModel'));
        return;
    }
    const button = document.getElementById('dt-to-hub');
    button.disabled = true;
    try {
        const body = await apiJson('/api/dt/project', { session: state.sessionId });
        interfaceState.rotorLibrary.push({ savedAnalyses: [], ...body.projectData });
        openRotorHub();
        await openCustomAlert([t('dtOpenedInHub'), ...body.warnings].join('\n\n'));
    } catch (error) {
        openCustomAlert(error.message);
    } finally {
        button.disabled = !state.hasModel;
    }
}

function renderTable() {
    const table = state.table;
    if (!table) return;
    const head = table.columns.map(name => `<th>${escapeHtml(name)}</th>`).join('');
    const rows = table.rows.map(row =>
        '<tr>' + row.map(cell => formatCell(cell)).join('') + '</tr>'
    ).join('');
    document.getElementById('dt-data-body').innerHTML =
        `<p class="panel-hint">${escapeHtml(t('dtSelectedTime'))} <b>${escapeHtml(state.piTime)}</b></p>` +
        `<div class="table-scroll"><table class="data-table"><thead><tr>${head}</tr></thead><tbody>${rows}</tbody></table></div>`;
    document.getElementById('dt-data-card').style.display = '';
}

function formatCell(cell) {
    if (cell === null) return `<td class="missing">${escapeHtml(t('dtEmpty'))}</td>`;
    if (typeof cell === 'number') return `<td>${cell.toFixed(4)}</td>`;
    return `<td>${escapeHtml(cell)}</td>`;
}

function redrawTexts() {
    if (!USE_PRE_ESTABLISHED) {
        describeUserRotor();
        return;
    }
    if (!state.platforms.length) return;
    const equipment = document.getElementById('dt-equipment').value;
    fillPlatforms();
    const platform = state.platforms.find(item => item.name === document.getElementById('dt-platform').value);
    const select = document.getElementById('dt-equipment');
    select.innerHTML = equipmentOptions(platform ? platform.equipment : []);
    select.value = equipment;
    describeEquipment();
    renderSettings();
    renderTable();
}
