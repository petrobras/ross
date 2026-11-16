// Digital Twin, steps 1 to 3: equipment, process data, rotor model.
// The server keeps the session; this file keeps only what is on screen.
import { openCustomAlert } from '/components/modals.js';
import { escapeHtml } from '/core/dom.js';
import { onLanguageChanged, t } from '/core/i18n.js';
import { setStatus } from '../components/status.js';
import { apiJson } from '../core/api.js';
import { useModel } from './dt_analyses.js';
import { openPanel } from './panels.js';

const state = {
    platforms: [],
    sessionId: null,
    sessionName: null,
    table: null,
    piTime: null,
};

export function startDigitalTwin() {
    if (!document.getElementById('dt-platform')) return;
    document.getElementById('dt-date').value = new Date().toISOString().slice(0, 10);
    document.getElementById('dt-platform').addEventListener('change', fillEquipment);
    document.getElementById('dt-equipment').addEventListener('change', onEquipmentChange);
    document.getElementById('dt-rotation').addEventListener('change', invalidate);
    document.getElementById('dt-load').addEventListener('click', loadData);
    document.getElementById('dt-create').addEventListener('click', createModel);
    onLanguageChanged(redrawTexts);
    loadEquipment();
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
    select.innerHTML = items.map(item =>
        `<option value="${escapeHtml(item.tag)}"${item.available ? '' : ' disabled'}>` +
        `${escapeHtml(item.tag)}${item.available ? '' : ' ' + escapeHtml(t('dtNeedsStreamlit'))}</option>`
    ).join('');
    const firstUsable = items.find(item => item.available);
    if (firstUsable) select.value = firstUsable.tag;
    onEquipmentChange();
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
    describeEquipment();
    invalidate();
}

// A new equipment, rotation or time makes the loaded data and the model stale.
function invalidate() {
    document.getElementById('dt-create').disabled = true;
    state.table = null;
    document.getElementById('dt-data-card').style.display = 'none';
    clearModel();
}

function clearModel() {
    useModel(null, null);
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
        });
        state.table = body.table;
        state.piTime = body.session.pi_time;
        renderTable();
        document.getElementById('dt-create').disabled = false;
        setStatus('dt-status', t('dtLoaded'), 'ok');
        openPanel('dt', 'model');
    } catch (error) {
        setStatus('dt-status', '');
        openCustomAlert(error.message);
    } finally {
        describeEquipment();
    }
}

async function createModel() {
    if (!state.table) {
        openCustomAlert(t('dtNeedData'));
        return;
    }
    const button = document.getElementById('dt-create');
    button.disabled = true;
    clearModel();
    setStatus('dt-status', t('dtCreating'), 'busy');
    try {
        const body = await apiJson('/api/dt/model', { session: state.sessionId });
        showModel(body);
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
    if (!state.platforms.length) return;
    const equipment = document.getElementById('dt-equipment').value;
    fillPlatforms();
    const platform = state.platforms.find(item => item.name === document.getElementById('dt-platform').value);
    const select = document.getElementById('dt-equipment');
    select.innerHTML = (platform ? platform.equipment : []).map(item =>
        `<option value="${escapeHtml(item.tag)}"${item.available ? '' : ' disabled'}>` +
        `${escapeHtml(item.tag)}${item.available ? '' : ' ' + escapeHtml(t('dtNeedsStreamlit'))}</option>`
    ).join('');
    select.value = equipment;
    describeEquipment();
    renderTable();
}
