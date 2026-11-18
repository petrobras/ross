// Digital Twin, step 4: measuring planes, balancing planes and sensors, as in
// section 4 of the Streamlit app. The server keeps the planes and writes each
// sensor's waveform to the session (`services/measurement.py`); this file
// draws what the server answers.
import { openCustomAlert } from '/components/modals.js';
import { escapeHtml } from '/core/dom.js';
import { onLanguageChanged, t } from '/core/i18n.js';
import { apiJson } from '../core/api.js';

const PLANES = ['X', 'Y'];
const Y_OFFSET = 90;
const MIN_MEASURING = 2;

const state = {
    sessionId: null,
    step: null,
};

export const MEASUREMENT_ACTIONS = {
    'dt-plane-add': element => changePlanes(element.dataset.kind, true),
    'dt-plane-remove': element => changePlanes(element.dataset.kind, false),
    'dt-sensor-apply': element => applySensor(Number(element.dataset.node), element.dataset.plane),
};

export function startMeasurement() {
    if (!document.getElementById('dt-planes-form')) return;
    onLanguageChanged(render);
    useMeasurement(null, null);
}

// Called when a model is created (with the step's state the server answered)
// and when the model is cleared (with nulls).
export function useMeasurement(sessionId, step) {
    state.sessionId = sessionId;
    state.step = step;
    document.getElementById('dt-planes-need-model').style.display = step ? 'none' : '';
    document.getElementById('dt-planes-form').style.display = step ? '' : 'none';
    if (!step) {
        document.getElementById('dt-sensors-card').style.display = 'none';
        document.getElementById('dt-sensors').innerHTML = '';
        document.getElementById('dt-sensors-status').textContent = '';
    }
    fillNodes();
    render();
}

function fillNodes() {
    const count = state.step ? state.step.nodes : 0;
    const options = Array.from({ length: count }, (_, node) => `<option value="${node}">${node}</option>`).join('');
    ['dt-measuring-node', 'dt-balancing-node'].forEach(id => {
        document.getElementById(id).innerHTML = options;
    });
}

async function changePlanes(kind, add) {
    if (!state.step) return;
    const node = Number(document.getElementById(`dt-${kind}-node`).value);
    const lists = { measuring: [...state.step.measuring], balancing: [...state.step.balancing] };
    const list = lists[kind];
    if (add) {
        if (list.includes(node)) {
            openCustomAlert(t('dtAlreadyPlane').replace('{node}', node));
            return;
        }
        list.push(node);
    } else {
        const index = list.indexOf(node);
        if (index === -1) return;
        list.splice(index, 1);
    }
    try {
        state.step = await apiJson('/api/dt/planes', { session: state.sessionId, ...lists });
    } catch (error) {
        openCustomAlert(error.message);
    }
    render();
}

async function applySensor(node, plane) {
    const input = document.getElementById(sensorId(node, plane, 'file'));
    const file = input.files[0];
    const body = {
        session: state.sessionId,
        node,
        plane,
        angle: Number(document.getElementById(sensorId(node, 'X', 'angle')).value),
    };
    if (file) {
        body.filename = file.name;
        body.content = await file.text();
    }
    const button = document.getElementById(sensorId(node, plane, 'apply'));
    button.disabled = true;
    try {
        state.step = await apiJson('/api/dt/sensor', body);
    } catch (error) {
        openCustomAlert(error.message);
    } finally {
        button.disabled = false;
    }
    render();
}

function render() {
    if (!state.step) return;
    const step = state.step;
    document.getElementById('dt-measuring-list').textContent = listText(step.measuring);
    document.getElementById('dt-balancing-list').textContent = listText(step.balancing);

    const box = document.getElementById('dt-sensors');
    const status = document.getElementById('dt-sensors-status');
    const card = document.getElementById('dt-sensors-card');
    if (!step.measuring.length) {
        box.innerHTML = '';
        card.style.display = 'none';
        status.textContent = t('dtNeedTwoPlanes');
        return;
    }
    // A sensor is offered for every measuring plane there is, so the files can
    // be sent as the planes are chosen. The analysis needs two of them, which
    // the status line asks for until the second one is there.
    card.style.display = '';
    box.innerHTML = step.measuring.map(node => PLANES.map(plane => sensorCard(node, plane)).join('')).join('');
    if (step.measuring.length < MIN_MEASURING) {
        status.textContent = t('dtNeedTwoPlanes');
    } else {
        status.textContent = step.complete
            ? t('dtSensorsComplete')
            : `${t('dtSensorsPending')} ${step.pending.join(', ')}`;
    }
    step.measuring.forEach(node => {
        const xAngle = document.getElementById(sensorId(node, 'X', 'angle'));
        xAngle.addEventListener('input', () => {
            const value = Number(xAngle.value);
            document.getElementById(sensorId(node, 'Y', 'angle')).value = Number.isFinite(value) ? value + Y_OFFSET : '';
        });
    });
}

function sensorCard(node, plane) {
    const applied = (state.step.sensors[String(node)] || {})[plane];
    const xApplied = (state.step.sensors[String(node)] || {}).X;
    const xAngle = xApplied ? xApplied.probes_angle : 0;
    const angle = plane === 'X' ? xAngle : xAngle + Y_OFFSET;
    const title = t('dtNodePlane').replace('{node}', node).replace('{plane}', plane);
    const file = applied
        ? `<p class="panel-hint"><b>${escapeHtml(t('dtFileApplied'))}</b> ${escapeHtml(applied.filename)} (${applied.rows} ${escapeHtml(t('dtRows'))})</p>`
        : '';
    return `
        <div class="form-box dt-sensor${applied ? ' applied' : ''}">
            <h4><i class="fas ${applied ? 'fa-check-circle' : 'fa-wave-square'}"></i> ${escapeHtml(title)}</h4>
            ${file}
            <div class="input-group">
                <label for="${sensorId(node, plane, 'file')}">${escapeHtml(t('dtSignalFile'))}</label>
                <input id="${sensorId(node, plane, 'file')}" type="file" accept=".csv,.CSV">
            </div>
            <div class="input-group">
                <label for="${sensorId(node, plane, 'angle')}">${escapeHtml(t('dtSensorAngle'))}</label>
                <input id="${sensorId(node, plane, 'angle')}" type="number" step="any" value="${angle}"${plane === 'Y' ? ' disabled' : ''}>
            </div>
            <button type="button" class="btn-primary" id="${sensorId(node, plane, 'apply')}" data-action="dt-sensor-apply" data-node="${node}" data-plane="${plane}">${escapeHtml(t('dtApply'))}</button>
        </div>`;
}

function sensorId(node, plane, part) {
    return `dt-sensor-${node}-${plane}-${part}`;
}

function listText(nodes) {
    return nodes.length ? `${t('dtNode')}: ${nodes.join(', ')}` : t('dtNoPlanes');
}
