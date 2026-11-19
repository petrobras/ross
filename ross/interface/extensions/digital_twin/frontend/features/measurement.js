// Digital Twin, step 4: measuring planes, balancing planes and sensors, as in
// section 4 of the Streamlit app. The server keeps the planes and writes each
// sensor's waveform to the session (`services/measurement.py`); this file
// draws what the server answers.
import { openCustomAlert } from '/components/modals.js';
import { escapeHtml } from '/core/dom.js';
import { onLanguageChanged, t } from '/core/i18n.js';
import { apiJson } from '../core/api.js';
import { showLoader } from '../components/loader.js';

const PLANES = ['X', 'Y'];
const Y_OFFSET = 90;
const MIN_MEASURING = 2;

const state = {
    sessionId: null,
    step: null,
    result: null,
};

export const MEASUREMENT_ACTIONS = {
    'dt-plane-add': element => changePlanes(element.dataset.kind, true),
    'dt-plane-remove': element => changePlanes(element.dataset.kind, false),
    'dt-sensor-apply': element => applySensor(Number(element.dataset.node), element.dataset.plane),
    'dt-run-balancing': () => runBalancing(),
    'dt-parameters-apply': () => applyParameters(),
    // The interface dispatches a select's `change` through the same actions
    // (core/actions.js), so the two pickers of the polar chart come here.
    'dt-polar-change': () => renderBalancing(state.result),
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
        renderBalancing(null);
        document.getElementById('dt-run').disabled = true;
    }
    fillNodes();
    render();
    refreshAnalysis();
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
    const lists = {
        measuring: [...state.step.measuring],
        balancing: state.step.balancing.map(plane => ({ ...plane })),
    };
    // A measuring plane is a node; a balancing plane is a node and the radius
    // where its correction mass goes, which is what turns the estimated
    // unbalance (in g.mm) into a mass.
    const list = lists[kind];
    const nodeOf = item => (kind === 'balancing' ? item.node : item);
    if (add) {
        if (list.some(item => nodeOf(item) === node)) {
            openCustomAlert(t('dtAlreadyPlane').replace('{node}', node));
            return;
        }
        if (kind === 'balancing') {
            const radius = Number(document.getElementById('dt-balancing-radius').value);
            if (!(radius > 0)) {
                openCustomAlert(t('dtNeedRadius'));
                return;
            }
            list.push({ node, radius });
        } else {
            list.push(node);
        }
    } else {
        const index = list.findIndex(item => nodeOf(item) === node);
        if (index === -1) return;
        list.splice(index, 1);
    }
    try {
        state.step = await apiJson('/api/dt/planes', { session: state.sessionId, ...lists });
    } catch (error) {
        openCustomAlert(error.message);
    }
    render();
    refreshAnalysis();
}

async function runBalancing() {
    const button = document.getElementById('dt-run');
    const status = document.getElementById('dt-run-refusal');
    button.disabled = true;
    status.textContent = t('dtRunning');
    // The card is shown while it waits, so the spinner and the seconds have
    // somewhere to sit: a run takes from a few seconds to minutes.
    const card = document.getElementById('dt-balancing-card');
    const body = document.getElementById('dt-balancing-body');
    card.style.display = '';
    const stop = showLoader('dt-balancing', body, { label: t('dtRunning'), top: '50%' });
    try {
        const answer = await apiJson('/api/dt/balancing', {
            session: state.sessionId,
            iterations: Number(document.getElementById('dt-iterations').value),
            population: Number(document.getElementById('dt-population').value),
            optimize: document.getElementById('dt-optimize').value === 'yes',
        });
        stop();
        showAnalysis(answer);
    } catch (error) {
        stop();
        if (!state.result) card.style.display = 'none';
        status.textContent = error.message;
        openCustomAlert(error.message);
        button.disabled = false;
    }
}

// The hyperparameters of the machine, the file the Streamlit app passes to
// rossml for every run. Without it the estimate uses the library defaults and
// does not reproduce that app, which is what `dt-parameters-status` says.
async function applyParameters() {
    const input = document.getElementById('dt-parameters-file');
    const file = input.files && input.files[0];
    const body = { session: state.sessionId, filename: null, content: null };
    if (file) {
        body.filename = file.name;
        body.content = await file.text();
    }
    const button = document.getElementById('dt-parameters-apply');
    button.disabled = true;
    try {
        showAnalysis(await apiJson('/api/dt/balancing-parameters', body));
    } catch (error) {
        openCustomAlert(error.message);
    } finally {
        button.disabled = false;
    }
}

// The analysis needs rossml, every sensor applied and a balancing plane. The
// server decides, and says why not, so the button and the reason stay in step
// with the session instead of with what this file last saw.
async function refreshAnalysis() {
    if (!state.sessionId) return;
    try {
        showAnalysis(await apiJson('/api/dt/balancing-settings?session=' + state.sessionId));
    } catch (error) {
        document.getElementById('dt-run-refusal').textContent = error.message;
    }
}

function showAnalysis(answer) {
    document.getElementById('dt-run').disabled = !answer.ready;
    document.getElementById('dt-run-refusal').textContent = answer.refusal;
    const status = document.getElementById('dt-parameters-status');
    status.textContent = answer.parameters
        ? `${t('dtParametersFrom' + (answer.parameters.source === 'model' ? 'Model' : 'File'))} ${answer.parameters.name}`
        : (answer.parameters_warning || '');
    status.classList.toggle('dt-warning', !answer.parameters);
    renderBalancing(answer.balancing);
}

function renderBalancing(result) {
    const card = document.getElementById('dt-balancing-card');
    const box = document.getElementById('dt-balancing-result');
    const summary = document.getElementById('dt-balancing-summary');
    state.result = result;
    if (!result) {
        card.style.display = 'none';
        box.innerHTML = '';
        summary.textContent = '';
        document.getElementById('dt-polar').innerHTML = '';
        return;
    }
    card.style.display = '';
    fillPolarChoices(result);
    summary.textContent = t('dtBalancingSummary')
        .replace('{speed}', number(result.speed[0], 0) + ' ' + result.speed[1])
        .replace('{nodes}', result.measuring.join(', '));
    box.innerHTML = `
        <table class="dt-result-table">
            <thead><tr>
                <th>${escapeHtml(t('dtNode'))}</th>
                <th>${escapeHtml(t('dtRadius'))}</th>
                <th>${escapeHtml(t('dtUnbalance'))}</th>
                <th>${escapeHtml(t('dtMass'))}</th>
                <th>${escapeHtml(t('dtPhase'))}</th>
            </tr></thead>
            <tbody>${result.planes.map((plane, index) => `
                <tr>
                    <td>${plane.node}</td>
                    <td>${number(plane.radius, 1)}</td>
                    <td>${number(plane.unbalance, 2)}</td>
                    <td>${number(plane.mass, 3)}</td>
                    <td>${number(correctionPhase(result, plane, index), 1)}</td>
                </tr>`).join('')}</tbody>
        </table>`;
    loadPolar();
}

// The unbalance angle is measured against one sensor, so the correction angle
// follows the sensor chosen above. Without a choice it is the one the server
// read against the first sensor.
function correctionPhase(result, plane, index) {
    const key = document.getElementById('dt-polar-sensor').value;
    const angles = (result.phases || {})[key];
    if (!angles || angles[index] === null || angles[index] === undefined) return plane.phase;
    return (angles[index] + 180) % 360;
}

function fillPolarChoices(result) {
    const sensors = document.getElementById('dt-polar-sensor');
    const views = document.getElementById('dt-polar-view');
    const chosen = { sensor: sensors.value, view: views.value };
    sensors.innerHTML = (result.sensors || []).map(sensor =>
        `<option value="${escapeHtml(sensor.key)}">${escapeHtml(t('dtNode'))} ${sensor.node} - ${escapeHtml(sensor.plane)}</option>`).join('');
    views.innerHTML = (result.views || []).map(view =>
        `<option value="${escapeHtml(view)}">${escapeHtml(t('dtView' + view))}</option>`).join('');
    // The first option is chosen here rather than left to the browser, which
    // selects it on its own: the choice has to be readable right away, because
    // the phase in the table is read against the sensor chosen.
    sensors.value = chosen.sensor || (result.sensors || [{}])[0].key || '';
    views.value = chosen.view || (result.views || [''])[0];
}

async function loadPolar() {
    const box = document.getElementById('dt-polar');
    if (!state.result || !state.sessionId) {
        box.innerHTML = '';
        return;
    }
    const sensor = document.getElementById('dt-polar-sensor').value;
    const view = document.getElementById('dt-polar-view').value;
    if (!sensor || !view) return;
    let answer = null;
    try {
        answer = await apiJson('/api/dt/balancing-polar?session=' + state.sessionId
            + '&view=' + encodeURIComponent(view) + '&sensor=' + encodeURIComponent(sensor));
    } catch (error) {
        box.innerHTML = `<p class="panel-hint">${escapeHtml(error.message)}</p>`;
        return;
    }
    box.innerHTML = answer.charts.map((chart, index) => `
        <div>
            <p class="dt-polar-title">${escapeHtml(t('dtNode'))} ${chart.node}</p>
            <div id="dt-polar-chart-${index}" class="dt-polar-chart"></div>
        </div>`).join('');
    answer.charts.forEach((chart, index) => {
        const layout = Object.assign({}, chart.figure.layout, { autosize: true });
        window.Plotly.newPlot(`dt-polar-chart-${index}`, chart.figure.data, layout,
                              { responsive: true, displaylogo: false });
    });
}

function number(value, digits) {
    return value === null || value === undefined ? '-' : Number(value).toFixed(digits);
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
    // The last sensor is what makes the analysis possible, so the server is
    // asked again here: without this the button stayed disabled until the
    // screen was reopened.
    refreshAnalysis();
}

function render() {
    if (!state.step) return;
    const step = state.step;
    document.getElementById('dt-measuring-list').textContent = listText(step.measuring);
    document.getElementById('dt-balancing-list').textContent = planesText(step.balancing);

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

function planesText(planes) {
    if (!planes.length) return t('dtNoPlanes');
    const items = planes.map(plane => `${plane.node} (${number(plane.radius, 1)} mm)`);
    return `${t('dtNode')}: ${items.join(', ')}`;
}
