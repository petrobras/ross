// Digital Twin, step 4: measuring planes, balancing planes and sensors, as in
// section 4 of the Digital Twin app. The server keeps the planes and writes each
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
    // The analysis parameters the server answered: the preset, its fields
    // and the presets on offer.
    parameters: null,
};

export const MEASUREMENT_ACTIONS = {
    'dt-plane-add': element => changePlanes(element.dataset.kind, true),
    'dt-plane-remove': element => changePlanes(element.dataset.kind, false),
    'dt-sensor-apply': element => applySensor(Number(element.dataset.node), element.dataset.plane),
    'dt-run-balancing': () => runBalancing(),
    'dt-parameters-preset': () => fillParameters(chosenPreset()),
    // The interface dispatches a select's `change` through the same actions
    // (core/actions.js), so the pickers of the result card come here.
    'dt-polar-change': () => renderBalancing(state.result),
    'dt-chart-change': () => showChart(),
    'dt-chart-draw': () => drawChart(),
};

// The charts of a run, in the order the picker offers them. `node` says which
// planes a chart is drawn for: the balancing planes the estimate used, the
// planes of the virtual sensing, or none (one chart per balancing plane, or
// the whole rotor).
const CHARTS = [
    { kind: 'polar', label: 'dtChartPolar', hint: 'dtPolarHint', node: null },
    { kind: 'decomposed', label: 'dtChartDecomposed', hint: 'dtDecomposedHint', node: 'planes' },
    { kind: 'forces_1d', label: 'dtChartForces1d', node: 'planes' },
    { kind: 'forces_2d', label: 'dtChartForces2d', node: 'planes' },
    { kind: 'vs_1d', label: 'dtChartVs1d', hint: 'dtVirtualSensingHint', node: 'vs' },
    { kind: 'vs_2d', label: 'dtChartVs2d', hint: 'dtVirtualSensingHint', node: 'vs' },
    { kind: 'vs_3d', label: 'dtChartVs3d', hint: 'dtVirtualSensingHint', node: null },
];

export function startMeasurement() {
    if (!document.getElementById('dt-planes-form')) return;
    onLanguageChanged(render);
    useMeasurement(null, null);
}

// Called when a model is created (with the step's state the server answered)
// and when the model is cleared (with nulls).
export function useMeasurement(sessionId, step) {
    if (state.sessionId !== sessionId) {
        state.parameters = null;
        document.getElementById('dt-parameters-fields').innerHTML = '';
    }
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
            preset: document.getElementById('dt-parameters-preset').value,
            parameters: parameterValues(),
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
    showParameters(answer.parameters);
    renderBalancing(answer.balancing);
}

// The analysis parameters: a preset (the rossml defaults or a machine of the
// Digital Twin app) fills every field, and any field can be changed after.
// The server answers them on every refresh; the fields are redrawn only when
// the preset differs, so what the user typed is kept until the run sends it.
const PARAMETER_GROUPS = ['signal', 'kalman', 'limits', 'direct', 'indirect'];

function showParameters(parameters) {
    if (!parameters) return;
    const changed = !state.parameters || state.parameters.preset !== parameters.preset;
    state.parameters = parameters;
    const select = document.getElementById('dt-parameters-preset');
    select.innerHTML = parameters.presets.map(item => {
        const label = item.models.length ? item.models.join(', ') : t('dtParametersDefaults');
        return `<option value="${escapeHtml(item.id)}">${escapeHtml(label)}</option>`;
    }).join('');
    select.value = parameters.preset;
    if (changed || !document.getElementById('dt-parameters-fields').children.length) {
        drawParameterFields(parameters.fields, parameters.values);
    }
}

function drawParameterFields(fields, values) {
    document.getElementById('dt-parameters-fields').innerHTML = PARAMETER_GROUPS.map(group => {
        const inputs = fields.filter(field => field.group === group).map(field => {
            const value = values[field.key];
            return `<div class="input-group">
                <label for="dt-param-${field.key}">${escapeHtml(t('dtParam_' + field.key))}</label>
                <input id="dt-param-${field.key}" type="number" step="${field.whole ? 1 : 'any'}"
                       value="${value === null || value === undefined ? '' : value}">
            </div>`;
        }).join('');
        return `<fieldset class="dt-parameter-group"><legend>${escapeHtml(t('dtParamGroup_' + group))}</legend>${inputs}</fieldset>`;
    }).join('');
}

function chosenPreset() {
    const id = document.getElementById('dt-parameters-preset').value;
    return (state.parameters?.presets || []).find(item => item.id === id) || null;
}

function fillParameters(preset) {
    if (!preset || !state.parameters) return;
    state.parameters = { ...state.parameters, preset: preset.id, values: preset.values };
    drawParameterFields(state.parameters.fields, preset.values);
}

// The fields as the run sends them: an empty one is null, the rossml default.
function parameterValues() {
    const values = {};
    (state.parameters?.fields || []).forEach(field => {
        const input = document.getElementById(`dt-param-${field.key}`);
        const text = input ? String(input.value).trim() : '';
        values[field.key] = text === '' ? null : Number(text);
    });
    return values;
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
        document.getElementById('dt-charts').innerHTML = '';
        return;
    }
    card.style.display = '';
    fillPolarChoices(result);
    fillChartChoices(result);
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
    showChart();
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
    const chosen = sensors.value;
    sensors.innerHTML = (result.sensors || []).map(sensor =>
        `<option value="${escapeHtml(sensor.key)}">${escapeHtml(t('dtNode'))} ${sensor.node} - ${escapeHtml(sensor.plane)}</option>`).join('');
    // The first option is chosen here rather than left to the browser, which
    // selects it on its own: the choice has to be readable right away, because
    // the phase in the table is read against the sensor chosen.
    const keys = (result.sensors || []).map(sensor => sensor.key);
    sensors.value = keys.includes(chosen) ? chosen : (keys[0] || '');
}

function sensorIndex() {
    const key = document.getElementById('dt-polar-sensor').value;
    const sensor = ((state.result || {}).sensors || []).find(item => item.key === key);
    return sensor ? sensor.index : 0;
}

function fillChartChoices(result) {
    const kinds = document.getElementById('dt-chart-kind');
    const chosen = kinds.value;
    const offered = CHARTS.filter(chart => (result.charts || []).includes(chart.kind));
    kinds.innerHTML = offered.map(chart =>
        `<option value="${chart.kind}">${escapeHtml(t(chart.label))}</option>`).join('');
    kinds.value = offered.some(chart => chart.kind === chosen) ? chosen : (offered[0] || {}).kind || '';
    // A new run brings its own planes: the positions start at the radius the
    // correction was computed for.
    const radius = ((result.planes || [])[0] || {}).radius;
    ['dt-decomp-radius-a', 'dt-decomp-radius-b'].forEach(id => {
        const input = document.getElementById(id);
        if (!input.value && radius) input.value = String(radius);
    });
}

function chosenChart() {
    const kind = document.getElementById('dt-chart-kind').value;
    return CHARTS.find(chart => chart.kind === kind) || null;
}

// The picker of planes follows the chart, and a chart is drawn as soon as it
// is chosen, except the decomposition, which waits for its positions.
function showChart() {
    const chart = chosenChart();
    const result = state.result;
    if (!chart || !result) return;
    const nodes = chartNodes(chart, result);
    const picker = document.getElementById('dt-chart-node');
    const chosen = Number(picker.value);
    picker.innerHTML = nodes.map(node => `<option value="${node}">${node}</option>`).join('');
    picker.value = String(nodes.includes(chosen) ? chosen : (nodes[0] ?? ''));
    document.getElementById('dt-chart-node-group').style.display = nodes.length ? '' : 'none';
    document.getElementById('dt-decomposition').style.display = chart.kind === 'decomposed' ? '' : 'none';
    document.getElementById('dt-chart-hint').textContent = chart.hint ? t(chart.hint) : '';
    if (chart.kind === 'decomposed') {
        document.getElementById('dt-charts').innerHTML = '';
        return;
    }
    drawChart();
}

function chartNodes(chart, result) {
    if (chart.node === 'planes') return result.planes.map(plane => plane.node);
    if (chart.node === 'vs') return result.vs_nodes || [];
    return [];
}

async function drawChart() {
    const box = document.getElementById('dt-charts');
    const chart = chosenChart();
    if (!state.result || !state.sessionId || !chart) {
        box.innerHTML = '';
        return;
    }
    const body = { session: state.sessionId, chart: chart.kind, sensor: sensorIndex() };
    if (chart.node) body.node = Number(document.getElementById('dt-chart-node').value);
    if (chart.kind === 'decomposed') {
        const value = id => Number(document.getElementById(id).value);
        body.positions = [value('dt-decomp-angle-a'), value('dt-decomp-angle-b')];
        body.radii = [value('dt-decomp-radius-a'), value('dt-decomp-radius-b')];
    }
    // The answer is drawn only while it is still the chart on screen: choosing
    // another one while rossml draws must not be overwritten by the older one.
    const asked = JSON.stringify(body);
    state.chartAsked = asked;
    box.innerHTML = '';
    const stop = showLoader('dt-chart', box, { label: t('dtDrawing'), top: '50%' });
    let answer = null;
    try {
        answer = await apiJson('/api/dt/balancing-figure', body);
    } catch (error) {
        stop();
        if (state.chartAsked === asked) {
            box.innerHTML = `<p class="panel-hint">${escapeHtml(error.message)}</p>`;
        }
        return;
    }
    stop();
    if (state.chartAsked !== asked) return;
    const wide = chart.kind !== 'polar' && chart.kind !== 'decomposed';
    box.classList.toggle('dt-chart-wide', wide);
    box.innerHTML = answer.charts.map((item, index) => `
        <div>
            ${item.node === null || item.node === undefined ? '' : `<p class="dt-polar-title">${escapeHtml(t('dtNode'))} ${item.node}</p>`}
            <div id="dt-chart-${index}" class="dt-polar-chart"></div>
        </div>`).join('');
    answer.charts.forEach((item, index) => {
        const layout = Object.assign({}, item.figure.layout, { autosize: true, width: null, height: null });
        window.Plotly.newPlot(`dt-chart-${index}`, item.figure.data, layout,
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
