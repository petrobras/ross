// The BRBalance form. The numbers are calculated by POST /api/balance/calculate;
// this file only keeps the fields and draws the answer.
//
// The field keys (`ref_vib_amp`, `test1_vib_amp`, `test_1_vib_phase`, ...) are
// those of the .brb files the Digital Twin app wrote, irregular
// spelling included: changing them would stop old files from opening.
import { apiFetch } from '/core/api.js';
import { openCustomAlert } from '/components/modals.js';
import { escapeHtml } from '/core/dom.js';
import { onLanguageChanged, t } from '/core/i18n.js';
import { setStatus } from '../components/status.js';
import { apiDownload, apiJson } from '../core/api.js';

const state = {
    session_name: '',
    n_sensors: 2,
    n_tests: 2,
    n_planes: 2,
    n_weights: 2,
    notes: '',
    invert: [],
    refQuantity: 'vib',
    currQuantity: 'vib',
    testIndex: 0,
};

let lastResult = null;

const COUNTS = [
    ['bal-n-sensors', 'n_sensors', 1, 20],
    ['bal-n-tests', 'n_tests', 1, 10],
    ['bal-n-planes', 'n_planes', 1, 5],
    ['bal-n-weights', 'n_weights', 1, 8],
];

export function startBalance() {
    const root = document.getElementById('balance-root');
    if (!root) return;

    COUNTS.forEach(([id, key]) => {
        document.getElementById(id).addEventListener('change', event => {
            state[key] = clampCount(key, event.target.value);
            event.target.value = state[key];
            render();
        });
    });
    document.getElementById('bal-name').addEventListener('input', event => {
        state.session_name = event.target.value;
    });
    document.getElementById('bal-notes').addEventListener('input', event => {
        state.notes = event.target.value;
    });
    document.getElementById('bal-ref-quantity').addEventListener('change', event => {
        state.refQuantity = event.target.value;
        render();
    });
    document.getElementById('bal-curr-quantity').addEventListener('change', event => {
        state.currQuantity = event.target.value;
        render();
    });
    document.getElementById('bal-test-pick').addEventListener('change', event => {
        state.testIndex = Number(event.target.value) || 0;
        render();
    });
    document.getElementById('bal-copy').addEventListener('click', copyReference);
    document.getElementById('bal-calc').addEventListener('click', calculate);
    document.getElementById('bal-save').addEventListener('click', saveFile);
    document.getElementById('bal-report').addEventListener('click', downloadReport);
    document.getElementById('bal-open').addEventListener('click', () => {
        document.getElementById('bal-file').click();
    });
    document.getElementById('bal-file').addEventListener('change', openFile);

    root.addEventListener('input', onField);
    root.addEventListener('change', onInvert);
    onLanguageChanged(() => {
        render();
        if (lastResult) showResult(lastResult);
    });
    render();
}

function render() {
    fitAll();
    document.getElementById('bal-ref').innerHTML = pairTable(
        sensors(), quantityKeys('ref', state.refQuantity)
    );
    renderTestPicker();
    renderTest(state.testIndex);
    document.getElementById('bal-current').innerHTML = pairTable(
        sensors(), quantityKeys('curr', state.currQuantity)
    );
    document.getElementById('bal-corr').innerHTML = massTable('corr_plane', state.n_planes, state.n_weights);
    renderInvert();
    document.querySelectorAll('#balance-root table').forEach(refreshSums);
}

function quantityKeys(prefix, quantity) {
    return [prefix + '_' + quantity + '_amp', prefix + '_' + quantity + '_phase'];
}

function renderTestPicker() {
    const select = document.getElementById('bal-test-pick');
    select.innerHTML = Array.from({ length: state.n_tests }, (_, index) =>
        `<option value="${index}">${escapeHtml(t('balTest'))} ${index + 1}</option>`
    ).join('');
    select.value = String(state.testIndex);
}

function renderTest(index) {
    const number = index + 1;
    document.getElementById('bal-tests').innerHTML =
        `<p class="panel-section-title">${escapeHtml(t('balTestVibration'))} ${number}</p>` +
        `<div class="table-scroll">${pairTable(sensors(), ['test' + number + '_vib_amp', 'test_' + number + '_vib_phase'])}</div>` +
        `<p class="panel-section-title">${escapeHtml(t('balTestMasses'))}</p>` +
        `<div class="table-scroll">${massTable('test' + index + '_plane', state.n_planes, state.n_weights)}</div>`;
}

function pairTable(rows, [ampKey, phaseKey]) {
    const body = rows.map((label, index) =>
        `<tr><td>${escapeHtml(label)}</td><td>${field(ampKey, index)}</td><td>${field(phaseKey, index)}</td></tr>`
    ).join('');
    return `<table class="data-table"><thead><tr><th>${escapeHtml(t('balIdentification'))}</th>` +
        `<th>${escapeHtml(t('balAmplitude'))}</th><th>${escapeHtml(t('balPhase'))}</th></tr></thead>` +
        `<tbody>${body}</tbody></table>`;
}

function massTable(prefix, planes, weights) {
    let header = `<tr><th rowspan="2">${escapeHtml(t('balIdentification'))}</th>`;
    let units = '<tr>';
    for (let plane = 0; plane < planes; plane += 1) {
        header += `<th colspan="2">${escapeHtml(t('balPlane'))} ${plane + 1}</th>`;
        units += `<th>${escapeHtml(t('balMass'))}</th><th>${escapeHtml(t('balPhase'))}</th>`;
    }
    let body = '';
    for (let weight = 0; weight < weights; weight += 1) {
        body += `<tr><td>${escapeHtml(t('balWeight'))} ${weight + 1}</td>`;
        for (let plane = 0; plane < planes; plane += 1) {
            const key = prefix + plane;
            body += `<td>${field(key + '_mass', weight, key)}</td><td>${field(key + '_phase', weight, key)}</td>`;
        }
        body += '</tr>';
    }
    let totals = `<tr class="total"><td>${escapeHtml(t('balTotal'))}</td>`;
    for (let plane = 0; plane < planes; plane += 1) {
        const key = prefix + plane;
        totals += `<td data-sum-mag="${key}"></td><td data-sum-phase="${key}"></td>`;
    }
    totals += '</tr>';
    return `<table class="data-table"><thead>${header}</tr>${units}</tr></thead><tbody>${body}${totals}</tbody></table>`;
}

function field(key, index, sumKey) {
    const list = state[key] || [];
    const sum = sumKey ? ` data-sum="${sumKey}"` : '';
    const value = list[index] == null ? '0' : list[index];
    return `<input data-key="${key}" data-index="${index}"${sum} value="${escapeHtml(value)}" inputmode="decimal">`;
}

function renderInvert() {
    document.getElementById('bal-invert').innerHTML = state.invert.map((flag, plane) =>
        `<div class="input-group"><label>${escapeHtml(t('balInvertPlane'))} ${plane + 1}</label>` +
        `<select data-invert="${plane}">` +
        `<option value="false"${flag ? '' : ' selected'}>${escapeHtml(t('balNo'))}</option>` +
        `<option value="true"${flag ? ' selected' : ''}>${escapeHtml(t('balYes'))}</option>` +
        '</select></div>'
    ).join('');
}

function onField(event) {
    const input = event.target;
    if (!input.dataset || !input.dataset.key) return;
    state[input.dataset.key][Number(input.dataset.index)] = input.value;
    if (input.dataset.sum) refreshSums(input.closest('table'));
}

function onInvert(event) {
    const select = event.target;
    if (!select.dataset || select.dataset.invert == null) return;
    state.invert[Number(select.dataset.invert)] = select.value === 'true';
}

function refreshSums(scope) {
    if (!scope) return;
    scope.querySelectorAll('[data-sum-mag]').forEach(cell => {
        const key = cell.dataset.sumMag;
        const [magnitude, phase] = polarSum(state[key + '_mass'], state[key + '_phase']);
        cell.textContent = magnitude.toFixed(2);
        cell.parentElement.querySelector(`[data-sum-phase="${key}"]`).textContent = phase.toFixed(2);
    });
}

function polarSum(amps, phases) {
    let real = 0;
    let imag = 0;
    const size = Math.max((amps || []).length, (phases || []).length);
    for (let index = 0; index < size; index += 1) {
        const magnitude = Number(amps[index]) || 0;
        const angle = ((Number(phases[index]) || 0) * Math.PI) / 180;
        real += magnitude * Math.cos(angle);
        imag += magnitude * Math.sin(angle);
    }
    let degrees = (Math.atan2(imag, real) * 180) / Math.PI;
    if (degrees < 0) degrees += 360;
    return [Math.hypot(real, imag), degrees];
}

function copyReference() {
    ['vib_amp', 'vib_phase', 'run_amp', 'run_phase'].forEach(suffix => {
        state['curr_' + suffix] = state['ref_' + suffix].slice();
    });
    render();
}

function fitAll() {
    ['ref', 'curr'].forEach(prefix => {
        ['vib_amp', 'vib_phase', 'run_amp', 'run_phase'].forEach(suffix => {
            fit(prefix + '_' + suffix, state.n_sensors);
        });
    });
    for (let test = 0; test < state.n_tests; test += 1) {
        fit('test' + (test + 1) + '_vib_amp', state.n_sensors);
        fit('test_' + (test + 1) + '_vib_phase', state.n_sensors);
        for (let plane = 0; plane < state.n_planes; plane += 1) {
            fit('test' + test + '_plane' + plane + '_mass', state.n_weights);
            fit('test' + test + '_plane' + plane + '_phase', state.n_weights);
        }
    }
    for (let plane = 0; plane < state.n_planes; plane += 1) {
        fit('corr_plane' + plane + '_mass', state.n_weights);
        fit('corr_plane' + plane + '_phase', state.n_weights);
    }
    while (state.invert.length < state.n_planes) state.invert.push(false);
    state.invert.length = state.n_planes;
    if (state.testIndex >= state.n_tests) state.testIndex = 0;
}

function fit(key, size) {
    if (!Array.isArray(state[key])) state[key] = [];
    while (state[key].length < size) state[key].push('0');
    state[key].length = size;
}

function sensors() {
    return Array.from({ length: state.n_sensors }, (_, index) => t('balSensor') + ' ' + (index + 1));
}

function clampCount(key, raw) {
    const found = COUNTS.find(item => item[1] === key);
    const value = Number(raw);
    if (!Number.isFinite(value)) return state[key];
    return Math.min(found[3], Math.max(found[2], Math.round(value)));
}

function payload() {
    const body = {
        session_name: state.session_name,
        n_sensors: state.n_sensors,
        n_tests: state.n_tests,
        n_planes: state.n_planes,
        n_weights: state.n_weights,
        notes: state.notes,
        invert: state.invert.slice(),
    };
    Object.keys(state).forEach(key => {
        if (Array.isArray(state[key]) && key !== 'invert') body[key] = state[key].slice();
    });
    return body;
}

async function calculate() {
    const button = document.getElementById('bal-calc');
    button.disabled = true;
    setStatus('bal-status', t('balCalculating'), 'busy');
    try {
        const body = await apiJson('/api/balance/calculate', payload());
        lastResult = body;
        showResult(body);
        setStatus('bal-status', t('balCalculated'), 'ok');
    } catch (error) {
        setStatus('bal-status', '');
        openCustomAlert(error.message);
    } finally {
        button.disabled = false;
    }
}

function showResult(body) {
    const groups = [
        ['balRecommended', body.correction, 'g'],
        ['balApplied', body.applied, 'g'],
        ['balPredicted', body.predicted, 'µm'],
    ];
    document.getElementById('bal-results').innerHTML = '<div class="result-grid">' + groups.map(([key, rows, unit]) => {
        const items = (rows || []).map(row =>
            `<li><span>${escapeHtml(row.name)}</span><span>${Number(row.magnitude).toFixed(2)} ${unit} ∠ ${Number(row.phase).toFixed(2)}°</span></li>`
        ).join('');
        return `<div><h4>${escapeHtml(t(key))}</h4><ul class="result-list">${items}</ul></div>`;
    }).join('') + '</div>';
    document.getElementById('bal-results-card').style.display = '';

    const plot = document.getElementById('bal-plot');
    if (body.figure && window.Plotly) {
        plot.innerHTML = '';
        window.Plotly.newPlot(plot, body.figure.data, body.figure.layout, { responsive: true, displaylogo: false });
    } else if (body.plot_error) {
        plot.innerHTML = `<div class="analysis-no-chart" style="height: 100%;"><i class="fas fa-exclamation-triangle"></i><p>${escapeHtml(body.plot_error)}</p></div>`;
    }
}

async function downloadReport() {
    try {
        await apiDownload('/api/balance/report', payload(), 'BRBalance_Report.docx');
    } catch (error) {
        openCustomAlert(error.message);
    }
}

async function saveFile() {
    try {
        await apiDownload('/api/balance/save', payload(), (state.session_name || 'balance') + '.brb');
    } catch (error) {
        openCustomAlert(error.message);
    }
}

async function openFile(event) {
    const file = event.target.files && event.target.files[0];
    event.target.value = '';
    if (!file) return;
    const body = new FormData();
    body.append('file', file);
    try {
        const response = await apiFetch('/api/balance/open', { method: 'POST', body });
        const loaded = await response.json();
        if (!response.ok) throw new Error(loaded.message || 'HTTP ' + response.status);
        applyLoaded(loaded);
        setStatus('bal-status', t('balLoaded'), 'ok');
    } catch (error) {
        openCustomAlert(error.message);
    }
}

function applyLoaded(loaded) {
    ['session_name', 'notes'].forEach(key => {
        if (typeof loaded[key] === 'string') state[key] = loaded[key];
    });
    COUNTS.forEach(([id, key]) => {
        if (loaded[key] != null) state[key] = clampCount(key, loaded[key]);
        document.getElementById(id).value = state[key];
    });
    document.getElementById('bal-name').value = state.session_name;
    document.getElementById('bal-notes').value = state.notes || '';
    Object.keys(loaded).forEach(key => {
        if (key === 'invert' && Array.isArray(loaded.invert)) {
            state.invert = loaded.invert.map(item => item === true || item === 'true');
        } else if (Array.isArray(loaded[key])) {
            state[key] = loaded[key].map(item => String(item));
        }
    });
    render();
}
