// Digital Twin analyses (Campbell, unbalance response, clearances, stability):
// one card per analysis, drawn like the reference interface's dashboard cards
// -- the chart on top, the controls underneath, Update/Help/Delete in the header.
//
// The rotor lives in the server session. A card sends its fields to
// POST /api/dt/analysis and draws the figures and tables that come back.
import { openCustomAlert, openCustomConfirm } from '../components/modals.js';
import { apiJson } from '../core/api.js';
import { escapeHtml } from '../core/dom.js';
import { t } from '../core/i18n.js';

// What each card asks for. `label` is an i18n key; the server reads the same
// `id`s and validates them (services/analysis/*.py).
const CATALOG = {
    campbell: {
        title: 'anCampbell',
        icon: 'fas fa-chart-line',
        fields: [
            { id: 'speed_min', label: 'anSpeedMin', val: 0 },
            { id: 'speed_max', label: 'anSpeedMax', val: 3000 },
            { id: 'points', label: 'anPoints', val: 25 },
            { id: 'harmonics', label: 'anHarmonics', val: '1', type: 'text' },
        ],
    },
    unbalance: {
        title: 'anUnbalance',
        icon: 'fas fa-wave-square',
        fields: [
            { id: 'freq_min', label: 'anFreqMin', val: 315 },
            { id: 'freq_max', label: 'anFreqMax', val: 1150 },
            { id: 'points', label: 'anPoints', val: 20 },
            { id: 'mcs', label: 'anMcs', val: 1150 },
            { id: 'percent', label: 'anPercent', val: 100 },
            { id: 'probe1_node', label: 'anProbe1Node', val: rotor => rotor.bearing_nodes[0] || 0 },
            { id: 'probe1_angle', label: 'anProbe1Angle', val: 0 },
            { id: 'probe2_node', label: 'anProbe2Node', val: rotor => rotor.bearing_nodes[rotor.bearing_nodes.length - 1] || 0 },
            { id: 'probe2_angle', label: 'anProbe2Angle', val: 0 },
            { id: 'unbalances', label: 'anExtraUnbalances', type: 'unbalance_list' },
        ],
    },
    clearance: {
        title: 'anClearance',
        icon: 'fas fa-arrows-alt-v',
        fields: [{ id: 'speed', label: 'anSpeed', val: 1150 }],
    },
    stability: {
        title: 'anStability',
        icon: 'fas fa-balance-scale-right',
        fields: [{ id: 'speed', label: 'anSpeed', val: 0 }],
    },
};

const state = {
    sessionId: null,
    rotor: null,
    counter: 0,
    cards: new Map(),
};

export function startAnalyses() {
    const select = document.getElementById('dt-analysis-type');
    if (!select) return;
    fillAnalysisTypes();
    document.getElementById('dt-analysis-add').addEventListener('click', () => {
        if (select.value) addCard(select.value);
    });
}

function fillAnalysisTypes() {
    const select = document.getElementById('dt-analysis-type');
    select.innerHTML = `<option value="" data-i18n="anPick">${escapeHtml(t('anPick'))}</option>` +
        Object.entries(CATALOG).map(([key, entry]) =>
            `<option value="${key}" data-i18n="${entry.title}">${escapeHtml(t(entry.title))}</option>`
        ).join('');
}

// A new model makes every card stale: the cards are removed with it.
export function useModel(sessionId, rotor) {
    state.sessionId = sessionId;
    state.rotor = rotor;
    state.cards.clear();
    const list = document.getElementById('dt-analysis-list');
    list.innerHTML = emptyNotice();
    const button = document.getElementById('dt-go-analysis');
    if (button) button.disabled = !rotor;
}

function emptyNotice() {
    return `<p class="dashboards-empty" style="color: #888; text-align: center; margin-top: 20%;" data-i18n="anWillAppear">${escapeHtml(t('anWillAppear'))}</p>`;
}

function addCard(type) {
    if (!state.rotor) {
        openCustomAlert(t('anNeedModel'));
        return;
    }
    const entry = CATALOG[type];
    const id = 'an' + (++state.counter);
    state.cards.set(id, { type });
    const list = document.getElementById('dt-analysis-list');
    if (list.querySelector('.dashboards-empty')) list.innerHTML = '';
    list.insertAdjacentHTML('afterbegin', `
        <div class="analysis-card" id="card-${id}">
            <div class="analysis-header" data-collapse="body-${id}">
                <span class="analysis-title"><i class="${entry.icon}"></i> <span data-i18n="${entry.title}">${escapeHtml(t(entry.title))}</span></span>
                <div class="analysis-actions">
                    <button class="btn-update-analysis" data-card-action="run" data-card="${id}"><i class="fas fa-sync-alt"></i> <span data-i18n="update">${escapeHtml(t('update'))}</span></button>
                    <button class="btn-help-analysis" data-help="${type}"><i class="fas fa-question-circle"></i> <span data-i18n="help">${escapeHtml(t('help'))}</span></button>
                    <button class="btn-delete-analysis" data-card-action="delete" data-card="${id}"><i class="fas fa-trash"></i> <span data-i18n="delete">${escapeHtml(t('delete'))}</span></button>
                    <span><i class="fas fa-chevron-down"></i></span>
                </div>
            </div>
            <div class="analysis-body" id="body-${id}" style="padding:0; background:var(--bg-workspace); position: relative;">
                <div id="plot-${id}" style="min-height: 400px; width: 100%; overflow: hidden; position:relative; background: var(--bg-card);"></div>
                ${controlsHTML(id, entry)}
            </div>
        </div>`);
    runCard(id);
}

function controlsHTML(id, entry) {
    const html = entry.fields.map(field => {
        if (field.type === 'unbalance_list') {
            return `<div class="dash-control-group" style="flex-direction: column; align-items: stretch; grid-column: 1 / -1;">
                <div style="display:flex; justify-content:space-between; align-items: center; width:100%; margin-bottom:8px;">
                    <label style="margin:0;" data-i18n="${field.label}">${escapeHtml(t(field.label))}</label>
                    <button type="button" class="btn-add-probe" data-card-action="add-unbalance" data-card="${id}"><i class="fas fa-plus"></i></button>
                </div>
                <div class="probe-list-container" id="unb-container-${id}"></div>
            </div>`;
        }
        const value = typeof field.val === 'function' ? field.val(state.rotor) : field.val;
        return `<div class="dash-control-group" style="flex-direction: column; align-items: stretch; gap: 6px;">
            <label style="margin: 0; min-width: auto;" data-i18n="${field.label}">${escapeHtml(t(field.label))}</label>
            <div class="unified-input"><input type="${field.type || 'number'}" id="input-${field.id}-${id}" value="${escapeHtml(value)}"></div>
        </div>`;
    }).join('');
    return `<div class="light-dashboard-controls">${html}</div>`;
}

function unbalanceRowHTML() {
    return `
    <div class="probe-row" style="align-items:center;">
        <span style="font-size:11px; color:var(--text-muted);" data-i18n="anNode">${escapeHtml(t('anNode'))}</span>
        <input type="number" class="unb-node" value="0" min="0">
        <span style="font-size:11px; color:var(--text-muted); margin-left:8px;" data-i18n="anUnbalanceGmm">${escapeHtml(t('anUnbalanceGmm'))}</span>
        <input type="number" class="unb-mag" value="0" step="any" min="0">
        <span style="font-size:11px; color:var(--text-muted); margin-left:8px;" data-i18n="anPhaseDeg">${escapeHtml(t('anPhaseDeg'))}</span>
        <input type="number" class="unb-phase" value="0" step="any">
        <button type="button" class="btn-remove-probe" style="margin-left:auto;" data-card-action="remove-row"><i class="fas fa-times"></i></button>
    </div>`;
}

// The global click listener in main.js hands card buttons here.
export async function cardAction(node) {
    const id = node.dataset.card;
    switch (node.dataset.cardAction) {
        case 'run':
            runCard(id);
            break;
        case 'delete':
            if (await openCustomConfirm(t('anConfirmDelete'))) {
                document.getElementById('card-' + id).remove();
                state.cards.delete(id);
                if (!state.cards.size) document.getElementById('dt-analysis-list').innerHTML = emptyNotice();
            }
            break;
        case 'add-unbalance':
            document.getElementById('unb-container-' + id).insertAdjacentHTML('beforeend', unbalanceRowHTML());
            break;
        case 'remove-row':
            node.closest('.probe-row').remove();
            break;
    }
}

function cardParameters(id, type) {
    const params = {};
    CATALOG[type].fields.forEach(field => {
        if (field.type === 'unbalance_list') {
            params[field.id] = [...document.querySelectorAll(`#unb-container-${id} .probe-row`)].map(row => ({
                node: row.querySelector('.unb-node').value,
                magnitude: row.querySelector('.unb-mag').value,
                phase: row.querySelector('.unb-phase').value,
            }));
        } else {
            params[field.id] = document.getElementById(`input-${field.id}-${id}`).value;
        }
    });
    return params;
}

async function runCard(id) {
    const card = state.cards.get(id);
    if (!card) return;
    const plot = document.getElementById('plot-' + id);
    const params = cardParameters(id, card.type);
    const token = (card.token || 0) + 1;
    card.token = token;
    const stopLoader = showLoader(id, plot);
    try {
        const result = await apiJson('/api/dt/analysis', { session: state.sessionId, type: card.type, params });
        if (card.token !== token) return;
        drawResult(id, plot, result);
    } catch (error) {
        if (card.token !== token) return;
        plot.innerHTML = `<div class="analysis-no-chart analysis-unsupported"><i class="fas fa-exclamation-triangle"></i><p>${escapeHtml(error.message)}</p></div>`;
    } finally {
        if (card.token === token) stopLoader();
    }
}

// The reference card's loader: a box over the chart, with the seconds running.
function showLoader(id, plot) {
    plot.style.opacity = '0.4';
    let loader = document.getElementById('loading-' + id);
    if (!loader) {
        loader = document.createElement('div');
        loader.id = 'loading-' + id;
        loader.innerHTML = `
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 8px;">
                <i class="fas fa-sync fa-spin fa-2x" style="color: var(--accent-primary);"></i>
                <span style="font-weight: 600; font-size: 13px;">${escapeHtml(t('updating'))}</span>
                <span id="loading-when-${id}" class="job-elapsed"></span>
            </div>`;
        Object.assign(loader.style, {
            position: 'absolute', top: '200px', left: '50%', transform: 'translate(-50%, -50%)',
            zIndex: '10', color: 'var(--text-main)', background: 'var(--bg-card)', padding: '15px 25px',
            borderRadius: 'var(--radius-ui)', boxShadow: 'var(--shadow-card)', border: '1px solid var(--border-color)',
        });
        plot.parentElement.appendChild(loader);
    }
    const started = Date.now();
    const when = document.getElementById('loading-when-' + id);
    const timer = setInterval(() => {
        when.textContent = Math.round((Date.now() - started) / 1000) + ' s';
    }, 1000);
    return () => {
        clearInterval(timer);
        loader.remove();
        plot.style.opacity = '1';
    };
}

function drawResult(id, plot, result) {
    const figures = result.figures || [];
    let html = '';
    if (result.layout === 'campbell') {
        html += `<div class="campbell-split" style="padding: 10px;">
            <div id="fig-${id}-0" class="campbell-diagram"></div>
            <div id="mode-${id}" class="campbell-mode"><div class="campbell-hint"><i class="fas fa-hand-pointer"></i><br><span data-i18n="anModeHint">${escapeHtml(t('anModeHint'))}</span></div></div>
        </div>`;
    } else if (result.layout === 'split') {
        html += `<div class="campbell-split" style="padding: 10px;">${figures.map((_, index) =>
            `<div id="fig-${id}-${index}" class="campbell-diagram"></div>`).join('')}</div>`;
    } else {
        html += figures.map((_, index) => `<div id="fig-${id}-${index}" style="min-height: 460px;"></div>`).join('');
    }
    if ((result.tables || []).length) {
        html += `<div class="analysis-body"><div class="result-grid">${result.tables.map(tableHTML).join('')}</div></div>`;
    }
    plot.innerHTML = html;
    plot.style.minHeight = figures.length ? '400px' : '0';
    figures.forEach((fig, index) => {
        const layout = Object.assign({}, fig.layout, { autosize: true });
        window.Plotly.newPlot(`fig-${id}-${index}`, fig.data, layout, { responsive: true, displaylogo: false });
    });
    if (result.layout === 'campbell') wireModeShape(id);
}

function tableHTML(table) {
    const head = table.columns.map(name => `<th>${escapeHtml(name)}</th>`).join('');
    const rows = table.rows.map(row => '<tr>' + row.map(cell => {
        if (cell === null) return '<td class="missing">—</td>';
        if (typeof cell === 'number' && !Number.isInteger(cell)) return `<td>${formatNumber(cell)}</td>`;
        return `<td>${escapeHtml(cell)}</td>`;
    }).join('') + '</tr>').join('');
    return `<div><h4>${escapeHtml(table.title)}</h4><div class="table-scroll"><table class="data-table"><thead><tr>${head}</tr></thead><tbody>${rows}</tbody></table></div></div>`;
}

function formatNumber(value) {
    const size = Math.abs(value);
    if (size !== 0 && (size < 1e-3 || size >= 1e6)) return value.toExponential(4);
    return value.toFixed(4);
}

// A click on the diagram asks for the 3D mode of that point. Only the last
// click owns the panel: an answer for an earlier point is dropped.
function wireModeShape(id) {
    const diagram = document.getElementById(`fig-${id}-0`);
    const panel = document.getElementById(`mode-${id}`);
    let latest = 0;
    diagram.on('plotly_click', async event => {
        const point = event && event.points && event.points[0];
        if (!point) return;
        const ticket = ++latest;
        panel.innerHTML = `<div class="campbell-hint"><i class="fas fa-spinner fa-spin"></i><br>${escapeHtml(t('anModeLoading'))}</div>`;
        try {
            const answer = await apiJson('/api/dt/campbell/mode-shape', {
                session: state.sessionId,
                point: { x: point.x, y: point.y, curve_name: (point.data && point.data.name) || null },
            });
            if (ticket !== latest) return;
            const fig = answer.figure;
            panel.innerHTML = '';
            window.Plotly.newPlot(panel, {
                data: fig.data,
                layout: Object.assign({}, fig.layout, { autosize: true }),
                frames: fig.frames || [],
                config: { responsive: true, displaylogo: false },
            });
        } catch (error) {
            if (ticket !== latest) return;
            panel.innerHTML = `<div class="campbell-hint campbell-error">${escapeHtml(error.message)}</div>`;
        }
    });
}
