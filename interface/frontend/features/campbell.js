// The Campbell diagram with mode shape: clicking a point asks the backend for the
// 3D figure of that mode and draws it alongside.
//
// This used to be an iframe pointing at a Dash server on a random port, which the
// backend discovered by spying on its own sys.stdout (BE-04).
import { ANALYSES } from '../core/analysis_store.js';
import { runJob, wasCancelled, projectForServer } from '../core/api.js';
import { escapeHtml } from '../core/dom.js';
import { state } from '../core/state.js';
import { t } from '../core/i18n.js';
// --- Campbell with mode shape ---------------------------------------------
//
// Three paths draw a card: running the analysis, restoring from memory when
// coming back from the Hub, and loading from a file. All three ask here.

export function isModeShape(type, params) {
    return type === 'campbell' && (params || {}).plot_type === 'Mode Shape';
}

export function prepareModeShapePanels(card, plotId) {
    card.innerHTML = ''
        + `<div class="campbell-split">`
        +   `<div id="${plotId}-campbell" class="campbell-diagram"></div>`
        +   `<div id="${plotId}-mode" class="campbell-mode">`
        +     `<div class="campbell-hint"><i class="fas fa-hand-pointer"></i><br>`
        +     `${escapeHtml(t('modeShapeHint'))}</div>`
        +   `</div>`
        + `</div>`;
    return plotId + '-campbell';
}

export function wireModeShapeClick(card, diagramId, uniqueId) {
    const diagram = document.getElementById(diagramId);
    const panel = document.getElementById(diagramId.replace(/-campbell$/, '-mode'));
    if (!diagram || !diagram.on || !panel) return;

    diagram.on('plotly_click', async event => {
        const point = event && event.points && event.points[0];
        if (!point) return;

        panel.innerHTML = `<div class="campbell-hint">`
            + `<i class="fas fa-spinner fa-spin"></i><br>${escapeHtml(t('modeShapeLoading'))}</div>`;

        try {
            // One request per card: clicking several points in a row cancels the earlier
            // ones, otherwise the answer for an old point can arrive later and show the
            // wrong mode.
            const data = await runJob('mode:' + uniqueId, '/api/campbell/mode_shape', {
                project: projectForServer(state.projectData),
                conversion_type: (ANALYSES.get(uniqueId) || {}).conversion || '',
                params: (ANALYSES.get(uniqueId) || {}).params || {},
                point: {
                    x: point.x,
                    y: point.y,
                    curve_name: (point.data && point.data.name) || null
                }
            });
            // Another point was clicked: that click owns the panel now.
            if (data && data.status === 'superseded') return;
            // The user stopped everything. Said, and not silently undone: the
            // panel is where they were looking when they pressed the button.
            if (data && data.status === 'cancelled') {
                panel.innerHTML = `<div class="campbell-hint">`
                    + `<i class="fas fa-ban"></i><br>${escapeHtml(t('workStopped'))}</div>`;
                return;
            }
            if (!data || data.status !== 'success') {
                panel.innerHTML = `<div class="campbell-hint campbell-error">`
                    + `${escapeHtml((data && data.message) || 'Error')}</div>`;
                return;
            }
            const fig = JSON.parse(data.plot_json);
            fig.layout.autosize = true;
            panel.innerHTML = '';
            Plotly.newPlot(panel, {
                data: fig.data,
                layout: fig.layout,
                frames: fig.frames || [],
                config: { responsive: true }
            });
        } catch (e) {
            if (wasCancelled(e)) return;
            panel.innerHTML = `<div class="campbell-hint campbell-error">`
                + `${escapeHtml(t('modeShapeError'))}</div>`;
        }
    });
}
