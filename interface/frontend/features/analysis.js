// The analysis cards: the parameter catalogue of each analysis, assembling the
// panel, running the computation and restoring the saved cards.
//
// The fields of each analysis come from /api/schema/analyses. Until slice 4 this
// was 198 lines of `AnalysisDashboards` here, with the same names the runners
// already wrote on the other side -- two hand-kept lists, in two languages.
import { openCustomAlert, openCustomConfirm } from '../components/modals.js';
import { ANALYSES, analysesInScreenOrder, analysesToSave, recordResult, cardConversion, forgetAnalysis, forgetAllAnalyses, conversionName, registerAnalysis, conversionBadge, hasChart } from '../core/analysis_store.js';
import { runJob, wasCancelled, projectForServer } from '../core/api.js';
import { escapeHtml } from '../core/dom.js';
import { state } from '../core/state.js';
import { t } from '../core/i18n.js';
import { analysisFieldsFor, analysisTitle, analysisTitles, analysisUnsupported, schemaReady, unitAlternativesFor } from '../core/schema.js';
import { isModeShape, wireModeShapeClick, prepareModeShapePanels } from './campbell.js';
import { switchScreen } from './screens.js';
// Rebuilds the analysis charts on the screen from memory.
//
// Asynchronous since slice 4: the fields come from the server, and assembling the
// card before they arrive would give an empty form.
export async function restoreAnalysesFromMemory(savedArray) {
    await schemaReady();
    const container = document.getElementById('analysis-list');
    if (!container) return;
    
    savedArray.slice().reverse().forEach(an => {
        const uniqueId = Date.now() + Math.random().toString().slice(2,8);
        const nid = 'plot-' + uniqueId;
        const cardId = 'card-' + uniqueId;
        let controlsHTML = '';
        
        if(an.type && an.params) {
            const config = analysisFieldsFor(an.type);
            if(config) {
                config.forEach(item => { 
                    if(an.params[item.id] !== undefined) item.val = an.params[item.id]; 
                    if(an.params[item.id + '_unit'] !== undefined) item.saved_unit = an.params[item.id + '_unit'];
                });
                controlsHTML = buildDashboardHTML(uniqueId, an.type, config);
            }
        }                
        
        let typeVal = an.type || 'campbell';                
        container.insertAdjacentHTML('afterbegin', `
            <div class="analysis-card" id="${cardId}">
                <div class="analysis-header" onclick="toggleAnalysis('${uniqueId}')">
                    <span class="analysis-title">${escapeHtml(an.title)} ${conversionBadge(an.conversion)}</span>
                    <div class="analysis-actions">
                        <button class="btn-update-analysis" onclick="event.stopPropagation(); runCardAnalysis('${uniqueId}', '${typeVal}')"><i class="fas fa-sync-alt"></i> ${escapeHtml(t('update'))}</button>
                        <button class="btn-help-analysis" onclick="openAnalysisCardHelp(event, '${typeVal}')"><i class="fas fa-question-circle"></i> ${escapeHtml(t('help'))}</button>
                        <button class="btn-delete-analysis" onclick="deleteAnalysis(event, '${cardId}')"><i class="fas fa-trash"></i> ${escapeHtml(t('delete'))}</button>
                        <span id="icon-${uniqueId}"><i class="fas fa-chevron-down"></i></span>
                    </div>
                </div>
                <div class="analysis-body" id="body-${uniqueId}" style="padding:0; background:var(--bg-workspace); position: relative;">
                    <div id="${nid}" style="min-height: 400px; display: block; width: 100%; overflow: hidden; position:relative;"></div>
                    ${controlsHTML}
                </div>
            </div>
        `);                
        
        const divNode = document.getElementById(nid);
        registerAnalysis(uniqueId, an.type, an.title || 'Analysis', an.conversion);
        recordResult(uniqueId, an.params || {}, an.conversion,
                        { data: an.data, layout: an.layout, frames: an.frames });

        if (!hasChart(ANALYSES.get(uniqueId))) {
            inviteToRecompute(divNode, uniqueId, an.type);
            return;
        }

        an.layout.autosize = true; 
        const withMode = isModeShape(an.type, an.params);
        const target = withMode ? prepareModeShapePanels(divNode, nid) : nid;
        Plotly.newPlot(target, an.data, an.layout, {responsive: true});
        if (withMode) wireModeShapeClick(divNode, target, uniqueId);
    });
}


// Global probe builders

const generateProbeRowHTML = function(uniqueId, id, type, node=0, dof=0) {
    return `
    <div class="probe-row" style="align-items:center;">
        <span style="font-size:11px; color:var(--text-muted);">${escapeHtml(t('probeNode'))}</span> 
        <input type="number" class="probe-node" value="${node}" min="0">
        <span style="font-size:11px; color:var(--text-muted); margin-left:8px;">${escapeHtml(t('probeDof'))}</span> 
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

export const addProbeRow = function(uniqueId, id, type) {
    const container = document.getElementById(`probe-container-${id}-${uniqueId}`);
    container.insertAdjacentHTML('beforeend', generateProbeRowHTML(uniqueId, id, type));
}

// Force generators

const generateForceRowHTML = function(uniqueId, id, type, node=0, dof=0, func="1000 * np.cos(speed * t)") {
    return `
    <div class="probe-row" style="flex-direction:column; align-items:stretch; gap:8px;">
        <div style="display:flex; gap:6px; align-items:center;">
            <span style="font-size:11px; color:var(--text-muted);">${escapeHtml(t('probeNode'))}</span> 
            <input type="number" class="force-node" value="${node}" min="0">
            <span style="font-size:11px; color:var(--text-muted); margin-left:8px;">${escapeHtml(t('probeDof'))}</span> 
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
            <span style="font-size:11px; color:var(--text-muted);">${escapeHtml(t('probeForce'))}</span> 
            <input type="text" class="force-func" value="${func}">
        </div>
    </div>`;
}

// Function to add a force

export const addForceRow = function(uniqueId, id, type) {
    const container = document.getElementById(`force-container-${id}-${uniqueId}`);
    container.insertAdjacentHTML('beforeend', generateForceRowHTML(uniqueId, id, type));
}

// Unbalance generators

const generateUnbalanceRowHTML = function(uniqueId, id, type, node=0, mag=0.01, phase=0) {
    return `
    <div class="probe-row" style="flex-direction:column; align-items:stretch; gap:8px;">
        <div style="display:flex; gap:6px; align-items:center;">
            <span style="font-size:11px; color:var(--text-muted);">${escapeHtml(t('probeNode'))}</span> 
            <input type="number" class="unb-node" value="${node}" min="0">
            <button type="button" class="btn-remove-probe" style="margin-left:auto;" onclick="this.parentElement.parentElement.remove();"><i class="fas fa-times"></i></button>
        </div>
        <div style="display:flex; gap:6px; align-items:center;">
            <span style="font-size:11px; color:var(--text-muted);">${escapeHtml(t('probeMag'))}</span> 
            <input type="number" class="unb-mag" value="${mag}" step="0.001" min="0">
            <span style="font-size:11px; color:var(--text-muted); margin-left:8px;">${escapeHtml(t('probePhase'))}</span> 
            <input type="number" class="unb-phase" value="${phase}" step="0.01">
        </div>
    </div>`;
}

// Function to add a unbalance

export const addUnbalanceRow = function(uniqueId, id, type) {
    const container = document.getElementById(`unb-container-${id}-${uniqueId}`);
    container.insertAdjacentHTML('beforeend', generateUnbalanceRowHTML(uniqueId, id, type));
}

// Angle Probe generators (Node + Angle)
const generateAngleProbeRowHTML = function(uniqueId, id, type, node=0, angle=0) {
    return `
    <div class="probe-row" style="align-items:center;">
        <span style="font-size:11px; color:var(--text-muted);">${escapeHtml(t('probeNode'))}</span> 
        <input type="number" class="probe-node" value="${node}" min="0">
        <span style="font-size:11px; color:var(--text-muted); margin-left:8px;">${escapeHtml(t('probeAngle'))}</span> 
        <input type="number" class="probe-angle" value="${angle}" step="0.01">
        <button type="button" class="btn-remove-probe" style="margin-left:auto;" onclick="this.parentElement.remove();"><i class="fas fa-times"></i></button>
    </div>`;
}

// Function to add a probe

export const addAngleProbeRow = function(uniqueId, id, type) {
    const container = document.getElementById(`angle-probe-container-${id}-${uniqueId}`);
    container.insertAdjacentHTML('beforeend', generateAngleProbeRowHTML(uniqueId, id, type));
}

export const toggleDashAdv = function(btn) {
    const container = btn.nextElementSibling;
    if (container.style.display === 'grid') {
        container.style.display = 'none';
        btn.innerHTML = btn.dataset.textOriginal + ' <i class="fas fa-chevron-down"></i>';
    } else {
        container.style.display = 'grid';
        btn.innerHTML = 'Hide ' + btn.dataset.textOriginal + ' <i class="fas fa-chevron-up"></i>';
    }
};

// Shows or hides the fields whose visibility depends on another field.
//
// Before (FE-07) this read **every** select on the card and showed the field if
// any of them held one of the allowed values. It worked because no pair of
// selects in the same analysis shared an option -- a coincidence, not a
// guarantee. Today the catalogue says which field each dependency comes from, and
// a test refuses the ambiguity that would make this wrong.
export const checkDeps = function(uniqueId) {
    document.querySelectorAll(`.dash-dep-${uniqueId}`).forEach(item => {
        const allowed = (item.dataset.deps || '').split(',').map(v => v.trim());
        const selector = document.getElementById(`input-${item.dataset.depsDe}-${uniqueId}`);
        const value = selector ? selector.value : null;
        item.style.display = allowed.includes(value) ? 'flex' : 'none';
    });
};

// configOverride: a copy with the values of this instance. Without it the default
// definition is used -- which must NEVER be written to, or every new card
// inherits old values.
export function buildDashboardHTML(uniqueId, type, configOverride) {
    const config = configOverride || analysisFieldsFor(type);
    if(!config) return '';
    
    let htmlStandard = '';
    let htmlAdvAnalysis = '';
    let htmlAdvPlot = '';
    
    config.forEach(item => {
        let html = '';
        // `data-deps-de` says **which** field this visibility depends on. Without it
        // `checkDeps` compared the allowed values against the value of every select on
        // the card (FE-07): it took only another select having the same option -- "1D",
        // "flex" -- for the field to appear at the wrong moment.
        const depsAttr = item.deps
            ? `data-deps="${item.deps.join(',')}" data-deps-de="${item.deps_de}" `
              + `class="dash-control-group dash-dep-${uniqueId}"`
            : `class="dash-control-group"`;
        
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
            
            if (item.type === 'probe_list') item.val.forEach(v => { html += generateProbeRowHTML(uniqueId, item.id, type, v.node, v.dof); });
            else if (item.type === 'force_list') item.val.forEach(v => { html += generateForceRowHTML(uniqueId, item.id, type, v.node, v.dof, v.func); });
            else if (item.type === 'angle_probe_list') item.val.forEach(v => { html += generateAngleProbeRowHTML(uniqueId, item.id, type, v.node, v.angle); });
            else item.val.forEach(v => { html += generateUnbalanceRowHTML(uniqueId, item.id, type, v.node, v.mag, v.phase); });
            
            html += `</div></div>`;
        } else {
            html += `<div ${depsAttr} style="flex-direction: column; align-items: stretch; gap: 6px;">`;
            html += `<label style="margin: 0; min-width: auto;">${item.label}</label>`;
            
            let activeUnit = item.saved_unit || item.default_unit;

            if (item.type === 'range') {
                html += `<div style="display: flex; align-items: center; gap: 10px; width: 100%;">`;                
                html += `<input type="range" id="range-${item.id}-${uniqueId}" min="${item.min}" max="${item.max}" step="${item.step}" value="${item.val}" oninput="document.getElementById('num-${item.id}-${uniqueId}').value = this.value;" style="flex: 1; margin: 0;">`;
                html += `<div class="unified-input" style="width: 140px; flex-shrink: 0;">`;
                html += `<input type="number" id="num-${item.id}-${uniqueId}" value="${item.val}" oninput="document.getElementById('range-${item.id}-${uniqueId}').value = this.value;">`;
                
                const alternatives = unitAlternativesFor(item.default_unit);
                if (item.default_unit && alternatives) {
                    html += `<select id="unit-${item.id}-${uniqueId}" class="unified-unit" data-prev="${activeUnit}" onchange="handleUnitChange(this)">`;
                    let addedOpts = new Set();
                    alternatives.forEach(u => {
                        let sel = (u === activeUnit) ? 'selected' : '';
                        html += `<option value="${u}" ${sel}>${u}</option>`;
                        addedOpts.add(u);
                    });
                    if (activeUnit && !addedOpts.has(activeUnit)) html += `<option value="${activeUnit}" selected>${activeUnit}</option>`;
                    html += `<option value="Others">${escapeHtml(t('others'))}</option></select>`;
                }
                html += `</div></div>`;

            } else if (item.type === 'select') {
                let isUnit = item.id.includes('unit');
                let changeEvent = (item.id === 'plot_type' || item.id === 'coupling') ? `onchange="checkDeps('${uniqueId}')"` : ``;            
                
                if (isUnit) {
                    changeEvent = `onchange="handleUnitChange(this)" data-prev="${item.val}"`;
                }

                html += `<select id="input-${item.id}-${uniqueId}" style="width: 100%;" ${changeEvent}>`;
                let addedOpts = new Set();
                item.options.forEach(opt => { 
                    let sel = (opt === item.val) ? 'selected' : '';
                    html += `<option value="${opt}" ${sel}>${opt}</option>`; 
                    addedOpts.add(opt);
                });
                
                if (isUnit) {
                    if (!addedOpts.has(item.val)) html += `<option value="${item.val}" selected>${item.val}</option>`;
                    html += `<option value="Others">${escapeHtml(t('others'))}</option>`;
                }
                html += `</select>`;

            } else {
                html += `<div class="unified-input">`;
                html += `<input type="${item.type}" id="input-${item.id}-${uniqueId}" value="${item.val}">`;
                
                const alternatives = unitAlternativesFor(item.default_unit);
                if (item.default_unit && alternatives) {
                    html += `<select id="unit-${item.id}-${uniqueId}" class="unified-unit" data-prev="${activeUnit}" onchange="handleUnitChange(this)">`;
                    let addedOpts = new Set();
                    alternatives.forEach(u => {
                        let sel = (u === activeUnit) ? 'selected' : '';
                        html += `<option value="${u}" ${sel}>${u}</option>`;
                        addedOpts.add(u);
                    });
                    if (activeUnit && !addedOpts.has(activeUnit)) html += `<option value="${activeUnit}" selected>${activeUnit}</option>`;
                    html += `<option value="Others">${escapeHtml(t('others'))}</option></select>`;
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
    
    if (htmlAdvAnalysis) finalHtml += `<div style="grid-column: 1 / -1;"><button type="button" class="btn-adv-dash" data-text-original="${escapeHtml(t('advancedAnalysis'))}" onclick="toggleDashAdv(this)">${escapeHtml(t('advancedAnalysis'))} <i class="fas fa-chevron-down"></i></button><div class="adv-dash-container">${htmlAdvAnalysis}</div></div>`;
    if (htmlAdvPlot) finalHtml += `<div style="grid-column: 1 / -1;"><button type="button" class="btn-adv-dash" data-text-original="${escapeHtml(t('advancedPlot'))}" onclick="toggleDashAdv(this)">${escapeHtml(t('advancedPlot'))} <i class="fas fa-chevron-down"></i></button><div class="adv-dash-container" id="adv-plot-${uniqueId}">${htmlAdvPlot}</div></div>`;
    
    finalHtml += '</div>';
    setTimeout(() => { checkDeps(uniqueId); }, 100);
    return finalHtml;
}

// Function to hide the analysis

export function toggleAnalysis(uniqueId) {
    const body = document.getElementById(`body-${uniqueId}`);
    const icon = document.getElementById(`icon-${uniqueId}`);
    if (body.style.display === 'none') { body.style.display = 'block'; icon.innerHTML = '<i class="fas fa-chevron-down"></i>'; }
    else { body.style.display = 'none'; icon.innerHTML = '<i class="fas fa-chevron-right"></i>'; }
}

// Function to delete the analysis

export async function deleteAnalysis(event, cardId) {
    event.stopPropagation();
    let isConfirmed = await openCustomConfirm(t('confirmDeleteDashboard'));
    if (isConfirmed) {
        document.getElementById(cardId).remove();
        // The card is the drawing; the record is the analysis. Removing only one of the
        // two would leave the analysis alive in whatever got saved, or an orphan card.
        forgetAnalysis(cardId.replace(/^card-/, ''));
    }
}

// Fills the analysis-type <select> with the titles from the catalogue.
//
// The twelve names were written in index.html and again in a `typeNames` in this
// file. Now they come from the server, already in the chosen language -- and a
// new analysis shows up on screen without anyone editing HTML.
export function fillAnalysisTypes() {
    const selector = document.getElementById('analysis-type');
    if (!selector) return;
    const picked = selector.value;
    const titles = analysisTitles();

    selector.querySelectorAll('option[value]:not([value=""])').forEach(o => o.remove());
    Object.keys(titles).forEach(type => {
        const option = document.createElement('option');
        option.value = type;
        option.textContent = titles[type];
        selector.appendChild(option);
    });
    if (picked) selector.value = picked;
}

export async function addAnalysis(event) {
    if(event) event.preventDefault();
    await schemaReady();          // the fields come from /api/schema/analyses
    const type = document.getElementById('analysis-type').value;
    if(!type) return await openCustomAlert(t('pickAnalysisFirst'));
    
    const conversionNode = document.getElementById('rotor-conversion-type');
    const conversionType = conversionNode ? conversionNode.value : '';

    const badge = conversionBadge(conversionType);

    const uniqueId = Date.now() + Math.random().toString().slice(2,8);
    const plotId = 'plot-' + uniqueId;
    const cardId = 'card-' + uniqueId;
    const list = document.getElementById('analysis-list');
    if(list.innerHTML.includes('dashboards-empty')) list.innerHTML = '';
    const title = analysisTitle(type);
    registerAnalysis(uniqueId, type, title, conversionType);
    const controlsHTML = buildDashboardHTML(uniqueId, type);
    
    list.insertAdjacentHTML('afterbegin', `
        <div class="analysis-card" id="${cardId}">
            <div class="analysis-header" onclick="toggleAnalysis('${uniqueId}')">
                <span class="analysis-title"><i class="fas fa-chart-line"></i> ${escapeHtml(title)} ${badge}</span>
                <div class="analysis-actions">
                    <button class="btn-update-analysis" onclick="event.stopPropagation(); runCardAnalysis('${uniqueId}', '${type}')"><i class="fas fa-sync-alt"></i> ${escapeHtml(t('update'))}</button>
                    <button class="btn-help-analysis" onclick="openAnalysisCardHelp(event, '${type}')"><i class="fas fa-question-circle"></i> ${escapeHtml(t('help'))}</button>
                    <button class="btn-delete-analysis" onclick="deleteAnalysis(event, '${cardId}')"><i class="fas fa-trash"></i> ${escapeHtml(t('delete'))}</button>
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

// The values the user typed into a card, read from the DOM.
//
// This lived inside `runCardAnalysis`, the only place that needed it. The
// language change came to need it too: it redraws the cards to show the labels in
// the new language, and without reading the DOM first anything typed and not yet
// computed would fall back to the values of the last run.
export function cardParameters(uniqueId, type) {
    const config = analysisFieldsFor(type);
    const p = {};
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
    return p;
}


// The empty-list notice. It lives here, with whoever recognises it: two places
// clear the list when they find it and one rewrites it on a language change. The
// mark of recognition is the class, not the text -- comparing by the English text
// is what the translation broke.
export function emptyListNotice() {
    return `<p class="dashboards-empty" style="color: #888; text-align: center; margin-top: 20%;">`
         + `${escapeHtml(t('dashboardsWillAppear'))}</p>`;
}


// Redraws the already-open cards in the current language.
//
// The language change only reached HTML marked with `data-i18n` -- and analysis
// cards are assembled by the JS: labels, buttons and the title itself stayed in
// the language they were created in. The path is the same as the page reload
// (save, forget, restore), so a chart already computed comes back from `figure`
// with no new trip to the server. The title comes from the catalogue, not from
// what was written on the card, which is what actually translates it.
export async function redrawAnalyses() {
    const container = document.getElementById('analysis-list');
    if (!container) return;
    const records = analysesInScreenOrder();
    if (!records.length) {
        if (container.innerHTML.includes('dashboards-empty')) {
            container.innerHTML = emptyListNotice();
        }
        return;
    }

    const cards = records.map(record => ({
        title: analysisTitle(record.type) || record.title,
        type: record.type,
        params: parametersOnScreen(record),
        conversion: record.conversion,
        data: (record.figure && record.figure.data) || [],
        layout: (record.figure && record.figure.layout) || {},
        frames: (record.figure && record.figure.frames) || []
    }));

    forgetAllAnalyses();
    container.innerHTML = '';
    await restoreAnalysesFromMemory(cards);
}

// What is typed on the screen, or the values of the last run if the card has not
// drawn its fields yet.
function parametersOnScreen(record) {
    if (!record.type || !analysisFieldsFor(record.type)) return record.params;
    try {
        return cardParameters(record.id, record.type);
    } catch (e) {
        return record.params;
    }
}


// What the card says while it waits, and it is two different sentences.
//
// Before 6c-2 there was one: "updating", shown whether the analysis was being
// computed or was sitting behind another card's. Two cards saying that at the
// same time read as two analyses at once -- and there is one worker, so they
// were taking turns. `ahead` comes from the queue because it is the one thing a
// card cannot know about itself.
//
// The elapsed time is the server's `waiting` and not a timer started here: one
// clock, and the one that is also counting when the browser is busy drawing.
export function showProgress(uniqueId, progress) {
    const what = document.getElementById('loading-what-' + uniqueId);
    const when = document.getElementById('loading-when-' + uniqueId);
    if (!progress) return;
    if (what) {
        const ahead = progress.ahead || 0;
        what.textContent = progress.state === 'running' || !ahead
            ? t('jobComputing')
            : `${t('jobInLine')} \u2014 ${ahead} ${t('jobAhead')}`;
    }
    if (when) {
        when.textContent = progress.waiting ? `${Math.round(progress.waiting)} s` : '';
    }
}


export async function runCardAnalysis(uniqueId, type) {
    await schemaReady();
    const plotId = 'plot-' + uniqueId;
    const div = document.getElementById(plotId);
    if(!div) return;
    const p = cardParameters(uniqueId, type);
    div.style.opacity = '0.4';
    const loadingIndicatorId = `loading-${uniqueId}`;
    let loader = document.getElementById(loadingIndicatorId);
    if(!loader) {
        loader = document.createElement('div'); 
        loader.id = loadingIndicatorId;        
        loader.innerHTML = `
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 8px;">
                <i class="fas fa-sync fa-spin fa-2x" style="color: var(--accent-primary);"></i>
                <span id="loading-what-${uniqueId}" style="font-weight: 600; font-size: 13px;">${escapeHtml(t('updating'))}</span>
                <span id="loading-when-${uniqueId}" class="job-elapsed"></span>
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
    // Not every ROSS analysis accepts the converted rotor. The warning comes before
    // the computation, and states the reason -- before, what arrived here was the raw
    // error from the library, or worse: a chart badged "4 DoF" carrying the numbers
    // of the full model, because the conversion swaps the matrix methods and an
    // analysis that builds its own does not notice.
    const conversionType = cardConversion(uniqueId);
    const impediment = analysisUnsupported(type, conversionType);
    if (impediment) {
        div.style.opacity = '1';
        if (loader) loader.style.display = 'none';
        warnUnsupported(div, conversionType, impediment.reason);
        return;
    }

    const payload = {
        analysis_type: type,
        params: p,
        conversion_type: conversionType,
        project: projectForServer(state.projectData)
    };
    try {
        const data = await runJob('card:' + uniqueId, '/run_analysis', payload,
            progress => showProgress(uniqueId, progress));
        // The server dropped this job because the same card asked for something
        // else: the card belongs to that newer request now, and touching it here
        // would undo what it has already drawn.
        if (data && data.status === 'superseded') return;
        div.style.opacity = '1';
        if (loader) loader.style.display = 'none';
        // The user pressed the button and was told what it would cost. A card
        // that simply went back to how it looked before would leave them with no
        // evidence the button did anything at all.
        if (data && data.status === 'cancelled') {
            div.innerHTML = `<div class="analysis-no-chart"><i class="fas fa-ban"></i>`
                + `<p>${escapeHtml(t('workStopped'))}</p></div>`;
            return;
        }

        if(data.status === "success") {
            div.innerHTML = ""; 
            const fig = JSON.parse(data.plot_json);
            fig.layout.autosize = true;
            recordResult(uniqueId, p, conversionType, fig);

            // In mode-shape mode the diagram shares the card with the 3D mode panel.
            const withModeShape = isModeShape(type, p);
            const target = withModeShape ? prepareModeShapePanels(div, plotId) : plotId;

            Plotly.newPlot(target, {
                data: fig.data, 
                layout: fig.layout, 
                frames: fig.frames || [], 
                config: {responsive: true}
            });

            if (withModeShape) wireModeShapeClick(div, target, uniqueId);
            
        } else {
            div.innerHTML = `<div style="color:red; text-align:center; padding: 20px;"><i class="fas fa-exclamation-triangle fa-2x"></i><br><b>${escapeHtml(t('errorLabel'))}</b> ${data.message}</div>`;
        }
    } catch(e) {
        if (wasCancelled(e)) return;   // a newer request has taken over this card
        div.style.opacity = '1'; if(loader) loader.style.display = 'none';
        div.innerHTML = `<p style="color:red; text-align:center;">${escapeHtml(t('serverConnectionError'))}</p>`; 
    }
}

// The card for a combination ROSS does not support. It is not the user's mistake
// nor the interface's: it is a limit of the library, and the card says which.
function warnUnsupported(card, conversion, reason) {
    card.innerHTML =
        `<div class="analysis-no-chart analysis-unsupported">`
        + `<i class="fas fa-triangle-exclamation"></i>`
        + `<p><strong>${escapeHtml(t('unsupportedTitle'))}</strong></p>`
        + `<p>${escapeHtml(reason)}</p>`
        + `<p class="analysis-hint">${escapeHtml(t('unsupportedHint'))} `
        + `${escapeHtml(conversionName(conversion))}</p>`
        + `</div>`;
}

// A card with no chart -- restored from another session, or from an analysis that
// failed -- shows the invitation to recompute. It used to call Plotly with
// `data: []` and the result was an empty box, which looks like a defect rather
// than a choice. Charts are deliberately not saved: yesterday's chart, from an
// earlier version of the rotor, would look current.
function inviteToRecompute(card, uniqueId, type) {
    card.innerHTML =
        `<div class="analysis-no-chart">`
        + `<i class="fas fa-rotate"></i>`
        + `<p>${escapeHtml(t('chartNotKept'))}</p>`
        + `<button class="btn-analysis-go" type="button" `
        + `onclick="runCardAnalysis('${uniqueId}', '${type}')">`
        + `${escapeHtml(t('recalculate'))}</button>`
        + `</div>`;
}

// Function to load the analysis

export function loadAnalysis(event) {
    const file = event.target.files[0]; if (!file) return;
    const reader = new FileReader();
    reader.onload = async e => {
        try {
            await schemaReady();
            const loaded = JSON.parse(e.target.result);
            const container = document.getElementById('analysis-list');
            if(container.innerHTML.includes('dashboards-empty')) container.innerHTML = '';            
            loaded.reverse().forEach(an => {
                const uniqueId = Date.now() + Math.random().toString().slice(2,8);
                const nid = 'plot-' + uniqueId;
                const cardId = 'card-' + uniqueId;
                let controlsHTML = '';
                if(an.type && an.params) {
                    const config = analysisFieldsFor(an.type);
                    if(config) {
                        config.forEach(item => { 
                            if(an.params[item.id] !== undefined) item.val = an.params[item.id]; 
                            if(an.params[item.id + '_unit'] !== undefined) item.saved_unit = an.params[item.id + '_unit'];
                        });
                        controlsHTML = buildDashboardHTML(uniqueId, an.type, config);
                    }
                }                
                let typeVal = an.type || 'campbell';                
                container.insertAdjacentHTML('afterbegin', `
                    <div class="analysis-card" id="${cardId}">
                        <div class="analysis-header" onclick="toggleAnalysis('${uniqueId}')">
                            <span class="analysis-title">${escapeHtml(an.title)} ${escapeHtml(t('loadedSuffix'))} ${conversionBadge(an.conversion)}</span>
                            <div class="analysis-actions">
                                <button class="btn-update-analysis" onclick="event.stopPropagation(); runCardAnalysis('${uniqueId}', '${typeVal}')"><i class="fas fa-sync-alt"></i> ${escapeHtml(t('update'))}</button>
                                <button class="btn-help-analysis" onclick="openAnalysisCardHelp(event, '${typeVal}')"><i class="fas fa-question-circle"></i> ${escapeHtml(t('help'))}</button>
                                <button class="btn-delete-analysis" onclick="deleteAnalysis(event, '${cardId}')"><i class="fas fa-trash"></i> ${escapeHtml(t('delete'))}</button>
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
                registerAnalysis(uniqueId, an.type, an.title || 'Analysis', an.conversion);
                recordResult(uniqueId, an.params || {}, an.conversion,
                                { data: an.data, layout: an.layout, frames: an.frames });

                if (!hasChart(ANALYSES.get(uniqueId))) {
                    inviteToRecompute(divNode, uniqueId, an.type);
                    return;
                }

                an.layout.autosize = true; 
                const withMode = isModeShape(an.type, an.params);
                const target = withMode ? prepareModeShapePanels(divNode, nid) : nid;
                Plotly.newPlot(target, {
                    data: an.data, 
                    layout: an.layout, 
                    frames: an.frames || [], 
                    config: {responsive: true}
                }).then(() => {
                    if (withMode) wireModeShapeClick(divNode, target, uniqueId);
                    window.dispatchEvent(new Event('resize'));
                });
            });
        } catch(err) { await openCustomAlert(t('badJson')); }
    };
    reader.readAsText(file); event.target.value = '';
}

// Function to load the analysis directly

export function loadAnalysisDirect(event) { switchScreen('screen-analysis'); loadAnalysis(event); }

// Function to save the analysis

export async function saveAnalysis(event) {
    if (event) event.preventDefault();
    const saved = analysesToSave();
    if(saved.length===0) return await openCustomAlert(t('nothingToSave'));
    const blob = new Blob([JSON.stringify(saved)], {type: "application/json"});
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = "analyses.json";
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
}
