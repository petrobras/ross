// The modeling screen: the element tabs, the form, the rotor figure that redraws
// itself on every change, and the node hub over the figure.
import { buildFormHTML, capturedFormValues, restoreFormValues, toggleAdvanced } from '../components/form.js';
import { getEffectiveNodes, positionFormBox, renderList } from '../components/list.js';
import { reapplyHelp } from '../components/help.js';
import { openCustomAlert } from '../components/modals.js';
import { apiFetch, apiFetchLatest, wasCancelled, projectForServer } from '../core/api.js';
import { escapeHtml } from '../core/dom.js';
import { state, getActiveData, syncBackToLibrary } from '../core/state.js';
import { applyLanguage, rememberLanguage, t } from '../core/i18n.js';
import { formSubtypes, loadElementSchema, schemaReady } from '../core/schema.js';
import { fillAnalysisTypes, redrawAnalyses } from './analysis.js';
import { openRotorHub, renderRotorHub } from './hub.js';

// How the rotor figure settles into the panel. This used to be done in the
// backend (BE-12): margin, background, size and legend position assembled on top
// of the Plotly JSON before sending. What knows the size of the panel is the
// screen.
const ROTOR_APPEARANCE = {
    autosize: true,
    width: null,
    height: null,
    margin: { l: 40, r: 40, t: 80, b: 80 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    legend: { orientation: 'h', yanchor: 'bottom', y: 1.05, xanchor: 'center', x: 0.5 },
};

const ROTOR_MENU = { y: -0.15, yanchor: 'top', x: 1.0, xanchor: 'right' };

// --- State of the screen itself ----------------------------------------------
//
// Four values only this screen reads and writes, which used to hang off
// `window`: any script on the page could touch them, and nothing said whose they
// were. Here they are module-level `let` -- unreachable from outside, and with
// the scope declaring the owner.
//
// `nodeMap` is the node -> z position table, rebuilt on every rotor figure;
// `addingFromNodeHub` marks that the form was opened from the node hub (which
// hides the LIST option); `nodeHubTarget` is the node the hub picked, and
// `hiddenNode` the one the form keeps until it is saved.
let nodeMap = [];
let addingFromNodeHub = false;
let nodeHubTarget = null;
let hiddenNode = null;

// Switches the language of the whole interface.
//
// The order matters and has bitten before: translating the page comes **before**
// the `if (!state.currentTab) return`. With the early return in its old place,
// changing the language with no element tab open loaded the new schema and left
// the screen in English.
export async function changeLanguage(language) {
    rememberLanguage(language);

    // An open form keeps what was typed: it is redrawn with the new labels, not
    // reopened from scratch.
    const formBox = document.getElementById('insertion-form');
    const wasOpen = !!formBox && formBox.style.display !== 'none';
    const editedIndex = state.editingIndex;
    const formValues = wasOpen ? capturedFormValues() : null;

    await loadElementSchema(language);

    applyLanguage();
    fillAnalysisTypes();

    // `applyLanguage()` only reaches the marked HTML of `index.html`. The Hub and the
    // analysis cards are assembled by the JS: without redrawing, they stayed in the
    // language they were created in and the screen came out half in each.
    renderRotorHub();
    reapplyHelp();
    await redrawAnalyses();

    if (!state.currentTab) return;
    refreshTabTitle();
    renderList();          // rescues the form before clearing the list
    if (!wasOpen) return;

    state.editingIndex = editedIndex;
    selectSubType(state.currentSubType);
    restoreFormValues(formValues);
    positionFormBox(editedIndex);
    document.getElementById('btn-add-item').style.display = 'none';
}

// The translated name of a category comes from the sidebar button that opens it:
// that button is the one carrying the `data-i18n`. A second `category -> key`
// table here could only diverge from the one already in `index.html`.
export function categoryName(category, button) {
    const target = button || Array.from(document.querySelectorAll('.tab-btn')).find(
        b => (b.getAttribute('onclick') || '').includes(`openTab('${category}')`));
    const key = target && target.dataset && target.dataset.i18n;
    return key ? t(key) : category;
}

function tabTitle(category, button) {
    const name = categoryName(category, button);
    return `<div style="display:flex; align-items:center;">
        <span>${escapeHtml(name)}</span> 
        <button class="btn-help-section" onclick="openSectionHelp('${category}')" title="${escapeHtml(t('helpAbout'))} ${escapeHtml(name)}"><i class="fas fa-question-circle"></i></button>
    </div>`;
}

// Rewrites the heading of the open tab in the current language. It exists apart
// from `openTab` because a language change must not reopen the tab: `openTab`
// closes the form, and whatever was being edited would be lost.
export function refreshTabTitle() {
    if (!state.currentTab) return;
    const heading = document.getElementById('tab-title');
    if (heading) heading.innerHTML = tabTitle(state.currentTab, null);
}

export function openTab(category) {
    state.currentTab = category;
    document.getElementById('empty-message').style.display = 'none';
    document.getElementById('list-area').style.display = 'block';
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    
    const activeBtn = Array.from(document.querySelectorAll('.tab-btn'))
        .find(b => b.getAttribute('onclick') && b.getAttribute('onclick').includes(`openTab('${category}')`));
    if (activeBtn) {
        activeBtn.classList.add('active');
    }
    
    let titleHTML = tabTitle(category, activeBtn);

    if (state.projectData.isMultiRotor) {
        let drvSel = (state.multiRotorEditTarget === 'driving') ? 'selected' : '';
        let drvnSel = (state.multiRotorEditTarget === 'driven') ? 'selected' : '';
        titleHTML += `
            <select id="mr-edit-target" onchange="switchMultiRotorTarget(this.value)" style="padding: 2px 6px; font-size:11px; font-weight: bold; border-radius: 4px; background: #e2e8f0; color: #334155; border: 1px solid #cbd5e1; outline: none; cursor: pointer; max-width: 160px; overflow: hidden; text-overflow: ellipsis;">
                <option value="driving" ${drvSel}>${escapeHtml(t('multiDriving'))}: ${escapeHtml(state.projectData.driving_rotor.name)}</option>
                <option value="driven" ${drvnSel}>${escapeHtml(t('multiDriven'))}: ${escapeHtml(state.projectData.driven_rotor.name)}</option>
            </select>
        `;
    }
    
    document.getElementById('tab-title').innerHTML = titleHTML;
    
    closeForm();
    renderList();    
    if (window.innerWidth <= 768) {
        const sidebar = document.querySelector('.sidebar');
        if (!sidebar.classList.contains('collapsed')) {
            sidebar.classList.add('collapsed');
            setTimeout(() => window.dispatchEvent(new Event('resize')), 300);
        }
    }
}

// Function to open the form

export async function openForm(isNew = true) {
    await schemaReady();          // the forms come from /api/schema/elements
    if (isNew) { state.editingIndex = -1; state.currentSubType = 'BASIC'; }
    let subTypes = formSubtypes(state.currentTab);
    
    if (addingFromNodeHub) {
        subTypes = subTypes.filter(type => type !== 'LIST');
    }
    
    if (isNew && subTypes.length > 1) {
        let html = `<h4 class="subtype-header">${escapeHtml(t('selectModel'))}</h4>`
                 + '<div class="subtype-grid">';
        subTypes.forEach(type => { html += `<button class="btn-subtype" onclick="selectSubType('${type}')">${type}</button>`; });
        html += '</div><button class="btn-cancel" style="width:100%; margin-top:15px;" onclick="closeForm()">' + escapeHtml(t('cancel')) + '</button>';
        document.getElementById('form-fields').innerHTML = html;
        document.querySelector('.form-actions').style.display = 'none';
    } else {
        const activeData = getActiveData();
        selectSubType(isNew ? 'BASIC' : activeData[state.currentTab][state.editingIndex].element_type || 'BASIC');
    }    
    
    document.getElementById('btn-add-item').style.display = 'none';    
    const formBox = document.getElementById('insertion-form');
    formBox.style.display = 'block';    
    
    positionFormBox(isNew ? -1 : state.editingIndex);
    if(!document.getElementById('btn-default-form')) {
        document.querySelector('.form-actions').insertAdjacentHTML('afterbegin', `<button type="button" id="btn-default-form" class="btn-default" onclick="fillDefault()"><i class="fas fa-magic"></i> ${escapeHtml(t('defaultButton'))}</button>`);
    }

    addingFromNodeHub = false; 
}

// Function for the 'Advanced' button

export function selectSubType(type) {
    state.currentSubType = type;
    document.getElementById('form-fields').innerHTML = buildFormHTML(state.currentTab, type);
    document.querySelector('.form-actions').style.display = 'flex';
    
    const activeData = getActiveData();
    
    const matSelects = document.getElementById('form-fields').querySelectorAll('select#inp-material');
    matSelects.forEach(sel => {
        sel.innerHTML = `<option value="Default (Steel)">${escapeHtml(t('defaultSteel'))}</option>`;
        
        activeData.materials.forEach(m => {
            let mName = m.name || 'MaterialCustom';
            sel.innerHTML += `<option value="${mName}">${mName}</option>`;
        });
    });
    
    if (state.editingIndex >= 0) {
        const item = activeData[state.currentTab][state.editingIndex];
        const inputs = document.getElementById('form-fields').querySelectorAll('input, select');
        let hasAdvancedVal = false;
        inputs.forEach(inp => {
            const key = inp.id.replace('inp-', '');
            if (item[key] !== undefined && item[key] !== null && item[key] !== '') {
                
                if (inp.tagName === 'SELECT') {
                    let exists = Array.from(inp.options).some(o => o.value === item[key]);
                    if (!exists) {
                        let newOpt = document.createElement('option');
                        newOpt.value = item[key];
                        newOpt.innerText = item[key];
                        let othersOpt = inp.querySelector('option[value="Others"]');
                        if (othersOpt) inp.insertBefore(newOpt, othersOpt);
                        else inp.appendChild(newOpt);
                    }
                }
                
                inp.value = item[key];
                if (inp.closest('.advanced-fields')) hasAdvancedVal = true;
            }
        });
        if (hasAdvancedVal) {
            const advBtn = document.getElementById('form-fields').querySelector('.btn-advanced');
            if(advBtn) toggleAdvanced(advBtn);
        }
    }

    if (nodeHubTarget !== undefined && nodeHubTarget !== null) {
        let nInput = document.getElementById('inp-n');
        
        if (state.currentTab !== 'shafts') {
            if (nInput) nInput.value = nodeHubTarget;
        } else {
            hiddenNode = parseInt(nodeHubTarget);
        }
        
        nodeHubTarget = null;
    }
}

// Function to close the form

export function closeForm() { 
    const formBox = document.getElementById('insertion-form');
    formBox.style.display = 'none'; 
    document.getElementById('list-area').appendChild(formBox);
    document.getElementById('btn-add-item').style.display = 'block'; 
    state.editingIndex = -1; 
    hiddenNode = null;
}

// Element editing function

export function editItem(index) { state.editingIndex = index; openForm(false); }

// Copy function for element

export function copyItem(index) { 
    const activeData = getActiveData();
    const original = activeData[state.currentTab][index];
    const copiedItem = JSON.parse(JSON.stringify(original));    
    
    if (copiedItem.tag) {
        let baseTag = copiedItem.tag.replace(/_\d+$/, '');
        let counter = 1;
        let newTag = `${baseTag}_${counter}`;
        const tagInUse = (cTag) => activeData[state.currentTab].some(item => item.tag === cTag);
        while (tagInUse(newTag)) {
            counter++;
            newTag = `${baseTag}_${counter}`;
        }
        copiedItem.tag = newTag;
    }
    
    activeData[state.currentTab].splice(index + 1, 0, copiedItem); 
    syncBackToLibrary();
    renderList(); 
    buildRotorLive(); 
}

// Delete function for the element

export function deleteItem(index) {
    const activeData = getActiveData();
    activeData[state.currentTab].splice(index, 1);    
    if (state.editingIndex === index) {
        closeForm();
    } 
    else if (state.editingIndex > index) {
        state.editingIndex--;
    }    
    syncBackToLibrary();
    renderList();
    buildRotorLive();
}

// Function to save the element

export function saveItem() {
    const activeData = getActiveData();
    const inputs = document.getElementById('form-fields').querySelectorAll('input, select');    
    if (state.currentSubType === 'LIST') {
        let parsedData = {};
        let maxLen = 0;        
        inputs.forEach(inp => {
            const key = inp.id.replace('inp-', '');
            let value = inp.value.trim();
            if (value !== '') {
                let arr = value.split(/,(?![^\[]*\])/).map(v => v.trim());
                parsedData[key] = arr;
                if (arr.length > maxLen) maxLen = arr.length;
            }
        });        
        if (maxLen === 0) {
            closeForm();
            return;
        }        
        for (let i = 0; i < maxLen; i++) {
            let newObj = { element_type: 'BASIC' };
            for (let key in parsedData) {
                let arr = parsedData[key];
                newObj[key] = arr[i] !== undefined ? arr[i] : arr[arr.length - 1];
            }
            if (newObj.tag) {
                newObj.tag = newObj.tag + "_" + (i + 1);
            }
            activeData[state.currentTab].push(newObj);
        }        
    } else {
        let newObject = { element_type: state.currentSubType };        
        inputs.forEach(inp => {
            const key = inp.id.replace('inp-', '');
            let value = inp.value.trim();
            if (value !== '') {
                newObject[key] = value; 
            }
        });
        
        if (state.editingIndex >= 0) {
            activeData[state.currentTab][state.editingIndex] = newObject;
        } else {
            let targetN = null;
            if (newObject.n !== undefined && newObject.n !== "") {
                targetN = parseInt(newObject.n);
            } else if (state.currentTab === 'shafts' && hiddenNode !== undefined && hiddenNode !== null) {
                targetN = hiddenNode;
            }

            if (targetN !== null) {
                let insertIdx = activeData[state.currentTab].length;
                
                const effNodes = getEffectiveNodes(activeData[state.currentTab]);
                for (let i = 0; i < effNodes.length; i++) {
                    if (effNodes[i] >= targetN) {
                        insertIdx = i;
                        break;
                    }
                }
                activeData[state.currentTab].splice(insertIdx, 0, newObject);
            } else {
                activeData[state.currentTab].push(newObject);
            }
        }    
    }
    
    syncBackToLibrary();
    closeForm();
    renderList();
    buildRotorLive();
}

let rotorUpdateActive = false;

let debounceTimer = null;

// Real-time rotor construction functions

export function buildRotorLive() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
        _fetchRotorLive();
    }, 600); 
}

async function _fetchRotorLive() {
    const plotContainer = document.getElementById('plot-rotor');
    const infoContainer = document.getElementById('rotor-info');    
    if (!state.projectData.isMultiRotor && (!state.projectData.shafts || state.projectData.shafts.length === 0)) {
        plotContainer.innerHTML =
            `<div style="display: flex; height: 100%; min-height: 400px; align-items: center; justify-content: center;">`
            + `<p style="color: #888; text-align: center; margin: 0;">${escapeHtml(t('addOneShaft'))}</p></div>`;
        if(infoContainer) infoContainer.style.opacity = '0';
        return;
    }    
    rotorUpdateActive = true;
    plotContainer.style.opacity = '0.4';
    plotContainer.style.pointerEvents = 'none';
    let loadingTimer = setTimeout(() => {
        if(rotorUpdateActive) {
            plotContainer.style.opacity = '1';
            plotContainer.innerHTML = `
                <div style="display:flex; flex-direction:column; justify-content:center; align-items:center; height:100%; min-height:400px; color: var(--text-main);">
                    <i class="fas fa-microchip fa-spin fa-3x" style="margin-bottom:15px; color: var(--accent-primary);"></i>
                    <h3 style="margin:0;">${escapeHtml(t('computingElement'))}</h3>
                    <p style="color: var(--text-muted); text-align:center; padding:0 20px;">${escapeHtml(t('usingCache'))}</p>
                </div>`;
        }
    }, 500);
    try {
        const response = await apiFetchLatest('rotor', '/build_rotor', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ project: projectForServer(state.projectData) })
        });
        const data = await response.json();        
        rotorUpdateActive = false;
        clearTimeout(loadingTimer);
        plotContainer.style.opacity = '1';
        plotContainer.style.pointerEvents = 'auto';        
        if(data.status === "success") {
            plotContainer.innerHTML = ""; 
            const fig = JSON.parse(data.plot_json);
            const layout = Object.assign(fig.layout, ROTOR_APPEARANCE);
            // Before drawing: positioning the menu after `newPlot` would not touch the
            // figure already rendered.
            (layout.updatemenus || []).forEach(menu => Object.assign(menu, ROTOR_MENU));
            Plotly.newPlot('plot-rotor', fig.data, layout, { responsive: true });
            setupPlotHoverEvents();
            if(infoContainer) {
                document.getElementById('info-mass').innerText = data.mass.toFixed(4);
                document.getElementById('info-ip').innerText = data.ip.toFixed(4);
                infoContainer.style.opacity = '1';
            }
        } else {
            plotContainer.innerHTML = `<div style="padding:20px; color:var(--accent-danger); text-align:center;"><i class="fas fa-exclamation-triangle fa-2x"></i><br><b>${escapeHtml(t('modelingError'))}</b><br>${data.message}</div>`;
            if(infoContainer) infoContainer.style.opacity = '0';
        }
    } catch (e) { 
        if (wasCancelled(e)) return;   // a newer request has taken over
        rotorUpdateActive = false; 
        clearTimeout(loadingTimer); 
        plotContainer.style.opacity = '1';
        plotContainer.innerHTML = `<p style="color:var(--accent-danger); text-align:center; margin-top:50%;">`
            + `${escapeHtml(t('serverConnectionError'))}</p>`; 
        if(infoContainer) infoContainer.style.opacity = '0';
    }
}

// Function to save the active rotor

export function saveRotor(event) {
    if (event) event.preventDefault();
    const fileName = (state.projectData.name || "my_rotor").replace(/\s+/g, '_') + ".json";
    const blob = new Blob([JSON.stringify(state.projectData, null, 2)], {type: "application/json"});
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = fileName;
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
}

// Function to load the rotor

export async function loadRotor(event) {
    const file = event.target.files[0]; 
    if (!file) return;    
    const reader = new FileReader();    
    reader.onload = async function(e) {
        const content = e.target.result;
        let isInterfaceJSON = false;
        
        try {
            const loaded = JSON.parse(content);
            if (loaded.shafts !== undefined || loaded.eixos !== undefined) {
                isInterfaceJSON = true;
                
                let newLoadedRotor = {
                    name: file.name.replace('.json', ''),
                    savedAnalyses: loaded.savedAnalyses || [],
                    materials: loaded.materials || loaded.materiais || [],
                    shafts: loaded.shafts || loaded.eixos || [],
                    disks: loaded.disks || loaded.discos || [],
                    gears: loaded.gears || loaded.engrenagens || [],
                    couplings: loaded.couplings || loaded.acoplamentos || [],
                    seals: loaded.seals || loaded.badges || [],
                    bearings: loaded.bearings || loaded.mancais || [],
                    pointmasses: loaded.pointmasses || []
                };

                state.rotorLibrary.push(newLoadedRotor);
                openRotorHub();
                await openCustomAlert(t('rotorLoaded'));
            }
        } catch (err) { }
        
        if (!isInterfaceJSON) {
            try {
                const response = await apiFetch('/load_ross_file', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ content: content })
                });
                const data = await response.json();
                if (data.status === 'success') {
                    let newConvertedRotor = {
                        name: file.name.replace('.json', '') + " (Converted)",
                        savedAnalyses: [],
                        materials: data.projectData.materials || [],
                        shafts: data.projectData.shafts || [],
                        disks: data.projectData.disks || [],
                        gears: data.projectData.gears || [],
                        couplings: data.projectData.couplings || [],
                        seals: data.projectData.seals || [],
                        bearings: data.projectData.bearings || [],
                        pointmasses: data.projectData.pointmasses || []
                    };
                    
                    state.rotorLibrary.push(newConvertedRotor);
                    openRotorHub();
                    await openCustomAlert(t('rossFileLoaded'));
                } else {
                    await openCustomAlert(t('rossFileError') + '\n' + data.message);
                }
            } catch (err) {
                await openCustomAlert(t('rossFileServerError'));
            }
        }
    };
    reader.readAsText(file); 
    event.target.value = '';
}

// ==========================================
// INTERACTIVE GRAPH ADD FEATURE (ADD BY NODE)
// ==========================================
let hoverButtonTimeout;

let activeHoverNode = null;

function setupPlotHoverEvents() {
    const plotDiv = document.getElementById('plot-rotor');
    if (!plotDiv || !plotDiv.on) return;
    
    nodeMap = [];
    let currZ = 0;
    const effNodes = getEffectiveNodes(state.projectData.shafts || []);
    
    if (effNodes.length > 0) {
        nodeMap.push({n: effNodes[0], z: currZ});
        (state.projectData.shafts || []).forEach((s, i) => {
            let length = parseFloat(s.L) || 0;
            let unit = s.L_unit || 'mm';
            
            if (unit === 'mm') length /= 1000;
            else if (unit === 'cm') length /= 100;
            else if (unit === 'in') length *= 0.0254;
            
            currZ += length;
            nodeMap.push({n: effNodes[i] + 1, z: currZ});
        });
    } else {
        nodeMap.push({n: 0, z: 0});
    }

    let totalLen = currZ > 0 ? currZ : 1;
    let tolerance = totalLen * 0.05; 
    if (tolerance < 0.02) tolerance = 0.02;
    if (tolerance > 0.15) tolerance = 0.15;

    plotDiv.on('plotly_hover', function(data) {
        if(state.projectData.isMultiRotor) return;
        
        let pt = data.points[0];
        let xVal = pt.x; 
        
        let closestNode = null;
        let minDist = Infinity;
        
        nodeMap.forEach(node => {
            let dist = Math.abs(node.z - xVal);
            if(dist < minDist) {
                minDist = dist;
                closestNode = node.n;
            }
        });
        
        if (minDist <= tolerance) {
            let btn = document.getElementById('floating-add-btn');
            let isHidden = !btn || btn.style.display === 'none';
            
            if (activeHoverNode !== closestNode || isHidden) {
                activeHoverNode = closestNode;
                showFloatingAddButton(data.event.clientX, data.event.clientY, closestNode);
            } else {
                clearTimeout(hoverButtonTimeout);
            }
        } else {
            clearTimeout(hoverButtonTimeout);
            hoverButtonTimeout = setTimeout(() => {
                hideFloatingAddButton();
                activeHoverNode = null;
            }, 800); 
        }
    });
    
    plotDiv.on('plotly_unhover', function(data) {
        clearTimeout(hoverButtonTimeout);
        hoverButtonTimeout = setTimeout(() => {
            hideFloatingAddButton();
            activeHoverNode = null;
        }, 800);
    });
}

function showFloatingAddButton(x, y, node) {
    clearTimeout(hoverButtonTimeout);
    let btn = document.getElementById('floating-add-btn');
    if(!btn) {
        btn = document.createElement('button');
        btn.id = 'floating-add-btn';
        btn.innerHTML = '<i class="fas fa-plus"></i>';
        btn.className = 'floating-add-btn';
        document.body.appendChild(btn);
        
        btn.addEventListener('mouseenter', () => clearTimeout(hoverButtonTimeout));
        btn.addEventListener('mouseleave', () => {
            hoverButtonTimeout = setTimeout(() => {
                hideFloatingAddButton();
                activeHoverNode = null;
            }, 300);
        });
    }
    btn.style.display = 'flex';
    btn.style.left = (x + 15) + 'px';
    btn.style.top = (y - 20) + 'px';
    btn.onclick = () => {
        activeHoverNode = null;
        openNodeHub(node);
    };
}

function hideFloatingAddButton() {
    let btn = document.getElementById('floating-add-btn');
    if(btn) btn.style.display = 'none';
}

function openNodeHub(nodeIndex) {
    hideFloatingAddButton();
    document.getElementById('node-hub-overlay').style.display = 'flex';
    document.getElementById('node-hub-target').innerText = nodeIndex;
}

export function closeNodeHub() {
    document.getElementById('node-hub-overlay').style.display = 'none';
}

export const addElementFromNodeHub = function(category) {
    let nodeIndex = document.getElementById('node-hub-target').innerText;
    closeNodeHub();
    openTab(category);
    
    addingFromNodeHub = true;
    nodeHubTarget = nodeIndex;
    openForm(true);
};
