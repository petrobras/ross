// The Hub: the rotor library. Create, copy, rename, delete, open, and save or
// export a rotor without having to open it.
import { openCustomConfirm, openCustomPrompt } from '../components/modals.js';
import { analysesToSave, forgetAllAnalyses } from '../core/analysis_store.js';
import { escapeHtml } from '../core/dom.js';
import { ensureUIDs, openProjectHistory, state, syncMultiRotors } from '../core/state.js';
import { saveState } from '../core/persistence.js';
import { CATEGORIES } from '../core/project_file.js';
import { emptyListNotice, restoreAnalysesFromMemory } from './analysis.js';
import { generatePythonFile } from './export.js';
import { buildRotorLive, closeForm } from './modeling.js';
import { switchScreen } from './screens.js';
import { t } from '../core/i18n.js';
// Function to open the Hub screen and render the list

export function openRotorHub() {
    if (state.activeRotorIndex !== -1) {
        state.rotorLibrary[state.activeRotorIndex].savedAnalyses = analysesToSave();
        forgetAllAnalyses();
        state.activeRotorIndex = -1;
        saveState();   // the moment the analyses settle
    }

    switchScreen('screen-rotor-hub');
    renderRotorHub();
}

// Renders the list of rotors on the screen

export function renderRotorHub() {
    const container = document.getElementById('rotor-hub-list');
    if (!container) return;
    container.innerHTML = '';
    
    if (state.rotorLibrary.length === 0) {
        container.innerHTML = `<div class="empty-message">${escapeHtml(t('noRotors'))}</div>`;
        return;
    }

    state.rotorLibrary.forEach((rotor, index) => {
        let name = rotor.name || `Rotor ${index + 1}`;
        let badge = "";
        
        if (rotor.isMultiRotor) {
            badge = `<span class="badge-conversion badge-multirotor"><i class="fas fa-link"></i> MultiRotor</span>`;
        } else {
            badge = `<span style="font-size:11px; color:var(--text-muted); font-weight:normal; margin-left:8px;">(${escapeHtml(elementsLabel(elementCount(rotor)))})</span>`;
        }

        container.innerHTML += `
            <div class="hub-card">
                <div class="hub-card-top">
                    <div class="hub-card-title">
                        <i class="${rotor.isMultiRotor ? 'fas fa-link' : 'fas fa-cogs'}"></i> 
                        <span style="cursor:pointer;" data-action="rename-rotor" data-index="${index}" title="${escapeHtml(t('editName'))}">
                            ${escapeHtml(name)} <i class="fas fa-pen" style="font-size:11px; color:var(--text-muted); margin-left:4px;"></i>
                        </span> 
                        ${badge}
                    </div>
                    <div class="hub-card-actions">
                        <button class="btn-action copy" data-action="copy-rotor" data-index="${index}" title="${escapeHtml(t('copy'))}"><i class="fas fa-copy"></i></button>
                        <button class="btn-action delete" data-action="delete-rotor" data-index="${index}" title="${escapeHtml(t('delete'))}"><i class="fas fa-trash"></i></button>
                    </div>
                </div>
                <div class="hub-card-bottom">
                    <button class="btn-primary" data-action="open-rotor" data-index="${index}" data-screen="screen-modeling"><i class="fas fa-tools"></i> ${escapeHtml(t('goToModeling'))}</button>
                    <button class="btn-secondary" data-action="open-rotor" data-index="${index}" data-screen="screen-analysis"><i class="fas fa-chart-line"></i> ${escapeHtml(t('goToAnalysis'))}</button>
                    <button class="btn-secondary" data-action="save-rotor-file" data-index="${index}"><i class="fas fa-save"></i> ${escapeHtml(t('saveJson'))}</button>
                    <button class="btn-secondary" data-action="export-rotor-python" data-index="${index}"><i class="fab fa-python"></i> ${escapeHtml(t('generatePython'))}</button>
                </div>
            </div>
        `;
    });
}

// Function to rename rotors and multi-rotors directly via the Hub

export async function editRotorName(index) {
    let currentName = state.rotorLibrary[index].name || `Rotor ${index + 1}`;
    let newName = await openCustomPrompt(t('renameRotor'), currentName);
    
    if (newName !== null && newName.trim() !== "") {
        state.rotorLibrary[index].name = newName.trim();
        renderRotorHub();
    }
}

// Creates a new empty rotor in the Hub

export async function createNewRotorInHub() {
    let defaultName = `Rotor ${state.rotorLibrary.length + 1}`;
    let baseName = await openCustomPrompt(t('nameNewRotor'), defaultName);
    
    if (baseName === null) return; 
    
    if (baseName.trim() === "") baseName = defaultName; 

    let newRotor = {
        name: baseName,
        savedAnalyses: [],
        materials: [], shafts: [], disks: [], gears: [], couplings: [], seals: [], bearings: [], pointmasses: []
    };
    state.rotorLibrary.push(newRotor);
    renderRotorHub();
}

// Deletes a rotor from the Hub

export async function deleteRotorInHub(index) {
    let isConfirmed = await openCustomConfirm(t('confirmDeleteRotor'));
    if (isConfirmed) {
        state.rotorLibrary.splice(index, 1);
        if (state.activeRotorIndex === index) state.activeRotorIndex = -1;
        renderRotorHub();
    }
}

// Copies an existing rotor

export function copyRotorInHub(index) {
    let copiedRotor = JSON.parse(JSON.stringify(state.rotorLibrary[index]));
    copiedRotor.name = copiedRotor.name + " (Copy)";
    state.rotorLibrary.push(copiedRotor);
    renderRotorHub();
}

// Opens the selected rotor in the Modeling or Analysis environment

export function openRotorWorkspace(index, targetScreen) {
    ensureUIDs();
    syncMultiRotors();

    state.activeRotorIndex = index;
    state.projectData = state.rotorLibrary[index]; 

    // A fresh history, and it starts holding this rotor rather than nothing:
    // the first change records *this* model as the step to come back to.
    // Undoing across this boundary would restore one rotor over another.
    //
    // `openProjectHistory` and not `resetHistory`: it announces, and the
    // buttons have to hear it. See the comment beside it in core/state.js.
    openProjectHistory(state.projectData);
    showOpenRotorName();
    
    // The form first, then the list. The form lives *inside* the list while an
    // element is being edited, and emptying the list with it there deleted it
    // from the page for good: every later `closeForm` threw, `openTab` stopped
    // before drawing anything, and the element lists stayed empty until the
    // page was reloaded. Leaving the modeling screen with a form open was
    // enough -- which is very likely the "lists vanished" of the week's report.
    // `closeForm` puts it back in `#list-area` and sets `editingIndex` to -1.
    closeForm();
    document.getElementById('element-list').innerHTML = '';
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    
    if (targetScreen === 'screen-modeling') {
        state.multiRotorEditTarget = 'driving';
    }

    switchScreen(targetScreen);
    
    if (targetScreen === 'screen-modeling') {
        if (state.currentTab) {
            document.getElementById('list-area').style.display = 'block';
            document.getElementById('empty-message').style.display = 'none';
            document.getElementById('btn-add-item').style.display = 'block';
        } else {
            document.getElementById('list-area').style.display = 'none';
            document.getElementById('empty-message').style.display = 'block';
        }

        document.querySelector('.sidebar').style.opacity = '1';
        document.querySelector('.sidebar').style.pointerEvents = 'auto';
        
        buildRotorLive();
    }

    const analysisContainer = document.getElementById('analysis-list');
    if (analysisContainer) {
        // The list and the state empty out together. Clearing only one of the two would
        // leave the previous rotor's analyses alive in whatever got saved under this one.
        forgetAllAnalyses();
        analysisContainer.innerHTML = ''; 
        if (state.projectData.savedAnalyses && state.projectData.savedAnalyses.length > 0) {
            restoreAnalysesFromMemory(state.projectData.savedAnalyses);   // async: the cards show up when the schema arrives
        } else {
            analysisContainer.innerHTML = emptyListNotice();
        }
    }
}

// Saves the JSON directly from the Hub

export function saveRotorFromHub(index) {
    const rotorToSave = state.rotorLibrary[index];
    const fileName = (rotorToSave.name || "my_rotor").replace(/\s+/g, '_') + ".json";
    const blob = new Blob([JSON.stringify(rotorToSave, null, 2)], {type: "application/json"});
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = fileName;
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
}

// Exports a rotor kept in the hub, with **its own** analyses.
//
// Until Phase 3 this function sent the empty list, and for a reason that was a
// good one at the time: the only source of analyses was the DOM, holding the
// cards of the open project -- which belong to another rotor and would come out
// with nodes this one does not even have. When the analysis state left the DOM,
// each rotor started carrying its own in savedAnalyses, and the reason stopped
// existing. The empty list stayed, and exporting from the Hub came with no
// analysis at all.
export function generatePythonFromHub(index) {
    const rotor = state.rotorLibrary[index] || {};
    const saved = (rotor.savedAnalyses || [])
        .filter(a => a && a.type && Object.keys(a.params || {}).length > 0)
        .map(a => ({ type: a.type, params: a.params, conversion: a.conversion || '' }));
    return generatePythonFile(rotor, saved);
}


// Which rotor is on the modelling screen, in its topbar.
//
// It was not shown anywhere before, which is a gap you only notice once there
// is an undo button: "undo" is a question about a particular model, and the
// screen was not saying which one.
// Every element of a rotor, whatever its kind. It used to add up shafts, disks
// and bearings only, so a rotor with gears, couplings, seals or point masses
// was announced with fewer elements than it had. Materials are not elements of
// the rotor -- they are what the shafts are made of -- and stay out.
// "1 element", not "1 elements".
function elementsLabel(count) {
    return (count === 1 ? t('elementsCountOne') : t('elementsCount')).replace('%1', () => count);
}

function elementCount(rotor) {
    return CATEGORIES
        .filter(category => category !== 'materials')
        .reduce((total, category) => total + (Array.isArray(rotor[category]) ? rotor[category].length : 0), 0);
}

export function showOpenRotorName() {
    const label = document.getElementById('modeling-rotor-name');
    if (!label) return;
    const name = (state.projectData && state.projectData.name) || '';
    label.textContent = name ? ' \u2014 ' + name : '';
    // A long name is cut with an ellipsis in the header; the whole of it is
    // one hover away.
    label.title = name;
}
