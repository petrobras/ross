// The modeling screen: the element tabs, the form, the rotor figure that redraws
// itself on every change, and the node hub over the figure.
import { buildFormHTML, capturedFormValues, restoreFormValues, toggleAdvanced } from '../components/form.js';
import { getEffectiveNodes, positionFormBox, renderList } from '../components/list.js';
import { reapplyHelp } from '../components/help.js';
import { openCustomAlert, openCustomConfirm } from '../components/modals.js';
import { apiFetch, apiFetchLatest, wasCancelled, projectForServer } from '../core/api.js';
import { busySpinner, escapeHtml } from '../core/dom.js';
import { listContext, projectChanged, state, getActiveData, syncBackToLibrary, writeBackToLibrary } from '../core/state.js';
import { pick, pickAll, picked } from '../core/selection.js';
import { applySnapshot, canRedo, canUndo, redo, undo } from '../core/history.js';
import { afterInsertion, afterRemoval } from '../core/editing.js';
import { elementsUsing, renameMaterial, rossMaterialName } from '../core/material_names.js';
import { themedLayout } from '../core/theme.js';
import { applyLanguage, rememberLanguage, t } from '../core/i18n.js';
import { formSubtypes, loadElementSchema, schemaReady } from '../core/schema.js';
import { projectFromFile } from '../core/project_file.js';
import { VERTICAL_SCALES, withVerticalScale } from '../core/rotor_scale.js';
import { fillAnalysisTypes, redrawAnalyses } from './analysis.js';
import { openRotorHub, renderRotorHub } from './hub.js';
import { splitProject } from './split.js';

// How the rotor figure settles into the panel. This used to be done in the
// backend (BE-12): margin, background, size and legend position assembled on top
// of the Plotly JSON before sending. What knows the size of the panel is the
// screen.
// What the screen says about the rotor figure, and it is now only the theme.
//
// This used to carry `height: null`, a `margin` of its own and a `legend`
// position, from BE-12 in Phase 3: the backend was assembling them and "what
// knows the size of the panel is the screen". That was right at the time.
//
// ROSS 3 changed the ground under it. `plot_rotor` now computes height, margin
// and legend position **together**, and the source says why they belong
// together: the height comes from a nominal width and the `scaleanchor`
// constraint, the bottom margin reserves fixed bands so showing the axes
// indicator does not resize the figure, and the legend hangs from a line below
// the title so a narrow container wraps it downward instead of over the title.
// Measured on a six-element rotor, it asks for `height: 332`,
// `margin {l:70, r:25, t:100, b:102}` and `legend.y = 1.4615`.
//
// We were replacing all three with values that knew nothing about each other --
// a bottom margin of 80 where ROSS reserves 102, and a legend at 1.05 where it
// puts 1.46. Taking one of a set of three and leaving the other two is how a
// figure ends up with its buttons clipped.
//
// So the line is drawn differently now: **ROSS owns the geometry, the screen
// owns the theme**. The two colours stay because ROSS sets neither (measured:
// `paper_bgcolor` and `plot_bgcolor` both come back as None), so making the
// figure transparent over a themed panel is genuinely ours. Width is ours too,
// but not through `responsive: true` -- see `startRotorFigureFollowsWidth`.
const ROTOR_APPEARANCE = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
};

// How many times the drawing is stretched vertically, and the last figure the
// server sent.
//
// The figure is kept so that changing the scale redraws from it instead of
// asking again: the stretch is presentation, the server would answer the same
// rotor, and a round trip for it would cost 600 ms of debounce to show
// something that is already in the browser.
let verticalScale = 1;

let lastRotorFigure = null;

export function setVerticalScale(value) {
    verticalScale = Number(value) || 1;
    if (lastRotorFigure) drawRotorFigure(lastRotorFigure);
    renderVerticalScalePicker();
}

function renderVerticalScalePicker() {
    const holder = document.getElementById('rotor-scale');
    if (!holder) return;
    const options = VERTICAL_SCALES.map(times =>
        `<option value="${times}" ${times === verticalScale ? 'selected' : ''}>${times}×</option>`
    ).join('');
    holder.innerHTML =
        `<span>${escapeHtml(t('verticalScale'))}</span>` +
        `<select data-action="set-vertical-scale">${options}</select>`;
}

// One place draws the rotor, whether the figure just arrived or the scale just
// changed. Two would be two chances to forget the stretch.
//
// The toggle buttons under the axis are ROSS's to place, like the rest of the
// geometry. There used to be a `ROTOR_MENU = { y: -0.15 }` applied here, and
// it was the fourth member of the set this file's comment above says cannot be
// split: `-0.15` is a fraction of the plot area's height, chosen when the area
// was whatever the panel gave (~500 px, so 75 px below the axis). With ROSS's
// own 130 px area it came to 20 px -- on top of the tick labels. ROSS puts them
// at `-0.4615` of its area, which is 60 px down, inside the 102 px band it
// reserves for them. Measured in a browser, not reasoned about.
//
// And `withVerticalScale` **returns** the stretched layout; it does not change
// the one it is given. The first version of this function called it for a side
// effect it no longer had, so 2× and 5× computed a stretch and threw it away --
// the contract of the helper changed and this call site was not revisited.
function drawRotorFigure(fig) {
    lastRotorFigure = fig;
    const dressed = Object.assign(JSON.parse(JSON.stringify(fig.layout)), ROTOR_APPEARANCE);
    const layout = withVerticalScale(
        dressed, verticalScale, t('verticalScaleNote').replace('%1', () => verticalScale),
    );
    Plotly.newPlot('plot-rotor', fig.data, themedLayout(layout), { responsive: false });
    setupPlotHoverEvents();
}

// Whatever takes the figure's place -- the "add a shaft" note, the spinner, an
// error -- goes through here, so that nothing can draw the old rotor back over
// it. `setVerticalScale` and the width redraw both draw from `lastRotorFigure`;
// with every shaft deleted, a resize would otherwise bring back a rotor that no
// longer exists.
function showInsteadOfFigure(container, html) {
    lastRotorFigure = null;
    container.innerHTML = html;
}

// --- following the width -----------------------------------------------------
//
// The figure used to follow the window through Plotly's `responsive: true`, and
// measuring it in a browser showed what that costs: Plotly answers a resize with
// `relayout({autosize: true})`, which **throws away the layout's height**.
// ROSS's 332 px -- and the height the vertical scale computes -- became the
// container's (676 px on a 1080p screen) at the first resize. And a resize is
// any of: resizing the window, zooming the page, hiding the sidebar or the
// list. So the figure a person saw depended on what they had clicked before,
// and the offsets ROSS computed for its buttons and legend were fractions of an
// area that was no longer there.
//
// So the rotor figure is drawn once with `responsive: false` and redrawn here,
// from the figure the server sent, with the geometry ROSS and the scale decided.
// Only the width changes, which is the one thing ROSS left to the screen.
let widthTimer = null;

export function startRotorFigureFollowsWidth() {
    window.addEventListener('resize', () => {
        // A window being dragged fires dozens of these; the figure is redrawn
        // once it settles.
        clearTimeout(widthTimer);
        widthTimer = setTimeout(redrawAtNewWidth, 120);
    });
}

function redrawAtNewWidth() {
    const div = document.getElementById('plot-rotor');
    // On another screen the div has no width, and Plotly would draw at its
    // default 700 px. `switchScreen` fires a resize when the modeling screen
    // comes back, and that one lands here with the real width.
    if (!lastRotorFigure || !div || div.offsetWidth === 0) return;
    drawRotorFigure(lastRotorFigure);
}

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
    const target = button || tabButton(category);
    const key = target && target.dataset && target.dataset.i18n;
    return key ? t(key) : category;
}

// The sidebar button of a category, found by what it *is* (`data-tab`) rather
// than by what it happens to call. It used to be found by reading `openTab('x')`
// out of the `onclick` text -- which is how this slice would have broken the
// title and the highlight the moment the buttons started calling `pickTab`.
function tabButton(category) {
    return Array.from(document.querySelectorAll('.tab-btn'))
        .find(b => b.dataset && b.dataset.tab === category);
}

function tabTitle(category, button) {
    const name = categoryName(category, button);
    return `<div style="display:flex; align-items:center;">
        <span>${escapeHtml(name)}</span> 
        <button class="btn-help-section" data-action="section-help" data-category="${category}" title="${escapeHtml(t('helpAbout'))} ${escapeHtml(name)}"><i class="fas fa-question-circle"></i></button>
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

// --- hiding the list -----------------------------------------------------------
//
// The list panel takes 360 px from the figure, and the figure is decided by its
// width: `plot_rotor` locks 1:1, so on a laptop a slender rotor at 1× is ~87 px
// tall with the list open and ~130 px with it hidden. Clicking the tab that is
// already open hides the list, the same gesture the sidebar already answers to.
//
// Only the **button** toggles. `openTab` is also called by `switchScreen` (on
// the way back from the analyses, with the same tab) and by the node hub; if it
// toggled, coming back to the modeling screen would hide the list every other
// time. So `openTab` keeps meaning "show this tab's list" and leaves the panel
// as it is, and `pickTab` is what the person's click means.
function listPanel() {
    return document.getElementById('list-panel');
}

function listPanelOpen() {
    return !listPanel().classList.contains('collapsed');
}

function showListPanel(open) {
    if (open === listPanelOpen()) return;
    if (open) listPanel().classList.remove('collapsed');
    else listPanel().classList.add('collapsed');
    // `responsive: true` makes Plotly follow the **window**, and the window did
    // not change -- only the panel beside the figure did. Same answer, and same
    // delay (the CSS transition), as `toggleSidebar`.
    setTimeout(() => window.dispatchEvent(new Event('resize')), 300);
}

export function pickTab(category) {
    if (category === state.currentTab && listPanelOpen()) {
        showListPanel(false);
        return;
    }
    showListPanel(true);
    // Reopening the tab that was hidden only shows it again. `openTab` would
    // close the form, and a form half filled in before hiding the list would be
    // lost for having been out of sight.
    if (category !== state.currentTab) openTab(category);
}

export function openTab(category) {
    state.currentTab = category;
    document.getElementById('empty-message').style.display = 'none';
    document.getElementById('list-area').style.display = 'block';
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    
    const activeBtn = tabButton(category);
    if (activeBtn) {
        activeBtn.classList.add('active');
    }
    
    let titleHTML = tabTitle(category, activeBtn);

    if (state.projectData.isMultiRotor) {
        let drvSel = (state.multiRotorEditTarget === 'driving') ? 'selected' : '';
        let drvnSel = (state.multiRotorEditTarget === 'driven') ? 'selected' : '';
        titleHTML += `
            <select id="mr-edit-target" data-action="switch-half" class="target-select">
                <option value="driving" ${drvSel}>${escapeHtml(t('multiDriving'))}: ${escapeHtml(state.projectData.driving_rotor.name)}</option>
                <option value="driven" ${drvnSel}>${escapeHtml(t('multiDriven'))}: ${escapeHtml(state.projectData.driven_rotor.name)}</option>
            </select>
        `;
    }
    
    document.getElementById('tab-title').innerHTML = titleHTML;
    
    closeForm();
    renderList();
}

// Function to open the form

export async function openForm(isNew = true) {
    // A form opened from the figure (the node hub) or from anywhere else has to
    // be seen: it lives in the list panel.
    showListPanel(true);
    await schemaReady();          // the forms come from /api/schema/elements
    if (isNew) { state.editingIndex = -1; state.currentSubType = 'BASIC'; }
    let subTypes = formSubtypes(state.currentTab);
    
    if (addingFromNodeHub) {
        subTypes = subTypes.filter(type => type !== 'LIST');
    }
    
    if (isNew && subTypes.length > 1) {
        let html = `<h4 class="subtype-header">${escapeHtml(t('selectModel'))}</h4>`
                 + '<div class="subtype-grid">';
        subTypes.forEach(type => { html += `<button class="btn-subtype" data-action="pick-subtype" data-subtype="${type}">${type}</button>`; });
        html += '</div><button class="btn-cancel" style="width:100%; margin-top:15px;" data-action="close-form">' + escapeHtml(t('cancel')) + '</button>';
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
        document.querySelector('.form-actions').insertAdjacentHTML('afterbegin', `<button type="button" id="btn-default-form" class="btn-default" data-action="fill-default"><i class="fas fa-magic"></i> ${escapeHtml(t('defaultButton'))}</button>`);
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
            // Escaped: a material's name is whatever was typed into it.
            const mName = escapeHtml(m.name || 'MaterialCustom');
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

// One copy of an element, with a name nobody else is using. Pulled out of
// `copyItem` so that copying several uses the same rule as copying one -- two
// naming rules for the same act is how a list ends up with `Stage_1` twice.
export function freshCopy(original, siblings) {
    const copiedItem = JSON.parse(JSON.stringify(original));
    if (copiedItem.tag) {
        let baseTag = copiedItem.tag.replace(/_\d+$/, '');
        let counter = 1;
        let newTag = `${baseTag}_${counter}`;
        const tagInUse = (cTag) => siblings.some(item => item.tag === cTag);
        while (tagInUse(newTag)) {
            counter++;
            newTag = `${baseTag}_${counter}`;
        }
        copiedItem.tag = newTag;
    }
    return copiedItem;
}

export function copyItem(index) { 
    const activeData = getActiveData();
    const original = activeData[state.currentTab][index];
    const copiedItem = freshCopy(original, activeData[state.currentTab]);

    activeData[state.currentTab].splice(index + 1, 0, copiedItem); 
    state.editingIndex = afterInsertion(state.editingIndex, index + 1);
    syncBackToLibrary();
    renderList(); 
    buildRotorLive(); 
}

// Delete function for the element

// The question, when there is one, is the only reason this is not the plain
// call it used to be: an `await` suspends even when the answer is already
// known, and the batteries -- like the screen -- expect a delete with nothing
// to ask about to have happened by the time the call returns. So the promise
// is only in the path that has a question.
export function deleteItem(index) {
    const activeData = getActiveData();
    const asking = askBeforeMaterialGoes(activeData, [index]);
    if (asking) return asking.then(yes => { if (yes) removeItem(activeData, index); });
    removeItem(activeData, index);
    return undefined;
}

function removeItem(activeData, index) {
    activeData[state.currentTab].splice(index, 1);
    const editing = afterRemoval(state.editingIndex, index);
    if (editing === null) closeForm();
    else state.editingIndex = editing;
    syncBackToLibrary();
    renderList();
    buildRotorLive();
}

// Split function for a shaft element
//
// Here and not in features/split.js because this is the third thing that
// mutates the element list, and the other two -- `copyItem` and `deleteItem` --
// already live side by side with the two calls that put the screen back in
// agreement with the data. The question, the trip to the server and the answer
// are in `splitProject`; what is added here is only what the screen owes.
export async function splitItem(index) {
    const activeData = getActiveData();
    if (!(await splitProject(activeData, index))) return;

    // The form is open on the element that just stopped existing as one thing:
    // saving it would write the whole original back over its left half. Below
    // the split nothing moved; above it, the right half was inserted.
    if (state.editingIndex === index) closeForm();
    else state.editingIndex = afterInsertion(state.editingIndex, index + 1);

    syncBackToLibrary();
    renderList();
    buildRotorLive();
}

// Undo and redo, for the modelling screen only
//
// Here, beside `copyItem`, `deleteItem` and `splitItem`, because restoring a
// snapshot owes the screen exactly what they owe it: the form closed if it was
// open on something that may no longer be there, the list redrawn, the figure
// rebuilt. The bookkeeping -- which snapshot, and whether there is one -- is in
// core/history.js, which knows nothing about any of this.
function restore(snapshot) {
    if (snapshot === null) return;

    applySnapshot(state.projectData, snapshot);

    // The form was open on an element of a model that no longer exists. Saving
    // it would write a row from one version of the rotor into another, at an
    // index that means something different now.
    closeForm();

    // `writeBackToLibrary` and not `syncBackToLibrary`: a restore that recorded
    // itself would push onto the stack the very step it just took off.
    writeBackToLibrary();
    // Through `projectChanged`, not by calling `refreshHistoryButtons` here.
    // This line used to name that one subscriber directly, which worked only
    // while there was one -- and it would have skipped the selection, whose
    // ticked positions point at a model that is no longer on screen.
    projectChanged();
    renderList();
    buildRotorLive();
}

export function undoModelling() {
    restore(undo());
}

export function redoModelling() {
    restore(redo());
}

// Greys the two buttons out when there is nowhere to go. Subscribed to
// `onProjectChanged` in main.js, so it runs after every mutation without any
// mutation having to remember it.
export function refreshHistoryButtons() {
    const back = document.getElementById('btn-undo');
    const forward = document.getElementById('btn-redo');
    if (back) back.disabled = !canUndo();
    if (forward) forward.disabled = !canRedo();
}

// Ticking, and the two things worth doing to several elements at once
//
// All four end in the same three lines as `copyItem` and `deleteItem`, and the
// two that change the model call `syncBackToLibrary` **once**. That is not an
// economy: it is what makes deleting eight elements one step of the undo
// instead of eight, which is what a person means by "undo that".
export function toggleSelected(index) {
    pick(listContext(), index);
    renderList();
}

export function toggleSelectAll() {
    pickAll(listContext(), (getActiveData()[state.currentTab] || []).length);
    renderList();
}

// Deleting a material that elements still use: the rotor stops building, and
// the message comes from the server, later, naming elements the person is no
// longer looking at. So the question is asked here, with the count, before it
// happens. `null` when there is nothing to ask.
function askBeforeMaterialGoes(project, positions) {
    if (state.currentTab !== 'materials') return null;
    const using = positions.reduce(
        (total, index) => total + elementsUsing(project, (project.materials[index] || {}).name), 0);
    if (!using) return null;
    const question = using === 1 ? t('deleteMaterialInUseOne') : t('deleteMaterialInUse');
    return openCustomConfirm(question.replace('%1', () => using));
}

export function deleteSelected() {
    const chosen = picked(listContext());
    if (!chosen.length) return undefined;
    const activeData = getActiveData();
    const asking = askBeforeMaterialGoes(activeData, chosen);
    if (asking) return asking.then(yes => { if (yes) removeSelected(activeData, chosen); });
    removeSelected(activeData, chosen);
    return undefined;
}

function removeSelected(activeData, chosen) {
    // Backwards, because deleting position 2 makes every later position mean
    // something else. Going forwards would delete the wrong elements and give
    // no sign of it -- the list would simply be shorter.
    chosen.slice().reverse().forEach(index => {
        activeData[state.currentTab].splice(index, 1);
    });

    // A bulk change is a bigger change than an open form can survive: the
    // element it was editing may be gone, and the ones after it have moved.
    closeForm();
    syncBackToLibrary();
    renderList();
    buildRotorLive();
}

export function copySelected() {
    const chosen = picked(listContext());
    if (!chosen.length) return;
    const activeData = getActiveData();
    const list = activeData[state.currentTab];

    // Backwards again, and for the same reason: each copy goes in right after
    // its own original, and inserting at position 2 moves everything after it.
    // Walking from the end leaves the positions still to be handled untouched.
    chosen.slice().reverse().forEach(index => {
        list.splice(index + 1, 0, freshCopy(list[index], list));
    });

    closeForm();
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
            if (state.currentTab === 'materials' && newObj.name !== undefined) {
                newObj.name = rossMaterialName(newObj.name);
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
        
        // ROSS refuses a space in a material's name (core/material_names.js);
        // and an edited name takes the elements that used it along.
        if (state.currentTab === 'materials' && newObject.name !== undefined) {
            newObject.name = rossMaterialName(newObject.name);
            if (state.editingIndex >= 0) {
                renameMaterial(activeData, activeData.materials[state.editingIndex].name, newObject.name);
            }
        }

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
        showInsteadOfFigure(plotContainer,
            `<div style="display: flex; height: 100%; min-height: 400px; align-items: center; justify-content: center;">`
            + `<p class="placeholder-text">${escapeHtml(t('addOneShaft'))}</p></div>`);
        if(infoContainer) infoContainer.style.opacity = '0';
        return;
    }    
    rotorUpdateActive = true;
    plotContainer.style.opacity = '0.4';
    plotContainer.style.pointerEvents = 'none';
    let loadingTimer = setTimeout(() => {
        if(rotorUpdateActive) {
            plotContainer.style.opacity = '1';
            showInsteadOfFigure(plotContainer, `
                <div style="display:flex; flex-direction:column; justify-content:center; align-items:center; height:100%; min-height:400px; color: var(--text-main);">
                    <span style="margin-bottom:15px; color: var(--accent-primary);">${busySpinner(3)}</span>
                    <h3 style="margin:0;">${escapeHtml(t('computingElement'))}</h3>
                    <p style="color: var(--text-muted); text-align:center; padding:0 20px;">${escapeHtml(t('usingCache'))}</p>
                </div>`);
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
            drawRotorFigure(JSON.parse(data.plot_json));
            renderVerticalScalePicker();
            if(infoContainer) {
                document.getElementById('info-mass').innerText = data.mass.toFixed(4);
                document.getElementById('info-ip').innerText = data.ip.toFixed(4);
                infoContainer.style.opacity = '1';
            }
        } else {
            // `data.message` is escaped like everything else: it quotes what the
            // user typed back ("could not read 'abc'"), and a tag typed into a
            // field would otherwise be drawn as a tag.
            showInsteadOfFigure(plotContainer, `<div class="analysis-error"><i class="fas fa-exclamation-triangle fa-2x"></i><br><b>${escapeHtml(t('modelingError'))}</b><br>${escapeHtml(data.message)}</div>`);
            if(infoContainer) infoContainer.style.opacity = '0';
        }
    } catch (e) { 
        if (wasCancelled(e)) return;   // a newer request has taken over
        rotorUpdateActive = false; 
        clearTimeout(loadingTimer); 
        plotContainer.style.opacity = '1';
        showInsteadOfFigure(plotContainer, `<p class="analysis-error analysis-error-tall">`
            + `${escapeHtml(t('serverConnectionError'))}</p>`);
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
            // The whole rule -- is this ours, and what project is it -- lives in
            // core/project_file.js, with no DOM around it, so it can be checked
            // case by case. Here what is left is pushing it and saying so.
            const project = projectFromFile(loaded, file.name.replace('.json', ''));
            if (project) {
                isInterfaceJSON = true;
                state.rotorLibrary.push(project);
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
