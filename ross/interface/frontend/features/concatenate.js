// Joining two rotors of the library into one shaft line.
//
// The hub now has two buttons side by side and they do different things:
// linking a MultiRotor couples two shafts through a gear mesh, each turning at
// its own speed; concatenating makes **one** shaft line turning at one speed.
// Nothing about the two names says that, so the modal says it, above the fold,
// before anything is chosen.
//
// The second thing the screen has to say is the one the measurement turned up:
// `Rotor.concatenate` shares the joint node. The last node of the first rotor
// *is* the first node of the second -- they are welded, not queued with a gap --
// and a user who expected two machines in a row would get a different rotor with
// no warning. So the modal says, in numbers, where the joint falls and how many
// nodes come out, and it says it while the selects are being changed rather than
// afterwards.
import { getEffectiveNodes, getListedNodes } from '../components/list.js';
import { openCustomAlert } from '../components/modals.js';
import { apiFetch } from '../core/api.js';
import { conversionName } from '../core/analysis_store.js';
import { escapeHtml } from '../core/dom.js';
import { t } from '../core/i18n.js';
import { ensureUIDs, state } from '../core/state.js';
import { renderRotorHub } from './hub.js';

// The screen's own count, for the sentence above the buttons only. What decides
// the concatenation is the numbering the server reads back from ROSS; this is a
// preview and the comment is here so nobody promotes it. `getEffectiveNodes` is
// the half of the node rule that already has a test comparing it, case by case,
// against `domain/node_resolver.effective_nodes` -- so this adds no second copy
// of anything, it just reads the existing one.
// Couplings are numbered by `getListedNodes`, as the builder numbers them.
export function structuralNodes(rotor) {
    const lefts = getEffectiveNodes(rotor.shafts || [])
        .concat(getListedNodes(rotor.couplings || []));
    if (!lefts.length) return 0;
    return Math.max(...lefts) + 1;
}

// The rotor model its analyses agree on, or null when they disagree, or
// undefined when there are none. The server decides the refusal -- this is only
// what the modal shows beside each name, so that a refusal is never the first
// time the user hears about it.
export function modelOfRotor(rotor) {
    const used = new Set((rotor.savedAnalyses || []).map(a => a.conversion || ''));
    if (!used.size) return undefined;
    if (used.size > 1) return null;
    return used.values().next().value;
}

// The name of a model comes from `conversionName`, which is the badge the cards
// already wear. Writing the three names again here would be a second table of
// the same thing, and the day one of them changes only one would follow.
function modelLabel(rotor) {
    const model = modelOfRotor(rotor);
    if (model === undefined) return t('concatNoAnalyses');
    if (model === null) return t('concatMixedModels');
    return conversionName(model);
}

function chosen(which) {
    const node = document.getElementById(which);
    return state.rotorLibrary[Number(node.value)];
}

// Runs on every change of either select. A sentence that only appears after the
// button is pressed is a sentence that explains what already happened.
export function describeJoint() {
    const hint = document.getElementById('cc-hint');
    if (!hint) return;
    const first = chosen('cc-first');
    const second = chosen('cc-second');
    // One property, always `innerHTML`. Writing `textContent` on one branch and
    // `innerHTML` on another leaves an element whose content depends on which
    // branch ran last, and the two are different fields to anything reading it
    // back -- the node battery could not see this sentence at all.
    if (!first || !second) {
        hint.innerHTML = '';
        return;
    }
    if (first === second) {
        hint.innerHTML = escapeHtml(t('concatSameRotor'));
        return;
    }
    const ends = structuralNodes(first);
    const total = ends + structuralNodes(second);
    hint.innerHTML =
        `<b>${escapeHtml(first.name || '')}</b> ${escapeHtml(t('concatEndsAtNode'))} ` +
        `<b>${ends}</b>. <b>${escapeHtml(second.name || '')}</b> ` +
        `${escapeHtml(t('concatStartsThere'))} ` +
        `${escapeHtml(t('concatResultHas'))} <b>${total + 1}</b> ` +
        `${escapeHtml(t('concatNodes'))}. ` +
        `${escapeHtml(t('concatModelsAre'))} ${escapeHtml(modelLabel(first))} ` +
        `${escapeHtml(t('concatAnd'))} ${escapeHtml(modelLabel(second))}.`;
}

export async function openConcatenateModal() {
    if (state.rotorLibrary.length < 2) {
        await openCustomAlert(t('concatNeedTwoRotors'));
        return;
    }
    let options = '';
    state.rotorLibrary.forEach((rotor, index) => {
        options += `<option value="${index}">${escapeHtml(rotor.name || '')}</option>`;
    });
    const first = document.getElementById('cc-first');
    const second = document.getElementById('cc-second');
    first.innerHTML = options;
    second.innerHTML = options;
    // `value` and not `selectedIndex`: the two are the same thing in a browser,
    // and only one of them is something the code states. `selectedIndex` also
    // made this untestable -- the fake DOM of the node batteries models `value`
    // and not the selection, so the screen's whole sentence went unchecked while
    // looking checked. `openMultiRotorModal` still has that spot.
    second.value = state.rotorLibrary.length > 1 ? '1' : '0';
    first.value = '0';
    document.getElementById('cc-name').value = '';
    describeJoint();
    document.getElementById('concatenate-modal-overlay').style.display = 'flex';
}

export function closeConcatenateModal() {
    document.getElementById('concatenate-modal-overlay').style.display = 'none';
}

// Order matters and the modal has to let it be changed without re-picking both:
// A then B is not the same rotor as B then A.
export function swapConcatenationOrder() {
    const first = document.getElementById('cc-first');
    const second = document.getElementById('cc-second');
    const keep = first.value;
    first.value = second.value;
    second.value = keep;
    describeJoint();
}

function conversionsOf(rotor) {
    return (rotor.savedAnalyses || []).map(analysis => analysis.conversion || '');
}

export async function saveConcatenation() {
    const first = chosen('cc-first');
    const second = chosen('cc-second');
    if (!first || !second) return;
    if (first === second) {
        await openCustomAlert(t('concatSameRotor'));
        return;
    }

    let answer;
    try {
        answer = await apiFetch('/api/rotor/concatenate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                first: withoutAnalyses(first),
                second: withoutAnalyses(second),
                first_conversions: conversionsOf(first),
                second_conversions: conversionsOf(second),
            }),
        });
    } catch (error) {
        await openCustomAlert(t('concatFailed'));
        return;
    }

    const body = await answer.json();
    // The refusals of the domain arrive as a 400 carrying the sentence. Showing
    // `body.message` and not a phrase of our own is what makes "these two rotors
    // were analysed under different rotor models" reach the person who has to
    // act on it -- the alternative is a generic failure and a guess.
    if (!answer.ok) {
        await openCustomAlert(body.message || t('concatFailed'));
        return;
    }

    ensureUIDs();
    state.rotorLibrary.push({
        name: document.getElementById('cc-name').value || defaultName(first, second),
        uid: 'rotor_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9),
        savedAnalyses: [],
        ...body.projectData,
    });
    closeConcatenateModal();
    renderRotorHub();
}

// The two source rotors stay in the library untouched: concatenating adds a
// third, it does not consume them. Their analyses do not come along -- they were
// computed on a rotor that no longer describes this one.
function withoutAnalyses(rotor) {
    const copy = JSON.parse(JSON.stringify(rotor));
    delete copy.savedAnalyses;
    return copy;
}

function defaultName(first, second) {
    return `${first.name || ''} + ${second.name || ''}`;
}
