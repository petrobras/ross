// The model survives a closed tab; the charts do not, deliberately. A chart
// from yesterday, of an earlier version of the rotor, would look current.
import { analysesToSave } from './analysis_store.js';
import { state } from './state.js';
// --- Persistence ------------------------------------------------------------
//
// Until Phase 3 the interface stored nothing: closing the tab lost the whole
// project (FE-09). `localStorage` was used only to remember the language.
//
// **We store the model, not the charts.** A Plotly figure of a Campbell runs
// past 15 KB; half a dozen saved analyses would blow the localStorage quota in
// a few projects. And a chart restored from yesterday, from an earlier version
// of the rotor, would be worse than none -- it would look current. On reopening,
// the cards come back with their configuration and an explicit Update.
//
// Saving is periodic and unconditional, rather than triggered on every
// mutation. Dozens of places touch the project (every field of every form,
// every analysis, every Hub operation) and forgetting one would leave a silent
// loss; a clock cannot forget. The model is small -- about 1 KB per rotor -- and
// the write only happens if something changed.

export const STATE_KEY = 'ross_interface_state_v1';

export const REFUSED_KEY = 'ross_interface_state_refused';

// The key used to be written in Portuguese (`..._estado_v1`). Renaming it with
// the rest of the codebase would have silently emptied the library of anyone
// who had already saved work: the new key is absent, and an absent key is a
// first run. So the old one is read once, when the new one has nothing, and
// what comes back is written under the new name on the next save.
const OLD_STATE_KEY = 'ross_interface_estado_v1';

export const STATE_VERSION = 1;

const SAVE_INTERVAL = 5000;

let lastStateWritten = null;

let persistenceOff = false;

// An analysis becomes configuration: title, type, parameters and conversion.
// The chart is left out by decision, not by oversight.
function analysisWithoutChart(analysis) {
    return {
        title: analysis.title,
        type: analysis.type,
        params: analysis.params || {},
        conversion: analysis.conversion || ''
    };
}

function rotorWithoutCharts(rotor) {
    const copy = Object.assign({}, rotor);
    copy.savedAnalyses = (rotor.savedAnalyses || []).map(analysisWithoutChart);
    ['driving_rotor', 'driven_rotor'].forEach(key => {
        if (copy[key]) copy[key] = rotorWithoutCharts(copy[key]);
    });
    return copy;
}

export function stateToDisk() {
    const library = state.rotorLibrary.map((rotor, index) => {
        // The open rotor has its live analyses in ANALYSES; its savedAnalyses is only
        // refreshed on the way back to the Hub, and would be stale here.
        if (index !== state.activeRotorIndex) return rotorWithoutCharts(rotor);
        const current = Object.assign({}, rotor, { savedAnalyses: analysesToSave() });
        return rotorWithoutCharts(current);
    });
    return { version: STATE_VERSION, library: library };
}

export function saveState() {
    if (persistenceOff) return false;
    let text;
    try {
        text = JSON.stringify(stateToDisk());
    } catch (error) {
        console.error('persistence: could not serialise the state', error);
        return false;
    }
    if (text === lastStateWritten) return false;

    try {
        localStorage.setItem(STATE_KEY, text);
        lastStateWritten = text;
        return true;
    } catch (error) {
        // Quota exceeded, or a browser with storage blocked. Switching off and saying
        // so once beats retrying every five seconds in silence.
        persistenceOff = true;
        console.error('persistence switched off:', error);
        return false;
    }
}

// Return the stored library, or null. A state that will not parse must not
// bring the interface down, nor be discarded without a trace: it is kept under
// another key, so it does not become work lost in silence.
export function stateFromDisk() {
    let text;
    try {
        text = localStorage.getItem(STATE_KEY) || localStorage.getItem(OLD_STATE_KEY);
    } catch (error) {
        return null;      // storage unavailable
    }
    if (!text) return null;

    let stored;
    try {
        stored = JSON.parse(text);
    } catch (error) {
        refuseState(text, 'not valid JSON');
        return null;
    }

    if (!stored || stored.version !== STATE_VERSION) {
        refuseState(text, 'written by another version of the interface');
        return null;
    }
    if (!Array.isArray(stored.library)) {
        refuseState(text, 'has no rotor library');
        return null;
    }
    return stored.library;
}

export function refuseState(text, reason) {
    console.warn('stored state refused (' + reason + '); ' +
                 'copy kept in ' + REFUSED_KEY);
    try {
        localStorage.setItem(REFUSED_KEY, text);
        localStorage.removeItem(STATE_KEY);
    } catch (error) {
        /* no room for the copy: the original stays where it is */
    }
}

export function restoreState() {
    const library = stateFromDisk();
    if (!library || library.length === 0) return 0;

    state.rotorLibrary = library;
    // Stays on the Hub on purpose: reopening a rotor by itself would fire a
    // computation on the server without the user having asked for anything.
    state.activeRotorIndex = -1;
    return state.rotorLibrary.length;
}

// Persistence switches itself off when localStorage refuses a write (quota
// full): retrying every five seconds would only wear the browser out.
// Switching back on is explicit, and that is what `startPersistence` does --
// before, the shutdown lasted until the page was reloaded, even if the quota
// had been freed.
export function resumePersistence() {
    persistenceOff = false;
    lastStateWritten = null;
}

export function persistenceIsOff() {
    return persistenceOff;
}

export function startPersistence() {
    resumePersistence();
    setInterval(saveState, SAVE_INTERVAL);
    // Closing the tab must not cost the last few seconds of work.
    window.addEventListener('beforeunload', saveState);
}
