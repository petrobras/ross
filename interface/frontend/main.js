// Entry point.
//
// No logic here: it wires the layers, fires the bootstrap and publishes on
// `window` the names the inline handlers of the HTML call. Everything else lives
// in `core/`, `components/` and `features/`.
//
// The bridge is temporary by construction: the next slice replaces the inline
// handlers with event delegation, and each converted handler erases a name from
// it.
import { closeHelpModal, openAnalysisCardHelp, openAnalysisHelp, openGeneralHelp, openSectionHelp } from './components/help.js';
import { fillDefault, handleUnitChange, toggleAdvanced } from './components/form.js';
import { onReorder } from './components/list.js';
import { closeCustomAlert, closeCustomConfirm, closeCustomPrompt, confirmCustomPrompt } from './components/modals.js';
import { applyLanguage } from './core/i18n.js';
import { startPersistence, restoreState } from './core/persistence.js';
import { schemaReady } from './core/schema.js';
import { addAnalysis, addAngleProbeRow, addForceRow, addProbeRow, addUnbalanceRow, checkDeps, deleteAnalysis, loadAnalysis, loadAnalysisDirect, fillAnalysisTypes, runCardAnalysis, saveAnalysis, toggleAnalysis, toggleDashAdv } from './features/analysis.js';
import { generatePythonFile } from './features/export.js';
import { copyRotorInHub, createNewRotorInHub, deleteRotorInHub, editRotorName, generatePythonFromHub, openRotorHub, openRotorWorkspace, renderRotorHub, returnToHub, saveRotorFromHub } from './features/hub.js';
import { addElementFromNodeHub, buildRotorLive, changeLanguage, closeForm, closeNodeHub, copyItem, deleteItem, editItem, loadRotor, openForm, openTab, saveItem, saveRotor, selectSubType } from './features/modeling.js';
import { closeMultiRotorModal, openMultiRotorModal, saveMultiRotor, switchMultiRotorTarget } from './features/multirotor.js';
import { startWorkBar } from './features/progress.js';
import { exitApplication, switchScreen, toggleAnalysisSidebar, toggleSidebar } from './features/screens.js';

// --- Wiring between layers -------------------------------------------------
//
// The element list does not know the rotor: it announces that the order changed,
// and whoever knows how to rebuild subscribes here. It was this inversion that
// removed the only path from a component to a feature -- and it was the one I
// created and forgot to connect, leaving dragging with no effect on the figure.
onReorder(buildRotorLive);


// The schema is loaded once at startup; openForm waits for it.
document.addEventListener('DOMContentLoaded', () => {
    // The page is born in English in the HTML; the chosen language is applied as
    // soon as the schema arrives, along with the analysis titles that come with it.
    applyLanguage();
    // The bar that says what is being computed and offers the only way to stop
    // it. Here and not at module level because it takes hold of elements of the
    // page; `features/progress.js` exports a function and runs nothing on load.
    startWorkBar();
    schemaReady()
        .then(() => { applyLanguage(); fillAnalysisTypes(); })
        .catch(error => console.error('schema:', error));

    const rotors = restoreState();
    if (rotors) {
        console.info('state restored: ' + rotors + ' rotor(s)');
        // `restoreState` returns and does not draw -- the caller draws. This line was
        // missing when the dependency was inverted, and the result was the same defect
        // as the list hook: the rotors came back from memory and the Hub stayed empty
        // until the user touched something.
        renderRotorHub();
    }
    startPersistence();
});


// --- Bridge to the inline handlers of the HTML -----------------------------
//
// An ES module has its own scope: nothing is global unless someone puts it
// there. index.html and the HTML the cards generate call these functions by
// name, in `onclick`/`onchange` attributes. While that is so, they have to be on
// `window`, and this is the only place that puts them there -- before, there were
// 21 `window.x =` scattered through the file.
//
// The bridge is temporary by construction. The next slice replaces the inline
// handlers with event delegation, and each converted handler erases a name from
// here: the bridge shrinking is the measure of progress.
//
// `tests/js/test_bridge.js` guards both sides -- that every name called in a
// handler is here, and that nothing here has stopped being called.
Object.assign(window, {
    addAnalysis, addAngleProbeRow, addElementFromNodeHub, addForceRow,
    addProbeRow, addUnbalanceRow, changeLanguage, checkDeps, closeCustomAlert,
    closeCustomConfirm, closeCustomPrompt, closeForm, closeHelpModal,
    closeMultiRotorModal, closeNodeHub, confirmCustomPrompt, copyItem,
    copyRotorInHub, createNewRotorInHub, deleteAnalysis, deleteItem,
    deleteRotorInHub, editItem, editRotorName, exitApplication, fillDefault,
    generatePythonFile, generatePythonFromHub, handleUnitChange,
    loadAnalysis, loadAnalysisDirect, loadRotor, openAnalysisCardHelp,
    openAnalysisHelp, openForm, openGeneralHelp, openMultiRotorModal,
    openRotorHub, openRotorWorkspace, openSectionHelp, openTab, returnToHub,
    runCardAnalysis, saveAnalysis, saveItem, saveMultiRotor, saveRotor,
    saveRotorFromHub, selectSubType, switchMultiRotorTarget, switchScreen,
    toggleAdvanced, toggleAnalysis, toggleAnalysisSidebar, toggleDashAdv,
    toggleSidebar,
});
