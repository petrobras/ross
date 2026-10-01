// Entry point.
//
// No logic here: it wires the layers, names what the buttons mean and fires
// the bootstrap. Everything else lives in `core/`, `components/` and
// `features/`.
//
// Nothing is published on `window`. Until slice 15 this file ended in a
// bridge block that hung on it every function an inline `onclick` of the HTML
// called (73 names at its largest); every button now says `data-action="..."`
// and one listener looks the name up (core/actions.js).
import { onReorder } from './components/list.js';
import { applyLanguage } from './core/i18n.js';
import { startPersistence, restoreState } from './core/persistence.js';
import { schemaReady } from './core/schema.js';
import { onProjectChanged } from './core/state.js';
import { fillAnalysisTypes } from './features/analysis.js';
import { renderRotorHub } from './features/hub.js';
import { buildRotorLive, refreshHistoryButtons, startRotorFigureFollowsWidth } from './features/modeling.js';
import { startHistoryShortcuts } from './features/shortcuts.js';
import { startWorkBar } from './features/progress.js';
import { startTheme } from './core/theme.js';
import { startErrorNotice } from './features/error_notice.js';
import { defineActions, startActions } from './core/actions.js';
import { SHELL_ACTIONS } from './features/shell_actions.js';
import { MODELING_ACTIONS } from './features/modeling_actions.js';
import { ANALYSIS_ACTIONS } from './features/analysis_actions.js';
import { DIALOG_ACTIONS } from './features/dialog_actions.js';

// --- Wiring between layers -------------------------------------------------
//
// The element list does not know the rotor: it announces that the order changed,
// and whoever knows how to rebuild subscribes here. It was this inversion that
// removed the only path from a component to a feature -- and it was the one I
// created and forgot to connect, leaving dragging with no effect on the figure.
onReorder(buildRotorLive);

// The buttons grey out when there is nothing left to undo. The reason this is a
// hook and not a call from core/state.js is written there.
onProjectChanged(refreshHistoryButtons);

// What the buttons mean, by name (core/actions.js). One table per area of the
// page.
defineActions(SHELL_ACTIONS);
defineActions(MODELING_ACTIONS);
defineActions(ANALYSIS_ACTIONS);
defineActions(DIALOG_ACTIONS);

// The schema is loaded once at startup; openForm waits for it.
document.addEventListener('DOMContentLoaded', () => {
    // First, so that a failure in anything below is seen on screen.
    startErrorNotice();
    startActions();
    // The theme first: the head script already set the attribute before the first
    // paint, and this wires the buttons and the system's preference to it.
    startTheme();
    // The page is born in English in the HTML; the chosen language is applied as
    // soon as the schema arrives, along with the analysis titles that come with it.
    applyLanguage();
    // The bar that says what is being computed and offers the only way to stop
    // it. Here and not at module level because it takes hold of elements of the
    // page; `features/progress.js` exports a function and runs nothing on load.
    startWorkBar();
    // Ctrl+Z / Ctrl+Y: takes hold of the document, like the other `start*`.
    startHistoryShortcuts();
    // The rotor figure follows the width without losing ROSS's height.
    startRotorFigureFollowsWidth();
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
