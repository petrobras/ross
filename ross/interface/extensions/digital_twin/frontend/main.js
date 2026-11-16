// Entry point of the Digital Twin extension. The ROSS interface loads this
// module into its own page (see ross/interface/api/extensions.py), and the
// screens of `screens.html` are already in the document when it runs.
//
// Everything is wired through the interface's own mechanisms: the buttons say
// `data-action="..."` and the names are defined here, the texts go into its
// dictionary, and the help into its help modal. Nothing is published on
// `window`.
import { addHelp, openSectionHelp } from '/components/help.js';
import { defineActions } from '/core/actions.js';
import { addTexts } from '/core/i18n.js';
import { startBalance } from './features/balance.js';
import { startDigitalTwin } from './features/digital_twin.js';
import { addSelectedCard, CARD_ACTIONS, startAnalyses } from './features/dt_analyses.js';
import { openPanel, toggleCard, toggleSidebarById } from './features/panels.js';
import { HELP_ENTRIES } from './help.js';
import { TEXTS } from './texts.js';

addTexts(TEXTS);
addHelp(HELP_ENTRIES);

defineActions({
    'dt-panel': element => openPanel(element.dataset.tabGroup, element.dataset.tab),
    'dt-toggle-sidebar': element => toggleSidebarById(element.dataset.target),
    'dt-collapse': element => toggleCard(element.dataset.target),
    'dt-help': element => openSectionHelp(element.dataset.help),
    'dt-add-card': () => addSelectedCard(),
    ...CARD_ACTIONS,
});

document.addEventListener('DOMContentLoaded', () => {
    startBalance();
    startDigitalTwin();
    startAnalyses();
});
