// Entry point. No logic here: it wires the layers and listens to the page.
//
// The HTML talks through data attributes (`data-screen`, `data-tab`,
// `data-help`, ...) and this file is the only place that listens to them, so no
// function has to be published on `window` -- the direction the reference
// interface's bridge is moving towards.
import { closeHelpModal, openHelp } from './components/help.js';
import { closeCustomAlert, closeCustomConfirm, closeCustomPrompt, confirmCustomPrompt } from './components/modals.js';
import { applyLanguage, changeLanguage, restoreLanguage } from './core/i18n.js';
import { startBalance } from './features/balance.js';
import { startDigitalTwin } from './features/digital_twin.js';
import { cardAction, startAnalyses } from './features/dt_analyses.js';
import { exitApplication, openPanel, switchScreen, toggleCard, toggleSidebar } from './features/screens.js';

const CLICKS = [
    ['[data-card-action]', node => cardAction(node)],
    ['[data-screen]', node => switchScreen(node.dataset.screen)],
    ['[data-tab-group]', node => openPanel(node.dataset.tabGroup, node.dataset.tab)],
    ['[data-toggle-sidebar]', node => toggleSidebar(node.dataset.toggleSidebar)],
    ['[data-collapse]', node => toggleCard(node.dataset.collapse)],
    ['[data-help]', node => openHelp(node.dataset.help)],
    ['[data-close-help]', () => closeHelpModal()],
    ['[data-exit]', () => exitApplication()],
    ['[data-prompt-confirm]', () => confirmCustomPrompt()],
    ['[data-prompt-cancel]', () => closeCustomPrompt(null)],
    ['[data-confirm]', node => closeCustomConfirm(node.dataset.confirm === 'true')],
    ['[data-alert-close]', () => closeCustomAlert()],
];

document.addEventListener('DOMContentLoaded', () => {
    restoreLanguage();
    applyLanguage();
    startBalance();
    startDigitalTwin();
    startAnalyses();

    // The nearest marked element wins: a button inside a card header acts as
    // itself, not as the header that folds the card.
    const anyMarked = CLICKS.map(([selector]) => selector).join(',');
    document.body.addEventListener('click', event => {
        const node = event.target.closest(anyMarked);
        if (!node || node.disabled) return;
        const found = CLICKS.find(([selector]) => node.matches(selector));
        if (found) found[1](node);
    });
    document.querySelectorAll('select.ui-language').forEach(selector => {
        selector.addEventListener('change', () => changeLanguage(selector.value));
    });
});
