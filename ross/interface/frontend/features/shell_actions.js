// The buttons that frame every screen: the home page, the hub's header and the
// three top bars. Phase 5, slice 12 -- the first stage of moving the page off
// the `window` bridge (see core/actions.js).
//
// Each entry is a thin adapter on purpose. The functions behind them were
// written to be called from an `onclick` with specific arguments --
// `generatePythonFile(project, analyses)` takes two optional ones, and would
// have received the button and the event -- so every action states exactly
// what it passes, and the element's own attributes carry anything it needs
// (`data-input`, `data-screen`).

import { openAbout } from '../components/about.js';
import { openAnalysisHelp, openGeneralHelp } from '../components/help.js';
import { toggleTheme } from '../core/theme.js';
import { loadAnalysis, loadAnalysisDirect, saveAnalysis } from './analysis.js';
import { openConcatenateModal } from './concatenate.js';
import { generatePythonFile } from './export.js';
import {
    copyRotorInHub, createNewRotorInHub, deleteRotorInHub, editRotorName, generatePythonFromHub,
    openRotorHub, openRotorWorkspace, saveRotorFromHub,
} from './hub.js';
import { changeLanguage, loadRotor, redoModelling, saveRotor, undoModelling } from './modeling.js';
import { openMultiRotorModal } from './multirotor.js';
import { exitApplication, switchScreen, toggleAnalysisSidebar, toggleSidebar } from './screens.js';

export const SHELL_ACTIONS = {
    // everywhere
    'about': () => openAbout(),
    'toggle-theme': () => toggleTheme(),
    'change-language': element => changeLanguage(element.value),
    'exit': () => exitApplication(),
    'export-python': () => generatePythonFile(),
    // A hidden file input is opened by a visible button; which input is the
    // button's `data-input`.
    'choose-file': element => document.getElementById(element.dataset.input).click(),

    // moving between screens
    'show-screen': element => switchScreen(element.dataset.screen),
    'open-hub': () => openRotorHub(),

    // the hub
    'new-rotor': () => createNewRotorInHub(),
    'open-multirotor': () => openMultiRotorModal(),
    'open-concatenate': () => openConcatenateModal(),
    'load-rotor': (element, event) => loadRotor(event),

    // a rotor's card on the hub; `data-index` is its place in the library
    'rename-rotor': element => editRotorName(Number(element.dataset.index)),
    'copy-rotor': element => copyRotorInHub(Number(element.dataset.index)),
    'delete-rotor': element => deleteRotorInHub(Number(element.dataset.index)),
    'open-rotor': element => openRotorWorkspace(Number(element.dataset.index), element.dataset.screen),
    'save-rotor-file': element => saveRotorFromHub(Number(element.dataset.index)),
    'export-rotor-python': element => generatePythonFromHub(Number(element.dataset.index)),

    // modelling
    'toggle-sidebar': () => toggleSidebar(),
    'help-modelling': () => openGeneralHelp(),
    'undo': () => undoModelling(),
    'redo': () => redoModelling(),
    'save-rotor': (element, event) => saveRotor(event),

    // analysis
    'toggle-analysis-sidebar': () => toggleAnalysisSidebar(),
    'help-analysis': () => openAnalysisHelp(),
    'save-analysis': (element, event) => saveAnalysis(event),
    'load-analysis': (element, event) => loadAnalysis(event),
    'load-analysis-direct': (element, event) => loadAnalysisDirect(event),
};
