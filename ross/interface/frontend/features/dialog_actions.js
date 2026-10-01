// The buttons of the dialogs: About, the help, the interface's own prompt /
// confirm / alert, and the two hub modals that build a rotor out of two
// (MultiRotor and concatenation). Phase 5, slice 15 -- the last stage of
// moving the page off the `window` bridge (see core/actions.js).
//
// The prompt, confirm and alert used to pass their answer written into the
// attribute (`closeCustomConfirm(true)`). Here each answer is its own name --
// `confirm-yes`, `confirm-no` -- because a dataset value is a string, and a
// `data-answer="false"` read as a truthy "false" would be a confirmation.
//
// The two rotor selects of each hub modal are fields: they act on `change`,
// like every other select of the page.

import { closeAbout } from '../components/about.js';
import { closeHelpModal } from '../components/help.js';
import {
    closeCustomAlert, closeCustomConfirm, closeCustomPrompt, confirmCustomPrompt,
} from '../components/modals.js';
import {
    closeConcatenateModal, describeJoint, saveConcatenation, swapConcatenationOrder,
} from './concatenate.js';
import { closeMultiRotorModal, describeCoupling, saveMultiRotor } from './multirotor.js';

export const DIALOG_ACTIONS = {
    // About and the help
    'close-about': () => closeAbout(),
    'close-help': () => closeHelpModal(),

    // the interface's own prompt, confirm and alert
    'prompt-ok': () => confirmCustomPrompt(),
    'prompt-cancel': () => closeCustomPrompt(null),
    'confirm-yes': () => closeCustomConfirm(true),
    'confirm-no': () => closeCustomConfirm(false),
    'alert-ok': () => closeCustomAlert(),

    // linking a MultiRotor
    'describe-coupling': () => describeCoupling(),
    'save-multirotor': () => saveMultiRotor(),
    'close-multirotor': () => closeMultiRotorModal(),

    // concatenating
    'describe-joint': () => describeJoint(),
    'swap-concatenation': () => swapConcatenationOrder(),
    'save-concatenation': () => saveConcatenation(),
    'close-concatenate': () => closeConcatenateModal(),
};
