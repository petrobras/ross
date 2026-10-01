// Saying so when something breaks.
//
// WHY THIS EXISTS. Until now there was no `window.onerror` and no handler for
// rejected promises anywhere in the frontend. An exception inside a click
// handler went to the console -- which nobody using the interface has open --
// and the screen simply stopped where it was. That is how a failure reached
// Leonardo as "the element lists vanished": the list had been emptied and the
// error that interrupted its rebuilding was never seen.
//
// This does not recover anything. It makes the failure visible, with the one
// line that says where it happened, so that the next report comes as a message
// rather than as a symptom.

import { wasCancelled } from '../core/api.js';
import { t } from '../core/i18n.js';

// Two things that arrive as errors and are not failures.
//
// A cancelled request is how `apiFetchLatest` drops an answer nobody wants any
// more: a newer request on the same subject took its place. And Chrome reports
// "ResizeObserver loop" as a window error when Plotly resizes inside a resize
// callback; it is a warning that no frame was lost, and every figure on the
// analysis screen can raise it.
export function isNoise(error) {
    if (wasCancelled(error)) return true;
    const message = String((error && error.message) || error || '');
    return message.includes('ResizeObserver loop');
}

// The line shown on screen. Exported so the suite can check it without a page.
//
// The message, and where it came from when the browser says: the file name and
// the line. The file is cut to its last part -- `modeling.js:412` is what
// someone can read out loud; the whole URL is not.
export function describeError(error, file, line) {
    const message = (error && error.message) || String(error || '') || '?';
    const where = file ? String(file).split(/[\\/]/).pop().split('?')[0] : '';
    return where ? message + ' (' + where + (line ? ':' + line : '') + ')' : message;
}

let shown = 0;

function report(error, file, line) {
    if (isNoise(error)) return;
    const notice = document.getElementById('error-notice');
    if (!notice) return;
    shown++;
    const text = document.getElementById('error-notice-text');
    // `textContent`, never `innerHTML`: an error message can quote whatever the
    // user typed into a field.
    if (text) {
        text.textContent = describeError(error, file, line)
            + (shown > 1 ? ' — ' + t('errorNoticeMore').replace('%1', () => shown) : '');
    }
    notice.style.display = 'flex';
}

function dismiss() {
    const notice = document.getElementById('error-notice');
    if (notice) notice.style.display = 'none';
    shown = 0;
}

export function startErrorNotice() {
    window.addEventListener('error', event => report(event.error || event.message, event.filename, event.lineno));
    window.addEventListener('unhandledrejection', event => report(event.reason));
    const close = document.getElementById('error-notice-close');
    if (close) close.addEventListener('click', dismiss);
}
