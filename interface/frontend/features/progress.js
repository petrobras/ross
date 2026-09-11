// The line of work, and the only button that can stop it.
//
// WHY THERE IS A BAR AT ALL. Slice 6c-1 made the screen stop blocking, and the
// first thing that came back from using it was a misreading: two cards said
// "updating" at the same time, so two analyses looked like they were running at
// the same time. There is one worker and one thread -- they take turns. The
// cards now say which of the two states they are in, and this bar says what the
// line as a whole is doing, because a card can only ever speak for itself.
//
// WHY ONE BUTTON AND NOT ONE PER CARD. Stopping a computation that is already
// inside ROSS means killing the process, and every card's work lives in that one
// process. A button inside a card would look like it belonged to that card and
// would silently take the others with it. One button, one scope, and the scope
// is written on it.
//
// WHY IT ASKS FIRST. The price is real and measured: the next analysis pays
// eight to twelve seconds to import ROSS again, plus up to sixteen of numba on
// the campbell and the crack. A cost that big is the user's decision, so it is
// quoted in the confirmation and not discovered afterwards.

import { openCustomAlert, openCustomConfirm } from '../components/modals.js';
import { apiFetch } from '../core/api.js';
import { t } from '../core/i18n.js';
import { watchWork } from '../core/work.js';

export function startWorkBar() {
    const bar = document.getElementById('work-bar');
    if (!bar) return;
    const stop = document.getElementById('work-bar-stop');
    if (stop) stop.addEventListener('click', interruptEverything);
    watchWork(work => drawWorkBar(bar, work));
}

// Exported so the suite can drive it with a list instead of a browser.
export function describeWork(work) {
    const running = work.filter(job => job.state === 'running').length;
    const queued = work.length - running;
    const parts = [];
    if (running) parts.push(running + ' ' + t('jobComputing').toLowerCase());
    if (queued) parts.push(queued + ' ' + t('jobInLine').toLowerCase());
    return parts.join(', ');
}

function drawWorkBar(bar, work) {
    if (!work.length) {
        bar.style.display = 'none';
        return;
    }
    bar.style.display = 'flex';
    const text = document.getElementById('work-bar-text');
    if (text) text.textContent = describeWork(work);
}

async function interruptEverything() {
    // The price, before it is spent. `openCustomConfirm` resolves false on
    // anything that is not a yes, which is the answer this question should have
    // by default.
    if (!await openCustomConfirm(t('workStopPrice'))) return;
    try {
        const answer = await apiFetch('/api/worker/interrupt', { method: 'POST' })
            .then(response => response.json());
        // The one case where the button cannot keep its promise: with no worker
        // the analysis is running in the server process itself, and there is
        // nothing to kill without closing the interface. Nothing will be drawn
        // -- the jobs are cancelled -- but the computation does finish, and the
        // machine stays busy until it does. Saying so is cheaper than letting
        // somebody wonder why the fan is still spinning.
        if (answer && answer.was_running && !answer.worker_killed) {
            openCustomAlert(t('workStopNoWorker'));
        }
    } catch (error) {
        openCustomAlert(t('serverConnectionError'));
    }
}
