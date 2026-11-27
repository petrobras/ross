// Every trip to the server goes through here: the session token, cancelling the
// previous request on the same subject, and pruning the project before sending.
//
// Flask serves this page, so the calls are same-origin (relative URL). The token
// is injected into index.html by the server and travels on every request.

import { forgetWork, noteWork } from './work.js';

export function apiFetch(path, options = {}) {
    const headers = Object.assign({}, options.headers || {}, {
        'X-ROSS-Token': window.ROSS_TOKEN || ''
    });
    return fetch(path, Object.assign({}, options, { headers }));
}

// One request per subject: a new one cancels the previous.
// Without this, dragging the slider fires several and the answer to an old
// configuration can arrive after the new one, overwriting the right chart -- the
// user sees a result that does not match the controls on screen.
const REQUESTS_IN_FLIGHT = new Map();

// The backend only needs what describes the rotor. `savedAnalyses` holds the
// already-rendered Plotly figures of each saved analysis, a bundle not worth
// resending on every keystroke in the form only to get a chart back -- and the
// server keys its cache by what it receives.
export function projectForServer(project) {
    if (!project || typeof project !== 'object') return {};
    const copy = Object.assign({}, project);
    delete copy.savedAnalyses;
    ['driving_rotor', 'driven_rotor'].forEach(key => {
        if (copy[key]) copy[key] = projectForServer(copy[key]);
    });
    return copy;
}

export function apiFetchLatest(subject, path, options = {}) {
    const previous = REQUESTS_IN_FLIGHT.get(subject);
    if (previous) previous.abort();

    const controller = new AbortController();
    REQUESTS_IN_FLIGHT.set(subject, controller);

    return apiFetch(path, Object.assign({}, options, { signal: controller.signal }))
        .finally(() => {
            if (REQUESTS_IN_FLIGHT.get(subject) === controller) {
                REQUESTS_IN_FLIGHT.delete(subject);
            }
        });
}

export function wasCancelled(error) {
    return !!error && error.name === 'AbortError';
}

// --- asking for work that takes a while --------------------------------------
//
// An analysis is not answered by the request that asks for it.
// The server puts the work in a line, answers with the name of the job, and
// this function asks about that name until there is a chart.
//
// Three things follow. A job that has not started can leave the queue when the
// same card asks again -- aborting a `fetch` alone would leave the computation
// running to the end, so a card run twice would make the second analysis wait
// for the first, which nobody wants any more. The screen stops being held by a request it is not showing.
// And a chart can only land in the card that asked for it, because the answer
// carries the name of the question.
//
// The subject is sent as `key`: it is what the server uses to take the older
// job out of the line, and it is the same string `apiFetchLatest` uses to abort
// the older request here. One idea, one name, both sides.
//
// Since 6c-2 every state the job passes through is announced twice: to the
// caller through `onProgress`, which is how the card tells "computing" from
// "third in line", and to `core/work.js`, which is how the bar at the bottom of
// the page knows there is anything to interrupt. Announcing and not returning,
// because these arrive while the answer is still unknown.
const POLL_INTERVAL_MS = 250;

const pause = ms => new Promise(resolve => setTimeout(resolve, ms));

export async function runJob(subject, path, body, onProgress) {
    const run = {};   // identity of this attempt; see core/work.js
    try {
        const accepted = await apiFetchLatest(subject, path, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(Object.assign({}, body, { key: subject }))
        }).then(response => response.json());

        // A body the envelope refused never becomes a job, and the caller shows
        // that message exactly as it did before there was a queue.
        if (!accepted || accepted.status !== 'accepted') return accepted;

        for (;;) {
            noteWork(subject, run, accepted);
            if (onProgress) onProgress(accepted);
            const state = await apiFetchLatest(subject, `/api/jobs/${accepted.job_id}`)
                .then(response => response.json());
            if (!state || state.status !== 'working') return state;
            Object.assign(accepted, state);
            await pause(POLL_INTERVAL_MS);
        }
    } finally {
        // Also on an abort and on a thrown request: a bar still offering to
        // interrupt work that ended is worse than no bar at all.
        forgetWork(subject, run);
    }
}
