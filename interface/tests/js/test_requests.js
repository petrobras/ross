// A new request cancels the previous one on the same subject.
//
// This was defect FE-05 of the audit: with no cancellation, dragging a slider
// fires several calls and the answer to an old configuration can arrive after
// the new one, overwriting the right chart. A wrong result, with no error at
// all on screen.
//
// The battery had existed since Phase 2 and **died in the move to modules**: it
// read `frontend/app.js` with `require()`, and the file stopped existing while
// the `package.json` here became `"type": "module"`. What followed was a light
// year of silence: `tests/test_js_dom.py` collected it, `node` blew up on the
// first line, and the node `skipif` on Windows hid all of it. The coverage of a
// critical defect already fixed vanished with nobody seeing.
import { check, shutDown } from './fake_dom.js';

// --- a `fetch` that obeys the signal and never resolves by itself ------------
const calls = [];

globalThis.fetch = (path, options) => new Promise((resolve, reject) => {
    const record = { path, cancelled: false };
    calls.push(record);

    options.signal.addEventListener('abort', () => {
        record.cancelled = true;
        const error = new Error('aborted');
        error.name = 'AbortError';
        reject(error);
    });

    record.answerWith = () => resolve({ ok: true, status: 200, json: async () => ({}) });
});

const { apiFetch, apiFetchLatest, wasCancelled, projectForServer } =
    await import('../../frontend/core/api.js');

// --- the cancellation ---------------------------------------------------------

const first = apiFetchLatest('rotor', '/build_rotor', { method: 'POST' });
first.catch(() => {});   // will be cancelled: without this node complains
check('the first went out', calls.length === 1);
check('and is still in flight', calls[0].cancelled === false);

const second = apiFetchLatest('rotor', '/build_rotor', { method: 'POST' });
second.catch(() => {});
check('the second went out', calls.length === 2);
check('and the first was cancelled', calls[0].cancelled === true);
check('the second was not', calls[1].cancelled === false);

// A different subject does not cancel: the rotor figure and the analysis figure
// are independent requests, and cancelling one for the other would wipe the
// wrong part of the screen.
const other = apiFetchLatest('analysis', '/run_analysis', { method: 'POST' });
other.catch(() => {});
check('a different subject does not cancel', calls[1].cancelled === false);
check('and goes out all the same', calls.length === 3);

// --- the caller can tell cancellation from failure ---------------------------

let errorOfTheFirst = null;
try { await first; } catch (e) { errorOfTheFirst = e; }
check('the cancelled promise rejects', errorOfTheFirst !== null);
check('and `wasCancelled` recognises it', wasCancelled(errorOfTheFirst) === true);
check('without mistaking it for a network error', wasCancelled(new TypeError('failed')) === false);
check('nor for nothing at all', wasCancelled(null) === false);

// --- the register cleans itself ----------------------------------------------
//
// If the map kept the controller of a request already finished, the next one on
// the same subject would try to abort a dead controller.

calls[2].answerWith();
await other;
const after = apiFetchLatest('analysis', '/run_analysis', { method: 'POST' });
after.catch(() => {});
check('a finished one does not cancel the next', calls[2].cancelled === false);

// --- the token travels on every call -----------------------------------------

globalThis.window = globalThis;
globalThis.ROSS_TOKEN = 'abc123';
let headers = null;
globalThis.fetch = async (path, options) => { headers = options.headers; return { ok: true }; };
await apiFetch('/anything');
check('the session token travels in the header', headers['X-ROSS-Token'] === 'abc123');

// --- pruning the project before sending --------------------------------------
//
// `savedAnalyses` holds the already-rendered Plotly figures. Until Phase 2 the
// interface resent that whole bundle on every keystroke.

const pruned = projectForServer({
    shafts: [{ L: '500' }],
    savedAnalyses: [{ data: 'huge figure' }],
    driving_rotor: { shafts: [], savedAnalyses: [{ data: 'other' }] },
});
check('pruning: the saved analyses do not go up', pruned.savedAnalyses === undefined);
check('pruning: the rotor stays whole', pruned.shafts.length === 1);
check('pruning: it goes down into multirotors', pruned.driving_rotor.savedAnalyses === undefined);
check('pruning: it invents no object', Object.keys(projectForServer(null)).length === 0);

// --- a job: asked for once, asked about until there is a chart ----------------
//
// Since slice 6c the analysis is not answered by the request that asks for it.
// What the screen used to do -- one fetch, one chart -- is now submit, then ask
// about the job by name. Three properties matter and each has a way of failing
// quietly: the payload must still arrive whole, the subject must travel as the
// `key` the server cancels by, and a job the server dropped must come back as
// dropped instead of as an error on screen.

const trips = [];
let polls = 0;

globalThis.fetch = async (path, options = {}) => {
    trips.push({ path, body: options.body ? JSON.parse(options.body) : null });
    if (path === '/run_analysis') {
        return { ok: true, json: async () => ({ status: 'accepted', job_id: 'j7' }) };
    }
    polls += 1;
    if (polls === 1) {
        return { ok: true, json: async () => ({ status: 'working', state: 'queued' }) };
    }
    return { ok: true, json: async () => ({ status: 'success', plot_json: '{}' }) };
};

const { runJob } = await import('../../frontend/core/api.js');

const chart = await runJob('card:3', '/run_analysis', { analysis_type: 'static' });
check('the chart arrives when the job is done', chart.status === 'success');
check('the work was asked for once', trips[0].path === '/run_analysis');
check('the payload still goes up whole', trips[0].body.analysis_type === 'static');
check('the subject travels as the key the server drops by', trips[0].body.key === 'card:3');
check('the job is asked about by name', trips[1].path === '/api/jobs/j7');
check('and asked about until it stopped working', polls === 2);

// A job the server took out of the line: the card belongs to a newer request,
// and this one has to go quiet rather than draw an error over it.
trips.length = 0;
globalThis.fetch = async (path, options = {}) => {
    trips.push({ path });
    if (path === '/run_analysis') {
        return { ok: true, json: async () => ({ status: 'accepted', job_id: 'j8' }) };
    }
    return { ok: true, json: async () => ({ status: 'superseded' }) };
};
const dropped = await runJob('card:3', '/run_analysis', {});
check('a dropped job says so', dropped.status === 'superseded');

// A body the envelope refused never becomes a job, and the message still
// reaches the caller exactly as it did before there was a queue.
trips.length = 0;
globalThis.fetch = async (path) => {
    trips.push({ path });
    return { ok: false, json: async () => ({ status: 'error', message: 'Unknown field' }) };
};
const refused = await runJob('card:3', '/run_analysis', {});
check('a refused body keeps its message', refused.message === 'Unknown field');
check('and is never asked about as a job', trips.length === 1);

// --- the progress of a job reaches both the card and the bar (slice 6c-2) -----
//
// Announced and not returned, because these arrive while the answer is still
// unknown. Two listeners, because they need different things: the card wants
// *its* position, and the bar -- the only place that can offer to stop the work
// -- wants to know that there is any work at all.
const { workNow } = await import('../../frontend/core/work.js');

trips.length = 0;
globalThis.fetch = async (path) => {
    trips.push({ path });
    if (path === '/run_analysis') {
        return { ok: true, json: async () => (
            { status: 'accepted', job_id: 'j9', state: 'queued', waiting: 0, ahead: 2 }) };
    }
    if (trips.length < 3) {
        return { ok: true, json: async () => ({ status: 'working', state: 'running', waiting: 5 }) };
    }
    return { ok: true, json: async () => ({ status: 'success', plot_json: '{}' }) };
};

const told = [];
const barHeld = [];
const finished = await runJob('card:7', '/run_analysis', {}, progress => {
    told.push({ state: progress.state, ahead: progress.ahead });
    barHeld.push(workNow().length);
});
check('the job finished', finished.status === 'success');
check('the card was told it is waiting, and behind how many', told[0].ahead === 2);
check('and then that it is being computed', told[1].state === 'running');
check('the bar knew there was work while there was', barHeld[0] === 1);
check('and the bar is empty once the work is over', workNow().length === 0);

// A body the envelope refused never becomes a job -- and must not leave the bar
// offering to interrupt something that never started.
trips.length = 0;
globalThis.fetch = async (path) => {
    trips.push({ path });
    return { ok: false, json: async () => ({ status: 'error', message: 'Unknown field' }) };
};
await runJob('card:7', '/run_analysis', {});
check('a refused body leaves nothing behind in the bar', workNow().length === 0);

shutDown();
