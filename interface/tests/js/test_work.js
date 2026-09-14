// The line of work as the screen sees it: who is computing, who is waiting, and
// who is allowed to say the work is over.
//
// WHY THIS BATTERY EXISTS. Slice 6c-1 made the screen stop blocking, and the
// first thing that came out of using it was a misreading: the Campbell and the
// crack both showed "updating" at the same time, so they looked like two
// analyses running at once. There is one worker and one thread -- they were
// taking turns. Everything here is about the difference between those two
// sentences, and about the bar that can only exist because something keeps the
// list in one place.
import { check, node, shutDown } from './fake_dom.js';

const { forgetWork, noteWork, watchWork, workNow } =
    await import('../../frontend/core/work.js');
const { describeWork } = await import('../../frontend/features/progress.js');
const { setSchemaLanguage } = await import('../../frontend/core/i18n.js');

// --- the registry -------------------------------------------------------------

const seen = [];
watchWork(work => seen.push(work.length));
check('a watcher is told what there is when it subscribes', seen.length === 1);
check('and there is nothing yet', seen[0] === 0);

const runA = {};
noteWork('card:1', runA, { state: 'running', waiting: 3 });
check('a job announces itself', workNow().length === 1);
check('and the watcher heard it', seen[seen.length - 1] === 1);

const runB = {};
noteWork('card:2', runB, { state: 'queued', waiting: 1, ahead: 1 });
check('two cards, two entries', workNow().length === 2);

forgetWork('card:2', runB);
check('a job that ended leaves', workNow().length === 1);

// --- the run token ------------------------------------------------------------
//
// A card run twice aborts its own previous request, and the loser unwinds
// *after* the winner has announced itself. Without an identity to compare, the
// loser's cleanup would delete the winner's entry: the bar would go quiet, and
// the only way to stop an analysis that was still running would be gone with it.

const older = {};
const newer = {};
noteWork('card:9', older, { state: 'queued', waiting: 0 });
noteWork('card:9', newer, { state: 'queued', waiting: 0 });
forgetWork('card:9', older);
check('the loser does not erase the winner', workNow().some(job => job.run === newer));

forgetWork('card:9', newer);
check('the winner can still end it', workNow().every(job => job.subject !== 'card:9'));

// --- what the bar says --------------------------------------------------------

setSchemaLanguage('en');
check('one of each is counted as one of each',
    describeWork([{ state: 'running' }, { state: 'queued' }]) === '1 computing, 1 in line');
check('nothing running is not reported as running',
    describeWork([{ state: 'queued' }, { state: 'queued' }]) === '2 in line');
check('nothing waiting is not reported as waiting',
    describeWork([{ state: 'running' }]) === '1 computing');
check('an empty line says nothing', describeWork([]) === '');

setSchemaLanguage('pt');
check('and it speaks the chosen language',
    describeWork([{ state: 'running' }]) === '1 calculando');

// --- what the card says, which is the whole reason for the slice --------------
//
// One sentence for two states is the defect. A card computing and a card
// waiting behind another card looked identical, so two analyses taking turns
// read as two analyses at once.

const { showProgress } = await import('../../frontend/features/analysis.js');

function cardSays(progress) {
    const what = node('loading-what-7');
    const when = node('loading-when-7');
    what.textContent = '';
    when.textContent = '';
    showProgress('7', progress);
    return [what.textContent, when.textContent];
}

setSchemaLanguage('en');
check('a card being computed says so',
    cardSays({ state: 'running', waiting: 12.4 })[0] === 'Computing');
check('and counts the wait in whole seconds',
    cardSays({ state: 'running', waiting: 12.4 })[1] === '12 s');
check('a card waiting behind another says a different thing',
    cardSays({ state: 'queued', waiting: 2, ahead: 1 })[0] === 'In line \u2014 1 ahead');
check('and it says how many, not just that there are some',
    cardSays({ state: 'queued', waiting: 2, ahead: 3 })[0] === 'In line \u2014 3 ahead');
check('a queued card with nobody in front is about to run, and says that',
    cardSays({ state: 'queued', waiting: 0, ahead: 0 })[0] === 'Computing');

setSchemaLanguage('pt');
check('the card speaks the chosen language too',
    cardSays({ state: 'queued', waiting: 1, ahead: 2 })[0] === 'Na fila \u2014 2 antes');

shutDown();
