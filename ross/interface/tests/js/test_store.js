// The analysis state: before Phase 3 this lived in properties hung off DOM
// nodes (`div.rossParams`, `div.rossType`, `div.rossFrames`), and the title was
// read back from the header's innerText. An analysis whose card did not render
// vanished from what was saved, with no warning.
import './fake_dom.js';
import {
    ANALYSES, analysesInScreenOrder, analysesToSave, recordResult,
    collectActiveAnalyses, forgetAnalysis, forgetAllAnalyses,
    registerAnalysis, hasChart,
} from '../../frontend/core/analysis_store.js';

let ok = 0, failed = 0;
function check(description, condition) {
    if (condition) { ok++; console.log('  ok      ' + description); }
    else { failed++; console.log('  FAILED  ' + description); }
}

const FIG = { data: [{ x: [1, 2] }], layout: { title: 'x' }, frames: [{ n: 1 }] };

// 1) registering stores what the screen needs, without depending on the screen
forgetAllAnalyses();
const record = registerAnalysis('a1', 'campbell', 'Campbell Diagram', '4dof');
check('the record is born with type, title and conversion',
    record.type === 'campbell' && record.title === 'Campbell Diagram'
    && record.conversion === '4dof');
check('and with no figure yet', record.figure === null);

// 2) recording the result stores parameters and figure
recordResult('a1', { speed_max: '4000' }, '4dof', FIG);
check('the parameters stay in the record',
    ANALYSES.get('a1').params.speed_max === '4000');
check('the figure too', ANALYSES.get('a1').figure.data.length === 1);
check('the frames included', ANALYSES.get('a1').figure.frames.length === 1);

// 3) the parameters are copied, not referenced
const params = { speed_max: '4000' };
recordResult('a1', params, '', FIG);
params.speed_max = '9999';
check('touching the original object does not change the record',
    ANALYSES.get('a1').params.speed_max === '4000');

// 4) screen order is the reverse of creation (cards enter with afterbegin)
forgetAllAnalyses();
registerAnalysis('first', 'campbell', 'A', '');
registerAnalysis('second', 'ucs', 'B', '');
registerAnalysis('third', 'modes', 'C', '');
check('the screen order runs from newest to oldest',
    analysesInScreenOrder().map(r => r.id).join(',') === 'third,second,first');

// 5) an analysis that never computed keeps its configuration
recordResult('third', { num_modes: '12' }, '', null);
const toSave = analysesToSave();
check('with no figure, the configuration survives',
    toSave[0].type === 'modes' && toSave[0].params.num_modes === '12');
check('and the chart comes out empty, instead of losing the whole analysis',
    toSave[0].data.length === 0 && toSave.length === 3);

// 6) the title comes from the record, not back from the screen
check('the title is the one that was registered', toSave[0].title === 'C');

// 7) forgetting takes it out of the state
forgetAnalysis('second');
check('forgetting removes it', ANALYSES.has('second') === false);
check('and does not touch the others', ANALYSES.size === 2);

// 8) a figure with no traces is not a figure
forgetAllAnalyses();
registerAnalysis('empty', 'campbell', 'A', '');
recordResult('empty', { speed_max: '4000' }, '', { data: [], layout: {}, frames: [] });
check('a figure with no traces does not count as a chart',
    hasChart(ANALYSES.get('empty')) === false);
check('and the real one does',
    hasChart(recordResult('empty', {}, '', FIG)) === true);

// 9) the export follows the configuration, not the drawing
//
// A card restored from another session lost its figure and kept its
// parameters. Filtering by figure would leave it out of the exported script
// -- precisely the analysis the user has just seen reappear on screen.
forgetAllAnalyses();
registerAnalysis('computed', 'campbell', 'A', '');
recordResult('computed', { speed_max: '4000' }, '', FIG);
registerAnalysis('restored', 'modes', 'B', '');
recordResult('restored', { num_modes: '12' }, '', null);   // no figure
registerAnalysis('fresh', 'ucs', 'C', '');                       // never configured
const toExport = collectActiveAnalyses();
check('the computed analysis is in',
    toExport.some(a => a.type === 'campbell'));
check('the restored one with no figure too',
    toExport.some(a => a.type === 'modes'));
check('the one never configured stays out',
    toExport.every(a => a.type !== 'ucs') && toExport.length === 2);

// 10) recording on an id that does not exist invents no record
check('recording on an unknown id gives back null',
    recordResult('does-not-exist', {}, '', FIG) === null);
check('and creates no entry', ANALYSES.has('does-not-exist') === false);

// 11) clearing wipes everything
forgetAllAnalyses();
check('clearing empties it', ANALYSES.size === 0 && analysesToSave().length === 0);

console.log(`\n${ok} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
