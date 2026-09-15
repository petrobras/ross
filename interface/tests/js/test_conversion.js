// Under which rotor model each card is computed, and what the header announces.
//
// The defect: after reloading the page, three cards of the same rotor -- one in
// 6 DoF, one in 4 DoF and one torsional -- all recomputed as 6 DoF.
// `runCardAnalysis` re-read the screen's "Rotor Model" selector on every
// computation, and the selector goes back to its default when the page loads.
// The same selector had already produced the export defect in the previous
// slice; it was the third place reading from the screen a value that belongs
// to the analysis.
//
// Along with it, the second symptom of the same cause: cards rebuilt from
// memory came with no badge, because the badge only existed in the HTML that
// `addAnalysis` wrote -- and "no badge" was indistinguishable from "6 DoF".
import { node, clearDom } from './fake_dom.js';
import { t } from '../../frontend/core/i18n.js';
import {
    ANALYSES, recordResult, analysesToSave, cardConversion,
    forgetAllAnalyses, registerAnalysis, conversionBadge,
} from '../../frontend/core/analysis_store.js';

const selector = node('rotor-conversion-type');

let ok = 0, failed = 0;
function check(description, condition) {
    if (condition) { ok++; console.log('  ok      ' + description); }
    else { failed++; console.log('  FAILED  ' + description); }
}
function clearAll() { forgetAllAnalyses(); selector.value = ''; }

// --- the badge ---------------------------------------------------------------
console.log('\nThe rotor model badge');

check('6 DoF has a badge', /6 DoF/.test(conversionBadge('')));
check('4 DoF has a badge', /4 DoF/.test(conversionBadge('4dof')));
check('torsional has a badge', /Torsional/.test(conversionBadge('torsional')));

check('6 DoF has a class of its own', /badge-6dof/.test(conversionBadge('')));
check('4 DoF has a class of its own', /badge-4dof/.test(conversionBadge('4dof')));
check('torsional keeps the old class',
          /badge-torsional/.test(conversionBadge('torsional')));

check('all three come out with the base class',
          ['', '4dof', 'torsional'].every(c => /badge-conversion/.test(conversionBadge(c))));

// An empty badge would pass "the three are different" further down without
// anything written inside it.
check('every badge has visible text',
          ['', '4dof', 'torsional'].every(c => />[^<]+</.test(conversionBadge(c))));

check('undefined means 6 DoF, as in the empty record',
          conversionBadge(undefined) === conversionBadge(''));

// The tooltip comes from the language dictionary, and not from an English
// literal written in the middle of the HTML -- which is how it was born.
check('the tooltip comes from the dictionary',
          conversionBadge('4dof').includes('title="' + t('conv4dof') + '"'));

// A future value nobody put in the map must not become "6 DoF" by omission;
// which is what would happen if the function gave back ''.
check('an unknown model still shows up',
          conversionBadge('8dof').includes('8dof'));

check('an unknown model does not become 6 DoF',
          !/6 DoF/.test(conversionBadge('8dof')));

check('the badge escapes what came from outside',
          !conversionBadge('<img>').includes('<img>'));

// --- who the card asks -------------------------------------------------------
console.log('\nWhere the conversion of a computation comes from');

clearAll();
registerAnalysis('a1', 'modes', 'Modal Analysis', '4dof');
selector.value = 'torsional';
check('the record decides, not the selector', cardConversion('a1') === '4dof');

clearAll();
selector.value = '4dof';
check('with no record, the selector answers for the card about to be born',
          cardConversion('fresh') === '4dof');

clearAll();
registerAnalysis('a1', 'modes', 'Modal Analysis', '');
selector.value = 'torsional';
check('a 6 DoF card stays in 6 DoF', cardConversion('a1') === '');

// --- the reported regression ------------------------------------------------
console.log('\nThree cards, three models, one reload');

clearAll();
// What memory keeps of each analysis, and what restoring does with it.
const saved = [
    { title: 'Modal Analysis', type: 'modes', params: { num_modes: '4' }, conversion: '' },
    { title: 'Modal Analysis', type: 'modes', params: { num_modes: '4' }, conversion: '4dof' },
    { title: 'Modal Analysis', type: 'modes', params: { num_modes: '4' }, conversion: 'torsional' },
];
const ids = ['c0', 'c1', 'c2'];
saved.forEach((an, i) => {
    registerAnalysis(ids[i], an.type, an.title, an.conversion);
    recordResult(ids[i], an.params, an.conversion, null);
});

// The page has just loaded: the selector is at its default.
selector.value = '';
check('the torsional card recomputes torsional', cardConversion('c2') === 'torsional');
check('the 4 DoF card recomputes 4 DoF', cardConversion('c1') === '4dof');
check('the 6 DoF card recomputes 6 DoF', cardConversion('c0') === '');

// Control: if the conversion were not coming from the record, all three would
// give the same value -- which is exactly the reported defect. Without this
// control the battery above would pass with a `cardConversion` returning ''.
selector.value = 'torsional';
check('control: with the selector on torsional the three stay different',
          new Set(ids.map(cardConversion)).size === 3);

check('the three cards come back with different badges',
          new Set(saved.map(an => conversionBadge(an.conversion))).size === 3);

// And what memory gives back really carries the conversion: if `analysesToSave`
// lost the field, the restoration above would be a fiction of the test.
check('the conversion survives what is saved',
          analysesToSave().map(a => a.conversion).sort().join(',') === ',4dof,torsional');

console.log('\n' + ok + ' checks ok, ' + failed + ' failed');
process.exit(failed ? 1 : 0);
