// Ticking several elements, and doing one thing to all of them.
//
// WHY THIS BATTERY EXISTS. An element of this project has no identity -- rotors
// have a `uid`, shafts and disks have nothing stable -- so a selection can only
// be a set of **positions**. Positions are the most perishable thing in the
// interface: they change when an element is deleted, copied, split or dragged,
// when the tab changes, when the other half of a MultiRotor is shown, and when
// an undo puts a different model on screen.
//
// A selection that outlives any of those does not fail loudly. It deletes the
// wrong elements and leaves a shorter list, which looks exactly like a list you
// shortened on purpose. That is what is checked here.
import { check, node, shutDown } from './fake_dom.js';

const { allPicked, clearSelection, isPicked, nowShowing, pick, pickAll, picked, pickedCount } =
    await import('../../frontend/core/selection.js');
const { listContext, onProjectChanged, state, syncBackToLibrary } =
    await import('../../frontend/core/state.js');

function rotor(names) {
    return {
        name: 'Compressor',
        uid: 'uid_compressor',
        savedAnalyses: [],
        materials: [],
        shafts: names.map(tag => ({ element_type: 'BASIC', L: '250', odl: '50', tag })),
        disks: [{ element_type: 'BASIC', n: '1', m: '10' }],
        gears: [],
        couplings: [],
        seals: [],
        bearings: [{ element_type: 'BASIC', n: '0', kxx: '1e6' }],
        pointmasses: [],
    };
}

// --- the stamp, which is the whole design ---------------------------------------

clearSelection();
pick('shafts/', 0);
pick('shafts/', 2);

check('what was ticked is ticked', isPicked('shafts/', 0) === true);
check('and comes back in order, not in the order it was clicked',
    JSON.stringify(picked('shafts/')) === '[0,2]');
check('ticking twice unticks', pick('shafts/', 0) === false && isPicked('shafts/', 0) === false);

pick('shafts/', 0);
// The point of the stamp: the same numbers read from another list give nothing.
// Without it, "positions 0 and 2 of the shafts" would silently become
// "positions 0 and 2 of the bearings", and deleting would take the wrong two.
check('the same positions read from another tab give nothing',
    picked('disks/').length === 0);
check('and nothing is ticked there', isPicked('disks/', 0) === false);
// Reading another list does **not** throw the first one away -- only writing
// does. A read with a side effect would mean that drawing the screen could
// empty a selection depending on the order things happened in; exactly one
// function drops it, and it is the one below.
check('but merely looking at another tab does not throw the first one away',
    JSON.stringify(picked('shafts/')) === '[0,2]');

// Which is `nowShowing`, called by `renderList`. A selection belongs to what is
// on screen: leaving the list ends it. Before this, the ticks merely went
// invisible and came back when the tab did -- safe, because any real change to
// the model drops them anyway, but a resurrection nobody chose.
nowShowing('disks/');
check('putting another list on screen ends the selection',
    picked('disks/').length === 0);
nowShowing('shafts/');
check('and coming back does not bring it out again',
    picked('shafts/').length === 0);

// The two halves of a MultiRotor are two lists with the same tab name.
clearSelection();
pick('shafts/driving', 1);
check('the driving half has its tick', isPicked('shafts/driving', 1) === true);
check('the driven half does not', isPicked('shafts/driven', 1) === false);

// --- tick everything --------------------------------------------------------------

clearSelection();
pickAll('shafts/', 4);
check('all four are ticked', pickedCount('shafts/') === 4);
check('and the header knows it', allPicked('shafts/', 4) === true);
pickAll('shafts/', 4);
check('doing it again unticks them all', pickedCount('shafts/') === 0);
// Control: a header that reported "all" for an empty list would start ticked on
// a list with nothing in it.
check('an empty list is not all-ticked', allPicked('shafts/', 0) === false);

clearSelection();
pick('shafts/', 0);
pickAll('shafts/', 4);
check('ticking all from a partial selection takes the rest', pickedCount('shafts/') === 4);

// --- deleting several ------------------------------------------------------------

const { copySelected, deleteSelected, toggleSelectAll, toggleSelected, refreshHistoryButtons } =
    await import('../../frontend/features/modeling.js');
const { clearSelection: forget } = await import('../../frontend/core/selection.js');
const { canUndo } = await import('../../frontend/core/history.js');
const { openProjectHistory } = await import('../../frontend/core/state.js');

// Wired like the application: main.js subscribes the buttons, and nothing
// subscribes the selection -- `projectChanged` drops it itself, because it is
// state and not something on the page. A battery that subscribed it here would
// be testing its own wiring instead of the application's.
onProjectChanged(refreshHistoryButtons);

globalThis.fetch = async () => ({
    ok: true,
    json: async () => ({ status: 'success', figure: { data: [], layout: {} } }),
});

// What `openRotorWorkspace` does, in the part that matters here.
//
// `openProjectHistory` and not `resetHistory`: the first announces and so drops
// the selection, the second does not. The first draft of this helper used the
// second, and three checks below failed -- not because the feature was wrong,
// but because ticks from an earlier section of this file were still standing
// when the next one began. A helper that opens a rotor less thoroughly than the
// application does is a helper that tests a situation the application never
// reaches.
function openRotor(names) {
    state.rotorLibrary = [rotor(names)];
    state.activeRotorIndex = 0;
    state.projectData = state.rotorLibrary[0];
    state.currentTab = 'shafts';
    state.editingIndex = -1;
    openProjectHistory(state.projectData);
}

openRotor(['A', 'B', 'C', 'D']);
toggleSelected(0);
toggleSelected(2);
deleteSelected();

// The one that says the walk goes backwards. Deleting 0 then 2 forwards would
// take A and then whatever had moved into position 2 -- D -- and the list would
// simply be shorter, with no sign that the wrong ones went.
check('exactly the ticked elements went',
    state.projectData.shafts.map(s => s.tag).join(',') === 'B,D');
check('and the selection did not survive the change', pickedCount(listContext()) === 0);
// Eight deletions in one step of the undo is what a person means by "undo that".
check('the whole batch is one step of the undo', canUndo() === true);

// --- copying several -------------------------------------------------------------

openRotor(['A', 'B', 'C']);
toggleSelected(0);
toggleSelected(2);
copySelected();

check('the list grew by two', state.projectData.shafts.length === 5);
// Each copy right after its own original, which is what the single copy button
// already does. Walking forwards would put the second copy in the wrong place.
check('each copy sits right after its original',
    state.projectData.shafts.map(s => s.tag).join(',') === 'A,A_1,B,C,C_1');

// Control on the naming rule, which is shared with `copyItem`: a second copy of
// the same element cannot take a name that is already there.
forget();
toggleSelected(0);
copySelected();
check('a second copy takes the next free name',
    state.projectData.shafts.map(s => s.tag).join(',') === 'A,A_2,A_1,B,C,C_1');

// --- nothing ticked ----------------------------------------------------------------

openRotor(['A', 'B']);
deleteSelected();
check('deleting with nothing ticked does nothing', state.projectData.shafts.length === 2);
copySelected();
check('and neither does copying', state.projectData.shafts.length === 2);
check('and neither of them recorded a step of the undo', canUndo() === false);

// --- the bar --------------------------------------------------------------------

const { renderList } = await import('../../frontend/components/list.js');

openRotor(['A', 'B', 'C']);
renderList();
const bar = () => node('selection-bar').innerHTML;

check('the bar is showing on a list with elements',
    node('selection-bar').style.display === 'flex');
check('it offers to tick everything', /Select all/.test(bar()));
// `/selected/` on its own matches the buttons' own titles ("Copy the selected
// elements"), which is how the first version of this check passed for the wrong
// reason. The count is a number followed by the word.
check('with nothing ticked it says no count', !/\d+ selected/.test(bar()));
check('and both actions are dead', (bar().match(/disabled/g) || []).length === 2);

toggleSelected(1);
check('one tick brings the actions to life', !/disabled/.test(bar()));
check('and the bar says how many', /1 selected/.test(bar()));

toggleSelectAll();
check('ticking all says three', /3 selected/.test(bar()));
check('and the header box is ticked', /checked/.test(bar()));

// --- leaving the list, through the screen ------------------------------------------
//
// The same rule as above, but driven the way the application drives it: nobody
// calls `nowShowing` by hand, `renderList` does. This is the check that would
// fail if the call were taken out of the list and left only in the module.

openRotor(['A', 'B', 'C']);
toggleSelected(1);
check('setting the scene: one tick on the shafts', pickedCount(listContext()) === 1);

state.currentTab = 'disks';
renderList();
state.currentTab = 'shafts';
renderList();

check('a round trip through another tab ends the selection',
    pickedCount(listContext()) === 0);
check('and the box on the row is not ticked either',
    !/checked/.test(node('element-list').innerHTML));

// --- the change that is not a change of list ---------------------------------------
//
// The case the stamp does not cover, and the reason `clearSelection` is
// subscribed to `onProjectChanged`: the tab has not changed, the rotor has not
// changed, but the positions have.

openRotor(['A', 'B', 'C']);
toggleSelected(2);
check('setting the scene: C is ticked', pickedCount(listContext()) === 1);

state.projectData.shafts.unshift({ element_type: 'BASIC', L: '250', odl: '50', tag: 'Z' });
syncBackToLibrary();

check('a change to the same list drops the selection, because position 2 moved',
    pickedCount(listContext()) === 0);

shutDown();
