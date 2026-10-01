// Reading back a project file the interface wrote.
//
// WHY THIS BATTERY EXISTS. The old loader rebuilt the rotor from eight keys
// named by hand, so anything outside that list was dropped without a word. A
// saved MultiRotor came back as an **empty ordinary rotor** carrying the file's
// name -- which is worse than a refusal, because it looks like it worked.
//
// So what is checked here is mostly what *survives* the trip, and the two
// normalisations that are decisions rather than plumbing: the identity is
// always fresh, and a loaded MultiRotor has its links to the library cut.
import { check, shutDown } from './fake_dom.js';

const { CATEGORIES, isProjectFile, projectFromFile } =
    await import('../../frontend/core/project_file.js');

function plainRotor() {
    return {
        name: 'Compressor',
        uid: 'rotor_111_aaa',
        savedAnalyses: [{ type: 'campbell' }],
        materials: [{ name: 'Steel' }],
        shafts: [{ element_type: 'BASIC', L: '250', odl: '50' }],
        disks: [],
        gears: [],
        couplings: [],
        seals: [],
        bearings: [{ element_type: 'BASIC', n: '0', kxx: '1e6' }],
        pointmasses: [],
    };
}

// The shape `saveMultiRotor` writes, which is what `saveRotor` then puts in the
// file. Note `shafts: []` at the root -- it is why the old test for "is this
// ours" passed and the rotor still came back empty.
function multiRotorFile() {
    return {
        name: 'Train',
        isMultiRotor: true,
        uid: 'rotor_222_bbb',
        driving_uid: 'rotor_111_aaa',
        driven_uid: 'rotor_333_ccc',
        savedAnalyses: [],
        driving_rotor: plainRotor(),
        driven_rotor: plainRotor(),
        multi_params: {
            coupled_nodes: '3,0',
            gear_mesh_stiffness: '1e8',
            backlash: { enable: false, initial_value: 0.0 },
        },
        materials: [],
        shafts: [],
        disks: [],
        gears: [],
        couplings: [],
        seals: [],
        bearings: [],
        pointmasses: [],
    };
}

// --- is this file ours --------------------------------------------------------

check('a rotor of ours is recognised', isProjectFile(plainRotor()) === true);
check('and so is a multirotor', isProjectFile(multiRotorFile()) === true);
check('a file written before the translation is too',
    isProjectFile({ eixos: [{ L: '250' }] }) === true);
check('something else is not', isProjectFile({ rotor: {}, Elements: [] }) === false);
check('and neither is a list, or nothing at all',
    isProjectFile([1, 2]) === false && isProjectFile(null) === false);

// --- the multirotor, which is why this exists ---------------------------------

const train = projectFromFile(multiRotorFile(), 'train');

check('it comes back as a multirotor', train.isMultiRotor === true);
check('with the driving half inside it', train.driving_rotor.shafts.length === 1);
check('and the driven half', train.driven_rotor.shafts.length === 1);
// The four keys the old loader dropped. Each one is the whole feature.
check('and the gear mesh parameters', train.multi_params.gear_mesh_stiffness === '1e8');
check('including what is nested in them', train.multi_params.backlash.enable === false);

// The links to the library are cut on purpose: loading the same file back into
// the session that wrote it would otherwise let `syncMultiRotors` overwrite the
// saved halves with whatever those rotors look like now.
check('the link to the driving parent is cut', train.driving_uid === undefined);
check('and to the driven one', train.driven_uid === undefined);

// --- the ordinary rotor still loads -------------------------------------------

const one = projectFromFile(plainRotor(), 'my_rotor');
check('the elements are there', one.shafts.length === 1 && one.bearings.length === 1);
check('and the materials', one.materials.length === 1);
check('the name comes from the file, which is the more recent of the two',
    one.name === 'my_rotor');
check('the saved analyses travel', one.savedAnalyses.length === 1);

// --- the identity --------------------------------------------------------------

check('the identity is not the one in the file', one.uid !== 'rotor_111_aaa');
const twice = projectFromFile(plainRotor(), 'my_rotor');
check('loading the same file twice gives two rotors that can be told apart',
    twice.uid !== one.uid);

// --- files written before the translation ---------------------------------------

const old = projectFromFile({
    eixos: [{ L: '250' }],
    discos: [{ n: '1' }],
    mancais: [{ n: '0' }],
    materiais: [{ name: 'Aco' }],
    engrenagens: [],
    acoplamentos: [],
    selos: [{ n: '2' }],
}, 'antigo');

check('the shafts arrive under their english name', old.shafts.length === 1);
check('and the disks', old.disks.length === 1);
check('and the bearings', old.bearings.length === 1);
check('and the materials', old.materials.length === 1);
// The one the old loader got wrong: it looked for `badges`, which is what an
// automatic translation makes of `selos`, so an old project lost its seals.
check('and the seals, which used to be lost', old.seals.length === 1);
check('no portuguese key is left behind', old.eixos === undefined && old.selos === undefined);

// `badges` is accepted too, because which of the two an old file really carries
// cannot be measured from here -- and accepting the wrong one costs nothing.
const badged = projectFromFile({ shafts: [], badges: [{ n: '2' }] }, 'x');
check('a file written with the mistranslated key also keeps its seals',
    badged.seals.length === 1);

// Control: a file that speaks both was written by a version that already spoke
// English, so the english key wins and nothing is doubled.
const both = projectFromFile({ shafts: [{ L: '1' }], eixos: [{ L: '2' }] }, 'x');
check('a file with both spellings keeps the current one',
    both.shafts.length === 1 && both.shafts[0].L === '1');

// --- a file missing a category ---------------------------------------------------
//
// `renderList` empties the list and *then* reads the category, so a project
// without `gears` leaves the screen blank with nothing in the console. Every
// category is filled in here instead of making each reader defensive.

const thin = projectFromFile({ shafts: [{ L: '250' }] }, 'thin');
CATEGORIES.forEach(category => {
    check('a missing ' + category + ' arrives as an empty list',
        Array.isArray(thin[category]));
});

// Control: filling the gaps must not overwrite what the file did bring.
check('and a category the file did have is untouched', one.shafts.length === 1);

// --- what is not ours ------------------------------------------------------------

check('a native ROSS file is not read as one of ours',
    projectFromFile({ Elements: [], rotor: {} }, 'x') === null);

shutDown();
