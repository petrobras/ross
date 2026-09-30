// Reading a project file that the interface itself wrote.
//
// WHY THIS IS NOT A REBUILD. `loadRotor` used to construct the loaded rotor
// from eight keys named by hand:
//
//     let newLoadedRotor = {
//         materials: loaded.materials || loaded.materiais || [],
//         shafts:    loaded.shafts    || loaded.eixos     || [],
//         ...
//     };
//
// Everything outside that list was dropped in silence -- which is what happened
// to the MultiRotor. `saveRotor` writes the whole `projectData`, so the file
// carries `isMultiRotor`, `driving_rotor`, `driven_rotor` and `multi_params`;
// none of the four is in the list, so a saved MultiRotor came back as an
// **empty ordinary rotor** with the file's name on it. Not a refusal: a rotor
// that looks loaded and is not.
//
// So this takes the object as the project and works on it, instead of listing
// what to keep. Adding a key to a project now costs nothing here, which is the
// point: the next `multi_params` will not go missing.

// The categories a project is expected to have.
//
// This is the one place in the frontend that needs them by name, and the reason
// is worth stating: a file may simply not have one. `renderList` does
// `container.innerHTML = ''` and *then* reads the category, so a project
// missing `gears` empties the element list and leaves nothing on screen and
// nothing in the console. Filling the gaps here is cheaper than making every
// reader defensive.
//
// `tests/test_frontend_structure.py` compares this list against
// `domain/element_registry.categories()`, so the two cannot drift.
export const CATEGORIES = [
    'materials',
    'shafts',
    'disks',
    'gears',
    'couplings',
    'bearings',
    'seals',
    'pointmasses',
];

// Keys a project written before the interface was translated still carries.
//
// A closed list: these names cannot grow, because nothing writes them any more.
//
// `selos` **and** `badges` both map to `seals`, and that is not belt and
// braces. The old loader looked for `badges`, which is the English word an
// automatic translation produces from `selos` -- every other alias in that list
// is a real Portuguese word, and `badges` is the only English one. So the
// translation pass almost certainly renamed the *data key* along with the
// prose, and no file ever had `badges` in it: an old project with seals lost
// them on load, silently. Both are accepted here because one of the two is
// right and accepting the wrong one costs nothing.
const LEGACY_NAMES = {
    materiais: 'materials',
    eixos: 'shafts',
    discos: 'disks',
    engrenagens: 'gears',
    acoplamentos: 'couplings',
    mancais: 'bearings',
    selos: 'seals',
    badges: 'seals',
};

// What says "this file came from here".
//
// `shafts` covers every ordinary rotor, `eixos` the ones written before the
// translation, and `isMultiRotor` the case that started this: a MultiRotor
// happens to carry `shafts: []` today, but that is an accident of how
// `saveMultiRotor` builds the object and not something to depend on.
export function isProjectFile(loaded) {
    if (!loaded || typeof loaded !== 'object' || Array.isArray(loaded)) return false;
    return loaded.isMultiRotor === true
        || loaded.shafts !== undefined
        || loaded.eixos !== undefined;
}

function freshUid() {
    return 'rotor_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

// The project as the library should hold it, or null when the file is not ours.
//
// `name` comes from the file name, as it always has: the person renamed the
// file, so the file name is the more recent of the two.
export function projectFromFile(loaded, fileName) {
    if (!isProjectFile(loaded)) return null;

    const project = {};
    Object.keys(loaded).forEach(key => {
        const name = LEGACY_NAMES[key] || key;
        // A translated key never wins over a current one: a file holding both
        // was written by a version that already spoke English.
        if (name !== key && loaded[name] !== undefined) return;
        project[name] = loaded[key];
    });
    Object.keys(LEGACY_NAMES).forEach(old => { delete project[old]; });

    CATEGORIES.forEach(category => {
        if (!Array.isArray(project[category])) project[category] = [];
    });

    project.name = fileName;
    project.savedAnalyses = project.savedAnalyses || [];
    // A fresh identity, always. Loading the same file twice has to give two
    // rotors that can be told apart, and a `uid` copied from the file would
    // collide with the rotor it was saved from.
    project.uid = freshUid();

    if (project.isMultiRotor) cutTheParentLinks(project);

    return project;
}

// A loaded MultiRotor stands on its own.
//
// `driving_uid` and `driven_uid` point at two rotors in the library of whoever
// saved it. Read in another library they find nothing, and the two functions
// that use them (`syncMultiRotors` and `syncBackToLibrary`) quietly do nothing
// -- which is the right behaviour for a MultiRotor with no parents.
//
// The danger is the opposite case: load the file back into the **same** session
// and the ids match, so `syncMultiRotors` overwrites the two halves that were
// saved with whatever those rotors look like now. The file would open showing
// something other than what is in it, and nothing would say so. Cutting the
// links makes that impossible rather than unlikely.
//
// The cost is stated plainly: a loaded MultiRotor is a snapshot. Editing its
// halves works and is saved, but no longer propagates to any loose rotor.
function cutTheParentLinks(project) {
    delete project.driving_uid;
    delete project.driven_uid;
}
