// The name a material has to have for ROSS to accept it.
//
// `rs.Material` refuses any space in the name ("Spaces are not allowed in
// Material name"). The form let a person type "Stainless Steel", and the rotor
// then failed to build, far from the field that caused it. So the name is
// written the way ROSS wants it when the material is saved, and the list shows
// what ROSS will be given.
//
// The server applies the same rule (domain/material_names.py) to projects
// saved before this, and the two are held to the same cases in
// tests/golden/material_names.json.

// Runs of whitespace become one underscore; the ends are trimmed.
export function rossMaterialName(name) {
    return String(name).split(/\s+/).filter(Boolean).join('_');
}

// How a reference to a material is matched: ROSS's name, without case.
export function materialKey(name) {
    return rossMaterialName(name).toLowerCase();
}

// How many elements name this material. What it is for: deleting a material
// that shafts still use leaves those shafts naming one that no longer exists,
// and the server refuses to build the rotor (domain/material_names.py). The
// screen says how many before the material goes, not after.
export function elementsUsing(project, name) {
    if (name === undefined || name === null) return 0;
    const key = materialKey(name);
    let count = 0;
    Object.keys(project).forEach(category => {
        if (category === 'materials' || !Array.isArray(project[category])) return;
        project[category].forEach(element => {
            if (element && element.material !== undefined && materialKey(element.material) === key) count++;
        });
    });
    return count;
}

// Points the elements that named a material at its new name, and says how
// many moved. Shafts and gears name their material by text; before this, an
// edited name left them naming one that no longer existed, and the server
// answers an unknown name with the first material of the list -- a rotor
// built out of the wrong metal, with nothing on screen.
export function renameMaterial(project, oldName, newName) {
    if (oldName === undefined || oldName === null || oldName === newName) return 0;
    const key = materialKey(oldName);
    let moved = 0;
    Object.keys(project).forEach(category => {
        if (category === 'materials' || !Array.isArray(project[category])) return;
        project[category].forEach(element => {
            if (element && element.material !== undefined && materialKey(element.material) === key) {
                element.material = newName;
                moved++;
            }
        });
    });
    return moved;
}
