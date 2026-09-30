// Cutting one shaft element in two, from the modelling list.
//
// WHY THE BUTTON IS ON THE ELEMENT AND NOT ON THE ROTOR. `Rotor.add_nodes`
// takes a position along the whole shaft line, in metres. A field asking for
// that puts two jobs on the user: adding up the lengths of everything to the
// left, and converting a form that speaks millimetres into metres. It also
// makes two refusals possible that nobody wants to meet -- a position past the
// end of the rotor, and a position that lands exactly on a node that already
// exists, which ROSS answers by doing nothing at all.
//
// Asking *this element, this far from its left face* -- in the unit its length
// is typed in -- removes the arithmetic and makes the first refusal impossible
// by construction. The second one survives
// (0 and L are still nodes), and the server names it.
import { openCustomAlert, openCustomPrompt } from '../components/modals.js';
import { apiFetch } from '../core/api.js';
import { t } from '../core/i18n.js';

// `%1`, `%2`, ... are filled here rather than in `t`. The dictionary is a flat
// table of strings and the two places that need a number are both in this file;
// teaching the whole translation layer about placeholders for them would be a
// feature with two users.
function fill(template, ...values) {
    return values.reduce(
        (text, value, position) => text.split('%' + (position + 1)).join(String(value)),
        template,
    );
}

// The default the dialog opens with: halfway, which is the answer often enough
// to be worth offering and never wrong enough to be dangerous -- it is the one
// position inside the element that cannot be refused.
export function middleOf(length) {
    const span = Number(String(length).trim());
    if (!isFinite(span) || span <= 0) return '';
    return String(span / 2);
}

export function splitPrompt(shaft, index) {
    const length = String(shaft.L === undefined || shaft.L === null ? '' : shaft.L).trim();
    const tag = String(shaft.tag === undefined || shaft.tag === null ? '' : shaft.tag).trim();
    // The unit the length is typed in, which is also the unit the server reads
    // the answer in; `mm` is the form's default, as in modeling.js.
    const unit = String(shaft.L_unit || '').trim() || 'mm';
    return tag
        ? fill(t('splitNamed'), index + 1, tag, length, unit)
        : fill(t('splitAsk'), index + 1, length, unit);
}

// Answers whether the project changed, and changes nothing else.
//
// Redrawing the list and the figure is not done here: it is done by
// `splitItem` in features/modeling.js, beside `copyItem` and `deleteItem`,
// because a list mutation that refreshes the screen from three different
// places is how one of them ends up forgetting. What is left here is the part
// worth testing on its own -- the question, the trip, and the answer.
export async function splitProject(data, index) {
    const shafts = (data && data.shafts) || [];
    const shaft = shafts[index];
    if (!shaft) return false;

    const typed = await openCustomPrompt(splitPrompt(shaft, index), middleOf(shaft.L));
    // `null` is Escape or Cancel, and an empty box is somebody who changed their
    // mind with the keyboard. Neither is a distance, and neither is an error.
    if (typed === null || typed === undefined || String(typed).trim() === '') return false;

    let answer;
    try {
        answer = await apiFetch('/api/rotor/split_shaft', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                project: withoutAnalyses(data),
                index,
                offset: String(typed).trim(),
            }),
        });
    } catch (error) {
        await openCustomAlert(t('splitFailed'));
        return false;
    }

    const body = await answer.json();
    // The server's sentence, not one of ours: "would land on node 2, which
    // already exists" is what tells the user what to type instead.
    if (!answer.ok) {
        await openCustomAlert(body.message || t('splitFailed'));
        return false;
    }

    applyProject(data, body.projectData);
    return true;
}

// Copy back every list the server returned, and nothing else.
//
// The alternative is naming the categories here -- shafts, disks, gears,
// couplings, seals, bearings, pointmasses, materials -- which would be the
// fourth copy of that list in this project and the one nobody would remember to
// update. What comes back is the project that went up with its element lists
// rewritten, so "every array" is exactly the right set, and `name`, `uid` and
// `savedAnalyses` are not arrays.
export function applyProject(target, answered) {
    Object.keys(answered || {}).forEach(key => {
        if (Array.isArray(answered[key])) target[key] = answered[key];
    });
    return target;
}

// The saved analyses do not go up: they are the heavy half of the project and
// splitting does not read them. They are also not thrown away, which is
// deliberate -- splitting an element is an edit like any other on this screen,
// and editing an element has never discarded the charts already computed.
function withoutAnalyses(project) {
    const copy = JSON.parse(JSON.stringify(project));
    delete copy.savedAnalyses;
    return copy;
}
