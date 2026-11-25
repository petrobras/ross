// The coupled nodes of a MultiRotor when one of its halves is renumbered.
//
// `multi_params.coupled_nodes` names a node of each half -- "2, 1": node 2 of
// the driving rotor meshes with node 1 of the driven one -- and ROSS wants a
// gear on each (`MultiRotor`: "Each rotor needs a GearElement in the coupled
// nodes!"). The node is a number, though, and what the person chose was the
// gear: cutting a shaft below it, meshing the line or adding an element before
// it moves the gear to another node, and the number, left as it was, pointed
// past it. The MultiRotor then refused to build, far from the cut that broke it.
//
// So the number follows the gear: whichever gear row sat on the coupled node
// before the change, the coupled node is that row's node after it. Splitting,
// meshing and inserting (features/split.js `applyProject`) keep the gear rows
// and their order; they only move their nodes.
//
// No DOM and no state here, so node can test it; the nodes come in already
// resolved, by the builder's rule for gears (components/list.js
// `getEffectiveNodes`, domain/node_resolver.py `effective_nodes`).

// "2, 1" as [2, 1], as the builder reads it (domain/rotor_builder.py): two
// whole numbers separated by a comma. Null when it is not that.
export function parseCoupledNodes(text) {
    const parts = String(text === undefined || text === null ? '' : text).split(',');
    if (parts.length !== 2) return null;
    const nodes = parts.map(part => part.trim());
    const digits = n => n.length > 0 && Array.from(n).every(c => c >= '0' && c <= '9');
    if (!nodes.every(digits)) return null;
    return nodes.map(Number);
}

// The node of the gear that was on `node`: `before` and `after` are the nodes
// of the same gear rows, in order, before and after the change. The node as it
// was when no gear was on it (the coupling was already broken, and moving it
// somewhere else would only hide that) or when the rows are not the same ones.
export function followedNode(before, after, node) {
    if (!Array.isArray(before) || !Array.isArray(after) || before.length !== after.length) return node;
    const k = before.indexOf(node);
    return k < 0 ? node : after[k];
}

// The `coupled_nodes` text after the half at `position` (0 driving, 1 driven)
// was renumbered. The text as it was when nothing moved or it cannot be read,
// so a value the person typed is never rewritten into another spelling.
export function followCoupling(text, position, before, after) {
    const nodes = parseCoupledNodes(text);
    if (!nodes || (position !== 0 && position !== 1)) return text;
    const moved = followedNode(before, after, nodes[position]);
    if (moved === nodes[position]) return text;
    nodes[position] = moved;
    return nodes.join(', ');
}
