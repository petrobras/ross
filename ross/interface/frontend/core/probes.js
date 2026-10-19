// The probes of the model, as the analyses offer them.
//
// A probe placed on the modelling screen is where the analyses are meant to
// read the response, and a row of an analysis's probe table can name one
// instead of typing a node and an angle (features/analysis.js). It names it
// by `id`: a name can be edited and a position in the list moves, and the
// row has to keep reading at the same probe through both. The server fills
// the row in from the model on every run (domain/probe_refs.py), so a probe
// turned, renamed or carried along by a split shaft needs no analysis touched.
import { t } from './i18n.js';

export function newProbeId() {
    return `probe-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

// The shaft lines of a project: the project, or the two halves of a MultiRotor.
function lines(project) {
    if (!project || typeof project !== 'object') return [];
    if (project.isMultiRotor) {
        return [['driving', project.driving_rotor], ['driven', project.driven_rotor]]
            .filter(([, line]) => line && typeof line === 'object');
    }
    return [[null, project]];
}

// Every probe with an id: one made before the ids existed, or brought in from
// a file, gets one here, once, and keeps it.
export function withProbeIds(project) {
    lines(project).forEach(([, line]) => {
        (Array.isArray(line.probes) ? line.probes : []).forEach(probe => {
            if (probe && typeof probe === 'object' && !probe.id) probe.id = newProbeId();
        });
    });
    return project;
}

// What a probe is called in a list of choices: its name, its line on a
// MultiRotor, its node, and how it reads.
function describe(probe, position, half) {
    const name = String(probe.tag || '').trim() || `${t('catProbe')} #${position + 1}`;
    const where = half ? `${t(half === 'driven' ? 'multiDriven' : 'multiDriving')} · ` : '';
    const how = probe.direction === 'axial'
        ? t('rotor3dProbeAxial')
        : `${String(probe.angle ?? '').trim() || '0'}${String(probe.angle_unit || 'deg') === 'rad' ? ' rad' : '°'}`;
    return `${name} · ${where}${t('probeNodeShort')} ${String(probe.n ?? '').trim()} · ${how}`;
}

// The probes an analysis can read at, in the order of the lists.
export function probeChoices(project) {
    withProbeIds(project);
    const choices = [];
    lines(project).forEach(([half, line]) => {
        (Array.isArray(line.probes) ? line.probes : []).forEach((probe, position) => {
            choices.push({ id: probe.id, label: describe(probe, position, half) });
        });
    });
    return choices;
}
