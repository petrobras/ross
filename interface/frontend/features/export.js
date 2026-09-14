// Generates the Python script equivalent to what is on screen. The script is
// assembled in the backend (`domain/python_export.py`); here we only decide what
// goes into it.
import { openCustomAlert } from '../components/modals.js';
import { collectActiveAnalyses, unanimousConversion } from '../core/analysis_store.js';
import { apiFetch, wasCancelled, projectForServer } from '../core/api.js';
import { downloadTextFile } from '../core/dom.js';
import { state } from '../core/state.js';
import { t } from '../core/i18n.js';
export async function generatePythonFile(project, analyses) {
    const target = project || state.projectData;
    const chosen = analyses !== undefined ? analyses : collectActiveAnalyses();

    // This used to come from the selector on the analysis screen, the same value
    // for every card -- including the ones that had been computed under another
    // conversion. The script came out with numbers that did not match the figures,
    // with nothing to warn about it.
    const conversion = unanimousConversion(chosen);
    if (conversion === null) {
        await openCustomAlert(t('mixedConversions'));
        return;
    }

    const body = {
        project: projectForServer(target),
        analyses: chosen.map(a => ({ type: a.type, params: a.params })),
        conversion_type: conversion
    };

    try {
        const resp = await apiFetch('/api/export/python', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        const data = await resp.json();
        if (!data || data.status !== 'success') {
            await openCustomAlert((data && data.message) || t('exportServerError'));
            return;
        }
        downloadTextFile('my_ross_script.py', data.script, 'text/x-python');
    } catch (e) {
        if (wasCancelled(e)) return;
        await openCustomAlert(t('exportServerError'));
    }
}
