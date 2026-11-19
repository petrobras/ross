// The box that sits over a result while it is being computed: a spinner, a
// word, and the seconds running. It was written inside the analysis cards
// (features/dt_analyses.js) and the balancing analysis of step 4 needs the
// same thing, so it lives here and both call it.
//
// The seconds matter: a run can take from a few seconds to minutes, depending
// on the extra iterations, and a spinner alone leaves the user guessing
// whether anything is happening.
import { escapeHtml } from '/core/dom.js';

export function showLoader(id, target, options) {
    const settings = options || {};
    target.style.opacity = '0.4';
    let loader = document.getElementById('loading-' + id);
    if (!loader) {
        loader = document.createElement('div');
        loader.id = 'loading-' + id;
        loader.className = 'dt-loader';
        loader.innerHTML = `
            <div class="dt-loader-body">
                <i class="fas fa-sync fa-spin fa-2x"></i>
                <span class="dt-loader-label">${escapeHtml(settings.label || '')}</span>
                <span id="loading-when-${id}" class="job-elapsed"></span>
            </div>`;
        loader.style.top = settings.top || '200px';
        target.parentElement.appendChild(loader);
    }
    const started = Date.now();
    const when = document.getElementById('loading-when-' + id);
    const timer = setInterval(() => {
        when.textContent = Math.round((Date.now() - started) / 1000) + ' s';
    }, 1000);
    return () => {
        clearInterval(timer);
        loader.remove();
        target.style.opacity = '1';
    };
}
