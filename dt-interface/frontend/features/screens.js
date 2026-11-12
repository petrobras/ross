// Navigation between the screens, the side panels, the panel tabs, and
// shutting down. Same rules as the reference interface.
import { openCustomConfirm } from '../components/modals.js';
import { apiFetch } from '../core/api.js';
import { t } from '../core/i18n.js';

export function switchScreen(screenId) {
    document.querySelectorAll('.screen').forEach(screen => screen.classList.remove('active'));
    document.getElementById(screenId).classList.add('active');
    setTimeout(() => {
        window.dispatchEvent(new Event('resize'));
    }, 150);
}

export function toggleSidebar(sidebarId) {
    const sidebar = document.getElementById(sidebarId);
    if (!sidebar) return;
    sidebar.classList.toggle('collapsed');
    setTimeout(() => window.dispatchEvent(new Event('resize')), 300);
}

// A sidebar button shows its panel and hides the others of the same group.
export function openPanel(group, name) {
    document.querySelectorAll(`.tab-btn[data-tab-group="${group}"]`).forEach(button => {
        button.classList.toggle('active', button.dataset.tab === name);
    });
    document.querySelectorAll(`[data-panel-group="${group}"]`).forEach(panel => {
        panel.style.display = panel.dataset.panel === name ? '' : 'none';
    });
}

// The header of a card folds its body, as the analysis cards of the reference do.
export function toggleCard(bodyId) {
    const body = document.getElementById(bodyId);
    if (!body) return;
    body.style.display = body.style.display === 'none' ? '' : 'none';
    setTimeout(() => window.dispatchEvent(new Event('resize')), 50);
}

export async function exitApplication() {
    const isConfirmed = await openCustomConfirm(t('confirmExit'));
    if (isConfirmed) {
        try { await apiFetch('/shutdown', { method: 'POST' }); } catch (e) { /* the process exits under the request */ }
        window.close();
        document.body.innerHTML = `<h2 style='text-align:center; margin-top:20%; color:#2c3e50;'><i class='fas fa-power-off'></i> ${t('serverShutdown')}</h2>`;
    }
}
