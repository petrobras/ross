// Navigation between the three screens, the side panels, and shutting down.
import { openCustomConfirm } from '../components/modals.js';
import { apiFetch } from '../core/api.js';
import { state } from '../core/state.js';
import { openTab } from './modeling.js';
import { t } from '../core/i18n.js';
// Function to switch between windows without leaving others active

export function switchScreen(screenId) {
    document.querySelectorAll('.screen').forEach(screen => screen.classList.remove('active'));
    document.getElementById(screenId).classList.add('active');
    if(screenId === 'screen-modeling' && state.currentTab) openTab(state.currentTab);

    setTimeout(() => {
        window.dispatchEvent(new Event('resize'));
    }, 150);
}

// Function to hide the sidebar of the modeling

export function toggleSidebar() {
    const sidebar = document.querySelector('.sidebar');
    sidebar.classList.toggle('collapsed');
    setTimeout(() => window.dispatchEvent(new Event('resize')), 300);
}

// Function to hide the sidebar of the analysis

export function toggleAnalysisSidebar() {
    const sidebar = document.querySelector('.analysis-controls-panel');
    sidebar.classList.toggle('collapsed');    
    setTimeout(() => window.dispatchEvent(new Event('resize')), 300);
}

// Function to exit the application

export async function exitApplication() {
    let isConfirmed = await openCustomConfirm(t('confirmExit'));
    if(isConfirmed) {
        try { await apiFetch('/shutdown', {method: 'POST'}); } catch(e) {} 
        window.close();
        document.body.innerHTML = `<h2 style='text-align:center; margin-top:20%; color:#2c3e50;'><i class='fas fa-power-off'></i> ${t('serverShutdown')}</h2>`;
    }
}
