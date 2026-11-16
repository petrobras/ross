// The side panels of the Digital Twin screens: the step buttons that show one
// panel of a group, the sidebars, and the cards that fold. The ROSS interface's
// `toggleSidebar` acts on its own modelling sidebar, so these screens keep
// theirs, addressed by id.

export function openPanel(group, name) {
    document.querySelectorAll(`.tab-btn[data-tab-group="${group}"]`).forEach(button => {
        button.classList.toggle('active', button.dataset.tab === name);
    });
    document.querySelectorAll(`[data-panel-group="${group}"]`).forEach(panel => {
        panel.style.display = panel.dataset.panel === name ? '' : 'none';
    });
}

export function toggleSidebarById(sidebarId) {
    const sidebar = document.getElementById(sidebarId);
    if (!sidebar) return;
    sidebar.classList.toggle('collapsed');
    setTimeout(() => window.dispatchEvent(new Event('resize')), 300);
}

export function toggleCard(bodyId) {
    const body = document.getElementById(bodyId);
    if (!body) return;
    body.style.display = body.style.display === 'none' ? '' : 'none';
    setTimeout(() => window.dispatchEvent(new Event('resize')), 50);
}
