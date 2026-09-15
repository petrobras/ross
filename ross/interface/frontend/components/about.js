// The About dialog: which ROSS this is, and where to go with a question.
//
// The version is the one the server wrote into the page next to the token
// (`window.ROSS_VERSION`), so the dialog, `ross-interface --version` and the
// selftest header can never disagree. The links are plain anchors: the page
// runs in the user's own browser, so a `target="_blank"` reaches GitHub even
// from the frozen executable.

export function openAbout() {
    document.getElementById('about-version').textContent = window.ROSS_VERSION || '';
    const overlay = document.getElementById('about-modal-overlay');
    // Clicking the scrim closes the dialog, like the help.
    overlay.onclick = event => { if (event.target === overlay) closeAbout(); };
    overlay.style.display = 'flex';
}

export function closeAbout() {
    document.getElementById('about-modal-overlay').style.display = 'none';
}
