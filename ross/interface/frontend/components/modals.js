// The interface's own dialogs (prompt, confirm, alert), which replace the
// browser ones and give back a Promise.

let customPromptResolver = null;

let customConfirmResolver = null;

export const openCustomPrompt = function(message, defaultValue = '') {
    return new Promise((resolve) => {
        customPromptResolver = resolve;
        document.getElementById('custom-prompt-message').innerText = message;
        const inputEl = document.getElementById('custom-prompt-input');
        inputEl.value = defaultValue;
        document.getElementById('custom-prompt-overlay').style.display = 'flex';
        inputEl.focus();
        inputEl.select();        
        inputEl.onkeydown = function(e) {
            if (e.key === 'Enter') confirmCustomPrompt();
            if (e.key === 'Escape') closeCustomPrompt(null);
        };
    });
};

export const confirmCustomPrompt = function() {
    const val = document.getElementById('custom-prompt-input').value;
    closeCustomPrompt(val);
};

export const closeCustomPrompt = function(value) {
    document.getElementById('custom-prompt-overlay').style.display = 'none';
    if (customPromptResolver) {
        customPromptResolver(value);
        customPromptResolver = null;
    }
};

export const openCustomConfirm = function(message) {
    return new Promise((resolve) => {
        customConfirmResolver = resolve;
        document.getElementById('custom-confirm-message').innerText = message;
        document.getElementById('custom-confirm-overlay').style.display = 'flex';
    });
};

export const closeCustomConfirm = function(value) {
    document.getElementById('custom-confirm-overlay').style.display = 'none';
    if (customConfirmResolver) {
        customConfirmResolver(value);
        customConfirmResolver = null;
    }
};

let customAlertResolver = null;

export const openCustomAlert = function(message) {
    return new Promise((resolve) => {
        customAlertResolver = resolve;
        document.getElementById('custom-alert-message').innerText = message;
        document.getElementById('custom-alert-overlay').style.display = 'flex';
    });
};

export const closeCustomAlert = function() {
    document.getElementById('custom-alert-overlay').style.display = 'none';
    if (customAlertResolver) {
        customAlertResolver();
        customAlertResolver = null;
    }
};
