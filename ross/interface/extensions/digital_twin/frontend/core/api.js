// The Digital Twin's requests, on top of the ROSS interface's `apiFetch`, which
// adds the session token.
import { apiFetch } from '/core/api.js';

// A JSON request that answers JSON. A refused or failed request becomes an
// `Error` carrying the server's message, so the caller has one path for both.
export async function apiJson(path, body, method) {
    const options = { method: method || (body === undefined ? 'GET' : 'POST') };
    if (body !== undefined) {
        options.headers = { 'Content-Type': 'application/json' };
        options.body = JSON.stringify(body);
    }
    const response = await apiFetch(path, options);
    let answer = null;
    try { answer = await response.json(); } catch (e) { answer = null; }
    // A route the server does not know means the running server predates this
    // page: Flask runs without the reloader, so new code needs a restart.
    if (response.status === 404 || response.status === 405) {
        throw new Error('O servidor não conhece ' + path + '. Reinicie o ross-interface: o processo em execução é anterior a esta versão da interface.');
    }
    if (!response.ok) {
        throw new Error((answer && answer.message) || ('HTTP ' + response.status));
    }
    return answer;
}

// A request whose answer is a file the user keeps.
export async function apiDownload(path, body, fallbackName) {
    const response = await apiFetch(path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
    if (!response.ok) {
        let answer = null;
        try { answer = await response.json(); } catch (e) { answer = null; }
        throw new Error((answer && answer.message) || ('HTTP ' + response.status));
    }
    const blob = await response.blob();
    const header = response.headers.get('Content-Disposition') || '';
    const found = /filename="?([^";]+)"?/.exec(header);
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = found ? found[1] : fallbackName;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(a.href);
}
