// Changing the language changes the screen -- measured on the screen, not in
// the dictionary.
//
// Everything this slice has as a guard in `test_fase3e.py` reads TEXT: that
// `index.html` is marked, that the dictionary has both languages, that no
// English sentence was left in the JS. None of that proves the change reaches
// the user. And the failure mode has already happened twice in this project:
// the key existed in the dictionary and nobody called it -- the Hub drew "Go to
// Modeling" with `goToModeling` translated right next to it.
//
// This battery runs the real modules and looks at the HTML that comes out.
import { node, disk, clearDisk, registerSelector, schemaResponse, check,
         shutDown } from './fake_dom.js';

globalThis.fetch = async path => {
    const schema = schemaResponse(path);
    return { ok: true, status: 200,
             json: async () => schema || { status: 'success' } };
};
globalThis.addEventListener = () => {};
globalThis.setInterval = () => 0;
globalThis.confirm = () => true;

const { setSchemaLanguage, applyLanguage, t, preferredLanguage,
        rememberLanguage } = await import('../../frontend/core/i18n.js');
const { state } = await import('../../frontend/core/state.js');
const { renderRotorHub } = await import('../../frontend/features/hub.js');

state.rotorLibrary = [
    { name: 'Compressor A', shafts: [{ L: '500' }], disks: [], bearings: [] },
];
state.activeRotorIndex = -1;

// --- the Hub, in both languages ----------------------------------------------

function hubEm(language) {
    setSchemaLanguage(language);
    node('rotor-hub-list').innerHTML = '';
    renderRotorHub();
    return node('rotor-hub-list').innerHTML;
}

const inEnglish = hubEm('en');
check('hub en: modeling button', inEnglish.includes('Go to Modeling'));
check('hub en: analysis button', inEnglish.includes('Go to Analysis'));
check('hub en: save json', inEnglish.includes('Save JSON'));
check('hub en: delete title', inEnglish.includes('title="Delete"'));

const inPortuguese = hubEm('pt');
check('hub pt: modeling button', inPortuguese.includes('Ir para Modelagem'));
check('hub pt: analysis button', inPortuguese.includes('Ir para Análises'));
check('hub pt: save json', inPortuguese.includes('Salvar JSON'));
check('hub pt: delete title', inPortuguese.includes('title="Apagar"'));

// The control that was missing the other times: it is not enough for the
// Portuguese to appear, the English has to disappear. A forgotten `innerHTML
// +=` would leave both.
for (const sentence of ['Go to Modeling', 'Go to Analysis', 'Save JSON',
                     'Generate Python', 'title="Delete"', 'title="Copy"',
                     'Click to edit name']) {
    check('hub pt: no "' + sentence + '"', !inPortuguese.includes(sentence));
}

// An empty list is a screen too.
state.rotorLibrary = [];
check('hub pt: empty library in Portuguese',
    hubEm('pt').includes('Nenhum rotor na biblioteca'));
check('hub en: empty library in English',
    hubEm('en').includes('No rotors available'));

// --- the static HTML ---------------------------------------------------------

const label = node('rot'); label.dataset.i18n = 'hubMyRotors';
const button = node('bt'); button.dataset.i18nTitle = 'delete';
const field = node('cp'); field.dataset.i18nPlaceholder = 'phCoupledNodes';
const selector = node('sel-language');
registerSelector('[data-i18n]', [label]);
registerSelector('[data-i18n-title]', [button]);
registerSelector('[data-i18n-placeholder]', [field]);
registerSelector('select.ui-language', [selector]);

setSchemaLanguage('pt');
applyLanguage();
check('applyLanguage: text', label.textContent === 'Meus Rotores');
check('applyLanguage: title', button.getAttribute('title') === 'Apagar');
check('applyLanguage: placeholder',
    field.getAttribute('placeholder') === 'nó_motor, nó_movido');
check('applyLanguage: the selector shows the current language', selector.value === 'pt');

setSchemaLanguage('en');
applyLanguage();
check('applyLanguage: back to English', label.textContent === 'My Rotors');
check('applyLanguage: title comes back', button.getAttribute('title') === 'Delete');

// --- the modeling tab title --------------------------------------------------
//
// The defect Leonardo found on screen: in Portuguese the big title above the
// element list stayed in English, and right after a language change it became
// the word "CATEGORIA". It was three things: the `<h3>` marked with
// `data-i18n` (`applyLanguage()` wrote over what `openTab` writes), the name
// coming from the internal key (`shafts`) instead of the label, and the
// language change not rewriting the title.

const { categoryName, refreshTabTitle } = await import('../../frontend/features/modeling.js');

function tabButton(category, key) {
    const b = node('tab:' + category);
    b.setAttribute('onclick', `openTab('${category}')`);
    b.dataset.i18n = key;
    return b;
}
registerSelector('.tab-btn', [tabButton('shafts', 'catShaft'),
                              tabButton('materials', 'catMaterial')]);

setSchemaLanguage('pt');
check('title: the name comes from the label, not from the internal key',
    categoryName('shafts', null) === 'Eixo');
setSchemaLanguage('en');
check('title: and back to English', categoryName('shafts', null) === 'Shaft');
check('title: a category with no button does not go blank',
    categoryName('does-not-exist', null) === 'does-not-exist');

state.currentTab = 'shafts';
setSchemaLanguage('pt');
refreshTabTitle();
const title = node('tab-title').innerHTML;
check('title: the language change rewrites it', title.includes('Eixo'));
check('title: no raw key on the screen', !title.includes('>shafts<'));
check('title: the help button tooltip translates too',
    title.includes('Ajuda sobre Eixo'));
check('title: never the bare word "Categoria"',
    !/>\s*Categoria\s*</.test(title));

// And the real path: `changeLanguage`, not `refreshTabTitle` by hand. Without
// this, removing the call from inside `changeLanguage` passed through here --
// only the orphan-function guard complained, and for another reason.
const { changeLanguage } = await import('../../frontend/features/modeling.js');

state.projectData = { shafts: [], bearings: [], materials: [], disks: [],
                       gears: [], seals: [], couplings: [], pointmasses: [] };
node('tab-title').innerHTML = 'before';
await changeLanguage('en');
check('changeLanguage: rewrites the tab title',
    node('tab-title').innerHTML.includes('Shaft'));
await changeLanguage('pt');
check('changeLanguage: and in the new language',
    node('tab-title').innerHTML.includes('Eixo'));


// --- the open help panel -----------------------------------------------------
//
// The same defect as the tab title, found by the guard while investigating
// that one: `#help-modal-title` was marked with `data-i18n="help"` and what
// writes it is `showHelpEntry()`. With the help open, changing the language
// replaced the entry title with the word "Ajuda".

const { openSectionHelp, reapplyHelp, closeHelpModal } =
    await import('../../frontend/components/help.js');

setSchemaLanguage('en');
openSectionHelp('shafts');
check('help: opens in English',
    node('help-modal-title').innerHTML.includes('Shafts Help'));

setSchemaLanguage('pt');
reapplyHelp();
check('help: follows the language change',
    node('help-modal-title').innerHTML.includes('Ajuda: Eixos'));
check('help: the body too',
    node('help-modal-body').innerHTML.includes('Comprimento (L):'));
check('help: the title does not become the word "Ajuda"',
    node('help-modal-title').innerHTML.trim() !== 'Ajuda');

closeHelpModal();
node('help-modal-title').innerHTML = 'untouched';
reapplyHelp();
check('help: closed, a language change does not reopen it',
    node('help-modal-title').innerHTML === 'untouched');

// --- the missing key ---------------------------------------------------------

setSchemaLanguage('pt');
check('t(): a key that does not exist gives back the key', t('doesNotExist') === 'doesNotExist');
check('t(): an unknown language falls back to English',
    (setSchemaLanguage('de'), t('delete') === 'Delete'));

// --- the preference survives -------------------------------------------------

clearDisk();
check('preferredLanguage: with nothing stored, English', preferredLanguage() === 'en');
rememberLanguage('pt');
check('rememberLanguage: writes', disk.content['ross-language'] === 'pt');
check('preferredLanguage: reads it back', preferredLanguage() === 'pt');

shutDown();
