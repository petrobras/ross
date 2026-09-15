// The About dialog shows the version the server put into the page.
//
// The dictionary guards prove the dialog is marked and translated; this
// battery runs the real module against the fake DOM and looks at what ends up
// on screen: the dialog opens, names the version, and closes.
import { node, clearDom, check, shutDown } from './fake_dom.js';

const { openAbout, closeAbout } = await import('../../frontend/components/about.js');

console.log('\nThe About dialog');

clearDom();
globalThis.ROSS_VERSION = '3.0.0.test';
openAbout();
check('opening shows the dialog', node('about-modal-overlay').style.display === 'flex');
check('the dialog names the version the server injected',
      node('about-version').textContent === '3.0.0.test');

const overlay = node('about-modal-overlay');
overlay.onclick({ target: node('about-box-child') });
check('a click inside the box keeps it open', overlay.style.display === 'flex');
overlay.onclick({ target: overlay });
check('a click on the scrim closes it', overlay.style.display === 'none');

openAbout();
closeAbout();
check('the close button closes it', overlay.style.display === 'none');

clearDom();
delete globalThis.ROSS_VERSION;
openAbout();
check('a page with no version shows an empty field, not "undefined"',
      node('about-version').textContent === '');

shutDown();
