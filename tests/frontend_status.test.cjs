const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('smartreplay/frontend/static/app.js', 'utf8');
const render = source.slice(source.indexOf('function renderStatus()'), source.indexOf('let validateTimer'));

function statusView(status, saved = {revision: 3}) {
  const label = {}, element = {};
  vm.runInNewContext(render + '\nrenderStatus();', {
    state: {status, saved, savedAt: 0}, performance: {now: () => 10000},
    $: selector => selector === '#player-status' ? element : label,
  });
  return {label, element};
}
const recording = {state: 'recording', message: 'Recording Stick Fight'};
test('recording retains failed hotkey warnings', () => {
  const {label, element} = statusView({running: true, revision: 3, runtime: recording,
    hotkeys: {toggle_replay: {state: 'failed', detail: 'Already registered'}}});
  assert.match(label.textContent, /not applied/);
  assert.match(element.className, /bad/);
  assert.match(element.title, /Already registered/);
});
test('recording retains unapplied revision warnings', () => {
  const {label} = statusView({running: true, revision: 2, runtime: recording});
  assert.match(label.textContent, /last save/);
});
test('model errors do not hide recording status', () => {
  const {label, element} = statusView({running: true, revision: 3,
    runtime: {...recording, model: {state: 'error', message: 'Offline'}}});
  assert.match(label.textContent, /Recording Stick Fight/);
  assert.match(label.textContent, /AI\/voice/);
  assert.match(element.title, /Offline/);
});
