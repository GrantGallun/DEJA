const {test} = require('node:test');
const assert = require('node:assert/strict');
const {RecordingViewer} = require('../smartreplay/frontend/static/recordings.js');

class Element {
  constructor() { this.children = []; this.classList = {toggle() {}}; this.value = ''; this.currentTime = 0; }
  replaceChildren(...nodes) { this.children = nodes; }
  append(...nodes) { this.children.push(...nodes); }
  setAttribute() {}
  removeAttribute(name) { delete this[name]; }
  addEventListener() {}
  pause() { this.paused = true; }
  load() {}
  play() { this.paused = false; return Promise.resolve(); }
}
function viewer() {
  const elements = new Map();
  const doc = {getElementById(id) { if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id); },
    createElement() { return new Element(); }};
  const v = new RecordingViewer(doc);
  v.rows = [{id:'one',game:'Stick Fight',duration_s:5,playable:true},
    {id:'two',game:'Another recording',duration_s:9,playable:true}];
  return v;
}
test('a late recording preparation cannot replace the newer selection', async () => {
  const v = viewer();
  let finish;
  v.request = path => path.includes('/one/') ? new Promise(r => finish = r) : Promise.resolve({state:'ready',url:'/two.mp4'});
  const first = v.select('one');
  await v.select('two');
  finish({state:'ready',url:'/one.mp4'});
  await first;
  assert.equal(v.video.src, '/two.mp4');
});
test('no-match answer clears previous result buttons and stops highlight playback', () => {
  const v = viewer();
  v.results.append(new Element());
  v.video.src = '/old-search-result.mp4';
  v.showSearch({result:{matches:[],note:'No evidence'}, question:'Absent event'});
  assert.equal(v.results.children.length, 0);
  assert.equal(v.video.paused, true);
  assert.equal(v.video.src, undefined);
  assert.match(v.searchStatus.textContent, /No matching moment/);
});

test('a pending full recording cannot overwrite a newer search playback', async () => {
  const v = viewer();
  let finish;
  v.request = path => path.endsWith('/prepare') ? new Promise(r => finish = r)
    : Promise.resolve({state:'ready',result:{matches:[]}});
  const selecting = v.select('one');
  await v.runSearch('/api/recordings/one/search', {});
  v.loadVideo('/matching-moment.mp4', 1, 3);
  finish({state:'ready',url:'/whole-recording.mp4'});
  await selecting;
  assert.equal(v.video.src, '/matching-moment.mp4');
});

test('canceling microphone while permission is pending cannot begin capture', async () => {
  const v = viewer(); v.selected = 'one';
  let grant, stopped = 0, constructed = 0;
  const originalNavigator = global.navigator, originalRecorder = global.MediaRecorder;
  global.navigator = {mediaDevices:{getUserMedia: () => new Promise(r => grant = r)}};
  global.MediaRecorder = class { static isTypeSupported() { return true; } constructor() { constructed++; } start() {} };
  try {
    const pending = v.toggleVoice();
    v.stopVoice(true);
    grant({getTracks:() => [{stop:() => stopped++}]});
    await pending;
    assert.equal(constructed, 0);
    assert.equal(stopped, 1);
  } finally { global.navigator = originalNavigator; global.MediaRecorder = originalRecorder; }
});
test('switching recording discards a late search result', async () => {
  const v = viewer();
  let finish;
  v.selected = 'one';
  v.request = path => path.endsWith('/search') ? new Promise(r => finish = r) : Promise.resolve({state:'ready',url:'/two.mp4'});
  const search = v.runSearch('/api/recordings/one/search', {});
  await v.select('two');
  finish({state:'ready',result:{matches:[]},question:'old question'});
  await search;
  assert.equal(v.selected, 'two');
  assert.equal(v.searchStatus.textContent, '');
});
