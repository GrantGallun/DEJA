const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

// The browser is external; exercise the actual page script against its DOM boundary.
class Element {
  constructor() {
    this.children = []; this.textContent = ''; this.hidden = false; this.value = '';
    this.style = {}; this.dataset = {}; this.className = ''; this.paused = true; this.readyState = 1;
    this.classList = {toggle: () => {}, remove: () => {}};
  }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = [...children]; }
  addEventListener() {}
  pause() { this.paused = true; }
  play() { this.paused = false; return Promise.resolve(); }
  load() {}
  removeAttribute() {}
  focus() {}
}

const initial = () => ({busy: false, phase: 'ready', match: {id: 'fresh-match', url: '/media/fresh.mp4',
  duration: 25, source: 'live', created_at: '2026-10-25T02:00:00Z'}, result: null, error: '',
  game: {running: true, minimized: false}, model: {state: 'connected', message: 'Test'}, backend: 'codex'});

async function page(failure = null, fetchOverride = null) {
  const elements = new Map();
  const element = id => { if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id); };
  const context = vm.createContext({console, setInterval: () => 0,
    document: {getElementById: element, createElement: () => new Element(),
      querySelectorAll: selector => selector === '.result-card' ? element('results').children.filter(c => c.className === 'result-card') : []},
    fetch: fetchOverride || (async route => ({ok: !(failure && route === '/api/record'),
      json: async () => failure && route === '/api/record' ? {error: failure} : initial()}))});
  const source = fs.readFileSync(path.join(__dirname, '../smartreplay/demo/static/app.js'), 'utf8');
  vm.runInContext(source, context);
  await vm.runInContext('poll()', context);
  return {context, element, render: state => { context.nextState = state; vm.runInContext('render(nextState)', context); }};
}

for (const outcome of ['absent', 'uncertain']) {
  test(`a new ${outcome} answer cannot leave the previous highlight active`, async () => {
    const p = await page();
    const state = initial();
    state.result = {match_id: 'fresh-match', answered_at: 'first', elapsed_s: 1,
      matches: [{start_s: 4, end_s: 9, time_s: 7, verdict: 'yes', evidence: 'Yellow falls'}]};
    p.render(state);
    assert.equal(p.element('player').paused, false);
    state.result = {match_id: 'fresh-match', answered_at: 'second', elapsed_s: 1,
      matches: outcome === 'absent' ? [] : [{start_s: 10, end_s: 15, time_s: 13, verdict: 'maybe', evidence: 'Obscured'}]};
    p.render(state);
    assert.equal(p.element('player').paused, true);
    assert.match(p.element('clipLabel').textContent, /full recording/);
  });
}

for (const message of ['Open Stick Fight first.', 'Wait for the current operation to finish.']) {
  test(`a failed request remains visible after polling: ${message}`, async () => {
    const p = await page(message);
    await vm.runInContext("post('/api/record')", p.context);
    assert.equal(p.element('error').hidden, false);
    assert.equal(p.element('error').textContent, message);
    await vm.runInContext('poll()', p.context);
    assert.equal(p.element('error').textContent, message);
  });
}

test('questions leave native recording controls and state independent', async () => {
  const p = await page();
  const state = initial();
  Object.assign(state, {busy: true, phase: 'searching', capture_control_busy: false,
    native: {available: true, enabled: true, recording: true, cache: {bytes: 4096, budget_bytes: 134217728}},
    sessions: [{id: 'native-one', game: 'Stick Fight', duration_s: 180, created_at: '2026-10-25T02:00:00Z'}],
    selected_session_id: 'native-one', index: {indexed_until_s: 125, duration_s: 180, indexing: true}});
  p.render(state);
  assert.equal(p.element('stop').hidden, false);
  assert.equal(p.element('stop').disabled, false);
  assert.match(p.element('captureNote').textContent, /recording/i);
  assert.equal(p.element('sessionPicker').disabled, true);
});

test('a native replay loading request cannot disable Stop recording', async () => {
  let finishPlay, requested;
  const p = await page(null, async (route, options) => {
    if (route === '/api/play') {
      requested = JSON.parse(options.body);
      return await new Promise(resolve => { finishPlay = resolve; });
    }
    return {ok: true, json: async () => initial()};
  });
  const state = initial();
  state.match.source = 'native';
  state.native = {available: true, enabled: true, recording: true, cache: {}};
  state.result = {match_id: 'fresh-match', answered_at: 'native-answer', elapsed_s: 1,
    matches: [{start_s: 4, end_s: 9, time_s: 7, verdict: 'yes', evidence: 'Yellow falls'}]};
  p.render(state);
  try {
    assert.deepEqual(requested, {match_id: 'fresh-match', start_s: 4, end_s: 9});
    assert.equal(p.element('stop').hidden, false);
    assert.equal(p.element('stop').disabled, false);
  } finally {
    finishPlay({ok: true, json: async () => state});
  }
});

test('a voice result already played by the service cannot trigger duplicate native playback', async () => {
  const played = [];
  const p = await page(null, async route => {
    if (route === '/api/play') played.push(route);
    return {ok: true, json: async () => initial()};
  });
  const state = initial();
  state.match.source = 'native';
  state.native = {available: true, enabled: true, recording: true, cache: {}};
  state.result = {match_id: 'fresh-match', answered_at: 'voice-answer', elapsed_s: 1,
    native_auto_played: true, matches: [{start_s: 4, end_s: 9, time_s: 7, verdict: 'yes', evidence: 'Yellow falls'}]};
  p.render(state);
  assert.deepEqual(played, []);
  assert.equal(p.element('player').paused, true);
  assert.match(p.element('clipLabel').textContent, /native replay/i);
  assert.equal(p.element('results').children.length, 1);
});

test('speech progress and the recognized question remain visible while recording continues', async () => {
  const p = await page();
  const state = initial();
  Object.assign(state, {busy: true, phase: 'transcribing',
    native: {available: true, enabled: true, recording: true},
    voice: {state: 'searching', request_id: 'voice-one', message: 'Finding your moment', transcript: 'Show the last yellow fall'}});
  p.render(state);
  assert.equal(p.element('voiceStatus').hidden, false);
  assert.match(p.element('voiceMessage').textContent, /Finding your moment/);
  assert.equal(p.element('voiceTranscript').textContent, '“Show the last yellow fall”');
  assert.equal(p.element('stop').disabled, false);
  state.voice = {state: 'no_match', message: 'No matching moment was confirmed.', transcript: 'Show the last yellow fall'};
  p.render(state);
  assert.match(p.element('voiceMessage').textContent, /No matching moment/);
});
