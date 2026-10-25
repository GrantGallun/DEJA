/* All results belong to an immutable match; text from the model is never HTML. */
const $ = id => document.getElementById(id);
let state = null, currentMatch = null, shownAnswer = null, clip = null, requestPending = false, requestError = '', sessionSignature = '';
let pendingRequests = 0, captureRequests = 0;
const stamp = seconds => `${Math.floor(seconds / 60)}:${Math.floor(seconds % 60).toString().padStart(2, '0')}`;

async function post(route, body = {}) {
  pendingRequests += 1;
  requestPending = true;
  const captureRequest = route === '/api/record' || route === '/api/stop';
  if (captureRequest) captureRequests += 1;
  requestError = '';
  if (state) render(state);
  try {
    const response = await fetch(route, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'The request failed.');
    render(data);
  } catch (error) {
    requestError = error.message;
    $('error').textContent = error.message;
    $('error').hidden = false;
  } finally {
    pendingRequests -= 1;
    requestPending = pendingRequests > 0;
    if (captureRequest) captureRequests -= 1;
    await poll();
  }
}

function playInterval(match, index) {
  clip = match;
  document.querySelectorAll('.result-card').forEach((card, i) => card.classList.toggle('selected', i === index));
  const player = $('player');
  const start = () => {
    if (clip !== match) return;
    player.currentTime = match.start_s;
    player.play().catch(() => { $('clipLabel').textContent += ' · Press play'; });
  };
  $('clipLabel').textContent = `${stamp(match.start_s)}–${stamp(match.end_s)} · moment at ${stamp(match.time_s)}`;
  if (player.readyState >= 1) start();
  else player.addEventListener('loadedmetadata', start, {once: true});
  if (state.match?.source === 'native' && state.native?.available) {
    post('/api/play', {match_id: state.match.id, start_s: match.start_s, end_s: match.end_s});
  }
}

function showResults(result) {
  $('results').replaceChildren();
  if (!result) return;
  result.matches.forEach((match, index) => {
    const button = document.createElement('button');
    button.className = 'result-card';
    const top = document.createElement('div');
    top.className = 'result-top';
    const time = document.createElement('span');
    time.className = 'timestamp'; time.textContent = `▶ ${stamp(match.session_time_s ?? match.time_s)}`;
    const label = document.createElement('span'); label.textContent = `Moment ${index + 1}`;
    const certainty = document.createElement('span'); certainty.className = 'certainty';
    certainty.textContent = match.verdict === 'yes' ? 'Supported by the frames' : 'Uncertain · take a look';
    const evidence = document.createElement('p'); evidence.textContent = match.evidence;
    top.append(time, label, certainty); button.append(top, evidence);
    button.addEventListener('click', () => playInterval(match, index));
    $('results').append(button);
  });
  if (result.note) {
    const note = document.createElement('p'); note.className = 'result-note'; note.textContent = result.note;
    $('results').append(note);
  }
  // Uncertain candidates are always opened explicitly by the viewer.
  const confirmed = result.matches.findIndex(match => match.verdict === 'yes'
    && Number.isFinite(match.start_s) && Number.isFinite(match.end_s)
    && 0 <= match.start_s && match.start_s < match.end_s && match.end_s <= state.match.duration);
  if (confirmed >= 0 && result.native_auto_played) {
    document.querySelectorAll('.result-card').forEach((card, i) => card.classList.toggle('selected', i === confirmed));
    $('clipLabel').textContent = `${stamp(result.matches[confirmed].start_s)}–${stamp(result.matches[confirmed].end_s)} · playing in native replay`;
  } else if (confirmed >= 0) playInterval(result.matches[confirmed], confirmed);
}

function render(next) {
  state = next;
  const busy = next.busy || requestPending;
  const native = next.native || {};
  const captureOn = native.available && (native.enabled || native.recording);
  const voice = next.voice || native.voice || {};
  const voiceLabels = {idle: 'Push to talk', listening: 'Listening… release to ask', transcribing: 'Transcribing your question…',
    searching: 'Finding your moment…', replaying: 'Replaying your moment', no_match: 'No confirmed moment', error: 'Voice needs attention'};
  $('voiceStatus').hidden = false;
  $('voiceStatus').dataset.state = voice.state || 'idle';
  $('voiceTitle').textContent = voiceLabels[voice.state] || 'Push to talk';
  $('voiceMessage').textContent = voice.message || (native.available
    ? 'Hold your configured push-to-talk shortcut, ask, then release. Default: 9.'
    : 'Start native recording, then hold your push-to-talk shortcut in the game. Default: 9.');
  $('voiceTranscript').textContent = voice.transcript ? `“${voice.transcript}”` : '';
  $('voiceTranscript').hidden = !voice.transcript;
  $('voiceModel').textContent = voice.model?.message || '';
  $('voiceModel').hidden = !voice.model?.message;
  $('record').disabled = captureRequests > 0 || !!next.capture_control_busy;
  $('record').hidden = captureOn;
  $('stop').hidden = !captureOn;
  $('stop').disabled = captureRequests > 0 || !!next.capture_control_busy;
  $('recordProgress').hidden = !native.recording;
  $('recordFill').style.width = native.recording ? '100%' : '0';
  const hasFootage = next.match || next.selected_session_id;
  $('question').disabled = busy || !hasFootage;
  $('ask').disabled = busy || !hasFootage || !$('question').value.trim();
  $('searchScope').disabled = busy || !next.selected_session_id;
  $('windowStart').disabled = busy;
  $('windowControls').hidden = $('searchScope').value !== 'window' || !next.selected_session_id;
  $('fullSession').disabled = busy || !next.selected_session_id;
  const sessions = next.sessions || [];
  const signature = sessions.map(item => item.id).join('|');
  if (signature !== sessionSignature) {
    sessionSignature = signature;
    $('sessionPicker').replaceChildren();
    if (!sessions.length) {
      const option = document.createElement('option'); option.value = ''; option.textContent = 'No recorded session yet';
      $('sessionPicker').append(option);
    }
    sessions.forEach(item => {
      const option = document.createElement('option'); option.value = item.id;
      option.textContent = `${item.game || 'Game'} · ${new Date(item.created_at).toLocaleTimeString()}`;
      $('sessionPicker').append(option);
    });
  }
  $('sessionPicker').value = next.selected_session_id || '';
  $('sessionPicker').disabled = busy || !sessions.length;
  const selected = sessions.find(item => item.id === next.selected_session_id);
  const index = next.index || {};
  $('indexNote').textContent = selected ? `Indexed ${stamp(index.indexed_until_s || 0)} / ${stamp(selected.duration_s)}${index.indexing ? ' · indexing in background' : ''}${index.error ? ' · ' + index.error : ''}${index.retry_after_s ? ' · retry in ' + Math.ceil(index.retry_after_s) + 's' : ''}` : 'Choose a session or import a short recording.';
  const cache = native.cache || {};
  $('cacheNote').textContent = `Replay cache: ${((Number(cache.bytes) || 0) / 1048576).toFixed(1)} / ${((Number(cache.budget_bytes) || 134217728) / 1048576).toFixed(0)} MiB · full sessions remain on disk.`;
  $('checkModel').disabled = busy;
  $('backend').disabled = busy;
  $('backend').value = next.backend || 'codex';
  document.querySelectorAll('[data-question]').forEach(button => { button.disabled = busy || !hasFootage; });
  $('gameBadge').textContent = native.recording ? 'Recording' : captureOn ? 'Waiting for game' : next.game.running ? 'Game ready' : 'Open the game';
  $('gameBadge').classList.toggle('ready', native.recording || next.game.running);
  $('modelDot').className = `dot ${next.model.state}`;
  $('modelTitle').textContent = {connected: 'Model connected', error: 'Model needs attention', checking: 'Checking model…', unchecked: 'Model connection'}[next.model.state];
  $('modelMessage').textContent = next.model.message;
  const visibleError = requestError || next.error || native.error;
  $('error').hidden = !visibleError;
  $('error').textContent = visibleError;
  const messages = {
    idle: selected ? 'Session available. Ask about a moment, or open the full recording.' : 'Start native recording, then switch to Stick Fight.',
    recording: `Recording the game · ${Math.min(next.record_seconds, Math.floor(next.recorded_s))} / ${next.record_seconds} seconds`,
    finalizing: 'Preserving your match…',
    checking: 'Checking the live model connection…',
    transcribing: 'Transcribing your question locally. Native recording continues independently…',
    searching: 'Finding and checking the closest moments. Native recording continues independently…',
    preparing: 'Preparing the full-session replay. Recording continues independently…',
    ready: next.result ? (next.result.matches.length ? `${next.result.matches.length} candidate moment${next.result.matches.length === 1 ? '' : 's'} · searched in ${next.result.elapsed_s}s` : 'No matching moment was confirmed. Try another description.') : 'Match preserved. Ask about something you just did.'
  };
  $('status').textContent = messages[next.phase] || '';
  $('status').classList.toggle('busy', busy);
  const id = next.match?.id || null;
  if (id !== currentMatch) {
    currentMatch = id; shownAnswer = null; clip = null;
    $('player').pause();
    $('player').hidden = !id; $('emptyState').hidden = !!id;
    if (id) { $('player').src = next.match.url; $('player').load(); }
    else { $('player').removeAttribute('src'); $('player').load(); }
    $('results').replaceChildren();
    $('clipLabel').textContent = id ? `${next.match.duration.toFixed(1)} seconds · full recording` : 'Record a match to begin';
  }
  $('matchLabel').textContent = id ? `${next.match.source === 'native' ? 'Native session' : next.match.source === 'live' ? 'Fresh capture' : 'Imported footage'} · ${next.match.duration.toFixed(1)}s` : selected ? `${stamp(selected.duration_s)} saved` : 'No session yet';
  $('wholeMatch').disabled = !id;
  $('captureNote').textContent = native.recording ? `Native recording continues while you ask and replay${selected ? ' · ' + stamp(selected.duration_s) + ' finalized on disk' : ''}.` : captureOn ? 'Recorder is armed. Switch to the game to begin; you can return here afterward.' : 'Start recording once. Full sessions stay on disk until you choose to remove them.';
  const answer = next.result?.answered_at || null;
  if (answer !== shownAnswer) {
    shownAnswer = answer;
    clip = null;
    $('player').pause();
    $('clipLabel').textContent = id ? `${next.match.duration.toFixed(1)} seconds · full recording` : 'Record a match to begin';
    if (next.result?.match_id === id) showResults(next.result);
    else $('results').replaceChildren();
  }
}

async function poll() {
  try {
    const response = await fetch('/api/status');
    if (!response.ok) throw new Error('Server unavailable');
    render(await response.json());
  } catch (_) {
    $('error').hidden = false;
    $('error').textContent = 'The demo server is not responding. Keep its launcher running, then refresh.';
  }
}
$('record').addEventListener('click', () => post('/api/record'));
$('stop').addEventListener('click', () => post('/api/stop'));
$('checkModel').addEventListener('click', () => post('/api/check-model'));
$('backend').addEventListener('change', async () => {
  const backend = $('backend').value;
  await post('/api/backend', {backend});
  if (state.backend === backend && !state.busy) await post('/api/check-model');
});
$('launch').addEventListener('click', () => post('/api/launch-game'));
$('sessionPicker').addEventListener('change', () => post('/api/select-session', {session_id: $('sessionPicker').value}));
$('searchScope').addEventListener('change', () => { if (state) render(state); });
$('fullSession').addEventListener('click', () => post('/api/full-session', {session_id: state.selected_session_id}));
$('questionForm').addEventListener('submit', event => {
  event.preventDefault();
  if (!state || state.busy || requestPending) return;
  if (state.selected_session_id) {
    post('/api/ask', {question: $('question').value.trim(), session_id: state.selected_session_id,
      scope: $('searchScope').value || 'session', start_s: Number($('windowStart').value)});
  } else if (state.match) post('/api/ask', {question: $('question').value.trim(), match_id: state.match.id});
});
$('question').addEventListener('input', () => { if (state) render(state); });
document.querySelectorAll('[data-question]').forEach(button => button.addEventListener('click', () => {
  $('question').value = button.dataset.question; $('question').focus(); if (state) render(state);
}));
$('wholeMatch').addEventListener('click', () => {
  clip = null; $('player').currentTime = 0; $('player').play().catch(() => {});
  $('clipLabel').textContent = `${state.match.duration.toFixed(1)} seconds · full recording`;
  document.querySelectorAll('.result-card').forEach(card => card.classList.remove('selected'));
});
$('player').addEventListener('timeupdate', () => {
  if (clip && $('player').currentTime >= clip.end_s) $('player').pause();
});
let polling = false;
setInterval(async () => { if (!polling) { polling = true; try { await poll(); } finally { polling = false; } } }, 700);
poll();
