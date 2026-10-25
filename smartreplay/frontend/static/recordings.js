"use strict";

function recordingTime(seconds) {
  const total = Math.max(0, Math.floor(Number(seconds) || 0));
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, '0')}`;
}

class RecordingViewer {
  constructor(doc) {
    this.doc = doc;
    const get = id => doc.getElementById(id);
    this.list = get('recording-list'); this.video = get('recording-video');
    this.title = get('recording-title'); this.status = get('recording-message');
    this.searchStatus = get('recording-search-status'); this.results = get('recording-results');
    this.question = get('recording-question'); this.voiceButton = get('recording-voice');
    this.searchButton = get('recording-search'); this.speed = get('recording-speed');
    this.rows = []; this.selected = null; this.generation = 0; this.searchGeneration = 0;
    this.playbackGeneration = 0; this.voiceGeneration = 0;
    this.clipEnd = null; this.recorder = null; this.micPending = false;
  }

  async request(path, options = {}) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 15000);
    try {
      const response = await fetch(path, {...options, signal: controller.signal});
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'The request failed. Please try again.');
      return data;
    } finally { clearTimeout(timer); }
  }

  async refresh() {
    if (this.refreshing) return;
    this.refreshing = true;
    try {
      const data = await this.request('/api/recordings');
      this.rows = data.recordings;
      this.renderList();
      this.doc.getElementById('recording-count').textContent = `${this.rows.length} recording${this.rows.length === 1 ? '' : 's'}`;
    } catch (error) { this.status.textContent = error.message; }
    finally { this.refreshing = false; }
  }

  renderList() {
    this.list.replaceChildren();
    if (!this.rows.length) {
      const empty = this.doc.createElement('p'); empty.className = 'recording-empty';
      empty.textContent = 'No recordings yet. Open a game with deja running. Saved footage will appear here.';
      this.list.append(empty); return;
    }
    for (const row of this.rows) {
      const button = this.doc.createElement('button');
      button.type = 'button'; button.className = 'recording-card';
      button.classList.toggle('selected', row.id === this.selected);
      button.setAttribute('aria-pressed', String(row.id === this.selected));
      const name = this.doc.createElement('strong'); name.textContent = row.game || 'Game recording';
      const date = new Date(row.created_at);
      const detail = this.doc.createElement('span');
      detail.textContent = `${Number.isNaN(date.getTime()) ? 'Unknown date' : date.toLocaleString()} · ${recordingTime(row.duration_s)}`;
      const badge = this.doc.createElement('small');
      badge.textContent = !row.playable ? 'Waiting for saved footage / unavailable' : row.state === 'recording' ? 'Recording · watch saved footage' : 'Saved session';
      button.append(name, detail, badge);
      button.addEventListener('click', () => this.select(row.id));
      this.list.append(button);
    }
  }

  async select(identity) {
    this.stopVoice(true);
    const generation = ++this.generation;
    const playback = ++this.playbackGeneration;
    ++this.searchGeneration;
    this.selected = identity; this.clipEnd = null;
    this.clearVideo();
    this.results.replaceChildren(); this.searchStatus.textContent = ''; this.question.value = '';
    this.searchButton.disabled = false; this.voiceButton.disabled = false;
    this.renderList();
    const row = this.rows.find(x => x.id === identity);
    this.title.textContent = row?.game || 'Recording';
    this.status.textContent = 'Preparing your recording… You can keep playing while it loads.';
    try {
      let job = await this.request(`/api/recordings/${identity}/prepare`, {method: 'POST'});
      while (job.state === 'preparing' && generation === this.generation) {
        await new Promise(resolve => setTimeout(resolve, 750));
        if (generation !== this.generation) return;
        job = await this.request(`/api/recordings/${identity}/playback`);
      }
      if (generation !== this.generation || playback !== this.playbackGeneration) return;
      if (job.state !== 'ready') throw new Error(job.error || 'This recording is not available.');
      this.loadVideo(job.url, 0);
      this.status.textContent = row?.state === 'recording'
        ? 'Watching footage saved so far. Open this session again for the latest footage.'
        : 'Full session · use the timeline to find your moment.';
    } catch (error) {
      if (generation === this.generation) this.status.textContent = error.message;
    }
  }

  loadVideo(url, start, end = null) {
    this.clipEnd = end;
    this.video.onloadedmetadata = () => {
      this.video.currentTime = start;
      this.video.playbackRate = Number(this.speed.value) || 1;
      this.video.play().catch(() => { this.status.textContent = 'Press Play to watch this recording.'; });
    };
    this.video.src = url;
    this.video.load();
  }

  clearVideo() {
    this.video.onloadedmetadata = null;
    this.video.pause(); this.video.removeAttribute('src'); this.video.load(); this.clipEnd = null;
  }

  async runSearch(path, options) {
    const generation = this.generation, search = ++this.searchGeneration;
    ++this.playbackGeneration;
    this.clearVideo(); this.results.replaceChildren();
    this.searchButton.disabled = true; this.voiceButton.disabled = true;
    this.searchStatus.textContent = path.endsWith('/voice') ? 'Transcribing your question locally…' : 'Searching the selected recording…';
    try {
      let job = await this.request(path, options);
      while (['transcribing', 'searching'].includes(job.state) && generation === this.generation && search === this.searchGeneration) {
        if (job.question) this.question.value = job.question;
        this.searchStatus.textContent = job.state === 'transcribing' ? 'Transcribing your question locally…' : 'Finding and checking the matching moment…';
        await new Promise(resolve => setTimeout(resolve, 750));
        if (generation !== this.generation || search !== this.searchGeneration) return;
        job = await this.request(`/api/recordings/${job.session_id}/search/${job.id}`);
      }
      if (generation !== this.generation || search !== this.searchGeneration) return;
      if (job.state === 'error') throw new Error(job.error);
      this.showSearch(job);
    } catch (error) {
      if (generation === this.generation && search === this.searchGeneration) this.searchStatus.textContent = error.message;
    } finally {
      if (generation === this.generation && search === this.searchGeneration) {
        this.searchButton.disabled = false; this.voiceButton.disabled = false;
      }
    }
  }

  showSearch(job) {
    this.clearVideo(); this.results.replaceChildren();
    this.question.value = job.question || this.question.value;
    const result = job.result || {}, matches = result.matches || [];
    const playable = matches.filter(m => job.url && Number.isFinite(m.start_s) && Number.isFinite(m.end_s)
      && m.start_s >= 0 && m.end_s > m.start_s && m.end_s <= job.duration_s);
    this.searchStatus.textContent = (playable.length ? 'Select a result to watch it. ' : 'No matching moment was confirmed. ') + (result.note || '');
    for (const match of playable) {
      const button = this.doc.createElement('button'); button.type = 'button'; button.className = 'recording-result';
      button.textContent = `${match.verdict === 'yes' ? 'Watch' : 'Uncertain candidate'} · ${recordingTime(match.session_time_s ?? match.time_s)} · ${match.evidence || match.description || 'Matching moment'}`;
      button.addEventListener('click', () => {
        this.loadVideo(job.url, match.start_s, match.end_s);
        this.status.textContent = 'Search result · open the session again to watch the full recording.';
      });
      this.results.append(button);
    }
  }

  stopVoice(discard = false) {
    if (discard) ++this.voiceGeneration;
    if (this.voiceCapture) this.voiceCapture.discard = discard || this.voiceCapture.discard;
    if (this.recorder && this.recorder.state !== 'inactive') this.recorder.stop();
  }

  async toggleVoice() {
    if (this.recorder?.state === 'recording') { this.stopVoice(); return; }
    if (!this.selected || this.micPending) return;
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
      this.searchStatus.textContent = 'Microphone recording is unavailable here. Type your question instead.'; return;
    }
    const generation = this.generation, identity = this.selected;
    const voiceGeneration = ++this.voiceGeneration;
    this.micPending = true;
    this.searchStatus.textContent = 'Allow microphone access to ask a question.';
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({audio: true});
      if (generation !== this.generation || voiceGeneration !== this.voiceGeneration) { stream.getTracks().forEach(t => t.stop()); return; }
      const type = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus'].find(t => MediaRecorder.isTypeSupported(t));
      const recorder = new MediaRecorder(stream, type ? {mimeType: type} : {});
      const capture = {discard: false}; this.voiceCapture = capture;
      const chunks = []; this.recorder = recorder;
      recorder.ondataavailable = e => { if (e.data.size) chunks.push(e.data); };
      const timer = setTimeout(() => { if (recorder.state === 'recording') recorder.stop(); }, 15000);
      recorder.onstop = () => {
        clearTimeout(timer); stream.getTracks().forEach(t => t.stop());
        if (this.recorder === recorder) this.recorder = null;
        if (generation !== this.generation) return;
        this.voiceButton.textContent = 'Ask with microphone';
        this.searchButton.disabled = false;
        if (capture.discard || generation !== this.generation) return;
        const blob = new Blob(chunks, {type: recorder.mimeType || 'audio/webm'});
        this.runSearch(`/api/recordings/${identity}/voice`, {method: 'POST', headers: {'Content-Type': blob.type}, body: blob});
      };
      recorder.onerror = () => {
        capture.discard = true; clearTimeout(timer); stream.getTracks().forEach(t => t.stop());
        if (recorder.state !== 'inactive') recorder.stop();
        if (generation === this.generation) this.searchStatus.textContent = 'Microphone recording failed. Try again or type your question.';
      };
      recorder.start(); this.searchButton.disabled = true;
      this.voiceButton.textContent = 'Stop and search';
      this.searchStatus.textContent = 'Listening… Click Stop and search when finished (up to 15 seconds).';
    } catch (error) {
      stream?.getTracks().forEach(t => t.stop());
      if (generation === this.generation) this.searchStatus.textContent = 'Microphone access failed. Allow microphone access, or type your question.';
    } finally { this.micPending = false; }
  }

  init() {
    this.doc.getElementById('recording-refresh').addEventListener('click', () => this.refresh());
    this.doc.getElementById('recording-search-form').addEventListener('submit', event => {
      event.preventDefault();
      if (this.selected && this.question.value.trim()) this.runSearch(`/api/recordings/${this.selected}/search`, {
        method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({question: this.question.value})});
    });
    this.voiceButton.addEventListener('click', () => this.toggleVoice());
    this.speed.addEventListener('change', () => { this.video.playbackRate = Number(this.speed.value); });
    this.video.addEventListener('timeupdate', () => { if (this.clipEnd !== null && this.video.currentTime >= this.clipEnd) this.video.pause(); });
    this.video.addEventListener('error', () => { if (this.video.getAttribute('src')) this.status.textContent = 'This recording could not be played. Open it again or select another session.'; });
    const route = () => {
      const viewing = !location.hash || location.hash === '#recordings';
      this.doc.body.classList.toggle('recordings-view', viewing);
      this.doc.getElementById('tab-recordings').setAttribute('aria-current', viewing ? 'page' : 'false');
      this.doc.getElementById('tab-settings').setAttribute('aria-current', viewing ? 'false' : 'page');
      if (!viewing) { this.video.pause(); this.stopVoice(true); }
    };
    window.addEventListener('hashchange', route);
    window.addEventListener('beforeunload', () => this.stopVoice(true));
    route(); this.refresh();
    setInterval(() => { if (!this.doc.hidden) this.refresh(); }, 10000);
  }
}

if (typeof module !== 'undefined') module.exports = {RecordingViewer, recordingTime};
if (typeof document !== 'undefined') new RecordingViewer(document).init();
