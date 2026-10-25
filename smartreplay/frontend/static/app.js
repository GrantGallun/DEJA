"use strict";
// Settings app for the replay player. The server (player_settings.py) owns
// validation; this page edits a draft, asks the server what is wrong with it,
// and saves it whole. The mock player mirrors ReplayWindow.qml's metrics, so
// what fits here fits there.

const HOTKEYS = [
  ["toggle_replay", "Show / hide replay", "Opens your recent replay at the configured lookback, or closes it."],
  ["toggle_controls", "Take / return controls", "Summons a hidden replay and takes controls, or returns controls to the game."],
  ["push_to_talk", "Hold to ask about gameplay", "Hold to speak, then release to find and replay a matching moment. Uses the native recorder's microphone."],
  ["quit", "Quit player", "Closes the replay player."],
];
const PLAYER_KEYS = [
  ["play_pause", "Play / pause"], ["seek_back", "Seek back"], ["seek_forward", "Seek forward"],
  ["speed_down", "Slower"], ["speed_up", "Faster"], ["mute", "Mute"],
  ["restart", "Jump to start"], ["history", "Full history"], ["expand", "Expand to screen"], ["close", "Close replay"],
];
const SPEEDS = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3, 4];
// Preset widths; the height fits a 16:9 video plus whatever chrome is on.
const SIZES = [["Small", 400], ["Medium", 536], ["Large", 720], ["Huge", 960]];
const ANCHORS = ["top_left", "top_center", "top_right", "middle_left", "center", "middle_right",
                 "bottom_left", "bottom_center", "bottom_right"];
const COLOURS = [["background", "Video background"], ["panel", "Control panel"], ["header", "Title bar"],
                 ["header_active", "Title bar in control"], ["text", "Text"], ["muted", "Secondary text"],
                 ["accent", "Accent"]];
const THEME_NAMES = { slate: "Slate", midnight: "Midnight", violet: "Violet", ember: "Ember", paper: "Paper",
                      contrast: "High contrast" };
// Shared with player_settings.BUTTON_WIDTHS and ReplayWindow.qml.
const BTN_W = { seek: 53, play_pause: 58, speed: 39, mute: 57, expand: 40, close: 34 };
const KEY_SYMBOLS = { Left: "←", Right: "→", Up: "↑", Down: "↓" };

const state = {
  saved: null, draft: null, defaults: null, themes: {}, issues: [],
  status: { running: false }, path: "", pvState: "passive", savedAt: 0, footage: null, providers: {},
};
const syncers = [];

// ------------------------------------------------------------------ helpers

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const clone = (v) => JSON.parse(JSON.stringify(v));
const get = (obj, path) => path.split(".").reduce((o, k) => (o == null ? o : o[k]), obj);
function set(obj, path, value) {
  const keys = path.split(".");
  const last = keys.pop();
  keys.reduce((o, k) => o[k], obj)[last] = value;
}
function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "style") Object.assign(el.style, v);
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c == null || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}
const fmtSpeed = (s) => `${s}×`;
const px = (n) => `${Math.round(n)}px`;

function toast(text, isError = false) {
  const t = $("#toast");
  t.textContent = text;
  t.className = "toast" + (isError ? " error" : "");
  t.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { t.hidden = true; }, isError ? 6000 : 2600);
}

async function api(method, url, body) {
  const res = await fetch(url, {
    method,
    headers: body === undefined ? {} : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  return { ok: res.ok, status: res.status, data };
}

// ------------------------------------------------------------------ chords

function chordParts(chord) { return chord ? chord.split("+").filter(Boolean) : []; }
function keycaps(chord, empty = "Not set") {
  const parts = chordParts(chord);
  if (!parts.length) return [h("span", { class: "unbound" }, empty)];
  const out = [];
  parts.forEach((p, i) => {
    if (i) out.push(h("span", { class: "plus" }, "+"));
    out.push(h("kbd", { title: p }, KEY_SYMBOLS[p] || p));
  });
  return out;
}
function plainChord(chord) {
  return chordParts(chord).map((p) => KEY_SYMBOLS[p] || p).join("+");
}

const CODE_NAMES = {
  Space: "Space", Escape: "Esc", Tab: "Tab", Backspace: "Backspace", Enter: "Return",
  NumpadEnter: "Enter", Insert: "Ins", Delete: "Del", Home: "Home", End: "End",
  PageUp: "PgUp", PageDown: "PgDown", ArrowLeft: "Left", ArrowRight: "Right",
  ArrowUp: "Up", ArrowDown: "Down", Comma: ",", Period: ".", Slash: "/", Semicolon: ";",
  Quote: "'", BracketLeft: "[", BracketRight: "]", Backslash: "\\", Minus: "-",
  Equal: "=", Backquote: "`",
};
const MODIFIER_KEYS = new Set(["Control", "Alt", "Shift", "Meta", "AltGraph", "OS"]);

function keyFromEvent(e) {
  const c = e.code;
  if (/^Key[A-Z]$/.test(c)) {
    // Prefer the layout's letter (AZERTY's A is KeyQ) when the key reports one.
    return e.key && /^[a-z]$/i.test(e.key) ? e.key.toUpperCase() : c.slice(3);
  }
  if (/^Digit\d$/.test(c)) return c.slice(5);
  if (/^F\d{1,2}$/.test(c) && +c.slice(1) <= 24) return c;
  return CODE_NAMES[c] || null;
}
function heldModifiers(e) {
  return [e.ctrlKey && "Ctrl", e.altKey && "Alt", e.shiftKey && "Shift"].filter(Boolean);
}
// Immediate feedback while recording; the server re-checks everything.
function captureProblem(mods, key, isGlobal, metaKey) {
  if (metaKey) return "The Windows key is reserved for system shortcuts.";
  if (!key) return "That key isn't supported.";
  if (!isGlobal) return null;
  if (!mods.length && !/^[0-9]$/.test(key)) return "Use a number-row digit, or add Ctrl, Alt or Shift.";
  if (!/^([A-Z0-9]|F\d{1,2})$/.test(key)) return "Global hotkeys use a letter, digit or F-key.";
  if (mods.length === 1 && mods[0] === "Alt" && key === "F4") return "Alt+F4 is reserved by Windows.";
  return null;
}

function keyCapture({ isGlobal, path, label }) {
  const btn = h("button", { type: "button", class: "capture", "aria-label": `${label} shortcut` });
  const msg = h("div", { class: "key-msg", hidden: true });
  let recording = false;
  let endedAt = 0;

  const show = () => {
    btn.replaceChildren(...keycaps(get(state.draft, path)));
    btn.setAttribute("aria-label", `${label}: ${plainChord(get(state.draft, path)) || "not set"}. Press Enter to change.`);
  };
  const stop = () => {
    if (!recording) return;
    recording = false;
    endedAt = performance.now();
    btn.classList.remove("recording");
    msg.hidden = true;
    show();
  };
  const start = () => {
    recording = true;
    btn.classList.add("recording");
    btn.replaceChildren(h("span", { class: "prompt" }, "Press keys…"));
    msg.hidden = false;
    msg.className = "key-msg";
    msg.textContent = isGlobal ? "Press a number-row digit, or hold Ctrl, Alt or Shift and press a key. Tab cancels." : "Press a key. Tab cancels.";
  };
  btn.addEventListener("click", () => {
    if (performance.now() - endedAt < 350) return; // the Space/Enter that ended a capture
    recording ? stop() : start();
  });
  btn.addEventListener("blur", stop);
  btn.addEventListener("keydown", (e) => {
    if (!recording) return;
    if (e.key === "Tab" && !e.ctrlKey && !e.altKey) return stop(); // let focus move on
    e.preventDefault();
    e.stopPropagation();
    const mods = heldModifiers(e);
    if (MODIFIER_KEYS.has(e.key)) {
      btn.replaceChildren(...mods.flatMap((m, i) => [i ? h("span", { class: "plus" }, "+") : null, h("kbd", {}, m)]),
                          h("span", { class: "plus" }, "+ …"));
      return;
    }
    const key = keyFromEvent(e);
    const problem = captureProblem(mods, key, isGlobal, e.metaKey);
    if (problem) {
      msg.className = "key-msg error";
      msg.textContent = problem;
      return;
    }
    set(state.draft, path, [...mods, key].join("+"));
    stop();
    changed();
  });
  btn.addEventListener("keyup", (e) => {
    if (recording && MODIFIER_KEYS.has(e.key) && !heldModifiers(e).length) {
      btn.replaceChildren(h("span", { class: "prompt" }, "Press keys…"));
    }
  });
  syncers.push(() => { if (!recording) show(); });
  return { btn, msg, isRecording: () => recording };
}

function keyRow({ action, label, help, group, isGlobal }) {
  const path = `${group}.${action}`;
  const cap = keyCapture({ isGlobal, path, label });
  const reg = h("span", { class: "reg", hidden: true });
  const clear = h("button", { type: "button", class: "icon-btn", title: "Unbind", "aria-label": `Unbind ${label}`,
    onclick: () => { set(state.draft, path, ""); changed(); } }, "✕");
  const reset = h("button", { type: "button", class: "icon-btn", title: "Back to default", "aria-label": `Reset ${label} to default`,
    onclick: () => { set(state.draft, path, get(state.defaults, path)); changed(); } }, "↺");
  const issue = h("div", { class: "key-msg", hidden: true });
  const row = h("div", { class: "keyrow", "data-issue-path": path },
    h("div", { class: "key-name" }, label, help ? h("small", {}, help) : null),
    isGlobal ? reg : h("span"), cap.btn, h("div", { style: { display: "flex" } }, clear, reset), cap.msg, issue);
  syncers.push(() => {
    const value = get(state.draft, path);
    clear.disabled = !value;
    reset.disabled = value === get(state.defaults, path);
    const mine = state.issues.filter((i) => i.path === path);
    issue.hidden = !mine.length;
    issue.className = "key-msg " + (mine.some((i) => i.level === "error") ? "error" : "warning");
    issue.textContent = mine.map((i) => i.message).join(" ");
    cap.btn.classList.toggle("invalid", mine.some((i) => i.level === "error"));
    if (group === "player_keys") row.classList.toggle("disabled", state.draft.input.mode === "mouse");
    if (isGlobal) renderRegistration(reg, action);
  });
  return row;
}

function renderRegistration(el, action) {
  const st = state.status;
  const info = st.running && st.hotkeys && st.hotkeys[action];
  const savedChord = state.saved && state.saved.hotkeys[action];
  // Only report on what the player has for the saved chord, never the draft.
  if (!info || (info.state !== "override" && info.chord !== savedChord)) { el.hidden = true; return; }
  const text = { registered: "Active", failed: "Taken", invalid: "Rejected", unbound: "Off",
                 override: `Overridden: ${info.chord}` }[info.state] || info.state;
  el.hidden = false;
  el.className = `reg ${info.state}`;
  el.textContent = text;
  el.title = info.detail || (info.state === "registered" ? "Registered with Windows by the running player." : "");
}

// ------------------------------------------------------------------ generic inputs

function bindInputs() {
  for (const el of $$("[data-path]")) {
    const path = el.dataset.path;
    const radios = $$("input[type=radio]", el);
    if (radios.length) {
      radios.forEach((r) => r.addEventListener("change", () => { if (r.checked) { set(state.draft, path, r.value); changed(); } }));
      syncers.push(() => radios.forEach((r) => { r.checked = r.value === get(state.draft, path); }));
    } else if (el.type === "checkbox") {
      el.addEventListener("change", () => { set(state.draft, path, el.checked); changed(); });
      syncers.push(() => { el.checked = !!get(state.draft, path); });
    } else if (el.type === "range" || el.type === "number") {
      const isInt = !String(el.step).includes(".");
      el.addEventListener("input", () => {
        if (el.value === "") return;
        set(state.draft, path, isInt ? Math.round(+el.value) : Math.round(+el.value * 100) / 100);
        changed();
      });
      syncers.push(() => { if (document.activeElement !== el || el.type === "range") el.value = get(state.draft, path); });
    } else if (el.tagName === "SELECT") {
      el.addEventListener("change", () => { set(state.draft, path, el.value); changed(); });
      syncers.push(() => { el.value = get(state.draft, path); });
    }
  }
  for (const out of $$("output[data-out]")) {
    syncers.push(() => { out.textContent = `${Math.round(get(state.draft, out.dataset.out) * 100)}%`; });
  }
  const seekOut = $("#seek-step-out");
  syncers.push(() => { seekOut.textContent = `${state.draft.input.seek_step_s} s`; });
}

// Roving-tabindex radio group for button grids (themes, anchors).
function radioButtons(container, values, { current, choose, columns = values.length, render }) {
  const buttons = values.map((v) => {
    const b = h("button", { type: "button", role: "radio" }, render(v));
    b.addEventListener("click", () => choose(v));
    b.addEventListener("keydown", (e) => {
      const i = values.indexOf(v);
      const step = { ArrowRight: 1, ArrowLeft: -1, ArrowDown: columns, ArrowUp: -columns }[e.key];
      if (step === undefined) return;
      e.preventDefault();
      const j = i + step;
      if (j < 0 || j >= values.length) return;
      choose(values[j]);
      buttons[j].focus();
    });
    container.append(b);
    return b;
  });
  syncers.push(() => {
    const cur = current();
    const anyChecked = values.includes(cur);
    buttons.forEach((b, i) => {
      const on = values[i] === cur;
      b.setAttribute("aria-checked", on);
      b.tabIndex = on || (!anyChecked && i === 0) ? 0 : -1;
    });
  });
  return buttons;
}

// ------------------------------------------------------------------ sections

function buildHotkeys() {
  const g = $("#global-keys");
  for (const [action, label, help] of HOTKEYS) g.append(keyRow({ action, label, help, group: "hotkeys", isGlobal: true }));
  const p = $("#player-keys");
  for (const [action, label] of PLAYER_KEYS) p.append(keyRow({ action, label, group: "player_keys", isGlobal: false }));
  syncers.push(() => {
    const mouse = state.draft.input.mode === "mouse";
    $("#mouse-keys-off").hidden = !mouse;
    $("#player-keys-hint").textContent = state.draft.input.mode === "keyboard"
      ? "Work only while the player has controls, so plain keys are fine. In Keyboard only mode these are the only way to control playback."
      : "Work only while the player has controls, so plain keys are fine.";
  });
}

function behaviourRows(s) {
  // Functions, not nodes: one node can only sit in one table cell.
  const tc = () => (s.hotkeys.toggle_controls ? h("kbd", {}, plainChord(s.hotkeys.toggle_controls)) : null);
  const tr = () => h("kbd", {}, plainChord(s.hotkeys.toggle_replay) || "—");
  const or = (a, b) => (a && b ? [a, " or ", b] : [a || b]);
  const keys = PLAYER_KEYS.filter(([a]) => s.player_keys[a]).slice(0, 3)
    .map(([a, l]) => [h("kbd", {}, plainChord(s.player_keys[a])), ` ${l.toLowerCase()}`]);
  const keyList = keys.length ? keys.flatMap((k, i) => (i ? [", ", ...k] : k)) : ["none bound"];
  const rows = {
    keyboard: [
      ["Passive replay", "Click-through: the game gets every click."],
      ["Take controls", tc() || "No hotkey set"],
      ["Give controls back", or(tc(), "click the game")],
      ["Playback", ["Player keys only (", ...keyList, "…). Buttons show what's happening but ignore clicks."]],
      ["Move and resize", "Not with the mouse: set size and position on this page."],
    ],
    mouse: [
      ["Passive replay", "Clickable: click it to take controls."],
      ["Take controls", or("Click the player", tc())],
      ["Give controls back", or("Click the game", tc())],
      ["Playback", "Buttons and timeline; scroll to magnify, drag to pan. Player keys are off."],
      ["Move and resize", "Drag the title bar (or the video when the title bar is hidden); drag any edge or corner."],
    ],
    hybrid: [
      ["Passive replay", "Click-through: the game gets every click."],
      ["Take controls", tc() || "No hotkey set"],
      ["Give controls back", or(tc(), "click the game")],
      ["Playback", ["Player keys (", ...keyList, "…), buttons and timeline; scroll to magnify, drag to pan."]],
      ["Move and resize", s.appearance.show_header
        ? "Drag the title bar; drag any edge or corner. Double-click the title bar to fill the screen."
        : "Drag the video; drag any edge or corner. Double-click the video to fill the screen."],
    ],
  }[s.input.mode];
  rows.push(["Close", [tr(), " any time, from anywhere"]]);
  return rows;
}

function buildFocus() {
  const box = $("#behaviour");
  syncers.push(() => {
    const s = state.draft;
    const table = h("table", {}, h("tbody", {}, behaviourRows(s).map(([k, v]) => h("tr", {}, h("th", { scope: "row" }, k), h("td", {}, v)))));
    const note = s.input.mode === "mouse"
      ? h("div", { class: "warn-note" }, "While a replay is showing in this mode it blocks clicks on the part of the game it covers.")
      : null;
    box.replaceChildren(table, note || "");
  });
}

// Height of the title bar and control panel, as ReplayWindow.qml lays them
// out. Minimal controls sit on the video and take none.
function chromeHeight(s) {
  const a = s.appearance, btn = Math.round(28 * a.font_scale);
  const header = a.show_header ? Math.round(36 * a.font_scale) : 0;
  const info = a.show_status ? 6 + Math.round(14 * a.font_scale) : 0;
  const controls = { full: 8 + 20 + 6 + btn + info + 8, compact: 8 + btn + 8, timeline: 8 + 20 + 8 }[a.controls] || 0;
  return header + controls;
}
const presetHeight = (s, w) => Math.round((w * 9) / 16) + chromeHeight(s);

function buildSize() {
  const chips = $("#size-presets");
  for (const [name, w] of SIZES) {
    const dims = h("small");
    const b = h("button", { type: "button", class: "chip", onclick: () => {
      state.draft.layout.width = w; state.draft.layout.height = presetHeight(state.draft, w); changed();
    } }, name, dims);
    chips.append(b);
    syncers.push(() => {
      const hgt = presetHeight(state.draft, w);
      dims.textContent = `${w}×${hgt}`;
      b.setAttribute("aria-pressed", state.draft.layout.width === w && state.draft.layout.height === hgt);
    });
  }
  radioButtons($("#anchor-grid"), ANCHORS, {
    columns: 3,
    current: () => state.draft.layout.anchor,
    choose: (v) => { state.draft.layout.anchor = v; changed(); },
    render: (v) => h("span", { class: "sr-only" }, v.replace("_", " ")),
  });
}

function buildModel() {
  const note = $("#provider-note");
  syncers.push(() => {
    const deepseek = state.draft.model.provider === "deepseek";
    const hasKey = state.providers.deepseek?.key;
    note.hidden = !deepseek;
    note.className = "banner " + (hasKey ? "info" : "warn");
    note.textContent = hasKey
      ? "DeepSeek API key found."
      : "No DeepSeek API key found. Add DEEPSEEK_API_KEY=… to the .env file in the Smart Replay folder, then save again.";
  });
}

function matchingTheme(colors) {
  return Object.keys(state.themes).find((t) =>
    COLOURS.every(([k]) => state.themes[t][k] === colors[k].toLowerCase())) || "custom";
}

function buildLook() {
  const names = [...Object.keys(state.themes), "custom"];
  radioButtons($("#themes"), names, {
    current: () => state.draft.appearance.theme,
    choose: (t) => {
      if (t === "custom") { state.draft.appearance.theme = "custom"; $(".colours").open = true; changed(); return; }
      state.draft.appearance.theme = t;
      state.draft.appearance.colors = clone(state.themes[t]);
      changed();
    },
    render: (t) => {
      const c = state.themes[t];
      const sw = c
        ? h("span", { class: "swatch" }, h("i", { style: { background: c.header } }),
            h("span", { class: "sw-mid", style: { background: c.background } }, h("b", { style: { background: c.accent } })),
            h("i", { style: { background: c.panel } }))
        : h("span", { class: "swatch", id: "custom-swatch" });
      return [sw, THEME_NAMES[t] || "Custom"];
    },
  }).forEach((b) => b.classList.add("theme"));
  syncers.push(() => {
    const c = state.draft.appearance.colors;
    const sw = $("#custom-swatch");
    sw.replaceChildren(h("i", { style: { background: c.header } }),
      h("span", { class: "sw-mid", style: { background: c.background } }, h("b", { style: { background: c.accent } })),
      h("i", { style: { background: c.panel } }));
  });

  const grid = $("#colours");
  for (const [key, label] of COLOURS) {
    const path = `appearance.colors.${key}`;
    const picker = h("input", { type: "color", "aria-label": `${label} colour` });
    const text = h("input", { type: "text", "aria-label": `${label} hex value`, spellcheck: "false", maxlength: 7 });
    const apply = (v) => {
      state.draft.appearance.colors[key] = v.toLowerCase();
      state.draft.appearance.theme = matchingTheme(state.draft.appearance.colors);
      changed();
    };
    picker.addEventListener("input", () => apply(picker.value));
    text.addEventListener("input", () => {
      const v = text.value.trim();
      const ok = /^#[0-9a-f]{6}$/i.test(v);
      text.classList.toggle("invalid", !ok);
      if (ok) apply(v);
    });
    text.addEventListener("blur", () => { text.value = get(state.draft, path); text.classList.remove("invalid"); });
    grid.append(h("label", { class: "colour", "data-issue-path": path }, picker, h("span", {}, label), text));
    syncers.push(() => {
      picker.value = get(state.draft, path);
      if (document.activeElement !== text) text.value = get(state.draft, path);
    });
  }
}

function buildButtons() {
  const chips = $("#speeds");
  for (const s of SPEEDS) {
    const b = h("button", { type: "button", class: "chip", onclick: () => {
      const list = new Set(state.draft.buttons.speeds);
      list.has(s) ? list.delete(s) : list.add(s);
      state.draft.buttons.speeds = [...list].sort((a, c) => a - c);
      changed();
    } }, fmtSpeed(s));
    chips.append(b);
    syncers.push(() => b.setAttribute("aria-pressed", state.draft.buttons.speeds.includes(s)));
  }
  const meter = $("#row-meter");
  syncers.push(() => {
    const s = state.draft;
    const layout = s.appearance.controls;
    const shown = layout === "full" || layout === "compact";
    const layoutName = { minimal: "Minimal", timeline: "Timeline only", none: "Video only" }[layout];
    $("#buttons-lede").textContent = shown
      ? "Shown in the player's controls."
      : `Hidden: the controls are set to “${layoutName}” under Appearance.`;
    meter.hidden = !shown;
    const need = buttonRowWidth(s) + (layout === "compact" ? 136 : 0);
    const have = s.layout.width;
    meter.className = "meter" + (need > have ? " over" : "");
    meter.replaceChildren(
      need ? `${layout === "compact" ? "Buttons and timeline need" : "Buttons need"} ${need} px of the player's ${have} px.` : "No buttons selected.",
      h("div", { class: "bar" }, h("i", { style: { width: `${Math.min(100, (need / have) * 100)}%` } })),
    );
  });
}

function buttonList(s) {
  const b = s.buttons, step = s.input.seek_step_s;
  const list = [];
  if (b.seek) list.push({ id: "seek", label: `−${step}s` });
  if (b.play_pause) list.push({ id: "play_pause", label: "Pause" });
  if (b.seek) list.push({ id: "seek", label: `+${step}s` });
  for (const sp of b.speeds) list.push({ id: "speed", label: fmtSpeed(sp), on: sp === 1 });
  if (b.mute) list.push({ id: "mute", label: "Muted" });
  if (b.expand) list.push({ id: "expand", label: "⤢" });
  if (b.close) list.push({ id: "close", label: "✕" });
  return list;
}
function buttonRowWidth(s) {
  const list = buttonList(s);
  if (!list.length) return 0;
  const sum = list.reduce((a, x) => a + BTN_W[x.id], 0);
  return Math.round(sum * s.appearance.font_scale + 4 * (list.length - 1)) + 16;
}

// ------------------------------------------------------------------ preview

// Real footage when the server has a recording: it plays in the mock player,
// and the same recording further along stands in for the game behind it. The
// elements persist across renders, so changing a setting never restarts them.
const footage = { player: null, backdrop: null };
const stillMotion = matchMedia("(prefers-reduced-motion: reduce)");
// Magnification as ReplayWindow.qml models it: the picture's offset as a
// fraction of its frame, in [1 - z, 0].
const zoom = { z: 1, x: 0, y: 0 };
const clampPan = (v) => Math.max(1 - zoom.z, Math.min(0, v));

function zoomAt(fx, fy, factor) {
  const next = Math.max(1, Math.min(4, zoom.z * factor));
  const cx = (fx - zoom.x) / zoom.z, cy = (fy - zoom.y) / zoom.z;  // the point under the pointer stays put
  zoom.z = next;
  zoom.x = clampPan(fx - cx * next);
  zoom.y = clampPan(fy - cy * next);
}

function positionFootage() {
  Object.assign(footage.player.style, {
    left: `${zoom.x * 100}%`, top: `${zoom.y * 100}%`, width: `${zoom.z * 100}%`, height: `${zoom.z * 100}%`,
  });
}

function footageVideo(className, startAt) {
  const v = h("video", { class: className, loop: true, playsinline: true, preload: "auto", "aria-hidden": "true" });
  v.muted = true;
  v.addEventListener("loadedmetadata", () => {
    v.currentTime = v.duration * startAt;
    if (!stillMotion.matches) v.play().catch(() => {});
  }, { once: true });
  v.src = state.footage;
  return v;
}

function setupFootage() {
  if (!state.footage) return;
  footage.player = footageVideo("pv-footage", 0.1);
  footage.backdrop = footageVideo("pv-backdrop", 0.55);
  const stage = $("#pv-stage");
  stage.prepend(footage.backdrop);
  stage.classList.add("has-footage");
  // Scroll magnifies and dragging pans, as in the player.
  stage.addEventListener("wheel", (e) => {
    const frame = e.target.closest(".pv-video");
    if (!frame) return;
    e.preventDefault();
    const r = frame.getBoundingClientRect();
    const dy = e.deltaY * (e.deltaMode === 1 ? 40 : 1);
    zoomAt((e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height, Math.pow(1.0015, -dy));
    renderPlayer();
  }, { passive: false });
  let pan = null;
  stage.addEventListener("pointerdown", (e) => {
    const frame = e.target.closest(".pv-video");
    if (!frame || zoom.z <= 1) return;
    const r = frame.getBoundingClientRect();
    pan = { id: e.pointerId, x: e.clientX, y: e.clientY, zx: zoom.x, zy: zoom.y, w: r.width, h: r.height };
    stage.setPointerCapture(e.pointerId);
  });
  stage.addEventListener("pointermove", (e) => {
    if (!pan || e.pointerId !== pan.id) return;
    zoom.x = clampPan(pan.zx + (e.clientX - pan.x) / pan.w);
    zoom.y = clampPan(pan.zy + (e.clientY - pan.y) / pan.h);
    positionFootage();
  });
  const endPan = () => { pan = null; };
  stage.addEventListener("pointerup", endPan);
  stage.addEventListener("pointercancel", endPan);
}

function hintText(s, active) {
  const tc = plainChord(s.hotkeys.toggle_controls);
  if (s.input.mode === "mouse") {
    if (active) return tc ? `CONTROLS · click the game or ${tc} to return` : "CONTROLS · click the game to return";
    return "PASSIVE · click for controls";
  }
  return active ? `CONTROLS · ${tc} return` : `PASSIVE · ${tc} controls`;
}

function legend(s) {
  const k = s.player_keys, step = s.input.seek_step_s;
  const pair = (a, b) => [k[a], k[b]].filter(Boolean).map(plainChord).join("/");
  const items = [
    [plainChord(k.play_pause), "play"], [pair("seek_back", "seek_forward"), `±${step}s`],
    [pair("speed_down", "speed_up"), "speed"], [plainChord(k.mute), "mute"],
    [plainChord(k.restart), "start"], [plainChord(k.expand), "expand"], [plainChord(k.close), "close"],
  ].filter(([key]) => key);
  return items.flatMap(([key, what], i) => [i ? " · " : "", h("b", {}, key), ` ${what}`]);
}

function renderPlayer() {
  const s = state.draft, a = s.appearance, c = a.colors, fs = a.font_scale;
  const active = state.pvState === "controls";
  const W = s.layout.width, H = s.layout.height;
  const p = $("#pv-player");
  p.className = "pv-player" + (active ? " active" : "") + (footage.player && zoom.z > 1 ? " zoomed" : "");
  Object.assign(p.style, { width: px(W), height: px(H), opacity: active ? a.active_opacity : a.passive_opacity });
  for (const [k, v] of Object.entries({ "--c-bg": c.background, "--c-panel": c.panel, "--c-header": c.header,
    "--c-header-active": c.header_active, "--c-text": c.text, "--c-muted": c.muted, "--c-accent": c.accent })) {
    p.style.setProperty(k, v);
  }
  const parts = [];
  if (a.show_header) {
    parts.push(h("div", { class: "pv-header", style: { height: px(36 * fs) } },
      h("span", { class: "pv-hint", style: { fontSize: px(11 * fs) } }, hintText(s, active))));
  }
  const layout = a.controls;
  const minimal = layout === "minimal";
  const fit = { fit: "contain", crop: "cover", stretch: "100% 100%" }[s.layout.fill];
  const video = h("div", { class: "pv-video" + (footage.player ? " live" : ""),
                           style: { backgroundSize: fit, backgroundColor: c.background } });
  if (footage.player) {
    footage.player.style.objectFit = { fit: "contain", crop: "cover", stretch: "fill" }[s.layout.fill];
    positionFootage();
    video.append(footage.player);
  }
  // Minimal controls' state badge; here only magnification can differ from plain playback.
  if (minimal && zoom.z > 1.01) {
    video.append(h("div", { class: "pv-badge", style: { fontSize: px(11 * fs) } }, `Zoom ${zoom.z.toFixed(1)}×`));
  }
  // With no title bar, Minimal shows the way to the full session in the corner while in control.
  if (minimal && active) {
    const key = plainChord(s.player_keys.history);
    video.append(h("span", { class: "pv-btn pv-history", style: { width: px(118 * fs), height: px(28 * fs), fontSize: px(12 * fs) } },
      key ? `Full session · ${key}` : "Full session"));
  }
  if (a.show_key_hints && active && s.input.mode !== "mouse") {
    video.append(h("div", { class: "pv-legend", style: { height: px(22 * fs), fontSize: px(11 * fs), bottom: minimal ? "12px" : 0 } }, legend(s)));
  }
  // Minimal: a 3 px timeline on the video's bottom edge (a 12 px grab strip in the player).
  if (minimal) video.append(h("div", { class: "pv-line" }, h("div", { class: "fill", style: { width: "40%" } })));
  parts.push(video);

  const slider = (style = {}) => h("div", { class: "pv-slider", style }, h("div", { class: "track" },
    h("div", { class: "fill", style: { width: "40%" } }), h("div", { class: "knob", style: { left: "40%" } })));
  const time = () => h("span", { class: "pv-time", style: { fontSize: px(11 * fs) } }, "12 / 30s");
  const buttons = () => buttonList(s).map((b) => h("span", {
    class: "pv-btn" + (b.on ? " on" : ""),
    style: { width: px(BTN_W[b.id] * fs), height: px(28 * fs), fontSize: px(12 * fs) },
  }, b.label));
  if (layout !== "none" && !minimal) {
    const panel = h("div", { class: "pv-controls" });
    if (layout === "full") {
      panel.append(h("div", { class: "pv-row" }, slider()));
      panel.append(h("div", { class: "pv-row", style: { marginTop: "6px", height: px(28 * fs) } }, buttons()));
      if (a.show_status) {
        const tr = plainChord(s.hotkeys.toggle_replay), q = plainChord(s.hotkeys.quit);
        panel.append(h("div", { class: "pv-info", style: { marginTop: "6px", height: px(14 * fs), fontSize: px(11 * fs), lineHeight: px(14 * fs) } },
          h("span", { class: "pv-time" }, "12 / 30s"),
          h("span", { class: "grow" }, active ? "Replay controls active" : "Game input active"),
          h("span", {}, [tr && `${tr} hide`, q && `${q} quit`].filter(Boolean).join(" · "))));
      }
    } else if (layout === "compact") {
      panel.append(h("div", { class: "pv-row", style: { height: px(28 * fs), gap: "6px" } },
        h("div", { class: "pv-row" }, buttons()), slider({ minWidth: "80px" }), time()));
    } else {
      panel.append(h("div", { class: "pv-row", style: { height: "20px", gap: "6px" } }, slider(), time()));
    }
    parts.push(panel);
  }
  if (a.focus_outline && active) parts.push(h("div", { class: "pv-outline" }));
  if (active && s.input.mode !== "keyboard") parts.push(h("div", { class: "pv-grip" }));
  p.replaceChildren(...parts);
  // Re-parenting can pause a playing video; carry on.
  const v = footage.player;
  if (v && v.paused && v.readyState > 0 && !stillMotion.matches) v.play().catch(() => {});

  const stage = $("#pv-stage");
  const scale = Math.min(1, (stage.clientWidth - 28) / W, (stage.clientHeight - 28) / H);
  const wrap = $("#pv-scale");
  Object.assign(wrap.style, { width: px(W), height: px(H), transform: `translate(-50%, -50%) scale(${scale})` });
  const how = { keyboard: "keyboard only", mouse: "mouse only", hybrid: "keyboard + mouse" }[s.input.mode];
  $("#pv-caption").textContent = `${W} × ${H} px, shown at ${Math.round(scale * 100)}% · ${active ? "in control" : "passive"} · ${how}`
    + (footage.player ? " · your latest replay, scroll it to magnify" : "");
  renderScreen();
}

// Same placement rule as PlayerSettings::placement in the player.
function placement(s, aw, ah) {
  const W = Math.min(s.layout.width, aw), H = Math.min(s.layout.height, ah);
  const [v, hz] = s.layout.anchor === "center" ? ["middle", "center"] : s.layout.anchor.split("_");
  const mx = s.layout.margin_x, my = s.layout.margin_y;
  let x = hz === "left" ? mx : hz === "right" ? aw - mx - W : (aw - W) / 2;
  let y = v === "top" ? my : v === "bottom" ? ah - my - H : (ah - H) / 2;
  x = Math.max(0, Math.min(x, aw - W));
  y = Math.max(0, Math.min(y, ah - H));
  return { x, y, W, H };
}

function renderScreen() {
  const s = state.draft;
  const aw = screen.availWidth || 1920, ah = screen.availHeight || 1040;
  const box = $("#pv-screen");
  const scale = box.clientWidth / aw;
  box.style.height = px(ah * scale);
  const r = placement(s, aw, ah);
  Object.assign($("#pv-screen-player").style, {
    left: px(r.x * scale), top: px(r.y * scale), width: px(r.W * scale), height: px(r.H * scale),
    background: s.appearance.colors.panel, border: `1px solid ${s.appearance.colors.accent}`,
  });
  const where = { game: "the game's screen", cursor: "the screen with the cursor", primary: "the primary screen" }[s.layout.screen];
  $("#pv-screen-caption").textContent = `Drawn on this display's usable area (${aw} × ${ah}); the player uses ${where}.`;
}

// ------------------------------------------------------------------ issues, status, saving

function renderIssues() {
  for (const el of $$(".field-msg.auto")) el.remove();
  for (const el of $$(".invalid[data-path]")) el.classList.remove("invalid");
  const placed = new Set();
  for (const issue of state.issues) {
    if (issue.path.startsWith("hotkeys.") || issue.path.startsWith("player_keys.")) { placed.add(issue); continue; }
    const target = $(`[data-path="${issue.path}"]`) || $(`[data-issue-path="${issue.path}"]`);
    if (!target) continue;
    if (issue.level === "error" && target.matches("input, select")) target.classList.add("invalid");
    const host = target.closest(".field") || target.parentElement;
    host.append(h("p", { class: `field-msg auto ${issue.level}` }, issue.message));
    placed.add(issue);
  }
  const box = $("#issues");
  const errs = state.issues.filter((i) => i.level === "error"), warns = state.issues.filter((i) => i.level === "warning");
  if (!state.issues.length) { box.replaceChildren(); return; }
  const title = [errs.length && `${errs.length} problem${errs.length > 1 ? "s" : ""} to fix before saving`,
                 warns.length && `${warns.length} warning${warns.length > 1 ? "s" : ""}`].filter(Boolean).join(" · ");
  box.replaceChildren(h("h3", {}, title), h("ul", {}, state.issues.map((i) => {
    const anchor = $(`[data-path="${i.path}"]`) || $(`[data-issue-path="${i.path}"]`);
    const go = anchor ? h("a", { href: "#", onclick: (e) => { e.preventDefault(); anchor.scrollIntoView({ block: "center" });
      (anchor.querySelector("button, input") || anchor).focus?.(); } }, i.message) : i.message;
    return h("li", { class: i.level }, go);
  })));
}

function isDirty() {
  if (!state.saved) return false;
  const strip = (s) => JSON.stringify({ ...s, revision: 0 });
  return strip(state.draft) !== strip(state.saved);
}

function renderChrome() {
  const dirty = isDirty();
  const hasErrors = state.issues.some((i) => i.level === "error");
  $("#dirty").hidden = !dirty;
  $("#dirty").textContent = hasErrors ? "Fix problems to save" : "Unsaved changes";
  $("#save").disabled = !dirty || hasErrors;
  $("#revert").disabled = !dirty;
  renderStatus();
}

function renderStatus() {
  const st = state.status, el = $("#player-status"), label = $(".label", el);
  if (st.runtime) {
    const rt = st.runtime;
    const errors = [rt.model, rt.voice_model, rt.voice].filter((x) => x && x.state === "error");
    const hotkeyErrors = Object.values(st.hotkeys || {}).filter((x) => x.state === "failed" || x.state === "invalid");
    const problems = (st.issues || []).length + hotkeyErrors.length;
    const behind = st.running && state.saved && (st.revision ?? -1) < state.saved.revision;
    el.className = `player-status ${rt.state === "error" ? "bad" : rt.state === "recording" ? "ok" : "pending"}`;
    label.textContent = rt.message + (errors.length ? " · AI/voice needs attention" : "");
    if (problems) {
      el.className = "player-status bad";
      label.textContent += ` · ${problems} setting${problems > 1 ? "s" : ""} not applied`;
    } else if (behind) {
      el.className = "player-status pending";
      label.textContent += performance.now() - state.savedAt > 5000 ? " · hasn't picked up the last save" : " · applying settings…";
    }
    el.title = [rt.message, ...errors.map((x) => x.message), ...hotkeyErrors.map((x) => x.detail || x.state), ...(st.issues || []).map((x) => x.message || x)].filter(Boolean).join("\n");
    return;
  }
  if (!st.running) {
    el.className = "player-status";
    label.textContent = "Player not running · settings apply when it starts";
    return;
  }
  const behind = state.saved && (st.revision ?? -1) < state.saved.revision;
  const problems = (st.issues || []).length + Object.values(st.hotkeys || {}).filter((x) => x.state === "failed" || x.state === "invalid").length;
  if (behind) {
    const stale = performance.now() - state.savedAt > 5000;
    el.className = "player-status pending";
    label.textContent = stale ? "Player running · hasn't picked up the last save" : "Player running · applying…";
  } else if (problems) {
    el.className = "player-status bad";
    label.textContent = `Player running · ${problems} setting${problems > 1 ? "s" : ""} not applied`;
  } else {
    el.className = "player-status ok";
    label.textContent = "Player running · settings applied";
  }
  el.title = (st.issues || []).map((i) => i.message || i).join("\n");
}

let validateTimer = 0, validateSeq = 0;
function changed() {
  render();
  clearTimeout(validateTimer);
  validateTimer = setTimeout(validate, 120);
}
async function validate() {
  const seq = ++validateSeq;
  try {
    const { data } = await api("POST", "/api/validate", state.draft);
    if (seq !== validateSeq) return;
    state.issues = data.issues || [];
  } catch {
    state.issues = [{ level: "error", path: "", message: "The settings server is not responding. Is it still running?" }];
  }
  render();
}

function render() {
  for (const fn of syncers) fn();
  renderPlayer();
  renderIssues();
  renderChrome();
}

async function save() {
  if ($("#save").disabled) return;
  clearTimeout(validateTimer);
  const { ok, status, data } = await api("PUT", "/api/settings", state.draft).catch(() => ({ ok: false, data: {} }));
  if (ok) {
    accept(data);
    state.savedAt = performance.now();
    toast(state.status.running ? "Saved · the player is applying it now" : "Saved · applies when the player starts");
  } else if (status === 422) {
    state.issues = data.issues || [];
    render();
    toast("Not saved: fix the problems listed first.", true);
  } else {
    toast(data.error || "Not saved: the settings server did not respond.", true);
  }
}

function accept(snapshot) {
  $("#app-version").textContent = snapshot.version ? `v${snapshot.version}` : "";
  state.saved = snapshot.settings;
  state.draft = clone(snapshot.settings);
  state.defaults = snapshot.defaults;
  state.themes = snapshot.themes;
  state.issues = snapshot.issues || [];
  state.status = snapshot.status || { running: false };
  state.path = snapshot.path;
  state.footage ??= snapshot.footage || null;  // keep playing the clip the page started with
  state.providers = snapshot.providers || {};
  $("#file-path").textContent = snapshot.path;
  const loadErrors = (snapshot.issues || []).filter((i) => i.level === "error");
  const banner = $("#load-error");
  banner.hidden = !loadErrors.length;
  banner.textContent = loadErrors.length
    ? `The settings file has problems, so defaults are shown where it was unreadable: ${loadErrors.map((i) => i.message).join(" ")} Saving rewrites it.`
    : "";
}

async function pollStatus() {
  try {
    const { data } = await api("GET", "/api/status");
    state.status = data;
    for (const fn of syncers) fn();
    renderStatus();
  } catch { /* server stopped; the next save reports it */ }
}

// ------------------------------------------------------------------ boot

function wireChrome() {
  $("#save").addEventListener("click", save);
  $("#revert").addEventListener("click", () => { state.draft = clone(state.saved); changed(); });
  $("#reset-all").addEventListener("click", () => {
    state.draft = { ...clone(state.defaults), revision: state.saved.revision };
    changed();
    toast("Everything reset to defaults · Save to keep it");
  });
  for (const b of $$("[data-reset]")) {
    b.addEventListener("click", () => {
      for (const key of b.dataset.reset.split(",")) state.draft[key] = clone(state.defaults[key]);
      changed();
    });
  }
  for (const r of $$("#pv-state input")) {
    r.addEventListener("change", () => { if (r.checked) { state.pvState = r.value; renderPlayer(); } });
  }
  document.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s") { e.preventDefault(); save(); }
  });
  window.addEventListener("beforeunload", (e) => { if (isDirty()) { e.preventDefault(); e.returnValue = ""; } });
  window.addEventListener("resize", () => renderPlayer());

  const links = $$(".sidenav a");
  const observer = new IntersectionObserver((entries) => {
    for (const en of entries) {
      if (!en.isIntersecting) continue;
      links.forEach((a) => a.classList.toggle("current", a.getAttribute("href") === `#${en.target.id}`));
    }
  }, { rootMargin: "-30% 0px -60% 0px" });
  $$("main section").forEach((s) => observer.observe(s));
}

async function boot() {
  let res;
  try {
    res = await api("GET", "/api/settings");
  } catch {
    $("#load-error").hidden = false;
    $("#load-error").textContent = "Could not reach the settings server. Start it with: python -m smartreplay.cli settings";
    return;
  }
  accept(res.data);
  setupFootage();
  bindInputs();
  buildFocus();
  buildHotkeys();
  buildSize();
  buildLook();
  buildButtons();
  buildModel();
  wireChrome();
  render();
  validate();
  setInterval(pollStatus, 2000);
}

boot();
