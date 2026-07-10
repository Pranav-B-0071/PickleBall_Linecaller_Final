/* page3.js - Analysis dashboard: synced playback + top view + verdicts.
   Uses the real video group when clips are present; otherwise a synthetic clock
   drives the top-view animation so a mock run is still fully demonstrable. */
(function () {
  "use strict";
  const PLAY = '<svg class="ic" viewBox="0 0 24 24"><path d="M7 5l12 7-12 7z" fill="currentColor"/></svg>';
  const PAUSE = '<svg class="ic" viewBox="0 0 24 24"><rect x="7" y="5" width="3.5" height="14" rx="1" fill="currentColor"/><rect x="13.5" y="5" width="3.5" height="14" rx="1" fill="currentColor"/></svg>';
  const VERDICT_WINDOW = 2.5;   // s a bounce verdict stays on the banner

  const pb = document.getElementById("playback");
  const seek = pb.querySelector("[data-seek]");
  const curEl = pb.querySelector("[data-cur]"), durEl = pb.querySelector("[data-dur]");
  const ppBtn = pb.querySelector("[data-playpause]");
  const verdictEl = document.getElementById("verdict");
  const topview = new window.TopView(document.getElementById("topview"), { ttl: 10 });

  let group = null, synthetic = null, clock = null, analysis = null, fps = 60, scrubbing = false;

  document.addEventListener("session-state", (e) => setup(e.detail));
  document.getElementById("calcBtn").addEventListener("click", calculate);
  document.getElementById("analyzeBtn").addEventListener("click", run);
  wirePlayback();

  function onTick(t, frame) {
    const dur = clock ? clock.duration() : 0;
    if (!scrubbing) seek.value = dur ? Math.round((t / dur) * 1000) : 0;
    curEl.textContent = t.toFixed(2);
    durEl.textContent = dur.toFixed(2);
    topview.render(t);
    updateVerdict(t);
    ppBtn.innerHTML = clock && clock.paused ? PLAY : PAUSE;
  }

  // ---- setup players from session ----
  let didSetup = false;
  function setup(state) {
    if (didSetup || !state) return; didSetup = true;
    fps = (state.analysis && state.analysis.fps) || 60;
    group = new window.SyncedPlayerGroup({ fps, onTick });
    const offsets = (state.sync && state.sync.offsets) || {};
    ["cam1", "cam2", "cam3"].forEach((role) => {
      const slot = document.querySelector(`.media[data-cam="${role}"]`);
      if (!slot) return;
      if (state.uploads && state.uploads[role]) {
        const v = slot.querySelector("[data-video]");
        v.src = API.mediaUrl(role); v.hidden = false;
        slot.querySelector("[data-empty]").classList.add("hidden");
        v.addEventListener("loadedmetadata", () => { if (clock === group) onTick(group.time(), 0); }, { once: true });
        group.add(v, role, (offsets[role] || {}).offset_frames || 0);
      }
    });
    clock = group.present ? group : ensureSynthetic();
    // restore a prior mock run's summary if present
    if (state.analysis && state.analysis.summary) setSource(state.analysis.source);
    onTick(0, 0);
  }

  function ensureSynthetic() {
    if (!synthetic) synthetic = new SyntheticClock({ fps, onTick });
    return synthetic;
  }

  // ---- calculate: run GridTrackNet per clip (progress shows in the terminal) ----
  async function calculate() {
    const btn = document.getElementById("calcBtn");
    const label = btn.dataset.label || btn.innerHTML;
    btn.disabled = true; btn.innerHTML = '<span class="spinner"></span> Calculating…';
    try {
      const res = await API.postJSON("/api/analysis/calculate", {});
      window.toast(`Tracked ${res.done.length} video(s) · watch the terminal for progress`, "success");
    } catch (err) { window.toast(err.message, "error"); }
    btn.disabled = false; btn.innerHTML = label;
  }

  // ---- analyze: bounces + IN/OUT from the tracked CSVs, play the tracked clips ----
  async function run() {
    const btn = document.getElementById("analyzeBtn");
    btn.disabled = true; btn.innerHTML = '<span class="spinner"></span> Analyzing…';
    try {
      const res = await API.postJSON("/api/analysis/run", {});
      analysis = res.analysis;
      fps = analysis.fps || 60;
      swapToTracked();   // show the annotated (yellow-trail) clips
      topview.setData({ trajectory: analysis.trajectory, bounces: analysis.bounces,
                        roi_ft: analysis.roi_ft, kitchen_zone_ft: analysis.kitchen_zone_ft,
                        cam3_calibrated: analysis.cam3_calibrated });
      if (!group || !group.present) { clock = ensureSynthetic(); synthetic.setDuration(analysis.duration_s); synthetic.fps = fps; }
      renderSummary(analysis.summary);
      renderFeed(analysis.bounces);
      setSource(analysis.source);
      clock.seek(0);
      window.toast(`Analysis complete · ${analysis.summary.total_bounces} bounces`, "success");
      if (analysis.note) window.toast(analysis.note, "");
      Workflow.refresh();
    } catch (err) { window.toast(err.message, "error"); }
    btn.disabled = false; btn.innerHTML = btn.dataset.label || 'Analyze';
  }

  // Point each already-loaded video slot at its tracked (trail) clip. Same
  // <video> elements, so the SyncedPlayerGroup keeps its offsets/master clock.
  function swapToTracked() {
    ["cam1", "cam2", "cam3"].forEach((role) => {
      const slot = document.querySelector(`.media[data-cam="${role}"]`);
      if (!slot) return;
      const v = slot.querySelector("[data-video]");
      if (v && v.src) { v.src = API.trackedUrl(role); v.hidden = false; }
    });
  }

  function setSource(src) {
    const el = document.getElementById("analysisSource");
    el.textContent = src === "mock" ? "mock data"
      : (src === "gridtracknet" ? "GridTrackNet · live" : src);
    el.className = "badge " + (src === "mock" ? "badge-info" : "badge-in") + " plain";
  }

  // ---- verdict banner + feed ----
  function updateVerdict(t) {
    if (!analysis) return;
    const calls = analysis.bounces.filter((b) => b.in_roi && (b.verdict === "IN" || b.verdict === "OUT"));
    let cur = null, idx = -1;
    calls.forEach((b, i) => { if (b.t <= t + 1e-6 && t - b.t <= VERDICT_WINDOW) { cur = b; idx = i; } });
    if (cur) {
      verdictEl.className = "verdict-banner " + cur.verdict.toLowerCase();
      verdictEl.innerHTML = `${cur.verdict} <span class="margin">${Math.abs(cur.margin_ft).toFixed(2)} ft from ${prettyLine(cur.nearest_line)}</span>`;
    } else {
      verdictEl.className = "verdict-banner idle";
      verdictEl.textContent = analysis ? "Tracking…" : "Run analysis to see live line calls";
    }
    document.querySelectorAll(".verdict-row").forEach((r, i) => r.classList.toggle("current", i === idx));
  }

  function renderFeed(bounces) {
    const feed = document.getElementById("verdictFeed");
    const calls = bounces.filter((b) => b.in_roi && (b.verdict === "IN" || b.verdict === "OUT"));
    if (!calls.length) { feed.innerHTML = '<div class="muted">No in-ROI bounces detected.</div>'; return; }
    feed.innerHTML = calls.map((b) => `<div class="verdict-row" data-t="${b.t}">
      <span class="badge ${b.verdict === "IN" ? "badge-in" : "badge-out"}">${b.verdict}</span>
      <span class="t">${b.t.toFixed(2)} s · ${prettyLine(b.nearest_line)}</span>
      <span class="num">${Math.abs(b.margin_ft).toFixed(2)} ft</span></div>`).join("");
    feed.querySelectorAll(".verdict-row").forEach((r) => r.addEventListener("click", () => clock && clock.seek(parseFloat(r.dataset.t))));
  }

  function renderSummary(s) {
    const kitchen = s.kitchen_bounces == null
      ? '<div class="stat" title="Calibrate CAM 3 (kitchen box) on the Calibration page"><div class="k">Kitchen · CAM 3</div><div class="v num">-</div></div>'
      : `<div class="stat"><div class="k">Kitchen · CAM 3</div><div class="v num">${s.kitchen_bounces}</div></div>`;
    document.getElementById("summaryCards").innerHTML = `
      <div class="stat"><div class="k">Bounces</div><div class="v num">${s.total_bounces}</div></div>
      <div class="stat"><div class="k">In ROI</div><div class="v num">${s.in_roi_bounces}</div></div>
      <div class="stat accent"><div class="k">IN</div><div class="v num">${s.in}</div></div>
      <div class="stat warn"><div class="k">OUT</div><div class="v num">${s.out}</div></div>
      ${kitchen}
      <div class="stat"><div class="k">Avg conf</div><div class="v num">${s.avg_confidence}</div></div>`;
  }

  function prettyLine(name) { return (name || "line").replace(/_/g, " "); }

  // ---- playback controls ----
  function wirePlayback() {
    ppBtn.addEventListener("click", () => clock && clock.toggle());
    pb.querySelector("[data-step-back]").addEventListener("click", () => clock && clock.stepFrames(-1));
    pb.querySelector("[data-step-fwd]").addEventListener("click", () => clock && clock.stepFrames(1));
    seek.addEventListener("input", () => { scrubbing = true; if (clock) clock.seek((seek.value / 1000) * clock.duration()); });
    seek.addEventListener("change", () => { scrubbing = false; });
    pb.querySelectorAll("[data-speeds] button").forEach((b) => b.addEventListener("click", () => {
      pb.querySelectorAll("[data-speeds] button").forEach((x) => x.classList.remove("active"));
      b.classList.add("active");
      if (clock) clock.setRate(parseFloat(b.dataset.rate));
    }));
  }

  // ---- synthetic clock (no videos) ----
  class SyntheticClock {
    constructor(o) { this.fps = o.fps || 60; this.onTick = o.onTick; this._t = 0; this._dur = 20; this._playing = false; this.rate = 1; this._raf = null; this._loop = this._loop.bind(this); }
    setDuration(d) { this._dur = d || 20; }
    duration() { return this._dur; }
    time() { return this._t; }
    get paused() { return !this._playing; }
    get present() { return true; }
    seek(t) { this._t = Math.max(0, Math.min(t, this._dur)); this.onTick && this.onTick(this._t, Math.round(this._t * this.fps)); }
    play() { if (this._playing) return; this._playing = true; this._last = performance.now(); this._raf = requestAnimationFrame(this._loop); }
    pause() { this._playing = false; if (this._raf) cancelAnimationFrame(this._raf); this._raf = null; this.onTick && this.onTick(this._t, 0); }
    toggle() { this.paused ? this.play() : this.pause(); }
    stepFrames(n) { this.pause(); this.seek(this._t + n / this.fps); }
    setRate(r) { this.rate = r; }
    _loop() {
      if (!this._playing) return;
      const now = performance.now(); this._t += (now - this._last) / 1000 * this.rate; this._last = now;
      if (this._t >= this._dur) { this._t = this._dur; this._playing = false; }
      this.onTick && this.onTick(this._t, Math.round(this._t * this.fps));
      if (this._playing) this._raf = requestAnimationFrame(this._loop);
    }
  }

  // capture each button's original label for restoration after a spinner
  ["calcBtn", "analyzeBtn"].forEach((id) => {
    const b = document.getElementById(id); b.dataset.label = b.innerHTML;
  });
})();
