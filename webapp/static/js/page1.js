/* page1.js - Upload + clap-sync + court calibration controller. */
(function () {
  "use strict";
  const q = (el, s) => el.querySelector(s);
  const lockBadge = (px) => `<span class="badge badge-locked plain">LOCKED · ${px.toFixed(2)} px</span>`;

  document.querySelectorAll(".cam-card").forEach(initCard);
  const autoBtn = document.getElementById("autoSync");
  if (autoBtn) autoBtn.addEventListener("click", autoSync);
  document.addEventListener("session-state", (e) => restore(e.detail));

  function initCard(card) {
    const role = q(card, "[data-dropzone]").dataset.role;
    card.dataset.role = role;
    card._frame = 0;
    wireDropzone(card, role);
    wireSync(card, role);
    wireFrameControls(card, role);
    if (card.dataset.mode === "court") wireCourtCalibration(card, role);
    else if (card.dataset.mode === "kitchen") wireKitchenCalibration(card, role);
  }

  // ---- uploads ----
  function wireDropzone(card, role) {
    const dz = q(card, "[data-dropzone]");
    const input = q(dz, "[data-fileinput]");
    // reset value so re-picking the SAME file still fires 'change' (a Replace)
    const pick = () => { input.value = ""; input.click(); };
    dz.addEventListener("click", pick);
    input.addEventListener("change", () => input.files[0] && doUpload(card, role, input.files[0]));
    ["dragover", "dragenter"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("dragover"); }));
    ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, () => dz.classList.remove("dragover")));
    dz.addEventListener("drop", (e) => { e.preventDefault(); e.dataTransfer.files[0] && doUpload(card, role, e.dataTransfer.files[0]); });
    // "Replace video" (shown once a clip is loaded) re-opens the same picker
    const replace = q(card, "[data-replace]");
    if (replace) replace.addEventListener("click", pick);
  }

  async function doUpload(card, role, file) {
    const dz = q(card, "[data-dropzone]");
    const wasLoaded = !q(card, "[data-stagewrap]").classList.contains("hidden");
    dz.querySelector(".dz-title").textContent = "Uploading… 0%";
    try {
      const res = await API.upload(role, file, (p) => { dz.querySelector(".dz-title").textContent = `Uploading… ${Math.round(p * 100)}%`; });
      // a Replace invalidates the old calibration/editor - clear it, then force
      // the <video> to fetch the new clip (same URL, so we cache-bust + reload).
      resetCalibration(card, role);
      showMedia(card, role, res.meta, { reload: true });
      window.toast(`${role.toUpperCase()} ${wasLoaded ? "replaced" : "uploaded"}`, "success");
      Workflow.refresh();
    } catch (err) {
      window.toast(err.message, "error");
      dz.querySelector(".dz-title").textContent = "Drop video or click to browse";
    }
  }

  // Clear the calibration editor + badges so a freshly (re)uploaded clip starts
  // clean - no stale keypoints, ROI, PnP, or LOCKED badge from the old video.
  function resetCalibration(card, role) {
    card._editor = null;
    card._frame = 0;
    const hide = (sel, cls) => { const e = q(card, sel); if (e) (cls ? e.classList.add(cls) : (e.hidden = true)); };
    hide("[data-overlay]");
    hide("[data-kptools]", "hidden");
    hide("[data-legend]", "hidden");
    hide("[data-confirm]", "hidden");
    hide("[data-pnp]", "hidden");
    const lock = q(card, "[data-lock]"); if (lock) lock.innerHTML = "";
    q(card, "[data-status]").innerHTML = '<span class="badge badge-neutral plain">not calibrated</span>';
    const det = q(card, "[data-detect]"); if (det && card._detectHTML) det.innerHTML = card._detectHTML;
    const draw = q(card, "[data-kitchen-draw]"); if (draw && card._drawHTML) draw.innerHTML = card._drawHTML;
  }

  // Page 1 calibrates on a still frame (camera is static). We show the real
  // <video> (the browser decodes any codec it supports) and step it frame by
  // frame; stills for the editor are captured client-side from that <video>,
  // so nothing depends on server-side OpenCV decoding.
  function showMedia(card, role, meta, opts) {
    opts = opts || {};
    q(card, "[data-dropzone]").classList.add("hidden");
    q(card, "[data-stagewrap]").classList.remove("hidden");
    q(card, "[data-frame]").hidden = true;            // legacy still, unused here
    if (meta) { card._fps = meta.fps || 60; card._nframes = meta.frame_count || 0; }
    const v = q(card, "[data-video]");
    v.hidden = false;
    v.preload = "auto";
    if (opts.reload) {
      // Replace: the media URL is the same, so bust the browser cache + reload
      // to pull the new clip instead of re-showing the old decoded frames.
      card._frame = 0;
      v.src = API.mediaUrl(role) + "?t=" + Date.now();
      v.load();
    } else if (!v.src) {
      v.src = API.mediaUrl(role);
    }
    const goto0 = () => setFrame(card, role, card._frame || 0);
    if (v.readyState >= 1 && !opts.reload) goto0();
    else v.addEventListener("loadedmetadata", goto0, { once: true });
  }

  // ---- frame stepper (find a frame where all 12 court points are visible) ----
  function wireFrameControls(card, role) {
    card.querySelectorAll("[data-fstep]").forEach((b) =>
      b.addEventListener("click", () =>
        setFrame(card, role, (card._frame || 0) + parseInt(b.dataset.fstep, 10))));
  }

  function setFrame(card, role, idx) {
    idx = Math.max(0, Math.round(idx));
    if (card._nframes && idx > card._nframes - 1) idx = card._nframes - 1;
    card._frame = idx;
    const v = q(card, "[data-video]");
    const fps = card._fps || 30;
    const editing = card._editor && !q(card, "[data-overlay]").hidden;
    if (editing) v.addEventListener("seeked", () => card._editor.setBackground(captureFrame(v)), { once: true });
    const dur = Number.isFinite(v.duration) ? v.duration : 1e9;
    v.currentTime = Math.max(0, Math.min(idx / fps, dur - 0.05));
    q(card, "[data-frameinfo]").textContent = "frame " + idx + (card._fps ? " · " + (idx / fps).toFixed(2) + " s" : "");
  }

  // Grab the <video>'s current frame as a data URL (client-side, no server decode).
  function captureFrame(v) {
    const c = document.createElement("canvas");
    c.width = v.videoWidth || 1920; c.height = v.videoHeight || 1080;
    c.getContext("2d").drawImage(v, 0, 0, c.width, c.height);
    try { return c.toDataURL("image/jpeg", 0.9); } catch (e) { return ""; }
  }

  // Resolve once the current frame is decodable, then return its data URL.
  function ensureFrame(v) {
    return new Promise((res) => {
      if (v.readyState >= 2 && v.videoWidth) return res(captureFrame(v));
      v.addEventListener("loadeddata", () => res(captureFrame(v)), { once: true });
    });
  }

  // ---- sync controls ----
  function wireSync(card, role) {
    const input = q(card, "[data-off-input]"), slider = q(card, "[data-off-slider]");
    const disp = q(card, "[data-off-display]"), method = q(card, "[data-sync-method]");
    const set = (val, persist) => {
      val = Math.round(val * 10) / 10;
      input.value = val; slider.value = val; disp.textContent = val.toFixed(1);
      if (persist) persistOffset(role, val, method);
    };
    q(card, "[data-off-inc]").addEventListener("click", () => set(parseFloat(input.value) + 1, true));
    q(card, "[data-off-dec]").addEventListener("click", () => set(parseFloat(input.value) - 1, true));
    q(card, "[data-off-reset]").addEventListener("click", () => set(0, true));
    input.addEventListener("input", () => { disp.textContent = parseFloat(input.value || 0).toFixed(1); slider.value = input.value; });
    input.addEventListener("change", () => set(parseFloat(input.value || 0), true));
    slider.addEventListener("input", () => { disp.textContent = parseFloat(slider.value).toFixed(1); input.value = slider.value; });
    slider.addEventListener("change", () => set(parseFloat(slider.value), true));
    card._setOffset = set;
    card._method = method;
  }

  async function persistOffset(role, val, methodEl) {
    if (role === "cam1") return; // reference - offset stays 0
    try {
      await API.postJSON("/api/sync/manual", { cam: role, offset_frames: val });
      if (methodEl) { methodEl.textContent = "manual"; methodEl.className = "badge badge-neutral plain"; }
      Workflow.refresh();
    } catch (err) { window.toast(err.message, "error"); }
  }

  async function autoSync() {
    autoBtn.disabled = true;
    autoBtn.innerHTML = '<span class="spinner"></span> Syncing…';
    try {
      const res = await API.postJSON("/api/sync", {});
      applySync(res.sync);
      window.toast(res.sync.message, res.sync.status === "manual_required" ? "error" : "success");
      Workflow.refresh();
    } catch (err) { window.toast(err.message, "error"); }
    autoBtn.disabled = false;
    autoBtn.innerHTML = "Auto-sync all";
  }

  function applySync(sync) {
    if (!sync || !sync.offsets) return;
    document.querySelectorAll(".cam-card").forEach((card) => {
      const o = sync.offsets[card.dataset.role];
      if (!o) return;
      if (card._setOffset) card._setOffset(o.offset_frames, false);
      if (card._method) { card._method.textContent = o.method; }
    });
  }

  // ---- calibration (shared editor chrome + tools) ----
  function wireEditorTools(card) {
    q(card, "[data-confirm]").addEventListener("click", () => card._confirm && card._confirm());
    q(card, "[data-undo]").addEventListener("click", () => card._editor && card._editor.undo());
    q(card, "[data-redo]").addEventListener("click", () => card._editor && card._editor.redo());
    q(card, "[data-kpreset]").addEventListener("click", () => card._editor && card._editor.reset());
    q(card, "[data-zoomin]").addEventListener("click", () => card._editor && card._editor.zoomBy(1.25));
    q(card, "[data-zoomout]").addEventListener("click", () => card._editor && card._editor.zoomBy(1 / 1.25));
  }

  function showEditChrome(card) {
    // keep the <video> in the DOM (behind the opaque editor canvas) so frames
    // can still be captured from it while stepping during edit.
    q(card, "[data-frame]").hidden = true;
    q(card, "[data-overlay]").hidden = false;
    q(card, "[data-kptools]").classList.remove("hidden");
    q(card, "[data-legend]").classList.remove("hidden");
    q(card, "[data-confirm]").classList.remove("hidden");
  }

  function makeEditor(card, drawFn) {
    const zv = q(card, "[data-zoomval]");
    return new window.KeypointEditor(q(card, "[data-overlay]"), {
      draw: drawFn,
      labels: !drawFn,   // court keypoints get index labels; box does not
      onChange: (_k, zoom) => { if (zv) zv.textContent = Math.round(zoom * 100) + "%"; },
    });
  }

  // ---- CAM1/CAM2: model court detection + 12-keypoint editor ----
  function wireCourtCalibration(card, role) {
    const det = q(card, "[data-detect]");
    card._detectHTML = det.innerHTML;   // snapshot label so Replace can restore it
    det.addEventListener("click", () => detect(card, role));
    card._confirm = () => confirmCourt(card, role);
    wireEditorTools(card);
  }

  async function detect(card, role) {
    const v = q(card, "[data-video]");
    const btn = q(card, "[data-detect]");
    btn.disabled = true; btn.innerHTML = '<span class="spinner"></span> Detecting...';
    try {
      const bg = await ensureFrame(v);   // client-side still of the current frame
      const det = await API.postJSON("/api/calibration/detect",
        { cam: role, frame: card._frame || 0, width: v.videoWidth, height: v.videoHeight, image: bg });
      showEditChrome(card);
      if (!card._editor) card._editor = makeEditor(card, null);
      card._editor.load(bg, det.keypoints);
      const note = q(card, "[data-detect-note]");
      if (note) note.textContent = det.source === "model"
        ? "model detected the court - fine-tune any point if needed"
        : (det.note || "mock guess - drag the 12 points onto the lines");
    } catch (err) { window.toast(err.message, "error"); }
    btn.disabled = false; btn.innerHTML = '<svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9"><circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3"/></svg> Re-detect';
  }

  async function confirmCourt(card, role) {
    if (!card._editor) return;
    const btn = q(card, "[data-confirm]");
    btn.disabled = true; btn.innerHTML = '<span class="spinner"></span> Locking...';
    try {
      const res = await API.postJSON("/api/calibration/confirm", { cam: role, keypoints: card._editor.getKeypoints() });
      const c = res.calibration;
      const badge = c.locked ? lockBadge(c.mean_reproj_px)
        : `<span class="badge badge-close plain">high error · ${c.mean_reproj_px.toFixed(1)} px</span>`;
      q(card, "[data-lock]").innerHTML = badge;
      q(card, "[data-status]").innerHTML = c.locked ? badge : '<span class="badge badge-close plain">check keypoints</span>';
      renderPnp(card, c.pnp);
      window.toast(`${role.toUpperCase()} calibrated · ${c.mean_reproj_px.toFixed(2)} px`, c.locked ? "success" : "");
      Workflow.refresh();
    } catch (err) { window.toast(err.message, "error"); }
    btn.disabled = false; btn.innerHTML = '<svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9"><path d="M20 6L9 17l-5-5"/></svg> Confirm calibration';
  }

  // measured (from config) vs PnP-recovered camera position (Recipe 3 check)
  function renderPnp(card, pnp) {
    const el = q(card, "[data-pnp]");
    if (!el || !pnp) return;
    el.classList.remove("hidden", "ok", "off");
    if (!pnp.available) { el.textContent = "Position check unavailable: " + pnp.reason; return; }
    const rec = "(" + pnp.recovered_ft.join(", ") + ")";
    const warn = pnp.warning ? `<div class="pnp-warn">${pnp.warning}</div>` : "";
    if (pnp.measured_ft) {
      const meas = "(" + pnp.measured_ft.join(", ") + ")";
      el.classList.add(pnp.ok ? "ok" : "off");
      el.innerHTML = `<span class="pnp-badge">${pnp.ok ? "✓ position check" : "⚠ position mismatch"}</span>`
        + ` <span class="mono">measured ${meas} vs PnP ${rec} ft · error ${pnp.error_ft} ft</span>` + warn;
    } else {
      el.innerHTML = `<span class="pnp-badge">PnP position</span> <span class="mono">${rec} ft (no measured value in config to compare)</span>` + warn;
    }
  }

  // ---- CAM3: manual 4-point kitchen box (no model) ----
  const kitchenBadge = '<span class="badge badge-locked plain">KITCHEN SET</span>';

  function wireKitchenCalibration(card, role) {
    const draw = q(card, "[data-kitchen-draw]");
    card._drawHTML = draw.innerHTML;    // snapshot label so Replace can restore it
    draw.addEventListener("click", () => drawKitchen(card, role));
    card._confirm = () => confirmKitchen(card, role);
    wireEditorTools(card);
  }

  function defaultKitchenBox(w, h) {
    // a sensible starting quad over the middle of the frame (clockwise)
    return [[0.24 * w, 0.40 * h, 2], [0.76 * w, 0.40 * h, 2],
            [0.80 * w, 0.82 * h, 2], [0.20 * w, 0.82 * h, 2]];
  }

  async function drawKitchen(card, role) {
    const v = q(card, "[data-video]");
    const bg = await ensureFrame(v);
    showEditChrome(card);
    if (!card._editor) card._editor = makeEditor(card, window.BoxOverlay.draw);
    card._editor.load(bg, defaultKitchenBox);
    q(card, "[data-kitchen-draw]").innerHTML = '<svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9"><circle cx="12" cy="12" r="8"/></svg> Redraw box';
  }

  async function confirmKitchen(card, role) {
    if (!card._editor) return;
    const btn = q(card, "[data-confirm]");
    btn.disabled = true; btn.innerHTML = '<span class="spinner"></span> Saving...';
    try {
      const points = card._editor.getKeypoints().map((p) => [p[0], p[1]]);
      const res = await API.postJSON("/api/calibration/kitchen", { points });
      q(card, "[data-lock]").innerHTML = kitchenBadge;
      q(card, "[data-status]").innerHTML = kitchenBadge;
      window.toast("CAM 3 kitchen box saved", "success");
      Workflow.refresh();
    } catch (err) { window.toast(err.message, "error"); }
    btn.disabled = false; btn.innerHTML = '<svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9"><path d="M20 6L9 17l-5-5"/></svg> Confirm calibration';
  }

  // ---- restore from persisted state ----
  let restored = false;
  function restore(state) {
    if (restored || !state) return; restored = true;
    document.querySelectorAll(".cam-card").forEach((card) => {
      const role = card.dataset.role;
      if (state.uploads && state.uploads[role]) {
        showMedia(card, role, (state.meta || {})[role] || null);
      }
      const cal = state.calibration && state.calibration[role];
      if (cal && cal.type === "kitchen_box") {
        q(card, "[data-status]").innerHTML = kitchenBadge;
        const lock = q(card, "[data-lock]"); if (lock) lock.innerHTML = kitchenBadge;
      } else if (cal && cal.locked) {
        q(card, "[data-status]").innerHTML = lockBadge(cal.mean_reproj_px);
        const lock = q(card, "[data-lock]"); if (lock) lock.innerHTML = lockBadge(cal.mean_reproj_px);
        if (cal.pnp) renderPnp(card, cal.pnp);
      }
    });
    if (state.sync) applySync(state.sync);
  }
})();
