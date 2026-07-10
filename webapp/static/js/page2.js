/* page2.js - Label Metrics: uploads, boundary line, compare, dashboard. */
(function () {
  "use strict";
  const q = (el, s) => el.querySelector(s);
  const TRUTH = ["IN", "OUT", "ON_LINE"], PRED = ["IN", "OUT", "CLOSE_IN", "UNKNOWN", "MISSED"];
  const ORIENT = { baseline: "horizontal", sideline: "vertical" };
  const POSMAX = { baseline: 44, sideline: 20 };

  const panels = {};
  document.querySelectorAll(".metrics-panel").forEach(initPanel);
  document.addEventListener("session-state", (e) => restore(e.detail));

  function badge(call) {
    const c = (call || "").toUpperCase();
    if (c === "IN") return '<span class="badge badge-in">IN</span>';
    if (c === "OUT") return '<span class="badge badge-out">OUT</span>';
    if (c === "CLOSE_IN" || c === "ON_LINE") return `<span class="badge badge-close">${c}</span>`;
    return `<span class="badge badge-neutral">${c || "-"}</span>`;
  }

  function initPanel(panel) {
    const kind = panel.dataset.panel;
    const state = { kind, position: parseFloat(q(panel, "[data-bpos]").max) / 2,
                    in_side: q(panel, "[data-bside]").value };
    panels[kind] = { el: panel, state };

    panel.querySelectorAll("[data-dropzone]").forEach((dz) => wireDropzone(panel, dz));

    const slider = q(panel, "[data-bpos]");
    slider.value = state.position;
    slider.addEventListener("input", () => { state.position = parseFloat(slider.value); drawBoundary(panel); });
    q(panel, "[data-bside]").addEventListener("change", (e) => { state.in_side = e.target.value; drawBoundary(panel); });
    q(panel, "[data-compare]").addEventListener("click", compareAll);
  }

  function wireDropzone(panel, dz) {
    const role = dz.dataset.role, input = q(dz, "[data-fileinput]");
    dz.addEventListener("click", () => input.click());
    input.addEventListener("change", () => input.files[0] && upload(panel, dz, role, input.files[0]));
    ["dragover", "dragenter"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("dragover"); }));
    ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, () => dz.classList.remove("dragover")));
    dz.addEventListener("drop", (e) => { e.preventDefault(); e.dataTransfer.files[0] && upload(panel, dz, role, e.dataTransfer.files[0]); });
  }

  async function upload(panel, dz, role, file) {
    dz.querySelector(".dz-title").textContent = "Uploading…";
    try {
      const res = await API.upload(role, file, null);
      dz.classList.add("has-file");
      dz.querySelector(".dz-title").textContent = file.name;
      if (role.endsWith("_video")) showVideo(panel, role);
      window.toast(`${role.replace("_", " ")} uploaded`, "success");
      Workflow.refresh();
    } catch (err) {
      window.toast(err.message, "error");
      dz.querySelector(".dz-title").textContent = "click to browse";
    }
  }

  function showVideo(panel, role) {
    q(panel, "[data-stagewrap]").classList.remove("hidden");
    const v = q(panel, "[data-video]"); v.hidden = false; v.src = API.mediaUrl(role);
    const c = q(panel, "[data-boundary]"); c.hidden = false;
    v.addEventListener("loadedmetadata", () => drawBoundary(panel), { once: true });
    bindBoundaryDrag(panel);
    drawBoundary(panel);
  }

  function drawBoundary(panel) {
    const c = q(panel, "[data-boundary]"); if (c.hidden) return;
    const rect = c.getBoundingClientRect();
    c.width = rect.width; c.height = rect.height;
    const ctx = c.getContext("2d");
    ctx.clearRect(0, 0, c.width, c.height);
    const st = panels[panel.dataset.panel].state, frac = st.position / POSMAX[st.kind];
    ctx.strokeStyle = "#F7D65A"; ctx.lineWidth = 2.5; ctx.setLineDash([8, 5]);
    ctx.beginPath();
    if (ORIENT[st.kind] === "horizontal") { const y = frac * c.height; ctx.moveTo(0, y); ctx.lineTo(c.width, y); }
    else { const x = frac * c.width; ctx.moveTo(x, 0); ctx.lineTo(x, c.height); }
    ctx.stroke(); ctx.setLineDash([]);
    // shade the IN side faintly
    ctx.fillStyle = "rgba(63,143,91,0.12)";
    if (ORIENT[st.kind] === "horizontal") {
      const y = frac * c.height;
      st.in_side === "above" ? ctx.fillRect(0, 0, c.width, y) : ctx.fillRect(0, y, c.width, c.height - y);
    } else {
      const x = frac * c.width;
      st.in_side === "left" ? ctx.fillRect(0, 0, x, c.height) : ctx.fillRect(x, 0, c.width - x, c.height);
    }
  }

  function bindBoundaryDrag(panel) {
    const c = q(panel, "[data-boundary]"); if (c._bound) return; c._bound = true;
    const st = panels[panel.dataset.panel].state, slider = q(panel, "[data-bpos]");
    let drag = false;
    const move = (e) => {
      const r = c.getBoundingClientRect();
      const frac = ORIENT[st.kind] === "horizontal"
        ? (e.clientY - r.top) / r.height : (e.clientX - r.left) / r.width;
      st.position = Math.max(0, Math.min(1, frac)) * POSMAX[st.kind];
      slider.value = st.position; drawBoundary(panel);
    };
    c.addEventListener("pointerdown", (e) => { drag = true; c.setPointerCapture(e.pointerId); move(e); });
    c.addEventListener("pointermove", (e) => drag && move(e));
    c.addEventListener("pointerup", () => { drag = false; });
  }

  async function compareAll() {
    const boundaries = {};
    Object.values(panels).forEach((p) => {
      boundaries[p.state.kind] = { orientation: ORIENT[p.state.kind], position: p.state.position, in_side: p.state.in_side };
    });
    try {
      const res = await API.postJSON("/api/metrics/compare", { boundaries });
      Object.entries(res.panels).forEach(([kind, data]) => data.available && renderPanel(kind, data));
      renderCards(res.cards);
      Workflow.refresh();
    } catch (err) { window.toast(err.message, "error"); }
  }

  function renderPanel(kind, data) {
    const panel = panels[kind].el;
    const acc = q(panel, "[data-accuracy]");
    acc.textContent = data.accuracy + "%";
    acc.classList.toggle("low", data.accuracy < 70);
    const tbody = q(panel, "[data-table]");
    tbody.innerHTML = data.rows.map((r) => `<tr>
      <td class="num">${r.i}</td>
      <td class="num">${r.t == null ? "-" : r.t.toFixed(2)}</td>
      <td class="num">${r.x == null ? "-" : "(" + r.x.toFixed(2) + ", " + r.y.toFixed(2) + ")"}</td>
      <td>${badge(r.call)}</td>
      <td>${badge(r.pred)}</td></tr>`).join("") ||
      '<tr><td colspan="5" class="muted">no rows</td></tr>';
    renderConfusion(panel, data.confusion);
  }

  function renderConfusion(panel, confusion) {
    const wrap = q(panel, "[data-confusion-wrap]"); wrap.hidden = false;
    let html = "<tr><th class='corner'></th>" + PRED.map((p) => `<th>${p}</th>`).join("") + "</tr>";
    TRUTH.forEach((t) => {
      html += `<tr><th>${t}</th>` + PRED.map((p) => {
        const v = (confusion[t] && confusion[t][p]) || 0;
        const diag = (t === p) || (t === "ON_LINE" && p === "IN");
        return `<td class="${diag ? "diag" : ""}">${v}</td>`;
      }).join("") + "</tr>";
    });
    q(panel, "[data-confusion]").innerHTML = html;
  }

  function renderCards(cards) {
    const host = document.getElementById("cards");
    host.innerHTML = cards.map((c) => {
      const accent = c.label === "Accuracy" ? "accent" : c.placeholder ? "placeholder" : "";
      const unit = c.unit ? ` <span class="u">${c.unit}</span>` : "";
      return `<div class="stat ${accent}"><div class="k">${c.label}</div><div class="v num">${c.value}${unit}</div></div>`;
    }).join("");
  }

  let restored = false;
  function restore(state) {
    if (restored || !state) return; restored = true;
    Object.values(panels).forEach((p) => {
      const vrole = p.state.kind + "_video", crole = p.state.kind + "_csv";
      if (state.uploads && state.uploads[vrole]) showVideo(p.el, vrole);
      if (state.uploads && state.uploads[crole]) {
        const dz = p.el.querySelector(`[data-role="${crole}"]`);
        if (dz) { dz.classList.add("has-file"); dz.querySelector(".dz-title").textContent = "CSV uploaded"; }
      }
    });
    if (state.metrics && Array.isArray(state.metrics.cards) && state.metrics.cards.length) renderCards(state.metrics.cards);
  }
})();
