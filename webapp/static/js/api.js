/* api.js - session-aware fetch helpers + toast. Exposes window.API / window.toast. */
(function () {
  "use strict";
  const SKEY = "pb_session";
  let sid = localStorage.getItem(SKEY);
  let ready = null;   // memoized: validate/create the session once per page load

  async function ensureSession() {
    if (ready) return ready;
    ready = (async () => {
      // Reuse the stored session only if the server still has it. After a
      // restart (fresh run) it won't, so we transparently start a new one -
      // this is why old uploads no longer reappear across runs.
      if (sid) {
        try {
          const r = await fetch("/api/session/state", { headers: { "X-Session-Id": sid } });
          if (r.ok) return sid;
        } catch (e) { /* fall through to create */ }
        localStorage.removeItem(SKEY);
        sid = null;
      }
      const r = await fetch("/api/session/new", { method: "POST" });
      sid = (await r.json()).session_id;
      localStorage.setItem(SKEY, sid);
      return sid;
    })();
    return ready;
  }

  function headers(extra) {
    return Object.assign({ "X-Session-Id": sid || "" }, extra || {});
  }

  async function handle(r) {
    let j;
    try { j = await r.json(); } catch (e) { j = { ok: false, error: "bad server response" }; }
    if (!r.ok || j.ok === false) throw new Error(j.error || ("HTTP " + r.status));
    return j;
  }

  async function getJSON(url) {
    await ensureSession();
    return handle(await fetch(url, { headers: headers() }));
  }

  async function postJSON(url, body) {
    await ensureSession();
    return handle(await fetch(url, {
      method: "POST",
      headers: headers({ "Content-Type": "application/json" }),
      body: JSON.stringify(body || {}),
    }));
  }

  function upload(role, file, onProgress) {
    return ensureSession().then(() => new Promise((res, rej) => {
      const fd = new FormData();
      fd.append("role", role);
      fd.append("file", file);
      fd.append("session_id", sid);
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/api/upload");
      xhr.setRequestHeader("X-Session-Id", sid);
      xhr.upload.onprogress = (e) => { if (onProgress && e.lengthComputable) onProgress(e.loaded / e.total); };
      xhr.onload = () => {
        let j; try { j = JSON.parse(xhr.responseText); } catch (e) { return rej(new Error("bad response")); }
        if (xhr.status < 300 && j.ok !== false) res(j); else rej(new Error(j.error || ("HTTP " + xhr.status)));
      };
      xhr.onerror = () => rej(new Error("network error during upload"));
      xhr.send(fd);
    }));
  }

  const mediaUrl = (role) => `/media/${sid}/video/${role}`;
  const trackedUrl = (role) => `/media/${sid}/tracked/${role}`;
  const frameUrl = (role, frame = 0) => `/media/${sid}/frame/${role}?frame=${frame | 0}`;

  window.API = { ensureSession, getJSON, postJSON, upload, mediaUrl, trackedUrl,
                 frameUrl, sid: () => sid };

  window.toast = function (msg, kind) {
    const host = document.getElementById("toastHost");
    if (!host) return;
    const el = document.createElement("div");
    el.className = "toast " + (kind || "");
    el.textContent = msg;
    host.appendChild(el);
    setTimeout(() => { el.style.opacity = "0"; el.style.transform = "translateY(8px)"; }, 4200);
    setTimeout(() => el.remove(), 4600);
  };
})();
