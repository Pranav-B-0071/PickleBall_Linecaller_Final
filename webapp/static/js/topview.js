/* topview.js - top-down HALF pickleball court + ball trajectory.
   GridTrackNet-style fading tail, bounce markers that auto-expire after a TTL.
   Court frame (ft): near half y in [22,44], x in [0,20]; net drawn at top. */
(function () {
  "use strict";
  const NET_Y = 22, BASE_Y = 44, KITCHEN_Y = 29, W = 20;
  const PAD = 26, TAIL_S = 0.65;

  class TopView {
    constructor(canvas, opts) {
      this.canvas = canvas;
      this.ctx = canvas.getContext("2d");
      this.ttl = (opts && opts.ttl) || 10;
      this.data = { trajectory: [], bounces: [] };
      this.dpr = window.devicePixelRatio || 1;
      this._resize();
    }

    _resize() {
      const rect = this.canvas.getBoundingClientRect();
      const w = rect.width || this.canvas.width, h = rect.height || this.canvas.height;
      this.canvas.width = Math.round(w * this.dpr);
      this.canvas.height = Math.round(h * this.dpr);
      this.vw = w; this.vh = h;
      const cw = w - 2 * PAD, ch = h - 2 * PAD;
      this.s = Math.min(cw / W, ch / (BASE_Y - NET_Y));
      this.ox = (w - W * this.s) / 2;
      this.oy = (h - (BASE_Y - NET_Y) * this.s) / 2;
    }

    setData(d) { this.data = d || { trajectory: [], bounces: [] }; this.render(0); }

    C(x, y) { return [this.ox + (x / W) * (W * this.s), this.oy + ((y - NET_Y) / (BASE_Y - NET_Y)) * ((BASE_Y - NET_Y) * this.s)]; }

    _court(ctx) {
      const line = (x1, y1, x2, y2, w, col) => {
        const a = this.C(x1, y1), b = this.C(x2, y2);
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]);
        ctx.lineWidth = w; ctx.strokeStyle = col; ctx.stroke();
      };
      // playing-surface fill
      const tl = this.C(0, NET_Y), br = this.C(W, BASE_Y);
      const grd = ctx.createLinearGradient(0, tl[1], 0, br[1]);
      grd.addColorStop(0, "#1c6f66"); grd.addColorStop(1, "#15564f");
      ctx.fillStyle = grd; ctx.fillRect(tl[0], tl[1], br[0] - tl[0], br[1] - tl[1]);
      // ROI wash (the whole near half is the ROI)
      ctx.fillStyle = "rgba(46,160,140,0.14)";
      ctx.fillRect(tl[0], tl[1], br[0] - tl[0], br[1] - tl[1]);

      const white = "rgba(233,240,238,0.85)";
      line(0, NET_Y, W, NET_Y, 3, "#e9f0ee");                 // net
      line(0, BASE_Y, W, BASE_Y, 2.5, white);                 // near baseline
      line(0, NET_Y, 0, BASE_Y, 2.5, white);                  // sidelines
      line(W, NET_Y, W, BASE_Y, 2.5, white);
      line(0, KITCHEN_Y, W, KITCHEN_Y, 2, "rgba(233,240,238,0.7)"); // kitchen
      line(W / 2, KITCHEN_Y, W / 2, BASE_Y, 2, "rgba(233,240,238,0.6)"); // center
      // kitchen zone (CAM3) - amber band from net to the kitchen line
      if (this.data.cam3_calibrated) {
        const k0 = this.C(0, NET_Y), k1 = this.C(W, KITCHEN_Y);
        ctx.fillStyle = "rgba(199,154,62,0.16)";
        ctx.fillRect(k0[0], k0[1], k1[0] - k0[0], k1[1] - k0[1]);
        ctx.fillStyle = "rgba(224,186,110,0.85)"; ctx.font = "10px sans-serif";
        ctx.fillText("KITCHEN · CAM 3", k0[0] + 6, this.C(0, KITCHEN_Y)[1] - 6);
      }

      // net label
      ctx.fillStyle = "rgba(233,240,238,0.55)"; ctx.font = "11px sans-serif";
      ctx.fillText("NET", this.C(0, NET_Y)[0] + 4, this.C(0, NET_Y)[1] - 6);
    }

    _sample(t) {
      const tr = this.data.trajectory;
      if (!tr.length) return null;
      if (t <= tr[0].t) return tr[0];
      if (t >= tr[tr.length - 1].t) return tr[tr.length - 1];
      let lo = 0, hi = tr.length - 1;
      while (hi - lo > 1) { const m = (lo + hi) >> 1; (tr[m].t <= t ? lo = m : hi = m); }
      const a = tr[lo], b = tr[hi], u = (t - a.t) / (b.t - a.t || 1);
      return { x: a.x + u * (b.x - a.x), y: a.y + u * (b.y - a.y), in_roi: b.in_roi };
    }

    render(t) {
      const ctx = this.ctx;
      ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
      ctx.clearRect(0, 0, this.vw, this.vh);
      this._court(ctx);

      // fading tail
      const tr = this.data.trajectory;
      const tail = tr.filter((p) => p.t <= t && p.t > t - TAIL_S);
      for (let i = 1; i < tail.length; i++) {
        const a = this.C(tail[i - 1].x, tail[i - 1].y), b = this.C(tail[i].x, tail[i].y);
        const age = (t - tail[i].t) / TAIL_S;
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]);
        ctx.lineWidth = 4 * (1 - age) + 1;
        ctx.strokeStyle = `rgba(247,214,90,${0.85 * (1 - age)})`;
        ctx.lineCap = "round"; ctx.stroke();
      }

      // bounce markers (auto-expire after ttl)
      (this.data.bounces || []).forEach((bnc) => {
        if (bnc.t > t || t - bnc.t > this.ttl) return;
        const age = (t - bnc.t) / this.ttl;
        const p = this.C(bnc.x, bnc.y);
        const col = bnc.verdict === "IN" ? "63,143,91" : bnc.verdict === "OUT" ? "192,80,58" : "150,150,150";
        ctx.beginPath(); ctx.arc(p[0], p[1], 9 + 6 * Math.min(1, (t - bnc.t) * 3), 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(${col},${0.9 * (1 - age)})`; ctx.lineWidth = 2.5; ctx.stroke();
        ctx.beginPath(); ctx.arc(p[0], p[1], 4, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${col},${0.95 * (1 - age * 0.6)})`; ctx.fill();
        if (bnc.in_kitchen) {   // amber square marks a kitchen bounce (CAM3)
          ctx.strokeStyle = `rgba(199,154,62,${0.9 * (1 - age)})`; ctx.lineWidth = 2;
          ctx.strokeRect(p[0] - 8, p[1] - 8, 16, 16);
        }
      });

      // live ball
      const s = this._sample(t);
      if (s) {
        const p = this.C(s.x, s.y);
        ctx.beginPath(); ctx.arc(p[0], p[1], 9, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(247,214,90,0.18)"; ctx.fill();
        ctx.beginPath(); ctx.arc(p[0], p[1], 5, 0, Math.PI * 2);
        ctx.fillStyle = "#F7D65A"; ctx.strokeStyle = "#7a6a1e"; ctx.lineWidth = 1.4; ctx.fill(); ctx.stroke();
      }
    }
  }

  window.TopView = TopView;
})();
