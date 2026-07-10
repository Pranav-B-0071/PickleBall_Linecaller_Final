/* keypoint-editor.js - drag / zoom / pan / undo / redo / reset over a frame.
   Keypoints are kept in the frame's natural pixel space; the canvas draws the
   frame + overlay under a fit-scale * zoom transform so everything stays aligned. */
(function () {
  "use strict";

  class KeypointEditor {
    constructor(canvas, opts) {
      this.canvas = canvas;
      this.ctx = canvas.getContext("2d");
      this.opts = opts || {};
      this.kps = [];
      this.initial = [];
      this.zoom = 1; this.panX = 0; this.panY = 0;
      this.dpr = window.devicePixelRatio || 1;
      this.dragIndex = -1; this.panning = false;
      this.history = []; this.future = [];
      this._bind();
    }

    // `points` may be an array of [u,v,vis], or a function (naturalW, naturalH)
    // -> array (used to seed a default box once the frame size is known).
    load(frameUrl, points) {
      this.history = []; this.future = [];
      const img = new Image();
      img.onload = () => {
        this.frame = img;
        const pts = typeof points === "function" ? points(img.naturalWidth, img.naturalHeight) : points;
        this.kps = pts.map((p) => p.slice());
        this.initial = pts.map((p) => p.slice());
        this.fit(); this.render(); this._emit();
      };
      img.onerror = () => window.toast("could not load the calibration frame", "error");
      img.src = frameUrl;
    }

    // Swap the background frame (static camera => same resolution/transform),
    // keeping the current keypoints and view. Used by the frame stepper.
    setBackground(frameUrl) {
      const img = new Image();
      img.onload = () => { this.frame = img; this.render(); };
      img.src = frameUrl;
    }

    fit() {
      const rect = this.canvas.getBoundingClientRect();
      this.canvas.width = Math.round(rect.width * this.dpr);
      this.canvas.height = Math.round(rect.height * this.dpr);
      const nw = this.frame.naturalWidth, nh = this.frame.naturalHeight;
      this.base = Math.min(rect.width / nw, rect.height / nh);
      this.zoom = 1;
      const eff = this.base;
      this.panX = (rect.width - nw * eff) / 2;
      this.panY = (rect.height - nh * eff) / 2;
    }

    get eff() { return this.base * this.zoom; }

    toScreen(u, v) { return [u * this.eff + this.panX, v * this.eff + this.panY]; }
    toImage(x, y) { return [(x - this.panX) / this.eff, (y - this.panY) / this.eff]; }

    render() {
      if (!this.frame) return;
      const ctx = this.ctx, d = this.dpr;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
      ctx.setTransform(this.eff * d, 0, 0, this.eff * d, this.panX * d, this.panY * d);
      ctx.drawImage(this.frame, 0, 0);
      const drawFn = this.opts.draw || window.CourtOverlay.draw;
      drawFn(ctx, this.kps, { scale: this.eff, activeIndex: this.dragIndex, labels: this.opts.labels !== false });
    }

    // -- history --
    snapshot() { this.history.push(this.kps.map((p) => p.slice())); if (this.history.length > 50) this.history.shift(); this.future = []; }
    undo() { if (!this.history.length) return; this.future.push(this.kps.map((p) => p.slice())); this.kps = this.history.pop(); this.render(); this._emit(); }
    redo() { if (!this.future.length) return; this.history.push(this.kps.map((p) => p.slice())); this.kps = this.future.pop(); this.render(); this._emit(); }
    reset() { this.snapshot(); this.kps = this.initial.map((p) => p.slice()); this.render(); this._emit(); }

    zoomBy(factor, cx, cy) {
      const rect = this.canvas.getBoundingClientRect();
      cx = cx == null ? rect.width / 2 : cx; cy = cy == null ? rect.height / 2 : cy;
      const [iu, iv] = this.toImage(cx, cy);
      this.zoom = Math.min(6, Math.max(1, this.zoom * factor));
      // keep the cursor point stationary
      this.panX = cx - iu * this.eff; this.panY = cy - iv * this.eff;
      this._clampPan(rect);
      this.render(); this._emit();
    }

    _clampPan(rect) {
      if (this.zoom <= 1) { this.fitPanCenter(rect); return; }
      const nw = this.frame.naturalWidth * this.eff, nh = this.frame.naturalHeight * this.eff;
      this.panX = Math.min(0, Math.max(rect.width - nw, this.panX));
      this.panY = Math.min(0, Math.max(rect.height - nh, this.panY));
    }
    fitPanCenter(rect) {
      const nw = this.frame.naturalWidth * this.eff, nh = this.frame.naturalHeight * this.eff;
      this.panX = (rect.width - nw) / 2; this.panY = (rect.height - nh) / 2;
    }

    _hit(x, y) {
      let best = -1, bestD = 14; // screen px
      this.kps.forEach((p, i) => {
        const [sx, sy] = this.toScreen(p[0], p[1]);
        const d = Math.hypot(sx - x, sy - y);
        if (d < bestD) { bestD = d; best = i; }
      });
      return best;
    }

    _bind() {
      const c = this.canvas;
      const pos = (e) => { const r = c.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };

      c.addEventListener("pointerdown", (e) => {
        if (!this.frame) return;
        const [x, y] = pos(e);
        const hit = this._hit(x, y);
        c.setPointerCapture(e.pointerId);
        if (hit >= 0) { this.snapshot(); this.dragIndex = hit; }
        else { this.panning = this.zoom > 1; this._panStart = [x, y, this.panX, this.panY]; }
        this.render();
      });

      c.addEventListener("pointermove", (e) => {
        if (!this.frame) return;
        const [x, y] = pos(e);
        if (this.dragIndex >= 0) {
          const [u, v] = this.toImage(x, y);
          this.kps[this.dragIndex][0] = u; this.kps[this.dragIndex][1] = v;
          if (this.kps[this.dragIndex][2] < 1) this.kps[this.dragIndex][2] = 2; // dragged => known
          this.render(); this._emit();
        } else if (this.panning) {
          this.panX = this._panStart[2] + (x - this._panStart[0]);
          this.panY = this._panStart[3] + (y - this._panStart[1]);
          this._clampPan(c.getBoundingClientRect());
          this.render();
        } else {
          c.style.cursor = this._hit(x, y) >= 0 ? "grab" : (this.zoom > 1 ? "move" : "default");
        }
      });

      const end = (e) => { this.dragIndex = -1; this.panning = false; this.render(); };
      c.addEventListener("pointerup", end);
      c.addEventListener("pointercancel", end);

      c.addEventListener("wheel", (e) => {
        if (!this.frame) return;
        e.preventDefault();
        const [x, y] = pos(e);
        this.zoomBy(e.deltaY < 0 ? 1.12 : 1 / 1.12, x, y);
      }, { passive: false });
    }

    _emit() { if (this.opts.onChange) this.opts.onChange(this.kps, this.zoom); }
    getKeypoints() { return this.kps.map((p) => [Math.round(p[0] * 100) / 100, Math.round(p[1] * 100) / 100, p[2]]); }
  }

  window.KeypointEditor = KeypointEditor;
})();
