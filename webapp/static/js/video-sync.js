/* video-sync.js - SyncedPlayerGroup: N <video>s sharing one aligned timeline.
   Alignment uses the sync convention aligned_frame = local_frame - offset, so a
   given master time t maps to each clip's currentTime = t + offset/fps. The
   first-added clip (CAM 1) is the master clock; others are drift-corrected. */
(function () {
  "use strict";
  const DRIFT = 0.033;   // ~1 frame @30fps before we hard-correct

  class SyncedPlayerGroup {
    constructor(opts) {
      this.fps = (opts && opts.fps) || 60;
      this.members = [];        // {video, role, offsetSec}
      this.rate = 1;
      this.onTick = (opts && opts.onTick) || null;
      this._raf = null;
      this._loop = this._loop.bind(this);
    }

    add(video, role, offsetFrames) {
      if (!video) return;
      this.members.push({ video, role, offsetSec: (offsetFrames || 0) / this.fps });
      video.playbackRate = this.rate;
    }
    setOffset(role, offsetFrames) {
      const m = this.members.find((x) => x.role === role);
      if (m) { m.offsetSec = (offsetFrames || 0) / this.fps; if (this.paused) this.seek(this.time()); }
    }
    get master() { return this.members[0]; }
    get present() { return this.members.length > 0; }

    duration() {
      if (!this.present) return 0;
      return Math.min(...this.members.map((m) => Math.max(0, (m.video.duration || 0) - m.offsetSec)));
    }
    time() { return this.master ? Math.max(0, this.master.video.currentTime - this.master.offsetSec) : 0; }
    get paused() { return !this.master || this.master.video.paused; }

    seek(t) {
      t = Math.max(0, Math.min(t, this.duration()));
      this.members.forEach((m) => {
        const target = t + m.offsetSec;
        if (Number.isFinite(m.video.duration)) m.video.currentTime = Math.max(0, Math.min(target, m.video.duration - 0.001));
      });
      if (this.onTick) this.onTick(t, this.frame(t));
    }
    frame(t) { return Math.round((t == null ? this.time() : t) * this.fps); }

    play() { if (!this.present) return; this.members.forEach((m) => m.video.play().catch(() => {})); this._start(); }
    pause() { this.members.forEach((m) => m.video.pause()); }
    toggle() { this.paused ? this.play() : this.pause(); }

    stepFrames(n) {
      this.pause();
      this.seek(this.time() + n / this.fps);
    }
    setRate(r) { this.rate = r; this.members.forEach((m) => (m.video.playbackRate = r)); }

    _start() { if (!this._raf) this._raf = requestAnimationFrame(this._loop); }
    _loop() {
      if (this.paused) { this._raf = null; if (this.onTick) this.onTick(this.time(), this.frame()); return; }
      const t = this.time();
      // correct drift on the follower clips
      for (let i = 1; i < this.members.length; i++) {
        const m = this.members[i];
        const want = t + m.offsetSec;
        if (Math.abs(m.video.currentTime - want) > DRIFT && Number.isFinite(m.video.duration)) {
          m.video.currentTime = Math.max(0, Math.min(want, m.video.duration - 0.001));
        }
      }
      if (t >= this.duration() - 0.02) { this.pause(); }
      if (this.onTick) this.onTick(t, this.frame(t));
      this._raf = requestAnimationFrame(this._loop);
    }
  }

  window.SyncedPlayerGroup = SyncedPlayerGroup;
})();
