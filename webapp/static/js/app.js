/* app.js - shared workflow behaviour: the stepper completion state.
   Pages call Workflow.refresh() after a state-changing action. */
(function () {
  "use strict";

  // Derive which workflow steps are complete from the session state.
  function completion(state) {
    const up = state.uploads || {};
    const sync = state.sync || {};
    const cal = state.calibration || {};
    const met = state.metrics || {};
    const ana = state.analysis || {};
    const offsets = sync.offsets || {};
    return {
      upload: !!(up.cam1 && up.cam2),
      sync: Object.keys(offsets).length > 1,
      calibrate: !!(cal.cam1 && cal.cam2 && cal.cam1.locked && cal.cam2.locked),
      metrics: Array.isArray(met.cards) && met.cards.length > 0,
      analysis: !!(ana.summary),
    };
  }

  function paint(done) {
    const active = document.body.dataset.active;
    document.querySelectorAll(".stepper .step").forEach((el) => {
      const key = el.dataset.step;
      el.classList.toggle("done", !!done[key] && key !== active);
      const dot = el.querySelector(".dot");
      if (done[key] && key !== active) dot.textContent = "✓";
    });
  }

  const Workflow = {
    async refresh() {
      try {
        const j = await API.getJSON("/api/session/state");
        this.state = j.state;
        paint(completion(j.state));
        document.dispatchEvent(new CustomEvent("session-state", { detail: j.state }));
        return j.state;
      } catch (e) { /* new session - nothing to paint yet */ }
    },
  };

  window.Workflow = Workflow;
  document.addEventListener("DOMContentLoaded", () => {
    API.ensureSession().then(() => Workflow.refresh());
  });
})();
