/* court-overlay.js - draw the 12 keypoints + derived near-half ROI.
   Pure drawing + the client-side ROI derivation (mirrors services/roi.py), so
   the editor can show the ROI update live as points are dragged. */
(function () {
  "use strict";
  const CORNERS = [0, 2, 9, 11];              // court_model.CORNER_IDX
  const LABELS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"];

  // Full court skeleton: every court line as index pairs (Appendix A order).
  //  0 1 2 = far baseline    5 4 3 = far kitchen    6 7 8 = near kitchen
  //  11 10 9 = near baseline   0-5-6-11 = left rail   2-3-8-9 = right rail
  //  1-4 = far centerline      7-10 = near centerline
  const LINES = [
    [0, 1], [1, 2], [5, 4], [4, 3], [6, 7], [7, 8], [11, 10], [10, 9],
    [0, 5], [5, 6], [6, 11], [2, 3], [3, 8], [8, 9], [1, 4], [7, 10],
  ];

  const mid = (kps, a, b) => [(kps[a][0] + kps[b][0]) / 2, (kps[a][1] + kps[b][1]) / 2];

  // Net line sits at y=22, exactly midway between the far kitchen line (y=15:
  // pts 5,3) and the near kitchen line (y=29: pts 6,8). Averaging those ADJACENT
  // pairs (5&6 left, 3&8 right) lands on the net far more accurately than the old
  // whole-court baseline average (0&11 / 2&9), which perspective badly skews.
  function netEndpoints(kps) {
    return [mid(kps, 5, 6), mid(kps, 3, 8)];   // [left, right]
  }

  function deriveRoi(kps) {
    const [netL, netR] = netEndpoints(kps);
    const nearL = [kps[11][0], kps[11][1]], nearR = [kps[9][0], kps[9][1]];
    return [netL, netR, nearR, nearL];        // net line -> near baseline quad
  }

  // Draw in IMAGE-space coordinates; `scale` keeps marks a constant screen size.
  function draw(ctx, kps, opts) {
    opts = opts || {};
    const s = opts.scale || 1;
    const r = 6 / s, rc = 8 / s;

    // near-half ROI fill + edge
    const roi = deriveRoi(kps);
    ctx.beginPath();
    roi.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])));
    ctx.closePath();
    ctx.fillStyle = "rgba(46,160,140,0.22)";
    ctx.fill();
    ctx.lineWidth = 2 / s;
    ctx.strokeStyle = "#2EA08C";
    ctx.stroke();

    // FULL court skeleton - every line connected (not just the corners), so a
    // mis-ordered/mis-placed keypoint shows up as a tangled court immediately.
    ctx.strokeStyle = "rgba(233,240,238,0.85)";
    ctx.lineWidth = 1.6 / s;
    ctx.beginPath();
    LINES.forEach(([a, b]) => { ctx.moveTo(kps[a][0], kps[a][1]); ctx.lineTo(kps[b][0], kps[b][1]); });
    ctx.stroke();

    // net line (derived: average of the kitchen-line pairs straddling the net)
    const [nl, nr] = netEndpoints(kps);
    ctx.beginPath(); ctx.moveTo(nl[0], nl[1]); ctx.lineTo(nr[0], nr[1]);
    ctx.strokeStyle = "rgba(120,200,235,0.9)"; ctx.lineWidth = 2 / s; ctx.stroke();

    // keypoints
    kps.forEach((p, i) => {
      const isCorner = CORNERS.includes(i);
      const isActive = i === opts.activeIndex;
      ctx.beginPath();
      ctx.arc(p[0], p[1], isCorner ? rc : r, 0, Math.PI * 2);
      ctx.fillStyle = isCorner ? "#2B2A26" : "#D8C39A";
      ctx.fill();
      ctx.lineWidth = (isActive ? 3 : 1.5) / s;
      ctx.strokeStyle = isActive ? "#3F8F5B" : "#ffffff";
      ctx.stroke();
      if (opts.labels) {
        ctx.fillStyle = "rgba(20,24,33,0.9)";
        ctx.font = `${12 / s}px sans-serif`;
        ctx.fillText(LABELS[i], p[0] + rc + 2 / s, p[1] - rc);
      }
    });
  }

  // 4-point kitchen box (CAM3 manual calibration): a draggable quad, no ROI.
  function drawBox(ctx, pts, opts) {
    opts = opts || {};
    const s = opts.scale || 1, r = 8 / s;
    ctx.beginPath();
    pts.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])));
    ctx.closePath();
    ctx.fillStyle = "rgba(199,154,62,0.24)";   // amber kitchen wash
    ctx.fill();
    ctx.lineWidth = 2 / s; ctx.strokeStyle = "#C79A3E"; ctx.stroke();
    pts.forEach((p, i) => {
      const active = i === opts.activeIndex;
      ctx.beginPath(); ctx.arc(p[0], p[1], r, 0, Math.PI * 2);
      ctx.fillStyle = "#2B2A26"; ctx.fill();
      ctx.lineWidth = (active ? 3 : 1.5) / s; ctx.strokeStyle = active ? "#3F8F5B" : "#ffffff"; ctx.stroke();
    });
  }

  window.CourtOverlay = { deriveRoi, draw, CORNERS };
  window.BoxOverlay = { draw: drawBox };
})();
