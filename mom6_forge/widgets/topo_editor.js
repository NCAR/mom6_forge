// TopoEditor front end (Python side: topo_editor.py).
//
// Draws a Topo in grid index space: cell (i, j) is one pixel of an nx × ny image, j up.
// Arrays are row-major with k = j * nx + i. No mom6_forge logic lives here: the page
// draws state and sends one message per user action.
//
// Python → page
//   traits: depth (float32), mask (uint8), lon, lat (float32), basins (int32, basinmask
//           view only), nx, ny, view, min_depth, max_depth, has_history, has_stats,
//           has_user_mask, can_undo, can_redo, branches, current_branch, tags, status,
//           show_open_boundaries, open_boundary_runs ({side: [{start, stop, tiny}]}, cell
//           indices along the side, stop exclusive; {} while the overlay is off)
//   msg:    {kind: "patch", n} + buffers [cells int32 (j, i) pairs, depth float32, mask uint8]
//           {kind: "reload"}: the depth and mask traits are current, redraw from them
// Page → Python
//   {op: "edit", edit: {op, value?, statistic?, cell?, rect?}}, plus one int32 buffer of
//   (j, i) pairs for a brush stroke; {op: "view", view}; {op: "undo" | "redo" | "reset"};
//   {op: "tag" | "create_branch" | "checkout", name}; {op: "open_boundaries", show};
//   {op: "sync"} once per render.
//
// Performance at 2000 × 2000: the image is rebuilt in full only when the arrays, the
// view or the colour scale change. Patches and brush strokes write single pixels and
// upload only their dirty rectangle; drawing is one scaled drawImage per frame.

/* ============================== colours ============================== */

// Depth colormap, light shelf to dark abyss, indexed on a square-root depth scale.
const DEPTH_STOPS = [
  [0, [214, 236, 228]], [0.16, [150, 208, 208]], [0.36, [86, 160, 194]],
  [0.6, [48, 102, 164]], [0.84, [32, 58, 120]], [1, [18, 28, 70]],
];
const DEPTH_LUT = []; // 256 entries of [r, g, b]
for (let q = 0; q < 256; q++) {
  const t = q / 255;
  let s = 0;
  while (s < DEPTH_STOPS.length - 2 && t > DEPTH_STOPS[s + 1][0]) s++;
  const [[t0, c0], [t1, c1]] = [DEPTH_STOPS[s], DEPTH_STOPS[s + 1]];
  DEPTH_LUT.push(c0.map((c, n) => Math.round(c + ((c1[n] - c) * (t - t0)) / (t1 - t0))));
}
const LAND = [206, 190, 152];
const OCEAN = DEPTH_LUT[150];
// Basin labels cycle through these; land stays LAND.
const BASIN_COLORS = [
  [31, 119, 180], [255, 127, 14], [44, 160, 44], [214, 39, 40], [148, 103, 189], [140, 86, 75],
  [227, 119, 194], [127, 127, 127], [188, 189, 34], [23, 190, 207], [174, 199, 232], [255, 187, 120],
];
const rgb = (c) => `rgb(${c.join(",")})`;
// Open-boundary overlay: open water along the domain edge, and runs too small to be good
// open boundaries ("tiny" in corner_diagnostics.open_boundary_runs).
const OBC_COLOR = "#CC79A7";
const OBC_TINY_COLOR = "#d7191c";
const OBC_GAP = 4; // screen pixels from the grid edge to the centre of an overlay line

/* ============================== small helpers ============================== */

// Copy a binary buffer from Python (DataView or ArrayBuffer) into an aligned typed array.
function typed(buf, Type) {
  if (!buf || !buf.byteLength) return new Type(0);
  const bytes = ArrayBuffer.isView(buf)
    ? new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength)
    : new Uint8Array(buf);
  return new Type(bytes.slice().buffer);
}

// h("button", {class, "data-action", textContent, ...}, ...children)
function h(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (key === "class") node.className = value;
    else if (key.startsWith("data-") || key.startsWith("aria-")) node.setAttribute(key, value);
    else node[key] = value;
  }
  node.append(...children);
  return node;
}

const STYLE = `
.mf-topo { display: flex; flex-wrap: wrap; gap: 12px; align-items: flex-start;
  font: 13px/1.4 var(--jp-ui-font-family, system-ui, sans-serif); color: var(--jp-ui-font-color1, #222); }
.mf-topo [hidden] { display: none !important; }
.mf-topo-rail { flex: 0 0 250px; max-width: 100%; display: flex; flex-direction: column; gap: 12px; }
.mf-topo-main { flex: 1 1 380px; min-width: 0; display: flex; flex-direction: column; gap: 6px; }
.mf-topo-section h4 { margin: 0 0 4px; font-size: 11px; font-weight: 600; letter-spacing: .04em;
  text-transform: uppercase; color: var(--jp-ui-font-color2, #666); }
.mf-topo-row { display: flex; flex-wrap: wrap; gap: 4px; align-items: center; margin-top: 4px; }
.mf-topo-row > input[type=number], .mf-topo-row > input[type=text], .mf-topo-row > select { flex: 1 1 80px; }
.mf-topo button { font: inherit; padding: 3px 8px; border: 1px solid var(--jp-border-color1, #bbb);
  border-radius: 4px; background: var(--jp-layout-color1, #fff); color: inherit; cursor: pointer; }
.mf-topo button:disabled { opacity: .45; cursor: default; }
.mf-topo button[aria-pressed=true] { background: var(--jp-brand-color1, #1976d2);
  border-color: var(--jp-brand-color1, #1976d2); color: #fff; }
.mf-topo input, .mf-topo select { font: inherit; min-width: 0; padding: 2px 4px; color: inherit;
  background: var(--jp-layout-color0, #fff); border: 1px solid var(--jp-border-color1, #bbb); border-radius: 4px; }
.mf-topo-hint { font-size: 12px; color: var(--jp-ui-font-color2, #666); }
.mf-topo-stage { position: relative; height: clamp(360px, 62vh, 640px); overflow: hidden; outline: none;
  border: 1px solid var(--jp-border-color1, #bbb); border-radius: 6px;
  background: var(--jp-layout-color2, #eee); touch-action: none; }
.mf-topo-stage:focus-visible { border-color: var(--jp-brand-color1, #1976d2); }
.mf-topo-stage canvas { display: block; width: 100%; height: 100%; cursor: crosshair; }
.mf-topo-stage.mf-topo-panning canvas { cursor: grabbing; }
.mf-topo-zoom { position: absolute; top: 8px; right: 8px; display: grid; gap: 4px; }
.mf-topo-zoom button { width: 30px; height: 30px; padding: 0; font-weight: 600; }
.mf-topo-readout { font: 12px var(--jp-code-font-family, ui-monospace, monospace); white-space: pre-wrap; }
.mf-topo-status { font-size: 12px; color: var(--jp-ui-font-color2, #666); min-height: 1.4em; }
.mf-topo-bar { height: 10px; border-radius: 3px; border: 1px solid var(--jp-border-color1, #bbb); }
.mf-topo-ticks { position: relative; height: 16px; font: 11px var(--jp-code-font-family, monospace);
  color: var(--jp-ui-font-color2, #666); }
.mf-topo-ticks span { position: absolute; transform: translateX(-50%); white-space: nowrap; }
.mf-topo-ticks span:first-child { transform: none; }
.mf-topo-ticks span:last-child { transform: translateX(-100%); }
.mf-topo-swatch { display: inline-block; width: 12px; height: 12px; margin: 0 4px 0 10px;
  vertical-align: -2px; border-radius: 2px; }
`;

/* ============================== render ============================== */

function render({ model, el, guard, errorBox }) {
  // ---------- bookkeeping: the returned cleanup removes every listener ----------
  const cleanups = [];
  const on = (target, type, fn, opts) => {
    const g = guard(fn);
    target.addEventListener(type, g, opts);
    cleanups.push(() => target.removeEventListener(type, g, opts));
  };
  const listen = (event, fn) => {
    const g = guard(fn);
    model.on(event, g);
    cleanups.push(() => model.off(event, g));
  };
  const send = (msg, buffers) => model.send(msg, undefined, buffers);
  const sendEdit = (edit, buffers) => send({ op: "edit", edit }, buffers);

  // ---------- DOM ----------
  const btn = (action, text, title = "") =>
    h("button", { type: "button", textContent: text, title, "data-action": action });
  const input = (action, type, props = {}) => h("input", { type, ...props, "data-action": action });
  const section = (title, ...children) =>
    h("div", { class: "mf-topo-section" }, h("h4", { textContent: title }), ...children);
  const row = (...children) => h("div", { class: "mf-topo-row" }, ...children);
  const hint = (text = "") => h("div", { class: "mf-topo-hint", textContent: text });
  const label = (text) => h("span", { class: "mf-topo-hint", textContent: text });

  const ui = {
    views: {
      depth: btn("view-depth", "depth", "Colour by depth"),
      mask: btn("view-mask", "mask", "Land and ocean"),
      basinmask: btn("view-basinmask", "basinmask", "Connected ocean basins"),
    },
    obc: input("show-open-boundaries", "checkbox"),
    obcInfo: hint(),
    tools: {
      select: btn("tool-select", "Select", "Click a cell, or drag a box"),
      depth: btn("tool-depth", "Paint depth", "Paint the depth set below"),
      land: btn("tool-land", "Paint land"),
      ocean: btn("tool-ocean", "Paint ocean"),
    },
    radius: input("brush-radius", "range", { min: 0, max: 40, value: 2 }),
    radiusLabel: label(""),
    selection: h("div", { class: "mf-topo-hint", "data-role": "selection" }),
    clearSel: btn("clear-selection", "Clear selection"),
    depthValue: input("depth-value", "number", { value: 100, step: 10, title: "Depth (m)" }),
    setDepth: btn("set-depth", "Set depth"),
    setLand: btn("set-land", "Set land"),
    setOcean: btn("set-ocean", "Set ocean"),
    fill: {
      mean: btn("fill-mean", "Mean", "Set the selection to the source bathymetry's mean"),
      max: btn("fill-max", "Max", "Set the selection to the source bathymetry's max"),
      min: btn("fill-min", "Min", "Set the selection to the source bathymetry's min"),
    },
    clearMask: btn("clear-mask", "Clear manual mask", "Go back to the mask derived from depth"),
    minDepth: input("min-depth-value", "number", { step: 1, title: "Minimum depth (m)" }),
    setMinDepth: btn("set-min-depth", "Apply"),
    basinInfo: hint(),
    eraseSel: btn("erase-selected-basin", "Erase selected basin"),
    eraseDisc: btn("erase-disconnected-basins", "Erase disconnected basins"),
    undo: btn("undo", "Undo"),
    redo: btn("redo", "Redo"),
    reset: btn("reset", "Reset", "Back to the original bathymetry"),
    tagName: input("tag-name", "text", { placeholder: "Tag name" }),
    tag: btn("tag", "Tag"),
    branchName: input("branch-name", "text", { placeholder: "New branch name" }),
    createBranch: btn("create-branch", "Create branch"),
    branchSelect: h("select", { "data-action": "branch-select", title: "Branches" }),
    checkout: btn("checkout", "Checkout"),
    tags: hint(),
    zoomIn: btn("zoom-in", "+", "Zoom in"),
    zoomOut: btn("zoom-out", "−", "Zoom out"),
    zoomFit: btn("zoom-fit", "⤢", "Fit the grid"),
    canvas: h("canvas", { "data-role": "canvas", "aria-label": "Bathymetry in grid index space" }),
    readout: h("div", { class: "mf-topo-readout", "data-role": "readout" }),
    legend: h("div", { "data-role": "legend" }),
    status: h("div", { class: "mf-topo-status", "data-role": "status" }),
  };
  ui.statsRow = row(label("From source:"), ...Object.values(ui.fill));
  ui.basinSection = section("Basins", ui.basinInfo, row(ui.eraseSel), row(ui.eraseDisc));
  ui.history = h(
    "div", {},
    row(ui.undo, ui.redo, ui.reset),
    row(ui.tagName, ui.tag),
    row(ui.branchName, ui.createBranch),
    row(ui.branchSelect, ui.checkout),
    ui.tags,
  );
  ui.noHistory = hint(
    "Create the Topo with git=True to keep an edit history (undo, redo, reset, branches).",
  );
  ui.stage = h(
    "div", { class: "mf-topo-stage", tabIndex: 0 },
    ui.canvas,
    h("div", { class: "mf-topo-zoom" }, ui.zoomIn, ui.zoomOut, ui.zoomFit),
  );

  const rail = h(
    "div", { class: "mf-topo-rail" },
    section(
      "View",
      row(...Object.values(ui.views)),
      row(h("label", { class: "mf-topo-hint" }, ui.obc, " Show open boundaries")),
      ui.obcInfo,
    ),
    section("Tool", row(...Object.values(ui.tools)), row(label("Brush"), ui.radius, ui.radiusLabel)),
    section(
      "Selection",
      ui.selection,
      row(ui.depthValue, label("m"), ui.setDepth),
      row(ui.setLand, ui.setOcean, ui.clearSel),
      ui.statsRow,
    ),
    section("Mask and minimum depth", row(ui.clearMask), row(ui.minDepth, ui.setMinDepth)),
    ui.basinSection,
    section("History", ui.history, ui.noHistory),
  );
  const main = h(
    "div", { class: "mf-topo-main" },
    ui.stage, ui.readout, ui.legend, ui.status,
    hint("Scroll to zoom. Pan with the right mouse button, or hold Space and drag."),
  );
  const root = h("div", { class: "mf-topo" }, rail, main);
  // Keep JupyterLab's notebook shortcuts (Space, a, b, ...) out of the editor.
  root.setAttribute("data-lm-suppress-shortcuts", "true");
  root.setAttribute("data-p-suppress-shortcuts", "true");
  el.replaceChildren(h("style", { textContent: STYLE }), errorBox(), root);

  // ---------- state ----------
  const S = {
    nx: model.get("nx"),
    ny: model.get("ny"),
    depth: null, // Float32Array, masked depth (m)
    mask: null, // Uint8Array, 1 = ocean
    basins: new Int32Array(0),
    nBasins: 0,
    lon: typed(model.get("lon"), Float32Array),
    lat: typed(model.get("lat"), Float32Array),
    view: model.get("view"),
    minDepth: 0,
    lo: 0, // colour scale: depth lo .. hi on a square-root ramp
    hi: 1,
    off: document.createElement("canvas"), // the nx × ny image
    img: null,
    dirty: null, // [i0, row0, i1, row1]: pixels changed but not yet uploaded
    rebuild: true, // repaint every pixel on the next frame
    frame: 0,
    s: 1, // screen pixels per cell
    ox: 0, // screen position of the image's top-left corner
    oy: 0,
    dpr: 1,
    fitted: false,
    tool: "select",
    radius: 2,
    sel: null, // selected box {i0, j0, i1, j1}, inclusive
    drag: null, // selection box being dragged
    stroke: null, // brush stroke being painted
    pan: null,
    space: false,
    hover: null, // pointer position on the canvas
    pending: [], // cells a stroke painted whose answer from Python has not arrived
  };
  S.off.width = S.nx;
  S.off.height = S.ny;
  const offCtx = S.off.getContext("2d");
  S.img = offCtx.createImageData(S.nx, S.ny);
  const ctx = ui.canvas.getContext("2d");

  function loadField() {
    S.depth = typed(model.get("depth"), Float32Array);
    S.mask = typed(model.get("mask"), Uint8Array);
  }
  function loadBasins() {
    S.basins = typed(model.get("basins"), Int32Array);
    S.nBasins = 0;
    for (let k = 0; k < S.basins.length; k++) if (S.basins[k] > S.nBasins) S.nBasins = S.basins[k];
  }
  function loadScale() {
    S.minDepth = model.get("min_depth");
    S.lo = S.minDepth;
    S.hi = Math.max(model.get("max_depth"), S.lo + 1);
  }

  /* ============================== image ============================== */

  function colorOf(d, m, b) {
    if (!m) return LAND;
    if (S.view === "mask") return OCEAN;
    if (S.view === "basinmask") return S.basins.length ? BASIN_COLORS[b % BASIN_COLORS.length] : OCEAN;
    const t = Math.min(Math.max((d - S.lo) / (S.hi - S.lo), 0), 1);
    return DEPTH_LUT[(Math.sqrt(t) * 255) | 0];
  }

  function fullImage() {
    const { nx, ny } = S;
    const px = S.img.data;
    for (let j = 0; j < ny; j++) {
      let p = (ny - 1 - j) * nx * 4;
      for (let i = 0, k = j * nx; i < nx; i++, k++, p += 4) {
        const c = colorOf(S.depth[k], S.mask[k], S.basins[k]);
        px[p] = c[0]; px[p + 1] = c[1]; px[p + 2] = c[2]; px[p + 3] = 255;
      }
    }
    offCtx.putImageData(S.img, 0, 0);
    S.dirty = null;
  }

  function setPixel(k, c) {
    const i = k % S.nx;
    const r = S.ny - 1 - ((k / S.nx) | 0);
    const p = (r * S.nx + i) * 4;
    const px = S.img.data;
    px[p] = c[0]; px[p + 1] = c[1]; px[p + 2] = c[2]; px[p + 3] = 255;
    const d = S.dirty;
    if (!d) S.dirty = [i, r, i, r];
    else [d[0], d[1], d[2], d[3]] = [Math.min(d[0], i), Math.min(d[1], r), Math.max(d[2], i), Math.max(d[3], r)];
  }
  const paintCell = (k) => setPixel(k, colorOf(S.depth[k], S.mask[k], S.basins[k]));

  function flush() {
    const d = S.dirty;
    if (!d) return;
    S.dirty = null;
    offCtx.putImageData(S.img, 0, 0, d[0], d[1], d[2] - d[0] + 1, d[3] - d[1] + 1);
  }

  // A brush stroke shows at once; Python's answer then repaints the cells for real.
  function previewColor(k, op, value) {
    if (op === "land") return LAND;
    if (op === "ocean") return colorOf(Math.max(S.depth[k], S.minDepth), 1, S.basins[k]);
    return colorOf(value, value > S.minDepth ? 1 : 0, S.basins[k]);
  }
  function repaintPending() {
    for (const k of S.pending) paintCell(k);
    S.pending = [];
  }

  /* ============================== view: pan, zoom, draw ============================== */

  const theme = (name, fallback) => getComputedStyle(el).getPropertyValue(name).trim() || fallback;

  function fit() {
    const [w, ht] = [ui.canvas.clientWidth, ui.canvas.clientHeight];
    const m = 2 * (OBC_GAP + 6); // room for the open-boundary lines outside the grid
    S.s = Math.min((w - m) / S.nx, (ht - m) / S.ny);
    S.ox = (w - S.nx * S.s) / 2;
    S.oy = (ht - S.ny * S.s) / 2;
    S.fitted = true;
  }

  function zoomAt(x, y, f) {
    const fitS = Math.min(ui.canvas.clientWidth / S.nx, ui.canvas.clientHeight / S.ny);
    const s2 = Math.min(Math.max(S.s * f, fitS * 0.5), 64);
    S.ox = x - ((x - S.ox) * s2) / S.s;
    S.oy = y - ((y - S.oy) * s2) / S.s;
    S.s = s2;
    requestDraw();
  }

  function draw() {
    const { dpr } = S;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = theme("--jp-layout-color2", "#eee");
    ctx.fillRect(0, 0, ui.canvas.width, ui.canvas.height);
    ctx.imageSmoothingEnabled = S.s < 1;
    ctx.setTransform(S.s * dpr, 0, 0, S.s * dpr, S.ox * dpr, S.oy * dpr);
    ctx.drawImage(S.off, 0, 0);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    drawOpenBoundaries();

    const box = S.drag ? boxFrom(S.drag.a, S.drag.b) : S.sel;
    if (box) {
      const x = S.ox + box.i0 * S.s;
      const y = S.oy + (S.ny - 1 - box.j1) * S.s;
      const w = (box.i1 - box.i0 + 1) * S.s;
      const ht = (box.j1 - box.j0 + 1) * S.s;
      ctx.setLineDash(S.drag ? [5, 4] : []);
      // A dark halo under the warn-coloured line keeps it visible on any colour.
      for (const [style, width] of [["rgba(0,0,0,.7)", 4], [theme("--jp-warn-color1", "#f57c00"), 2]]) {
        ctx.strokeStyle = style;
        ctx.lineWidth = width;
        ctx.strokeRect(x - 1, y - 1, w + 2, ht + 2);
        if (!S.drag && selectedCell() && S.s < 6) {
          ctx.beginPath(); // a ring, so a single cell stays visible when zoomed out
          ctx.arc(x + w / 2, y + ht / 2, 9, 0, 2 * Math.PI);
          ctx.stroke();
        }
      }
      ctx.setLineDash([]);
    }
    if (S.hover && S.tool !== "select" && !S.pan) {
      ctx.strokeStyle = theme("--jp-ui-font-color0", "#000");
      ctx.globalAlpha = 0.7;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(S.hover[0], S.hover[1], Math.max((S.radius + 0.5) * S.s, 3), 0, 2 * Math.PI);
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
  }

  /* ============================== open boundaries ============================== */

  // Screen endpoints [x0, y0, x1, y1] of the cells start..stop-1 along one side of the
  // grid, OBC_GAP pixels outside it. South is row j = 0 (the bottom of the image).
  function sideLine(side, start, stop) {
    const { s, ox, oy, nx, ny } = S;
    if (side === "south" || side === "north") {
      const y = side === "south" ? oy + ny * s + OBC_GAP : oy - OBC_GAP;
      return [ox + start * s, y, ox + stop * s, y];
    }
    const x = side === "west" ? ox - OBC_GAP : ox + nx * s + OBC_GAP;
    return [x, oy + (ny - start) * s, x, oy + (ny - stop) * s]; // j counts up the screen
  }

  // One thick line per open-water run, labelled with its side inside the grid if it fits.
  function drawOpenBoundaries() {
    const runs = model.get("open_boundary_runs") || {};
    const inward = { south: [0, -1], north: [0, 1], west: [1, 0], east: [-1, 0] };
    ctx.font = "600 11px system-ui, sans-serif";
    ctx.textBaseline = "middle";
    for (const [side, list] of Object.entries(runs)) {
      for (const { start, stop, tiny } of list) {
        const [x0, y0, x1, y1] = sideLine(side, start, stop);
        ctx.strokeStyle = tiny ? OBC_TINY_COLOR : OBC_COLOR;
        ctx.lineWidth = 5;
        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineTo(x1, y1);
        ctx.stroke();
        if (Math.hypot(x1 - x0, y1 - y0) < 60) continue; // no room for a label
        const [dx, dy] = inward[side];
        const [x, y] = [(x0 + x1) / 2 + dx * (OBC_GAP + 6), (y0 + y1) / 2 + dy * (OBC_GAP + 10)];
        ctx.textAlign = side === "west" ? "left" : side === "east" ? "right" : "center";
        ctx.lineWidth = 3; // a dark halo keeps the white label readable on any colour
        ctx.strokeStyle = "rgba(0,0,0,.7)";
        ctx.strokeText(side, x, y);
        ctx.fillStyle = "#fff";
        ctx.fillText(side, x, y);
      }
    }
  }

  // The checkbox and the summary under it follow Python's state.
  function showOpenBoundaries() {
    const runs = model.get("open_boundary_runs") || {};
    ui.obc.checked = model.get("show_open_boundaries");
    ui.obcInfo.hidden = !ui.obc.checked;
    const sides = Object.keys(runs).filter((side) => runs[side].length);
    const n = Object.values(runs).flat().filter((r) => r.tiny).length;
    // The same wording as corner_diagnostics.quality_messages.
    const warn = n ? ` Warning: ${n} small open boundar${n === 1 ? "y" : "ies"} along coastline (red).` : "";
    ui.obcInfo.textContent = `Open boundaries: ${sides.join(", ") || "none"}.${warn}`;
    requestDraw();
  }

  // All drawing happens here, at most once per animation frame.
  const frame = guard(() => {
    S.frame = 0;
    if (S.rebuild) {
      S.rebuild = false;
      fullImage();
    }
    flush();
    draw();
  });
  const requestDraw = () => {
    if (!S.frame) S.frame = requestAnimationFrame(frame);
  };
  const rebuild = () => {
    S.rebuild = true;
    requestDraw();
  };

  function resize() {
    const [w, ht] = [ui.canvas.clientWidth, ui.canvas.clientHeight];
    if (!w || !ht) return; // not on screen yet
    S.dpr = window.devicePixelRatio || 1;
    ui.canvas.width = Math.round(w * S.dpr);
    ui.canvas.height = Math.round(ht * S.dpr);
    if (!S.fitted) fit();
    cancelAnimationFrame(S.frame);
    frame(); // resizing cleared the canvas: draw now, not next frame
  }

  /* ============================== cells and selection ============================== */

  const local = (e) => {
    const r = ui.canvas.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  };
  // [i, j] of the cell under a canvas position (may be outside the grid).
  const cellAt = (x, y) => [Math.floor((x - S.ox) / S.s), S.ny - 1 - Math.floor((y - S.oy) / S.s)];
  const inGrid = (c) => c[0] >= 0 && c[1] >= 0 && c[0] < S.nx && c[1] < S.ny;

  // The box between two cells, clipped to the grid; null if it misses the grid.
  function boxFrom(a, b) {
    const i0 = Math.max(Math.min(a[0], b[0]), 0);
    const i1 = Math.min(Math.max(a[0], b[0]), S.nx - 1);
    const j0 = Math.max(Math.min(a[1], b[1]), 0);
    const j1 = Math.min(Math.max(a[1], b[1]), S.ny - 1);
    return i0 <= i1 && j0 <= j1 ? { i0, j0, i1, j1 } : null;
  }
  // The selected cell {i, j, k} when the selection is a single cell, else null.
  function selectedCell() {
    const b = S.sel;
    if (!b || b.i0 !== b.i1 || b.j0 !== b.j1) return null;
    return { i: b.i0, j: b.j0, k: b.j0 * S.nx + b.i0 };
  }

  function select(box) {
    S.sel = box;
    const cell = selectedCell();
    if (cell && S.mask[cell.k]) ui.depthValue.value = S.depth[cell.k].toFixed(1);
    updateControls();
    requestDraw();
  }

  function readout(c) {
    if (!c || !inGrid(c)) return "Move over the grid to read a cell.";
    const [i, j] = c;
    const k = j * S.nx + i;
    let text = `i ${i}  j ${j}   lon ${S.lon[k].toFixed(3)}°  lat ${S.lat[k].toFixed(3)}°   `;
    text += S.mask[k] ? `depth ${S.depth[k].toFixed(1)} m` : "land";
    if (S.view === "basinmask" && S.basins.length) text += `   basin ${S.basins[k]}`;
    return text;
  }

  /* ============================== brush ============================== */

  function stamp(st, ci, cj) {
    const r = S.radius;
    for (let dj = -r; dj <= r; dj++) {
      for (let di = -r; di <= r; di++) {
        const [i, j] = [ci + di, cj + dj];
        if (di * di + dj * dj > r * r + r || i < 0 || j < 0 || i >= S.nx || j >= S.ny) continue;
        const k = j * S.nx + i;
        if (st.seen.has(k)) continue;
        st.seen.add(k);
        st.cells.push(k);
        setPixel(k, previewColor(k, st.op, st.value));
      }
    }
  }

  // Stamp along the line from the last cell, so fast strokes leave no gaps.
  function paintTo(c) {
    const st = S.stroke;
    const last = st.last || c;
    const steps = Math.max(Math.abs(c[0] - last[0]), Math.abs(c[1] - last[1]), 1);
    for (let s = 0; s <= steps; s++) {
      const f = s / steps;
      stamp(st, Math.round(last[0] + (c[0] - last[0]) * f), Math.round(last[1] + (c[1] - last[1]) * f));
    }
    st.last = c;
  }

  // One stroke is one action: its cells go to Python as one buffer of (j, i) pairs.
  function endStroke() {
    const st = S.stroke;
    S.stroke = null;
    if (!st.cells.length) return;
    const ij = new Int32Array(st.cells.length * 2);
    st.cells.forEach((k, n) => {
      ij[2 * n] = (k / S.nx) | 0;
      ij[2 * n + 1] = k % S.nx;
    });
    for (const k of st.cells) S.pending.push(k);
    sendEdit(st.op === "depth" ? { op: "depth", value: st.value } : { op: st.op }, [ij.buffer]);
  }

  /* ============================== controls ============================== */

  function legend() {
    if (S.view === "depth") {
      const stops = DEPTH_STOPS.map(([t, c]) => `${rgb(c)} ${(t * 100).toFixed(1)}%`);
      const bar = h("div", { class: "mf-topo-bar" });
      bar.style.background = `linear-gradient(90deg, ${stops.join(", ")})`;
      // The ramp is square-root: bar position p is depth lo + p² (hi - lo).
      const ticks = [0, 0.25, 0.5, 0.75, 1].map((p) => {
        const d = (S.lo + p * p * (S.hi - S.lo)).toFixed(0);
        const span = h("span", { textContent: p === 1 ? `${d} m` : d });
        span.style.left = `${p * 100}%`;
        return span;
      });
      ui.legend.replaceChildren(bar, h("div", { class: "mf-topo-ticks" }, ...ticks));
    } else if (S.view === "mask") {
      const swatch = (c) => {
        const s = h("span", { class: "mf-topo-swatch" });
        s.style.background = rgb(c);
        return s;
      };
      ui.legend.replaceChildren(
        h("div", { class: "mf-topo-hint" }, swatch(LAND), "land", swatch(OCEAN), "ocean"),
      );
    } else {
      const n = S.nBasins;
      ui.legend.replaceChildren(
        hint(
          `${n} basin${n === 1 ? "" : "s"}. The largest has the highest label;` +
            ` colours repeat every ${BASIN_COLORS.length} labels.`,
        ),
      );
    }
  }

  function updateControls() {
    for (const [name, b] of Object.entries(ui.views)) b.setAttribute("aria-pressed", String(name === S.view));
    for (const [name, b] of Object.entries(ui.tools)) b.setAttribute("aria-pressed", String(name === S.tool));
    ui.radiusLabel.textContent = `${S.radius} cell${S.radius === 1 ? "" : "s"}`;

    const box = S.sel;
    const cell = selectedCell();
    if (!box) ui.selection.textContent = "Nothing selected. With Select, click a cell or drag a box.";
    else if (cell) ui.selection.textContent = `Cell i ${cell.i}, j ${cell.j}`;
    else {
      const n = (box.i1 - box.i0 + 1) * (box.j1 - box.j0 + 1);
      ui.selection.textContent = `Box i ${box.i0}–${box.i1}, j ${box.j0}–${box.j1}: ${n.toLocaleString()} cells`;
    }
    for (const b of [ui.setDepth, ui.setLand, ui.setOcean, ui.clearSel, ...Object.values(ui.fill)]) {
      b.disabled = !box;
    }
    ui.statsRow.hidden = !model.get("has_stats");
    ui.clearMask.disabled = !model.get("has_user_mask");

    ui.basinSection.hidden = S.view !== "basinmask";
    const ocean = cell && S.mask[cell.k];
    ui.eraseSel.disabled = ui.eraseDisc.disabled = !ocean;
    if (!cell) ui.basinInfo.textContent = "Select an ocean cell in the basin to keep or erase.";
    else if (!ocean) ui.basinInfo.textContent = "The selected cell is land. Select an ocean cell.";
    else ui.basinInfo.textContent = S.basins.length ? `Selected cell is in basin ${S.basins[cell.k]}.` : "";

    const history = model.get("has_history");
    ui.history.hidden = !history;
    ui.noHistory.hidden = history;
    ui.undo.disabled = !model.get("can_undo");
    ui.redo.disabled = !model.get("can_redo");
    const tags = model.get("tags") || [];
    ui.tags.textContent = tags.length ? `Tags: ${tags.join(", ")}` : "";
  }

  // Only on branch changes, so a choice made in the dropdown is not reset under the person.
  function updateBranches() {
    const current = model.get("current_branch");
    ui.branchSelect.replaceChildren(
      ...(model.get("branches") || []).map((b) =>
        h("option", { value: b, textContent: b === current ? `${b} (current)` : b }),
      ),
    );
    ui.branchSelect.value = current;
  }

  /* ============================== events: canvas ============================== */

  on(ui.canvas, "contextmenu", (e) => e.preventDefault());
  on(ui.canvas, "pointerdown", (e) => {
    ui.stage.focus({ preventScroll: true });
    ui.canvas.setPointerCapture(e.pointerId);
    const [x, y] = local(e);
    const c = cellAt(x, y);
    if (e.button !== 0 || S.space) {
      S.pan = { x, y, ox: S.ox, oy: S.oy };
      ui.stage.classList.add("mf-topo-panning");
    } else if (S.tool === "select") {
      S.drag = { a: c, b: c, x, y, moved: false };
    } else {
      const value = ui.depthValue.valueAsNumber;
      S.stroke = { op: S.tool, value, cells: [], seen: new Set(), last: null };
      paintTo(c);
    }
    requestDraw();
  });
  on(ui.canvas, "pointermove", (e) => {
    const [x, y] = local(e);
    S.hover = [x, y];
    ui.readout.textContent = readout(cellAt(x, y));
    if (S.pan) {
      S.ox = S.pan.ox + x - S.pan.x;
      S.oy = S.pan.oy + y - S.pan.y;
    } else if (S.drag) {
      S.drag.b = cellAt(x, y);
      if (Math.abs(x - S.drag.x) + Math.abs(y - S.drag.y) > 3) S.drag.moved = true;
    } else if (S.stroke) {
      paintTo(cellAt(x, y));
    }
    requestDraw();
  });
  const pointerUp = () => {
    if (S.pan) ui.stage.classList.remove("mf-topo-panning");
    S.pan = null;
    if (S.drag) {
      const d = S.drag;
      S.drag = null;
      select(boxFrom(d.a, d.moved ? d.b : d.a)); // a click selects one cell
    }
    if (S.stroke) endStroke();
    requestDraw();
  };
  on(ui.canvas, "pointerup", pointerUp);
  on(ui.canvas, "pointercancel", pointerUp);
  on(ui.canvas, "pointerleave", () => {
    S.hover = null;
    ui.readout.textContent = readout(null);
    requestDraw();
  });
  const onWheel = (e) => {
    e.preventDefault();
    zoomAt(...local(e), Math.exp(-e.deltaY * 0.0015));
  };
  on(ui.canvas, "wheel", onWheel, { passive: false });
  // Space-drag pans. The stage takes focus on click, so the keys reach it.
  on(ui.stage, "keydown", (e) => {
    if (e.code !== "Space") return;
    S.space = true;
    e.preventDefault();
  });
  on(ui.stage, "keyup", (e) => {
    if (e.code === "Space") S.space = false;
  });
  on(ui.stage, "blur", () => (S.space = false));
  const zoomCentre = (f) => zoomAt(ui.canvas.clientWidth / 2, ui.canvas.clientHeight / 2, f);
  on(ui.zoomIn, "click", () => zoomCentre(1.6));
  on(ui.zoomOut, "click", () => zoomCentre(1 / 1.6));
  on(ui.zoomFit, "click", () => {
    fit();
    requestDraw();
  });

  /* ============================== events: controls ============================== */

  for (const [view, b] of Object.entries(ui.views)) on(b, "click", () => send({ op: "view", view }));
  on(ui.obc, "change", () => send({ op: "open_boundaries", show: ui.obc.checked }));
  for (const [name, b] of Object.entries(ui.tools)) {
    on(b, "click", () => {
      S.tool = name;
      updateControls();
      requestDraw(); // the brush ring shows only for paint tools
    });
  }
  on(ui.radius, "input", () => {
    S.radius = Number(ui.radius.value);
    updateControls();
    requestDraw();
  });
  on(ui.clearSel, "click", () => select(null));

  const editSelection = (edit) => {
    const b = S.sel;
    sendEdit({ ...edit, rect: [b.j0, b.i0, b.j1, b.i1] });
  };
  on(ui.setDepth, "click", () => editSelection({ op: "depth", value: ui.depthValue.valueAsNumber }));
  on(ui.setLand, "click", () => editSelection({ op: "land" }));
  on(ui.setOcean, "click", () => editSelection({ op: "ocean" }));
  for (const [statistic, b] of Object.entries(ui.fill)) {
    on(b, "click", () => editSelection({ op: "fill_from_stats", statistic }));
  }
  on(ui.clearMask, "click", () => sendEdit({ op: "clear_mask" }));
  on(ui.setMinDepth, "click", () => sendEdit({ op: "min_depth", value: ui.minDepth.valueAsNumber }));

  const eraseBasin = (op) => {
    const c = selectedCell();
    sendEdit({ op, cell: [c.j, c.i] });
  };
  on(ui.eraseSel, "click", () => eraseBasin("erase_selected_basin"));
  on(ui.eraseDisc, "click", () => eraseBasin("erase_disconnected_basin"));

  on(ui.undo, "click", () => send({ op: "undo" }));
  on(ui.redo, "click", () => send({ op: "redo" }));
  on(ui.reset, "click", () => send({ op: "reset" }));
  on(ui.tag, "click", () => send({ op: "tag", name: ui.tagName.value }));
  on(ui.createBranch, "click", () => send({ op: "create_branch", name: ui.branchName.value }));
  on(ui.checkout, "click", () => send({ op: "checkout", name: ui.branchSelect.value }));

  /* ============================== events: model ============================== */

  listen("msg:custom", (msg, buffers) => {
    if (msg.kind === "patch" && msg.n) {
      const cells = typed(buffers[0], Int32Array);
      const depth = typed(buffers[1], Float32Array);
      const mask = typed(buffers[2], Uint8Array);
      for (let n = 0; n < msg.n; n++) {
        const k = cells[2 * n] * S.nx + cells[2 * n + 1];
        S.depth[k] = depth[n];
        S.mask[k] = mask[n];
        paintCell(k);
      }
    } else if (msg.kind === "reload") {
      loadField();
      S.rebuild = true;
    }
    repaintPending(); // stroke cells the patch left alone go back to their real colour
    updateControls();
    requestDraw();
  });
  listen("change:errors", () => {
    if (!(model.get("errors") || []).length) return;
    repaintPending(); // the action failed: drop the stroke's preview
    requestDraw();
  });
  listen("change:view", () => {
    S.view = model.get("view");
    legend();
    updateControls();
    rebuild();
  });
  listen("change:basins", () => {
    loadBasins();
    legend();
    updateControls();
    if (S.view === "basinmask") rebuild();
  });
  const onScale = () => {
    loadScale();
    ui.minDepth.value = String(S.minDepth);
    legend();
    rebuild();
  };
  listen("change:min_depth", onScale);
  listen("change:max_depth", onScale);
  for (const name of ["has_stats", "has_user_mask", "can_undo", "can_redo", "tags"]) {
    listen(`change:${name}`, updateControls);
  }
  listen("change:show_open_boundaries", showOpenBoundaries);
  listen("change:open_boundary_runs", showOpenBoundaries);
  listen("change:branches", updateBranches);
  listen("change:current_branch", updateBranches);
  const showStatus = () => (ui.status.textContent = model.get("status") || "");
  listen("change:status", showStatus);

  /* ============================== start ============================== */

  loadField();
  loadBasins();
  onScale();
  updateControls();
  updateBranches();
  showOpenBoundaries();
  showStatus();
  ui.readout.textContent = readout(null);
  const observer = new ResizeObserver(guard(resize));
  observer.observe(ui.stage);
  send({ op: "sync" }); // edits patched in since the traits were last sent are not in them

  return () => {
    for (const off of cleanups) off();
    observer.disconnect();
    cancelAnimationFrame(S.frame);
  };
}

export default defineWidget(render);
