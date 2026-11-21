// GridSketcher front end (Python side: grid_sketcher.py).
//
// Draws the outline and the grid preview on an OpenLayers map whose view is in the very
// projection Python solves the grid in (the proj4 trait), so an edge drawn straight on
// screen is straight in the solver's plane. Python owns the outline (lon/lat degrees) and
// every rule; this file projects to and from the view only to draw and to edit.
//
// Python → page (traits)
//   vertices      float64 (lon, lat) pairs of the outline, not closed; empty: draw anew
//   corners       corner vertex indices; with 4, corners[k] is labelled k + 1
//   flat_corners  corner vertices flatter than 170° (ringed red)
//   kind, proj4   the projection of the solve and of the map; recommended: the best kind
//   nodes         float64 (rows, cols, 2) lon/lat of the drawn preview grid nodes, with
//   node_shape    [rows, cols]; flags: uint8 (rows - 1, cols - 1), worst status of the
//                 cells in each block between drawn lines (0 ok, 1 warn, 2 bad)
//   land          GeoJSON (lon/lat): a "land" MultiPolygon and "graticule" lines
//   resolution_km, name, cells, summary, details, preview_errors, preview_warnings,
//   status, can_build, can_save, can_undo, can_redo, errors
// Page → Python (actions)
//   {op: "outline"} + float64 (lon, lat) buffer   a vertex drag or an edge insert ended
//   {op: "draw"} + float64 (lon, lat) buffer      a new outline: clicked in, or a Box
//   {op: "corner", i}   {op: "delete", i}   {op: "undo" | "redo" | "clear"}
//   {op: "projection", kind}   {op: "resolution", km}
//   {op: "build", name}   {op: "save", name, overwrite}
//   {op: "view"} + float64 (lon, lat) buffer      points round the map's edge after a move
//
// OpenLayers and proj4 come from jsDelivr as pinned classic scripts, loaded once per page.
// The full OpenLayers build is one script, so every part of it shares one projection
// registry (separate "+esm" modules would each carry their own copy).
// Every button and input has a data-action attribute; the map element has data-role=map
// and holds its OpenLayers map as .olMap, for debugging and the browser tests.

/* ============================== libraries ============================== */

const OL_JS = "https://cdn.jsdelivr.net/npm/ol@10.6.1/dist/ol.js";
const OL_CSS = "https://cdn.jsdelivr.net/npm/ol@10.6.1/ol.css";
const PROJ4_JS = "https://cdn.jsdelivr.net/npm/proj4@2.15.0/dist/proj4.js";

// Load a classic script once per page; the promise resolves once it has run.
// `global` names the window global a UMD build sets. On a page with require.js (the docs
// site's saved widgets), such a build registers with require.js instead of setting its
// global, so it is loaded through require.js and the global set here.
function sketchScript(url, global) {
  const cache = (window.__mfSketchScripts ||= {});
  cache[url] ||= new Promise((resolve, reject) => {
    const fail = () => {
      delete cache[url]; // a later render tries again
      reject(new Error(`Could not load ${url}: the map needs cdn.jsdelivr.net.`));
    };
    const amd = typeof window.define === "function" && window.define.amd;
    if (global && amd && typeof window.require === "function") {
      window.require([url], (lib) => resolve((window[global] = lib)), fail);
      return;
    }
    const tag = document.createElement("script");
    tag.src = url;
    tag.onload = resolve;
    tag.onerror = fail;
    document.head.appendChild(tag);
  });
  return cache[url];
}

// OpenLayers (window.ol) and proj4 (window.proj4), plus OpenLayers' stylesheet.
function sketchLibraries() {
  if (!document.querySelector(`link[href="${OL_CSS}"]`)) {
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = OL_CSS;
    document.head.appendChild(link);
  }
  // ol.js sets window.ol on any page; proj4.js is a UMD build.
  return Promise.all([sketchScript(OL_JS), sketchScript(PROJ4_JS, "proj4")]);
}

// The OpenLayers projection code for a proj4 string, registered on first use.
function sketchProjection(proj4String) {
  const codes = (window.__mfSketchCodes ||= new Map());
  if (!codes.has(proj4String)) {
    const code = `MF-SKETCH-${codes.size + 1}`;
    proj4.defs(code, proj4String);
    ol.proj.proj4.register(proj4);
    codes.set(proj4String, code);
  }
  return codes.get(proj4String);
}

/* ============================== look ============================== */

// Projection buttons: [kind, label, tooltip]
const SKETCH_KINDS = [
  ["lcc", "Lambert", "Lambert conformal conic"],
  ["merc", "Mercator", "Mercator"],
  ["tmerc", "Transverse", "Transverse Mercator"],
  ["stere", "Polar", "Polar stereographic"],
];

// The map keeps one palette in light and dark themes, like a printed map.
const SKETCH_COLORS = {
  water: "#dfe9f2",
  land: "#ece5d8",
  graticule: "rgba(60, 70, 90, 0.2)",
  outline: "#222",
  line: "rgba(112, 112, 112, 0.55)",
  flat: "#d62728",
  flag: { 1: "#e08214", 2: "#d7191c" }, // corner_diagnostics.STATUS_COLORS: warn, bad
};

const SKETCH_CSS = `
.mf-sketch { display: flex; flex-wrap: wrap; gap: 12px; align-items: flex-start;
  font: 13px/1.4 var(--jp-ui-font-family, system-ui, sans-serif); color: var(--jp-ui-font-color1, #222); }
.mf-sketch [hidden] { display: none !important; }
.mf-sketch-rail { flex: 0 0 300px; max-width: 100%; display: flex; flex-direction: column; gap: 10px; }
.mf-sketch-main { flex: 1 1 420px; min-width: 0; display: flex; flex-direction: column; gap: 6px; }
.mf-sketch-section h4 { margin: 0 0 4px; font-size: 11px; font-weight: 600; letter-spacing: .04em;
  text-transform: uppercase; color: var(--jp-ui-font-color2, #666); }
.mf-sketch-row { display: flex; flex-wrap: wrap; gap: 4px; align-items: center; margin-top: 4px; }
.mf-sketch button { font: inherit; padding: 3px 8px; border: 1px solid var(--jp-border-color1, #bbb);
  border-radius: 4px; background: var(--jp-layout-color1, #fff); color: inherit; cursor: pointer; }
.mf-sketch button:disabled { opacity: .45; cursor: default; }
.mf-sketch button[aria-pressed=true] { background: var(--jp-brand-color1, #1976d2);
  border-color: var(--jp-brand-color1, #1976d2); color: #fff; }
.mf-sketch button.mf-sketch-best { outline: 2px solid var(--jp-success-color1, #2e7d32); outline-offset: 1px; }
.mf-sketch button.mf-sketch-warn:not(:disabled) { border-color: var(--jp-warn-color1, #b26a00);
  box-shadow: inset 0 0 0 1px var(--jp-warn-color1, #b26a00); }
.mf-sketch input { font: inherit; min-width: 0; padding: 2px 4px; color: inherit;
  background: var(--jp-layout-color0, #fff); border: 1px solid var(--jp-border-color1, #bbb); border-radius: 4px; }
.mf-sketch input[type=number] { width: 90px; }
.mf-sketch input[type=text] { flex: 1 1 120px; }
.mf-sketch-hint { font-size: 12px; color: var(--jp-ui-font-color2, #666); }
.mf-sketch-list { margin: 0; padding-left: 18px; font-size: 12px; }
.mf-sketch-list li { margin-top: 2px; }
.mf-sketch-errors { color: var(--jp-error-color1, #b42318); }
.mf-sketch-warnings { color: var(--jp-warn-color1, #b26a00); }
.mf-sketch table { border-collapse: collapse; width: 100%; font-size: 12px; margin-top: 4px; }
.mf-sketch td { padding: 2px 4px; vertical-align: top; border-bottom: 1px solid var(--jp-border-color2, #e0e0e0); }
.mf-sketch summary { cursor: pointer; font-size: 12px; }
.mf-sketch-map { position: relative; height: clamp(360px, 62vh, 640px); overflow: hidden;
  border: 1px solid var(--jp-border-color1, #bbb); border-radius: 6px; background: ${SKETCH_COLORS.water}; }
.mf-sketch-loading { position: absolute; inset: 0; display: grid; place-items: center; color: #555; }
`;

/* ============================== small helpers ============================== */

// Copy a binary trait (DataView or ArrayBuffer) into an aligned typed array.
function sketchTyped(buf, Type) {
  if (!buf || !buf.byteLength) return new Type(0);
  const bytes = ArrayBuffer.isView(buf)
    ? new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength)
    : new Uint8Array(buf);
  return new Type(bytes.slice().buffer);
}

// sketchEl("button", {class, "data-action", textContent, ...}, ...children)
function sketchEl(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (key === "class") node.className = value;
    else if (key.startsWith("data-") || key.startsWith("aria-")) node.setAttribute(key, value);
    else node[key] = value;
  }
  node.append(...children);
  return node;
}

// The map's styles, made once OpenLayers has loaded.
function sketchStyles() {
  const { Style, Stroke, Fill, Circle } = ol.style;
  const stroke = (color, width) => new Stroke({ color, width });
  return {
    land: new Style({ fill: new Fill({ color: SKETCH_COLORS.land }) }),
    graticule: new Style({ stroke: stroke(SKETCH_COLORS.graticule, 0.8) }),
    line: new Style({ stroke: stroke(SKETCH_COLORS.line, 0.7) }),
    flag: {
      1: new Style({ stroke: stroke(SKETCH_COLORS.flag[1], 1.4), zIndex: 1 }),
      2: new Style({ stroke: stroke(SKETCH_COLORS.flag[2], 1.6), zIndex: 2 }),
    },
    outline: new Style({ stroke: stroke(SKETCH_COLORS.outline, 1.6) }),
    vertex: new Circle({ radius: 4, fill: new Fill({ color: "#fff" }), stroke: stroke("#333", 1.2) }),
    corner: new Circle({ radius: 8, fill: new Fill({ color: "#111" }) }),
    flat: new Circle({ radius: 11, stroke: stroke(SKETCH_COLORS.flat, 2.2) }),
    label: new Fill({ color: "#fff" }),
  };
}

/* ============================== render ============================== */

function render({ model, el, guard, errorBox }) {
  // ---------- bookkeeping: the returned cleanup removes every listener ----------
  const cleanups = [];
  const listen = (event, fn) => {
    const g = guard(fn);
    model.on(event, g);
    cleanups.push(() => model.off(event, g));
  };
  const on = (target, type, fn) => {
    const g = guard(fn);
    target.addEventListener(type, g);
    cleanups.push(() => target.removeEventListener(type, g));
  };
  const send = (msg, buffers) => model.send(msg, undefined, buffers);

  // ---------- DOM ----------
  const btn = (action, text, title = "") =>
    sketchEl("button", { type: "button", textContent: text, title, "data-action": action });
  const input = (action, type, props = {}) => sketchEl("input", { type, ...props, "data-action": action });
  const row = (...children) => sketchEl("div", { class: "mf-sketch-row" }, ...children);
  const hint = (text = "", role = "") =>
    sketchEl("div", { class: "mf-sketch-hint", textContent: text, ...(role && { "data-role": role }) });
  const section = (title, ...children) =>
    sketchEl("div", { class: "mf-sketch-section" }, sketchEl("h4", { textContent: title }), ...children);

  const ui = {
    undo: btn("undo", "Undo"),
    redo: btn("redo", "Redo"),
    clear: btn("clear", "Clear", "Remove the outline, to click in a new one"),
    box: btn("box", "Draw box", "Drag a box on the map: a 4-corner outline"),
    fit: btn("fit", "Zoom to outline"),
    kinds: Object.fromEntries(SKETCH_KINDS.map(([kind, label]) => [kind, btn(`projection-${kind}`, label)])),
    resolution: input("resolution", "number", { min: 0, step: "any", title: "Target cell size (km)" }),
    cells: sketchEl("span", { class: "mf-sketch-hint", "data-role": "cells" }),
    name: input("name", "text", { placeholder: "Grid name" }),
    build: btn("build", "Build grid", "Solve the full grid; it is then widget.grid"),
    save: btn("save", "Save", "Write GridLibrary/grid_<name>.nc"),
    overwrite: input("overwrite", "checkbox", { title: "Replace a saved grid of this name" }),
    status: hint("", "status"),
    errors: sketchEl("ul", { class: "mf-sketch-list mf-sketch-errors", "data-role": "preview-errors" }),
    warnings: sketchEl("ul", { class: "mf-sketch-list mf-sketch-warnings", "data-role": "preview-warnings" }),
    summary: sketchEl("div", { "data-role": "summary" }),
    details: sketchEl("table", { "data-role": "details" }),
    map: sketchEl("div", { class: "mf-sketch-map", "data-role": "map", tabIndex: 0 }),
    loading: sketchEl("div", { class: "mf-sketch-loading", textContent: "Loading the map…" }),
  };
  ui.errorSection = section("Errors", ui.errors);
  ui.warningSection = section("Warnings", ui.warnings);
  ui.preview = section(
    "Preview",
    ui.summary,
    sketchEl("details", {}, sketchEl("summary", { textContent: "Details" }), ui.details),
  );
  ui.map.append(ui.loading);

  const rail = sketchEl(
    "div", { class: "mf-sketch-rail" },
    section("Outline", row(ui.undo, ui.redo, ui.clear), row(ui.box, ui.fit)),
    section(
      "Projection",
      row(...Object.values(ui.kinds)),
      hint("Green outline: the projection whose cells vary least in size."),
    ),
    section("Resolution", row(ui.resolution, sketchEl("span", { textContent: "km" }), ui.cells)),
    section(
      "Build",
      row(ui.name),
      row(ui.build, ui.save, sketchEl("label", { class: "mf-sketch-hint" }, ui.overwrite, " Overwrite")),
      ui.status,
    ),
    ui.errorSection,
    ui.warningSection,
    ui.preview,
  );
  const main = sketchEl(
    "div", { class: "mf-sketch-main" },
    ui.map,
    hint(
      "Drag a vertex to move it, or an edge to add a vertex. Right-click (or Alt-click) a " +
        "vertex to delete it; double-click one to make or unmake a corner (4 needed). " +
        "After Clear, click points, then the first one to close. Draw box: drag a box. " +
        "Scroll to zoom; drag the map to pan.",
    ),
  );
  const root = sketchEl("div", { class: "mf-sketch" }, rail, main);
  // Keep JupyterLab's notebook shortcuts out of the widget.
  root.setAttribute("data-lm-suppress-shortcuts", "true");
  root.setAttribute("data-p-suppress-shortcuts", "true");
  el.replaceChildren(sketchEl("style", { textContent: SKETCH_CSS }), errorBox(), root);

  // ---------- state ----------
  const S = {
    code: null, // OpenLayers code of the view's projection
    n: 0, // outline vertices, as Python last sent them
    corners: [],
    flat: [],
    boxMode: false, // Draw box is on
    sketching: false, // a new outline is being clicked in
    fitted: false, // the view has been fitted to the outline
  };
  let map = null; // the OpenLayers map, once the libraries have loaded
  let layers, outline, tools, styles;
  let disposed = false;

  /* ============================== panel ============================== */

  const fillList = (list, items) => {
    list.replaceChildren(...items.map((text) => sketchEl("li", { textContent: text })));
    return items.length > 0;
  };

  const drawPanel = () => {
    const kind = model.get("kind");
    const best = model.get("recommended");
    for (const [k, label, title] of SKETCH_KINDS) {
      const b = ui.kinds[k];
      b.setAttribute("aria-pressed", String(k === kind));
      b.classList.toggle("mf-sketch-best", k === best);
      b.title = title + (k === best ? " (recommended: its cells vary least in size)" : "");
    }
    ui.box.setAttribute("aria-pressed", String(S.boxMode));
    ui.undo.disabled = !model.get("can_undo") && !S.sketching;
    ui.redo.disabled = !model.get("can_redo");
    ui.clear.disabled = S.n === 0 && !S.sketching;
    ui.build.disabled = !model.get("can_build");
    ui.save.disabled = !model.get("can_save");
    ui.cells.textContent = model.get("cells");
    ui.status.textContent = model.get("status");
    ui.errorSection.hidden = !fillList(ui.errors, model.get("preview_errors") || []);
    const warned = fillList(ui.warnings, model.get("preview_warnings") || []);
    ui.warningSection.hidden = !warned;
    ui.build.classList.toggle("mf-sketch-warn", warned);
    ui.summary.textContent = model.get("summary");
    const rows = model.get("details") || [];
    ui.details.replaceChildren(
      ...rows.map(([label, value]) =>
        sketchEl("tr", {}, sketchEl("td", { textContent: label }), sketchEl("td", { textContent: value })),
      ),
    );
    ui.preview.hidden = !model.get("summary");
  };

  // Inputs are set from Python only when their trait changes, or (resolution) when Python
  // rejected a value: a rejected edit changes no trait, but it sets errors.
  const showResolution = () => (ui.resolution.value = model.get("resolution_km"));
  const showName = () => (ui.name.value = model.get("name"));

  /* ============================== map: drawing ============================== */

  // Flat (lon, lat) pairs to view coordinates.
  function project(pairs) {
    const out = [];
    for (let k = 0; k + 1 < pairs.length; k += 2) out.push(ol.proj.fromLonLat([pairs[k], pairs[k + 1]], S.code));
    return out;
  }

  // Put the view in the projection of the proj4 trait, keeping the same place and true
  // scale in the middle of the map. Returns whether the projection changed.
  function syncView() {
    const code = sketchProjection(model.get("proj4"));
    if (code === S.code) return false;
    const old = map.getView();
    const oldCode = S.code;
    S.code = code;
    let center = [NaN, NaN];
    let resolution = NaN;
    if (oldCode && old.getCenter()) {
      center = ol.proj.fromLonLat(ol.proj.toLonLat(old.getCenter(), oldCode), code);
      const metres = ol.proj.getPointResolution(oldCode, old.getResolution(), old.getCenter());
      resolution = metres / ol.proj.getPointResolution(code, 1, center);
    }
    if (!center.every(Number.isFinite) || !(resolution > 0)) {
      [center, resolution, S.fitted] = [[0, 0], 1e4, false]; // fit the outline instead
    }
    map.setView(new ol.View({ projection: code, center, resolution }));
    return true;
  }

  function drawLand() {
    const source = layers.land.getSource();
    source.clear();
    const text = model.get("land");
    if (!text) return;
    const format = new ol.format.GeoJSON();
    source.addFeatures(format.readFeatures(text, { dataProjection: "EPSG:4326", featureProjection: S.code }));
  }

  function drawOutline() {
    const pairs = sketchTyped(model.get("vertices"), Float64Array);
    S.n = pairs.length / 2;
    S.corners = model.get("corners") || [];
    S.flat = model.get("flat_corners") || [];
    const ring = project(pairs);
    outline.setGeometry(S.n ? new ol.geom.Polygon([[...ring, ring[0]]]) : undefined);
    updateModes();
    drawPanel();
  }

  // Grid lines through the drawn nodes (the boundary is the outline's), and flagged
  // blocks outlined in the warn or bad colour.
  function drawGrid() {
    const source = layers.grid.getSource();
    source.clear();
    const [rows, cols] = model.get("node_shape") || [0, 0];
    if (!rows || !cols) return;
    const nodes = project(sketchTyped(model.get("nodes"), Float64Array));
    const at = (r, c) => nodes[r * cols + c];
    const lines = [];
    for (let r = 1; r < rows - 1; r++) lines.push(Array.from({ length: cols }, (_, c) => at(r, c)));
    for (let c = 1; c < cols - 1; c++) lines.push(Array.from({ length: rows }, (_, r) => at(r, c)));
    source.addFeature(new ol.Feature({ geometry: new ol.geom.MultiLineString(lines), status: 0 }));
    const flags = sketchTyped(model.get("flags"), Uint8Array);
    const blocks = { 1: [], 2: [] };
    for (let r = 0; r < rows - 1; r++) {
      for (let c = 0; c < cols - 1; c++) {
        const s = flags[r * (cols - 1) + c];
        if (s) blocks[s].push([[at(r, c), at(r, c + 1), at(r + 1, c + 1), at(r + 1, c), at(r, c)]]);
      }
    }
    for (const s of [1, 2]) {
      if (blocks[s].length) source.addFeature(new ol.Feature({ geometry: new ol.geom.MultiPolygon(blocks[s]), status: s }));
    }
  }

  // The outline, its vertices, its corners (numbered once there are 4) and red rings
  // round nearly flat corners. Drawn from the feature's own geometry, so the markers
  // follow a vertex while it is dragged.
  function outlineStyle(feature) {
    const geometry = feature.getGeometry();
    if (!geometry) return null;
    const ring = geometry.getCoordinates()[0].slice(0, -1);
    const out = [styles.outline, new ol.style.Style({ geometry: new ol.geom.MultiPoint(ring), image: styles.vertex })];
    if (ring.length !== S.n) return out; // mid-insert the indices shift: corners return after
    const numbered = S.corners.length === 4;
    S.corners.forEach((i, k) => {
      const at = new ol.geom.Point(ring[i]);
      if (S.flat.includes(i)) out.push(new ol.style.Style({ geometry: at, image: styles.flat }));
      const text = numbered
        ? new ol.style.Text({ text: String(k + 1), fill: styles.label, font: "bold 11px sans-serif" })
        : undefined;
      out.push(new ol.style.Style({ geometry: at, image: styles.corner, text }));
    });
    return out;
  }

  // Redraw what changed, once per batch of trait changes (Python sends one batch per action).
  const dirty = new Set();
  const flush = guard(() => {
    const parts = new Set(dirty);
    dirty.clear();
    if (!map) return; // startMap draws everything once the map exists
    if (syncView()) ["land", "outline", "grid"].forEach((p) => parts.add(p));
    if (parts.has("land")) drawLand();
    if (parts.has("outline")) drawOutline();
    if (parts.has("grid")) drawGrid();
    if (!S.fitted) fitOutline();
  });
  const mark = (part) => {
    if (!dirty.size) queueMicrotask(flush);
    dirty.add(part);
  };

  /* ============================== map: view ============================== */

  const hasSize = () => {
    const size = map.getSize();
    return Boolean(size && size[0] && size[1]);
  };

  // Zoom to the outline, once the map has a size (a hidden tab has none).
  function fitOutline() {
    if (!hasSize()) return;
    const geometry = outline.getGeometry();
    if (geometry) map.getView().fit(geometry.getExtent(), { padding: [40, 40, 40, 40] });
    S.fitted = true;
  }

  // Tell Python where the map now looks: lon/lat of 8 points round its edge, for land.
  function sendView() {
    if (!hasSize()) return;
    const [x0, y0, x1, y1] = map.getView().calculateExtent(map.getSize());
    const [xm, ym] = [(x0 + x1) / 2, (y0 + y1) / 2];
    const edge = [[x0, y0], [xm, y0], [x1, y0], [x1, ym], [x1, y1], [xm, y1], [x0, y1], [x0, ym]];
    const pairs = new Float64Array(edge.flatMap((xy) => ol.proj.toLonLat(xy, S.code)));
    send({ op: "view" }, [pairs.buffer]);
  }

  /* ============================== map: editing ============================== */

  // Send a ring of view coordinates (closed: last = first) as float64 lon/lat pairs.
  function sendRing(op, ring) {
    const pairs = new Float64Array(2 * (ring.length - 1));
    for (let k = 0; k < ring.length - 1; k++) pairs.set(ol.proj.toLonLat(ring[k], S.code), 2 * k);
    send({ op }, [pairs.buffer]);
  }

  // The outline vertex nearest a pixel, within 8 px; -1 if none.
  function vertexAt(pixel) {
    const geometry = outline.getGeometry();
    if (!geometry) return -1;
    let best = -1;
    let bestDistance = 8;
    geometry.getCoordinates()[0].slice(0, -1).forEach((xy, i) => {
      const p = map.getPixelFromCoordinate(xy);
      const d = Math.hypot(p[0] - pixel[0], p[1] - pixel[1]);
      if (d <= bestDistance) [best, bestDistance] = [i, d];
    });
    return best;
  }

  const deleteAt = (pixel) => {
    const i = vertexAt(pixel);
    if (i >= 0) send({ op: "delete", i });
  };

  // Which tool the mouse drives: Draw box, clicking a new outline in, or editing it.
  function updateModes() {
    if (!map) return;
    tools.box.setActive(S.boxMode);
    tools.draw.setActive(!S.boxMode && S.n === 0);
    tools.modify.setActive(!S.boxMode && S.n > 0);
    ui.map.style.cursor = S.boxMode || S.n === 0 ? "crosshair" : "";
  }

  function startMap() {
    if (disposed) return;
    ui.loading.remove();
    styles = sketchStyles();
    const vector = (style) => new ol.layer.Vector({ source: new ol.source.Vector(), style });
    layers = {
      land: vector((f) => (f.get("kind") === "land" ? styles.land : styles.graticule)),
      grid: vector((f) => (f.get("status") ? styles.flag[f.get("status")] : styles.line)),
      outline: vector(outlineStyle),
    };
    outline = new ol.Feature();
    layers.outline.getSource().addFeature(outline);
    map = new ol.Map({
      target: ui.map,
      layers: [layers.land, layers.grid, layers.outline],
      controls: ol.control.defaults.defaults({ attribution: false, rotate: false }),
      // No double-click zoom: a double-click on a vertex makes or unmakes a corner.
      interactions: ol.interaction.defaults.defaults({ doubleClickZoom: false }),
    });
    ui.map.olMap = map;

    const { condition } = ol.events;
    tools = {
      // Drag a vertex, or an edge to insert one. Deleting is ours (right/Alt-click).
      modify: new ol.interaction.Modify({
        source: layers.outline.getSource(),
        condition: (e) => condition.primaryAction(e) && condition.noModifierKeys(e),
        deleteCondition: condition.never,
      }),
      // A new outline: click points, then the first one (or double-click) to close.
      draw: new ol.interaction.Draw({ type: "Polygon" }),
      // A box in the view's projection: press, drag, release.
      box: new ol.interaction.Draw({
        type: "Circle",
        geometryFunction: ol.interaction.Draw.createBox(),
        freehandCondition: condition.always,
      }),
    };
    Object.values(tools).forEach((tool) => map.addInteraction(tool));
    tools.modify.on("modifyend", guard(() => sendRing("outline", outline.getGeometry().getCoordinates()[0])));
    tools.draw.on("drawstart", guard(() => ((S.sketching = true), drawPanel())));
    tools.draw.on("drawabort", guard(() => ((S.sketching = false), drawPanel())));
    tools.draw.on("drawend", guard((e) => {
      S.sketching = false;
      sendRing("draw", e.feature.getGeometry().getCoordinates()[0]);
    }));
    tools.box.on("drawend", guard((e) => {
      S.boxMode = false;
      updateModes();
      drawPanel();
      sendRing("draw", e.feature.getGeometry().getCoordinates()[0]);
    }));
    map.on("dblclick", guard((e) => {
      const i = vertexAt(e.pixel);
      if (i >= 0) send({ op: "corner", i });
    }));
    map.on("click", guard((e) => e.originalEvent.altKey && deleteAt(e.pixel)));
    map.on("moveend", guard(sendView));

    syncView();
    drawLand();
    drawOutline();
    drawGrid();
    map.updateSize();
    fitOutline();
  }

  /* ============================== actions ============================== */

  on(ui.undo, "click", () => (S.sketching ? tools.draw.removeLastPoint() : send({ op: "undo" })));
  on(ui.redo, "click", () => send({ op: "redo" }));
  on(ui.clear, "click", () => {
    if (S.sketching) tools.draw.abortDrawing();
    else send({ op: "clear" });
  });
  on(ui.box, "click", () => {
    S.boxMode = !S.boxMode;
    updateModes();
    drawPanel();
  });
  on(ui.fit, "click", () => map && fitOutline());
  for (const [kind, button] of Object.entries(ui.kinds)) {
    on(button, "click", () => kind !== model.get("kind") && send({ op: "projection", kind }));
  }
  on(ui.resolution, "change", () => send({ op: "resolution", km: Number(ui.resolution.value) }));
  on(ui.build, "click", () => send({ op: "build", name: ui.name.value }));
  on(ui.save, "click", () => send({ op: "save", name: ui.name.value, overwrite: ui.overwrite.checked }));
  on(ui.map, "contextmenu", (e) => {
    if (!map) return;
    e.preventDefault(); // a right-click deletes a vertex, not opens the browser's menu
    deleteAt(map.getEventPixel(e));
  });

  /* ============================== listen to Python ============================== */

  const panelTraits = ["kind", "recommended", "cells", "summary", "details", "preview_errors"];
  panelTraits.push("preview_warnings", "status", "can_build", "can_save", "can_undo", "can_redo");
  for (const name of panelTraits) listen(`change:${name}`, drawPanel);
  listen("change:proj4", () => mark("view"));
  for (const name of ["vertices", "corners", "flat_corners"]) listen(`change:${name}`, () => mark("outline"));
  for (const name of ["nodes", "node_shape", "flags"]) listen(`change:${name}`, () => mark("grid"));
  listen("change:land", () => mark("land"));
  listen("change:resolution_km", showResolution);
  listen("change:errors", showResolution);
  listen("change:name", showName);

  /* ============================== start ============================== */

  S.n = sketchTyped(model.get("vertices"), Float64Array).length / 2;
  drawPanel();
  showResolution();
  showName();
  const observer = new ResizeObserver(guard(() => {
    if (!map) return;
    map.updateSize();
    if (!S.fitted) fitOutline();
  }));
  observer.observe(ui.map);
  sketchLibraries().then(
    guard(startMap),
    guard((err) => {
      ui.loading.textContent = "The map could not load.";
      throw err;
    }),
  );

  return () => {
    disposed = true;
    cleanups.forEach((off) => off());
    observer.disconnect();
    if (map) map.dispose();
  };
}

export default defineWidget(render);
