// VGridCreator front end (Python side: vgrid_creator.py).
//
// Python owns every rule. This file draws the traits and sends one action per click or edit.
//   Traits read:   dz (float64 bytes, top layer first), name, kind, nk, depth, ratio,
//                  source, warning, library, errors
//   Actions sent:  params {kind, nk, depth, ratio}    reset {}
//                  save {name, message, overwrite}    load {file}
//   Message in:    {confirm_overwrite: name}  ->  ask before replacing a saved grid
// Every button and input has a data-action attribute, used by the browser contract test.

// ------------------------------------------------------------------------------ styles
const VGRID_CSS = `
.mf-vgrid { display: flex; flex-wrap: wrap; gap: 12px 20px; padding: 8px;
  font: 13px/1.4 var(--jp-ui-font-family, system-ui, sans-serif);
  color: var(--jp-ui-font-color1, #222); background: var(--jp-layout-color0, #fff); }
.mf-vgrid [hidden] { display: none !important; }
.mf-vgrid > .mf-errors, .mf-vgrid-library { flex: 1 1 100%; }
.mf-vgrid-panel { flex: 1 1 260px; max-width: 360px; display: flex; flex-direction: column; gap: 8px; }
.mf-vgrid-plot { flex: 1 1 300px; max-width: 440px; }
.mf-vgrid h3 { margin: 0; font-size: 15px; }
.mf-vgrid h4 { margin: 0 0 6px; font-size: 13px; }
.mf-vgrid label { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
.mf-vgrid input { font: inherit; color: inherit; background: var(--jp-layout-color1, #fff);
  border: 1px solid var(--jp-border-color1, #bbb); border-radius: 4px; padding: 3px 6px; }
.mf-vgrid input[type=number] { width: 110px; }
.mf-vgrid button { font: inherit; color: inherit; cursor: pointer; padding: 3px 10px;
  background: var(--jp-layout-color1, #f3f3f3); border: 1px solid var(--jp-border-color1, #bbb);
  border-radius: 4px; }
.mf-vgrid button.mf-vgrid-on { color: #fff; background: var(--jp-brand-color1, #1976d2);
  border-color: var(--jp-brand-color1, #1976d2); }
.mf-vgrid-toggle { display: flex; }
.mf-vgrid-toggle button { flex: 1; }
.mf-vgrid-row { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin-bottom: 6px; }
.mf-vgrid-help, .mf-vgrid-caption { font-size: 12px; color: var(--jp-ui-font-color2, #666); }
.mf-vgrid-note { font-size: 12px; padding: 4px 8px; border-left: 3px solid var(--jp-brand-color1, #1976d2); }
.mf-vgrid-warning { padding: 6px 8px; border-radius: 6px; color: var(--jp-warn-color1, #b26a00);
  border: 1px solid var(--jp-warn-color1, #b26a00); }
.mf-vgrid table { border-collapse: collapse; width: 100%; font-size: 12px; }
.mf-vgrid th, .mf-vgrid td { text-align: left; vertical-align: top; padding: 3px 6px;
  border-bottom: 1px solid var(--jp-border-color1, #ddd); }
.mf-vgrid svg { width: 100%; height: auto; display: block; }
.mf-vgrid-layer { fill: var(--jp-brand-color1, #1976d2); fill-opacity: 0.25;
  stroke: var(--jp-brand-color1, #1976d2); stroke-width: 0.8; }
.mf-vgrid-centre { fill: var(--jp-brand-color1, #1976d2); }
.mf-vgrid-axis { stroke: var(--jp-ui-font-color2, #666); stroke-width: 1; }
.mf-vgrid-label { fill: var(--jp-ui-font-color1, #333); font-size: 11px;
  font-family: var(--jp-ui-font-family, system-ui, sans-serif); }
`;

// ------------------------------------------------------------------------------ helpers
const VGRID_SVG_TAGS = ["svg", "rect", "circle", "line", "text"];

// Make an element (SVG or HTML) with attributes and optional text, appended to parent.
function vgridEl(parent, tag, attrs = {}, text = "") {
  const node = VGRID_SVG_TAGS.includes(tag)
    ? document.createElementNS("http://www.w3.org/2000/svg", tag)
    : document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  if (text !== "") node.textContent = text;
  if (parent) parent.appendChild(node);
  return node;
}

// A bytes trait (DataView, typed array or ArrayBuffer) as a Float64Array (copied, so aligned).
function vgridFloats(value) {
  if (!value || !value.byteLength) return new Float64Array(0);
  const bytes =
    value instanceof ArrayBuffer
      ? new Uint8Array(value)
      : new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
  return new Float64Array(bytes.slice().buffer);
}

// A number for display: at most two decimals, no float noise.
function vgridFmt(value) {
  return String(Math.round(value * 100) / 100);
}

// About five round tick values (steps of 1, 2 or 5 x 10^n) from 0 up to max.
function vgridTicks(max) {
  if (!(max > 0)) return [0];
  const raw = max / 5;
  const power = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * power).find((s) => s >= raw);
  const ticks = [];
  for (let i = 0; i * step <= max * (1 + 1e-9); i++) ticks.push(i * step);
  return ticks;
}

// ------------------------------------------------------------------------------ plot
// Depth runs down the y axis. Each layer is a box from its top interface to its bottom
// interface, as wide as its thickness; a dot marks the layer centre.
function vgridPlot(svg, dz) {
  const W = 420, H = 420, L = 60, R = 14, T = 12, B = 42;
  svg.replaceChildren();
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  if (dz.length === 0) return;
  const zi = [0];
  for (const d of dz) zi.push(zi[zi.length - 1] + d);
  const depth = zi[zi.length - 1];
  const xmax = Math.max(...dz) * 1.1;
  const x = (v) => L + (v / xmax) * (W - L - R);
  const y = (v) => T + (v / depth) * (H - T - B);

  for (let k = 0; k < dz.length; k++) {
    const top = y(zi[k]), bottom = y(zi[k + 1]);
    vgridEl(svg, "rect", { class: "mf-vgrid-layer", x: L, y: top, width: x(dz[k]) - L, height: bottom - top });
    vgridEl(svg, "circle", { class: "mf-vgrid-centre", cx: x(dz[k]), cy: (top + bottom) / 2, r: 1.8 });
  }

  // Axes, ticks and titles.
  vgridEl(svg, "line", { class: "mf-vgrid-axis", x1: L, y1: T, x2: L, y2: H - B });
  vgridEl(svg, "line", { class: "mf-vgrid-axis", x1: L, y1: H - B, x2: W - R, y2: H - B });
  for (const t of vgridTicks(depth)) {
    vgridEl(svg, "line", { class: "mf-vgrid-axis", x1: L - 4, y1: y(t), x2: L, y2: y(t) });
    vgridEl(svg, "text", { class: "mf-vgrid-label", x: L - 7, y: y(t) + 4, "text-anchor": "end" }, vgridFmt(t));
  }
  for (const t of vgridTicks(xmax)) {
    vgridEl(svg, "line", { class: "mf-vgrid-axis", x1: x(t), y1: H - B, x2: x(t), y2: H - B + 4 });
    vgridEl(svg, "text", { class: "mf-vgrid-label", x: x(t), y: H - B + 16, "text-anchor": "middle" }, vgridFmt(t));
  }
  const midY = (T + H - B) / 2;
  vgridEl(svg, "text", { class: "mf-vgrid-label", x: 12, y: midY, "text-anchor": "middle",
    transform: `rotate(-90 12 ${midY})` }, "Depth (m)");
  vgridEl(svg, "text", { class: "mf-vgrid-label", x: (L + W - R) / 2, y: H - 6, "text-anchor": "middle" },
    "Layer thickness (m)");
}

// ------------------------------------------------------------------------------ render
function render({ model, el, guard, errorBox }) {
  const listeners = [];
  const on = (event, fn) => {
    const handler = guard(fn);
    model.on(event, handler);
    listeners.push([event, handler]);
  };
  const click = (node, fn) => node.addEventListener("click", guard(fn));

  // --- Layout: errors on top, controls and plot side by side, library below.
  const root = vgridEl(null, "div", { class: "mf-vgrid" });
  vgridEl(root, "style", {}, VGRID_CSS);
  root.appendChild(errorBox());

  const panel = vgridEl(root, "div", { class: "mf-vgrid-panel" });
  const title = vgridEl(panel, "h3");
  const toggle = vgridEl(panel, "div", { class: "mf-vgrid-toggle" });
  const kindButtons = {
    uniform: vgridEl(toggle, "button", { "data-action": "kind-uniform" }, "Uniform"),
    hyperbolic: vgridEl(toggle, "button", { "data-action": "kind-hyperbolic" }, "Hyperbolic"),
  };
  const numberInput = (text, action, step) =>
    vgridEl(vgridEl(panel, "label", {}, text), "input", { type: "number", step, "data-action": action });
  const nkInput = numberInput("Levels", "nk", "1");
  const depthInput = numberInput("Depth (m)", "depth", "any");
  const ratioInput = numberInput("Bottom/top thickness ratio", "ratio", "any");
  const ratioHelp = vgridEl(panel, "div", { class: "mf-vgrid-help" },
    "Bottom layer thickness divided by top layer thickness. Above 1 gives thin layers " +
    "near the surface and thick ones at depth; below 1 the reverse; 1 is uniform.");
  const note = vgridEl(panel, "div", { class: "mf-vgrid-note" });
  const warning = vgridEl(panel, "div", { class: "mf-vgrid-warning" });
  const resetButton = vgridEl(panel, "button", { "data-action": "reset" }, "Reset to the starting grid");

  const plotBox = vgridEl(root, "div", { class: "mf-vgrid-plot" });
  const caption = vgridEl(plotBox, "div", { class: "mf-vgrid-caption" });
  const svg = vgridEl(plotBox, "svg", { role: "img", "aria-label": "Layer thickness against depth" });

  const library = vgridEl(root, "div", { class: "mf-vgrid-library" });
  vgridEl(library, "h4", {}, "Library (VGridLibrary folder)");
  const saveRow = vgridEl(library, "div", { class: "mf-vgrid-row" });
  const nameInput = vgridEl(saveRow, "input", { type: "text", placeholder: "Name", "data-action": "save-name" });
  const messageInput = vgridEl(saveRow, "input", { type: "text", placeholder: "Message", "data-action": "save-message" });
  const saveButton = vgridEl(saveRow, "button", { "data-action": "save" }, "Save");
  const confirmRow = vgridEl(library, "div", { class: "mf-vgrid-row mf-vgrid-warning", hidden: "" });
  const confirmText = vgridEl(confirmRow, "span");
  const overwriteButton = vgridEl(confirmRow, "button", { "data-action": "save-overwrite" }, "Replace it");
  const cancelButton = vgridEl(confirmRow, "button", { "data-action": "save-cancel" }, "Cancel");
  const table = vgridEl(library, "table");
  el.appendChild(root);

  // --- Draw state from the traits.
  const drawControls = () => {
    const kind = model.get("kind");
    for (const [name, button] of Object.entries(kindButtons)) {
      button.classList.toggle("mf-vgrid-on", name === kind);
      button.setAttribute("aria-pressed", String(name === kind));
    }
    nkInput.value = model.get("nk");
    depthInput.value = model.get("depth");
    ratioInput.value = model.get("ratio");
    ratioInput.parentElement.hidden = ratioHelp.hidden = kind !== "hyperbolic";
  };

  const drawGrid = () => {
    const dz = vgridFloats(model.get("dz"));
    const name = model.get("name");
    title.textContent = name ? `Vertical grid: ${name}` : "Vertical grid (unnamed)";
    caption.textContent = dz.length
      ? `${dz.length} layers, ${vgridFmt(dz.reduce((a, b) => a + b, 0))} m deep. ` +
        `Top layer ${vgridFmt(dz[0])} m, bottom layer ${vgridFmt(dz[dz.length - 1])} m.`
      : "";
    vgridPlot(svg, dz);
    const source = model.get("source");
    note.hidden = !source;
    note.textContent = source
      ? `${source}. The controls show the nearest uniform or hyperbolic settings; ` +
        "changing one replaces this grid with one built from the controls."
      : "";
    warning.hidden = !model.get("warning");
    warning.textContent = model.get("warning");
  };

  const drawLibrary = () => {
    const rows = model.get("library") || [];
    table.replaceChildren();
    if (rows.length === 0) {
      vgridEl(vgridEl(table, "tr"), "td", { class: "mf-vgrid-help" }, "No saved grids yet.");
      return;
    }
    const head = vgridEl(table, "tr");
    for (const text of ["Name", "Levels", "Depth (m)", "Saved", "Message", ""]) vgridEl(head, "th", {}, text);
    for (const row of rows) {
      const tr = vgridEl(table, "tr");
      const depth = row.depth == null ? "" : vgridFmt(row.depth);
      for (const text of [row.name, row.nk ?? "", depth, row.date, row.message]) vgridEl(tr, "td", {}, String(text));
      vgridEl(vgridEl(tr, "td"), "button", { "data-action": "load", "data-file": row.file }, "Load");
    }
  };

  // --- Actions: one message each.
  let pendingSave = null; // the save Python asked to confirm, resent with overwrite: true
  const hideConfirm = () => {
    pendingSave = null;
    confirmRow.hidden = true;
  };
  const sendParams = (kind) => {
    hideConfirm();
    const number = (input) => Number(input.value);
    model.send({ op: "params", kind, nk: number(nkInput), depth: number(depthInput), ratio: number(ratioInput) });
  };

  for (const [kind, button] of Object.entries(kindButtons)) {
    click(button, () => kind !== model.get("kind") && sendParams(kind));
  }
  for (const input of [nkInput, depthInput, ratioInput]) {
    input.addEventListener("change", guard(() => sendParams(model.get("kind"))));
  }
  click(resetButton, () => {
    hideConfirm();
    model.send({ op: "reset" });
  });
  click(saveButton, () => {
    hideConfirm();
    pendingSave = { name: nameInput.value, message: messageInput.value };
    model.send({ op: "save", ...pendingSave, overwrite: false });
  });
  click(overwriteButton, () => {
    if (pendingSave) model.send({ op: "save", ...pendingSave, overwrite: true });
    hideConfirm();
  });
  click(cancelButton, hideConfirm);
  click(table, (event) => {
    const button = event.target.closest("[data-action=load]");
    if (!button) return;
    hideConfirm();
    model.send({ op: "load", file: button.getAttribute("data-file") });
  });

  // --- Listen to Python.
  on("msg:custom", (msg) => {
    if (!msg || msg.confirm_overwrite === undefined) return;
    confirmText.textContent = `A saved grid named "${msg.confirm_overwrite}" already exists. Replace it?`;
    confirmRow.hidden = false;
  });
  // errors too: a rejected edit changes no trait, so the errors change is the cue to put
  // the inputs back to the grid's real values (the error box says what was rejected).
  for (const name of ["kind", "nk", "depth", "ratio", "errors"]) on(`change:${name}`, drawControls);
  for (const name of ["dz", "name", "source", "warning"]) on(`change:${name}`, drawGrid);
  on("change:library", drawLibrary);
  on("change:name", () => (nameInput.value = model.get("name") || ""));

  drawControls();
  drawGrid();
  drawLibrary();
  nameInput.value = model.get("name") || "";
  return () => listeners.forEach(([event, handler]) => model.off(event, handler));
}

export default defineWidget(render);
