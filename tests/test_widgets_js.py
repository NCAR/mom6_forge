"""Front-end contract tests for mom6_forge's web widgets, in headless Chromium.

Each test renders a widget's real JavaScript (``widget._esm``) against a small mock of
anywidget's ``model``, starting from the state the Python widget would send. It then makes
gestures and checks which actions the front end sends, and that no JavaScript error happened.
The Python side of each action is tested in the widget's own test file.

Every test leaves a screenshot, ``widget.png``, in its pytest ``tmp_path`` for a look after a
failure. These tests skip themselves when Playwright or its Chromium is not installed:
``pip install playwright && python -m playwright install chromium-headless-shell``.
"""

import base64
from pathlib import Path

import pytest

sync_api = pytest.importorskip("playwright.sync_api")

import mom6_forge.widgets._base as base

# A stand-in for anywidget's model: state, change events, and a record of sent messages.
MOCK = """
<div id="root"></div>
<script>
window.__errors = [];
window.addEventListener("error", (e) => window.__errors.push(String(e.message)));
window.__render = async (esm, rawState) => {
  const state = {};
  for (const [k, v] of Object.entries(rawState)) {
    if (v && typeof v === "object" && "__bytes__" in v) {
      const bin = Uint8Array.from(atob(v.__bytes__), (c) => c.charCodeAt(0));
      state[k] = new DataView(bin.buffer);
    } else state[k] = v;
  }
  const handlers = {};
  const model = {
    sent: [],
    get: (k) => state[k],
    set: (k, v) => { state[k] = v; (handlers["change:" + k] || []).forEach((f) => f()); },
    save_changes: () => {},
    on: (evt, fn) => { (handlers[evt] ||= []).push(fn); },
    off: (evt, fn) => { handlers[evt] = (handlers[evt] || []).filter((f) => f !== fn); },
    send: (msg, _callbacks, buffers) =>
      model.sent.push({ ...msg, _buffers: (buffers || []).map((b) => b.byteLength) }),
    emit: (evt, ...args) => (handlers[evt] || []).forEach((f) => f(...args)),
  };
  window.__model = model;
  const url = URL.createObjectURL(new Blob([esm], { type: "text/javascript" }));
  const mod = await import(url);
  mod.default.render({ model, el: document.getElementById("root") });
};
</script>
"""


def _encode(value):
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"__bytes__": base64.b64encode(bytes(value)).decode()}
    return value


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as p:
        try:
            chromium = p.chromium.launch()
        except Exception as exc:  # browser not installed here
            pytest.skip(f"headless Chromium unavailable: {exc}")
        yield chromium
        chromium.close()


@pytest.fixture
def page(browser, tmp_path):
    page = browser.new_page(viewport={"width": 1100, "height": 760})
    page.console_errors = []
    page.on(
        "console", lambda m: m.type == "error" and page.console_errors.append(m.text)
    )
    page.on("pageerror", lambda e: page.console_errors.append(str(e)))
    yield page
    page.screenshot(path=tmp_path / "widget.png")
    page.close()


def render(page, widget):
    """Render ``widget``'s front end in ``page`` from its current Python state."""
    state = {
        k: _encode(v) for k, v in widget.get_state().items() if not k.startswith("_")
    }
    page.set_content(MOCK)
    page.evaluate("([esm, state]) => window.__render(esm, state)", [widget._esm, state])
    return page


def sent(page):
    """Actions the front end has sent so far, oldest first (buffers as byte lengths)."""
    return page.evaluate("window.__model.sent")


def assert_clean(page):
    """No JavaScript error was raised, logged, or shown in the widget."""
    assert page.console_errors == [], page.console_errors
    assert page.evaluate("window.__errors") == []
    assert page.locator(".mf-js-error").count() == 0
    assert [m for m in sent(page) if m["op"] == "js_error"] == []


# --- the shared prelude ---------------------------------------------------------------

PRELUDE = Path(base.__file__).with_name("_prelude.js").read_text()


class _Probe(base.ActionWidget):
    """A widget whose only button throws, to test the shared prelude."""

    _esm = PRELUDE + """
function render({ model, el, guard, errorBox }) {
  const b = document.createElement("button");
  b.dataset.action = "boom";
  b.addEventListener("click", guard(() => { throw new Error("handler broke"); }));
  el.append(errorBox(), b);
}
export default defineWidget(render);
"""


def test_prelude_reports_handler_errors_to_person_and_python(page):
    """An error in a guarded handler shows a box and reaches Python as js_error."""
    render(page, _Probe())
    page.click("[data-action=boom]")
    assert "handler broke" in page.inner_text(".mf-js-error")
    reports = [m for m in sent(page) if m["op"] == "js_error"]
    assert [r["message"] for r in reports] == ["handler broke"]


# --- VGridCreator -----------------------------------------------------------------------


def test_vgrid_draws_python_state_and_sends_params(page, tmp_path):
    """The profile shows every layer from the dz bytes; edits send one params action."""
    from mom6_forge.vgrid import VGrid
    from mom6_forge.widgets import VGridCreator

    widget = VGridCreator(
        VGrid.hyperbolic(12, 400.0, 5.0, name="demo"), repo_root=tmp_path
    )
    render(page, widget)
    assert page.locator(".mf-vgrid-layer").count() == 12

    page.fill("[data-action=nk]", "20")
    page.dispatch_event("[data-action=nk]", "change")
    page.click("[data-action=kind-uniform]")
    params = [m for m in sent(page) if m["op"] == "params"]
    assert params[0]["nk"] == 20 and params[0]["kind"] == "hyperbolic"
    assert params[-1]["kind"] == "uniform"
    assert_clean(page)


def test_vgrid_asks_before_replacing_a_saved_grid(page, tmp_path):
    """Python's confirm_overwrite message shows a confirm row; Replace resends with overwrite."""
    from mom6_forge.widgets import VGridCreator

    render(page, VGridCreator(repo_root=tmp_path))
    page.fill("[data-action=save-name]", "shelf")
    page.fill("[data-action=save-message]", "for the shelf case")
    page.click("[data-action=save]")
    page.evaluate("window.__model.emit('msg:custom', {confirm_overwrite: 'shelf'})")
    page.click("[data-action=save-overwrite]")
    saves = [m for m in sent(page) if m["op"] == "save"]
    assert [s["overwrite"] for s in saves] == [False, True]
    assert saves[-1]["name"] == "shelf"
    assert_clean(page)


# --- TopoEditor -------------------------------------------------------------------------


def _drag(page, start, end):
    """Drag across the canvas between two (x, y) fractions of its size."""
    box = page.locator("[data-role=canvas]").bounding_box()
    at = lambda f: (box["x"] + f[0] * box["width"], box["y"] + f[1] * box["height"])
    page.mouse.move(*at(start))
    page.mouse.down()
    page.mouse.move(*at(end), steps=8)
    page.mouse.up()


def test_topo_brush_stroke_is_one_edit_and_patches_apply(
    page, get_rect_topo_without_vc
):
    """A stroke sends one depth edit with its cells as one buffer; a patch then applies."""
    from mom6_forge.widgets import TopoEditor

    render(page, TopoEditor(get_rect_topo_without_vc))
    assert not page.locator("[data-action=undo]").is_visible()  # no history without git
    page.click("[data-action=tool-depth]")
    _drag(page, (0.3, 0.5), (0.6, 0.5))
    edits = [m for m in sent(page) if m["op"] == "edit"]
    assert len(edits) == 1 and edits[0]["edit"]["op"] == "depth"
    assert edits[0]["_buffers"][0] > 0 and edits[0]["_buffers"][0] % 8 == 0
    page.evaluate("""window.__model.emit("msg:custom", {kind: "patch", n: 1}, [
             new DataView(new Int32Array([0, 0]).buffer),
             new DataView(new Float32Array([42]).buffer),
             new DataView(new Uint8Array([1]).buffer)])""")
    assert_clean(page)


def test_topo_box_selection_and_history_send_actions(page, get_rect_topo_with_vc):
    """Select a box, set it to land (sent as a rect), then undo."""
    from mom6_forge.widgets import TopoEditor

    render(page, TopoEditor(get_rect_topo_with_vc, open_boundaries=True))
    page.click("[data-action=tool-select]")
    _drag(page, (0.4, 0.4), (0.6, 0.6))
    page.click("[data-action=set-land]")
    page.click("[data-action=show-open-boundaries]")  # turn the overlay off
    page.evaluate("window.__model.set('can_undo', true)")
    page.click("[data-action=undo]")
    ops = [m["op"] for m in sent(page) if m["op"] != "sync"]
    land = next(m["edit"] for m in sent(page) if m["op"] == "edit")
    assert ops == ["edit", "open_boundaries", "undo"]
    assert land["op"] == "land" and len(land["rect"]) == 4
    assert_clean(page)


# --- GridSketcher (loads OpenLayers and proj4 from cdn.jsdelivr.net) --------------------

# The outline's vertex i, as a page position; and the map's layers' feature counts
_VERTEX = """(i) => { const map = document.querySelector('[data-role=map]').olMap;
  const ring = map.getLayers().item(2).getSource().getFeatures()[0].getGeometry();
  const p = map.getPixelFromCoordinate(ring.getCoordinates()[0][i]);
  const box = map.getTargetElement().getBoundingClientRect();
  return [box.left + p[0], box.top + p[1]]; }"""
_MAP = """() => { const map = document.querySelector('[data-role=map]').olMap;
  return {code: map.getView().getProjection().getCode(),
          features: map.getLayers().getArray().map((l) => l.getSource().getFeatures().length)}; }"""


def _sketch(page, widget):
    """Render a GridSketcher and wait for its map (the libraries load from the CDN)."""
    render(page, widget)
    page.wait_for_function(
        "document.querySelector('[data-role=map]').olMap", timeout=30_000
    )


def test_sketcher_draws_python_state_in_its_projection(page):
    """The map is drawn in the solve projection, and follows Python to a polar one."""
    from mom6_forge.widgets import GridSketcher

    _sketch(page, GridSketcher())
    first = page.evaluate(_MAP)
    assert first["features"] == [2, 1, 1]  # land + graticule, grid lines, outline
    assert page.get_attribute("[data-action=projection-merc]", "aria-pressed") == "true"
    polar = GridSketcher(
        outline=dict(lon=[0, 90, 180, 270], lat=[75] * 4, corners=[0, 1, 2, 3]),
        resolution_km=150,
    )
    state = {k: v for k, v in polar.get_state().items() if not k.startswith("_")}
    page.evaluate(
        """(state) => { for (const [k, v] of Object.entries(state)) {
             const bytes = v && v.__bytes__ !== undefined;
             const bin = bytes && Uint8Array.from(atob(v.__bytes__), (c) => c.charCodeAt(0));
             window.__model.set(k, bytes ? new DataView(bin.buffer) : v); } }""",
        {k: _encode(v) for k, v in state.items()},
    )
    page.wait_for_function(f"({_MAP})().code !== {first['code']!r}")
    assert (
        page.get_attribute("[data-action=projection-stere]", "aria-pressed") == "true"
    )
    assert_clean(page)


def test_sketcher_gestures_send_actions(page):
    """Drag a vertex, double- and right-click one, drag a Box, pick a projection."""
    from mom6_forge.widgets import GridSketcher

    _sketch(page, GridSketcher())
    x, y = page.evaluate(_VERTEX, 2)
    page.mouse.move(x, y)
    page.mouse.down()
    page.mouse.move(x + 30, y - 20, steps=6)
    page.mouse.up()
    x, y = page.evaluate(_VERTEX, 3)
    page.mouse.dblclick(x, y)
    page.mouse.click(x, y, button="right")
    page.click("[data-action=box]")
    page.mouse.move(600, 200)
    page.mouse.down()
    page.mouse.move(700, 300, steps=8)
    page.mouse.up()
    page.click("[data-action=projection-lcc]")
    actions = [m for m in sent(page) if m["op"] != "view"]
    assert [(m["op"], m.get("i"), m["_buffers"]) for m in actions] == [
        ("outline", None, [64]),  # 4 (lon, lat) float64 pairs
        ("corner", 3, []),
        ("delete", 3, []),
        ("draw", None, [64]),  # the box: 4 vertices
        ("projection", None, []),
    ]
    assert actions[-1]["kind"] == "lcc"
    assert_clean(page)
