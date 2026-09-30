"""Run by test_grid_sketcher_ipympl.py in its own process: drive a GridSketcher on the
real ipympl backend with the messages the browser sends, and print the results."""

import json

import matplotlib
import numpy as np

matplotlib.use("module://ipympl.backend_nbagg")

from mom6_forge.grid_sketcher import GridSketcher, Outline  # noqa: E402

UNASKED = []  # frames sent other than in reply to the browser's "draw"


def send(canvas, kind, button=0, modifiers=(), **kw):
    msg = dict(type=kind, button=button, buttons=1, modifiers=list(modifiers))
    msg.update(guiEvent={}, **kw)
    canvas.answering = kind
    canvas._handle_message(canvas, msg, [])
    canvas.answering = None


def sketch(ratio=1, blank=False):
    """A sketch whose canvas records what it sends, after the frontend's start-up."""
    s = GridSketcher(Outline.from_bbox(-10, 10, -5, 5), resolution_km=110, blank=blank)
    c, s.sent, c.answering = s.fig.canvas, [], None

    def record(name, real):
        def wrapped(m):
            s.sent.append((name, m))
            if name == "send_binary" and c.answering != "draw":
                UNASKED.append(m)
            return real(m)

        return wrapped

    for name in ("send_json", "send_binary"):
        setattr(c, name, record(name, getattr(c, name)))
    if ratio != 1:
        send(c, "set_dpi_ratio", dpi_ratio=ratio)
        send(c, "set_device_pixel_ratio", device_pixel_ratio=ratio)
    for kind in ("refresh", "send_image_mode", "initialized", "draw"):
        send(c, kind)
    return s, c


def at(s, i):
    """Vertex i where the browser reports it: device pixels, y from the top."""
    vx, vy = s._vertex_pixels()
    return dict(x=vx[i], y=s.fig.canvas.get_renderer().height - vy[i])


def click(c, button=0, modifiers=(), **xy):
    send(c, "button_press", button, modifiers, **xy)
    send(c, "button_release", button, modifiers, buttons=0, **xy)


def fire(s, name):
    if name in s._timers:
        func, args, kwargs = s._timers[name].callbacks[0]
        func(*args, **kwargs)


def counts(s):
    """Draw requests and frames sent since the last call."""
    r = [sum(k == "send_json" and m["type"] == "draw" for k, m in s.sent)]
    r.append(sum(k == "send_binary" for k, m in s.sent))
    s.sent.clear()
    return r


def png_size(ratio):
    s, c = sketch(ratio)
    return [s.fig.dpi, c.get_renderer().width, c.get_renderer().height]


def layout():
    s, c = sketch()

    def walk(w):
        yield w
        for child in getattr(w, "children", ()):
            yield from walk(child)

    tips = [w for w in walk(s) if getattr(w, "tooltip", None)]
    # "p" and "o" would switch the hidden toolbar to pan or zoom
    for key in ("kp", "ko"):
        send(c, "key_press", key=key)
    bars = [c.toolbar_visible, c.header_visible, c.footer_visible, c.resizable]
    bars.append(str(c.toolbar.mode))
    return dict(
        flow=s.layout.flex_flow,
        canvas=[c.layout.width, c.layout.flex, c.layout.border, c.capture_scroll],
        bars=bars,
        panel=s.control_panel.layout.width,
        help=[s.help_html.layout.width, s.children[0].children.index(s.help_html)],
        edits=[
            b.description
            for row in s.control_panel.children
            if s.undo_button in getattr(row, "children", ())
            for b in row.children
        ],
        tips_without_description=[
            type(w).__name__ for w in tips if not w.has_trait("description")
        ],
    )


def double_click():
    s, c = sketch()
    xy = at(s, 1)
    click(c, **xy)
    click(c, **xy)
    send(c, "dblclick", **xy)
    return s.outline.corners


def delete(button, modifiers):
    s, c = sketch()
    s.outline.insert(0, 0.0, -5.0)
    s._update_outline_artists()
    click(c, button, modifiers, **at(s, 1))
    return s.outline.n


def drawing():
    """Blank: clicks add points, a short drag pans (the globe stays), a right-click
    deletes, point 1 closes."""
    s, c = sketch(blank=True)
    b, h = s.ax.bbox, c.get_renderer().height

    def on_map(fx, fy):
        return dict(x=b.x0 + fx * b.width, y=h - b.y0 - fy * b.height)

    for fxy in [(0.3, 0.3), (0.7, 0.3), (0.7, 0.7), (0.3, 0.7), (0.5, 0.2)]:
        click(c, **on_map(*fxy))
    r, centre, (x, y) = [s.outline.n], s.globe.centre, on_map(0.5, 0.5).values()
    send(c, "button_press", x=x, y=y)
    send(c, "motion_notify", x=x + 40, y=y)
    send(c, "button_release", buttons=0, x=x + 40, y=y)
    r += [s.globe.centre == centre, s.outline.n]
    click(c, 2, **at(s, 4))
    click(c, **at(s, 0))
    return r + [s.outline.n, s.outline.corners]


def drag():
    s, c = sketch()
    x0 = at(s, 0)
    to = dict(x=x0["x"] + 15, y=x0["y"] + 10)
    counts(s)
    send(c, "button_press", **x0)
    send(c, "motion_notify", **to)
    r = dict(sketched=len(s.frame_times), motion=counts(s))
    send(c, "draw")
    r.update(answer=counts(s), mode=c._current_image_mode)
    send(c, "button_release", buttons=0, **to)
    x1 = at(s, 0)
    r.update(dx=x1["x"] - x0["x"], dy=x1["y"] - x0["y"], release=counts(s))
    return r


def grab(ratio):
    """Press near the rim of corner 1's black disc, in the browser's device pixels."""
    s, c = sketch(ratio)
    xy = at(s, 0)
    send(c, "button_press", x=xy["x"] + 0.95 * 6.5 * s.fig.dpi / 72, y=xy["y"])
    return [s._drag, s.outline.n]


def tooltip():
    s, c = sketch()
    x, y = s.ax.transData.transform((s._shown.x[1, 1], s._shown.y[1, 1]))
    xy = dict(x=x, y=c.get_renderer().height - y)
    # Leaving the canvas straight from the map sends no axes_leave
    send(c, "motion_notify", **xy)
    send(c, "figure_leave", **xy)
    r = dict(left="hover" not in s._timers)
    send(c, "motion_notify", **xy)
    counts(s)
    fire(s, "hover")
    r.update(visible=s.hover_annotation.get_visible(), footer=c._message)
    r.update(rest=counts(s))
    send(c, "draw")
    r.update(answer=counts(s))
    send(c, "button_press", x=(at(s, 0)["x"] + at(s, 1)["x"]) / 2, y=at(s, 0)["y"])
    r.update(hidden=not s.hover_annotation.get_visible(), edge_press=counts(s))
    return r


def view():
    s, c = sketch()
    mid = dict(x=c.get_renderer().width / 2, y=c.get_renderer().height / 2)
    counts(s)
    for _ in range(5):
        send(c, "scroll", step=1, **mid)
    r = dict(scroll=counts(s), pending="redraw" in s._timers, lite=s._lite_on)
    fire(s, "redraw")
    fire(s, "lite")
    r.update(after=counts(s), lite_after=s._lite_on)
    send(c, "draw")
    r.update(end_mode=c._current_image_mode, end=counts(s))
    before = s.outline.to_dict()
    send(c, "button_press", **mid)
    for k in range(5):
        send(c, "motion_notify", x=mid["x"] - 10 * k, y=mid["y"])
    send(c, "button_release", buttons=0, x=mid["x"] - 40, y=mid["y"])
    r.update(pan=counts(s), moved=s.outline.to_dict() != before)
    return r


def gestures():
    """A pan released off the canvas, then Box pressed as the browser does, then a
    drag on the map; then the move handle and the rotate knob dragged."""
    s, c = sketch()
    b, h = s.ax.bbox, c.get_renderer().height
    x, y = b.x0 + 0.2 * b.width, h - b.y0 - 0.2 * b.height
    send(c, "button_press", x=x, y=y)
    send(c, "motion_notify", x=x + 40, y=y)
    send(c, "figure_leave", x=x + 40, y=y)
    state = dict(method="update", state=dict(value=True), buffer_paths=[])
    s.box_button._handle_msg(dict(content=dict(data=state), buffers=[]))
    lim = s.ax.get_xlim()
    send(c, "motion_notify", buttons=0, x=x - 60, y=y)
    r = dict(still=s.ax.get_xlim() == lim)
    send(c, "button_press", x=x, y=y)
    for k in range(1, 5):
        send(c, "motion_notify", x=x + 30 * k, y=y - 20 * k)
    send(c, "button_release", buttons=0, x=x + 120, y=y - 80)
    r.update(box=[s.outline.n, s.box_button.value])

    def handle(line):
        hx, hy = s.ax.transData.transform(np.column_stack(line.get_data()))[0]
        return hx, h - hy

    def drag(line, dx, dy):
        (x, y), lon = handle(line), list(s.outline.lon)
        send(c, "button_press", x=x, y=y)
        send(c, "motion_notify", x=x + dx / 2, y=y + dy / 2)
        send(c, "button_release", buttons=0, x=x + dx, y=y + dy)
        return s.outline.lon != lon

    r.update(moved=drag(s.move_handle, 30, 0))
    (mx, my), (kx, ky) = handle(s.move_handle), handle(s.rotate_handle)
    r.update(turned=drag(s.rotate_handle, my - ky, my - ky))
    (mx, my), (kx, ky) = handle(s.move_handle), handle(s.rotate_handle)
    r.update(knob=float(np.degrees(np.arctan2(my - ky, kx - mx))))
    return r


if __name__ == "__main__":
    results = dict(png=[png_size(1), png_size(2)], layout=layout())
    results.update(double_click=double_click(), drag=drag(), tooltip=tooltip())
    results.update(delete=[delete(2, ()), delete(0, ("ctrl",))], view=view())
    results.update(grab=[grab(1), grab(2)], drawing=drawing())
    results.update(gestures=gestures(), unasked=len(UNASKED))
    print(json.dumps(results))
