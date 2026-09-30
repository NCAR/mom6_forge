"""GridSketcher's map: a globe facing the outline, zoomable out to all of it."""

import re
from types import SimpleNamespace
import numpy as np
import xarray as xr
from mom6_forge.grid_sketcher import GridSketcher, Outline


def _scroll(sk, step, times):
    for _ in range(times):
        (x0, x1), (y0, y1) = sk.ax.get_xlim(), sk.ax.get_ylim()
        xy = dict(xdata=(x0 + x1) / 2, ydata=(y0 + y1) / 2)
        sk._on_scroll(SimpleNamespace(inaxes=sk.ax, step=step, **xy))
    sk._set_lite(False)


def test_zooming_out_shows_the_whole_arctic_and_zooming_in_stops_at_1_50():
    sk = GridSketcher(Outline.from_bbox(20, 60, 70, 80))  # Barents Sea
    fit = np.ptp(sk.ax.get_xlim())
    _scroll(sk, -1, 40)
    (x0, x1), (y0, y1) = sk.ax.get_xlim(), sk.ax.get_ylim()
    assert min(x1 - x0, y1 - y0) >= 2 * sk.globe.radius
    lon = np.arange(-180.0, 180.0, 5.0)
    x, y = sk.globe.to_xy(lon, np.full_like(lon, 50.0))  # 50 N all round
    assert x0 < x.min() and x.max() < x1 and y0 < y.min() and y.max() < y1
    _scroll(sk, 1, 80)
    assert np.isclose(np.ptp(sk.ax.get_xlim()), 0.02 * fit)
    sk.close()


def test_land_on_the_whole_globe_is_finite_on_it_and_has_no_seams():
    sk = GridSketcher(Outline.from_bbox(165, 205, -78, -68))  # Ross Sea
    _scroll(sk, -1, 40)
    xy = np.concatenate([p.vertices for p in sk.land_collection.get_paths()])
    assert len(xy) > 1000 and np.isfinite(xy).all()
    assert np.hypot(*xy.T).max() < 1.0001 * sk.globe.radius
    # Natural Earth cuts Antarctica along the dateline to the pole: no coast drawn there
    pole = sk.globe.to_xy(0.0, -90.0)
    for path in sk.coast_collection.get_paths():
        v = path.vertices
        drawn = np.isfinite(v[:-1] + v[1:]).all(axis=1)
        near = np.hypot(*(v - pole).T) < 3e5  # the coast is 490 km off at the closest
        assert not (drawn & (near[:-1] | near[1:])).any()
    sk.close()


def test_the_globe_faces_the_outline_and_stays_when_the_solve_plane_changes():
    # Across the dateline, solved on the polar plane
    sk = GridSketcher(Outline.from_bbox(150, 210, 70, 80), projection="stere")
    o, cg = sk.outline, sk.preview
    assert sk.projection.kind == "stere" and sk.globe.centre == (180.0, 75.0)
    assert np.allclose(sk.vertex_scatter.get_data(), sk.globe.to_xy(o.lon, o.lat))
    assert np.allclose(sk._shown.x, sk.globe.to_xy(cg.lon, cg.lat)[0])
    view = (sk.ax.get_xlim(), sk.ax.get_ylim())
    sk.projection_buttons["lcc"].value = True
    assert sk.projection.kind == "lcc" and (sk.ax.get_xlim(), sk.ax.get_ylim()) == view
    # A vertex dragged off the globe stops on its rim
    o.begin_drag()
    sk._drag = 0
    sk._move_dragged(3 * sk.globe.radius, 0.0, sketch=False)
    assert np.isfinite([o.lon[0], o.lat[0]]).all()
    sk.close()


def test_an_outline_round_the_pole_is_fitted_on_a_globe_facing_it():
    sk = GridSketcher(Outline([0, 90, 180, 270], [78] * 4, [0, 1, 2, 3]))
    assert sk.globe.centre[1] == 90.0
    (x0, x1), (y0, y1) = sk.ax.get_xlim(), sk.ax.get_ylim()
    x, y = sk.vertex_scatter.get_data()
    assert x0 < x.min() and x.max() < x1 and y0 < y.min() and y.max() < y1
    sk.close()


def _eastward(lon):
    """A uniform eastward transport of 100 m2/s and its streamfunction on a global
    lon/lat grid: psi falls 11.1 Sv per degree north."""
    lat = np.arange(-80.0, 90.0, 1.0)
    u = np.full((lat.size, lon.size), 100.0)
    psi = -100.0 * 6.371e6 * np.radians(lat + 80.0)[:, None] / 1e6 + 0 * u
    names = {"psi": psi, "U": u, "V": 0 * u}
    return xr.Dataset(
        {k: (("lat", "lon"), v) for k, v in names.items()},
        coords={"lat": lat, "lon": lon},
    )


def test_streamlines_follow_the_parallels_and_hide_while_panning():
    for lon in (np.arange(0.0, 360.0, 1.0), np.arange(-180.0, 180.0, 1.0)):
        sk = GridSketcher(currents=_eastward(lon))
        lines = sk.psi_lines.get_segments()
        assert 5 <= len(lines) <= 20 and sk.timings["currents"] > 0
        lat = [sk.globe.to_lonlat(*v.T)[1] for v in lines]
        assert max(np.ptp(v) for v in lat) < 0.1
        psi = sk._value_at("psi", *sk.globe.to_xy(240.0, 35.0))
        assert abs(psi + 100.0 * 6.371 * np.radians(115.0)) < 6
        sk._set_lite(True)
        assert not sk.psi_lines.get_visible()
        sk._set_lite(False)
        sk.show_currents.value = False
        assert not sk.psi_lines.get_segments()
        sk.close()


def test_details_give_the_transport_and_angle_across_each_open_side():
    sk = GridSketcher(currents=_eastward(np.arange(0.0, 360.0, 1.0)))
    html = sk.details_html.value
    assert re.search(r"West side: net \+\d+\.\d Sv into the grid, flow \d°", html)
    assert re.search(
        r"South side: net [+-]0\.\d Sv into the grid, flow (8\d|90)°", html
    )
    sk.close()
