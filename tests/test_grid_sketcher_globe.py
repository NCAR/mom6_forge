"""GridSketcher's map: a globe facing the outline, zoomable out to all of it."""

from types import SimpleNamespace
import numpy as np
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
