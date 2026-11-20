"""Contract tests for :mod:`linetess.tessellator`.

All geometry is in screen-space pixels. Every test calls at least one of the
stub functions and fails until the implementation is fixed.
"""

import math

import pytest

from linetess import (
    LineError,
    bevel_join,
    dash_pattern,
    expand_segment,
    miter_join,
    snap_to_pixel,
    tessellate_batch,
    tessellate_bezier,
    validate_mesh,
)

ABS = 1e-6


def _signed_area(poly):
    total = 0.0
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        total += x1 * y2 - x2 * y1
    return 0.5 * total


def _assert_points(actual, expected, abs_tol=ABS):
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        assert got[0] == pytest.approx(want[0], abs=abs_tol)
        assert got[1] == pytest.approx(want[1], abs=abs_tol)


def _assert_intervals(actual, expected):
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        assert got[0] == pytest.approx(want[0], abs=ABS)
        assert got[1] == pytest.approx(want[1], abs=ABS)


def _sample_quadratic(p0, p1, p2, count=2001):
    samples = []
    for i in range(count):
        t = i / (count - 1)
        u = 1.0 - t
        samples.append(
            (
                u * u * p0[0] + 2.0 * u * t * p1[0] + t * t * p2[0],
                u * u * p0[1] + 2.0 * u * t * p1[1] + t * t * p2[1],
            )
        )
    return samples


def _point_segment_distance(pt, a, b):
    ax, ay = a
    bx, by = b
    px, py = pt
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom <= 0.0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / denom
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _point_polyline_distance(pt, polyline):
    return min(
        _point_segment_distance(pt, a, b)
        for a, b in zip(polyline, polyline[1:])
    )


def test_expand_segment_normals_point_outward():
    quad = expand_segment((0.0, 0.0), (10.0, 0.0), 2.0)
    assert len(quad) == 4
    # Left normal (0, 1) with the documented [p0+n, p0-n, p1-n, p1+n]
    # order gives a counter-clockwise quad with positive area.
    assert _signed_area(quad) == pytest.approx(20.0, abs=ABS)
    ys = sorted(y for _, y in quad)
    assert ys == pytest.approx([-1.0, -1.0, 1.0, 1.0], abs=ABS)


def test_miter_join_respects_limit():
    prev, corner, nxt = (0.0, 0.0), (10.0, 0.0), (10.0, 10.0)
    # A right angle has miter ratio sqrt(2)/2 ~= 0.707, so limit 0.5 forces
    # the bevel fallback.
    join = miter_join(prev, corner, nxt, width=2.0, limit=0.5)
    assert len(join) == 2
    _assert_points(join, bevel_join(prev, corner, nxt, width=2.0))


def test_bezier_subdivision_follows_curvature():
    p0, p1, p2 = (0.0, 0.0), (100.0, 100.0), (200.0, 0.0)
    mesh = tessellate_bezier(p0, p1, p2, tolerance=0.01, width=1.0)
    points = mesh["points"]
    assert len(points) >= 4
    assert len(points) % 2 == 0
    center = [
        (
            (points[i][0] + points[i + 1][0]) / 2.0,
            (points[i][1] + points[i + 1][1]) / 2.0,
        )
        for i in range(0, len(points), 2)
    ]
    mids = [
        ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
        for a, b in zip(center, center[1:])
    ]
    dense = _sample_quadratic(p0, p1, p2)
    worst = max(_point_polyline_distance(mid, dense) for mid in mids)
    # 16 fixed segments on this curve deviate by roughly 0.25 px.
    assert worst <= 0.02


def test_bezier_tolerance_controls_subdivision():
    p0, p1, p2 = (0.0, 0.0), (100.0, 100.0), (200.0, 0.0)
    coarse = tessellate_bezier(p0, p1, p2, tolerance=1.0, width=2.0)
    fine = tessellate_bezier(p0, p1, p2, tolerance=0.001, width=2.0)
    assert len(coarse["segments"]) > 0
    assert len(fine["segments"]) > 2 * len(coarse["segments"])


def test_dash_negative_offset_wraps():
    base = dash_pattern(10.0, 2.0, 1.0, 0.0)
    _assert_intervals(base, [(0.0, 2.0), (3.0, 5.0), (6.0, 8.0), (9.0, 10.0)])

    # -1 mod 3 == 2: the pattern starts inside the gap, first dash at t=1.
    negative = dash_pattern(10.0, 2.0, 1.0, -1.0)
    _assert_intervals(negative, [(1.0, 3.0), (4.0, 6.0), (7.0, 9.0)])
    _assert_intervals(negative, dash_pattern(10.0, 2.0, 1.0, 2.0))


def test_snap_to_pixel_uses_pixel_centers():
    snapped = snap_to_pixel([(10.2, 20.8), (10.8, 20.2)], 1.0)
    # round() would give (10, 21) and (11, 20): different pixel columns for
    # two points that belong to the same pixel center.
    _assert_points(snapped, [(10.5, 20.5), (10.5, 20.5)])
    for x, y in snapped:
        assert abs((x - math.floor(x)) - 0.5) <= 1e-9
        assert abs((y - math.floor(y)) - 0.5) <= 1e-9


def test_zero_length_segment_is_dropped():
    mesh = tessellate_batch(
        [[(0.0, 0.0), (0.0, 0.0), (10.0, 0.0)]],
        {"width": 2.0, "tolerance": 0.1},
    )
    assert len(mesh["polylines"]) == 1
    assert mesh["polylines"][0]["point_count"] == 4
    assert all(
        math.isfinite(value)
        for point in mesh["points"]
        for value in point
    )
    issues = validate_mesh([(1.0, 1.0), (1.0, 1.0)], [[0, 1]])
    assert any("degenerate" in issue.lower() for issue in issues)


def test_validate_mesh_reports_bounds_and_self_intersection():
    points = [(0.0, 0.0), (10.0, 10.0), (0.0, 10.0), (10.0, 0.0)]
    # Segments 0 and 1 cross at (5, 5); segment 2 points at index 99.
    issues = validate_mesh(points, [[0, 1], [2, 3], [1, 99]])
    assert any("out of bounds" in issue.lower() for issue in issues)
    assert any("intersect" in issue.lower() for issue in issues)


def test_tessellate_batch_isolates_buffers_and_rolls_back():
    lines = [
        [(0.0, 0.0), (10.0, 0.0)],
        [(0.0, 5.0), (10.0, 5.0)],
    ]
    out = tessellate_batch(lines, {"width": 2.0, "tolerance": 0.1})
    assert len(out["polylines"]) == 2
    for meta in out["polylines"]:
        start = meta["point_start"]
        count = meta["point_count"]
        seg_start = meta["segment_start"]
        seg_count = meta["segment_count"]
        assert count == 4
        assert seg_count > 0
        for i, j in out["segments"][seg_start:seg_start + seg_count]:
            assert start <= i < start + count
            assert start <= j < start + count

    second = out["polylines"][1]
    ys = [
        out["points"][k][1]
        for k in range(second["point_start"],
                       second["point_start"] + second["point_count"])
    ]
    assert min(ys) == pytest.approx(4.0, abs=ABS)
    assert max(ys) == pytest.approx(6.0, abs=ABS)

    # One bad polyline invalidates the whole batch.
    with pytest.raises(LineError):
        tessellate_batch(
            [[(0.0, 0.0), (10.0, 0.0)], [(3.0, 3.0)]],
            {"width": 2.0, "tolerance": 0.1},
        )


def test_zero_width_bezier_raises():
    with pytest.raises(LineError):
        tessellate_bezier((0.0, 0.0), (5.0, 5.0), (10.0, 0.0), 0.1, 0.0)


def test_horizontal_quad_area_equals_width_times_length():
    quad = expand_segment((0.0, 0.0), (10.0, 0.0), 3.0)
    assert abs(_signed_area(quad)) == pytest.approx(30.0, abs=ABS)


def test_miter_used_when_limit_is_large():
    join = miter_join(
        (0.0, 0.0), (10.0, 0.0), (10.0, 10.0), width=2.0, limit=10.0
    )
    assert len(join) == 3
    _assert_points(join, [(10.0, 1.0), (9.0, 1.0), (9.0, 0.0)])


def test_zero_width_expand_raises():
    with pytest.raises(LineError):
        expand_segment((0.0, 0.0), (10.0, 0.0), 0.0)


def test_empty_batch_returns_empty():
    out = tessellate_batch([], {"width": 2.0, "tolerance": 0.1})
    assert out == {"points": [], "segments": [], "polylines": []}


def test_batch_success_preserves_order():
    lines = [
        [(0.0, 0.0), (4.0, 0.0)],
        [(0.0, 10.0), (4.0, 10.0)],
        [(0.0, 20.0), (4.0, 20.0)],
    ]
    out = tessellate_batch(lines, {"width": 1.0, "tolerance": 0.1})
    assert len(out["polylines"]) == 3
    centers = []
    for meta in out["polylines"]:
        start = meta["point_start"]
        count = meta["point_count"]
        ys = [out["points"][k][1] for k in range(start, start + count)]
        centers.append(sum(ys) / len(ys))
    assert centers == pytest.approx([0.0, 10.0, 20.0], abs=ABS)
