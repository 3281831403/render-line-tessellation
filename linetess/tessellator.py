"""Screen-space stroke tessellation primitives.

Coordinates are 2D screen-space pixels: ``x`` grows to the right and ``y``
grows downwards. Stroke widths are in pixels and flattening tolerances are
in pixels as well. All geometric comparisons use ``EPS = 1e-6``.
"""

from __future__ import annotations

import math

EPS = 1e-6

# Maximum recursion depth for adaptive Bezier flattening. Real inputs reach
# their tolerance long before this; the cap only guards pathological input.
_MAX_BEZIER_DEPTH = 16

# Miter ratio cap used by the batch ribbon builder. Sharp corners are clamped
# to this ratio instead of growing an unbounded spike.
_BATCH_MITER_LIMIT = 4.0


class LineError(Exception):
    """Raised when the input geometry cannot be tessellated."""


# ---------------------------------------------------------------------------
# Small vector helpers (all inputs are (x, y) tuples).
# ---------------------------------------------------------------------------


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def _length(v):
    return math.hypot(v[0], v[1])


def _normalize(v):
    length = math.hypot(v[0], v[1])
    if length < EPS:
        raise LineError("cannot normalize a zero-length vector")
    return (v[0] / length, v[1] / length)


def _left_normal(d):
    """Unit left normal of a unit direction ``d`` (screen space)."""
    return (-d[1], d[0])


def _is_finite_point(p):
    return math.isfinite(p[0]) and math.isfinite(p[1])


def _check_width(width):
    if not (width > 0) or not math.isfinite(width):
        raise LineError("width must be greater than zero")


# ---------------------------------------------------------------------------
# Segment expansion / joins
# ---------------------------------------------------------------------------


def expand_segment(p0, p1, width):
    """Expand a single line segment into a stroke quad.

    The segment goes from ``p0`` to ``p1`` in screen space. The returned
    polygon is closed implicitly (the first point is not repeated) and wound
    counter-clockwise, so its signed shoelace area is exactly
    ``width * |p1 - p0|``.

    Let ``L = |p1 - p0|``, ``hw = width / 2`` and let
    ``n = (-(p1.y - p0.y) / L, (p1.x - p0.x) / L)`` be the unit left normal.
    The result is::

        [p0 + n*hw, p0 - n*hw, p1 - n*hw, p1 + n*hw]

    Args:
        p0: First endpoint ``(x, y)`` in pixels.
        p1: Second endpoint ``(x, y)`` in pixels.
        width: Stroke width in pixels; must be > 0.

    Returns:
        list[tuple[float, float]]: Four quad corners.

    Raises:
        LineError: If ``width <= 0`` or ``|p1 - p0| < EPS`` (a zero-length
            segment has no direction, so its normal would be NaN).
    """
    _check_width(width)
    if not (_is_finite_point(p0) and _is_finite_point(p1)):
        raise LineError("segment endpoints must be finite")
    delta = _sub(p1, p0)
    if _length(delta) < EPS:
        raise LineError("zero-length segment has no direction")
    d = _normalize(delta)
    n = _left_normal(d)
    hw = width / 2.0
    nx, ny = n[0] * hw, n[1] * hw
    return [
        (p0[0] + nx, p0[1] + ny),
        (p0[0] - nx, p0[1] - ny),
        (p1[0] - nx, p1[1] - ny),
        (p1[0] + nx, p1[1] + ny),
    ]


def _join_endpoints(p_prev, p, p_next, width, limit=None):
    """Shared geometry for :func:`miter_join` and :func:`bevel_join`.

    Returns ``(a, b, n_in, n_out, dot)`` or raises :class:`LineError`.
    """
    _check_width(width)
    if limit is not None and not (limit > 0):
        raise LineError("miter limit must be greater than zero")
    if not (
        _is_finite_point(p_prev)
        and _is_finite_point(p)
        and _is_finite_point(p_next)
    ):
        raise LineError("join points must be finite")

    d_in = _sub(p, p_prev)
    d_out = _sub(p_next, p)
    if _length(d_in) < EPS or _length(d_out) < EPS:
        raise LineError("adjacent segment is shorter than EPS")

    d_in = _normalize(d_in)
    d_out = _normalize(d_out)
    n_in = _left_normal(d_in)
    n_out = _left_normal(d_out)

    dot = d_in[0] * d_out[0] + d_in[1] * d_out[1]
    dot = max(-1.0, min(1.0, dot))
    if dot <= -1.0 + 1e-12:
        # The path doubles back exactly; the miter vertex is undefined.
        raise LineError("path doubles back; miter is undefined")

    hw = width / 2.0
    a = (p[0] + n_in[0] * hw, p[1] + n_in[1] * hw)
    b = (p[0] + n_out[0] * hw, p[1] + n_out[1] * hw)
    return a, b, n_in, n_out, dot, hw, p


def miter_join(p_prev, p, p_next, width, limit):
    """Build the outer join patch at ``p`` for a polyline corner.

    Incoming direction ``d_in = normalize(p - p_prev)``, outgoing direction
    ``d_out = normalize(p_next - p)``. Let ``n_in`` / ``n_out`` be their unit
    left normals and ``hw = width / 2``. The two offset edge endpoints are::

        a = p + n_in  * hw
        b = p + n_out * hw

    The miter vertex is the intersection of the two offset lines::

        m = p + normalize(n_in + n_out) * (hw / cos(phi / 2))

    where ``phi`` is the turn angle between ``d_in`` and ``d_out``
    (``phi = 0`` means straight ahead). The miter ratio is
    ``|m - p| / width``: ``0.5`` for a straight continuation, about ``0.707``
    for a right angle, and unbounded as the corner gets sharper.

    If ``miter_ratio <= limit`` the join is a miter and the result is
    ``[a, m, b]`` (three points). Otherwise it degrades to a bevel and the
    result is ``[a, b]`` (two points), matching :func:`bevel_join`.

    Args:
        p_prev: Previous polyline point ``(x, y)``.
        p: Corner point ``(x, y)``.
        p_next: Next polyline point ``(x, y)``.
        width: Stroke width in pixels; must be > 0.
        limit: Maximum allowed miter ratio ``|m - p| / width``; must be > 0.
            Typical values are 2-10.

    Returns:
        list[tuple[float, float]]: ``[a, m, b]`` for a miter join, ``[a, b]``
        for a bevel join.

    Raises:
        LineError: If either adjacent segment is shorter than ``EPS``,
            ``width <= 0``, ``limit <= 0``, or the path doubles back exactly
            (``d_in == -d_out``, where the miter vertex is undefined).
    """
    a, b, n_in, n_out, dot, hw, p = _join_endpoints(
        p_prev, p, p_next, width, limit
    )

    cos_half = math.sqrt(max(0.0, (1.0 + dot) / 2.0))
    if cos_half <= 1e-12:
        raise LineError("path doubles back; miter is undefined")

    # |m - p| = hw / cos(phi / 2), so ratio = 0.5 / cos(phi / 2).
    ratio = 0.5 / cos_half
    if ratio <= limit:
        denom = 1.0 + dot
        if denom <= 1e-12:
            raise LineError("path doubles back; miter is undefined")
        # normalize(n_in + n_out) * hw / cos(phi/2)
        #   == (n_in + n_out) * hw / (1 + dot)
        m = (
            p[0] + (n_in[0] + n_out[0]) * hw / denom,
            p[1] + (n_in[1] + n_out[1]) * hw / denom,
        )
        return [a, m, b]
    return [a, b]


def bevel_join(p_prev, p, p_next, width):
    """Build the bevel join patch at ``p``.

    Uses the same outer endpoints as :func:`miter_join`::

        a = p + n_in  * (width / 2)
        b = p + n_out * (width / 2)

    and returns the straight bevel edge ``[a, b]``.

    Args:
        p_prev: Previous polyline point ``(x, y)``.
        p: Corner point ``(x, y)``.
        p_next: Next polyline point ``(x, y)``.
        width: Stroke width in pixels; must be > 0.

    Returns:
        list[tuple[float, float]]: Two points ``[a, b]``.

    Raises:
        LineError: If either adjacent segment is shorter than ``EPS``,
            ``width <= 0``, or the path doubles back exactly.
    """
    a, b, _n_in, _n_out, _dot, _hw, _p = _join_endpoints(
        p_prev, p, p_next, width
    )
    return [a, b]


# ---------------------------------------------------------------------------
# Quadratic Bezier flattening
# ---------------------------------------------------------------------------


def _quadratic_point(p0, p1, p2, t):
    u = 1.0 - t
    return (
        u * u * p0[0] + 2.0 * u * t * p1[0] + t * t * p2[0],
        u * u * p0[1] + 2.0 * u * t * p1[1] + t * t * p2[1],
    )


def _bezier_flatness(p0, p1, p2):
    """Upper bound on the distance between the curve and its chord."""
    return math.hypot(
        p0[0] - 2.0 * p1[0] + p2[0],
        p0[1] - 2.0 * p1[1] + p2[1],
    ) * 0.25


def _flatten_bezier(p0, p1, p2, tolerance):
    """Return parameter values ``t`` (including 0 and 1) to sample at."""
    ts = [0.0]

    def recurse(a, b, c, t0, t1, depth):
        if _bezier_flatness(a, b, c) <= tolerance or depth >= _MAX_BEZIER_DEPTH:
            ts.append(t1)
            return
        ab = ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)
        bc = ((b[0] + c[0]) * 0.5, (b[1] + c[1]) * 0.5)
        mid = ((ab[0] + bc[0]) * 0.5, (ab[1] + bc[1]) * 0.5)
        t_mid = (t0 + t1) * 0.5
        recurse(a, ab, mid, t0, t_mid, depth + 1)
        recurse(mid, bc, c, t_mid, t1, depth + 1)

    recurse(p0, p1, p2, 0.0, 1.0, 0)
    return ts


def tessellate_bezier(p0, p1, p2, tolerance, width):
    """Adaptively tessellate a quadratic Bezier stroke.

    Samples the quadratic Bezier
    ``B(t) = (1-t)^2*p0 + 2*(1-t)*t*p1 + t^2*p2`` for ``t in [0, 1]`` so that
    the maximum distance between the exact curve and the sampled centerline
    polyline is at most ``tolerance`` pixels. Subdivision is curvature-aware:
    a smaller tolerance must produce more samples than a larger tolerance.
    When a tangent vanishes at an endpoint, the chord direction is used
    instead.

    Each centerline sample ``B(t_i)`` is expanded to two stroke vertices with
    the unit left normal ``n_i`` of the local tangent::

        points[2*i]     = B(t_i) + n_i * (width / 2)
        points[2*i + 1] = B(t_i) - n_i * (width / 2)

    Segments (index pairs into ``points``) form the stroke ribbon:

    * for every sample ``i``: ``[2*i, 2*i + 1]`` (a rung),
    * for every span ``i``: ``[2*i, 2*i + 2]`` and
      ``[2*i + 1, 2*i + 3]`` (the two rails).

    Args:
        p0: Start control point ``(x, y)``.
        p1: Middle control point ``(x, y)``.
        p2: End control point ``(x, y)``.
        tolerance: Maximum flattening error in pixels; must be > 0.
        width: Stroke width in pixels; must be > 0.

    Returns:
        dict: ``{"points": list[tuple[float, float]],
        "segments": list[list[int]]}``.

    Raises:
        LineError: If ``width <= 0``, ``tolerance <= 0``, or all three control
            points collapse to one location (no tangent anywhere).
    """
    _check_width(width)
    if not (tolerance > 0) or not math.isfinite(tolerance):
        raise LineError("tolerance must be greater than zero")
    for point in (p0, p1, p2):
        if not _is_finite_point(point):
            raise LineError("control points must be finite")
    if (
        _length(_sub(p1, p0)) < EPS
        and _length(_sub(p2, p1)) < EPS
    ):
        raise LineError("all control points collapse to one location")

    ts = _flatten_bezier(p0, p1, p2, tolerance)
    centers = [_quadratic_point(p0, p1, p2, t) for t in ts]

    chord = _sub(p2, p0)
    hw = width / 2.0
    points = []
    for i, t in enumerate(ts):
        # Derivative of the quadratic Bezier at t.
        tangent = (
            2.0 * ((1.0 - t) * (p1[0] - p0[0]) + t * (p2[0] - p1[0])),
            2.0 * ((1.0 - t) * (p1[1] - p0[1]) + t * (p2[1] - p1[1])),
        )
        if _length(tangent) < EPS:
            # Vanishing tangent: fall back to a neighboring chord, then to the
            # overall chord direction.
            if i > 0:
                tangent = _sub(centers[i], centers[i - 1])
            if _length(tangent) < EPS and i + 1 < len(centers):
                tangent = _sub(centers[i + 1], centers[i])
            if _length(tangent) < EPS:
                tangent = chord
            if _length(tangent) < EPS:
                raise LineError("curve has no tangent anywhere")

        n = _left_normal(_normalize(tangent))
        nx, ny = n[0] * hw, n[1] * hw
        cx, cy = centers[i]
        points.append((cx + nx, cy + ny))
        points.append((cx - nx, cy - ny))

    sample_count = len(points) // 2
    segments = []
    for i in range(sample_count):
        segments.append([2 * i, 2 * i + 1])
    for i in range(sample_count - 1):
        segments.append([2 * i, 2 * i + 2])
        segments.append([2 * i + 1, 2 * i + 3])

    return {"points": points, "segments": segments}


# ---------------------------------------------------------------------------
# Dashes
# ---------------------------------------------------------------------------


def dash_pattern(length, dash, gap, offset):
    """Split a path length into dash intervals.

    The dash pattern is the periodic sequence ``dash`` on, ``gap`` off, with
    ``period = dash + gap``. ``offset`` shifts the pattern along the path and
    is normalized modulo ``period``, so a negative offset or an offset larger
    than one period behaves exactly like its wrapped equivalent::

        phase(t) = (t + offset) mod period

    A point ``t`` is "on" when ``phase(t) < dash``. Returned intervals are
    clamped to ``[0, length]``, ordered by start position, and zero-length
    intervals are dropped. With ``gap == 0`` the whole path is one dash.

    Args:
        length: Path length in pixels; must be >= 0.
        dash: Dash length in pixels; must be > 0.
        gap: Gap length in pixels; must be >= 0.
        offset: Pattern offset in pixels; any finite value, wrapped modulo
            ``dash + gap``.

    Returns:
        list[tuple[float, float]]: ``(start, end)`` intervals with
        ``0 <= start < end <= length``.

    Raises:
        LineError: If ``length < 0``, ``dash <= 0``, or ``gap < 0``.
    """
    if not math.isfinite(length) or length < 0:
        raise LineError("length must be finite and non-negative")
    if not math.isfinite(dash) or dash <= 0:
        raise LineError("dash must be finite and greater than zero")
    if not math.isfinite(gap) or gap < 0:
        raise LineError("gap must be finite and non-negative")
    if not math.isfinite(offset):
        raise LineError("offset must be finite")

    if length == 0:
        return []
    if gap == 0:
        return [(0.0, length)]

    period = dash + gap
    # Normalize the offset modulo the period; Python's % is non-negative for a
    # positive divisor, so negative offsets wrap correctly.
    phase_offset = offset % period

    # Dash k covers [k*period - offset, k*period - offset + dash].
    k_min = int(math.floor((-phase_offset - dash) / period)) - 1
    k_max = int(math.ceil((length + phase_offset) / period)) + 1

    intervals = []
    for k in range(k_min, k_max + 1):
        start = k * period - phase_offset
        end = start + dash
        clamped_start = max(0.0, start)
        clamped_end = min(length, end)
        if clamped_end > clamped_start:
            intervals.append((clamped_start, clamped_end))

    intervals.sort()
    return intervals


# ---------------------------------------------------------------------------
# Pixel snapping
# ---------------------------------------------------------------------------


def snap_to_pixel(points, pixel):
    """Snap points to pixel centers to keep stroke widths stable.

    Each coordinate ``c`` is mapped to the center of the pixel cell that
    contains it::

        snapped = (floor(c / pixel) + 0.5) * pixel

    This is a half-pixel offset alignment, not ``round(c / pixel) * pixel``.
    Lines of odd pixel width land on pixel centers, so even and odd widths do
    not drift by one pixel. Input order and length are preserved.

    Args:
        points: Iterable of ``(x, y)`` screen-space points.
        pixel: Pixel size (device pixel ratio aware); must be > 0.

    Returns:
        list[tuple[float, float]]: Snapped points.

    Raises:
        LineError: If ``pixel <= 0``.
    """
    if not (pixel > 0) or not math.isfinite(pixel):
        raise LineError("pixel size must be greater than zero")
    return [
        (
            (math.floor(x / pixel) + 0.5) * pixel,
            (math.floor(y / pixel) + 0.5) * pixel,
        )
        for x, y in points
    ]


# ---------------------------------------------------------------------------
# Mesh validation
# ---------------------------------------------------------------------------


def _sign(value):
    if value > EPS:
        return 1
    if value < -EPS:
        return -1
    return 0


def _orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _properly_cross(a, b, c, d):
    """True when segments AB and CD cross at a single interior point."""
    o1 = _sign(_orient(a, b, c))
    o2 = _sign(_orient(a, b, d))
    o3 = _sign(_orient(c, d, a))
    o4 = _sign(_orient(c, d, b))
    return o1 * o2 < 0 and o3 * o4 < 0


def validate_mesh(points, segments):
    """Report structural and geometric problems in a segment mesh.

    ``segments`` is a list of index pairs into ``points``. The returned list
    contains one human-readable message per problem and is empty for a valid
    mesh. Checks, in this order:

    1. Degenerate segments: ``|points[i] - points[j]| < EPS``.
    2. Index out of bounds: ``i`` or ``j`` outside ``[0, len(points))``.
    3. Self-intersection: two non-adjacent segments properly cross. Segments
       that share an endpoint index are not reported.

    Message formats::

        "segment {k} is degenerate"
        "segment {k} index out of bounds: {i}"
        "segments {a} and {b} intersect"

    Args:
        points: List of ``(x, y)`` vertices.
        segments: List of ``[i, j]`` index pairs.

    Returns:
        list[str]: Problems found; empty when the mesh is valid.

    Raises:
        LineError: If a segment entry is not a pair of integers.
    """
    point_count = len(points)
    pairs = []
    for k, segment in enumerate(segments):
        if (
            not isinstance(segment, (list, tuple))
            or len(segment) != 2
            or not all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in segment
            )
        ):
            raise LineError(
                "segment {0} is not a pair of integers".format(k)
            )
        pairs.append((int(segment[0]), int(segment[1])))

    issues = []

    # 1. Degenerate segments, then 2. out-of-bounds indices. Bounds are
    # checked first internally so points can be indexed safely, but the
    # messages are emitted in the documented order.
    bounds_issues = []
    valid = []
    for k, (i, j) in enumerate(pairs):
        i_ok = 0 <= i < point_count
        j_ok = 0 <= j < point_count
        if not i_ok or not j_ok:
            bad = i if not i_ok else j
            bounds_issues.append(
                "segment {0} index out of bounds: {1}".format(k, bad)
            )
            valid.append(False)
            continue
        valid.append(True)
        if math.hypot(
            points[i][0] - points[j][0], points[i][1] - points[j][1]
        ) < EPS:
            issues.append("segment {0} is degenerate".format(k))

    issues.extend(bounds_issues)

    # 3. Self-intersections between non-adjacent, in-bounds segments.
    for a in range(len(pairs)):
        if not valid[a]:
            continue
        i1, j1 = pairs[a]
        for b in range(a + 1, len(pairs)):
            if not valid[b]:
                continue
            i2, j2 = pairs[b]
            if i1 in (i2, j2) or j1 in (i2, j2):
                continue
            if _properly_cross(
                points[i1], points[j1], points[i2], points[j2]
            ):
                issues.append(
                    "segments {0} and {1} intersect".format(a, b)
                )

    return issues


# ---------------------------------------------------------------------------
# Batch tessellation
# ---------------------------------------------------------------------------


def _vertex_normals(pts):
    """Offsets (unit-ish normals) at each vertex of a cleaned polyline.

    Interior vertices use the miter direction, capped at
    ``_BATCH_MITER_LIMIT`` so sharp corners cannot create unbounded spikes.
    """
    count = len(pts)
    normals = []
    for i in range(count):
        if i == 0:
            normals.append(_left_normal(_normalize(_sub(pts[1], pts[0]))))
        elif i == count - 1:
            normals.append(
                _left_normal(_normalize(_sub(pts[i], pts[i - 1])))
            )
        else:
            d_in = _normalize(_sub(pts[i], pts[i - 1]))
            d_out = _normalize(_sub(pts[i + 1], pts[i]))
            n_in = _left_normal(d_in)
            n_out = _left_normal(d_out)
            dot = max(-1.0, min(1.0, d_in[0] * d_out[0] + d_in[1] * d_out[1]))
            denom = 1.0 + dot
            if denom <= 1e-9:
                # Path doubles back; choose one side rather than dividing by
                # an (almost) zero denominator.
                normals.append(n_out)
                continue
            mx = (n_in[0] + n_out[0]) / denom
            my = (n_in[1] + n_out[1]) / denom
            magnitude = math.hypot(mx, my)
            if magnitude > _BATCH_MITER_LIMIT:
                scale = _BATCH_MITER_LIMIT / magnitude
                mx *= scale
                my *= scale
            normals.append((mx, my))
    return normals


def tessellate_batch(polylines, config):
    """Tessellate several polylines as independent stroke ribbons.

    Each polyline is a list of ``(x, y)`` points. Consecutive duplicate
    points (distance < ``EPS``) are dropped before tessellation; a polyline
    with fewer than two distinct points is invalid. Vertex ranges are
    independent: the segments of one polyline may only index that polyline's
    points.

    ``config`` keys:

    * ``width`` (float > 0): stroke width in pixels.
    * ``tolerance`` (float > 0): flattening tolerance in pixels.

    Returns:
        dict: ``{"points": [...], "segments": [[i, j], ...],
        "polylines": [{"point_start": int, "point_count": int,
        "segment_start": int, "segment_count": int}, ...]}``.
        The per-polyline metadata appears in input order and describes slices
        of the global ``points`` / ``segments`` lists. Empty input returns
        ``{"points": [], "segments": [], "polylines": []}``.

    The operation is transactional: if any polyline is invalid (fewer than
    two distinct points, non-finite coordinates) or the config is invalid,
    the whole batch raises :class:`LineError` and no partial result is
    returned.

    Args:
        polylines: List of polylines, each a list of ``(x, y)`` points.
        config: Mapping with ``width`` and ``tolerance`` in pixels.

    Returns:
        dict: Combined points, segments and per-polyline metadata.

    Raises:
        LineError: If ``width <= 0``, ``tolerance <= 0``, or any polyline is
            invalid.
    """
    try:
        width = float(config["width"])
        tolerance = float(config["tolerance"])
    except (KeyError, TypeError, ValueError):
        raise LineError("config must provide numeric width and tolerance")

    if not math.isfinite(width) or not width > 0:
        raise LineError("width must be greater than zero")
    if not math.isfinite(tolerance) or not tolerance > 0:
        raise LineError("tolerance must be greater than zero")

    # Validate and clean every polyline before producing any output, so the
    # whole batch is atomic.
    cleaned = []
    for index, polyline in enumerate(polylines):
        try:
            raw_points = list(polyline)
        except TypeError:
            raise LineError("polyline {0} is not iterable".format(index))
        pts = []
        for point in raw_points:
            try:
                x = float(point[0])
                y = float(point[1])
            except (TypeError, ValueError, IndexError):
                raise LineError(
                    "polyline {0} has an invalid point".format(index)
                )
            if not (math.isfinite(x) and math.isfinite(y)):
                raise LineError(
                    "polyline {0} has non-finite coordinates".format(index)
                )
            if not pts or math.hypot(
                x - pts[-1][0], y - pts[-1][1]
            ) >= EPS:
                pts.append((x, y))
        if len(pts) < 2:
            raise LineError(
                "polyline {0} needs at least two distinct points".format(
                    index
                )
            )
        cleaned.append(pts)

    points = []
    segments = []
    metadata = []
    hw = width / 2.0

    for pts in cleaned:
        point_start = len(points)
        segment_start = len(segments)
        normals = _vertex_normals(pts)

        for (px, py), (nx, ny) in zip(pts, normals):
            points.append((px + nx * hw, py + ny * hw))
            points.append((px - nx * hw, py - ny * hw))

        count = len(pts)
        # Rungs at each sample.
        for i in range(count):
            segments.append([point_start + 2 * i, point_start + 2 * i + 1])
        # Rails between consecutive samples.
        for i in range(count - 1):
            segments.append(
                [point_start + 2 * i, point_start + 2 * i + 2]
            )
            segments.append(
                [point_start + 2 * i + 1, point_start + 2 * i + 3]
            )

        metadata.append(
            {
                "point_start": point_start,
                "point_count": 2 * count,
                "segment_start": segment_start,
                "segment_count": len(segments) - segment_start,
            }
        )

    return {
        "points": points,
        "segments": segments,
        "polylines": metadata,
    }
