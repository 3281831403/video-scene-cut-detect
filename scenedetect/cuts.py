"""Scene-cut detection primitives for a small grayscale video pipeline.

Shared contract
---------------
* A frame is a rectangular two-dimensional grayscale image represented as
  ``list[list[float]]``.  Pixel values live in ``[0.0, 1.0]`` and every
  frame in one clip has the same shape.
* A ``threshold`` is a float in ``[0.0, 1.0]``.
* Floating-point results are compared with an absolute tolerance of ``1e-6``.
* Invalid arguments raise :class:`SceneError`.
"""

import math
import numbers

# Bin count used by :func:`detect_cuts` (and the default for
# :func:`detect_batch`).  Histograms are always normalized, so the exact
# count only controls colour resolution.
HISTOGRAM_BINS = 16


class SceneError(ValueError):
    """Raised when a frame, histogram, shot list, or configuration is invalid."""


def _is_number(value) -> bool:
    """Return True for real, non-boolean numbers."""
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _require_int(value, name: str, minimum: int) -> int:
    """Validate an integer argument that must be at least ``minimum``."""
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise SceneError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


def _require_threshold(threshold) -> float:
    """Validate a distance threshold in ``[0.0, 1.0]``."""
    if not _is_number(threshold) or not math.isfinite(threshold):
        raise SceneError(f"threshold must be a finite number, got {threshold!r}")
    value = float(threshold)
    if not 0.0 <= value <= 1.0:
        raise SceneError(f"threshold must be in [0.0, 1.0], got {value!r}")
    return value


def _validate_frame(frame) -> tuple[int, int]:
    """Validate one frame and return its ``(rows, columns)`` shape."""
    if not isinstance(frame, (list, tuple)) or len(frame) == 0:
        raise SceneError("frame must be a non-empty rectangular 2D image")

    columns = None
    for row in frame:
        if not isinstance(row, (list, tuple)) or len(row) == 0:
            raise SceneError("frame rows must be non-empty sequences")
        if columns is None:
            columns = len(row)
        elif len(row) != columns:
            raise SceneError("frame must be rectangular (ragged rows found)")
        for pixel in row:
            if not _is_number(pixel) or not math.isfinite(pixel):
                raise SceneError(f"pixel must be a finite number, got {pixel!r}")
            if not 0.0 <= float(pixel) <= 1.0:
                raise SceneError(f"pixel must be in [0.0, 1.0], got {pixel!r}")

    return len(frame), columns


def _validate_clip(frames) -> tuple[int, int]:
    """Validate every frame in a clip and return the common shape."""
    if not isinstance(frames, (list, tuple)):
        raise SceneError("clip must be a list of frames")

    shape = None
    for frame in frames:
        frame_shape = _validate_frame(frame)
        if shape is None:
            shape = frame_shape
        elif frame_shape != shape:
            raise SceneError("all frames in a clip must share the same shape")

    return shape if shape is not None else (0, 0)


def _validate_histogram(values, name: str) -> float:
    """Validate a normalized histogram and return its total."""
    if not isinstance(values, (list, tuple)) or len(values) == 0:
        raise SceneError(f"histogram {name} must be a non-empty sequence")

    total = 0.0
    for value in values:
        if not _is_number(value) or not math.isfinite(value) or value < 0.0:
            raise SceneError(
                f"histogram {name} entries must be non-negative finite numbers"
            )
        total += float(value)

    if abs(total - 1.0) > 1e-6:
        raise SceneError(f"histogram {name} must sum to 1.0, got {total!r}")
    return total


def _histogram(frame, bins: int) -> list[float]:
    """Internal histogram implementation with an already-validated frame."""
    rows = len(frame)
    columns = len(frame[0])
    counts = [0.0] * bins

    for row in frame:
        for pixel in row:
            index = min(bins - 1, int(float(pixel) * bins))
            counts[index] += 1.0

    total = float(rows * columns)
    return [count / total for count in counts]


def histogram(frame: list[list[float]], bins: int) -> list[float]:
    """Return a normalized luminance histogram for ``frame``.

    The grayscale range ``[0.0, 1.0]`` is divided into ``bins`` equal-width
    bins.  A pixel ``v`` falls in ``min(bins - 1, int(v * bins))`` so that
    ``1.0`` belongs to the last bin.  Pixel counts are divided by the total
    number of pixels; consequently the returned list has length ``bins``,
    contains only non-negative values, and sums to ``1.0`` within ``1e-6``.
    Normalization is essential: a bright, large frame must not dominate a
    distance merely because it has more pixels.

    Raises :class:`SceneError` when ``frame`` is empty or ragged, a pixel is
    not a finite number in ``[0.0, 1.0]``, or ``bins`` is not a positive
    integer (in particular ``bins == 0``).
    """
    _validate_frame(frame)
    bins = _require_int(bins, "bins", 1)
    return _histogram(frame, bins)


def hist_diff(a: list[float], b: list[float]) -> float:
    """Return the normalized chi-square distance between two histograms.

    ``a`` and ``b`` must be normalized histograms of the same positive
    length, with non-negative finite entries and sums equal to ``1.0``
    (absolute tolerance ``1e-6``).  With terms where ``a_i + b_i == 0``
    contributing nothing, the distance is::

        sqrt(sum((a_i - b_i) ** 2 / (a_i + b_i)))

    clamped to ``[0.0, 1.0]``.  Thus identical histograms return ``0.0`` and
    two disjoint histograms (all black versus all white) return ``1.0``.
    The square root keeps the value on the same 0..1 scale as the configured
    threshold; returning the un-rooted sum makes thresholds incomparable.

    Raises :class:`SceneError` for malformed or mismatched histograms.
    """
    _validate_histogram(a, "a")
    _validate_histogram(b, "b")
    if len(a) != len(b):
        raise SceneError("histograms must have the same length")

    total = 0.0
    for left, right in zip(a, b):
        left = float(left)
        right = float(right)
        denominator = left + right
        if denominator == 0.0:
            continue
        difference = left - right
        total += (difference * difference) / denominator

    return min(1.0, max(0.0, math.sqrt(total)))


def _gradient_magnitudes(frame) -> list[list[float]]:
    """Forward-difference gradient magnitudes; borders contribute zero."""
    rows = len(frame)
    columns = len(frame[0])
    magnitudes = [[0.0] * columns for _ in range(rows)]

    for r in range(rows):
        for c in range(columns):
            value = float(frame[r][c])
            gx = float(frame[r][c + 1]) - value if c + 1 < columns else 0.0
            gy = float(frame[r + 1][c]) - value if r + 1 < rows else 0.0
            magnitudes[r][c] = math.hypot(gx, gy)

    return magnitudes


def edge_change(a: list[list[float]], b: list[list[float]]) -> float:
    """Return the mean absolute difference of the two gradient maps.

    For each pixel of a frame, the gradient magnitude is computed with
    forward differences::

        gx = f[r][c + 1] - f[r][c]
        gy = f[r + 1][c] - f[r][c]
        magnitude = sqrt(gx ** 2 + gy ** 2)

    Missing neighbours are treated as zero.  The function returns the mean
    absolute difference between the two magnitude maps, divided by
    ``sqrt(2.0)``, so the result is in ``[0.0, 1.0]``.  Identical frames and
    a global brightness shift that preserves every edge both return ``0.0``;
    moved or changed edges return a larger value.  Comparing edges in
    addition to colour prevents illumination changes from being reported as
    cuts while still capturing motion.

    Raises :class:`SceneError` for empty, ragged, or differently sized
    frames, or for non-finite pixels outside ``[0.0, 1.0]``.
    """
    shape_a = _validate_frame(a)
    shape_b = _validate_frame(b)
    if shape_a != shape_b:
        raise SceneError("frames must have the same shape for edge comparison")

    rows, columns = shape_a
    magnitudes_a = _gradient_magnitudes(a)
    magnitudes_b = _gradient_magnitudes(b)

    total = 0.0
    for r in range(rows):
        for c in range(columns):
            total += abs(magnitudes_a[r][c] - magnitudes_b[r][c])

    return (total / float(rows * columns)) / math.sqrt(2.0)


def _detect_cuts(
    frames: list[list[list[float]]],
    threshold: float,
    min_len: int,
    bins: int,
) -> list[int]:
    """Cut detection core shared by :func:`detect_cuts`/:func:`detect_batch`."""
    if len(frames) <= 1:
        return []

    # Histograms are computed for this clip alone.  Never cache them across
    # clips keyed by frame index: frames in different clips are unrelated.
    histograms = [_histogram(frame, bins) for frame in frames]

    cuts: list[int] = []
    previous_cut = 0
    for index in range(1, len(frames)):
        colour_score = hist_diff(histograms[index - 1], histograms[index])
        motion_score = edge_change(frames[index - 1], frames[index])
        score = 0.5 * colour_score + 0.5 * motion_score

        if score >= threshold and index - previous_cut >= min_len:
            cuts.append(index)
            previous_cut = index

    return cuts


def detect_cuts(
    frames: list[list[list[float]]],
    threshold: float,
    min_len: int,
) -> list[int]:
    """Return the frame indices at which a new shot starts.

    Frames are consecutive grayscale frames of equal shape.  For every
    boundary ``i`` (``1 <= i < len(frames)``) the candidate score is::

        0.5 * hist_diff(histogram(frames[i - 1]), histogram(frames[i]))
        + 0.5 * edge_change(frames[i - 1], frames[i])

    where the histograms use the module's fixed bin count.  A boundary is a
    cut when its score is at least ``threshold`` and it is at least
    ``min_len`` frames after the previous accepted cut (or after index 0 for
    the first cut).  This shortest-shot constraint prevents a one-second
    sequence of flashes from producing a cut at every frame.  Returned
    indices are strictly increasing.  A clip with zero or one frame returns
    ``[]``.

    Raises :class:`SceneError` for an invalid threshold, a non-positive
    ``min_len``, or an invalid clip/frame.
    """
    threshold = _require_threshold(threshold)
    min_len = _require_int(min_len, "min_len", 1)

    if not isinstance(frames, (list, tuple)):
        raise SceneError("frames must be a list of frames")
    _validate_clip(frames)

    return _detect_cuts(frames, threshold, min_len, HISTOGRAM_BINS)


def merge_short(shots: list[int], min_len: int) -> list[int]:
    """Merge shots shorter than ``min_len`` frames into their neighbour.

    ``shots`` is an increasing list of shot start indices whose first entry
    is ``0`` (an empty list is allowed).  Walking from left to right, a
    boundary is kept only when it is at least ``min_len`` frames after the
    previously kept boundary.  Every returned boundary is one of the input
    indices and the leading ``0`` is preserved: the function never invents
    an averaged boundary such as ``(2 + 5) / 2``, which would make cut points
    drift away from real frame boundaries.  When several short shots meet,
    the boundary of the longer merged run is retained.

    With ``shots=[0, 2, 5, 10]`` and ``min_len=4`` the result is
    ``[0, 5, 10]``: the short 0..2 shot is merged and the real boundary at 5
    is kept.

    Raises :class:`SceneError` for a non-positive ``min_len``, non-integer
    entries, a first entry other than ``0``, or a non-increasing list.
    """
    min_len = _require_int(min_len, "min_len", 1)

    if not isinstance(shots, (list, tuple)):
        raise SceneError("shots must be a list of integers")
    for shot in shots:
        if not isinstance(shot, int) or isinstance(shot, bool):
            raise SceneError(f"shot indices must be integers, got {shot!r}")

    if len(shots) == 0:
        return []
    if shots[0] != 0:
        raise SceneError("shots must start with index 0")

    for previous, current in zip(shots, shots[1:]):
        if current <= previous:
            raise SceneError("shots must be strictly increasing")

    kept = [0]
    for shot in shots[1:]:
        if shot - kept[-1] >= min_len:
            kept.append(shot)

    return kept


def fade_detect(diffs: list[float], window: int) -> list[tuple[int, int]]:
    """Detect gradual transitions from frame-to-frame differences.

    ``diffs`` is a sequence of non-negative change magnitudes.  A fade is a
    maximal contiguous run of strictly positive values.  Runs whose length is
    at least ``window`` are returned as inclusive ``(start, end)`` index
    pairs into ``diffs``; shorter runs are ignored.  The run is followed
    adaptively to its real end, so one long fade is reported as one interval
    rather than being chopped into fixed-size windows.  For example, a run of
    seven positive values at indices 1..7 is ``[(1, 7)]`` for any
    ``window <= 7``.

    Values equal to zero separate runs.  Raises :class:`SceneError` for an
    empty ``diffs`` list, negative/non-finite differences, or a non-positive
    ``window``.
    """
    if not isinstance(diffs, (list, tuple)) or len(diffs) == 0:
        raise SceneError("diffs must be a non-empty sequence")

    for difference in diffs:
        if not _is_number(difference) or not math.isfinite(difference):
            raise SceneError(f"differences must be finite numbers, got {difference!r}")
        if difference < 0.0:
            raise SceneError(f"differences must be non-negative, got {difference!r}")

    window = _require_int(window, "window", 1)

    runs: list[tuple[int, int]] = []
    start = None
    for index, difference in enumerate(diffs):
        if difference > 0.0:
            if start is None:
                start = index
        elif start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(diffs) - 1))

    return [(start, end) for start, end in runs if end - start + 1 >= window]


def validate_shots(shots: list[int], total: int) -> list[str]:
    """Validate shot start indices and return every detected problem.

    ``shots`` contains shot start indices in frame coordinates and normally
    starts with ``0``; ``total`` is the number of frames in the clip.  One
    human-readable string is appended per problem, in this order:

    * out of bounds: ``shot < 0`` or ``shot >= total`` (the message contains
      the offending index);
    * unordered: a strict increase is required, so duplicates and decreases
      are reported (the message contains both values);
    * too short: two valid consecutive starts less than two frames apart,
      or a final shot shorter than two frames.  The minimum shot length is
      fixed at two frames because this function has no ``min_len`` argument.

    Invalid entries are not used as the previous valid index, and negative
    gaps are reported only as unordered rather than also as short.  For
    ``shots=[0, 1, 99]`` and ``total=10`` there are two issues: the 0..1
    shot is too short and index 99 is out of bounds.  Returns an empty list
    when every shot is valid.

    Raises :class:`SceneError` if ``total`` is not a positive integer or
    ``shots`` is not a list of integers.
    """
    total = _require_int(total, "total", 1)

    if not isinstance(shots, (list, tuple)):
        raise SceneError("shots must be a list of integers")
    for shot in shots:
        if not isinstance(shot, int) or isinstance(shot, bool):
            raise SceneError(f"shot indices must be integers, got {shot!r}")

    out_of_bounds: list[str] = []
    unordered: list[str] = []
    too_short: list[str] = []

    previous_valid = None
    for shot in shots:
        if shot < 0 or shot >= total:
            out_of_bounds.append(
                f"shot index {shot} is out of bounds for a {total}-frame clip"
            )
            continue

        if previous_valid is not None and shot <= previous_valid:
            unordered.append(
                f"shot indices are not strictly increasing: "
                f"{previous_valid} followed by {shot}"
            )
            continue

        if previous_valid is not None and shot - previous_valid < 2:
            too_short.append(
                f"shot from {previous_valid} to {shot} is too short: "
                f"{shot - previous_valid} frame(s)"
            )

        previous_valid = shot

    if previous_valid is not None and total - previous_valid < 2:
        too_short.append(
            f"final shot starting at {previous_valid} is too short: "
            f"{total - previous_valid} frame(s)"
        )

    return out_of_bounds + unordered + too_short


def detect_batch(
    clips: list[list[list[list[float]]]],
    config: dict,
) -> dict:
    """Cut-detect a batch of clips atomically and independently.

    ``clips`` is a list of clips, each clip being a list of grayscale frames.
    ``config`` may contain ``threshold`` (default ``0.5``), ``min_len``
    (default ``1``), and ``bins`` (default ``16``).  The return value is::

        {"cuts": [cut_list_for_clip_0, cut_list_for_clip_1, ...],
         "count": len(clips)}

    Each clip is processed with its own histograms; histogram state must
    never be keyed only by frame position, because that would let one clip
    contaminate the next.  The order of ``clips`` and of their cut lists is
    preserved, and an empty batch returns ``{"cuts": [], "count": 0}``.

    Processing is atomic: all configuration, clips, and frames are validated
    before any result is produced.  If anything is invalid (including
    ``bins == 0``) the function raises :class:`SceneError` and returns no
    partial result, and it does not mutate the input clips.

    Raises :class:`SceneError` for malformed config/clips/frames or invalid
    values.
    """
    if not isinstance(config, dict):
        raise SceneError("config must be a dict")

    threshold = _require_threshold(config.get("threshold", 0.5))
    min_len = _require_int(config.get("min_len", 1), "min_len", 1)
    bins = _require_int(config.get("bins", HISTOGRAM_BINS), "bins", 1)

    if not isinstance(clips, (list, tuple)):
        raise SceneError("clips must be a list of clips")

    # Validate the entire batch (configuration, clips, and frames) before
    # computing anything, so an invalid late clip cannot leave a partial
    # result behind.
    for clip in clips:
        _validate_clip(clip)

    results = [
        _detect_cuts(clip, threshold, min_len, bins)
        for clip in clips
    ]

    return {"cuts": results, "count": len(clips)}
