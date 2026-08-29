
from __future__ import annotations
from collections import deque
import time
import numpy as np
from depth_utils import (get_depth_in_region, get_depth_with_mask, depth_to_meters,
                         estimate_distance, metric_depth_in_region)

NOVA_CLASSES = [
    "bench",
    "bicycle",
    "bus",
    "car",
    "chair",
    "computer",
    "curb",
    "door",
    "person",
    "pot-plant",
    "pothole",
    "ramp",
    "speed-breaker",
    "stairs",
    "table",
    "truck",
    "water dispenser",
    "white-board",
]

GROUND_HAZARD   = {"stairs", "pothole", "curb", "speed-breaker", "ramp"}
SAFETY_CRITICAL = {
    "pole", "fence", "wall",
    "ramp", "stairs", "pothole", "speed-breaker", "curb",
    "car", "bicycle", "bus", "truck",
}
NAVIGATION      = {"person", "door", "bench"}
OBSTACLE        = {"chair", "empty-chair", "occupied-chair", "table", "sofa",
                   "tree", "pot-plant", "vegitation"}
CONTEXT_ONLY    = {"computer", "projector", "white-board", "fire-extinguisher",
                   "water dispenser"}

_MASK_DEPTH_CLASSES = GROUND_HAZARD | {"pole", "fence", "tree", "bicycle"}

CONFIDENCE_THRESHOLDS: dict[str, float] = {
    **{c: 0.25 for c in SAFETY_CRITICAL},
    **{c: 0.25 for c in NAVIGATION},
    **{c: 0.25 for c in OBSTACLE},
    **{c: 0.40 for c in CONTEXT_ONLY},
    "ramp":          0.20,
    "curb":          0.20,
    "pothole":       0.30,
    "stairs":        0.20,
    "speed-breaker": 0.20,
    "bus":           0.60,
    "car":           0.40,
    "truck":         2.0,
    "bicycle":       2.0,
    "curb":          2.0,
    "pot-plant":     0.40,
    "sofa":          0.35,
    "table":         0.20,
    "chair":         0.20,
    "bench":         0.20,
}
DEFAULT_THRESHOLD = 0.25
MIN_FLOOR         = 0.20

PRIORITY: dict[str, int] = {
    "stairs": 1, "pothole": 1, "ramp": 1, "curb": 1, "speed-breaker": 1,
    "person": 2, "car": 2, "bus": 2, "bicycle": 2, "truck": 2,
    "door": 3, "fence": 3, "pole": 3, "tree": 3, "wall": 3,
    "unknown obstacle": 2,
}
DEFAULT_PRIORITY = 4

DANGER_DISTANCE  = 1.5
WARNING_DISTANCE = 3.0
CAUTION_DISTANCE = 5.0

MAX_ALERTS = 3

_pstats = {"calls": 0, "raw": 0, "passed": 0, "dedup_removed": 0, "by_class": {}}

_astats = {"calls": 0, "by_priority": {}}

_TEMPORAL_FILTER_CLASSES: frozenset[str] = frozenset(OBSTACLE | CONTEXT_ONLY)

TEMPORAL_WINDOW   = 4

HIGH_CONF_BYPASS               = 0.55
CONFIDENCE_ACCUMULATION_THRESHOLD = 0.90

_temporal_history: dict[str, deque] = {}

_last_path:       str   = "CONTINUE_FORWARD"
_last_path_time:  float = 0.0
DIRECTION_COMMIT_S = 1.5

_unknown_candidate_buffer: dict[str, int] = {}
_last_unknown_results: list[dict] = []

_known_bbox_history: list[list] = []
KNOWN_BBOX_HISTORY_FRAMES = 3

SCAN_BAND_TOP      = 0.15
SCAN_BAND_BOTTOM   = 0.80
SCAN_COLS          = 32
FLOOR_FIT_TOP      = 0.55
DEV_MARGIN         = 0.075
ABS_CLOSE          = 0.85
MIN_OBST_ROWS_FRAC = 0.18
FRONTAL_WALL_FRAC  = 0.60

UNKNOWN_CLOSE_CONFIRM_FRM   = 3
UNKNOWN_CAUTION_CONFIRM_FRM = 4

UNKNOWN_EMIT_COOLDOWN_S = 3.0
_ZONE_RANK = {"MEDIUM": 0, "CLOSE": 1, "CRITICAL": 2}
_unknown_last_emit: dict[str, tuple[float, str]] = {}


def update_temporal_history(detections: list[dict]) -> None:
    best_conf: dict[str, float] = {}
    for d in detections:
        cls = d["class"]
        if cls in _TEMPORAL_FILTER_CLASSES:
            best_conf[cls] = max(best_conf.get(cls, 0.0), d["confidence"])
    for cls in _TEMPORAL_FILTER_CLASSES:
        if cls not in _temporal_history:
            _temporal_history[cls] = deque(maxlen=TEMPORAL_WINDOW)
        _temporal_history[cls].append(best_conf.get(cls, 0.0))


def temporal_filter(detections: list[dict]) -> list[dict]:
    result = []
    for det in detections:
        cls = det["class"]
        if cls not in _TEMPORAL_FILTER_CLASSES:
            result.append(det)
            continue
        if det["confidence"] >= HIGH_CONF_BYPASS:
            result.append(det)
            continue
        history = _temporal_history.get(cls)
        if history and sum(history) >= CONFIDENCE_ACCUMULATION_THRESHOLD:
            result.append(det)
    return result


def reset_session_state() -> None:
    global _last_path, _last_path_time, _unknown_candidate_buffer, _last_unknown_results
    global _known_bbox_history
    _temporal_history.clear()
    _last_path = "CONTINUE_FORWARD"
    _last_path_time = 0.0
    _unknown_candidate_buffer.clear()
    _last_unknown_results = []
    _unknown_last_emit.clear()
    _known_bbox_history = []



def parse_detections(yolo_result, frame_h: int, frame_w: int,
                     depth_map: np.ndarray,
                     masks_resized: "np.ndarray | None" = None) -> list[dict]:
    detections = []
    boxes = yolo_result.boxes

    if boxes is None or len(boxes) == 0:
        _pstats["calls"] += 1
        return detections

    xyxyn = boxes.xyxyn.cpu().numpy()
    confs  = boxes.conf.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy().astype(int)

    n_raw = len(xyxyn)
    _pstats["calls"] += 1
    _pstats["raw"] += n_raw

    for i in range(len(xyxyn)):
        cls_id = int(cls_ids[i])
        if cls_id < 0 or cls_id >= len(NOVA_CLASSES):
            continue
        class_name = NOVA_CLASSES[cls_id]
        conf = float(confs[i])
        threshold = max(CONFIDENCE_THRESHOLDS.get(class_name, DEFAULT_THRESHOLD), MIN_FLOOR)
        if conf < threshold:
            continue

        x1, y1, x2, y2 = xyxyn[i].tolist()
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(1.0, x2), min(1.0, y2)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        bbox_h = y2 - y1

        if class_name == "ramp":
            h_dm, w_dm = depth_map.shape
            rx1 = max(0, int(x1 * w_dm)); ry1 = max(0, int(y1 * h_dm))
            rx2 = min(w_dm, int(x2 * w_dm)); ry2 = min(h_dm, int(y2 * h_dm))
            if rx2 > rx1 and ry2 > ry1:
                ramp_region = depth_map[ry1:ry2, rx1:rx2]
                scene_range = float(depth_map.max() - depth_map.min()) + 1e-6
                norm_std = float(np.std(ramp_region)) / scene_range
                if norm_std < 0.06:
                    print(f"[RAMP_GUARD] suppressed flat ramp (norm_std={norm_std:.3f})")
                    continue

        mask_i = None
        if (masks_resized is not None
                and i < len(masks_resized)
                and class_name in _MASK_DEPTH_CLASSES):
            mask_i = masks_resized[i]
            rel_depth = get_depth_with_mask(depth_map, mask_i, [x1, y1, x2, y2])
        else:
            rel_depth = get_depth_in_region(depth_map, [x1, y1, x2, y2])

        direction = classify_direction(cx)
        metric_dist = metric_depth_in_region([x1, y1, x2, y2], mask_i)
        dist_m    = depth_to_meters(rel_depth, class_name, bbox_h, direction, metric_dist)
        zone      = classify_zone(dist_m)

        detections.append({
            "class":      class_name,
            "confidence": round(conf, 4),
            "bbox":       [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)],
            "cx":         cx,
            "cy":         cy,
            "direction":  direction,
            "distance_m": round(dist_m, 3),
            "zone":       zone,
        })
        _pstats["by_class"][class_name] = _pstats["by_class"].get(class_name, 0) + 1

    _pstats["passed"] += len(detections)

    if _pstats["calls"] % 100 == 0:
        total_raw    = _pstats["raw"]
        total_passed = _pstats["passed"]
        pass_rate    = total_passed / total_raw * 100 if total_raw > 0 else 0
        top_cls = sorted(_pstats["by_class"].items(), key=lambda x: -x[1])[:5]
        cls_str = " ".join(f"{c}:{n}" for c, n in top_cls) if top_cls else "none"
        print(
            f"[ALERT_STATS:call{_pstats['calls']}] "
            f"raw={total_raw} passed={total_passed} ({pass_rate:.1f}%) "
            f"dedup_removed={_pstats['dedup_removed']} | top: {cls_str}"
        )
        _pstats.update({"raw": 0, "passed": 0, "dedup_removed": 0, "by_class": {}})

    return detections



def dedup_same_class(detections: list[dict]) -> list[dict]:
   
    sorted_dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    kept: list[dict] = []
    for det in sorted_dets:
        duplicate = any(
            k["class"] == det["class"] and _bbox_iou(k["bbox"], det["bbox"]) > 0.30
            for k in kept
        )
        if not duplicate:
            kept.append(det)
    removed = len(detections) - len(kept)
    if removed > 0:
        _pstats["dedup_removed"] += removed
    return kept



def filter_ground_hazard_positions(detections: list[dict]) -> list[dict]:
    
    result = []
    for det in detections:
        if det["class"] not in GROUND_HAZARD:
            result.append(det)
            continue
        if det["class"] == "ramp" and det["confidence"] < 0.20:
            continue
        bbox = det["bbox"]
        center_y = (bbox[1] + bbox[3]) / 2.0
        width    = bbox[2] - bbox[0]
        if center_y > 0.35 and width < 0.85:
            result.append(det)
    return result



def classify_direction(cx_norm: float) -> str:
    
    if cx_norm < 0.15:
        return "far_left"
    if cx_norm < 0.35:
        return "left"
    if cx_norm < 0.65:
        return "center"
    if cx_norm < 0.85:
        return "right"
    return "far_right"


def classify_zone(distance_m: float) -> str:

    if distance_m < DANGER_DISTANCE:
        return "CRITICAL"
    if distance_m < WARNING_DISTANCE:
        return "CLOSE"
    if distance_m < CAUTION_DISTANCE:
        return "MEDIUM"
    return "FAR"



def generate_alerts(detections: list[dict]) -> list[dict]:
   
    alerts = []
    all_classes = frozenset(d["class"] for d in detections)
    for det in detections:
        zone  = det["zone"]
        cls   = det["class"]

        if zone == "FAR":
            continue
        if zone == "MEDIUM" and cls in CONTEXT_ONLY:
            continue

        priority = PRIORITY.get(cls, DEFAULT_PRIORITY)
        if zone == "CRITICAL" and (cls in GROUND_HAZARD or cls in SAFETY_CRITICAL
                                   or cls == "unknown obstacle"):
            priority = 1

        haptic  = select_haptic(zone)
        message = generate_alert_message(cls, det["direction"], det["distance_m"], zone, all_classes)

        alerts.append({
            "priority":   priority,
            "message":    message,
            "class":      cls,
            "direction":  det["direction"],
            "distance_m": det["distance_m"],
            "haptic":     haptic,
        })

    alerts.sort(key=lambda a: (a["priority"], a["class"] == "unknown obstacle"))
    result = alerts[:MAX_ALERTS]

    _astats["calls"] += 1
    for a in result:
        p = a["priority"]
        _astats["by_priority"][p] = _astats["by_priority"].get(p, 0) + 1
    if _astats["calls"] % 100 == 0:
        d = _astats["by_priority"]
        print(
            f"[ALERT_PRI:call{_astats['calls']}] "
            f"P1={d.get(1,0)} P2={d.get(2,0)} P3={d.get(3,0)} P4={d.get(4,0)}"
        )
        _astats["by_priority"].clear()

    return result


def select_haptic(zone: str) -> str:
    return {
        "CRITICAL": "BUZZ",
        "CLOSE":    "DOUBLE_TAP",
        "MEDIUM":   "PING",
        "FAR":      "PING",
    }.get(zone, "PING")


_DIR_CLOCK = {
    "far_left":  "9 o'clock",
    "left":      "10 o'clock",
    "center":    "12 o'clock",
    "right":     "2 o'clock",
    "far_right": "3 o'clock",
}

_CLASS_DISPLAY = {
    "vegitation":        "vegetation",
    "speed-breaker":     "speed breaker",
    "fire-extinguisher": "fire extinguisher",
    "empty-chair":       "empty chair",
    "occupied-chair":    "occupied chair",
    "white-board":       "whiteboard",
    "water dispenser":   "water dispenser",
    "pot-plant":         "pot plant",
}


def _display_name(cls: str) -> str:
    return _CLASS_DISPLAY.get(cls, cls.replace("-", " ").replace("_", " "))


def generate_alert_message(class_name: str, direction: str,
                            distance_m: float, zone: str,
                            all_classes: "frozenset | None" = None) -> str:
    """
    Human-readable alert message. Mirrors Android AlertPhraseGenerator.
    Direction-first format so VIP can orient before the phrase completes.
    Distance ≤ 3m → steps (1 step ≈ 0.75m); >3m → X.X metres.
    all_classes: set of all detected class names this frame (used for stairs hint).
    """
    dir_spoken = _DIR_CLOCK.get(direction, "12 o'clock")
    obj = _display_name(class_name)

    if distance_m <= WARNING_DISTANCE:
        steps = max(1, round(distance_m / 0.75))
        dist_str = f"{steps} step" if steps == 1 else f"{steps} steps"
    else:
        dist_str = f"{distance_m:.1f} metres"

    if class_name == "stairs":
        hint = "step down" if (all_classes and "ramp" in all_classes) else "mind the steps"
        if zone == "CRITICAL":
            return f"{dir_spoken}. Stop! stairs, {hint}. {dist_str}."
        if zone == "CLOSE":
            return f"{dir_spoken}. Stairs, {hint}. {dist_str}."
        return f"{dir_spoken}. Stairs ahead, {hint}. {dist_str}."

    if class_name == "unknown obstacle":
        if zone == "CRITICAL":
            return f"{dir_spoken}. Stop! unknown object ahead."
        if zone == "CLOSE":
            return f"{dir_spoken}. Unknown object, slow down."
        return f"{dir_spoken}. Unknown object ahead."

    if class_name in NAVIGATION and zone == "CRITICAL":
        return f"{dir_spoken}. {obj}. {dist_str}."
    if zone == "CRITICAL":
        return f"{dir_spoken}. Stop! {obj}. {dist_str}."
    if zone == "CLOSE":
        return f"{dir_spoken}. {obj}. {dist_str}."
    return f"{dir_spoken}. {obj} ahead. {dist_str}."



def calculate_free_path(detections: list[dict], depth_zones: dict) -> str:
    """
    Replicates Android FreePathCalculator.calculate().
    Path blocking driven by YOLO detections only.

    depth_zones are NOT used for path blocking: run_depth() (depth_utils.py) takes
    the DA3-Large Metric / DAV2-Metric-Indoor-Large output and renormalises it to
    [0,1] per-frame (max always = 1.0) for the zone/geometry map, so absolute
    thresholds over depth_zones would still produce false STOP_ALL_BLOCKED on
    almost every frame — same failure mode as raw MiDaS, just for a different
    reason now that the underlying model is metric.
    Path blocking instead uses each detection's own `zone` field (set via
    classify_zone() below), which comes from true metric distance
    (depth_to_meters() + metric_depth_in_region()/_metric_depth_map) — NOT from
    depth_zones. depth_zones are kept in the response for client-side
    visualisation only.

    Returns: CONTINUE_FORWARD / SLOW_DOWN / MOVE_LEFT / MOVE_RIGHT / STOP_ALL_BLOCKED
    """
    global _last_path, _last_path_time

    status = {"left": "CLEAR", "center": "CLEAR", "right": "CLEAR"}

    _DIR_BUCKET = {"far_left": "left", "left": "left",
                   "center": "center",
                   "right": "right", "far_right": "right"}

    for det in detections:
        d    = _DIR_BUCKET.get(det["direction"], "center")
        zone = det["zone"]
        if zone == "FAR":
            continue
        new_status = "BLOCKED" if zone == "CRITICAL" else "CAUTION"
        if _status_rank(new_status) > _status_rank(status[d]):
            status[d] = new_status

    c, l, r = status["center"], status["left"], status["right"]

    if c == "CLEAR":
        raw = "CONTINUE_FORWARD"
    elif c == "CAUTION":
        raw = "SLOW_DOWN"
    elif l == "CLEAR":
        raw = "MOVE_LEFT"
    elif r == "CLEAR":
        raw = "MOVE_RIGHT"
    elif l != "BLOCKED" or r != "BLOCKED":
        raw = "SLOW_DOWN"
    else:
        raw = "STOP_ALL_BLOCKED"

    now = time.monotonic()
    is_lateral_flip = (
        (raw == "MOVE_LEFT" and _last_path == "MOVE_RIGHT") or
        (raw == "MOVE_RIGHT" and _last_path == "MOVE_LEFT")
    )
    if is_lateral_flip and now - _last_path_time < DIRECTION_COMMIT_S:
        raw = _last_path

    if raw != _last_path:
        _last_path = raw
        _last_path_time = now

    return raw


def _max_true_run(mask: np.ndarray) -> int:
    """Length of the longest run of consecutive True values in a 1-D boolean array."""
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        if cur > best:
            best = cur
    return best


def _span_overlap(a1: float, a2: float, b1: float, b2: float) -> float:
    """Fraction of span [a1,a2] that lies inside [b1,b2]."""
    inter = max(0.0, min(a2, b2) - max(a1, b1))
    span = a2 - a1
    return inter / span if span > 0 else 0.0


def _merge_strips_to_spans(strip_vals: list, edges: np.ndarray, w: int, kind: str) -> list[tuple]:
    """
    Merge adjacent flagged strips into spans: [(left_frac, right_frac, value, frontal, kind), ...].
    Reports the most severe (max = closest) value across merged strips.
    """
    spans: list[tuple] = []
    ci = 0
    n = len(strip_vals)
    while ci < n:
        if strip_vals[ci] is None:
            ci += 1
            continue
        cj = ci
        vals = []
        while cj < n and strip_vals[cj] is not None:
            vals.append(strip_vals[cj])
            cj += 1
        l = float(edges[ci]) / w
        r = float(edges[cj]) / w
        value = max(vals)
        spans.append((l, r, value, (r - l) >= FRONTAL_WALL_FRAC, kind))
        ci = cj
    return spans


def _scan_free_space(depth_map: np.ndarray) -> list[tuple]:
    
    h, w = depth_map.shape
    rows = np.arange(h, dtype=np.float32)
    row_med = np.median(depth_map, axis=1)
    fit_lo = int(FLOOR_FIT_TOP * h)
    a, b = np.polyfit(rows[fit_lo:], row_med[fit_lo:], 1)
    pred_floor = a * rows + b

    br1, br2 = int(SCAN_BAND_TOP * h), int(SCAN_BAND_BOTTOM * h)
    band_rows = max(1, br2 - br1)
    pf_band = pred_floor[br1:br2].reshape(-1, 1)

    edges = np.linspace(0, w, SCAN_COLS + 1).astype(int)

    strip_obst: list = [None] * SCAN_COLS
    for ci in range(SCAN_COLS):
        c1, c2 = int(edges[ci]), int(edges[ci + 1])
        if c2 <= c1:
            continue

        band = depth_map[br1:br2, c1:c2]
        obst = (band > pf_band + DEV_MARGIN) | (band >= ABS_CLOSE)
        rows_hit = obst.mean(axis=1) >= 0.5
        if _max_true_run(rows_hit) >= MIN_OBST_ROWS_FRAC * band_rows:
            vals = band[obst]
            strip_obst[ci] = float(np.percentile(vals, 75)) if vals.size else float(band.max())

    return _merge_strips_to_spans(strip_obst, edges, w, "obstacle")


def find_unknown_obstacles(depth_map, known_detections: list[dict],
                           fresh: bool = True, raw_known_bboxes: list | None = None) -> list[dict]:
    
    global _unknown_candidate_buffer, _last_unknown_results, _known_bbox_history

    if not fresh:
        return _last_unknown_results

    spans = _scan_free_space(depth_map)

    current_known = [d["bbox"] for d in known_detections]
    if raw_known_bboxes:
        current_known = current_known + list(raw_known_bboxes)
    _known_bbox_history.append(current_known)
    if len(_known_bbox_history) > KNOWN_BBOX_HISTORY_FRAMES:
        _known_bbox_history = _known_bbox_history[-KNOWN_BBOX_HISTORY_FRAMES:]
    known_bboxes = [kb for frame_boxes in _known_bbox_history for kb in frame_boxes]

    new_buffer: dict[str, int] = {}
    results: list[dict] = []

    for (l, r, value, frontal, kind) in spans:
        cx = (l + r) / 2.0

        if any(_span_overlap(l, r, kb[0], kb[2]) > 0.5 or _span_overlap(kb[0], kb[2], l, r) > 0.5
               for kb in known_bboxes):
            continue

        band_top = SCAN_BAND_TOP

        span_bbox = [l, band_top, r, SCAN_BAND_BOTTOM]
        metric_dist = metric_depth_in_region(span_bbox)

        centered = 0.30 <= cx <= 0.70
        if metric_dist is not None:
            zone = classify_zone(metric_dist)
            if metric_dist >= WARNING_DISTANCE:
                continue
        elif frontal and centered:
            zone = "CRITICAL"
        elif value >= 0.85:
            zone = "CRITICAL"
        elif value >= 0.65:
            zone = "CLOSE"
        else:
            zone = "MEDIUM"

        direction = classify_direction(cx)

        persist_key = "FRONTAL" if frontal else direction

        frames_seen = _unknown_candidate_buffer.get(persist_key, 0) + 1
        new_buffer[persist_key] = max(new_buffer.get(persist_key, 0), frames_seen)

        required = UNKNOWN_CLOSE_CONFIRM_FRM if zone in ("CRITICAL", "CLOSE") else UNKNOWN_CAUTION_CONFIRM_FRM
        if frames_seen < required:
            continue

        now_t = time.monotonic()
        last_emit_t, last_zone = _unknown_last_emit.get(persist_key, (0.0, None))
        escalated = last_zone is None or _ZONE_RANK.get(zone, 0) > _ZONE_RANK.get(last_zone, 0)
        if not escalated and (now_t - last_emit_t) < UNKNOWN_EMIT_COOLDOWN_S:
            continue
        _unknown_last_emit[persist_key] = (now_t, zone)

        dist_m = metric_dist if metric_dist is not None else estimate_distance(value)
        results.append({
            "class":      "unknown obstacle",
            "confidence": 0.50,
            "bbox":       [round(l, 3), round(band_top, 3),
                           round(r, 3), round(SCAN_BAND_BOTTOM, 3)],
            "cx":         cx,
            "cy":         (band_top + SCAN_BAND_BOTTOM) / 2,
            "direction":  direction,
            "distance_m": round(dist_m, 3),
            "zone":       zone,
        })

    _unknown_candidate_buffer = new_buffer
    _last_unknown_results = results
    return results


def _status_rank(s: str) -> int:
    return {"CLEAR": 0, "CAUTION": 1, "BLOCKED": 2}.get(s, 0)



def _bbox_iou(a: list, b: list) -> float:
    """[x1,y1,x2,y2] normalized IoU."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0
    i_area = (ix2 - ix1) * (iy2 - iy1)
    a_area = (a[2] - a[0]) * (a[3] - a[1])
    b_area = (b[2] - b[0]) * (b[3] - b[1])
    u_area = a_area + b_area - i_area
    return i_area / u_area if u_area > 0 else 0.0


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    H, W = 240, 320

    floor = np.tile((np.arange(H) / (H - 1)).reshape(H, 1), (1, W)).astype(np.float32)
    assert _scan_free_space(floor) == [], "clean floor gradient must yield no obstacle"

    boxed = floor.copy()
    boxed[int(0.30 * H):int(0.70 * H), int(0.40 * W):int(0.60 * W)] = 0.97
    spans = _scan_free_space(boxed)
    assert spans, "planted near box must be detected"
    cx = (spans[0][0] + spans[0][1]) / 2
    assert 0.30 < cx < 0.70, f"box should sit near centre, got cx={cx:.2f}"

    mid = floor.copy()
    mid[int(0.40 * H):int(0.60 * H), int(0.42 * W):int(0.58 * W)] = 0.60
    mspans = _scan_free_space(mid)
    assert mspans, "mid-range obstacle protruding above the floor line must be detected"

    wall = np.full((H, W), 0.90, np.float32)
    wspans = _scan_free_space(wall)
    assert wspans and wspans[0][3] is True, "uniform near wall must flag a frontal span"
    assert wspans[0][4] == "obstacle", "uniform near wall must be kind='obstacle'"

    assert _max_true_run(np.array([0, 1, 1, 0, 1, 1, 1, 0], bool)) == 3

    print("alert_logic self-check passed:",
          f"clean=ok box_cx={cx:.2f} wall_frontal={wspans[0][3]}")
