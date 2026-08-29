import os
import time
import cv2
import numpy as np
import torch
from PIL import Image


_depth_model     = None
_depth_processor = None   
_device          = None
_depth_backend: "str | None" = None   


_depth_call_count   = 0   
_mask_call_count    = 0   
_mask_fallback_count = 0  



_last_dist_m: dict[tuple, float] = {}


_STATIC_CLASSES = frozenset({
    "wall", "door", "fence", "pole", "stairs", "ramp",
    "table", "bench", "sofa", "chair", "empty-chair", "occupied-chair",
    "white-board", "projector", "water dispenser", "fire-extinguisher",
})



_depth_skip_counter: int = 0
_last_depth_map: "np.ndarray | None" = None
_last_depth_was_fresh: bool = False   
DEPTH_SKIP_FRAMES: int = 3   


_metric_depth_map: "np.ndarray | None" = None
_metric_active: bool = False   
_metric_log_count: int = 0     

_MODEL_DA3_METRIC         = "depth-anything/da3metric-large"
_MODEL_DAV2_METRIC_INDOOR = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"


KNOWN_HEIGHTS = {
    "person":             1.70,
    "door":               2.10,
    "bench":              0.80,
    "car":                1.50,
    "bus":                3.20,
    "truck":              2.50,
    "bicycle":            1.00,
    "chair":              0.90,
    "empty-chair":        0.90,
    "occupied-chair":     0.90,
    "table":              0.75,
    "sofa":               0.85,
    "pole":               3.50,
    "fence":              1.20,
    "wall":               2.40,
    "white-board":        1.20,
    "stairs":             0.20,
    "ramp":               0.30,
    "speed-breaker":      0.15,
    "curb":               0.12,
    "pothole":            0.10,
    "tree":               3.00,
    "vegitation":         1.00,
    "pot-plant":          0.60,
    "computer":           0.50,
    "projector":          0.30,
    "fire-extinguisher":  0.60,
    "water dispenser":    1.50,
}



METRIC_HEIGHT_DIVERGENCE_M = 1.0



def load_depth_model(device: torch.device) -> None:
    global _depth_model, _depth_processor, _device, _metric_active, _depth_backend
    _device = device

    
    try:
        print(f"[depth_utils] Loading DA3-Large Metric ({_MODEL_DA3_METRIC}) ...")
        from depth_anything_3.api import DepthAnything3
        model = DepthAnything3.from_pretrained(_MODEL_DA3_METRIC)
        
        
        
        
        
        
        model = model.to(device).eval()

        
        
        
        
        if device.type == "cuda" and hasattr(model, "model"):
            try:
                import numpy as _np
                t_compile0 = time.perf_counter()
                model.model = torch.compile(model.model, mode="reduce-overhead")
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                dummy = (_np.random.rand(640, 480, 3) * 255).astype(_np.uint8)
                model.inference([dummy])   
                print(f"[depth_utils] torch.compile warmup done in {time.perf_counter()-t_compile0:.1f}s "
                      f"(dummy input shape={dummy.shape})")
            except Exception as e:
                print(f"[depth_utils] ⚠ torch.compile unavailable, staying eager: {e}")

        _depth_model = model
        _depth_backend = "da3"
        _metric_active = True
        vram_mb = torch.cuda.memory_allocated(0) / 1e6 if device.type == "cuda" else 0
        print(f"[depth_utils] DA3-Large Metric ready (metric_active=True). VRAM: {vram_mb:.0f}MB")
        return
    except Exception as e:
        print(f"[depth_utils] ⚠ DA3-Large Metric failed: {e}")

    
    try:
        print(f"[depth_utils] Falling back to {_MODEL_DAV2_METRIC_INDOOR} ...")
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        _depth_processor = AutoImageProcessor.from_pretrained(_MODEL_DAV2_METRIC_INDOOR)
        _depth_model = AutoModelForDepthEstimation.from_pretrained(_MODEL_DAV2_METRIC_INDOOR)
        _depth_model = _depth_model.to(device).eval()
        if device.type == "cuda":
            _depth_model = _depth_model.half()
        _depth_backend = "dav2"
        _metric_active = True
        vram_mb = torch.cuda.memory_allocated(0) / 1e6 if device.type == "cuda" else 0
        print(f"[depth_utils] DAV2-Metric-Indoor-Large ready (fallback, metric_active=True). VRAM: {vram_mb:.0f}MB")
        return
    except Exception as e:
        print(f"[depth_utils] ⚠ DAV2-Metric-Indoor-Large failed: {e}")

    raise RuntimeError("[depth_utils] Could not load any depth model (DA3 and DAV2 both failed).")




def run_depth(frame_rgb: np.ndarray) -> np.ndarray:
    
    global _depth_call_count, _metric_depth_map
    _depth_call_count += 1
    t0 = time.perf_counter()

    if _depth_backend == "da3":
        if _depth_call_count == 1:
            
            
            
            
            print(f"[depth_utils] First live frame shape={frame_rgb.shape} "
                  f"(compare to torch.compile warmup dummy shape logged at startup)")
        
        
        prediction = _depth_model.inference([frame_rgb])
        raw = prediction.depth[0]
        depth_np = raw.cpu().float().numpy() if hasattr(raw, "cpu") else np.asarray(raw, dtype=np.float32)

        
        
        
        
        
        
        
        
        
        
        
        
        
        if depth_np.shape[:2] != frame_rgb.shape[:2]:
            depth_np = cv2.resize(
                depth_np, (frame_rgb.shape[1], frame_rgb.shape[0]), interpolation=cv2.INTER_LINEAR
            )
    else:
        
        image = Image.fromarray(frame_rgb)
        inputs = _depth_processor(images=image, return_tensors="pt")
        if _device.type == "cuda":
            inputs = {k: v.to(_device).half() for k, v in inputs.items()}
        else:
            inputs = {k: v.to(_device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = _depth_model(**inputs)
            predicted_depth = outputs.predicted_depth  
        depth = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=frame_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        depth_np = depth.cpu().float().numpy()

    
    _metric_depth_map = depth_np.copy()

    
    d_min, d_max = depth_np.min(), depth_np.max()
    if d_max - d_min > 1e-6:
        depth_np = (depth_np - d_min) / (d_max - d_min)
    else:
        depth_np = np.zeros_like(depth_np)
    depth_np = 1.0 - depth_np

    depth_ms = (time.perf_counter() - t0) * 1000
    if _depth_call_count <= 5 or _depth_call_count % 100 == 0:
        d_min2, d_max2 = float(depth_np.min()), float(depth_np.max())
        print(
            f"[DEPTH] call={_depth_call_count} {depth_ms:.0f}ms "
            f"shape={depth_np.shape} range=[{d_min2:.3f},{d_max2:.3f}] "
            f"backend={_depth_backend}"
        )

    return depth_np.astype(np.float32)


def run_depth_skippable(frame_rgb: np.ndarray) -> np.ndarray:
    """
    Run depth every DEPTH_SKIP_FRAMES frames; reuse the last map on skipped frames.
    At 5fps a skipped frame is ~200ms stale — acceptable for walking-speed navigation.
    Call reset_depth_skip() on each new client session.
    """
    global _depth_skip_counter, _last_depth_map, _last_depth_was_fresh
    _depth_skip_counter += 1
    if _last_depth_map is None or (_depth_skip_counter % DEPTH_SKIP_FRAMES) == 1:
        _last_depth_map = run_depth(frame_rgb)
        _last_depth_was_fresh = True
    else:
        _last_depth_was_fresh = False
    return _last_depth_map


def depth_was_fresh() -> bool:
    """True if the last run_depth_skippable() recomputed depth (vs. reusing the cached map).
    Stage 8 scans only on fresh frames — a stale map would double-confirm the same candidate."""
    return _last_depth_was_fresh


def reset_depth_skip() -> None:
    """Call on client connect to clear stale depth map from previous session."""
    global _depth_skip_counter, _last_depth_map, _last_depth_was_fresh, _metric_depth_map
    _depth_skip_counter = 0
    _last_depth_map = None
    _last_depth_was_fresh = False
    _metric_depth_map = None


def reset_calibration() -> None:
    """Call on client connect to clear per-class temporal-coherence distance history."""
    _last_dist_m.clear()


def metric_active() -> bool:
    """True when a metric depth head is loaded → metric_depth_in_region() returns real metres."""
    return _metric_active


def metric_depth_in_region(bbox_norm: list, mask_full: "np.ndarray | None" = None) -> "float | None":
    
    if _metric_depth_map is None:
        return None
    h, w = _metric_depth_map.shape
    x1 = max(0, int(bbox_norm[0] * w)); y1 = max(0, int(bbox_norm[1] * h))
    x2 = min(w, int(bbox_norm[2] * w)); y2 = min(h, int(bbox_norm[3] * h))
    if x2 <= x1 or y2 <= y1:
        return None
    depth_crop = _metric_depth_map[y1:y2, x1:x2]
    if mask_full is not None:
        mask_crop = mask_full[y1:y2, x1:x2]
        foreground = depth_crop[mask_crop > 0.5]
        if len(foreground) >= 50:
            return float(np.median(foreground))
    return float(np.median(depth_crop))


_NO_METRIC_FALLBACK_SCALE = 5.0  


def estimate_distance(relative_depth: float) -> float:

    return float(max(0.1, min(20.0, (1.0 - relative_depth) * _NO_METRIC_FALLBACK_SCALE)))




def get_depth_in_region(depth_map: np.ndarray, bbox_norm: list) -> float:
    """
    Mean depth in a normalized bbox [x1, y1, x2, y2] (values 0-1).
    Returns float in [0, 1]. HIGH = CLOSE.
    """
    h, w = depth_map.shape
    x1 = max(0, int(bbox_norm[0] * w))
    y1 = max(0, int(bbox_norm[1] * h))
    x2 = min(w, int(bbox_norm[2] * w))
    y2 = min(h, int(bbox_norm[3] * h))
    if x2 <= x1 or y2 <= y1:
        return 0.5
    region = depth_map[y1:y2, x1:x2]
    return float(region.mean())


def get_depth_with_mask(depth_map: np.ndarray, mask_full: np.ndarray,
                        bbox_norm: list) -> float:
    
    h, w = depth_map.shape
    x1 = max(0, int(bbox_norm[0] * w))
    y1 = max(0, int(bbox_norm[1] * h))
    x2 = min(w, int(bbox_norm[2] * w))
    y2 = min(h, int(bbox_norm[3] * h))
    if x2 <= x1 or y2 <= y1:
        return 0.5
    depth_crop = depth_map[y1:y2, x1:x2]
    mask_crop  = mask_full[y1:y2, x1:x2]
    foreground = depth_crop[mask_crop > 0.5]

    global _mask_call_count, _mask_fallback_count
    _mask_call_count += 1

    if len(foreground) < 50:
        _mask_fallback_count += 1
        
        if _mask_fallback_count % 20 == 0:
            print(
                f"[MASK_DEPTH] {_mask_fallback_count} sparse fallbacks "
                f"(total calls={_mask_call_count}, "
                f"rate={_mask_fallback_count/max(1,_mask_call_count)*100:.1f}%)"
            )
        return float(depth_crop.mean())  

    return float(foreground.mean())


def resize_masks(masks_np: np.ndarray, frame_h: int, frame_w: int) -> np.ndarray:
    """
    Resize YOLO seg mask tensor [N, H', W'] → [N, frame_h, frame_w] via nearest-neighbor.
    Pure numpy — no cv2 needed.
    """
    _, mh, mw = masks_np.shape
    row_idx = (np.arange(frame_h) * mh / frame_h).astype(np.int32).clip(0, mh - 1)
    col_idx = (np.arange(frame_w) * mw / frame_w).astype(np.int32).clip(0, mw - 1)
    return masks_np[:, row_idx[:, None], col_idx[None, :]]


def classify_depth_zones(depth_map: np.ndarray) -> dict:
    
    h, w = depth_map.shape
    thirds = {
        "left":   depth_map[:, :w // 3],
        "center": depth_map[:, w // 3: 2 * w // 3],
        "right":  depth_map[:, 2 * w // 3:],
    }
    result = {}
    for name, region in thirds.items():
        mean_d = float(np.percentile(region, 90))
        result[name] = _depth_to_zone_str(mean_d)
    return result


def _depth_to_zone_str(normalized_depth: float) -> str:
    if normalized_depth > 0.82:
        return "CRITICAL"
    elif normalized_depth > 0.65:
        return "CLOSE"
    elif normalized_depth > 0.45:
        return "MEDIUM"
    elif normalized_depth > 0.0:
        return "FAR"
    return "CLEAR"




def depth_to_meters(relative_depth: float, class_name: str, bbox_h_norm: float,
                    direction: str = "center", metric_dist: "float | None" = None) -> float:
    
    global _last_dist_m, _metric_log_count

    known_h = KNOWN_HEIGHTS.get(class_name)
    if known_h is not None and bbox_h_norm > 0.05:
        estimated_dist = known_h / (bbox_h_norm * 1.3)

        
        
        if metric_dist is not None:
            _metric_log_count += 1
            if _metric_log_count <= 10 or _metric_log_count % 50 == 0:
                print(f"[METRIC_CHECK] {class_name}: height_prior={estimated_dist:.2f}m "
                      f"metric={metric_dist:.2f}m gap={metric_dist - estimated_dist:+.2f}m "
                      f"bbox_h={bbox_h_norm:.3f}")

        if metric_dist is not None and bbox_h_norm < 0.15:
            
            
            dist = float(max(0.1, min(20.0, metric_dist)))
            if abs(metric_dist - estimated_dist) > 0.5:
                print(f"[METRIC_PREFER] {class_name}: bbox_h={bbox_h_norm:.3f} < 0.15 "
                      f"→ metric={metric_dist:.2f}m (height_prior={estimated_dist:.2f}m)")
        elif metric_dist is not None:
            if abs(metric_dist - estimated_dist) > METRIC_HEIGHT_DIVERGENCE_M:
                dist = min(metric_dist, estimated_dist)   
                print(f"[METRIC_CLAMP] {class_name}: diverge metric={metric_dist:.2f}m "
                      f"height_prior={estimated_dist:.2f}m (bbox_h={bbox_h_norm:.3f}) "
                      f"→ using closer={dist:.2f}m")
            else:
                dist = metric_dist
            dist = float(max(0.1, min(20.0, dist)))
        else:
            dist = float(max(0.1, min(20.0, estimated_dist)))
    elif metric_dist is not None:
        
        dist = float(max(0.1, min(20.0, metric_dist)))
    else:
        
        dist = float(max(0.1, min(20.0, (1.0 - relative_depth) * _NO_METRIC_FALLBACK_SCALE)))

    
    if dist > 15.0:
        print(
            f"[DEPTH_OUTLIER] {class_name} {dist:.1f}m "
            f"(bbox_h={bbox_h_norm:.3f} relative_depth={relative_depth:.3f})"
        )

    
    
    
    max_outward = 0.5 if class_name in _STATIC_CLASSES else 2.0
    _dm_key = (class_name, direction)
    prev_dist = _last_dist_m.get(_dm_key)
    if prev_dist is not None and dist > prev_dist + max_outward:
        print(
            f"[DEPTH_COHERENCE] {class_name}@{direction} outward clamp "
            f"{prev_dist:.1f}→{dist:.1f}m (max={max_outward})"
        )
        dist = prev_dist + max_outward
    _last_dist_m[_dm_key] = dist

    return dist
