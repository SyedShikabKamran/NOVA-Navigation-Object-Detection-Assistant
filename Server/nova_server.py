import asyncio
import base64
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from PIL import Image
from pydantic import BaseModel
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox as _LetterBox

# Match training-time letterbox: model was trained on black-edge (0,0,0) images, not grey (114).
_lb_defs = list(_LetterBox.__init__.__defaults__ or ())
_LetterBox.__init__.__defaults__ = tuple((0, 0, 0) if d == (114, 114, 114) else d for d in _lb_defs)

# Local modules (same directory)
sys.path.insert(0, os.path.dirname(__file__))
import depth_utils
import alert_logic

# ── App + global state ─────────────────────────────────────────────────────────
@asynccontextmanager
async def _lifespan(app: FastAPI):
    await _startup()
    yield

app = FastAPI(title="NOVA Inference Server", lifespan=_lifespan)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

yolo_model: YOLO | None = None      
frame_counter: int = 0


GROUNDING_DINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
grounding_dino_model: AutoModelForZeroShotObjectDetection | None = None
grounding_dino_processor: AutoProcessor | None = None

_frame_timestamps: list[float] = []  
_decode_error_count: int = 0         

_stat_total_ms:  list[int]   = []   
_stat_yolo_ms:   list[float] = []   
_stat_depth_ms:  list[float] = []   
_session_start:  float       = 0.0  

_inference_lock = asyncio.Lock()

_inference_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nova-inference")


_DEFAULT_WEIGHTS = "nova_yolo_run6_alt_best.pt"
MODEL_PATH = os.environ.get(
    "NOVA_MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "..", _DEFAULT_WEIGHTS)
)


async def _startup() -> None:
    global yolo_model, grounding_dino_model, grounding_dino_processor

    print(f"[NOVA] Device: {device}")
    if device.type == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[NOVA] GPU: {torch.cuda.get_device_name(0)} ({vram_gb:.1f}GB)")

    pt_path = os.path.abspath(MODEL_PATH)
    if not os.path.exists(pt_path):
        raise FileNotFoundError(
            f"[NOVA] Model not found: {pt_path}\n"
            f"Set NOVA_MODEL_PATH env var or place nova_yolo_run6_alt_best.pt "
            f"in the parent directory."
        )

    print(f"[NOVA] Loading YOLO-640 from {pt_path} ...")
    yolo_model = YOLO(pt_path)
    yolo_model.to(device)
    if device.type == "cuda":
        yolo_model.model.half()
        vram_used = torch.cuda.memory_allocated(0) / 1e6
        print(f"[NOVA] YOLO-640 ready (FP16). VRAM: {vram_used:.0f}MB")
    else:
        print("[NOVA] YOLO-640 ready (FP32, CPU).")

    await asyncio.get_running_loop().run_in_executor(
        _inference_executor, depth_utils.load_depth_model, device
    )

    print(f"[NOVA] Loading Grounding DINO ({GROUNDING_DINO_MODEL_ID}) ...")
    grounding_dino_processor = AutoProcessor.from_pretrained(GROUNDING_DINO_MODEL_ID)
    grounding_dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_DINO_MODEL_ID)
    grounding_dino_model = grounding_dino_model.to(device).eval()
   
    vram_mb = torch.cuda.memory_allocated(0) / 1e6 if device.type == "cuda" else 0
    print(f"[NOVA] Grounding DINO ready (fp32 weights, autocast fp16 on CUDA). VRAM: {vram_mb:.0f}MB")

    print("[NOVA] ✓ Server ready.")

@app.get("/health")
async def health():
    return {
        "status":           "ok",
        "device":           str(device),
        "yolo_loaded":      yolo_model is not None,
        # True once DA3-Large (or its DAV2 fallback) has loaded successfully — lets you
        # confirm from /health alone whether distance readings are real metric depth or
        # the relative-depth + scale-factor guess (see depth_to_meters()'s fallback path).
        "metric_active":    depth_utils.metric_active(),
        "frames_processed": frame_counter,
    }

@app.get("/stats")
async def stats():
    """Session-level latency summary. Polled by Colab Cell 5."""
    n = len(_stat_total_ms)
    if n == 0:
        return {"frames": 0, "session_s": 0}

    session_s = time.time() - _session_start if _session_start else 0
    fps = n / session_s if session_s > 0 else 0

    def _summarise(vals: list) -> dict:
        if not vals:
            return {}
        return {
            "avg_ms":  round(sum(vals) / len(vals), 1),
            "min_ms":  round(min(vals), 1),
            "max_ms":  round(max(vals), 1),
            "samples": len(vals),
        }

    recent = _stat_total_ms[-30:] if len(_stat_total_ms) >= 30 else _stat_total_ms

    return {
        "frames":        n,
        "session_s":     round(session_s, 1),
        "fps":           round(fps, 2),
        "total_latency": _summarise(_stat_total_ms),
        "total_recent":  _summarise(recent),
        "yolo":          _summarise(_stat_yolo_ms),
        "depth":         _summarise(_stat_depth_ms),   # fresh frames only (every other frame)
        "decode_errors": _decode_error_count,
    }

class FindRequest(BaseModel):
    image_b64: str
    query: str
    threshold: float = 0.3


@app.post("/find")
async def find(req: FindRequest):
    """
    Open-vocabulary "find X" for the Finder tab. Shares _inference_lock with the nav
    pipeline so a Finder query and an in-flight /ws frame never touch the GPU concurrently.
    """
    if grounding_dino_model is None or grounding_dino_processor is None:
        return {"boxes": [], "scores": [], "labels": [], "error": "grounding_dino_not_loaded"}

    async with _inference_lock:
        result = await asyncio.get_running_loop().run_in_executor(
            _inference_executor, _process_find, req.image_b64, req.query, req.threshold
        )
    return result


def _process_find(image_b64: str, query: str, threshold: float) -> dict:
    try:
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return {"boxes": [], "scores": [], "labels": [], "error": f"decode_failed: {e}"}

    t_start = time.perf_counter()
    # Grounding DINO expects lowercase, period-separated object phrases (e.g. "person. water bottle.").
    # Strip voice-command prefix ("find the X" / "find X") so only the object name is matched.
    text_query = query.strip().lower()
    for _prefix in ("find the ", "find a ", "find an ", "find "):
        if text_query.startswith(_prefix):
            text_query = text_query[len(_prefix):]
            break
    if not text_query.endswith("."):
        text_query += "."

    inputs = grounding_dino_processor(images=image, text=text_query, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}   # native dtype — no .half() cast, see _startup()

    try:
        with torch.no_grad():
            # autocast, not model.half(): keeps softmax/layernorm in fp32 (Grounding DINO's
            # cross-attention has no internal fp32 upcast — see the comment in _startup())
            # while still getting fp16 speed on the bulk matmuls.
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                outputs = grounding_dino_model(**inputs)

        # transformers 4.38+ requires input_ids for text-label decoding.
        results = grounding_dino_processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs["input_ids"],
            threshold=threshold,
            text_threshold=threshold,
            target_sizes=[image.size[::-1]],  # (height, width) — rescales boxes to original image
        )[0]
    except Exception as e:
        print(f"[NOVA] /find inference error: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        return {"boxes": [], "scores": [], "labels": [], "error": f"inference_failed: {type(e).__name__}: {e}"}

    # Normalize pixel-space boxes to [0, 1] so the Android client doesn't need image dimensions.
    w, h = image.size
    boxes_norm = [
        [
            round(max(0.0, min(1.0, x1 / w)), 4),
            round(max(0.0, min(1.0, y1 / h)), 4),
            round(max(0.0, min(1.0, x2 / w)), 4),
            round(max(0.0, min(1.0, y2 / h)), 4),
        ]
        for x1, y1, x2, y2 in results["boxes"].tolist()
    ]
    labels = results.get("text_labels", results.get("labels", []))
    find_ms = (time.perf_counter() - t_start) * 1000
    print(f"[NOVA] /find '{query}' (query='{text_query}') -> {len(boxes_norm)} box(es) in {find_ms:.0f}ms")

    return {
        "boxes": boxes_norm,
        "scores": [round(s, 4) for s in results["scores"].tolist()],
        "labels": labels,
    }

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    global frame_counter
    await ws.accept()
    client = ws.client
    print(f"[NOVA] Client connected: {client}")

    alert_logic.reset_session_state()
    depth_utils.reset_calibration()
    depth_utils.reset_depth_skip()
    # Reset per-session latency accumulators
    global _stat_total_ms, _stat_yolo_ms, _stat_depth_ms, _session_start
    _stat_total_ms, _stat_yolo_ms, _stat_depth_ms = [], [], []
    _session_start = time.time()

    try:
        while True:
            # Receive raw JPEG bytes from Android
            jpeg_bytes: bytes = await ws.receive_bytes()
            t_start = time.perf_counter()

            try:
                async with _inference_lock:
                    response = await asyncio.get_running_loop().run_in_executor(
                        _inference_executor, _process_frame, jpeg_bytes
                    )
            except Exception as e:
                # A single bad frame must not kill the whole session — the client would
                # have to fully reconnect (lost frames + reconnect overhead) for what's
                # often a transient inference hiccup, not a dead connection. Log it,
                # respond with a safe empty frame, and keep the session (and Stage 8/
                # temporal state) alive for the next frame.
                print(f"[NOVA] Frame processing error: {e}")
                import traceback; traceback.print_exc()
                response = _empty_response(f"processing_error: {e}")

            frame_counter += 1
            response["frame"] = frame_counter

            total_ms = int((time.perf_counter() - t_start) * 1000)
            response["inference_ms"] = total_ms

            # Accumulate latency stats for /stats endpoint
            _stat_total_ms.append(total_ms)

            # FPS tracking (circular buffer, last 60 frames)
            _frame_timestamps.append(time.perf_counter())
            if len(_frame_timestamps) > 60:
                _frame_timestamps.pop(0)

            try:
                await ws.send_text(json.dumps(response))
            except RuntimeError as e:

                if "after sending 'websocket.close'" in str(e):
                    print(f"[NOVA] Client disconnected mid-response after {frame_counter} frames "
                          f"(frame took {total_ms}ms).")
                    break
                raise

            # Per-frame header: frame number, total latency, path
            n_dets = len(response["detections"])
            path_str = response["path"]
            path_flag = " ◄◄" if path_str != "CONTINUE_FORWARD" else ""
            print(
                f"Frame[{frame_counter}] {total_ms}ms "
                f"dets={n_dets} path={path_str}{path_flag}"
            )

            if frame_counter % 50 == 0 and len(_frame_timestamps) >= 2:
                elapsed_s = _frame_timestamps[-1] - _frame_timestamps[0]
                fps = (len(_frame_timestamps) - 1) / elapsed_s if elapsed_s > 0 else 0
                if device.type == "cuda":
                    vram_used_mb  = torch.cuda.memory_allocated(0) / 1e6
                    vram_rsvd_mb  = torch.cuda.memory_reserved(0)  / 1e6
                    vram_str = f" vram={vram_used_mb:.0f}MB/{vram_rsvd_mb:.0f}MB"
                else:
                    vram_str = ""
                print(
                    f"[PERF:frame{frame_counter}] fps={fps:.1f}{vram_str} "
                    f"decode_errors={_decode_error_count}"
                )

    except WebSocketDisconnect:
        print(f"[NOVA] Client disconnected after {frame_counter} frames.")
    except Exception as e:
        print(f"[NOVA] Error in frame loop: {e}")
        import traceback; traceback.print_exc()

def _process_frame(jpeg_bytes: bytes) -> dict:
    nparr = np.frombuffer(jpeg_bytes, np.uint8)
    frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        return _empty_response("JPEG decode failed")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_h, frame_w = frame_bgr.shape[:2]

    use_half = device.type == "cuda"

    t_yolo = time.perf_counter()
    results = yolo_model(frame_bgr, verbose=False, imgsz=640, half=use_half, conf=0.20)
    yolo_ms = (time.perf_counter() - t_yolo) * 1000
    _stat_yolo_ms.append(yolo_ms)

    # ── DA3-Large Metric / DAV2-Metric-Indoor fallback (skip every other frame) ──
    t_depth = time.perf_counter()
    depth_map = depth_utils.run_depth_skippable(frame_rgb)
    depth_ms = (time.perf_counter() - t_depth) * 1000
    if depth_utils.depth_was_fresh():
        _stat_depth_ms.append(depth_ms)

    r = results[0]
    masks_resized = None
    if r.masks is not None:
        raw_cls_ids = r.boxes.cls.cpu().numpy().astype(int)
        needs_mask = any(
            0 <= c < len(alert_logic.NOVA_CLASSES)
            and alert_logic.NOVA_CLASSES[c] in alert_logic._MASK_DEPTH_CLASSES
            for c in raw_cls_ids
        )
        if needs_mask:
            masks_np = r.masks.data.cpu().float().numpy()
            masks_resized = depth_utils.resize_masks(masks_np, frame_h, frame_w)

    detections = alert_logic.parse_detections(r, frame_h, frame_w, depth_map, masks_resized)
    n_raw = len(detections)
    raw_known_bboxes = [d["bbox"] for d in detections]
    detections = alert_logic.dedup_same_class(detections)
    n_dedup_removed = n_raw - len(detections)
    detections = alert_logic.filter_ground_hazard_positions(detections)
    n_after_pos_filter = len(detections)
    alert_logic.update_temporal_history(detections)
    detections = alert_logic.temporal_filter(detections)
    n_temporal_dropped = n_after_pos_filter - len(detections)
    unknown_obs = alert_logic.find_unknown_obstacles(
        depth_map, detections, fresh=depth_utils.depth_was_fresh(),
        raw_known_bboxes=raw_known_bboxes
    )
    if unknown_obs:
        detections = detections + unknown_obs
    depth_zones = depth_utils.classify_depth_zones(depth_map)
    alerts = alert_logic.generate_alerts(detections)
    path = alert_logic.calculate_free_path(detections, depth_zones)

    json_detections = [
        {
            "class":      d["class"],
            "confidence": round(d["confidence"], 3),
            "bbox":       d["bbox"],
            "direction":  d["direction"],
            "distance_m": round(d["distance_m"], 2),
            "zone":       d["zone"],
        }
        for d in detections
    ]

    print(
        f"  yolo={yolo_ms:.0f}ms depth={depth_ms:.0f}ms "
        f"[raw={n_raw} dedup-{n_dedup_removed} temporal-{n_temporal_dropped} "
        f"-> final={len(detections)}{'(+' + str(len(unknown_obs)) + 'unk)' if unknown_obs else ''}]"
    )
    if detections:
        det_parts = []
        for d in sorted(detections, key=lambda x: x["distance_m"]):
            det_parts.append(
                f"{d['class']}({d['confidence']:.2f}) "
                f"{d['direction']} {d['distance_m']:.1f}m [{d['zone']}]"
            )
        print(f"  dets: {' | '.join(det_parts)}")
    if alerts:
        for a in alerts:
            print(f"  [P{a['priority']}] {a['message']}")

    return {
        "frame":        0,
        "detections":   json_detections,
        "depth_zones":  depth_zones,
        "alerts":       alerts,
        "path":         path,
        "inference_ms": 0,
    }


def _empty_response(reason: str = "") -> dict:
    """Fallback response when frame decoding fails."""
    global _decode_error_count
    _decode_error_count += 1
    print(f"[NOVA] Empty response #{_decode_error_count}: {reason}")
    return {
        "frame":       0,
        "detections":  [],
        "depth_zones": {"left": "CLEAR", "center": "CLEAR", "right": "CLEAR"},
        "alerts":      [],
        "path":        "CONTINUE_FORWARD",
        "inference_ms": 0,
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
