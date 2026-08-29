package com.nova.assistant.util

import android.os.Build

/**
 * NOVA Constants — All configuration values for the app.
 * Centralized here so any tuning requires changing only this file.
 */
object NovaConstants {

    // ── Model Files ──
    // ACTIVE: nova_yolo_run6_alt_fp16.tflite — YOLO26s-seg, E2E NMS output [1,300,38], mAP50=0.715, 18 classes.
    //   NMS applied internally — zero CPU NMS post-processing required.
    //   NNAPI INCOMPATIBLE: End2End NMS ops rejected by all tested NNAPI drivers → GPU/CPU only.
    //
    // ── NMS-FREE MODEL — HISTORY ──────────────────────────────────────────────────────────────
    // Tried:     nova_yolo26s_run2_nms_free_fp16.tflite — output [1,8400,65] raw anchors.
    // Reason:    MediaTek Neuron NNAPI driver rejects TFLite graphs containing the E2E NMS op,
    //            forcing GPU fallback at 0.33 FPS instead of expected ~3-5 FPS via NNAPI.
    // Abandoned: CPU NMS post-inference = 400–2400ms/frame bottleneck on device.
    //            Heavy inference moved to Colab/Kaggle server GPU (E2E at ~8ms).
    //            Android retains E2E TFLite as offline fallback only.
    // ──────────────────────────────────────────────────────────────────────────────────────────
    // ── Run6-alt model switch (28→18 classes) ─────────────────────────────────────────────────
    // Single source of truth for the run6-alt swap. To activate, once the trained + VALIDATED
    // run6-alt model is ready:
    //   1. Drop app/src/main/assets/nova_yolo_run6_alt_fp16.tflite  (18-class E2E TFLite).
    //   2. Set USE_RUN6_ALT = true here.
    // That flips BOTH the active class list (SplashScreen.NOVA_CLASSES → NOVA_CLASSES_RUN6_ALT)
    // and the asset file below. NO YoloDetector change: the E2E output carries a single class_id
    // float (not per-class logits), so the 38-feature layout is identical for 18 or 28 classes.
    // Dropped class names (pole/fence/wall/sofa/tree/vegitation/projector/fire-extinguisher/
    // empty-chair/occupied-chair) simply never appear in detections — their membership in the
    // ClassConfig sets + CONFIDENCE_THRESHOLDS below is inert, so those need no edit.
    const val USE_RUN6_ALT = true

    const val YOLO_RUN4_MODEL_FILE     = "nova_yolo_run4_fp16.tflite"
    const val YOLO_RUN6_ALT_MODEL_FILE = "nova_yolo_run6_alt_fp16.tflite"
    const val YOLO_RUN2_E2E_MODEL_FILE = "nova_yolo26s_run2_best_float16.tflite"
    // Not `const` — a const val can't hold an `if`. Plain val is fine (used as a runtime filename).
    @JvmField val YOLO_MODEL_FILE      = if (USE_RUN6_ALT) YOLO_RUN6_ALT_MODEL_FILE else YOLO_RUN4_MODEL_FILE

    // Depth model: MiDaS Small (on-device local inference only).
    // Server-side depth: DA3-Large Metric (primary) + DAV2-Metric-Indoor-Large (fallback).
    // MiDaS v2.1 Small — true FP16 TFLite. Input: 256×256, outputs [1,256,256].
    const val MIDAS_MODEL_FILE  = "midas_small_fp16_true.tflite"
    const val YOLO_INPUT_SIZE   = 640
    const val MIDAS_INPUT_SIZE  = 256

    // ── Distance Zones (meters) ──
    // NOTE: @Volatile var (not const) so Settings screen can update at runtime (H7).
    // At 1.4 m/s walking pace, 1.5m DANGER ≈ 1.07s reaction time. Old 0.8m was
    // physically unsafe — TTS couldn't finish before collision.
    @Volatile var DANGER_DISTANCE  = 1.5f   // C2: was 0.8f
    @Volatile var WARNING_DISTANCE = 3.0f   // C2: was 2.0f
    @Volatile var CAUTION_DISTANCE = 5.0f   // C2: new advance-awareness tier

    // ── Suppression Timers (milliseconds) ──
    // C4/M3: tiered by priority — P1 danger re-announces every 800ms,
    // P4 info suppressed for 4 seconds (was 8s — too slow for real-world nav).
    const val SUPPRESSION_P1_MS        =  800L   // C4: replaces "never suppress P1" bypass
    const val SUPPRESSION_P2_MS        = 3000L   // was 1500ms — raised to break TTS→STT AEC feedback loop
    const val SUPPRESSION_P3_MS        = 4000L   // direction guidance (was 2000ms)
    const val SUPPRESSION_P4_MS        = 4000L   // informational / context (was 8000ms)
    const val SUPPRESSION_UNKNOWN_MS   = 4000L   // depth-scan "unknown obstacle" — longer than P1
    // because unknown obstacles can persist every YOLO cycle (800ms) but VIP only needs
    // re-alert every ~4s for static unknown objects. Real P1 hazards (stairs, car) keep 800ms.
    const val DIRECTION_SUPPRESSION_MS = 2000L   // kept for backward compat (was 3000ms)

    // Post-speech quiet period: after ANY TTS output ends, non-hazard alerts (P3/P4)
    // are held back for this long so the mic gets uncontested priority for a follow-up
    // command. P1/P2 always bypass this — safety alerts are never held back.
    const val LISTENING_WINDOW_MS = 2500L

    // ── Adaptive Frame Rate ──
    const val FPS_MOVING = 10
    const val FPS_STATIONARY = 5   // was 3 — 3fps at standstill is too slow for obstacle detection
    const val FPS_LOW_BATTERY = 5
    const val FPS_CRITICAL = 3
    const val MOTION_THRESHOLD = 0.5f // m/s²
    const val STATIONARY_DELAY_MS = 3000L

    // ── Battery Thresholds (%) ──
    const val BATTERY_LOW = 20
    const val BATTERY_CRITICAL = 10

    // ── Thermal Thresholds (°C) ──
    const val THERMAL_WARM = 40f
    const val THERMAL_HOT = 45f
    const val THERMAL_CRITICAL = 50f

    // ── Low Light (lux) ──
    const val LOW_LIGHT_LUX = 60f
    const val VERY_LOW_LIGHT_LUX = 20f

    // ── OCR Quiet Period ──
    // "Read text" means the user is deliberately stationary, holding the phone close to the
    // object containing the text. Without this, YOLO can classify that very object as an
    // "unknown obstacle" at close range and repeatedly interrupt/flush the OCR announcement
    // via P1/P2 hazard alerts — the attention gate always re-fires P1/P2 (see
    // FeedbackOrchestrator.deliverAlert), so this would loop for as long as the user stands
    // there. FeedbackOrchestrator.beginOcrQuietPeriod() suppresses ALL alerts (even P1/P2)
    // for a bounded window instead.
    const val OCR_INITIAL_QUIET_MS = 5000L   // covers "Reading text..." announcement + ML Kit recognition latency
    const val OCR_QUIET_MIN_MS = 4000L       // floor once the result is known — covers short phrases
    const val OCR_QUIET_MAX_MS = 45000L      // cap — a long OCR result must not silence hazards for minutes

    // ── NNAPI Toggle ──
    // HiSilicon/Kirin: disabled — ANEURALNETWORKS_BAD_DATA on YOLO FP16 NMS ops.
    //   Known hardware codenames (Build.HARDWARE):
    //     "hi6250"  → Kirin 710 / 710F   (Huawei JNY-LX1 test device: bengal equiv)
    //     "hi6470"  → Kirin 810           (Huawei JNY-LX1 actual)
    //     "hi3660"  → Kirin 960
    //   All match via prefix "hi6" or "hi3"; "kirin" catches any literal kirin strings.
    // Qualcomm (SD662/SD680): enabled — YoloDetector routes:
    //   FP16 model → tryAttachNnapi (allowFp16=true, Adreno OpenCL via NNAPI)
    // Others (MediaTek, Exynos): GPU delegate used; NNAPI attempted only if GPU fails.
    val NNAPI_ENABLED: Boolean = Build.HARDWARE.lowercase().let { hw ->
        !hw.contains("hi6") && !hw.contains("hi3") && !hw.contains("kirin")
    }

    // NNAPI construction succeeding is not proof the accelerator is real hardware — some SoCs
    // (observed: MediaTek MT6789 "generic(runtime-routed)" candidate) silently route to an
    // internal CPU path instead. A warm-up inference timed after attach catches this: real
    // hardware delegates (NNAPI/GPU) land well under 1s on this model; a CPU-routed "NNAPI"
    // takes 2-6s. Threshold set conservatively above GPU's worst case, below CPU's best case.
    const val NNAPI_LATENCY_FALLBACK_MS = 1200L

    // ── Depth Frame-Skip (adaptive for device performance tier) ──
    // 1 = every frame (flagship/upper-mid with GPU delegate)
    // 2 = every other frame (mid-range, ~SD 680 / Helio G99)
    // 3 = every third frame (budget-mid, ~SD 665 / Helio G85, CPU-only)
    // Set at runtime by NovaNavigationService based on observed inference latency.
    // Default 1 until device tier is established after warmup.
    const val DEPTH_FRAME_SKIP_HIGH = 1
    const val DEPTH_FRAME_SKIP_MID  = 2
    const val DEPTH_FRAME_SKIP_LOW  = 3

    // ── Model Warm-up ──
    const val WARMUP_PASSES = 5
    // Maximum passes for adaptive GPU warm-up loop (adaptive exits early once stable)
    const val MAX_WARMUP_PASSES = 25

    // ── Free Path Zones ──
    const val ZONE_FAR_LEFT_END = 0.15f   // 0% - 15% of frame width
    const val ZONE_LEFT_END = 0.35f       // 15% - 35%
    const val ZONE_CENTER_END = 0.65f     // 35% - 65%
    const val ZONE_RIGHT_END = 0.85f      // 65% - 85%
    // FAR_RIGHT: 85% - 100%

    // ── Unknown Obstacle Corridor (metric, not angular) ─────────────────────
    // ZONE_*_END above bucket by a FIXED FRACTION of frame width — physically that band
    // covers a wider swath of real space the farther away it is (angular, not metric).
    // That's fine for named YOLO detections, where lateral context matters for orientation.
    // It's wrong for the class-agnostic unknown-obstacle scanner, whose only question is
    // "is this in my walking path" — so that scanner is gated to a physically-constant
    // corridor instead. Half-width 0.65m (1.3m total) is generous relative to shoulder
    // width (~0.45m) to cover white-cane sweep arc + lateral drift + depth-scale noise,
    // per ADA guidance that cane sweep clears roughly body width. Named-class detections
    // are NOT restricted by this — only DepthEstimator.findDepthAnomalies().
    const val UNKNOWN_OBSTACLE_CORRIDOR_HALF_WIDTH_M = 0.65f

    // Horizontal FOV factor for the corridor projection. DepthCalibrator's known-height
    // triangulation already uses 1.3 (tan(32.5°)x2 ~= 65 deg vertical FOV) for the VERTICAL
    // axis. There's no separate horizontal-FOV calibration in this codebase, and the
    // on-device working frame is square (YOLO 640x640, depth map 256x256) with x/y
    // normalized independently — so this reuses the same constant, approximating the
    // working frame's FOV as isotropic. Coarser than a true per-device camera intrinsics
    // model, but consistent with the precision level already accepted in DepthCalibrator.
    const val UNKNOWN_OBSTACLE_CORRIDOR_FOV_FACTOR = 1.3f

    // Below this fraction of the depth model's own running-average raw dynamic range,
    // treat the frame as low-texture/low-confidence (blank wall, glossy floor, dim
    // light) — MiDaS-Small's per-frame min-max normalization amplifies noise when the
    // raw range is unusually narrow. Field log 2026-07-15 raw ranges spanned 250-1355
    // (median ~778, p10 ~556) — a frame under 50% of its own recent baseline is a real
    // outlier, not normal variation.
    const val LOW_CONFIDENCE_RANGE_FRACTION = 0.5f

    // ── NOVA Inference Server (GPU server via Render + Colab/Kaggle ngrok) ──
    // Render URL registry — always-on, stores the current Colab/Kaggle ngrok WSS URL.
    // Replace with your actual Render deployment URL after deploying url_registry/.
    const val SERVER_REGISTRY_URL = "https://nova-url-registry-production.up.railway.app"

    // ponytail: temporary direct fallback — paste the wss:// URL here when Colab is running.
    // Leave blank to use on-device inference only. Remove once Render registry is deployed.
    const val SERVER_DIRECT_WSS_URL = ""

    // Long side (px) of the aspect-preserved frame sent to the GPU server.
    // Server runs YOLO at imgsz=640 on a model trained on 640×640 black-edge images.
    // Send at 640px long-side so server only adds padding (no content rescale).
    const val SERVER_SEND_LONG_SIDE = 640

    // ── Grounding DINO (Finder tab) ──
    // Server's POST /find endpoint — boxes below this score are discarded.
    const val GROUNDING_DINO_CONFIDENCE = 0.3f

    // ── NMS ──
    // MAX_DETECTIONS_PER_FRAME raised 8 → 10.
    // Nav-YOLO (MDPI 2025) and similar indoor navigation systems use 10–15 for multi-directional
    // spatial coverage. Suppression timers (SUPPRESSION_P*_MS) govern TTS frequency, not this cap.
    const val MAX_DETECTIONS_PER_FRAME = 10   // raised from 8 for better spatial coverage

    // ── Temporal Tracker ──
    const val BBOX_GROWTH_THRESHOLD = 1.15f  // 15% larger = approaching
    const val BBOX_SHRINK_THRESHOLD = 0.85f  // 15% smaller = receding

    // ── Ramp Tilt Suppression (on-device fallback) ──
    // When camera tilts downward, flat floor fills the frame and misclassifies as ramp.
    // Raise the ramp confidence requirement proportionally to tilt magnitude.
    const val RAMP_TILT_LOW_DEG   = 10f    // mild tilt: apply mid confidence floor
    const val RAMP_TILT_HIGH_DEG  = 20f    // significant tilt: apply high confidence floor
    const val RAMP_TILT_MID_CONF  = 0.20f  // min conf during mild tilt
    const val RAMP_TILT_HIGH_CONF = 0.35f  // min conf during significant tilt

    // ── Direction Hysteresis (on-device fallback) ──
    // Suppress MOVE_LEFT↔MOVE_RIGHT flip-flops for this many milliseconds.
    const val DIRECTION_COMMITMENT_MS = 1500L
}

/**
 * Detection class categories with per-class confidence thresholds.
 *
 * ── WHY THRESHOLDS ARE 0.20–0.40 ────────────────────────────────────────────
 * Run4 uses YOLO26 end-to-end NMS output [1, 300, 37] (28 classes, mAP50=0.646).
 * Post-NMS confidence scores are significantly higher than raw pre-NMS scores:
 *   Old (raw pre-NMS [1,8400,65]): 0.001-0.052 range on device
 *   New (end-to-end NMS [1,300,37]): 0.30-0.85 range (post-NMS, model-filtered)
 *
 * Thresholds are set conservatively for the first on-device run.
 * CALIBRATION: Check "E2EParse topScores" in LogCat after first run.
 *   thresh=0 every frame  -> lower all thresholds by 0.05-0.10
 *   e2e=300 every frame   -> raise all thresholds by 0.05-0.10
 *   good range: thresh=3-10 detections/frame with topScores in 0.30-0.85
 *
 * ── CLASS GROUPING RATIONALE ──────────────────────────────────────────────────
 * Vehicle reclassification (car/bicycle/bus/truck → SAFETY_CRITICAL):
 *   Research: YOLO-OD (PMC, Nov 2024), Nav-YOLO (MDPI, 2025), and PC-CS-YOLO (MDPI, Jan 2025)
 *   all classify vehicles as safety-critical for VIP navigation assistance.
 *   Effect: vehicles at WARNING distance (1.5–3m) now trigger P2_DANGER instead of
 *   P3_DIRECTION. Correct — a car at 2m is a danger, not a navigation hint.
 *
 * Ground-hazard sub-tier (stairs/pothole/curb/speed-breaker/ramp → 0.012f):
 *   Research: PC-CS-YOLO (2025) specifically calls out ground-level obstacle detection
 *   for VIPs as the highest-priority recall problem. A missed pothole causes a fall;
 *   a false-positive pothole costs one unnecessary detour. Recall > Precision here.
 *
 * TEST MODE NOTE: Two sets defined — COCO (yolo26s.pt testing) and NOVA (production).
 * Toggle USE_COCO_TEST_CLASSES below.
 */
object ClassConfig {

    // ── Toggle: true = COCO test model (yolo26s.onnx), false = custom NOVA model ──
    // ACTIVE: false — nova_yolo_run4_fp16.tflite (mAP50=0.646, 28 NOVA classes, E2E NMS [1,300,37]).
    private const val USE_COCO_TEST_CLASSES = false

    // ── NOVA custom model class sets ────────────────────────────────────────────

    // Ground-level fall/trip hazards — overrides to lowest threshold (0.012f).
    // A missed detection for these classes directly causes user injury.
    // All are subsets of SAFETY_CRITICAL_NOVA (so they also inherit P2_DANGER alert priority).
    private val GROUND_HAZARD_NOVA = setOf(
        "stairs", "pothole", "curb", "speed-breaker", "ramp"
    )

    // Structural + ground + vehicle hazards.
    // CHANGE from prior version: car, bicycle, bus, truck moved here from NAVIGATION_NOVA.
    // Vehicles at WARNING distance (1.5–3m) need P2_DANGER, not P3_DIRECTION.
    private val SAFETY_CRITICAL_NOVA = setOf(
        "pole", "fence", "wall",                              // structural barriers
        "ramp", "stairs", "pothole", "speed-breaker", "curb", // ground hazards (also in GROUND_HAZARD)
        "car", "bicycle", "bus", "truck"                      // vehicles (moved from NAVIGATION)
    )

    // Non-vehicle navigation targets — person and door are directional navigation cues,
    // not collision hazards at typical distances. Bench retained here as a rest marker.
    private val NAVIGATION_NOVA = setOf(
        "person", "door", "bench"
    )

    private val OBSTACLE_NOVA = setOf(
        "chair", "empty-chair", "occupied-chair", "table",
        "sofa", "tree", "pot-plant", "vegitation"
    )

    // Pure context objects — no collision risk in typical usage.
    // fire-extinguisher and water dispenser are physically present but wall-mounted;
    // false positives waste TTS time with no safety upside.
    private val CONTEXT_ONLY_NOVA = setOf(
        "computer", "projector", "white-board",
        "fire-extinguisher", "water dispenser"
    )

    // ── COCO 80-class test model sets ────────────────────────────────────────────
    private val GROUND_HAZARD_COCO: Set<String> = emptySet()  // no COCO ground-hazard equivalents

    // Vehicles moved to SAFETY_CRITICAL for consistency with NOVA class logic.
    private val SAFETY_CRITICAL_COCO = setOf(
        "person",           // Collision hazard
        "motorcycle",       // Fast-moving hazard
        "bicycle",          // Active hazard
        "car", "bus", "truck", "train",  // vehicles — safety-critical per Nav-YOLO 2025
        "traffic light"     // Navigation signal awareness
    )
    private val NAVIGATION_COCO = setOf(
        "bench", "fire hydrant", "stop sign"
    )
    private val OBSTACLE_COCO = setOf(
        "chair", "couch", "dining table", "bed",
        "potted plant", "suitcase", "backpack",
        "refrigerator", "toilet", "sink", "oven"
    )
    private val CONTEXT_ONLY_COCO = setOf(
        "tv", "laptop", "keyboard", "cell phone",
        "book", "clock", "bottle", "cup"
    )

    // ── Active sets (switched by flag above) ──────────────────────────────────

    val GROUND_HAZARD: Set<String> =
        if (USE_COCO_TEST_CLASSES) GROUND_HAZARD_COCO else GROUND_HAZARD_NOVA

    val SAFETY_CRITICAL: Set<String> =
        if (USE_COCO_TEST_CLASSES) SAFETY_CRITICAL_COCO else SAFETY_CRITICAL_NOVA

    val NAVIGATION: Set<String> =
        if (USE_COCO_TEST_CLASSES) NAVIGATION_COCO else NAVIGATION_NOVA

    val OBSTACLE: Set<String> =
        if (USE_COCO_TEST_CLASSES) OBSTACLE_COCO else OBSTACLE_NOVA

    val CONTEXT_ONLY: Set<String> =
        if (USE_COCO_TEST_CLASSES) CONTEXT_ONLY_COCO else CONTEXT_ONLY_NOVA

    /**
     * Per-class confidence thresholds — Run4 E2E NMS model (nova_yolo_run4_fp16.tflite).
     *
     * E2E post-NMS scores (observed 0.30–0.85 on device). The model applies NMS internally
     * and only outputs up to 300 detections; low-confidence padding slots score ~0.
     * These thresholds filter those 300 slots to the relevant detections.
     *
     * Run4 calibration (per PROGRESS.md empirical tuning — mAP50 per class informed picks):
     *   0.10  ramp                              -- very low val count, max recall
     *   0.15  curb, pothole                     -- ground hazard recall
     *   0.20  stairs + remaining GROUND_HAZARD  -- speed-breaker mAP50=0.985, stairs=0.806
     *   0.25  SAFETY_CRITICAL / NAV / OBSTACLE  -- E2E post-NMS default
     *   0.40  CONTEXT_ONLY                      -- precision focus, no safety risk
     *
     * RE-TUNING (check "E2EParse topScores" in nova_debug.log after first device run):
     *   thresh=0 every frame  → lower all tiers by 0.05
     *   e2e=300 every frame   → raise all tiers by 0.05 (model slots all filled)
     *   Target: 3-10 detections/frame in a typical cluttered scene.
     */
    // Floor stays below the lowest per-class override (ramp=curb=0.20f as of the 2026-07-15
    // curb sync) so no override is silently clamped.
    const val MIN_CONFIDENCE_FLOOR = 0.10f  // E2E post-NMS floor (was 0.20f — clamped ramp/curb overrides)

    val CONFIDENCE_THRESHOLDS: Map<String, Float> = buildMap {
        SAFETY_CRITICAL.forEach { put(it, 0.25f) }  // structural + vehicles
        NAVIGATION.forEach     { put(it, 0.25f) }   // person, door, bench
        OBSTACLE.forEach       { put(it, 0.25f) }   // furniture, vegetation
        CONTEXT_ONLY.forEach   { put(it, 0.40f) }   // informational — precision focus
        // Run4 ground hazard overrides — must come AFTER SAFETY_CRITICAL so they overwrite 0.25f.
        GROUND_HAZARD.forEach  { put(it, 0.20f) }   // stairs, speed-breaker default
        // 2026-07-14 Run6-alt F1 retune (val set: see NOVA_PROGRESS.md Session 36).
        // ramp raised 0.10→0.20: old value assumed "low val count"; real data shows
        // F1=0.794 (solid) — no longer needs an aggressive recall floor. Now matches server.
        put("ramp",    0.20f)
        // 2026-07-15: synced to server's 0.20 (alert_logic.py CONFIDENCE_THRESHOLDS) — server
        // raised curb 0.15→0.20 after confirming 0.15 fires on pavement/floor textures at
        // camera tilt. Same run6_alt weights on both sides, so the FP mode applies here too;
        // device was never updated when the server threshold was retuned.
        put("curb",    0.20f)
        // pothole raised 0.15→0.30: aligns with server's empirical floor-shadow fix
        // (real potholes score 0.50+); F1=0.673/recall=0.61 doesn't justify the old floor.
        put("pothole", 0.30f)
        // table/chair/bench lowered 0.25→0.20: recall 0.60-0.66 with decent precision
        // (0.76-0.82), no conflicting FP note, both temporally filtered as an FP backstop.
        put("table",   0.20f)
        put("chair",   0.20f)
        put("bench",   0.20f)
        // person LEFT UNCHANGED at 0.25 despite F1=0.559/recall=0.48 (worst in Run6-alt).
        // This is a model-capability gap, not a threshold problem — lowering it trades
        // recall for a flood of false "person" alerts. Tracked as a Run7 retrain target
        // (project memory: Q6 person floor 0.55 → target 0.70). Revisit after Run7 lands.
        // Dropped 2026-07-02: bicycle/truck detection unreliable (Run4 mAP50 0.316/0.348;
        // Run6-alt carries the same weakness — 0.356/0.322). Threshold set above the
        // confidence ceiling (device confidence never exceeds 1.0, see YoloDetector.kt:567)
        // so neither class ever passes the `confidence >= threshold` gate at
        // YoloDetector.kt:581 — no model change, no retrain. Revert by deleting these 2 lines.
        // 2026-07-16: raised 0.55→0.60 (false positive on bus-less indoor scene; server matched)
        put("bus",     0.60f)
        put("bicycle", 2.0f)
        put("truck",   2.0f)
    }

    /** Default threshold for unknown/unlisted classes. */
    const val DEFAULT_THRESHOLD = 0.25f  // E2E post-NMS default (was 0.15f for NMS-free)

    fun getThreshold(className: String): Float =
        maxOf(CONFIDENCE_THRESHOLDS[className] ?: DEFAULT_THRESHOLD, MIN_CONFIDENCE_FLOOR)

    fun isSafetyCritical(className: String): Boolean =
        className in SAFETY_CRITICAL

    fun isGroundHazard(className: String): Boolean =
        className in GROUND_HAZARD

    fun isNavigational(className: String): Boolean =
        className in NAVIGATION || className in SAFETY_CRITICAL

    // True for NAVIGATION classes (person, door, bench) — these are navigable targets,
    // not collision hazards, so DANGER-zone alerts omit "Stop!".
    fun isNavigable(className: String): Boolean =
        className in NAVIGATION
}

/**
 * Alert priority levels.
 */
enum class AlertPriority(val level: Int) {
    P1_EMERGENCY(1),    // DANGER zone + unknown obstacles
    P2_DANGER(2),       // WARNING zone obstacles
    P3_DIRECTION(3),    // Path guidance ("move left")
    P4_INFO(4),         // Path clear, hazard memory
    P5_CONTEXT(5);      // Room context, on-demand info

    companion object {
        // C2: added CAUTION tier between WARNING and FAR
        fun fromDistance(distanceMeters: Float, isSafetyCritical: Boolean): AlertPriority {
            return when {
                distanceMeters < NovaConstants.DANGER_DISTANCE  -> P1_EMERGENCY
                distanceMeters < NovaConstants.WARNING_DISTANCE ->
                    if (isSafetyCritical) P2_DANGER else P3_DIRECTION
                distanceMeters < NovaConstants.CAUTION_DISTANCE ->
                    if (isSafetyCritical) P3_DIRECTION else P4_INFO
                else -> P4_INFO
            }
        }
    }
}

/**
 * Distance zones for obstacle classification.
 * C2: Added CAUTION tier (between WARNING and FAR) for advance awareness at 3–5m.
 * WARNING: Adding CAUTION causes non-exhaustive `when` compile errors in
 * FreePathCalculator, AlertPhraseGenerator, DetectionOverlay — all must be updated.
 */
enum class DistanceZone {
    DANGER,     // < 1.5m  — immediate stop / critical TTS
    WARNING,    // 1.5m–3.0m — slow down / P2 alert
    CAUTION,    // 3.0m–5.0m — advance awareness, P3/P4
    FAR;        // > 5.0m  — skip unless moving toward VIP

    companion object {
        fun fromDistance(meters: Float): DistanceZone = when {
            meters < NovaConstants.DANGER_DISTANCE  -> DANGER
            meters < NovaConstants.WARNING_DISTANCE -> WARNING
            meters < NovaConstants.CAUTION_DISTANCE -> CAUTION
            else -> FAR
        }
    }
}

/**
 * Spatial direction of detected objects.
 */
enum class SpatialDirection {
    FAR_LEFT, LEFT, CENTER, RIGHT, FAR_RIGHT;

    // H3: Clock-position directions are unambiguous for blind users.
    // "far left/right" is vague — "9 o'clock" maps to a physical body reference.
    fun toSpokenDirection(): String = when (this) {
        FAR_LEFT  -> "9 o'clock"
        LEFT      -> "10 o'clock"
        CENTER    -> "12 o'clock"
        RIGHT     -> "2 o'clock"
        FAR_RIGHT -> "3 o'clock"
    }

    companion object {
        fun fromNormalizedX(x: Float): SpatialDirection = when {
            x < NovaConstants.ZONE_FAR_LEFT_END -> FAR_LEFT
            x < NovaConstants.ZONE_LEFT_END -> LEFT
            x < NovaConstants.ZONE_CENTER_END -> CENTER
            x < NovaConstants.ZONE_RIGHT_END -> RIGHT
            else -> FAR_RIGHT
        }
    }
}
