package com.nova.assistant.engine

import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import kotlin.math.roundToInt
import com.nova.assistant.ml.DepthEstimator
import com.nova.assistant.ml.YoloDetector
import com.nova.assistant.util.*
import com.nova.assistant.util.FileLogger
import kotlinx.coroutines.async
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import javax.inject.Inject
import javax.inject.Singleton

/**
 * NOVA Inference Pipeline — The complete 10-stage processing chain.
 *
 * Camera Frame → [10 stages] → NovaAlert (ready to speak)
 *
 * This is the brain of the app. Each frame passes through:
 *   Stage 1-2: Preprocessing (handled before calling this)
 *   Stage 3: YOLO detection
 *   Stage 4: Per-class confidence filtering (inside YOLO detector)
 *   Stage 5: Depth estimation
 *   Stage 6: Distance zone classification
 *   Stage 7: Temporal tracking (moving vs static)
 *   Stage 8: Unknown obstacle detection
 *   Stage 9: Free path calculation
 *   Stage 10: Priority queue + suppression
 */
@Singleton
class InferencePipeline @Inject constructor(
    private val yoloDetector: YoloDetector,
    private val depthEstimator: DepthEstimator,
    private val depthCalibrator: DepthCalibrator,
    private val temporalTracker: TemporalTracker,
    private val freePathCalculator: FreePathCalculator,
    private val priorityQueue: PriorityAlertQueue,
    private val suppressionFilter: SuppressionFilter,
    private val phraseGenerator: AlertPhraseGenerator,
    private val temporalFilter: DetectionTemporalFilter,
    private val fileLogger: FileLogger
) {
    companion object {
        private const val TAG = "InferencePipeline"
    }

    private var isRunning = false
    private var frameCount = 0L
    private var accPipelineMs = 0L  // 100-frame latency accumulator for health log

    // Depth frame-skip: skip depth estimation on intermediate frames to reduce latency
    // on mid/budget-mid devices. lastDepthMap reuses the previous frame's result.
    private var depthFrameCounter = 0
    @Volatile private var lastDepthMap: FloatArray? = null
    @Volatile var depthFrameSkip: Int = 2  // every other frame; service may tighten/loosen after warmup

    // Cross-frame confirmation for unknown obstacles. The on-device scanner previously
    // alerted off a single depth-processed frame; this mirrors the server's confirm-frame
    // gate (alert_logic.py UNKNOWN_CLOSE_CONFIRM_FRM) at reduced cost — 1 hit normally
    // (unchanged behavior), 2 consecutive hits when DepthEstimator flags the frame as
    // low-confidence (narrow raw disparity range — see DepthEstimator.isLowConfidenceFrame).
    private var previousRawAnomalies: List<RectF> = emptyList()

    // Rolling window of YOLO detection bboxes from the last N depth-processed frames.
    // Used as Stage 8 exclusion zones — prevents "unknown obstacle" from firing at the
    // same position where a YOLO-identified object was seen 1-2 frames ago, which happens
    // whenever YOLO briefly misses a real known object due to low FPS / model uncertainty.
    // Size 3 covers ~1 YOLO cycle (1 inference frame + 2 quick render frames ≈ 450ms).
    private val recentYoloBboxHistory = ArrayDeque<List<RectF>>(3)

    /**
     * Process a single camera frame through all 10 stages.
     * Call this on the inference coroutine (not main thread).
     */
    suspend fun processFrame(
        bitmap: Bitmap,
        sensorData: SensorData
    ): FrameResult = withContext(Dispatchers.Default) {
        val startTime = System.currentTimeMillis()

        // Allow exactly one scaleFactor calibration update this frame (first qualifying anchor).
        depthCalibrator.beginFrameCalibration()

        // ── STAGES 3 + 5: YOLO and depth run concurrently ──
        // Depth frame-skip decision is made before launching async work so the counter stays
        // consistent regardless of which coroutine finishes first.
        depthFrameCounter++
        val didRunDepth = depthFrameCounter % depthFrameSkip == 0 || lastDepthMap == null

        val yoloStart = System.currentTimeMillis()
        val yoloDeferred  = async { yoloDetector.detect(bitmap) }
        val depthDeferred = if (didRunDepth) async { depthEstimator.estimateDepth(bitmap) } else null

        // ── STAGE 3 result: await YOLO so 3b/3c run while depth may still be inferring ──
        val rawDetections = yoloDeferred.await()
        val yoloMs = System.currentTimeMillis() - yoloStart

        // ── STAGE 3b: Post-NMS deduplication + ground hazard position filter ──
        //
        // Problem 1 — duplicate same-class detections (3× "computer" for one monitor):
        //   End-to-end NMS uses a fixed IoU threshold; boxes just below that threshold survive.
        //   Sort by confidence desc, keep only the first detection whose IoU with every already-
        //   kept same-class box is ≤ 0.40. This is a second, tighter pass over the NMS output.
        //
        // Problem 2 — ramp/stairs false positives on flat floors:
        //   When the camera is tilted slightly downward, floor texture can fill the entire frame
        //   and be misclassified as a ramp. Guard with: bbox must be in the lower 55 % of the
        //   frame (center Y > 0.45) and must not cover > 85 % of frame width.
        val dedupedDetections = deduplicateSameClass(rawDetections)
        val spatialFiltered = filterGroundHazardPositions(dedupedDetections, sensorData.tiltAngleDegrees)
        val nAfterDedup   = dedupedDetections.size
        val nAfterSpatial = spatialFiltered.size

        // ── STAGE 3c: Temporal consistency filter ──
        //
        // Eliminates single-frame false positives for rare/visually-confused classes
        // (computer, water dispenser, table, chair) without suppressing safety-critical classes.
        // Also resolves the stairs/pothole class confusion: if a zone oscillates between
        // the two across recent frames, the detection is locked to "stairs".
        val filteredRawDetections = temporalFilter.filter(spatialFiltered)
        val nAfterTemporal = filteredRawDetections.size

        // ── STAGE 5 result: depth ran in parallel above; collect now ──
        // depthFrameSkip=1: every frame. =2: every other (default). =3: every third (budget-mid).
        val depthStart = System.currentTimeMillis()
        val depthMap = if (depthDeferred != null) {
            depthDeferred.await().also { if (it != null) lastDepthMap = it }
        } else {
            lastDepthMap
        }
        val depthMs = System.currentTimeMillis() - depthStart

        // ── STAGE 6: Assign depth + classify distance zones ──
        val detectionsWithDepth = filteredRawDetections.map { det ->
            val relativeDepth = if (depthMap != null) {
                // Use mask-weighted depth for classes where bbox over-includes background pixels.
                // Poles: bbox is mostly empty space around a 3-5cm object.
                // Stairs/pothole/curb: bbox edges extend beyond the actual step/edge surface.
                val mask = det.mask
                if (mask != null) {
                    depthEstimator.getDepthWithMask(depthMap, det.bbox, mask)
                } else {
                    depthEstimator.getDepthInRegion(depthMap, det.bbox)
                }
            } else {
                // Fallback: estimate depth from bbox size
                estimateDepthFromBbox(det.bbox)
            }

            val distanceMeters = depthCalibrator.relativeToMetric(
                relativeDepth, det.className, det.bbox
            )
            val zone = DistanceZone.fromDistance(distanceMeters)
            val direction = SpatialDirection.fromNormalizedX(det.bbox.centerX())

            DetectionWithDepth(
                classId = det.classId,
                className = det.className,
                confidence = det.confidence,
                bbox = det.bbox,
                distanceMeters = distanceMeters,
                distanceZone = zone,
                direction = direction
            )
        }

        // ── STAGE 7: Temporal Tracking (moving vs static) ──
        val trackedDetections = temporalTracker.update(detectionsWithDepth)
        val nMovingToward = trackedDetections.count { it.isMovingToward }

        // ── STAGE 8: Unknown Obstacle Detection ──
        // Skip when depth is stale (frame-skipped) — scanning old data produces the same
        // anomalies already announced last frame; wastes ~5-10ms for no new information.
        val unknownObstacles = if (depthMap != null && didRunDepth) {
            // Build extended exclusion zones: current frame + recent history.
            // This prevents Stage 8 from triggering at positions where YOLO identified a
            // known object in the last few frames but missed it this frame (common at low FPS).
            val currentBboxes = trackedDetections.map { it.bbox }
            recentYoloBboxHistory.addFirst(currentBboxes)
            if (recentYoloBboxHistory.size > 3) recentYoloBboxHistory.removeLast()
            val extendedExclusionZones = recentYoloBboxHistory.flatten()

            val rawAnomalies = depthEstimator.findDepthAnomalies(
                depthMap, extendedExclusionZones, scaleFactor = depthCalibrator.scaleFactor
            )

            // Confidence-aware confirmation: MiDaS-Small has no native uncertainty output,
            // so on a low-confidence frame (see field-log-derived threshold in Constants),
            // only anomalies that also appeared in the immediately preceding depth-processed
            // frame are trusted. Normal-confidence frames keep the existing single-frame
            // behavior — this only adds caution, it doesn't change the common case.
            val anomalies = if (depthEstimator.isLowConfidenceFrame) {
                rawAnomalies.filter { rect -> previousRawAnomalies.any { RectF.intersects(it, rect) } }
            } else {
                rawAnomalies
            }
            previousRawAnomalies = rawAnomalies

            anomalies.map { rect ->
                val relDepth = depthEstimator.getDepthInRegion(depthMap, rect)
                val distMeters = depthCalibrator.relativeToMetric(relDepth, "unknown", rect)

                DetectionWithDepth(
                    classId = -1,
                    className = "unknown obstacle",
                    confidence = 0.5f,
                    bbox = rect,
                    distanceMeters = distMeters,
                    distanceZone = DistanceZone.fromDistance(distMeters),
                    direction = SpatialDirection.fromNormalizedX(rect.centerX()),
                    isUnknownObstacle = true
                )
            }
        } else {
            emptyList()
        }

        val allDetections = trackedDetections + unknownObstacles

        // ── STAGE 9: Free Path Calculation ──
        val freePath = freePathCalculator.calculate(allDetections)

        // ── STAGE 10: Priority Queue + Suppression ──
        val rawAlerts = buildAlerts(allDetections, freePath)
        val filteredAlerts = suppressionFilter.filter(rawAlerts)
        val nSuppressed = rawAlerts.size - filteredAlerts.size
        val topAlert = filteredAlerts.minByOrNull { it.priority.level }

        val elapsed = System.currentTimeMillis() - startTime
        frameCount++
        accPipelineMs += elapsed

        // Dense log: every frame for first 100, then every 5
        if (frameCount <= 100L || frameCount % 5L == 0L) {
            val depthLabel = when {
                depthMap == null -> "null"
                didRunDepth      -> "fresh"
                else             -> "stale"
            }
            fileLogger.d(TAG, "Pipeline[$frameCount] ${elapsed}ms: yolo=${yoloMs}ms depth=${depthMs}ms($depthLabel) | " +
                "raw=${rawDetections.size} →dedup=${nAfterDedup} →spatial=${nAfterSpatial} →temporal=${nAfterTemporal} →final=${allDetections.size}")
            if (allDetections.isNotEmpty()) {
                val zoneStr = allDetections.groupBy { it.distanceZone.name }.entries
                    .joinToString(" ") { "${it.key}=${it.value.size}" }
                fileLogger.d(TAG, "Pipeline[$frameCount] zones=[$zoneStr] moving=$nMovingToward unknown=${unknownObstacles.size} | " +
                    "rawAlerts=${rawAlerts.size} suppressed=$nSuppressed active=${filteredAlerts.size} path=${freePath.suggestedDirection}")
                fileLogger.d(TAG, "Pipeline[$frameCount] objs: ${allDetections.map { "${it.className}@${"%.1f".format(it.distanceMeters)}m/${it.distanceZone.name}/${it.direction.name}" }.joinToString()}")
            } else {
                fileLogger.d(TAG, "Pipeline[$frameCount] NO_DETECTIONS | depth=$depthLabel skip=${!didRunDepth} path=${freePath.suggestedDirection}")
            }
        }

        // 100-frame health summary: avg latency + bbox-depth quality
        if (frameCount % 100L == 0L) {
            val avgMs = accPipelineMs / 100L
            val prioritySummary = filteredAlerts.groupBy { it.priority.name }
                .entries.joinToString(" ") { "${it.key}×${it.value.size}" }
                .ifEmpty { "none" }
            fileLogger.i(TAG, "Pipeline[100f-avg] avg=${avgMs}ms frames=$frameCount depthSkip=$depthFrameSkip scaleFactor=${"%.2f".format(depthCalibrator.scaleFactor)} | activeAlerts=[$prioritySummary]")
            accPipelineMs = 0L
        }

        FrameResult(
            detections = allDetections,
            freePath = freePath,
            alerts = filteredAlerts,
            topAlert = topAlert,
            frameTimestampMs = System.currentTimeMillis(),
            inferenceTimeMs = elapsed
        )
    }

    /**
     * Build alert list from detections and free path analysis.
     */
    private fun buildAlerts(
        detections: List<DetectionWithDepth>,
        freePath: FreePathResult
    ): List<NovaAlert> {
        val alerts = mutableListOf<NovaAlert>()

        // Alerts from detections
        for (det in detections) {
            val priority = AlertPriority.fromDistance(
                det.distanceMeters,
                ClassConfig.isSafetyCritical(det.className)
            )

            // FAR zone (>5m): only alert if actively approaching
            if (det.distanceZone == DistanceZone.FAR && !det.isMovingToward) continue
            // CAUTION zone (3–5m): suppress ONLY context-only classes (projectors, whiteboards,
            // computers) that are static — they have no path-blocking relevance at 3–5m.
            // OBSTACLE classes (chairs, tables, sofa) ARE announced in CAUTION zone so the VIP
            // gets advance warning before the object enters the WARNING/DANGER zone.
            if (det.distanceZone == DistanceZone.CAUTION &&
                !det.isMovingToward &&
                !det.isUnknownObstacle &&
                !ClassConfig.isSafetyCritical(det.className) &&
                !ClassConfig.isNavigational(det.className) &&
                det.className in ClassConfig.CONTEXT_ONLY) continue

            val haptic = when {
                det.isUnknownObstacle -> HapticPattern.HEARTBEAT
                det.isMovingToward -> HapticPattern.TRIPLE_PULSE
                det.distanceZone == DistanceZone.DANGER -> HapticPattern.BUZZ
                det.distanceZone == DistanceZone.WARNING -> HapticPattern.DOUBLE_TAP
                det.distanceZone == DistanceZone.CAUTION -> HapticPattern.PING
                else -> HapticPattern.PING   // FAR / moving-toward
            }

            alerts.add(NovaAlert(
                priority = if (det.isMovingToward) AlertPriority.P1_EMERGENCY else priority,
                spokenText = phraseGenerator.generatePhrase(det),
                direction = det.direction,
                distanceMeters = det.distanceMeters,
                hapticPattern = haptic,
                sourceClassName = det.className
            ))
        }

        // Direction guidance alert
        if (!freePath.isPathClear) {
            alerts.add(NovaAlert(
                priority = AlertPriority.P3_DIRECTION,
                spokenText = freePath.suggestedDirection.toSpokenGuidance(),
                direction = SpatialDirection.CENTER,
                distanceMeters = 0f,
                hapticPattern = HapticPattern.PING,
                isPathGuidance = true
            ))
        }

        return alerts
    }

    /**
     * Second-pass same-class deduplication after end-to-end NMS.
     *
     * The model's built-in NMS uses a fixed IoU threshold (e.g. 0.45). Boxes that are nearly
     * the same object but whose IoU falls just below that threshold both survive. This pass
     * sorts by confidence descending and suppresses any box that overlaps an already-kept box
     * of the SAME class with IoU > 0.40.
     *
     * This is what fixes "3× computer at 3.4m / 3.5m / 3.7m" for a single monitor.
     */
    private fun deduplicateSameClass(detections: List<RawDetection>): List<RawDetection> {
        val sorted = detections.sortedByDescending { it.confidence }
        val kept = mutableListOf<RawDetection>()
        for (det in sorted) {
            val isDuplicate = kept.any { k ->
                k.className == det.className && iou(k.bbox, det.bbox) > 0.40f
            }
            if (!isDuplicate) kept.add(det)
        }
        return kept
    }

    /**
     * Removes ground-hazard false positives caused by floor-texture patterns.
     *
     * Guards applied in order:
     *   1. Vertical position: bbox centre Y must be in the lower 65% of frame (> 0.35).
     *   2. Width cap: bbox must not cover > 85% of frame width (floor-spanning = FP).
     *   3. Ramp tilt compensation: downward camera tilt makes flat floor look like a ramp.
     *      Raise the confidence requirement proportionally to tilt magnitude so that
     *      borderline detections during tilt are suppressed.
     */
    private fun filterGroundHazardPositions(
        detections: List<RawDetection>,
        tiltDeg: Float
    ): List<RawDetection> {
        val absTilt = kotlin.math.abs(tiltDeg)
        return detections.filter { det ->
            if (!ClassConfig.isGroundHazard(det.className)) return@filter true
            if (det.bbox.centerY() <= 0.35f || det.bbox.width() >= 0.85f) return@filter false
            if (det.className == "ramp") {
                when {
                    absTilt > NovaConstants.RAMP_TILT_HIGH_DEG -> det.confidence >= NovaConstants.RAMP_TILT_HIGH_CONF
                    absTilt > NovaConstants.RAMP_TILT_LOW_DEG  -> det.confidence >= NovaConstants.RAMP_TILT_MID_CONF
                    else -> true
                }
            } else {
                true
            }
        }
    }


    /**
     * Fallback depth estimation from bounding box size.
     * Larger bbox = closer object.
     */
    private fun estimateDepthFromBbox(bbox: RectF): Float {
        val area = bbox.width() * bbox.height()
        // Heuristic: bbox area 0.5 = very close, 0.01 = far
        return (1f - area.coerceIn(0f, 1f))
    }

    fun initialize(classNames: List<String>) {
        fileLogger.i(TAG, "=== NOVA PIPELINE INIT ===")
        fileLogger.i(TAG, "Class set: ${classNames.size} classes — [${classNames.take(8).joinToString()}${if (classNames.size > 8) "..." else ""}]")
        yoloDetector.initialize(classNames)
        depthEstimator.initialize()

        // Warm up both models
        yoloDetector.warmup()
        depthEstimator.warmup()

        isRunning = true
        fileLogger.i(TAG, "=== PIPELINE READY — depthFrameSkip=$depthFrameSkip ===")
    }

    /**
     * Reset the suppression filter — call on RESUME after a pause so objects near the user
     * that were already announced before the pause get re-announced immediately.
     */
    fun resetSuppression() {
        suppressionFilter.reset()
        temporalFilter.reset()
    }

    fun shutdown() {
        isRunning = false
        yoloDetector.close()
        depthEstimator.close()
    }
}

// ponytail: package-level so InferencePipeline and TemporalTracker share one copy
private fun iou(a: RectF, b: RectF): Float {
    val iL = maxOf(a.left, b.left); val iT = maxOf(a.top, b.top)
    val iR = minOf(a.right, b.right); val iB = minOf(a.bottom, b.bottom)
    if (iL >= iR || iT >= iB) return 0f
    val iArea = (iR - iL) * (iB - iT)
    val uArea = a.width() * a.height() + b.width() * b.height() - iArea
    return if (uArea > 0f) iArea / uArea else 0f
}

// ─────────────────────────────────────────────────────────
// Supporting pipeline components
// ─────────────────────────────────────────────────────────

/**
 * Converts depth-model relative depth to metric distance (meters).
 */
@Singleton
class DepthCalibrator @Inject constructor() {

    // Known average heights of reference objects (meters) — 28 NOVA classes (Run4).
    // Ground hazards (stairs, curb, speed-breaker, ramp) use step/edge height, not total structure height.
    private val KNOWN_HEIGHTS = mapOf(
        // People + navigation
        "person"              to 1.70f,
        "door"                to 2.10f,
        "bench"               to 0.80f,
        // Vehicles
        "car"                 to 1.50f,
        "bus"                 to 3.20f,
        "truck"               to 2.50f,
        "bicycle"             to 1.00f,
        // Furniture
        "chair"               to 0.90f,
        "empty-chair"         to 0.90f,
        "occupied-chair"      to 0.90f,
        "table"               to 0.75f,
        "sofa"                to 0.85f,
        // Structural
        "pole"                to 3.50f,
        "fence"               to 1.20f,
        "wall"                to 2.40f,
        "white-board"         to 1.20f,
        // Ground hazards — bbox height ≈ visible edge, not physical height.
        // Still used for distance estimation; scale factor update is skipped for these.
        "stairs"              to 0.20f,
        "ramp"                to 0.30f,
        "speed-breaker"       to 0.15f,
        "curb"                to 0.12f,
        "pothole"             to 0.10f,
        // Vegetation
        "tree"                to 3.00f,
        "vegitation"          to 1.00f,
        "pot-plant"           to 0.60f,
        // Office / context
        "computer"            to 0.50f,
        "projector"           to 0.30f,
        "fire-extinguisher"   to 0.60f,
        "water dispenser"     to 1.50f,
    )

    // Ground-level objects: bbox shows only visible edge, not full object height.
    // Distance estimation (height-based) is still done, but these objects must NOT
    // update the global scaleFactor — their unrepresentative bbox heights would corrupt
    // depth estimates for all other objects in the scene.
    // ponytail: ClassConfig.GROUND_HAZARD is the single source of truth; no inline copy
    private val SKIP_SCALE_UPDATE = ClassConfig.GROUND_HAZARD

    // Calibrated scale factor (updated from reference objects via EMA)
    // Default: empirically tuned so relativeDepth≈0.7 (typical indoor 2m scene) ≈ 2.1m
    // internal: exposed for pipeline health logging
    internal var scaleFactor = 3.0f

    // Per-frame calibration lock — only the FIRST qualifying anchor each frame updates
    // scaleFactor (mirrors the server's begin_frame_calibration). Prevents multiple
    // detections in one frame from compounding the EMA. Reset each frame by beginFrameCalibration().
    private var frameCalibLocked = false

    /** Call once at the start of each frame, before assigning depth to detections. */
    fun beginFrameCalibration() {
        frameCalibLocked = false
    }

    /**
     * Convert depth-model relative depth to metric distance (meters).
     *
     * IMPORTANT: MiDaS/DA V2 outputs INVERSE depth (disparity).
     * After normalization to 0-1: HIGH value = CLOSE, LOW value = FAR.
     * We invert (1 - relativeDepth) so that small values → small distances.
     *
     * Calibration fixes:
     *   1. Min bbox guard 0.05: small/partial boxes give wildly inaccurate estimates.
     *   2. Max distance sanity clamp (< 8m): reject physically impossible estimates.
     *   3. EMA smoothing on scaleFactor (α=0.25): one bad detection can't corrupt all estimates.
     *   4. FOV factor 1.3: tan(32.5°)×2 ≈ 1.27 for ~65° vertical FOV (typical phone camera).
     *   5. SKIP_SCALE_UPDATE: ground hazards excluded from scale factor updates.
     */
    fun relativeToMetric(relativeDepth: Float, className: String, bbox: RectF): Float {
        val invertedDepth = 1f - relativeDepth

        KNOWN_HEIGHTS[className]?.let { knownHeight ->
            val bboxHeight = bbox.height()
            if (bboxHeight > 0.05f) {
                val estimatedDistance = knownHeight / (bboxHeight * 1.3f)

                // Update scaleFactor only from a well-visible, non-ground-hazard anchor, and
                // only once per frame. Guards ported from the server's depth_to_meters() —
                // the on-device path previously had none, so a single mis-sized bbox could
                // swing scaleFactor and corrupt the metric distance of every other object:
                //   • outlier reject: raw scale must lie within [scale/3.5, scale*3.5]
                //   • slow EMA (α=0.10) + tight ±1.25×/step cap
                //   • hard clamp to [1.0, 12.0]
                if (!frameCalibLocked &&
                    className !in SKIP_SCALE_UPDATE &&
                    bboxHeight > 0.15f && estimatedDistance < 8f && invertedDepth > 0.01f
                ) {
                    val rawScale = estimatedDistance / invertedDepth
                    if (rawScale in (scaleFactor / 3.5f)..(scaleFactor * 3.5f)) {
                        val newScale = 0.90f * scaleFactor + 0.10f * rawScale
                        scaleFactor = newScale
                            .coerceIn(scaleFactor / 1.25f, scaleFactor * 1.25f)
                            .coerceIn(1.0f, 12.0f)
                        frameCalibLocked = true   // one update per frame
                    }
                    // else: outlier rejected — leave scaleFactor unchanged
                }
                return estimatedDistance.coerceIn(0.1f, 20f)
            }
        }

        val meters = invertedDepth * scaleFactor
        return meters.coerceIn(0.1f, 20f)
    }
}

/**
 * Tracks objects across frames to detect movement and smooth depth estimates.
 */
@Singleton
class TemporalTracker @Inject constructor() {

    private var previousDetections: List<DetectionWithDepth> = emptyList()

    // EMA-smoothed depth per (className, direction) slot.
    // Key: "${className}_${direction.name}" — stable across frames for the same object type/zone.
    // Bounded: 29 classes × 5 directions = 145 max slots.
    private val smoothedDepths = mutableMapOf<String, Float>()

    fun update(current: List<DetectionWithDepth>): List<DetectionWithDepth> {
        val activeKeys = mutableSetOf<String>()

        val tracked = current.map { det ->
            val match = previousDetections.find { prev ->
                prev.className == det.className && iou(prev.bbox, det.bbox) > 0.3f
            }

            // EMA depth smoothing: α=0.3 damps frame-to-frame noise while responding
            // to genuine movement within ~10 frames at 5–10 fps.
            val depthKey = "${det.className}_${det.direction.name}"
            activeKeys.add(depthKey)

            val rawDist = det.distanceMeters
            val smoothedDist = if (match != null) {
                val prev = smoothedDepths[depthKey] ?: rawDist
                (0.7f * prev + 0.3f * rawDist).coerceIn(0.1f, 20f)
            } else {
                rawDist
            }
            smoothedDepths[depthKey] = smoothedDist

            if (match != null) {
                val currentArea = det.bbox.width() * det.bbox.height()
                val prevArea = match.bbox.width() * match.bbox.height()
                val areaRatio = if (prevArea > 0) currentArea / prevArea else 1f

                det.copy(
                    distanceMeters = smoothedDist,
                    distanceZone = DistanceZone.fromDistance(smoothedDist),
                    isMovingToward = areaRatio > NovaConstants.BBOX_GROWTH_THRESHOLD,
                    approachSpeed = areaRatio - 1f
                )
            } else {
                det.copy(
                    distanceMeters = smoothedDist,
                    distanceZone = DistanceZone.fromDistance(smoothedDist)
                )
            }
        }

        // Evict depth history for objects absent this frame to keep map bounded.
        smoothedDepths.keys.retainAll { it in activeKeys }

        previousDetections = current
        return tracked
    }

}

/**
 * Calculates which zones are clear/blocked for navigation guidance.
 */
@Singleton
class FreePathCalculator @Inject constructor() {

    // ponytail: lateral hysteresis state — suppress MOVE_LEFT↔MOVE_RIGHT oscillation
    private var lastLateralGuidance = NavigationGuidance.CONTINUE_FORWARD
    private var lastLateralChangeMs = 0L

    fun calculate(detections: List<DetectionWithDepth>): FreePathResult {
        val zones = mutableMapOf<SpatialDirection, ZoneStatus>()
        SpatialDirection.entries.forEach { zones[it] = ZoneStatus.CLEAR }

        for (det in detections) {
            if (det.distanceZone == DistanceZone.FAR && !det.isMovingToward) continue
            val status = when (det.distanceZone) {
                DistanceZone.DANGER  -> ZoneStatus.BLOCKED
                DistanceZone.WARNING -> ZoneStatus.CAUTION
                DistanceZone.CAUTION -> ZoneStatus.CAUTION
                DistanceZone.FAR     -> if (det.isMovingToward) ZoneStatus.CAUTION else continue
            }
            val currentStatus = zones[det.direction] ?: ZoneStatus.CLEAR
            if (status.ordinal > currentStatus.ordinal) zones[det.direction] = status
        }

        // Fold FAR_LEFT/FAR_RIGHT into lateral zones for guidance decisions
        val center = zones[SpatialDirection.CENTER] ?: ZoneStatus.CLEAR
        val leftStatus = run {
            val l = zones[SpatialDirection.LEFT] ?: ZoneStatus.CLEAR
            val fl = zones[SpatialDirection.FAR_LEFT] ?: ZoneStatus.CLEAR
            if (l.ordinal >= fl.ordinal) l else fl
        }
        val rightStatus = run {
            val r = zones[SpatialDirection.RIGHT] ?: ZoneStatus.CLEAR
            val fr = zones[SpatialDirection.FAR_RIGHT] ?: ZoneStatus.CLEAR
            if (r.ordinal >= fr.ordinal) r else fr
        }

        val rawGuidance = when {
            center == ZoneStatus.CLEAR   -> NavigationGuidance.CONTINUE_FORWARD
            center == ZoneStatus.CAUTION -> NavigationGuidance.SLOW_DOWN
            leftStatus == ZoneStatus.CLEAR && rightStatus == ZoneStatus.CLEAR ->
                NavigationGuidance.MOVE_LEFT    // both clear: default left
            leftStatus == ZoneStatus.CLEAR   -> NavigationGuidance.MOVE_LEFT
            rightStatus == ZoneStatus.CLEAR  -> NavigationGuidance.MOVE_RIGHT
            // Sides only CAUTION (not BLOCKED) — hazards exist laterally but passable slowly
            leftStatus != ZoneStatus.BLOCKED || rightStatus != ZoneStatus.BLOCKED ->
                NavigationGuidance.SLOW_DOWN
            else -> NavigationGuidance.STOP_ALL_BLOCKED
        }

        // Hysteresis: suppress MOVE_LEFT↔MOVE_RIGHT flips within commitment window.
        // STOP and all other escalation transitions pass through immediately.
        val now = System.currentTimeMillis()
        val isLateralFlip =
            (rawGuidance == NavigationGuidance.MOVE_LEFT  && lastLateralGuidance == NavigationGuidance.MOVE_RIGHT) ||
            (rawGuidance == NavigationGuidance.MOVE_RIGHT && lastLateralGuidance == NavigationGuidance.MOVE_LEFT)
        val guidance = if (isLateralFlip && now - lastLateralChangeMs < NovaConstants.DIRECTION_COMMITMENT_MS) {
            lastLateralGuidance
        } else {
            if (rawGuidance != lastLateralGuidance &&
                (rawGuidance == NavigationGuidance.MOVE_LEFT || rawGuidance == NavigationGuidance.MOVE_RIGHT)) {
                lastLateralGuidance = rawGuidance
                lastLateralChangeMs = now
            }
            rawGuidance
        }

        return FreePathResult(
            zones = zones,
            suggestedDirection = guidance,
            isPathClear = center == ZoneStatus.CLEAR
        )
    }
}

/**
 * Priority queue — sorts alerts and picks the most urgent.
 */
@Singleton
class PriorityAlertQueue @Inject constructor() {
    fun prioritize(alerts: List<NovaAlert>): List<NovaAlert> =
        alerts.sortedBy { it.priority.level }
}

/**
 * Suppresses repeated alerts for the same static objects.
 */
@Singleton
class SuppressionFilter @Inject constructor() {

    // Key: className+direction → last announcement timestamp
    private val lastAnnounced = mutableMapOf<String, Long>()

    fun filter(alerts: List<NovaAlert>): List<NovaAlert> {
        val now = System.currentTimeMillis()
        return alerts.filter { alert ->
            val key = "${alert.sourceClassName}_${alert.direction}"
            // C4/M3: tiered cooldowns — P1 re-announces every 800ms (was never suppressed →
            // caused 10 TTS flushes/sec at 10 FPS; now has minimum interval).
            val cooldown = when {
                alert.isPathGuidance                            -> NovaConstants.SUPPRESSION_P3_MS
                // Unknown obstacles get a longer cooldown than real P1 hazards: they are
                // depth-scan results (no semantic class, confidence always 0.5) and fire on
                // every YOLO cycle if an object is persistently close. 4s limits TTS chatter
                // while still alerting when the situation genuinely changes. Named P1 hazards
                // (stairs at 1m, car at 1m) keep the 800ms re-alert interval.
                alert.sourceClassName == "unknown obstacle"     -> NovaConstants.SUPPRESSION_UNKNOWN_MS
                alert.priority == AlertPriority.P1_EMERGENCY   -> NovaConstants.SUPPRESSION_P1_MS
                alert.priority == AlertPriority.P2_DANGER      -> NovaConstants.SUPPRESSION_P2_MS
                alert.priority == AlertPriority.P3_DIRECTION   -> NovaConstants.SUPPRESSION_P3_MS
                else                                           -> NovaConstants.SUPPRESSION_P4_MS
            }

            val lastTime = lastAnnounced[key] ?: 0L
            if (now - lastTime > cooldown) {
                lastAnnounced[key] = now
                true
            } else {
                false
            }
        }
    }

    fun reset() {
        lastAnnounced.clear()
    }
}

/**
 * Generates human-readable phrases from detections.
 *
 * D1: Direction-first format — direction is spoken first so VIP can immediately
 * orient before the rest of the phrase completes.
 * Example: "10 o'clock. Stop! Pole. 2 steps." (not "Stop! Pole at 10 o'clock, 1.5 metres.")
 *
 * F9b: Distance ≤ 3m expressed in steps (1 step ≈ 0.75m) — more actionable for cane users.
 * B1: Stairs get a direction hint (step down / step up) based on bbox vertical position.
 */
@Singleton
class AlertPhraseGenerator @Inject constructor() {

    private fun formatDistance(metres: Float, zone: DistanceZone): String {
        return when (zone) {
            DistanceZone.DANGER, DistanceZone.WARNING, DistanceZone.CAUTION -> {
                val steps = (metres / 0.75f).roundToInt().coerceAtLeast(1)
                if (steps == 1) "1 step" else "$steps steps"
            }
            else -> "%.1f metres".format(metres)
        }
    }

    fun generatePhrase(detection: DetectionWithDepth): String {
        // "vegitation" is the model's misspelled class name — map to correct spoken form.
        val rawName = if (detection.className == "vegitation") "vegetation" else detection.className
        val obj  = rawName.replace("-", " ").replace("_", " ")
        val dir  = detection.direction.toSpokenDirection()
        val dist = formatDistance(detection.distanceMeters, detection.distanceZone)

        // B1: Stairs vertical-position heuristic.
        // Lower frame (bottom > 0.75) = camera looking down = step descends away from user.
        // Upper frame (top < 0.25) = camera looking up = stairs ascending ahead.
        val stairsHint = if (detection.className == "stairs") {
            when {
                detection.bbox.bottom > 0.75f -> " step down"
                detection.bbox.top < 0.25f    -> " step up"
                else                           -> ""
            }
        } else ""

        return when {
            detection.isUnknownObstacle && detection.distanceZone == DistanceZone.DANGER ->
                "$dir. Unknown obstacle. Stop. $dist."

            detection.isUnknownObstacle ->
                "$dir. Unknown obstacle. $dist."

            detection.isMovingToward && detection.distanceZone == DistanceZone.DANGER ->
                "$dir. $obj approaching. Stop. $dist."

            detection.isMovingToward ->
                "$dir. $obj approaching. $dist."

            // NAVIGATION classes (door, bench, person) are navigable targets — no "Stop!"
            detection.distanceZone == DistanceZone.DANGER && ClassConfig.isNavigable(detection.className) ->
                "$dir. $obj. $dist."

            // "Stop!" only for hazards that justify it: structural barriers, vehicles, ground hazards.
            // OBSTACLE classes (chair, sofa, pot-plant, tree…) in DANGER zone get urgent tone
            // without "Stop!" — a pot plant at 1 step does not need a stop command.
            detection.distanceZone == DistanceZone.DANGER &&
                (ClassConfig.isSafetyCritical(detection.className) || ClassConfig.isGroundHazard(detection.className)) ->
                "$dir. Stop! $obj$stairsHint. $dist."

            detection.distanceZone == DistanceZone.DANGER ->
                "$dir. $obj. $dist."

            detection.distanceZone == DistanceZone.WARNING ->
                "$dir. $obj$stairsHint. $dist."

            detection.distanceZone == DistanceZone.CAUTION ->
                "$dir. $obj ahead. $dist."

            else ->
                "$dir. $obj. $dist."
        }
    }
}
