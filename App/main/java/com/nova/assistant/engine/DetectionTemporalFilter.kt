package com.nova.assistant.engine

import android.graphics.RectF
import com.nova.assistant.util.ClassConfig
import com.nova.assistant.util.FileLogger
import javax.inject.Inject
import javax.inject.Singleton
import kotlin.math.abs

/**
 * Temporal consistency filter — eliminates one-off false positives without touching safety classes.
 *
 * Root cause: mAP50=0.600 produces frequent single-frame false positives for rare/confused classes
 * (computer, water dispenser, table, pole) due to visual feature sharing between classes.
 *
 * Strategy:
 *  - GROUND_HAZARD (stairs, pothole, curb, ramp, speed-breaker): always pass immediately.
 *    False negative cost (missed hazard) outweighs false positive cost (extra announcement).
 *  - SAFETY_CRITICAL (pole, fence, wall, car, bicycle, bus, truck): always pass immediately.
 *  - OBSTACLE + CONTEXT_ONLY (chair, table, computer, water dispenser, etc.):
 *    confidence-weighted confirmation (2026-07-14) — confidences of same-zone same-class hits
 *    across the history window are summed; the current detection passes once that sum clears
 *    CONFIDENCE_ACCUMULATION_THRESHOLD, OR immediately if its own confidence alone clears
 *    HIGH_CONF_BYPASS. Three moderate hits (e.g. 0.42+0.71+0.63=1.76) now confirm as reliably as
 *    one strong hit, and tolerate an occasional miss without resetting — the binary "≥1 prior hit"
 *    rule this replaces would drop a real object outright if any single frame missed it.
 *
 * Stairs/pothole disambiguation:
 *  Stairs and potholes share near-identical visual features from a downward-angled camera
 *  (horizontal dark band on a gray surface at ground level). When a zone oscillates between
 *  the two classes across recent frames, the current "pothole" detection is overridden to
 *  "stairs" — stairs is the more specific and more dangerous class (action required).
 *
 * Thread-safety: single-threaded. Must only be called from the inference coroutine.
 */
@Singleton
class DetectionTemporalFilter @Inject constructor(
    private val fileLogger: FileLogger
) {
    companion object {
        private const val TAG = "TemporalFilter"

        // Rolling window depth: maximum stored frames regardless of time.
        private const val WINDOW_SIZE = 5

        // Time window for history lookback. Frames older than this are ignored when
        // counting prior hits. Replaces pure frame-count logic: at 0.33 FPS a 5-frame
        // window spans 15 seconds — stale history would incorrectly confirm or conflict.
        // At 2 FPS (post-320px fix) this covers ~1.5s of real history.
        private const val HISTORY_WINDOW_MS = 3000L

        // Normalized bbox-center distance threshold for "same spatial zone".
        // Widened from 0.20 to 0.30: at low FPS objects can shift more between frames.
        private const val ZONE_RADIUS = 0.30f

        // High-confidence threshold: detections above this pass without history confirmation.
        // A confident first-appearance is more likely real than a spurious single-frame hit.
        private const val HIGH_CONF_BYPASS = 0.55f

        // Confidence-weighted confirmation: sum of same-zone same-class confidences across the
        // window (current frame's own confidence included) must clear this to pass. Tuned so two
        // moderate hits (~0.45 each) confirm, but one hit alone — even a fairly solid ~0.7 — does
        // not, preserving the "needs more than a single frame" guarantee the old count rule gave.
        private const val CONFIDENCE_ACCUMULATION_THRESHOLD = 0.90f

        // Classes that must appear in ≥1 previous frame before being forwarded.
        // Populated lazily on first use so ClassConfig flag is resolved first.
        private val FILTER_CLASSES: Set<String> by lazy {
            ClassConfig.CONTEXT_ONLY + ClassConfig.OBSTACLE
        }
    }

    // Index 0 = most recent frame, paired with its wall-clock timestamp.
    // Capacity capped at WINDOW_SIZE; time-based expiry applied at query time.
    private val frameHistory = ArrayDeque<Pair<Long, List<RawDetection>>>(WINDOW_SIZE)

    /**
     * Filter [current] frame detections using the rolling history window.
     *
     * Call once per frame, in order. Must NOT be called concurrently.
     * Returns a subset of [current] with confirmed detections only.
     */
    fun filter(current: List<RawDetection>): List<RawDetection> {
        val now = System.currentTimeMillis()

        // Maintain fixed-size rolling window with timestamps
        if (frameHistory.size >= WINDOW_SIZE) frameHistory.removeLast()
        frameHistory.addFirst(Pair(now, current))

        // Insufficient recent history — pass everything through until at least one prior
        // frame exists within the time window (avoids stale frames from a previous session
        // being counted as confirmation for the current session).
        val recentPriorFrames = frameHistory.drop(1).count { (ts, _) -> now - ts <= HISTORY_WINDOW_MS }
        if (recentPriorFrames == 0) return current

        val kept = mutableListOf<RawDetection>()
        var filteredCount = 0
        var disambiguatedCount = 0

        for (det in current) {
            when {
                // Ground hazards and safety-critical: never suppress, resolve stairs/pothole conflict
                ClassConfig.isGroundHazard(det.className) || ClassConfig.isSafetyCritical(det.className) -> {
                    val resolved = resolveStairsPotholeConflict(det)
                    if (resolved.className != det.className) disambiguatedCount++
                    kept.add(resolved)
                }

                // Obstacle/context: require accumulated-confidence confirmation, or bypass for a
                // single high-confidence hit. High-confidence first appearances are more likely
                // real than spurious single-frame noise; at low FPS the history window may simply
                // not have seen the object yet.
                det.className in FILTER_CLASSES -> {
                    val accumConf = accumulatedConfidence(det.className, det.bbox, det.confidence)
                    if (accumConf >= CONFIDENCE_ACCUMULATION_THRESHOLD || det.confidence >= HIGH_CONF_BYPASS) {
                        kept.add(det)
                    } else {
                        filteredCount++
                    }
                }

                // Navigation and anything else — pass through
                else -> kept.add(det)
            }
        }

        if (filteredCount > 0 || disambiguatedCount > 0) {
            fileLogger.d(TAG, "in=${current.size} kept=${kept.size} filtered=$filteredCount disambig=$disambiguatedCount")
        }

        return kept
    }

    /**
     * If a "pothole" detection's spatial zone also contained "stairs" in any recent frame,
     * override to "stairs". Both classes produce the same visual signature from a ground-level
     * camera angle; stairs is more specific and more dangerous.
     */
    private fun resolveStairsPotholeConflict(det: RawDetection): RawDetection {
        if (det.className != "pothole") return det

        val now = System.currentTimeMillis()
        // Skip index 0 (current frame itself) — check frames 1..N within time window
        val stairsSeenNearby = (1 until frameHistory.size).any { i ->
            val (ts, dets) = frameHistory[i]
            now - ts <= HISTORY_WINDOW_MS &&
            dets.any { other -> other.className == "stairs" && isSameZone(det.bbox, other.bbox) }
        }

        return if (stairsSeenNearby) {
            fileLogger.d(TAG, "pothole→stairs at (%.2f,%.2f) — stairs seen in same zone last frame".format(
                det.bbox.centerX(), det.bbox.centerY()
            ))
            det.copy(className = "stairs")
        } else {
            det
        }
    }

    /**
     * Sum [currentConfidence] with the best same-class same-zone confidence from each history
     * frame (indexes 1..N, i.e. excluding the current frame at 0) within [HISTORY_WINDOW_MS].
     * Takes the max per frame (not a sum) so a single frame with duplicate same-zone boxes
     * can't inflate the total beyond what one real detection per frame would contribute.
     */
    private fun accumulatedConfidence(className: String, bbox: RectF, currentConfidence: Float): Float {
        val now = System.currentTimeMillis()
        var sum = currentConfidence
        for (i in 1 until frameHistory.size) {
            val (ts, dets) = frameHistory[i]
            if (now - ts > HISTORY_WINDOW_MS) continue  // ignore frames outside time window
            val bestMatch = dets.filter { it.className == className && isSameZone(bbox, it.bbox) }
                .maxOfOrNull { it.confidence }
            if (bestMatch != null) sum += bestMatch
        }
        return sum
    }

    private fun isSameZone(a: RectF, b: RectF): Boolean =
        abs(a.centerX() - b.centerX()) < ZONE_RADIUS &&
        abs(a.centerY() - b.centerY()) < ZONE_RADIUS

    /** Reset history — call when the pipeline restarts or camera feed is interrupted. */
    fun reset() {
        frameHistory.clear()
    }
}
