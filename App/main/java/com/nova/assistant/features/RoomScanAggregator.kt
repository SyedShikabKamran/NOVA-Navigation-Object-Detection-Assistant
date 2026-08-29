package com.nova.assistant.features

import kotlin.math.abs

/**
 * RoomScanAggregator — counts DISTINCT physical objects in a room scan.
 *
 * WHY: the old aggregation set `count` = number of frames a (class, direction) appeared in,
 * then announced it as an object count. One chair held in view for 7 frames became
 * "7 chairs at 2 o'clock" — the app told a blind user about furniture that does not exist.
 *
 * HOW: greedy IoU + distance-gated track-by-detection (SORT minus the Kalman filter — a slow
 * look-around scan has no object motion to model, so the predictor would add tuning for zero
 * gain). Across the scan's frames, each physical object becomes one track; `count` per class =
 * number of distinct tracks, not frame appearances. A track is associated to a detection when
 * their bboxes overlap (IoU) AND their monocular depths are close — the depth gate separates a
 * near chair from a far chair that line up in-image. A direction bucket is coarse (5 buckets),
 * so association uses the continuous bbox, NOT the direction.
 *
 * Pure Kotlin, no Android types (bbox passed as plain floats) → JVM unit-testable
 * (RoomScanAggregatorTest), important because a wrong count misinforms a non-visual user.
 *
 * ASSUMPTION (stated): the scan is slow enough that the same object's bbox overlaps frame-to-
 * frame (IoU >= threshold). For a deliberate look-around this is the normal case. Residual
 * limitation: two adjacent same-class objects at the same distance can merge into one track
 * (under-count) — and under-count is the safe direction for an assistive announcement.
 */

/** One detection in a scan frame — plain floats so this stays Android-free and unit-testable. */
data class ScanDetection(
    val className: String,
    val left: Float, val top: Float, val right: Float, val bottom: Float, // normalized 0..1
    val distanceMeters: Float,
    val direction: String,   // pre-computed spoken bucket, e.g. "2 o'clock"
    val confidence: Float,
)

/** One row of the room inventory — same shape the JSON contract + loadRoomById already expect. */
data class InventoryItem(val className: String, val direction: String, val count: Int)

object RoomScanAggregator {
    const val IOU_THRESHOLD = 0.30f       // min bbox overlap to call it the same object
    const val DISTANCE_GATE_M = 1.5f      // max depth difference to associate (generous — depth is noisy)
    const val MIN_FRAMES = 3              // a real object persists; single-frame blips are dropped
    const val TRACK_MAX_GAP = 5           // don't match a track unseen for >N frames (avoids stale cross-matches)
    const val STAIRS_CONF_FLOOR = 0.60f   // stairs need higher confidence (indoor false-positive prone)

    private class Track(
        val className: String,
        var l: Float, var t: Float, var r: Float, var b: Float,
        var lastDist: Float,
        var minDist: Float,          // closest observation — its bearing is the most reliable
        var minDistDir: String,
        var maxConf: Float,
        var frames: Int,
        var lastFrameIdx: Int,
    )

    /**
     * @param frames detections grouped per scan frame, in capture order.
     * @return distinct-object inventory: one entry per (class, direction) with count = distinct objects.
     */
    fun aggregate(frames: List<List<ScanDetection>>): List<InventoryItem> {
        val tracks = ArrayList<Track>()

        frames.forEachIndexed { frameIdx, frame ->
            val used = HashSet<Track>()
            // Process strongest detections first so they claim their track before weaker overlaps.
            for (det in frame.sortedByDescending { it.confidence }) {
                var best: Track? = null
                var bestIoU = IOU_THRESHOLD
                for (track in tracks) {
                    if (track.className != det.className || track in used) continue
                    if (frameIdx - track.lastFrameIdx > TRACK_MAX_GAP) continue       // too stale to be the same object
                    if (abs(track.lastDist - det.distanceMeters) > DISTANCE_GATE_M) continue
                    val iou = iou(track, det)
                    if (iou >= bestIoU) { bestIoU = iou; best = track }
                }
                if (best != null) {
                    best.l = det.left; best.t = det.top; best.r = det.right; best.b = det.bottom
                    best.lastDist = det.distanceMeters
                    best.frames += 1
                    best.lastFrameIdx = frameIdx
                    best.maxConf = maxOf(best.maxConf, det.confidence)
                    if (det.distanceMeters < best.minDist) {
                        best.minDist = det.distanceMeters
                        best.minDistDir = det.direction
                    }
                    used.add(best)
                } else {
                    val nt = Track(
                        det.className, det.left, det.top, det.right, det.bottom,
                        det.distanceMeters, det.distanceMeters, det.direction,
                        det.confidence, frames = 1, lastFrameIdx = frameIdx,
                    )
                    tracks.add(nt)
                    used.add(nt)
                }
            }
        }

        // Keep persistent tracks (drops single-frame blips); stairs demand a higher confidence floor.
        val kept = tracks.filter {
            it.frames >= MIN_FRAMES && (it.className != "stairs" || it.maxConf >= STAIRS_CONF_FLOOR)
        }

        // Distinct-object count per (class, closest-observed direction). Preserves the
        // [{class, direction, count}] inventory shape, but count is now objects, not frames.
        val grouped = LinkedHashMap<Pair<String, String>, Int>()
        for (track in kept) {
            val key = track.className to track.minDistDir
            grouped[key] = (grouped[key] ?: 0) + 1
        }
        return grouped.entries.map { (key, count) -> InventoryItem(key.first, key.second, count) }
    }

    private fun iou(track: Track, d: ScanDetection): Float {
        val ix1 = maxOf(track.l, d.left); val iy1 = maxOf(track.t, d.top)
        val ix2 = minOf(track.r, d.right); val iy2 = minOf(track.b, d.bottom)
        if (ix1 >= ix2 || iy1 >= iy2) return 0f
        val inter = (ix2 - ix1) * (iy2 - iy1)
        val areaA = (track.r - track.l) * (track.b - track.t)
        val areaB = (d.right - d.left) * (d.bottom - d.top)
        val union = areaA + areaB - inter
        return if (union > 0f) inter / union else 0f
    }
}
