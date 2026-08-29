package com.nova.assistant.server

import android.graphics.RectF
import com.nova.assistant.engine.*
import com.nova.assistant.util.*
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * ServerFrameResult — top-level JSON response from NOVA inference server.
 *
 * Matches the server JSON protocol exactly:
 *   {"frame":42, "detections":[...], "depth_zones":{...}, "alerts":[...],
 *    "path":"MOVE_LEFT", "inference_ms":22}
 */
@Serializable
data class ServerFrameResult(
    val frame: Int,
    val detections: List<ServerDetection>,
    @SerialName("depth_zones") val depthZones: Map<String, String>,
    val alerts: List<ServerAlert>,
    val path: String,
    @SerialName("inference_ms") val inferenceMs: Int
)

/**
 * A single detected object from the server.
 * bbox: [x1, y1, x2, y2] normalized 0-1 (same space as Android RawDetection.bbox)
 * zone: "CRITICAL" | "CLOSE" | "MEDIUM" | "FAR"
 * direction: "left" | "center" | "right"
 */
@Serializable
data class ServerDetection(
    @SerialName("class")      val className: String,
    val confidence: Float,
    val bbox: List<Float>,
    val direction: String,
    @SerialName("distance_m") val distanceM: Float,
    val zone: String
)

/**
 * Alert from server — ready to speak and vibrate.
 * priority: 1=EMERGENCY, 2=DANGER, 3=DIRECTION, 4=INFO
 * haptic: "PING" | "DOUBLE_TAP" | "BUZZ" | "TRIPLE_PULSE" | "HEARTBEAT"
 */
@Serializable
data class ServerAlert(
    val priority: Int,
    val message: String,
    @SerialName("class")      val className: String,
    val direction: String,
    @SerialName("distance_m") val distanceM: Float,
    val haptic: String
)

// ── Mapping: ServerFrameResult → existing FrameResult ────────────────────────

fun ServerFrameResult.toFrameResult(): FrameResult {
    val mappedDetections = detections.map { det ->
        DetectionWithDepth(
            classId        = -1,
            className      = det.className,
            confidence     = det.confidence,
            bbox           = RectF(det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3]),
            distanceMeters = det.distanceM,
            distanceZone   = when (det.zone) {
                "CRITICAL" -> DistanceZone.DANGER
                "CLOSE"    -> DistanceZone.WARNING
                "MEDIUM"   -> DistanceZone.CAUTION
                else       -> DistanceZone.FAR
            },
            direction = when (det.direction) {
                "far_left"  -> SpatialDirection.FAR_LEFT
                "left"      -> SpatialDirection.LEFT
                "right"     -> SpatialDirection.RIGHT
                "far_right" -> SpatialDirection.FAR_RIGHT
                else        -> SpatialDirection.CENTER
            }
        )
    }

    val zoneStatusMap: Map<SpatialDirection, ZoneStatus> = buildMap {
        depthZones.forEach { (key, value) ->
            val dir = when (key) {
                "left"  -> SpatialDirection.LEFT
                "right" -> SpatialDirection.RIGHT
                else    -> SpatialDirection.CENTER
            }
            put(dir, when (value) {
                "CRITICAL", "CLOSE" -> ZoneStatus.BLOCKED
                "MEDIUM"            -> ZoneStatus.CAUTION
                else                -> ZoneStatus.CLEAR
            })
        }
    }

    val guidance = when (path) {
        "MOVE_LEFT"        -> NavigationGuidance.MOVE_LEFT
        "MOVE_RIGHT"       -> NavigationGuidance.MOVE_RIGHT
        "STOP_ALL_BLOCKED" -> NavigationGuidance.STOP_ALL_BLOCKED
        "SLOW_DOWN"        -> NavigationGuidance.SLOW_DOWN
        else               -> NavigationGuidance.CONTINUE_FORWARD
    }

    val freePathResult = FreePathResult(
        zones              = zoneStatusMap,
        suggestedDirection = guidance,
        isPathClear        = (path == "CONTINUE_FORWARD")
    )

    val novaAlerts = alerts.map { sa ->
        NovaAlert(
            priority       = when (sa.priority) {
                1    -> AlertPriority.P1_EMERGENCY
                2    -> AlertPriority.P2_DANGER
                3    -> AlertPriority.P3_DIRECTION
                else -> AlertPriority.P4_INFO
            },
            spokenText     = sa.message,
            direction      = when (sa.direction) {
                "far_left"  -> SpatialDirection.FAR_LEFT
                "left"      -> SpatialDirection.LEFT
                "right"     -> SpatialDirection.RIGHT
                "far_right" -> SpatialDirection.FAR_RIGHT
                else        -> SpatialDirection.CENTER
            },
            distanceMeters  = sa.distanceM,
            hapticPattern   = when (sa.haptic) {
                "BUZZ"         -> HapticPattern.BUZZ
                "DOUBLE_TAP"   -> HapticPattern.DOUBLE_TAP
                "TRIPLE_PULSE" -> HapticPattern.TRIPLE_PULSE
                "HEARTBEAT"    -> HapticPattern.HEARTBEAT
                else           -> HapticPattern.PING
            },
            sourceClassName = sa.className
        )
    }

    return FrameResult(
        detections       = mappedDetections,
        freePath         = freePathResult,
        alerts           = novaAlerts,
        topAlert         = novaAlerts.minByOrNull { it.priority.level },
        frameTimestampMs = System.currentTimeMillis(),
        inferenceTimeMs  = inferenceMs.toLong()
    )
}
