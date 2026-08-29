package com.nova.assistant.engine

import android.graphics.Bitmap
import android.graphics.RectF
import com.nova.assistant.util.AlertPriority
import com.nova.assistant.util.DistanceZone
import com.nova.assistant.util.SpatialDirection

// ─────────────────────────────────────────────
// Stage 3 Output: Raw YOLO detections
// ─────────────────────────────────────────────
data class RawDetection(
    val classId: Int,
    val className: String,
    val confidence: Float,
    val bbox: RectF,           // Normalized 0-1
    val mask: FloatArray? = null
)

// ─────────────────────────────────────────────
// Stage 6 Output: Detection with distance info
// ─────────────────────────────────────────────
data class DetectionWithDepth(
    val classId: Int,
    val className: String,
    val confidence: Float,
    val bbox: RectF,
    val distanceMeters: Float,
    val distanceZone: DistanceZone,
    val direction: SpatialDirection,
    val isMovingToward: Boolean = false,
    val approachSpeed: Float = 0f,  // relative speed, 0 = static
    val isUnknownObstacle: Boolean = false
)

// ─────────────────────────────────────────────
// Stage 9 Output: Free path analysis
// ─────────────────────────────────────────────
data class FreePathResult(
    val zones: Map<SpatialDirection, ZoneStatus>,
    val suggestedDirection: NavigationGuidance,
    val isPathClear: Boolean
)

enum class ZoneStatus { CLEAR, CAUTION, BLOCKED }

enum class NavigationGuidance {
    CONTINUE_FORWARD,
    MOVE_LEFT,
    MOVE_RIGHT,
    STOP_ALL_BLOCKED,
    SLOW_DOWN;

    // H3/M5: clock-position directions + clearance confirmation so the user
    // knows which side is clear before committing to movement.
    fun toSpokenGuidance(): String = when (this) {
        CONTINUE_FORWARD -> "Path clear, continue forward."
        MOVE_LEFT        -> "Path clear to your left at 9 o'clock. Move left."
        MOVE_RIGHT       -> "Path clear to your right at 3 o'clock. Move right."
        STOP_ALL_BLOCKED -> "All paths blocked. Stand still and wait."
        SLOW_DOWN        -> "Obstacle ahead. Slow down."
    }
}

// ─────────────────────────────────────────────
// Stage 10 Output: Final alert to speak
// ─────────────────────────────────────────────
data class NovaAlert(
    val priority: AlertPriority,
    val spokenText: String,
    val direction: SpatialDirection,
    val distanceMeters: Float,
    val hapticPattern: HapticPattern,
    val timestamp: Long = System.currentTimeMillis(),
    val sourceClassName: String = "",
    val isPathGuidance: Boolean = false
)

enum class HapticPattern {
    PING,           // Single 80ms — object FAR
    DOUBLE_TAP,     // 80-pause-80ms — WARNING zone
    BUZZ,           // Continuous 400ms — DANGER zone
    TRIPLE_PULSE,   // 60ms × 3 — moving object approaching
    HEARTBEAT,      // 200ms on/off repeating — unknown obstacle
    SOS_PATTERN     // Morse SOS — emergency
}

// ─────────────────────────────────────────────
// Complete pipeline frame result
// ─────────────────────────────────────────────
data class FrameResult(
    val detections: List<DetectionWithDepth>,
    val freePath: FreePathResult,
    val alerts: List<NovaAlert>,
    val topAlert: NovaAlert?,
    val frameTimestampMs: Long,
    val inferenceTimeMs: Long
)

// ─────────────────────────────────────────────
// Sensor data bundle
// ─────────────────────────────────────────────
data class SensorData(
    val isMoving: Boolean = false,
    val lightLux: Float = 200f,
    val tiltAngleDegrees: Float = 0f,
    val batteryPercent: Int = 100,
    val temperatureCelsius: Float = 25f,
    val motionMagnitude: Float = 0f,
    val latitude: Double = 0.0,
    val longitude: Double = 0.0,
    val hasGpsFix: Boolean = false,
    val azimuthDeg: Float = -1f  // compass 0=N,90=E,180=S,270=W; -1=unknown
)
