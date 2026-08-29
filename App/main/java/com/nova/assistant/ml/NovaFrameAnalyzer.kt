package com.nova.assistant.ml

import android.graphics.Bitmap
import android.graphics.Matrix
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.nova.assistant.engine.FrameResult
import com.nova.assistant.engine.InferencePipeline
import com.nova.assistant.engine.SensorData
import com.nova.assistant.sensors.NovaSensorManager
import com.nova.assistant.server.NovaInferenceClient
import com.nova.assistant.server.toFrameResult
import com.nova.assistant.util.FileLogger
import com.nova.assistant.util.NovaConstants
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference
import javax.inject.Inject
import javax.inject.Singleton

/**
 * NOVA Frame Analyzer — bridges CameraX to InferencePipeline.
 *
 * Key behaviors:
 * - Always processes the LATEST frame, drops stale frames silently
 * - Adaptive frame rate (3-10 fps based on motion + battery)
 * - Runs inference on a dedicated coroutine
 */
@Singleton
class NovaFrameAnalyzer @Inject constructor(
    private val pipeline: InferencePipeline,
    private val sensorManager: NovaSensorManager,
    private val preprocessor: Preprocessor,
    private val fileLogger: FileLogger,
    private val inferenceClient: NovaInferenceClient,
    private val depthEstimator: DepthEstimator,
) : ImageAnalysis.Analyzer {

    // Set false to force on-device inference (useful for debugging / offline testing)
    var useServerMode: Boolean = true

    // True while the Finder tab is active. Finder only needs a fresh preview bitmap — it does
    // NOT need YOLO/depth navigation results — but analyze() previously ran the full pipeline
    // regardless of which tab was showing. That coupled Finder's preview freshness to the full
    // server round trip (300-500ms+, sometimes up to the 2s timeout): isProcessing stayed true
    // for that whole round trip, silently dropping every camera frame that arrived meanwhile,
    // and _lastBitmap only updated on every 5th frame that DID get through. Compounded, pointing
    // the camera at a new object could take several seconds to be reflected in Finder. When this
    // flag is set, analyze() skips the nav pipeline entirely and refreshes _lastBitmap on every
    // accepted frame instead.
    @Volatile
    var suspendNavPipeline: Boolean = false

    companion object {
        private const val TAG = "NovaFrameAnalyzer"
    }

    private val isProcessing = AtomicBoolean(false)
    private val frameCounter = AtomicLong(0L)

    // @Volatile var — not val — because this is a @Singleton that outlives any single ViewModel.
    // When MainNavigationViewModel.onCleared() calls shutdown(), the old scope is cancelled
    // and immediately replaced with a fresh one. The next ViewModel lifecycle picks it up
    // automatically without any re-initialization needed.
    @Volatile
    private var analysisScope = CoroutineScope(Dispatchers.Default + SupervisorJob())

    private val _frameResults = MutableStateFlow<FrameResult?>(null)
    val frameResults: StateFlow<FrameResult?> = _frameResults

    private val _isUsingServer = MutableStateFlow(false)
    val isUsingServer: StateFlow<Boolean> = _isUsingServer

    // Latest bitmap exposed for on-demand features (OCR, room snapshot, Finder preview).
    //
    // BUG FIX 2026-07-16: AtomicReference only makes the POINTER swap atomic — it does NOT
    // protect the underlying Bitmap object from being recycled by analyze()'s writer coroutine
    // while a reader (Finder's 500ms preview poll, OCR, PEOPLE_NEARBY) is still using the SAME
    // instance it read from _lastBitmap.get(). Finder in particular crashed the app on open:
    // FinderViewModel.init polled `frameAnalyzer.lastBitmap` directly into Compose state every
    // 500ms with no copy, and opening Finder sets suspendNavPipeline=true, which makes analyze()
    // recycle-and-replace _lastBitmap on EVERY accepted frame instead of every 5th — a much
    // tighter recycle cadence landing right when the read races against it. Compose then calls
    // .asImageBitmap() on an instance that can already be recycled -> IllegalStateException /
    // "Canvas: trying to use a recycled bitmap" crash.
    // Fix: all reads now go through snapshotLastBitmap(), which copies under the same lock the
    // writer recycles under, so a copy and a recycle of the same instance can never interleave.
    private val bitmapLock = Any()
    private val _lastBitmap = AtomicReference<Bitmap?>(null)

    /** Race-free copy of the latest frame, or null if none yet / the source was already recycled. */
    fun snapshotLastBitmap(): Bitmap? = synchronized(bitmapLock) {
        val bmp = _lastBitmap.get() ?: return@synchronized null
        if (bmp.isRecycled) null else bmp.copy(bmp.config ?: Bitmap.Config.ARGB_8888, false)
    }

    private fun replaceLastBitmap(newBitmap: Bitmap) {
        synchronized(bitmapLock) {
            _lastBitmap.getAndSet(newBitmap)?.recycle()
        }
    }

    private var lastInferenceTime = 0L
    private var isPaused = false

    override fun analyze(image: ImageProxy) {
        if (isPaused) {
            image.close()
            return
        }
        // Drop frame if already processing (always use latest) — only applies to the nav
        // pipeline; Finder mode (suspendNavPipeline) never sets isProcessing, so this check
        // is skipped there and every accepted frame gets through.
        if (!suspendNavPipeline && isProcessing.get()) {
            image.close()
            return
        }

        // Adaptive frame rate
        val sensorData = sensorManager.getCurrentSensorData()
        val targetIntervalMs = computeFrameInterval(sensorData)
        val now = System.currentTimeMillis()
        if (now - lastInferenceTime < targetIntervalMs) {
            image.close()
            return
        }
        lastInferenceTime = now

        // Convert image to bitmap
        val bitmap = imageToBitmap(image)
        image.close()

        if (bitmap == null) return

        if (suspendNavPipeline) {
            // Finder tab: skip YOLO/depth/server entirely — just keep the preview fresh.
            val copy = bitmap.copy(bitmap.config ?: Bitmap.Config.ARGB_8888, false)
            replaceLastBitmap(copy)
            bitmap.recycle()
            return
        }

        // Store latest frame for on-demand access (OCR, room snapshot).
        // Updated every 5 frames (~500ms at 10fps) — fresh enough for user-triggered features
        // while saving 4 full-resolution bitmap copies per second.
        if (frameCounter.get() % 5L == 0L) {
            val copy = bitmap.copy(bitmap.config ?: Bitmap.Config.ARGB_8888, false)
            replaceLastBitmap(copy)
        }

        // Process on coroutine
        if (isProcessing.compareAndSet(false, true)) {
            analysisScope.launch {
                try {
                    val frameNum = frameCounter.incrementAndGet()

                    // ── Server mode (primary path) ──────────────────────────
                    // Send an ASPECT-PRESERVED, long-side-capped frame (not the 640² on-device
                    // squish): the server runs YOLO at imgsz=1280, so it needs real detail and
                    // correct geometry for its height-based distance estimate. Preprocessing for
                    // the on-device path is deferred until we actually fall back — saving a scale
                    // op on every server-served frame.
                    if (useServerMode && inferenceClient.isConnected()) {
                        val serverFrame = scaleForServer(bitmap)
                        val serverResult = inferenceClient.sendFrame(serverFrame)
                        if (serverFrame != bitmap) serverFrame.recycle()
                        if (serverResult != null) {
                            if (!_isUsingServer.value) {
                                _isUsingServer.value = true
                                fileLogger.i(TAG, "InferenceSource → SERVER (frame[$frameNum])")
                            }
                            if (frameNum % 50L == 0L) {
                                fileLogger.i(TAG, "Frame[$frameNum] depthSource=SERVER(DAV2-Large)")
                            }
                            _frameResults.value = serverResult.toFrameResult()
                            bitmap.recycle()
                            return@launch
                        }
                        // null = timeout or parse error → fall through to on-device
                        fileLogger.w(TAG, "Server frame[$frameNum] failed — using on-device fallback")
                    }

                    // ── On-device fallback (existing pipeline) ───────────────
                    // Build the 640² preprocessed frame only now (server path didn't need it).
                    if (_isUsingServer.value) {
                        _isUsingServer.value = false
                        fileLogger.i(TAG, "InferenceSource → LOCAL (frame[$frameNum])")
                    }
                    val preprocessed = preprocessor.process(bitmap, sensorData)
                    val result = pipeline.processFrame(preprocessed, sensorData)
                    _frameResults.value = result

                    // Periodic thermal/battery state log — helps correlate latency spikes in log analysis
                    if (frameNum % 50L == 0L) {
                        fileLogger.i(TAG, "Frame[$frameNum] batt=${sensorData.batteryPercent}% temp=${"%.1f".format(sensorData.temperatureCelsius)}°C moving=${sensorData.isMoving} light=${"%.0f".format(sensorData.lightLux)}lux tilt=${"%.1f".format(sensorData.tiltAngleDegrees)}° depthSource=LOCAL(${depthEstimator.activeDepthModel})")
                    }

                    // Recycle bitmap
                    if (preprocessed != bitmap) preprocessed.recycle()
                    bitmap.recycle()

                } catch (e: Exception) {
                    Log.e(TAG, "Frame processing error", e)
                } finally {
                    isProcessing.set(false)
                }
            }
        }
    }

    /**
     * Downscale to NovaConstants.SERVER_SEND_LONG_SIDE on the long side, preserving aspect.
     * Returns the source unchanged if it is already within the cap (caller must not recycle
     * the returned bitmap when it === src). Aspect preservation keeps the server's letterboxing
     * and the height-based distance estimate geometrically correct.
     */
    private fun scaleForServer(src: Bitmap): Bitmap {
        val longSide = maxOf(src.width, src.height)
        val cap = NovaConstants.SERVER_SEND_LONG_SIDE
        if (longSide <= cap) return src
        val scale = cap.toFloat() / longSide
        val w = (src.width * scale).toInt().coerceAtLeast(1)
        val h = (src.height * scale).toInt().coerceAtLeast(1)
        return Bitmap.createScaledBitmap(src, w, h, true)
    }

    /**
     * Compute frame interval based on sensor data.
     */
    private fun computeFrameInterval(sensorData: SensorData): Long {
        val targetFps = when {
            sensorData.batteryPercent < NovaConstants.BATTERY_CRITICAL ->
                NovaConstants.FPS_CRITICAL
            sensorData.batteryPercent < NovaConstants.BATTERY_LOW ->
                NovaConstants.FPS_LOW_BATTERY
            sensorData.temperatureCelsius > NovaConstants.THERMAL_CRITICAL ->
                NovaConstants.FPS_CRITICAL
            sensorData.temperatureCelsius > NovaConstants.THERMAL_HOT ->
                NovaConstants.FPS_LOW_BATTERY
            sensorData.isMoving -> NovaConstants.FPS_MOVING
            else -> NovaConstants.FPS_STATIONARY
        }
        return 1000L / targetFps
    }

    fun pause() { isPaused = true }

    fun resume() {
        isPaused = false
    }

    /**
     * Reset suppression timestamps so objects near the user are announced immediately on resume.
     * Delegates to InferencePipeline → SuppressionFilter.
     */
    fun resetSuppression() {
        pipeline.resetSuppression()
    }

    fun shutdown() {
        // Cancel the current scope, then immediately replace it so the next
        // ViewModel lifecycle (after back-navigation) gets a live scope.
        analysisScope.cancel()
        analysisScope = CoroutineScope(Dispatchers.Default + SupervisorJob())
        isProcessing.set(false) // Reset flag in case it was stuck mid-frame
    }

    /**
     * Convert RGBA_8888 ImageProxy → ARGB Bitmap.
     *
     * CameraX delivers RGBA_8888 when OUTPUT_IMAGE_FORMAT_RGBA_8888 is set on
     * ImageAnalysis. The single plane buffer contains packed RGBA pixels with
     * possible row-stride padding — we strip padding row-by-row when present.
     *
     * This replaces the old YUV→JPEG→Bitmap path which cost ~40-100ms per frame
     * due to compressToJpeg()+decodeByteArray() on CPU.
     */
    private fun imageToBitmap(image: ImageProxy): Bitmap? {
        return try {
            val plane     = image.planes[0]
            val buf       = plane.buffer
            val rowStride = plane.rowStride
            val width     = image.width
            val height    = image.height

            val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)

            if (rowStride == width * 4) {
                // No padding — copy buffer directly
                bitmap.copyPixelsFromBuffer(buf)
            } else {
                // Row-stride padding present (16-byte alignment on some HALs)
                val rowBytes = ByteArray(width * 4)
                val pixels = IntArray(width)  // hoisted — one allocation reused for all rows
                for (row in 0 until height) {
                    buf.position(row * rowStride)
                    buf.get(rowBytes)
                    for (x in 0 until width) {
                        val r = rowBytes[x * 4].toInt() and 0xFF
                        val g = rowBytes[x * 4 + 1].toInt() and 0xFF
                        val b = rowBytes[x * 4 + 2].toInt() and 0xFF
                        pixels[x] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                    }
                    bitmap.setPixels(pixels, 0, width, 0, row, width, 1)
                }
            }

            // Rotate to match phone orientation if needed
            if (image.imageInfo.rotationDegrees != 0) {
                val matrix = Matrix()
                matrix.postRotate(image.imageInfo.rotationDegrees.toFloat())
                val rotated = Bitmap.createBitmap(bitmap, 0, 0, width, height, matrix, true)
                bitmap.recycle()
                rotated
            } else {
                bitmap
            }
        } catch (e: Exception) {
            Log.e(TAG, "Bitmap conversion failed", e)
            null
        }
    }
}

/**
 * Preprocessing: scale to YOLO input size and apply tilt compensation.
 * Low-light handling removed — the phone ISP auto-exposure covers it in hardware.
 */
@Singleton
class Preprocessor @Inject constructor() {

    fun process(bitmap: Bitmap, sensorData: SensorData): Bitmap {
        var result = bitmap

        // Pre-scale to YOLO input size before any per-pixel operations.
        // CLAHE and tilt compensation operate on 640×640 instead of the full camera resolution
        // (e.g. 1920×1080 = 2M pixels → 0.4M pixels = ~5× fewer pixels to process).
        val yoloSize = NovaConstants.YOLO_INPUT_SIZE
        val needsPreScale = result.width > yoloSize || result.height > yoloSize
        if (needsPreScale) {
            val scaled = Bitmap.createScaledBitmap(result, yoloSize, yoloSize, true)
            if (result != bitmap) result.recycle()
            result = scaled
        }

        // Tilt compensation: shift crop window based on phone angle
        if (kotlin.math.abs(sensorData.tiltAngleDegrees) > 15f) {
            result = applyTiltCompensation(result, sensorData.tiltAngleDegrees)
        }

        return result
    }

    /**
     * Tilt compensation: shift crop window to counteract phone angle.
     */
    private fun applyTiltCompensation(src: Bitmap, tiltDegrees: Float): Bitmap {
        val clampedTilt = tiltDegrees.coerceIn(-30f, 30f)
        val shiftFraction = (clampedTilt / 30f) * 0.15f
        val shiftPixels = (src.height * shiftFraction).toInt()
        val cropTop = shiftPixels.coerceAtLeast(0)
        val cropHeight = (src.height - cropTop).coerceAtLeast(1)
        return Bitmap.createBitmap(src, 0, cropTop, src.width, minOf(cropHeight, src.height - cropTop))
    }
}
