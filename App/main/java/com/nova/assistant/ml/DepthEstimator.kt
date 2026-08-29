package com.nova.assistant.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import com.nova.assistant.util.FileLogger
import com.nova.assistant.util.NovaConstants
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Depth Estimator — TFLite wrapper for MiDaS Small (on-device only).
 * Server-side uses DA3-Large Metric. Output: relative inverse depth, 0-1, HIGH=CLOSE.
 */
@Singleton
class DepthEstimator @Inject constructor(
    private val context: Context,
    private val fileLogger: FileLogger
) {
    companion object {
        private const val TAG = "DepthEstimator"
    }

    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    private var isInitialized = false
    private lateinit var inputBuffer: ByteBuffer
    // F1: Pre-allocated output buffer — reused each frame (rewind before inference).
    // Avoids ~256 KB allocateDirect() per frame that was causing GC pressure on 2 GB devices.
    private var outputBuffer: ByteBuffer? = null
    // F1: Pre-allocated pixel scratch buffer for getPixels() — same pattern as YoloDetector.
    private var depthPixelBuffer: IntArray? = null
    private var depthFrameCount = 0L

    // Raw (pre-normalization) disparity range of the most recent frame, and a slow EMA
    // baseline of it — used by findDepthAnomalies callers as a cheap confidence signal.
    // MiDaS-Small has no native uncertainty output; a frame whose raw range collapses well
    // below its own recent baseline is a proxy for a low-texture scene (blank wall, glossy
    // floor, dim light) where per-frame min-max normalization amplifies noise.
    var lastRawRange: Float = 0f
        private set
    private var rawRangeEma: Float = 0f

    /** True when [lastRawRange] has collapsed well below this session's recent baseline. */
    val isLowConfidenceFrame: Boolean
        get() = rawRangeEma > 0f && lastRawRange < rawRangeEma * NovaConstants.LOW_CONFIDENCE_RANGE_FRACTION

    private var depthInputSize = NovaConstants.MIDAS_INPUT_SIZE

    // Human-readable name of the active model — read by NovaFrameAnalyzer for frame logs.
    var activeDepthModel: String = "none"
        private set

    // Output depth map dimensions
    private var outputHeight = 256
    private var outputWidth = 256

    fun initialize() {
        val (modelBuffer, modelFile) = tryLoadModel(NovaConstants.MIDAS_MODEL_FILE)
            ?: run {
                fileLogger.e(TAG, "MiDaS model asset missing — depth estimation disabled. Expected: ${NovaConstants.MIDAS_MODEL_FILE}")
                return
            }

        try {
            val options = Interpreter.Options().apply { setNumThreads(2) }

            val gpuCompat = CompatibilityList()
            if (gpuCompat.isDelegateSupportedOnThisDevice) {
                try {
                    gpuDelegate = GpuDelegate(gpuCompat.bestOptionsForThisDevice)
                    options.addDelegate(gpuDelegate!!)
                    fileLogger.i(TAG, "GPU delegate attached to depth model")
                } catch (e: Exception) {
                    fileLogger.w(TAG, "Depth GPU delegate failed: ${e.message} — CPU fallback (2 threads)")
                    gpuDelegate?.close(); gpuDelegate = null
                }
            } else {
                fileLogger.w(TAG, "GPU not supported for depth model — CPU only (2 threads)")
            }

            interpreter = try {
                Interpreter(modelBuffer, options)
            } catch (e: Exception) {
                fileLogger.w(TAG, "Depth interpreter with GPU failed: ${e.message} — retrying CPU-only")
                gpuDelegate?.close(); gpuDelegate = null
                Interpreter(modelBuffer, Interpreter.Options().apply { setNumThreads(2) })
            }

            val inputTensor = interpreter!!.getInputTensor(0)
            val inputShape = inputTensor.shape()  // [1, H, W, 3] or [1, 3, H, W]
            depthInputSize = if (inputShape.size >= 3) inputShape[1] else NovaConstants.MIDAS_INPUT_SIZE

            inputBuffer = ByteBuffer.allocateDirect(inputTensor.numBytes())
            inputBuffer.order(ByteOrder.nativeOrder())
            depthPixelBuffer = IntArray(depthInputSize * depthInputSize)

            val outputShape = interpreter!!.getOutputTensor(0).shape()
            if (outputShape.size >= 3) {
                outputHeight = outputShape[1]
                outputWidth  = outputShape[2]
            }
            outputBuffer = ByteBuffer.allocateDirect(interpreter!!.getOutputTensor(0).numBytes())
                .order(ByteOrder.nativeOrder())

            activeDepthModel = "MiDaS-Small(${depthInputSize}px)"
            isInitialized = true
            fileLogger.i(TAG, "DepthModel loaded: $activeDepthModel  file=$modelFile  output=${outputHeight}x${outputWidth}  norm=0-1")

        } catch (e: Exception) {
            fileLogger.e(TAG, "Depth model init failed for $modelFile", e)
            isInitialized = false
        }
    }

    private fun tryLoadModel(filename: String): Pair<MappedByteBuffer, String>? {
        return try {
            Pair(loadModelFile(filename), filename)
        } catch (_: Exception) {
            null
        }
    }

    /**
     * Estimate depth map from camera frame.
     * @return FloatArray normalized 0-1 (HIGH = close), or null if not initialized.
     */
    fun estimateDepth(bitmap: Bitmap): FloatArray? {
        if (!isInitialized) {
            fileLogger.w(TAG, "estimateDepth() called but not initialized")
            return null
        }

        val startTime = System.currentTimeMillis()
        val inputSize = depthInputSize

        // Resize
        val resizeStart = System.currentTimeMillis()
        val resized = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        val resizeMs = System.currentTimeMillis() - resizeStart

        // Fill input buffer with per-model normalization
        inputBuffer.rewind()
        val pixels = depthPixelBuffer ?: IntArray(inputSize * inputSize).also { depthPixelBuffer = it }
        resized.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        for (pixel in pixels) {
            inputBuffer.putFloat(((pixel shr 16) and 0xFF) / 255.0f)
            inputBuffer.putFloat(((pixel shr 8) and 0xFF) / 255.0f)
            inputBuffer.putFloat((pixel and 0xFF) / 255.0f)
        }

        if (resized != bitmap) resized.recycle()

        // Run inference — reuse pre-allocated outputBuffer (F1: avoids per-frame allocateDirect).
        val inferenceStart = System.currentTimeMillis()
        val outBuf = outputBuffer ?: ByteBuffer.allocateDirect(
            interpreter!!.getOutputTensor(0).numBytes()
        ).order(ByteOrder.nativeOrder()).also { outputBuffer = it }
        outBuf.rewind()

        inputBuffer.rewind()
        interpreter!!.run(inputBuffer, outBuf)
        val inferenceMs = System.currentTimeMillis() - inferenceStart

        // Extract depth map
        outBuf.rewind()
        val depthMap = FloatArray(outputHeight * outputWidth)
        outBuf.asFloatBuffer().get(depthMap)

        // Normalize to 0-1 range
        var minDepth = Float.MAX_VALUE
        var maxDepth = Float.MIN_VALUE
        for (d in depthMap) {
            if (d < minDepth) minDepth = d
            if (d > maxDepth) maxDepth = d
        }

        val range = maxDepth - minDepth
        lastRawRange = range
        rawRangeEma = if (rawRangeEma <= 0f) range else 0.9f * rawRangeEma + 0.1f * range
        if (range > 0) {
            for (i in depthMap.indices) {
                depthMap[i] = (depthMap[i] - minDepth) / range
            }
        }

        val totalMs = System.currentTimeMillis() - startTime
        depthFrameCount++
        // Log every frame for first 200, then every 10 — matches YoloDetector cadence
        if (depthFrameCount <= 200L || depthFrameCount % 10L == 0L) {
            fileLogger.d(TAG, "Depth[$depthFrameCount] ${totalMs}ms (resize=${resizeMs}ms infer=${inferenceMs}ms) | depthSource=LOCAL model=$activeDepthModel rawRange=${"%.4f".format(minDepth)}..${"%.4f".format(maxDepth)}")
        }

        return depthMap
    }

    /**
     * Get depth value at a specific normalized position in the depth map.
     * @param depthMap The full depth map (MiDaS convention: HIGH value = CLOSE to camera)
     * @param normX Normalized X position (0-1)
     * @param normY Normalized Y position (0-1)
     * @return Relative disparity: 1.0 = closest, 0.0 = farthest
     */
    fun getDepthAt(depthMap: FloatArray, normX: Float, normY: Float): Float {
        val x = (normX * outputWidth).toInt().coerceIn(0, outputWidth - 1)
        val y = (normY * outputHeight).toInt().coerceIn(0, outputHeight - 1)
        return depthMap[y * outputWidth + x]
    }

    /**
     * Get average depth within a bounding box region.
     */
    fun getDepthInRegion(depthMap: FloatArray, bbox: RectF): Float {
        val x1 = (bbox.left * outputWidth).toInt().coerceIn(0, outputWidth - 1)
        val y1 = (bbox.top * outputHeight).toInt().coerceIn(0, outputHeight - 1)
        val x2 = (bbox.right * outputWidth).toInt().coerceIn(0, outputWidth - 1)
        val y2 = (bbox.bottom * outputHeight).toInt().coerceIn(0, outputHeight - 1)

        if (x2 <= x1 || y2 <= y1) return 0.5f

        var sum = 0f
        var count = 0
        for (y in y1..y2) {
            for (x in x1..x2) {
                sum += depthMap[y * outputWidth + x]
                count++
            }
        }

        return if (count > 0) sum / count else 0.5f
    }

    /**
     * Get average depth using a segmentation mask to exclude background pixels.
     *
     * The mask is a MASK_OUT×MASK_OUT (40×40) float array in normalized original-image space,
     * values 0-1 where >0.5 = foreground (the actual object). This avoids the depth dilution
     * caused by background pixels inside a bounding box (critical for poles, stairs, curbs).
     *
     * Falls back to [getDepthInRegion] if fewer than 10 foreground pixels are found —
     * e.g. when the mask is mostly empty because the object is at the frame edge.
     *
     * @param mask   40×40 float mask in normalized original-image space (from RawDetection.mask)
     * @param maskW  mask grid width  (default 40, matches YoloDetector.MASK_OUT)
     * @param maskH  mask grid height (default 40, matches YoloDetector.MASK_OUT)
     */
    fun getDepthWithMask(
        depthMap: FloatArray,
        bbox: RectF,
        mask: FloatArray,
        maskW: Int = 40,
        maskH: Int = 40
    ): Float {
        // Iterate over mask cells within the bbox bounds
        val xStart = (bbox.left  * maskW).toInt().coerceIn(0, maskW - 1)
        val yStart = (bbox.top   * maskH).toInt().coerceIn(0, maskH - 1)
        val xEnd   = (bbox.right * maskW).toInt().coerceIn(0, maskW - 1)
        val yEnd   = (bbox.bottom * maskH).toInt().coerceIn(0, maskH - 1)

        var sum   = 0f
        var count = 0

        for (my in yStart..yEnd) {
            for (mx in xStart..xEnd) {
                if (mask[my * maskW + mx] <= 0.5f) continue

                // Map mask cell center to depth map pixel
                val normX  = (mx + 0.5f) / maskW
                val normY  = (my + 0.5f) / maskH
                val depthX = (normX * outputWidth ).toInt().coerceIn(0, outputWidth  - 1)
                val depthY = (normY * outputHeight).toInt().coerceIn(0, outputHeight - 1)

                sum += depthMap[depthY * outputWidth + depthX]
                count++
            }
        }

        // Need at least 10 foreground pixels for a reliable average; fall back otherwise.
        return if (count >= 10) sum / count else getDepthInRegion(depthMap, bbox)
    }

    /**
     * Detect depth anomalies — regions with significant depth change
     * but no YOLO detection (potential unknown obstacles).
     *
     * Exclusion zones (structural surfaces, not real obstacles):
     *   - Top 20% of frame  → ceiling (always "present", not an obstacle)
     *   - Bottom 30% of frame → floor (always close to camera, always false-positive)
     *
     * Coverage guard:
     *   If > 35% of the active middle band tests as "close", the scene is a
     *   corridor / close wall — treat the entire frame as a room boundary and
     *   return nothing (the YOLO pipeline handles detected walls/fences separately).
     *
     * Corridor guard:
     *   Cells are further restricted to NovaConstants.UNKNOWN_OBSTACLE_CORRIDOR_HALF_WIDTH_M
     *   of physical walking-corridor width, converted per-cell from its own depth via
     *   [scaleFactor] (see isWithinCorridor). A cell 2m off to the side at 4m distance is
     *   not a collision risk even though it may sit in a "CENTER" angular bucket; this
     *   scans in constant-metric-width instead of constant-angular-width. Applies only to
     *   this class-agnostic scanner — named YOLO detections keep full lateral awareness.
     *
     * @param scaleFactor DepthCalibrator's current disparity->meters scale (same formula as
     *   its unclassified fallback: meters = (1 - relativeDepth) * scaleFactor). Passed in
     *   rather than duplicated so both call sites stay derived from one calibration.
     */
    fun findDepthAnomalies(
        depthMap: FloatArray,
        detectedRegions: List<RectF>,
        scaleFactor: Float,
        gridSize: Int = 8
    ): List<RectF> {
        val cellW = 1f / gridSize
        val cellH = 1f / gridSize

        // Rows to skip: top 20% = ceiling, bottom 30% = floor.
        // For gridSize=8: skip rows 0-1 (ceiling) and rows 6-7 (floor).
        val skipTopRows    = (gridSize * 0.20f).toInt().coerceAtLeast(1)  // ≥1 row
        val skipBottomRows = (gridSize * 0.30f).toInt().coerceAtLeast(1)  // ≥1 row
        val rowStart = skipTopRows
        val rowEnd   = gridSize - skipBottomRows  // exclusive

        // ── Pass 1: count active cells and anomalous cells ──────────────────
        // Used for the coverage guard: if most of the middle band is "close",
        // it is a wall or corridor, not a discrete unknown obstacle.
        var totalActive    = 0
        var totalAnomalous = 0

        for (row in rowStart until rowEnd) {
            for (col in 0 until gridSize) {
                val cellRect = RectF(
                    col * cellW, row * cellH,
                    (col + 1) * cellW, (row + 1) * cellH
                )
                val overlapsDetection = detectedRegions.any { det ->
                    RectF.intersects(cellRect, det)
                }
                if (overlapsDetection) continue

                val cellDepth = getDepthInRegion(depthMap, cellRect)
                if (!isWithinCorridor(cellRect.centerX(), cellDepth, scaleFactor)) continue

                totalActive++
                // Dual-gate threshold: relative depth > 0.80 AND metric distance < WARNING_DISTANCE.
                // The relative-depth-only gate (old: cellDepth > 0.80f) fires in distant scenes
                // because MiDaS normalizes per-frame — the closest cell always maps to ~1.0
                // even if nothing is actually within arm's reach. The metric gate ensures we
                // only fire when the inferred metric distance is a genuine navigation hazard.
                val cellDistMeters = ((1f - cellDepth) * scaleFactor).coerceIn(0.1f, 20f)
                if (cellDepth > 0.80f && cellDistMeters < NovaConstants.WARNING_DISTANCE) totalAnomalous++
            }
        }

        // Coverage guard: > 35% of the active band is "close" → wall/corridor.
        // Structural surfaces are not unknown obstacles — return nothing.
        if (totalActive > 0 && totalAnomalous.toFloat() / totalActive > 0.35f) {
            return emptyList()
        }

        // ── Pass 2: collect individual anomaly cells ─────────────────────────
        val anomalies = mutableListOf<RectF>()
        for (row in rowStart until rowEnd) {
            for (col in 0 until gridSize) {
                val cellRect = RectF(
                    col * cellW, row * cellH,
                    (col + 1) * cellW, (row + 1) * cellH
                )
                val overlapsDetection = detectedRegions.any { det ->
                    RectF.intersects(cellRect, det)
                }
                if (overlapsDetection) continue

                val cellDepth = getDepthInRegion(depthMap, cellRect)
                if (!isWithinCorridor(cellRect.centerX(), cellDepth, scaleFactor)) continue

                val cellDistMeters = ((1f - cellDepth) * scaleFactor).coerceIn(0.1f, 20f)
                if (cellDepth > 0.80f && cellDistMeters < NovaConstants.WARNING_DISTANCE) {
                    anomalies.add(cellRect)
                }
            }
        }

        // Merge adjacent anomaly cells
        return mergeAdjacentRects(anomalies)
    }

    /**
     * Is this cell within the physical walking corridor, given its own depth?
     * Similar-triangles projection (same style as DepthCalibrator's known-height
     * triangulation): lateral offset in meters = (x - 0.5) * distanceMeters * FOV_FACTOR.
     */
    private fun isWithinCorridor(cellCenterX: Float, cellDepth: Float, scaleFactor: Float): Boolean {
        val distMeters = ((1f - cellDepth) * scaleFactor).coerceIn(0.1f, 20f)
        val lateralOffsetM = (cellCenterX - 0.5f) * distMeters * NovaConstants.UNKNOWN_OBSTACLE_CORRIDOR_FOV_FACTOR
        return kotlin.math.abs(lateralOffsetM) <= NovaConstants.UNKNOWN_OBSTACLE_CORRIDOR_HALF_WIDTH_M
    }

    private fun mergeAdjacentRects(rects: List<RectF>): List<RectF> {
        if (rects.isEmpty()) return emptyList()
        // Simple merge: combine overlapping/adjacent rects
        val merged = mutableListOf<RectF>()
        val used = BooleanArray(rects.size)

        for (i in rects.indices) {
            if (used[i]) continue
            val current = RectF(rects[i])
            used[i] = true

            var changed = true
            while (changed) {
                changed = false
                for (j in rects.indices) {
                    if (used[j]) continue
                    // Check if adjacent (touching or overlapping)
                    val expanded = RectF(current)
                    expanded.inset(-0.02f, -0.02f) // small margin
                    if (RectF.intersects(expanded, rects[j])) {
                        current.union(rects[j])
                        used[j] = true
                        changed = true
                    }
                }
            }
            merged.add(current)
        }

        return merged
    }

    fun warmup() {
        if (!isInitialized) return
        val dummy = Bitmap.createBitmap(depthInputSize, depthInputSize, Bitmap.Config.ARGB_8888)
        estimateDepth(dummy)
        dummy.recycle()
        fileLogger.i(TAG, "Depth model warm-up complete")
    }

    private fun loadModelFile(filename: String): MappedByteBuffer {
        val fd = context.assets.openFd(filename)
        val inputStream = FileInputStream(fd.fileDescriptor)
        val channel = inputStream.channel
        return channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
    }

    fun close() {
        interpreter?.close()
        gpuDelegate?.close()
        interpreter = null
        gpuDelegate = null
        isInitialized = false
    }
}
