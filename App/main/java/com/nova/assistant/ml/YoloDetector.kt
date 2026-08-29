package com.nova.assistant.ml

import android.content.Context
import android.graphics.Bitmap
import android.os.Build
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.Log
import com.nova.assistant.engine.RawDetection
import com.nova.assistant.util.ClassConfig
import com.nova.assistant.util.FileLogger
import com.nova.assistant.util.NovaConstants
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import com.qualcomm.qti.QnnDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.min
import kotlin.math.roundToInt
import javax.inject.Inject
import javax.inject.Singleton

/**
 * YOLO TFLite Inference Wrapper — E2E NMS models only.
 *
 * // HISTORY: NMS-free model (nms=False TFLite, output [1,8400,65]) was tested June 2026.
 * // Reason tried: MediaTek Neuron NNAPI driver rejects TFLite graphs containing the E2E NMS op,
 * //   forcing GPU fallback at 0.33 FPS instead of expected ~3-5 FPS via NNAPI.
 * // Abandoned: CPU NMS post-inference = 400–2400ms/frame bottleneck on device.
 * //   Heavy inference moved to Colab/Kaggle server GPU. Android uses E2E TFLite offline fallback only.
 *
 * Active model: nova_yolo_run6_alt_fp16.tflite (YOLO26s-seg, 18 classes, mAP50=0.715)
 * Input:  NHWC ByteBuffer [1, 640, 640, 3]  float32, normalized 0-1
 * Output (E2E NMS):
 *   output[0]: [1, 300, 38] — post-NMS detections (Run4, 28 classes)
 *              38 = 4 (x1,y1,x2,y2 xyxy) + 1 (confidence) + 1 (class_id float) + 32 (mask coefficients)
 *              Empty padding slots have confidence ~0 and are skipped.
 *   output[1]: prototype masks, [1, 32, 160, 160] NCHW (standard Ultralytics export) or
 *              [1, 160, 160, 32] NHWC (seen on some exported builds) — layout is detected
 *              at init (protoIsNhwc) and computeMask() indexes accordingly either way.
 *
 * NO manual NMS — model applies NMS internally.
 * Post-NMS confidence scores: 0.30–0.85 range on real detections.
 *
 * Confs:  Check "E2EParse" in nova_debug.log after first device run:
 *         - thresh=0 every frame → ClassConfig thresholds too high
 *         - e2e=300 every frame  → threshold too low (all slots filled)
 *         - Target: 3-10 detections/frame in a cluttered scene
 */
@Singleton
class YoloDetector @Inject constructor(
    private val context: Context,
    private val fileLogger: FileLogger
) {
    companion object {
        private const val TAG = "YoloDetector"

        // Classes where segmentation masks give meaningful depth improvement over bbox averaging.
        // Poles: bbox is wide but pole is 3-5cm — mask isolates the actual pole pixels.
        // Stairs/pothole/curb: bbox includes floor above/below the edge — mask follows the surface.
        // Person: silhouette is narrower than bbox, especially when partially occluded.
        private val MASK_CLASSES = setOf("pole", "stairs", "pothole", "curb", "person")

        private const val PROTO_SIZE = 160    // prototype mask resolution (model output[1])
        // Run4 E2E [1,300,38]: 4+1+1+32 = 38 → 32 mask coefficients (same as Run2).
        private const val MASK_COEFF_COUNT = 32
        private const val MASK_OUT = 40      // output mask grid — normalized original-image space
    }

    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    private var nnApiDelegate: NnApiDelegate? = null
    private var qnnDelegate: QnnDelegate? = null
    private var isInitialized = false
    private var classNames: List<String> = emptyList()
    private var numClasses: Int = 0

    // Pre-allocated buffers (reused each frame -- zero per-frame heap allocation)
    private var inputBuffer: ByteBuffer? = null
    private var outputBuffers: Array<ByteBuffer> = emptyArray()
    private val outputsMap = HashMap<Int, Any>()
    // Pixel scratch buffer for bitmapToNHWC -- allocated once at first use, reused thereafter.
    private var pixelBuffer: IntArray? = null
    // F3: Pre-allocated canvas bitmap for letterboxBitmap() — avoids 1.6 MB ARGB8888 alloc/recycle
    // per frame. Bitmap is cleared with drawColor() before each use. Do NOT recycle after detect().
    private var canvasBitmap: Bitmap? = null
    private var canvasObj: Canvas? = null

    // Output detection tensor shape (determined at init from model metadata)
    private var outputDim1: Int = 0
    private var outputDim2: Int = 0

    // Coordinate space: logged once on first frame with non-zero detections
    @Volatile private var coordSpaceLogged = false

    // Frame counter for throttled file logging
    private var frameCount = 0L

    @Volatile private var resolvedModelFile = NovaConstants.YOLO_MODEL_FILE

    // Per-class rolling detection counts — logged every 100 frames to show which classes fire.
    private val classDetectionCounts = HashMap<String, Int>()
    private var classCountFrameWindow = 0

    // E2E Ultralytics export is always NCHW [1,31,160,160]. Detected at init; warn if NHWC.
    private var protoIsNhwc: Boolean = false

    // First inference timestamp — used once to log actual delegate latency for NNAPI diagnosis.
    private var firstInferenceLogged = false

    fun initialize(classNamesList: List<String>) {
        try {
            classNames = classNamesList
            numClasses = classNames.size

            // ── Session header ── written once per app launch to nova_debug.log
            fileLogger.i(TAG, "════════════════════════════════════════════════")
            fileLogger.i(TAG, "NOVA SESSION START")
            fileLogger.i(TAG, "Device: ${Build.MANUFACTURER} ${Build.MODEL} (${Build.DEVICE})")
            fileLogger.i(TAG, "Android: ${Build.VERSION.RELEASE}  API=${Build.VERSION.SDK_INT}  Board=${Build.BOARD}  hw=${Build.HARDWARE}")
            fileLogger.i(TAG, "RAM_max=${Runtime.getRuntime().maxMemory() / 1_048_576}MB  cores=${Runtime.getRuntime().availableProcessors()}")
            fileLogger.i(TAG, "NNAPI_ENABLED=${NovaConstants.NNAPI_ENABLED}  threads=4  yolo_input=${NovaConstants.YOLO_INPUT_SIZE}  classes=$numClasses")
            val yoloSize  = getAssetSize(NovaConstants.YOLO_MODEL_FILE)
            val midasSize = getAssetSize(NovaConstants.MIDAS_MODEL_FILE)
            fileLogger.i(TAG, "Assets — YOLO:${yoloSize}B  MiDaS:${midasSize}B")
            fileLogger.i(TAG, "════════════════════════════════════════════════")
            Log.i(TAG, "=== NOVA SESSION START === ${Build.MODEL} API=${Build.VERSION.SDK_INT}")

            if (yoloSize < 0) {
                throw java.io.FileNotFoundException(
                    "YOLO model not found: ${NovaConstants.YOLO_MODEL_FILE}. " +
                    "Add ${NovaConstants.YOLO_MODEL_FILE} to app/src/main/assets/"
                )
            }
            resolvedModelFile = NovaConstants.YOLO_MODEL_FILE
            val modelBuffer = loadModelFile(NovaConstants.YOLO_MODEL_FILE)
            fileLogger.i(TAG, "YOLO model loaded: ${modelBuffer.capacity()} bytes  file=${NovaConstants.YOLO_MODEL_FILE}")
            Log.i(TAG, "Model file loaded: size=${modelBuffer.capacity()} bytes")

            // Delegate selection — each tryAttach* function tests delegate + interpreter creation
            // atomically per candidate and returns the live Interpreter on success, null on failure.
            // Fallback chain: NNAPI → GPU → CPU/XNNPACK-builtin (all devices).
            interpreter = if (isQualcommDevice()) {
                // Qualcomm: QNN (Hexagon HTP) → NNAPI (legacy fallback) → GPU → CPU
                fileLogger.i(TAG, "Qualcomm SoC (hw=${Build.HARDWARE}) — trying QNN (Hexagon HTP)")
                val qnnInterp = tryAttachQnn(modelBuffer)
                val nnInterp = if (qnnInterp == null && NovaConstants.NNAPI_ENABLED) {
                    fileLogger.i(TAG, "QNN unavailable — trying NNAPI (FP16 → Adreno NNAPI)")
                    tryAttachNnapi(modelBuffer)
                } else null
                val fastInterp = qnnInterp ?: nnInterp
                if (fastInterp != null) {
                    fastInterp
                } else {
                    fileLogger.w(TAG, "QNN + NNAPI unavailable on Qualcomm — GPU fallback")
                    val gpuOpts = Interpreter.Options().apply { setNumThreads(4) }
                    if (tryAttachGpu(gpuOpts)) {
                        try {
                            val interp = Interpreter(modelBuffer, gpuOpts)
                            fileLogger.i(TAG, "GPU delegate active (Adreno)")
                            interp
                        } catch (e: Exception) {
                            fileLogger.w(TAG, "GPU failed: ${e.message} — CPU/XNNPACK fallback")
                            gpuDelegate?.close(); gpuDelegate = null
                            null
                        }
                    } else null
                }
            } else if (NovaConstants.NNAPI_ENABLED) {
                // Non-Qualcomm (MediaTek, Exynos, etc.)
                fileLogger.i(TAG, "Non-Qualcomm SoC (hw=${Build.HARDWARE}) — trying NNAPI then GPU then CPU")
                val nnApiInterp = tryAttachNnapiMtk(modelBuffer)
                if (nnApiInterp != null) {
                    nnApiInterp
                } else {
                    // Try GPU. Previous measurement (Helio G85/Mali-G52) showed GPU slower than
                    // XNNPACK, but G99/Mali-G57 and other SoCs may differ — let the CompatibilityList
                    // decide. If GPU init fails or is unsupported, falls through to CPU.
                    val gpuOpts = Interpreter.Options().apply { setNumThreads(4) }
                    if (tryAttachGpu(gpuOpts)) {
                        try {
                            val interp = Interpreter(modelBuffer, gpuOpts)
                            fileLogger.i(TAG, "GPU delegate active (${Build.HARDWARE})")
                            interp
                        } catch (e: Exception) {
                            fileLogger.w(TAG, "GPU failed: ${e.message} — CPU/XNNPACK fallback")
                            gpuDelegate?.close(); gpuDelegate = null
                            null
                        }
                    } else null
                }
            } else {
                // NNAPI_ENABLED=false (HiSilicon/Kirin or manual override) — GPU only
                fileLogger.i(TAG, "NNAPI disabled — GPU delegate only")
                val gpuOpts = Interpreter.Options().apply { setNumThreads(4) }
                if (tryAttachGpu(gpuOpts)) {
                    try { Interpreter(modelBuffer, gpuOpts) }
                    catch (e: Exception) {
                        fileLogger.w(TAG, "GPU failed: ${e.message}")
                        gpuDelegate?.close(); gpuDelegate = null
                        null
                    }
                } else null
            } ?: run {
                fileLogger.w(TAG, "Final fallback: CPU/XNNPACK-builtin (4 threads)")
                Interpreter(modelBuffer, Interpreter.Options().apply { setNumThreads(4) })
            }
            val interp = interpreter!!

            // -- Log input tensor info --
            val inputTensor = interp.getInputTensor(0)
            val inputShape  = inputTensor.shape()
            fileLogger.i(TAG, "Input tensor: shape=${inputShape.toList()} dtype=${inputTensor.dataType()}")
            Log.i(TAG, "Input tensor: shape=${inputShape.toList()} dtype=${inputTensor.dataType()} numBytes=${inputTensor.numBytes()}")

            check(inputTensor.dataType() == DataType.FLOAT32) {
                "YOLO input tensor dtype is ${inputTensor.dataType()}, expected FLOAT32. " +
                "bitmapToNHWC writes 4-byte floats -- a true FP16 input tensor would " +
                "overflow the buffer silently. Re-export the model with FLOAT32 I/O."
            }

            inputBuffer = ByteBuffer.allocateDirect(inputTensor.numBytes())
                .order(ByteOrder.nativeOrder())

            // -- Log + allocate ALL output tensors --
            val numOutputs = interp.outputTensorCount
            Log.i(TAG, "Output tensor count: $numOutputs")
            fileLogger.i(TAG, "Output tensor count: $numOutputs")

            outputBuffers = Array(numOutputs) { i ->
                val t = interp.getOutputTensor(i)
                fileLogger.i(TAG, "Output[$i]: shape=${t.shape().toList()} dtype=${t.dataType()} bytes=${t.numBytes()} name=${t.name()}")
                Log.i(TAG, "Output[$i]: shape=${t.shape().toList()} dtype=${t.dataType()} bytes=${t.numBytes()} name=${t.name()}")
                ByteBuffer.allocateDirect(t.numBytes()).order(ByteOrder.nativeOrder())
            }
            for (i in outputBuffers.indices) outputsMap[i] = outputBuffers[i]

            if (numOutputs >= 2) {
                val s0 = interp.getOutputTensor(0).shape().toList()
                val s1 = interp.getOutputTensor(1).shape().toList()
                val det0   = s0.any { it == 300 }
                val proto1 = s1.any { it == 160 } && s1.any { it == MASK_COEFF_COUNT }
                if (!det0)   fileLogger.w(TAG, "TENSOR_ORDER_WARN: output[0] shape=$s0 — expected E2E detections [1,300,N]")
                if (!proto1) fileLogger.w(TAG, "TENSOR_ORDER_WARN: output[1] shape=$s1 — expected proto masks [1,31,160,160]")
                if (det0 && proto1) fileLogger.i(TAG, "Tensor order OK: output[0]=detections[E2E] output[1]=proto")
                if (s0.any { it == 8400 } || s1.any { it == 8400 }) {
                    fileLogger.w(TAG, "UNEXPECTED: 8400 anchors detected — model may be NMS-free. " +
                        "This build only supports E2E NMS (nms=True export). Use server inference for NMS-free models.")
                }
            }

            val outShape = interp.getOutputTensor(0).shape()
            check(outShape.size == 3) {
                "YOLO output[0] shape ${outShape.toList()} is rank-${outShape.size}, expected rank-3. " +
                "Expected [1,300,37] (Run4 E2E) or [1,8400,65] (NMS-free). Check model export."
            }
            outputDim1 = outShape[1]
            outputDim2 = outShape[2]
            fileLogger.i(TAG, "Detection output[0]: [1,$outputDim1,$outputDim2] format=E2E_NMS")

            // Proto layout: Ultralytics E2E export is normally NCHW [1,32,160,160], but this
            // exported .tflite is NHWC [1,160,160,32] on some builds. Detected here so
            // computeMask() can pick the matching index formula — NOT a defect, both layouts
            // are handled correctly (see computeMask()'s protoIsNhwc branch). Logged at INFO,
            // not WARN, so it doesn't read as a live bug during field-log triage.
            if (outputBuffers.size > 1) {
                val protoShape = interp.getOutputTensor(1).shape()
                protoIsNhwc = protoShape.size == 4 && protoShape[1] != MASK_COEFF_COUNT && protoShape[3] == MASK_COEFF_COUNT
                val protoLayout = if (protoIsNhwc) "NHWC" else "NCHW"
                fileLogger.i(TAG, "Proto output[1]: shape=${protoShape.toList()} layout=$protoLayout")
                if (protoIsNhwc) {
                    fileLogger.i(TAG, "PROTO_LAYOUT_NHWC: proto tensor is NHWC — using NHWC-compensated mask indexing (masks unaffected).")
                }
            } else {
                fileLogger.w(TAG, "Proto output[1] missing — mask computation disabled")
            }

            // NNAPI on some SoCs (observed: MediaTek MT6789 "generic(runtime-routed)" candidate)
            // accepts the model and constructs an Interpreter successfully, but silently executes
            // on an internal CPU reference path rather than real hardware — Interpreter() not
            // throwing is NOT proof the accelerator is in use. setUseNnapiCpu(false) is supposed
            // to prevent this but doesn't cover every vendor's "generic" routing. Confirm with one
            // real warm-up inference; if NNAPI is clearly too slow to be hardware-accelerated,
            // fall back to GPU (then CPU) instead of running the whole session at CPU speed while
            // still logging "NNAPI attached ✓".
            if (nnApiDelegate != null) {
                val warmStart = System.currentTimeMillis()
                try {
                    inputBuffer!!.rewind()
                    interp.runForMultipleInputsOutputs(arrayOf(inputBuffer!!), outputsMap)
                } catch (e: Exception) {
                    fileLogger.w(TAG, "NNAPI warm-up inference failed: ${e.message}")
                }
                val warmMs = System.currentTimeMillis() - warmStart
                fileLogger.i(TAG, "NNAPI warm-up inference: ${warmMs}ms")
                if (warmMs > NovaConstants.NNAPI_LATENCY_FALLBACK_MS) {
                    fileLogger.w(TAG, "NNAPI warm-up ${warmMs}ms > ${NovaConstants.NNAPI_LATENCY_FALLBACK_MS}ms — " +
                        "treating as CPU-routed, not real hardware acceleration. Falling back to GPU.")
                    nnApiDelegate?.close()
                    nnApiDelegate = null
                    val gpuOpts = Interpreter.Options().apply { setNumThreads(4) }
                    interpreter = if (tryAttachGpu(gpuOpts)) {
                        try {
                            fileLogger.i(TAG, "GPU delegate active after NNAPI fallback")
                            Interpreter(modelBuffer, gpuOpts)
                        } catch (e: Exception) {
                            fileLogger.w(TAG, "GPU fallback failed: ${e.message} — CPU/XNNPACK fallback")
                            gpuDelegate?.close(); gpuDelegate = null
                            Interpreter(modelBuffer, Interpreter.Options().apply { setNumThreads(4) })
                        }
                    } else {
                        fileLogger.w(TAG, "GPU unavailable — CPU/XNNPACK fallback")
                        Interpreter(modelBuffer, Interpreter.Options().apply { setNumThreads(4) })
                    }
                }
            }

            val delegateLabel = when {
                nnApiDelegate != null -> "NNAPI"
                gpuDelegate   != null -> "GPU"
                else                  -> "CPU/XNNPACK-builtin"
            }
            fileLogger.i(TAG, "=== YOLO INIT OK ===")
            fileLogger.i(TAG, "  model   : $resolvedModelFile")
            fileLogger.i(TAG, "  format  : E2E_NMS — model-internal NMS (no CPU NMS)")
            fileLogger.i(TAG, "  delegate: $delegateLabel")
            fileLogger.i(TAG, "  classes : $numClasses")
            fileLogger.i(TAG, "  proto   : layout=${if (protoIsNhwc) "NHWC" else "NCHW"} NHWC=$protoIsNhwc")
            Log.i(TAG, "=== YOLO INIT OK === format=E2E_NMS delegate=$delegateLabel proto_nhwc=$protoIsNhwc")

            canvasBitmap = Bitmap.createBitmap(NovaConstants.YOLO_INPUT_SIZE, NovaConstants.YOLO_INPUT_SIZE, Bitmap.Config.ARGB_8888)
            canvasObj = Canvas(canvasBitmap!!)
            isInitialized = true

        } catch (e: Exception) {
            fileLogger.e(TAG, "YOLO INIT FAILED", e)
            isInitialized = false
        }
    }

    /**
     * Letterbox metadata: scale + padding offsets applied during preprocessing.
     */
    private data class LetterboxInfo(
        val bitmap: Bitmap,
        val scale: Float,
        val padLeft: Int,
        val padTop: Int
    )

    fun detect(bitmap: Bitmap): List<RawDetection> {
        if (!isInitialized) {
            fileLogger.w(TAG, "detect() called but detector not initialized")
            return emptyList()
        }
        val interp = interpreter ?: run {
            fileLogger.w(TAG, "detect() called but interpreter is null")
            return emptyList()
        }
        val buf = inputBuffer ?: run {
            fileLogger.w(TAG, "detect() called but inputBuffer is null")
            return emptyList()
        }

        val startTime  = System.currentTimeMillis()
        val origWidth  = bitmap.width
        val origHeight = bitmap.height

        // Preprocessing
        val preprocessStart = System.currentTimeMillis()
        val lb = letterboxBitmap(bitmap, NovaConstants.YOLO_INPUT_SIZE)
        bitmapToNHWC(lb.bitmap, buf)
        // F3: Do NOT recycle lb.bitmap — it is the pre-allocated canvasBitmap, reused next frame.
        val preprocessMs = System.currentTimeMillis() - preprocessStart

        buf.rewind()
        outputBuffers.forEach { it.rewind() }

        // Hoisted so they're accessible for the combined file-log after the try block
        var inferenceMs = 0L
        var parseMs = 0L
        var confNonZero = 0
        var confMin = 0f
        var confMax = 0f

        // Inference
        val inferenceStart = System.currentTimeMillis()
        val detections = try {
            interp.runForMultipleInputsOutputs(arrayOf(buf), outputsMap)
            inferenceMs = System.currentTimeMillis() - inferenceStart

            // Log first inference with actual latency — lets you confirm whether NNAPI is
            // actually running or silently fell back to CPU. Expected ranges:
            //   CPU/XNNPACK 4-thread: ~2000-5000ms    GPU delegate: ~200-600ms
            //   NNAPI Adreno FP16:   ~200-600ms       NNAPI MTK neuron: ~150-400ms
            if (!firstInferenceLogged) {
                firstInferenceLogged = true
                val delegate = when {
                    nnApiDelegate != null -> "NNAPI"
                    gpuDelegate   != null -> "GPU"
                    else                  -> "CPU/XNNPACK"
                }
                val warning = when {
                    nnApiDelegate != null && inferenceMs > 2000 ->
                        " ⚠ SLOW — NNAPI may have fallen back to CPU (expected <600ms)"
                    gpuDelegate != null && inferenceMs > 2000 ->
                        " ⚠ SLOW — GPU delegate may have fallen back to CPU"
                    else -> " ✓"
                }
                fileLogger.i(TAG, "First inference: ${inferenceMs}ms [$delegate]$warning")
            }

            outputBuffers[0].rewind()
            val totalFloats = outputDim1 * outputDim2
            val data = FloatArray(totalFloats)
            outputBuffers[0].asFloatBuffer().get(data)

            // Read prototype masks from output[1]: [1, 32, 160, 160] NCHW layout.
            // Needed to reconstruct per-detection masks for depth accuracy improvement.
            val protos: FloatArray? = if (outputBuffers.size > 1) {
                try {
                    outputBuffers[1].rewind()
                    FloatArray(outputBuffers[1].capacity() / 4).also {
                        outputBuffers[1].asFloatBuffer().get(it)
                    }
                } catch (e: Exception) {
                    fileLogger.w(TAG, "Proto mask read failed: ${e.message}")
                    null
                }
            } else null

            // Scan all 300 detection slots for raw confidence statistics
            var localMin = Float.MAX_VALUE; var localMax = 0f
            for (b in 0 until outputDim1) {
                val conf = data[b * outputDim2 + 4]
                if (conf > 1e-6f) {
                    confNonZero++
                    if (conf < localMin) localMin = conf
                    if (conf > localMax) localMax = conf
                }
            }
            if (confNonZero > 0) { confMin = localMin; confMax = localMax }

            // E2E NMS: feature[4] is sigmoid confidence (0-1). GPU overflow shows as > 1.0.
            if (confMax >= 1.0f || confMax.isNaN()) {
                fileLogger.w(TAG, "Frame[$frameCount] GPU_OVERFLOW: confRange=${"%.5f".format(confMin)}-${"%.5f".format(confMax)} nonzero=$confNonZero — skipping frame (tensor not yet stable)")
                emptyList<RawDetection>()
            } else {
                val parseStart = System.currentTimeMillis()
                val parsed = parseRawOutput(
                    data, outputDim1, outputDim2,
                    lb.scale, lb.padLeft, lb.padTop,
                    origWidth, origHeight,
                    protos
                )
                parseMs = System.currentTimeMillis() - parseStart
                parsed
            }
        } catch (e: Exception) {
            fileLogger.e(TAG, "TFLite inference error", e)
            emptyList()
        }

        frameCount++
        val elapsed = System.currentTimeMillis() - startTime

        // Log every frame for first 200 frames (dense diagnostic window), then every 10 frames.
        if (shouldLogToFile()) {
            val confRange = if (confNonZero > 0) "%.5f-%.5f".format(confMin, confMax) else "none"
            fileLogger.d(TAG, "Frame[$frameCount] ${elapsed}ms: prep=${preprocessMs}ms infer=${inferenceMs}ms parse=${parseMs}ms | slots=$outputDim1 nonzero=$confNonZero confRange=$confRange | dets=${detections.size}")
            if (detections.isNotEmpty()) {
                fileLogger.d(TAG, "Frame[$frameCount] dets: ${detections.map { "${it.className}(${"%.3f".format(it.confidence)})" }.joinToString()}")
            } else {
                fileLogger.d(TAG, "Frame[$frameCount] NO DETECTIONS — $confNonZero nonzero confs all below per-class thresholds")
            }
        }

        return detections
    }

    // -- Preprocessing --

    private fun letterboxBitmap(src: Bitmap, targetSize: Int): LetterboxInfo {
        val scale   = min(targetSize.toFloat() / src.width, targetSize.toFloat() / src.height)
        val scaledW = (src.width  * scale).toInt().coerceAtLeast(1)
        val scaledH = (src.height * scale).toInt().coerceAtLeast(1)
        val padLeft = (targetSize - scaledW) / 2
        val padTop  = (targetSize - scaledH) / 2

        // F3: Reuse pre-allocated canvas bitmap — avoids 1.6 MB alloc+recycle per frame.
        // Fall back to fresh allocation only if canvasBitmap was not yet initialized.
        val bitmap = canvasBitmap ?: Bitmap.createBitmap(targetSize, targetSize, Bitmap.Config.ARGB_8888)
        val c = canvasObj ?: Canvas(bitmap)
        c.drawColor(Color.BLACK)
        // Skip createScaledBitmap when src already matches — avoids a copy that would never be recycled.
        val scaled = if (src.width == scaledW && src.height == scaledH) src
                     else Bitmap.createScaledBitmap(src, scaledW, scaledH, true)
        c.drawBitmap(scaled, padLeft.toFloat(), padTop.toFloat(), Paint().apply { isFilterBitmap = true })
        if (scaled != src) scaled.recycle()

        return LetterboxInfo(bitmap, scale, padLeft, padTop)
    }

    private fun bitmapToNHWC(bitmap: Bitmap, buf: ByteBuffer) {
        val w = bitmap.width
        val h = bitmap.height
        val size = w * h
        val pixels = pixelBuffer?.takeIf { it.size == size } ?: IntArray(size).also { pixelBuffer = it }
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)

        buf.rewind()
        for (pixel in pixels) {
            buf.putFloat(((pixel shr 16) and 0xFF) / 255f)  // R
            buf.putFloat(((pixel shr 8)  and 0xFF) / 255f)  // G
            buf.putFloat((pixel          and 0xFF) / 255f)  // B
        }
    }

    // -- Postprocessing --

    /**
     * Parse YOLO26 end-to-end NMS output into RawDetection list.
     *
     * YOLO26 end-to-end TFLite produces [1, 300, 37] (Run4, 28 classes):
     *   300 = pre-filtered detection slots (NMS already applied by model internally)
     *   37  = 4 (x1,y1,x2,y2 xyxy corners) + 1 (confidence) + 1 (class_id float) + 31 (mask coefficients)
     *   (Run2 was [1,300,38] with 32 mask coefficients — code auto-adapts via numFeatures)
     *
     * Key differences from raw pre-NMS [1, 65, 8400] format:
     *   Coordinates are xyxy CORNERS, not cx,cy,w,h center format.
     *   class_id is a single float (cast to int), not an array of per-class scores.
     *   Confidence is post-NMS score (0.30-0.85 range), not raw logit (0.001-0.052 range).
     *   NO manual NMS call -- model already deduplicated internally.
     *   Empty padding slots have confidence ~0 and are skipped.
     */
    private fun parseRawOutput(
        data: FloatArray,
        dim1: Int,
        dim2: Int,
        lbScale: Float,
        lbPadLeft: Int,
        lbPadTop: Int,
        origW: Int,
        origH: Int,
        protos: FloatArray?
    ): List<RawDetection> {

        val numBoxes: Int
        val numFeatures: Int
        val transposed: Boolean

        if (dim1 >= dim2) {
            numBoxes = dim1; numFeatures = dim2; transposed = false
        } else {
            numBoxes = dim2; numFeatures = dim1; transposed = true
        }

        if (numFeatures < 6) {
            fileLogger.e(TAG, "E2E output: features=$numFeatures < 6. Shape=[1,$dim1,$dim2]. Expected [1,300,37] (Run4 E2E) or [1,300,38] (Run2 E2E).")
            Log.e(TAG, "E2E output: features=$numFeatures < 6. Shape=[1,$dim1,$dim2]. Expected [1,300,37] (Run4 E2E) or [1,300,38] (Run2 E2E).")
            return emptyList()
        }

        // -- Coordinate space auto-detection --
        val sampleCount = minOf(50, numBoxes)
        var maxX1Sample = 0f
        for (i in 0 until sampleCount) {
            val x1Val = if (transposed) data[0 * numBoxes + i] else data[i * numFeatures + 0]
            if (x1Val > maxX1Sample) maxX1Sample = x1Val
        }
        val coordsNormalized = maxX1Sample < 2.0f
        val toPixel = if (coordsNormalized) NovaConstants.YOLO_INPUT_SIZE.toFloat() else 1.0f

        if (!coordSpaceLogged && maxX1Sample > 0.05f) {
            val msg = "E2E coord space: maxX1=${"%.4f".format(maxX1Sample)} normalized=$coordsNormalized toPixel=$toPixel"
            fileLogger.i(TAG, msg)
            coordSpaceLogged = true
        }

        val maskCoeffStart = 6  // E2E NMS: 4(xyxy) + 1(conf) + 1(cls_id)

        var passedThreshold = 0
        val detections = mutableListOf<RawDetection>()
        val top5Confs = FloatArray(5) { 0f }

        for (b in 0 until numBoxes) {
            // E2E NMS: x1,y1,x2,y2 (xyxy corners) | conf | cls_id (float) | mask[31]
            val rawX1: Float; val rawY1: Float; val rawX2: Float; val rawY2: Float
            val confidence: Float; val classId: Int

            if (transposed) {
                rawX1 = data[0 * numBoxes + b] * toPixel; rawY1 = data[1 * numBoxes + b] * toPixel
                rawX2 = data[2 * numBoxes + b] * toPixel; rawY2 = data[3 * numBoxes + b] * toPixel
                confidence = data[4 * numBoxes + b]
                classId    = data[5 * numBoxes + b].roundToInt().coerceIn(0, numClasses - 1)
            } else {
                val base = b * numFeatures
                rawX1 = data[base + 0] * toPixel; rawY1 = data[base + 1] * toPixel
                rawX2 = data[base + 2] * toPixel; rawY2 = data[base + 3] * toPixel
                confidence = data[base + 4]
                classId    = data[base + 5].roundToInt().coerceIn(0, numClasses - 1)
            }
            if (confidence < 1e-4f) continue
            if (confidence > 1.0f || confidence.isNaN() || confidence.isInfinite()) continue

            if (confidence > top5Confs[4]) {
                top5Confs[4] = confidence
                var k = 4
                while (k > 0 && top5Confs[k] > top5Confs[k - 1]) {
                    val tmp = top5Confs[k]; top5Confs[k] = top5Confs[k - 1]; top5Confs[k - 1] = tmp
                    k--
                }
            }

            val className = classNames.getOrElse(classId) { "unknown" }
            val threshold = ClassConfig.getThreshold(className)

            if (confidence >= threshold) {
                passedThreshold++

                val bbox = RectF(
                    ((rawX1 - lbPadLeft) / (lbScale * origW)).coerceIn(0f, 1f),
                    ((rawY1 - lbPadTop)  / (lbScale * origH)).coerceIn(0f, 1f),
                    ((rawX2 - lbPadLeft) / (lbScale * origW)).coerceIn(0f, 1f),
                    ((rawY2 - lbPadTop)  / (lbScale * origH)).coerceIn(0f, 1f)
                )

                val mask: FloatArray? = if (
                    protos != null &&
                    className in MASK_CLASSES &&
                    numFeatures >= maskCoeffStart + MASK_COEFF_COUNT
                ) {
                    computeMask(
                        data, b, transposed, numBoxes, numFeatures,
                        protos, lbScale, lbPadLeft, lbPadTop, origW, origH,
                        maskCoeffStart
                    )
                } else null

                detections.add(RawDetection(
                    classId    = classId,
                    className  = className,
                    confidence = confidence,
                    bbox       = bbox,
                    mask       = mask
                ))
            }
        }

        val top5 = top5Confs.filter { it > 0f }.map { "%.4f".format(it) }
        if (shouldLogToFile()) {
            fileLogger.d(TAG, "E2EParse: [1,$dim1,$dim2] transposed=$transposed passedThresh=$passedThreshold dets=${detections.size} coordNorm=$coordsNormalized topScores=$top5")
        }

        val result = detections.sortedByDescending { it.confidence }.take(NovaConstants.MAX_DETECTIONS_PER_FRAME)
        updateClassCounts(result)
        return result
    }

    /**
     * Reconstruct per-detection segmentation mask in normalized original-image space.
     * Formula: sigmoid(coefficients[32] @ protos.reshape(32, 160×160)) → 40×40 float grid.
     * Proto tensor layout is NCHW [1,32,160,160] or NHWC [1,160,160,32] depending on the
     * export (see protoIsNhwc, set at init) — indexed accordingly below either way.
     */
    private fun computeMask(
        data: FloatArray,
        b: Int,
        transposed: Boolean,
        numBoxes: Int,
        numFeatures: Int,
        protos: FloatArray,
        lbScale: Float,
        lbPadLeft: Int,
        lbPadTop: Int,
        origW: Int,
        origH: Int,
        maskCoeffStart: Int
    ): FloatArray {
        val coeffs = FloatArray(MASK_COEFF_COUNT) { ci ->
            if (transposed) data[(maskCoeffStart + ci) * numBoxes + b]
            else            data[b * numFeatures + maskCoeffStart + ci]
        }

        val yoloSize = NovaConstants.YOLO_INPUT_SIZE.toFloat()
        val outMask  = FloatArray(MASK_OUT * MASK_OUT)

        for (oy in 0 until MASK_OUT) {
            for (ox in 0 until MASK_OUT) {
                val normX = (ox + 0.5f) / MASK_OUT
                val normY = (oy + 0.5f) / MASK_OUT

                val lbX = normX * origW * lbScale + lbPadLeft
                val lbY = normY * origH * lbScale + lbPadTop
                if (lbX < 0f || lbX >= yoloSize || lbY < 0f || lbY >= yoloSize) continue

                val protoX = (lbX / yoloSize * PROTO_SIZE).toInt().coerceIn(0, PROTO_SIZE - 1)
                val protoY = (lbY / yoloSize * PROTO_SIZE).toInt().coerceIn(0, PROTO_SIZE - 1)

                // Dot product: coefficients × proto column → sigmoid.
                // Index formula depends on tensor layout written by the exporter:
                //   NCHW [1,32,160,160] (Ultralytics/E2E): protos[ci * H*W + py*W + px]
                //   NHWC [1,160,160,32] (onnx2tf/NMS-free): protos[py * W*C + px * C + ci]
                var dot = 0f
                if (protoIsNhwc) {
                    val base = protoY * PROTO_SIZE * MASK_COEFF_COUNT + protoX * MASK_COEFF_COUNT
                    for (ci in 0 until MASK_COEFF_COUNT) dot += coeffs[ci] * protos[base + ci]
                } else {
                    val spatialIdx = protoY * PROTO_SIZE + protoX
                    for (ci in 0 until MASK_COEFF_COUNT) {
                        dot += coeffs[ci] * protos[ci * PROTO_SIZE * PROTO_SIZE + spatialIdx]
                    }
                }
                outMask[oy * MASK_OUT + ox] = 1f / (1f + kotlin.math.exp(-dot))
            }
        }
        return outMask
    }

    private fun calculateIoU(a: RectF, b: RectF): Float {
        val iL = maxOf(a.left, b.left);  val iT = maxOf(a.top, b.top)
        val iR = minOf(a.right, b.right); val iB = minOf(a.bottom, b.bottom)
        if (iL >= iR || iT >= iB) return 0f
        val iArea = (iR - iL) * (iB - iT)
        val uArea = a.width() * a.height() + b.width() * b.height() - iArea
        return if (uArea > 0) iArea / uArea else 0f
    }

    // -- Utilities --

    /** Log every frame for first 200 frames, then every 10 — dense diagnostic window. */
    private fun shouldLogToFile(): Boolean = frameCount <= 200L || frameCount % 10L == 0L

    /** Track per-class detection counts; log a summary every 100 frames. */
    private fun updateClassCounts(detections: List<RawDetection>) {
        for (d in detections) classDetectionCounts[d.className] = (classDetectionCounts[d.className] ?: 0) + 1
        classCountFrameWindow++
        if (classCountFrameWindow >= 100) {
            val sorted = classDetectionCounts.entries.sortedByDescending { it.value }
            val summary = sorted.joinToString(" | ") { "${it.key}=${it.value}" }
            fileLogger.i(TAG, "ClassCounts[last 100 frames]: $summary")
            classDetectionCounts.clear()
            classCountFrameWindow = 0
        }
    }

    /** Returns the declared byte size of an assets file, or -1 if the file is absent. */
    private fun getAssetSize(filename: String): Long {
        return try {
            context.assets.openFd(filename).use { it.declaredLength }
        } catch (e: Exception) {
            fileLogger.w(TAG, "Asset not found: $filename — ${e.message}")
            -1L
        }
    }

    private fun loadModelFile(filename: String): MappedByteBuffer {
        return context.assets.openFd(filename).use { fd ->
            FileInputStream(fd.fileDescriptor).use { stream ->
                stream.channel.use { channel ->
                    channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
                }
            }
        }
    }

    /**
     * Adaptive GPU warm-up: runs inferences until output tensor values are stable (no overflow),
     * or until MAX_WARMUP_PASSES is reached.
     *
     * Mali-G57 MC2 driver (Helio G85) does not reset OpenCL shader state between app sessions.
     * The first N inferences after a cold start produce FP16 tensor values >> 1.0 (observed up
     * to 16,607,732 in logs) because residual GPU memory from the previous session is interpreted
     * as live FP16 outputs. Running inferences until confMax drops below 1.0 for 3 consecutive
     * passes ensures the shader pipeline is fully initialized before live frames are processed.
     *
     * Fixed pass-count warmup (WARMUP_PASSES=5) was insufficient: logs show overflow persisting
     * for 8-12 frames on SM-A155F after a fresh session start.
     */
    fun warmup() {
        val interp = interpreter ?: return
        val buf = inputBuffer ?: return
        if (!isInitialized) return

        fileLogger.i(TAG, "Warm-up: flushing GPU shader state (max ${NovaConstants.MAX_WARMUP_PASSES} passes, need 3 clean)")

        // Small random noise — zeros cause GATHER_ND index=0 crash on End2End NMS graphs.
        // Seed fixed so warmup is deterministic across sessions.
        buf.rewind()
        val rng = java.util.Random(42L)
        repeat(buf.capacity() / 4) { buf.putFloat(rng.nextFloat() * 0.1f) }

        // Allocate once — avoids up to 25 × FloatArray(outputDim1 × outputDim2) GC allocs inside the loop.
        val data = FloatArray(outputDim1 * outputDim2)

        var cleanStreak = 0
        var pass = 0
        while (pass < NovaConstants.MAX_WARMUP_PASSES && cleanStreak < 3) {
            buf.rewind()
            outputBuffers.forEach { it.rewind() }
            try {
                interp.runForMultipleInputsOutputs(arrayOf(buf), outputsMap)
                outputBuffers[0].rewind()
                outputBuffers[0].asFloatBuffer().get(data)
                val confMax = (0 until outputDim1).maxOfOrNull { i -> data[i * outputDim2 + 4] } ?: 0f
                // E2E NMS: feature[4] is sigmoid confidence (0-1). Overflow shows as > 1.0.
                if (confMax >= 1.0f || confMax.isNaN()) {
                    fileLogger.d(TAG, "Warm-up[$pass]: GPU_OVERFLOW confMax=${"%.2f".format(confMax)} — continuing")
                    cleanStreak = 0
                } else {
                    cleanStreak++
                    fileLogger.d(TAG, "Warm-up[$pass]: clean confMax=${"%.5f".format(confMax)} streak=$cleanStreak/3")
                }
            } catch (e: Exception) {
                fileLogger.w(TAG, "Warm-up[$pass] failed: ${e.message}")
                break
            }
            pass++
        }
        fileLogger.i(TAG, "Warm-up complete: $pass passes, finalStreak=$cleanStreak/3 ${if (cleanStreak >= 3) "✓" else "⚠ max reached"}")
    }

    private fun isQualcommDevice(): Boolean {
        val hw = Build.HARDWARE.lowercase()
        if (hw.contains("qcom") || hw.contains("qualcomm")) return true
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            val soc = Build.SOC_MODEL.lowercase()
            if (soc.contains("sm") || soc.contains("qcm") || soc.contains("snapdragon")) return true
        }
        return false
    }

    private fun tryAttachQnn(modelBuffer: java.nio.MappedByteBuffer): Interpreter? {
        return try {
            val opts = QnnDelegate.Options()
            opts.setBackendType(QnnDelegate.Options.BackendType.HTP_BACKEND)
            opts.setSkelLibraryDir(context.applicationInfo.nativeLibraryDir)
            qnnDelegate = QnnDelegate(opts)
            val interpOpts = Interpreter.Options().apply { setNumThreads(4); addDelegate(qnnDelegate!!) }
            val interp = Interpreter(modelBuffer, interpOpts)
            fileLogger.i(TAG, "QNN delegate active — Hexagon HTP (hw=${Build.HARDWARE}) ✓")
            interp
        } catch (e: Exception) {
            fileLogger.w(TAG, "QNN delegate unavailable: ${e.message?.take(100)}")
            qnnDelegate?.close(); qnnDelegate = null
            null
        }
    }

    private fun tryAttachGpu(options: Interpreter.Options): Boolean {
        val gpuCompat = CompatibilityList()
        if (!gpuCompat.isDelegateSupportedOnThisDevice) {
            fileLogger.w(TAG, "GPU not supported on ${Build.MODEL}")
            return false
        }
        return try {
            gpuDelegate = GpuDelegate(gpuCompat.bestOptionsForThisDevice)
            options.addDelegate(gpuDelegate!!)
            fileLogger.i(TAG, "GPU delegate attached — OpenCL/GLES active (${Build.MODEL})")
            true
        } catch (e: Exception) {
            fileLogger.w(TAG, "GPU delegate failed: ${e.message}")
            gpuDelegate?.close(); gpuDelegate = null
            false
        }
    }

    /**
     * Non-Qualcomm NNAPI path — tests each candidate atomically (delegate + interpreter).
     *
     * Previous failure mode: NnApiDelegate("mtk-neuron") constructor succeeds even when the
     * accelerator doesn't exist; the failure only surfaces at Interpreter() creation time.
     * This meant the candidate loop never advanced past the first name. Fix: create and test
     * the Interpreter for each candidate before committing.
     *
     * setUseNnapiCpu(false) prevents routing to "nnapi-reference" (software CPU fallback),
     * which is slower than TFLite's built-in XNNPACK. If only the reference implementation
     * is available, the candidate fails fast.
     */
    private fun tryAttachNnapiMtk(modelBuffer: java.nio.MappedByteBuffer): Interpreter? {
        if (!NovaConstants.NNAPI_ENABLED) {
            fileLogger.i(TAG, "NNAPI disabled — skipping")
            return null
        }
        context.cacheDir.mkdirs()
        val hw  = Build.HARDWARE
        val soc = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) Build.SOC_MODEL else "n/a"
        fileLogger.i(TAG, "NNAPI attempt: hw=$hw  soc=$soc  api=${Build.VERSION.SDK_INT}")

        val modelSlug = resolvedModelFile.removeSuffix(".tflite").replace("-", "_")
        // null entry = generic (no explicit accelerator name)
        val candidates = listOf("mtk-neuron", "neuron-ann", "neuron", null)
        for (accelName in candidates) {
            val label = accelName ?: "generic(runtime-routed)"
            var delegate: NnApiDelegate? = null
            try {
                delegate = NnApiDelegate(NnApiDelegate.Options().apply {
                    if (accelName != null) setAcceleratorName(accelName)
                    setAllowFp16(true)
                    setUseNnapiCpu(false)
                    setExecutionPreference(NnApiDelegate.Options.EXECUTION_PREFERENCE_FAST_SINGLE_ANSWER)
                    setModelToken("${modelSlug}_${(accelName ?: "generic").replace("-", "_")}")
                    setCacheDir(context.cacheDir.absolutePath)
                })
                val testOpts = Interpreter.Options().apply { setNumThreads(4); addDelegate(delegate) }
                val interp = Interpreter(modelBuffer, testOpts)
                nnApiDelegate = delegate
                fileLogger.i(TAG, "NNAPI attached: accel='$label' hw=$hw ✓")
                return interp
            } catch (e: Exception) {
                fileLogger.w(TAG, "NNAPI_FAILED accel='$label': ${e.message?.take(120)}")
                delegate?.close()
            }
        }
        fileLogger.w(TAG, "NNAPI unavailable on $hw — no hardware accelerator found")
        return null
    }

    /**
     * Qualcomm FP16 NNAPI path — routes to Adreno GPU via NNAPI (allowFp16=true).
     *
     * On first launch the NNAPI compilation cache file does not exist yet. Previous behaviour:
     * Interpreter() throws "file couldn't be opened for reading". Fix: on cache-miss error,
     * retry without model token so NNAPI compiles from scratch; subsequent launches use cache.
     */
    private fun tryAttachNnapi(modelBuffer: java.nio.MappedByteBuffer): Interpreter? {
        if (!NovaConstants.NNAPI_ENABLED) {
            fileLogger.i(TAG, "NNAPI disabled (${Build.HARDWARE}) — skipping")
            return null
        }
        context.cacheDir.mkdirs()
        val modelSlug = resolvedModelFile.removeSuffix(".tflite").replace("-", "_")

        // Attempt 1: with compilation cache
        var delegate: NnApiDelegate? = null
        try {
            delegate = NnApiDelegate(NnApiDelegate.Options().apply {
                setAllowFp16(true)
                setUseNnapiCpu(false)
                setModelToken("${modelSlug}_nnapi_fp16")
                setCacheDir(context.cacheDir.absolutePath)
            })
            val testOpts = Interpreter.Options().apply { setNumThreads(4); addDelegate(delegate) }
            val interp = Interpreter(modelBuffer, testOpts)
            nnApiDelegate = delegate
            fileLogger.i(TAG, "NNAPI attached (FP16, hw=${Build.HARDWARE}, cache=active) ✓")
            return interp
        } catch (e: Exception) {
            fileLogger.w(TAG, "NNAPI FP16 failed: ${e.message?.take(120)}")
            delegate?.close(); delegate = null
        }

        // Attempt 2: cache miss on first launch — compile without token, cache builds afterward
        val msg = "NNAPI cache miss on first launch — retrying without cache token"
        fileLogger.i(TAG, msg)
        try {
            delegate = NnApiDelegate(NnApiDelegate.Options().apply {
                setAllowFp16(true)
                setUseNnapiCpu(false)
            })
            val testOpts = Interpreter.Options().apply { setNumThreads(4); addDelegate(delegate) }
            val interp = Interpreter(modelBuffer, testOpts)
            nnApiDelegate = delegate
            fileLogger.i(TAG, "NNAPI attached (FP16, hw=${Build.HARDWARE}, cache=building) ✓")
            return interp
        } catch (e2: Exception) {
            fileLogger.w(TAG, "NNAPI FP16 no-cache also failed: ${e2.message?.take(80)}")
            delegate?.close()
        }
        return null
    }

    fun close() {
        interpreter?.close()
        gpuDelegate?.close()
        nnApiDelegate?.close()
        qnnDelegate?.close()
        interpreter = null
        gpuDelegate = null
        nnApiDelegate = null
        qnnDelegate = null
        isInitialized = false
        coordSpaceLogged = false
        pixelBuffer = null
        canvasBitmap?.recycle(); canvasBitmap = null
        canvasObj = null
        frameCount = 0L
    }
}
