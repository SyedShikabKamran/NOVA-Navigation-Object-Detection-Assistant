package com.nova.assistant.ml

import android.graphics.Bitmap
import android.graphics.RectF
import com.nova.assistant.server.NovaInferenceClient
import com.nova.assistant.util.FileLogger
import com.nova.assistant.util.NovaConstants
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.util.Base64
import java.util.concurrent.TimeUnit
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Result of a Finder search — distinguishes "nothing matched the query" from "couldn't even
 * reach the server", since a blind user relying on TTS has no other way to tell the difference
 * ("try moving the camera" is useless advice when the real problem is no connection).
 */
sealed class FindResult {
    data class Found(val box: RectF) : FindResult()
    object NotFound : FindResult()
    object ServerUnavailable : FindResult()
}

/**
 * Finder-tab open-vocabulary "find X".
 *
 * Calls the NOVA GPU server's POST /find endpoint (Grounding DINO tiny, same T4 running
 * YOLO/depth). Returns the highest-confidence box as a normalized RectF, or null if the
 * server is unreachable or nothing was detected above threshold.
 */
@Singleton
class GroundingDinoClient @Inject constructor(
    private val fileLogger: FileLogger,
    private val inferenceClient: NovaInferenceClient,
) {
    private companion object {
        private const val TAG = "GroundingDinoClient"
        val JSON_TYPE = "application/json; charset=utf-8".toMediaType()
        const val IMAGE_MAX_WIDTH = 960
    }

    private val http = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .build()

    /**
     * Find [query] in [bitmap]. See [FindResult] — ServerUnavailable and NotFound are
     * deliberately distinct so the caller can speak different feedback for each.
     */
    suspend fun locate(bitmap: Bitmap, query: String): FindResult = withContext(Dispatchers.IO) {
        fileLogger.i(TAG, "locate: query=\"$query\" bitmap=${bitmap.width}x${bitmap.height}")
        val base = inferenceClient.httpBaseUrl() ?: run {
            fileLogger.w(TAG, "locate: no server URL — Finder unavailable")
            return@withContext FindResult.ServerUnavailable
        }
        fileLogger.i(TAG, "locate: posting to $base/find threshold=${NovaConstants.GROUNDING_DINO_CONFIDENCE}")
        val b64Start = System.currentTimeMillis()
        val b64 = bitmapToBase64(bitmap)
        fileLogger.d(TAG, "locate: base64 encoded ${b64.length} chars in ${System.currentTimeMillis() - b64Start}ms")

        val requestBody = JSONObject().apply {
            put("image_b64", b64)
            put("query", query)
            put("threshold", NovaConstants.GROUNDING_DINO_CONFIDENCE.toDouble())
        }.toString().toRequestBody(JSON_TYPE)

        val request = Request.Builder().url("$base/find").post(requestBody).build()
        val t0 = System.currentTimeMillis()
        val response = runCatching { http.newCall(request).execute() }.getOrNull()
            ?: run {
                fileLogger.w(TAG, "locate: HTTP call threw — server unreachable (${System.currentTimeMillis() - t0}ms)")
                return@withContext FindResult.ServerUnavailable
            }
        val elapsed = System.currentTimeMillis() - t0
        val body = response.body?.string()
        if (!response.isSuccessful || body == null) {
            fileLogger.w(TAG, "locate: HTTP ${response.code} after ${elapsed}ms body=${body?.take(200)}")
            return@withContext FindResult.ServerUnavailable
        }
        fileLogger.i(TAG, "locate: HTTP 200 in ${elapsed}ms bodyLen=${body.length}")
        val box = parseBoxes(body)
        if (box != null) {
            fileLogger.i(TAG, "locate: FOUND box=[${box.left},${box.top},${box.right},${box.bottom}] cx=${box.centerX()}")
            FindResult.Found(box)
        } else {
            fileLogger.i(TAG, "locate: NOT FOUND (no boxes above threshold) rawBody=${body.take(300)}")
            FindResult.NotFound
        }
    }

    /** Server returns boxes normalized to [0, 1] as [x1, y1, x2, y2]. */
    private fun parseBoxes(json: String): RectF? = runCatching {
        val obj = JSONObject(json)
        val boxes = obj.getJSONArray("boxes")
        val scores = obj.getJSONArray("scores")
        fileLogger.d(TAG, "parseBoxes: ${boxes.length()} boxes returned")
        if (boxes.length() == 0) return null

        var bestIdx = 0
        var bestScore = -1.0
        for (i in 0 until scores.length()) {
            val s = scores.getDouble(i)
            fileLogger.d(TAG, "parseBoxes: box[$i] score=${"%.3f".format(s)}")
            if (s > bestScore) { bestScore = s; bestIdx = i }
        }
        fileLogger.i(TAG, "parseBoxes: best box[$bestIdx] score=${"%.3f".format(bestScore)}")
        val box = boxes.getJSONArray(bestIdx)
        RectF(
            box.getDouble(0).toFloat(),
            box.getDouble(1).toFloat(),
            box.getDouble(2).toFloat(),
            box.getDouble(3).toFloat()
        )
    }.getOrElse { e ->
        fileLogger.e(TAG, "parseBoxes: JSON parse error: ${e.message} json=${json.take(200)}")
        null
    }

    private fun bitmapToBase64(bitmap: Bitmap): String {
        val scaled = if (bitmap.width > IMAGE_MAX_WIDTH) {
            val ratio = IMAGE_MAX_WIDTH.toFloat() / bitmap.width
            val sw = IMAGE_MAX_WIDTH
            val sh = (bitmap.height * ratio).toInt()
            fileLogger.d(TAG, "bitmapToBase64: downscaling ${bitmap.width}x${bitmap.height} → ${sw}x${sh}")
            Bitmap.createScaledBitmap(bitmap, sw, sh, true)
        } else {
            fileLogger.d(TAG, "bitmapToBase64: no downscale needed (width=${bitmap.width} <= $IMAGE_MAX_WIDTH)")
            bitmap
        }
        val out = ByteArrayOutputStream()
        scaled.compress(Bitmap.CompressFormat.JPEG, 85, out)
        val bytes = out.toByteArray()
        fileLogger.d(TAG, "bitmapToBase64: JPEG compressed to ${bytes.size / 1024}KB")
        return Base64.getEncoder().encodeToString(bytes)
    }
}
