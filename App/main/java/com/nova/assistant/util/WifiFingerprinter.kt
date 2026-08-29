package com.nova.assistant.util

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.net.wifi.WifiManager
import androidx.core.app.ActivityCompat
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import kotlin.math.sqrt

object WifiFingerprinter {

    /**
     * Capture the top-10 strongest APs as a JSON fingerprint.
     * Uses cached scan results — no startScan(), no throttle risk.
     * Returns "[]" if permission denied or WiFi off.
     * suspend + IO: WifiManager.scanResults is a Binder IPC call; keep it off Main.
     */
    suspend fun capture(context: Context): String = withContext(Dispatchers.IO) {
        if (ActivityCompat.checkSelfPermission(context, Manifest.permission.ACCESS_FINE_LOCATION)
            != PackageManager.PERMISSION_GRANTED
        ) return@withContext "[]"

        val wm = context.getSystemService(Context.WIFI_SERVICE) as WifiManager
        val results = try {
            wm.scanResults
        } catch (_: SecurityException) {
            return@withContext "[]"
        }

        val arr = JSONArray()
        results
            .sortedByDescending { it.level }
            .take(10)
            .forEach { ap ->
                arr.put(JSONObject().apply {
                    put("b", ap.BSSID ?: "")
                    put("r", ap.level)
                })
            }
        arr.toString()
    }

    /**
     * Cosine similarity between two fingerprints.
     * Requires ≥3 shared BSSIDs to avoid false positives from a single shared AP.
     * Magnitudes computed over each full vector (not just the intersection) so
     * scores are not inflated when fingerprints share only a subset of their APs.
     * Returns 0f if insufficient overlap.
     */
    fun similarity(fp1: String, fp2: String): Float {
        val m1 = parseFingerprint(fp1)
        val m2 = parseFingerprint(fp2)
        val common = m1.keys.intersect(m2.keys)
        if (common.size < 3) return 0f

        // Shift RSSI by +100 so values are positive (dBm range -100..-30 → 0..70)
        val dot = common.sumOf { b -> ((m1[b]!! + 100f) * (m2[b]!! + 100f)).toDouble() }.toFloat()
        val mag1 = sqrt(m1.values.sumOf { ((it + 100f) * (it + 100f)).toDouble() }.toFloat())
        val mag2 = sqrt(m2.values.sumOf { ((it + 100f) * (it + 100f)).toDouble() }.toFloat())

        return if (mag1 == 0f || mag2 == 0f) 0f else (dot / (mag1 * mag2)).coerceIn(0f, 1f)
    }

    private fun parseFingerprint(fp: String): Map<String, Float> {
        return try {
            val arr = JSONArray(fp)
            buildMap {
                for (i in 0 until arr.length()) {
                    val o = arr.getJSONObject(i)
                    val bssid = o.optString("b", "")
                    if (bssid.isNotEmpty()) put(bssid, o.getInt("r").toFloat())
                }
            }
        } catch (_: Exception) { emptyMap() }
    }
}
