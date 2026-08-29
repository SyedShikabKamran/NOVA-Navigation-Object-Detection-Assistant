package com.nova.assistant.features

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Build
import android.os.Bundle
import android.os.Looper
import android.telephony.SmsManager
import android.util.Log
import androidx.core.app.ActivityCompat
import com.nova.assistant.util.FileLogger
import com.google.android.gms.location.FusedLocationProviderClient
import com.google.android.gms.location.LocationCallback
import com.google.android.gms.location.LocationRequest
import com.google.android.gms.location.LocationResult
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.Priority
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import com.nova.assistant.data.local.RoomSnapshotDao
import com.nova.assistant.data.local.RoomSnapshotEntity
import com.nova.assistant.data.local.SettingsDao
import com.nova.assistant.data.local.SettingsKeys
import com.nova.assistant.engine.DetectionWithDepth
import com.nova.assistant.feedback.FeedbackOrchestrator
import com.nova.assistant.util.NovaConstants
import com.nova.assistant.util.WifiFingerprinter
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.tasks.await
import org.json.JSONArray
import org.json.JSONObject
import javax.inject.Inject
import javax.inject.Singleton
import kotlin.coroutines.resume

// ═════════════════════════════════════════════════════════════════
// ROOM SNAPSHOT ENGINE
// ═════════════════════════════════════════════════════════════════
@Singleton
class RoomSnapshotEngine @Inject constructor(
    @dagger.hilt.android.qualifiers.ApplicationContext private val context: Context,
    private val roomDao: RoomSnapshotDao,
    private val feedback: FeedbackOrchestrator,
    private val fileLogger: FileLogger,
) {
    companion object {
        private const val TAG = "RoomSnapshotEngine"
    }

    private var isRecording = false
    private val collectedDetections = mutableListOf<List<DetectionWithDepth>>()

    // Set by SavedRoomsScreen "Scan New Room" button; consumed by MainNavigationScreen.
    val pendingScanName = kotlinx.coroutines.flow.MutableStateFlow<String?>(null)

    // Dedup guard: don't re-announce same room within 5 minutes
    @Volatile private var lastAnnouncedRoom = ""
    @Volatile private var lastAnnouncedAt = 0L

    fun startRecording() {
        if (isRecording) return
        isRecording = true
        collectedDetections.clear()
        feedback.speakSystem("Scanning room. Please slowly look around for 20 seconds.")
        fileLogger.i(TAG, "Started room scan")
    }

    /**
     * Add a frame's detections during recording.
     * Called on every frame result while recording (see MainNavigationScreen's frame-results
     * collector) during the ~20-second scan (10s + "Half way" + 8s + "Almost done" + 2s).
     */
    fun addFrame(detections: List<DetectionWithDepth>) {
        if (!isRecording) return
        collectedDetections.add(detections)
        val n = collectedDetections.size
        if (n == 1 || n % 20 == 0) {
            val classes = detections.map { it.className }.distinct().joinToString()
            fileLogger.d(TAG, "addFrame: frame#$n detections=${detections.size} classes=[$classes]")
        }
    }

    /**
     * Finish recording and save room snapshot.
     * Bug fix: guard on collectedDetections.isEmpty() not !isRecording — pauseRecording()
     * sets isRecording=false before this is called, so the old guard always returned null.
     */
    suspend fun finishRecording(roomName: String, headingDeg: Float = -1f): RoomSnapshotEntity? {
        val totalFrames = collectedDetections.size
        fileLogger.i(TAG, "finishRecording: roomName=\"$roomName\" heading=$headingDeg° frames=$totalFrames")
        if (collectedDetections.isEmpty()) {
            isRecording = false
            // Previously silent: caller neither checked this null nor announced anything, so a
            // stalled camera/pipeline during the scan window (e.g. permission hiccup, frame
            // analyzer paused) meant the room silently failed to save with no user-facing signal
            // at all in a voice-first, non-visual app. Announce the failure explicitly.
            feedback.speakSystem("Room scan failed. No data was captured. Please try again.")
            fileLogger.w(TAG, "Room scan aborted for '$roomName' — zero frames collected")
            return null
        }
        isRecording = false

        val totalDets = collectedDetections.sumOf { it.size }
        val allClasses = collectedDetections.flatten().map { it.className }.groupBy { it }
            .map { (cls, list) -> "$cls×${list.size}" }.joinToString()
        fileLogger.i(TAG, "finishRecording: aggregating — totalDetections=$totalDets across $totalFrames frames classes=[$allClasses]")

        // Count DISTINCT physical objects via greedy IoU + distance-gated tracking — NOT frame
        // appearances. The old code summed per-frame detections into `count`, so one chair held
        // in view for N frames was announced as "N chairs". RoomScanAggregator tracks each object
        // across the scan; count = distinct tracks. New inclusion rule: a track must persist
        // >= MIN_FRAMES (replaces the "appeared in >=2 frames OR maxConf>=0.70" heuristic);
        // stairs keep the >=0.60 confidence floor. JSON shape [{class,direction,count}] unchanged.
        val scanFrames = collectedDetections.map { frameDets ->
            frameDets.map { det ->
                ScanDetection(
                    className = det.className,
                    left = det.bbox.left, top = det.bbox.top,
                    right = det.bbox.right, bottom = det.bbox.bottom,
                    distanceMeters = det.distanceMeters,
                    direction = det.direction.toSpokenDirection(),
                    confidence = det.confidence,
                )
            }
        }
        val aggregated = RoomScanAggregator.aggregate(scanFrames)
        fileLogger.i(TAG, "finishRecording: aggregation result — ${aggregated.size} distinct objects: ${aggregated.joinToString { "${it.className}×${it.count} at ${it.direction}" }}")

        val inventory = JSONArray()
        aggregated.forEach { item ->
            inventory.put(JSONObject().apply {
                put("class", item.className)
                put("direction", item.direction)
                put("count", item.count)
            })
        }

        val entity = RoomSnapshotEntity(
            roomName = roomName,
            objectInventoryJson = inventory.toString(),
            createdAt = System.currentTimeMillis(),
            wifiFingerprint = WifiFingerprinter.capture(context),
            scanHeadingDeg = headingDeg
        )

        val id = roomDao.insert(entity)
        feedback.speakSystem("Room saved as $roomName.")
        fileLogger.i(TAG, "Room saved: $roomName id=$id objects=${inventory.length()} heading=${headingDeg}° inventory=${inventory}")

        return entity.copy(id = id)
    }

    /**
     * H1: Pause recording (freeze frame collection) without clearing collectedDetections.
     * Used by SCAN_ROOM mid-flow: scan runs 8s, then we pause and ask for the room name.
     * After the name is captured, finishRecording(name) saves the accumulated data.
     */
    fun pauseRecording() {
        isRecording = false
        fileLogger.i(TAG, "Recording paused — ${collectedDetections.size} frames accumulated")
    }

    /**
     * H2: Return all saved rooms as a one-shot list (not a Flow).
     * Used by LOAD_ROOM to build the audio menu of room options.
     */
    suspend fun getAllRooms(): List<RoomSnapshotEntity> {
        return roomDao.getAll().first()
    }

    /**
     * H2: Load a room by entity reference and announce its layout via TTS.
     * Fix 5: if both scanHeadingDeg and currentHeadingDeg are valid, rotate stored
     * clock-positions to match the user's current entry angle.
     */
    suspend fun loadRoomById(room: RoomSnapshotEntity, currentHeadingDeg: Float = -1f) {
        val canRotate = room.scanHeadingDeg >= 0f && currentHeadingDeg >= 0f
        val inventory = JSONArray(room.objectInventoryJson)
        val description = StringBuilder("Room ${room.roomName} loaded. ")
        for (i in 0 until inventory.length()) {
            val obj = inventory.getJSONObject(i)
            val cls = obj.getString("class").replace("-", " ")
            val count = obj.getInt("count")
            val dir = if (canRotate) {
                rotateDirection(obj.getString("direction"), room.scanHeadingDeg, currentHeadingDeg)
            } else {
                obj.getString("direction")
            }
            description.append(if (count > 1) "$count ${cls}s at $dir. " else "$cls at $dir. ")
        }
        if (inventory.length() == 0) {
            description.append("No objects recorded in this room.")
        }
        feedback.speakSystem(description.toString())
        fileLogger.i(TAG, "Loaded room: ${room.roomName} (rotated=$canRotate)")
    }

    /** Convert stored direction string to relative degrees (-90..+90).
     *  Uses exact equality against SpatialDirection.toSpokenDirection() output. */
    private fun spokenToRelDeg(dir: String): Float = when (dir) {
        "9 o'clock"  -> -90f
        "10 o'clock" -> -60f
        "12 o'clock" ->   0f
        "2 o'clock"  ->  60f
        "3 o'clock"  ->  90f
        else         ->   0f
    }

    /** Rotate a stored clock-position label to account for heading change. */
    private fun rotateDirection(dir: String, scanHeading: Float, currentHeading: Float): String {
        val absAngle = spokenToRelDeg(dir) + scanHeading
        var relAngle = absAngle - currentHeading
        // Normalize to -180..180
        relAngle = ((relAngle + 180f) % 360f + 360f) % 360f - 180f
        val spatialDir = when {
            relAngle <= -75f -> com.nova.assistant.util.SpatialDirection.FAR_LEFT
            relAngle <= -30f -> com.nova.assistant.util.SpatialDirection.LEFT
            relAngle <= 30f  -> com.nova.assistant.util.SpatialDirection.CENTER
            relAngle <= 75f  -> com.nova.assistant.util.SpatialDirection.RIGHT
            else             -> com.nova.assistant.util.SpatialDirection.FAR_RIGHT
        }
        return spatialDir.toSpokenDirection()
    }

    /**
     * AirRoom auto-match: compare live detections against all saved rooms using
     * Jaccard similarity on object class sets + WiFi cosine similarity.
     * Returns the matched room name, or null if no confident match or same room
     * was announced within the last 5 minutes.
     */
    suspend fun checkAutoMatch(currentDetections: List<DetectionWithDepth>): String? {
        val rooms = roomDao.getAll().first()
        if (rooms.isEmpty()) return null

        val liveClasses = currentDetections.map { it.className }.toSet()
        if (liveClasses.size < 2) return null  // not enough context to identify a room

        val currentWifi = WifiFingerprinter.capture(context)
        fileLogger.d(TAG, "checkAutoMatch: liveClasses=${liveClasses} rooms=${rooms.size}")

        data class Candidate(val name: String, val score: Float)

        val best = rooms.mapNotNull { room ->
            val roomClasses = parseInventoryClasses(room.objectInventoryJson)
            if (roomClasses.isEmpty()) return@mapNotNull null

            val jaccard = jaccardSimilarity(liveClasses, roomClasses)
            val commonCount = liveClasses.intersect(roomClasses).size
            val wifiSim = WifiFingerprinter.similarity(currentWifi, room.wifiFingerprint)

            val strongJaccard = jaccard >= 0.55f && commonCount >= 3
            // strongWifi requires a minimum Jaccard floor to prevent false matches
            // in offices/apartments where adjacent rooms share many APs.
            val strongWifi = wifiSim >= 0.80f && jaccard >= 0.20f
            fileLogger.d(TAG, "checkAutoMatch: room=\"${room.roomName}\" jaccard=${"%.2f".format(jaccard)} common=$commonCount wifiSim=${"%.2f".format(wifiSim)} strongJaccard=$strongJaccard strongWifi=$strongWifi")
            if (!strongJaccard && !strongWifi) return@mapNotNull null

            Candidate(room.roomName, jaccard + wifiSim)
        }.maxByOrNull { it.score } ?: run {
            fileLogger.d(TAG, "checkAutoMatch: no room matched thresholds")
            return null
        }

        val now = System.currentTimeMillis()
        if (best.name == lastAnnouncedRoom && now - lastAnnouncedAt < 5 * 60 * 1000L) {
            fileLogger.d(TAG, "checkAutoMatch: suppressed re-announce of \"${best.name}\" (${(now - lastAnnouncedAt) / 1000}s ago)")
            return null
        }

        fileLogger.i(TAG, "checkAutoMatch: MATCHED \"${best.name}\" score=${"%.2f".format(best.score)}")
        lastAnnouncedRoom = best.name
        lastAnnouncedAt = now
        return best.name
    }

    private fun parseInventoryClasses(json: String): Set<String> = try {
        val arr = JSONArray(json)
        buildSet { for (i in 0 until arr.length()) add(arr.getJSONObject(i).getString("class")) }
    } catch (_: Exception) { emptySet() }

    private fun jaccardSimilarity(a: Set<String>, b: Set<String>): Float {
        if (a.isEmpty() || b.isEmpty()) return 0f
        return a.intersect(b).size.toFloat() / (a + b).size.toFloat()
    }

    // R17: Legacy loadRoom(roomName: String) removed — it used the stale "on the $dir"
    // direction format and bypassed loadRoomById(). All callers now use loadRoomById().
}

// ═════════════════════════════════════════════════════════════════
// EMERGENCY SOS
// ═════════════════════════════════════════════════════════════════
@Singleton
class EmergencySOSEngine @Inject constructor(
    private val context: Context,
    private val settingsDao: SettingsDao,
    private val feedback: FeedbackOrchestrator,
    private val fileLogger: FileLogger,
) {
    companion object {
        private const val TAG = "EmergencySOSEngine"
    }

    private var isActive = false

    suspend fun activate(currentLocation: Location?) {
        if (isActive) return
        isActive = true

        val contactNumber = settingsDao.get(SettingsKeys.EMERGENCY_CONTACT) ?: ""
        val contactName = settingsDao.get(SettingsKeys.EMERGENCY_NAME) ?: "your emergency contact"
        val userName = settingsDao.get(SettingsKeys.USER_NAME) ?: "NOVA user"

        if (contactNumber.isBlank()) {
            feedback.speakSystem("No emergency contact set. Please add one in settings.")
            isActive = false
            return
        }

        // Build SMS
        val locationStr = currentLocation?.let {
            "https://maps.google.com/?q=${it.latitude},${it.longitude}"
        } ?: "Location unavailable"

        val message = "NOVA Emergency Alert: $userName needs assistance. " +
                "Location: $locationStr. Time: ${java.text.SimpleDateFormat(
                    "yyyy-MM-dd HH:mm",
                    java.util.Locale.US
                ).format(java.util.Date())}"

        // Send SMS
        try {
            if (ActivityCompat.checkSelfPermission(context, Manifest.permission.SEND_SMS)
                == PackageManager.PERMISSION_GRANTED) {
                val smsManager = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                    context.getSystemService(SmsManager::class.java)
                } else {
                    @Suppress("DEPRECATION") SmsManager.getDefault()
                }
                val parts = smsManager.divideMessage(message)
                smsManager.sendMultipartTextMessage(contactNumber, null, parts, null, null)
                fileLogger.i(TAG, "SOS SMS sent to $contactNumber")

                feedback.speakSystem("Emergency alert sent to $contactName. Help is coming.")
            } else {
                feedback.speakSystem("SMS permission not granted. Cannot send alert.")
                isActive = false
            }
        } catch (e: Exception) {
            fileLogger.e(TAG, "Failed to send SOS SMS: ${e.message}")
            feedback.speakSystem("Failed to send emergency alert.")
            isActive = false
        }
    }

    suspend fun cancel() {
        if (!isActive) return
        isActive = false

        val contactNumber = settingsDao.get(SettingsKeys.EMERGENCY_CONTACT) ?: ""
        val userName = settingsDao.get(SettingsKeys.USER_NAME) ?: "NOVA user"

        if (contactNumber.isNotBlank()) {
            try {
                val smsManager = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                    context.getSystemService(SmsManager::class.java)
                } else {
                    @Suppress("DEPRECATION") SmsManager.getDefault()
                }
                smsManager.sendTextMessage(
                    contactNumber, null,
                    "NOVA: Emergency cancelled by $userName. They are safe.",
                    null, null
                )
            } catch (e: Exception) {
                fileLogger.e(TAG, "Failed to send cancel SMS: ${e.message}")
            }
        }

        feedback.speakSystem("Emergency alert cancelled.")
    }

    fun isEmergencyActive(): Boolean = isActive
}

// ═════════════════════════════════════════════════════════════════
// OCR TEXT READER
// ═════════════════════════════════════════════════════════════════
@Singleton
class TextReaderEngine @Inject constructor(
    private val feedback: FeedbackOrchestrator,
    private val fileLogger: FileLogger,
) {
    companion object {
        private const val TAG = "TextReaderEngine"
        private const val OCR_UPSCALE_MAX_SOURCE_WIDTH = 2000  // skip upscale above this to bound memory
    }

    private val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)

    suspend fun readText(bitmap: Bitmap) {
        fileLogger.i(TAG, "readText: start bitmap=${bitmap.width}x${bitmap.height}")
        // At distance, text occupies very few pixels in-frame — ML Kit's line/word grouping
        // needs a minimum glyph height to link characters into words instead of returning each
        // one as an isolated block (spoken back "letter by letter"). A 1.5x upscale adds no real
        // detail, but does raise glyph height above that internal grouping threshold in the
        // common case. Capped so we never upscale an already-huge frame into a memory problem.
        val ocrBitmap = if (bitmap.width in 1..OCR_UPSCALE_MAX_SOURCE_WIDTH) {
            val ow = (bitmap.width * 1.5f).toInt()
            val oh = (bitmap.height * 1.5f).toInt()
            fileLogger.d(TAG, "readText: upscaling ${bitmap.width}x${bitmap.height} → ${ow}x${oh}")
            Bitmap.createScaledBitmap(bitmap, ow, oh, true)
        } else {
            fileLogger.d(TAG, "readText: no upscale (width=${bitmap.width} > $OCR_UPSCALE_MAX_SOURCE_WIDTH)")
            bitmap
        }
        val image = InputImage.fromBitmap(ocrBitmap, 0)
        val t0 = System.currentTimeMillis()

        suspendCancellableCoroutine<Unit> { continuation ->
            recognizer.process(image)
                .addOnSuccessListener { result ->
                    val elapsed = System.currentTimeMillis() - t0
                    val text = result.text.trim()
                    val blockCount = result.textBlocks.size
                    val wordCount = text.split(Regex("\\s+")).size
                    if (text.isNotEmpty()) {
                        val estimatedMs = estimateSpeechDurationMs(text)
                        fileLogger.i(TAG, "readText: ML Kit SUCCESS in ${elapsed}ms — blocks=$blockCount words=$wordCount chars=${text.length} estimatedSpeechMs=$estimatedMs")
                        // Extend the quiet period to cover the actual result before speaking it —
                        // otherwise a hazard alert queued during recognition could flush right as
                        // this starts. See NovaConstants.OCR_QUIET_MIN_MS/MAX_MS.
                        feedback.beginOcrQuietPeriod(estimatedMs)
                        feedback.speakSystem("Text reads: $text")
                    } else {
                        fileLogger.i(TAG, "readText: ML Kit SUCCESS in ${elapsed}ms — no text (blocks=$blockCount)")
                        feedback.speakSystem("No text detected. Try pointing your phone at the text.")
                    }
                    continuation.resume(Unit)
                }
                .addOnFailureListener { e ->
                    val elapsed = System.currentTimeMillis() - t0
                    fileLogger.e(TAG, "readText: ML Kit FAILED in ${elapsed}ms — ${e.javaClass.simpleName}: ${e.message}")
                    feedback.speakSystem("Text reading failed. Please try again.")
                    continuation.resume(Unit)
                }
        }
        if (ocrBitmap !== bitmap) ocrBitmap.recycle()
    }

    /**
     * Rough spoken-duration estimate (~2.3 words/sec — a deliberately conservative/slow rate).
     * Erring long just means a few extra seconds of alert silence; erring short risks a hazard
     * alert cutting in mid-sentence, which is the failure mode this whole mechanism prevents.
     */
    private fun estimateSpeechDurationMs(text: String): Long {
        val words = text.trim().split(Regex("\\s+")).size
        val estimateMs = (words / 2.3 * 1000).toLong()
        return estimateMs.coerceIn(NovaConstants.OCR_QUIET_MIN_MS, NovaConstants.OCR_QUIET_MAX_MS)
    }
}

// ═════════════════════════════════════════════════════════════════
// EMPTY CHAIR FINDER
// ═════════════════════════════════════════════════════════════════
@Singleton
class EmptyChairFinder @Inject constructor(
    private val feedback: FeedbackOrchestrator,
    private val fileLogger: FileLogger,
) {
    companion object {
        private const val TAG = "EmptyChairFinder"
    }

    fun findChair(detections: List<DetectionWithDepth>) {
        fileLogger.d(TAG, "findChair: scanning ${detections.size} detections — classes=${detections.map { it.className }.distinct()}")

        // Primary: confirmed empty chair
        val emptyChairs = detections.filter { it.className == "empty-chair" }
        if (emptyChairs.isNotEmpty()) {
            val nearest = emptyChairs.minByOrNull { it.distanceMeters }!!
            val dir = nearest.direction.toSpokenDirection()
            val dist = "%.1f".format(nearest.distanceMeters)
            fileLogger.i(TAG, "Empty chair found: $dir, ${dist}m (${emptyChairs.size} in view)")
            feedback.speakSystem("Empty chair at $dir, $dist meters away.")
            return
        }

        // Fallback: generic "chair" class (model sees it but cannot confirm occupancy)
        val anyChairs = detections.filter { it.className == "chair" }
        if (anyChairs.isNotEmpty()) {
            val nearest = anyChairs.minByOrNull { it.distanceMeters }!!
            val dir = nearest.direction.toSpokenDirection()
            val dist = "%.1f".format(nearest.distanceMeters)
            fileLogger.i(TAG, "Chair found (occupancy unknown): $dir, ${dist}m (${anyChairs.size} in view)")
            // R13: "at $dir" matches clock-position convention ("at your 9 o'clock")
            feedback.speakSystem("Chair at $dir, $dist meters. Occupancy unclear.")
            return
        }

        fileLogger.d(TAG, "No chairs found in ${detections.size} detections")
        feedback.speakSystem("No chair visible. Try looking around.")
    }
}

// ═════════════════════════════════════════════════════════════════
// FACE DETECTOR ENGINE
// ═════════════════════════════════════════════════════════════════
@Singleton
class FaceDetectorEngine @Inject constructor(
    private val feedback: FeedbackOrchestrator,
    private val fileLogger: FileLogger,
) {
    companion object {
        private const val TAG = "FaceDetectorEngine"
    }

    // PERFORMANCE_MODE_FAST: skips landmarks/contours — we only need bbox + count.
    private val detector = FaceDetection.getClient(
        FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .setMinFaceSize(0.08f)  // detect faces down to 8% of image width (~arm's length)
            .build()
    )

    suspend fun detectAndAnnounce(bitmap: Bitmap) {
        fileLogger.i(TAG, "detectAndAnnounce: bitmap=${bitmap.width}x${bitmap.height}")
        val image = InputImage.fromBitmap(bitmap, 0)
        val t0 = System.currentTimeMillis()
        suspendCancellableCoroutine<Unit> { continuation ->
            detector.process(image)
                .addOnSuccessListener { faces ->
                    val elapsed = System.currentTimeMillis() - t0
                    fileLogger.i(TAG, "detectAndAnnounce: ${faces.size} faces found in ${elapsed}ms")
                    if (faces.isEmpty()) {
                        feedback.speakSystem("No people detected nearby.")
                    } else {
                        val directions = faces
                            .sortedBy { it.boundingBox.exactCenterX() }
                            .map { face ->
                                val normX = face.boundingBox.exactCenterX() / bitmap.width
                                val dir = com.nova.assistant.util.SpatialDirection
                                    .fromNormalizedX(normX).toSpokenDirection()
                                fileLogger.d(TAG, "face at normX=${"%.2f".format(normX)} → $dir bbox=${face.boundingBox}")
                                dir
                            }
                            .distinct()
                        val countWord = if (faces.size == 1) "1 person" else "${faces.size} people"
                        fileLogger.i(TAG, "detectAndAnnounce: announcing $countWord at ${directions.joinToString()}")
                        feedback.speakSystem("$countWord detected. ${directions.joinToString(", ")}.")
                    }
                    continuation.resume(Unit)
                }
                .addOnFailureListener { e ->
                    val elapsed = System.currentTimeMillis() - t0
                    fileLogger.e(TAG, "detectAndAnnounce: ML Kit FAILED in ${elapsed}ms — ${e.javaClass.simpleName}: ${e.message}")
                    feedback.speakSystem("Face detection failed. Try again.")
                    continuation.resume(Unit)
                }
        }
    }
}

// F4: GMS availability check — returns false on Huawei devices without Google Mobile Services.
// Used by LocationProvider to select FusedLocation (GMS) vs LocationManager (native) path.
private fun isGmsAvailable(context: Context): Boolean {
    return try {
        val result = com.google.android.gms.common.GoogleApiAvailability.getInstance()
            .isGooglePlayServicesAvailable(context)
        result == com.google.android.gms.common.ConnectionResult.SUCCESS
    } catch (_: Exception) { false }
}

// ═════════════════════════════════════════════════════════════════
// LOCATION PROVIDER
// ═════════════════════════════════════════════════════════════════
@Singleton
class LocationProvider @Inject constructor(
    private val context: Context,
    private val fileLogger: FileLogger,
) {
    companion object {
        private const val TAG = "LocationProvider"
    }

    // F4: Detect GMS at construction — Huawei devices without GMS use LocationManager fallback.
    // GMS check is synchronous and cheap (~1ms) so it's safe to do at injection time.
    private val useGms: Boolean = isGmsAvailable(context)
    private val client: FusedLocationProviderClient? =
        if (useGms) LocationServices.getFusedLocationProviderClient(context) else null
    private val locationManager: LocationManager =
        context.getSystemService(Context.LOCATION_SERVICE) as LocationManager

    // Cached live location — updated every 5 seconds by startLocationUpdates().
    // @Volatile so reads from other threads always see the current value.
    @Volatile private var cachedLocation: Location? = null

    private val locationCallback = object : LocationCallback() {
        override fun onLocationResult(result: LocationResult) {
            cachedLocation = result.lastLocation
        }
    }

    // Native android.location.LocationListener for non-GMS devices.
    private val legacyLocationListener = object : LocationListener {
        override fun onLocationChanged(location: Location) {
            cachedLocation = location
        }
        @Deprecated("Deprecated in API 29")
        override fun onStatusChanged(provider: String?, status: Int, extras: Bundle?) {}
        override fun onProviderEnabled(provider: String) {}
        override fun onProviderDisabled(provider: String) {}
    }

    private var isUpdating = false

    fun startLocationUpdates() {
        if (isUpdating) return
        if (ActivityCompat.checkSelfPermission(
                context, Manifest.permission.ACCESS_FINE_LOCATION
            ) != PackageManager.PERMISSION_GRANTED
        ) return
        isUpdating = true

        if (useGms) {
            val request = LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY, 5_000L)
                .setMinUpdateIntervalMillis(3_000L)
                .build()
            try {
                client!!.requestLocationUpdates(request, locationCallback, Looper.getMainLooper())
                fileLogger.i(TAG, "Live location updates started via FusedLocation (GMS)")
            } catch (e: Exception) {
                fileLogger.w(TAG, "GMS requestUpdates failed: ${e.message} — falling back to LocationManager")
                startLegacyUpdates()
            }
        } else {
            fileLogger.i(TAG, "GMS unavailable — using android.location.LocationManager")
            startLegacyUpdates()
        }
    }

    private fun startLegacyUpdates() {
        val provider = when {
            locationManager.isProviderEnabled(LocationManager.GPS_PROVIDER)     -> LocationManager.GPS_PROVIDER
            locationManager.isProviderEnabled(LocationManager.NETWORK_PROVIDER) -> LocationManager.NETWORK_PROVIDER
            else -> {
                fileLogger.w(TAG, "No location providers available — GPS and Network both disabled")
                isUpdating = false
                return
            }
        }
        try {
            locationManager.requestLocationUpdates(provider, 5_000L, 0f, legacyLocationListener, Looper.getMainLooper())
            fileLogger.i(TAG, "Live location updates started via $provider (GMS unavailable)")
        } catch (e: Exception) {
            fileLogger.e(TAG, "LocationManager.requestLocationUpdates failed: ${e.message}")
            isUpdating = false
        }
    }

    fun stopLocationUpdates() {
        if (!isUpdating) return
        isUpdating = false
        if (useGms) {
            client?.removeLocationUpdates(locationCallback)
        } else {
            locationManager.removeUpdates(legacyLocationListener)
        }
    }

    // Returns the live cached value; falls back to one-shot lastLocation if not yet updated.
    suspend fun getCurrentLocation(): Location? {
        cachedLocation?.let { return it }
        if (ActivityCompat.checkSelfPermission(
                context, Manifest.permission.ACCESS_FINE_LOCATION
            ) != PackageManager.PERMISSION_GRANTED
        ) return null

        return if (useGms) {
            try {
                client!!.lastLocation.await().also { cachedLocation = it }
            } catch (e: Exception) { null }
        } else {
            val provider = if (locationManager.isProviderEnabled(LocationManager.GPS_PROVIDER))
                LocationManager.GPS_PROVIDER else LocationManager.NETWORK_PROVIDER
            locationManager.getLastKnownLocation(provider)?.also { cachedLocation = it }
        }
    }
}
