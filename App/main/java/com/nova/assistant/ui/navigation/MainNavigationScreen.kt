package com.nova.assistant.ui.navigation

import android.Manifest
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.IBinder
import android.view.accessibility.AccessibilityManager
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.nova.assistant.data.local.SettingsDao
import com.nova.assistant.data.local.SettingsKeys
import com.nova.assistant.engine.FrameResult
import com.nova.assistant.feedback.FeedbackOrchestrator
import com.nova.assistant.features.EmergencySOSEngine
import com.nova.assistant.features.EmptyChairFinder
import com.nova.assistant.features.LocationProvider
import com.nova.assistant.features.RoomSnapshotEngine
import com.nova.assistant.features.FaceDetectorEngine
import com.nova.assistant.features.TextReaderEngine
import com.nova.assistant.features.voicecommand.NovaCommand
import com.nova.assistant.features.voicecommand.VoiceCommandProcessor
import com.nova.assistant.ml.NovaFrameAnalyzer
import com.nova.assistant.sensors.NovaSensorManager
import com.nova.assistant.service.NovaNavigationService
import com.nova.assistant.util.DistanceZone
import com.nova.assistant.util.NovaConstants
import androidx.compose.foundation.border
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.annotation.DrawableRes
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Settings
import androidx.compose.ui.res.painterResource
import com.nova.assistant.R
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import dagger.hilt.android.lifecycle.HiltViewModel
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlinx.coroutines.withTimeoutOrNull
import javax.inject.Inject
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.foundation.layout.IntrinsicSize
import androidx.compose.material.icons.filled.Warning
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.liveRegion
import com.nova.assistant.engine.NavigationGuidance
import com.nova.assistant.engine.NovaAlert
import com.nova.assistant.engine.SensorData
import com.nova.assistant.util.AlertPriority
import com.nova.assistant.util.FileLogger
import com.nova.assistant.util.SpatialDirection
import com.nova.assistant.util.parseSpokenMenuIndex

/** Navigation destinations that voice commands can trigger. */
sealed class NavEvent {
    object Settings  : NavEvent()
    object Rooms     : NavEvent()
    object Help      : NavEvent()
}

@HiltViewModel
class MainNavigationViewModel @Inject constructor(
    @ApplicationContext private val context: Context,
    val frameAnalyzer: NovaFrameAnalyzer,
    private val sensorManager: NovaSensorManager,
    private val feedback: FeedbackOrchestrator,
    private val voiceProcessor: VoiceCommandProcessor,
    private val roomSnapshot: RoomSnapshotEngine,
    private val emergencySOS: EmergencySOSEngine,
    private val textReader: TextReaderEngine,
    private val faceDetector: FaceDetectorEngine,
    private val emptyChairFinder: EmptyChairFinder,
    private val locationProvider: LocationProvider,
    private val settingsDao: SettingsDao,
    private val fileLogger: FileLogger,
) : ViewModel() {

    val frameResults = frameAnalyzer.frameResults
    val isUsingServer = frameAnalyzer.isUsingServer
    val sensorData = sensorManager.sensorData

    private val _isPaused = androidx.compose.runtime.mutableStateOf(false)
    val isPaused: androidx.compose.runtime.State<Boolean> = _isPaused

    // R1: NNAPI silent-fail watchdog — fires once per session.
    // Tracks consecutive frames with zero detections while not paused.
    // At ~10 FPS, 150 frames ≈ 15 seconds. If NNAPI silently miscomputes on
    // Kirin 710F HiAI, detect() returns emptyList() with no exception thrown.
    // The blind user needs a spoken warning so they know NOVA may not be protecting them.
    private var nnApiWatchdogFired = false

    private val _isEmergency = androidx.compose.runtime.mutableStateOf(false)
    val isEmergency: androidx.compose.runtime.State<Boolean> = _isEmergency

    // Navigation events emitted by voice commands — collected by the Composable via LaunchedEffect.
    // extraBufferCapacity = 1 ensures the event isn't dropped if the collector is briefly busy.
    private val _navEvents = MutableSharedFlow<NavEvent>(extraBufferCapacity = 1)
    val navEvents: SharedFlow<NavEvent> = _navEvents.asSharedFlow()

    /** Toggle emergency state — called from SOSButton and EMERGENCY/CANCEL_EMERGENCY voice commands. */
    fun toggleEmergency() {
        viewModelScope.launch {
            if (_isEmergency.value) {
                emergencySOS.cancel()
                _isEmergency.value = false
            } else {
                _isEmergency.value = true
                val location = locationProvider.getCurrentLocation()
                emergencySOS.activate(location)
            }
        }
    }

    var visionMode by mutableStateOf("partial")
        private set

    init {
        // Sensor/voice lifecycle is now owned by NovaNavigationService.
        // ViewModel only subscribes to the shared flows — no hardware management here.

        // Listen for voice commands
        viewModelScope.launch {
            voiceProcessor.commandFlow.collectLatest { handleCommand(it) }
        }

        // Consume scan requests from SavedRoomsScreen "Scan New Room" button.
        viewModelScope.launch {
            roomSnapshot.pendingScanName.collect { name ->
                if (name == null) return@collect
                roomSnapshot.pendingScanName.value = null  // consume immediately
                roomSnapshot.startRecording()
                kotlinx.coroutines.delay(10_000L)
                feedback.speakSystem("Half way.")
                kotlinx.coroutines.delay(8_000L)
                feedback.speakSystem("Almost done.")
                kotlinx.coroutines.delay(2_000L)
                roomSnapshot.pauseRecording()
                val heading = sensorManager.getCurrentSensorData().azimuthDeg
                roomSnapshot.finishRecording(name, heading)
            }
        }

        // Auto-deliver alerts from frame results
        viewModelScope.launch {
            frameAnalyzer.frameResults.collectLatest { result ->
                result?.topAlert?.let { alert ->
                    // Speech priority hierarchy: EMERGENCY/hazard > voice commands > everything else.
                    // While the user is actively talking, only real hazards (P1/P2) still get
                    // through — a "Stop! stairs" warning must never wait on a "what time is it"
                    // command. Navigation/context alerts (P3/P4) wait until the command finishes.
                    val userTalking = voiceProcessor.isUserSpeaking.value
                    if (!userTalking || alert.priority.level <= AlertPriority.P2_DANGER.level) {
                        feedback.deliverAlert(alert)
                    }
                }

                result?.detections?.let { dets ->
                    roomSnapshot.addFrame(dets)
                }
            }
        }

        // R1: NNAPI silent-fail watchdog.
        // Uses collect (not collectLatest) so the counter persists across frame emissions.
        viewModelScope.launch {
            var zeroFrameCount = 0
            frameAnalyzer.frameResults.collect { result ->
                if (nnApiWatchdogFired || _isPaused.value || result == null) {
                    if (result == null || _isPaused.value) zeroFrameCount = 0
                    return@collect
                }
                if (result.detections.isEmpty()) {
                    zeroFrameCount++
                    // ~10 FPS × 150 frames = 15 seconds of zero detections
                    if (zeroFrameCount >= 150) {
                        nnApiWatchdogFired = true
                        feedback.speakSystem(
                            "Warning: no obstacles detected for 15 seconds. " +
                            "If objects are nearby, obstacle detection may be inactive. " +
                            "Check NOVA settings or restart the app."
                        )
                    }
                } else {
                    zeroFrameCount = 0
                }
            }
        }

        // Load vision mode
        viewModelScope.launch {
            visionMode = settingsDao.get(SettingsKeys.VISION_MODE) ?: "partial"
        }

        // C6: Startup TTS — announce app is ready on cold start.
        // Guard on STARTUP_ANNOUNCED so this only fires once per session.
        viewModelScope.launch {
            val alreadyAnnounced = settingsDao.get(SettingsKeys.STARTUP_ANNOUNCED) == "true"
            if (!alreadyAnnounced) {
                // Brief delay lets TTS engine fully initialize before speaking
                delay(1500L)
                feedback.speakSystem(
                    "NOVA ready. Obstacle detection is active. " +
                    "Say 'what can I say' for available commands."
                )
                settingsDao.set(com.nova.assistant.data.local.SettingsEntity(
                    SettingsKeys.STARTUP_ANNOUNCED, "true"
                ))
            }
        }

        // H5/R10: Alive signal — interval is now configurable via Settings (2/5/10 min or Off).
        // Re-reads the setting before each cycle so changes take effect without a restart.
        // When "off", loops every 60 seconds to detect if the user re-enables the signal.
        viewModelScope.launch {
            while (true) {
                val intervalSetting = settingsDao.get(SettingsKeys.ALIVE_SIGNAL_INTERVAL) ?: "5"
                val intervalMs = when (intervalSetting) {
                    "2"   -> 2 * 60 * 1000L
                    "10"  -> 10 * 60 * 1000L
                    "off" -> 60 * 1000L    // poll every minute in case user re-enables
                    else  -> 5 * 60 * 1000L
                }
                delay(intervalMs)
                if (intervalSetting == "off") continue   // skip announcement; loop to re-check
                val sensor = sensorManager.getCurrentSensorData()
                feedback.speakSystem("NOVA active. Battery ${sensor.batteryPercent} percent.")
            }
        }

        // Proactive low-battery spoken warning — fires once per threshold crossing per session.
        // Keeps quiet above 20%; announces at 20% then again at 10% as the battery drains.
        viewModelScope.launch {
            var announcedAt20 = false
            var announcedAt10 = false
            while (true) {
                delay(60_000L)
                val batt = sensorManager.getCurrentSensorData().batteryPercent
                if (!announcedAt10 && batt <= 10) {
                    feedback.speakSystem("Critical battery: $batt percent. Consider charging soon.")
                    announcedAt10 = true; announcedAt20 = true
                } else if (!announcedAt20 && batt <= 20) {
                    feedback.speakSystem("Low battery: $batt percent.")
                    announcedAt20 = true
                }
            }
        }

        // H7: Apply WARNING_DISTANCE setting at startup and re-apply every 30s so
        // changes in the Settings screen take effect without a full navigation restart.
        // Also re-applied in the RESUME command handler for instant effect after
        // user changes setting and says "resume".
        viewModelScope.launch { applyWarningDistanceSetting() }
        viewModelScope.launch {
            while (true) {
                delay(30_000L)
                applyWarningDistanceSetting()
            }
        }

        // AirRoom auto-match: every 10s check live detections against saved rooms.
        // Speaks "You appear to be in <name>" when Jaccard+WiFi thresholds are met.
        // RoomSnapshotEngine deduplicates — same room won't re-announce within 5 minutes.
        viewModelScope.launch {
            while (true) {
                delay(10_000L)
                if (_isPaused.value) continue
                val dets = frameResults.value?.detections ?: continue
                val matched = roomSnapshot.checkAutoMatch(dets)
                if (matched != null) {
                    feedback.speakSystem("You appear to be in $matched.")
                    fileLogger.i("MainNav", "AutoRoom matched: $matched")
                }
            }
        }

        // TTS echo prevention — stop the mic while TTS is speaking, restart after 500ms buffer.
        // This prevents NOVA's own voice output from being transcribed as voice commands.
        // notifyTtsBusy() is safe to call from any thread; it dispatches to the main Handler.
        viewModelScope.launch {
            feedback.isBusy.collect { busy ->
                voiceProcessor.notifyTtsBusy(busy)
            }
        }

        // M6/R15: Detect TalkBack and disable custom VoiceCommandProcessor if active.
        // Letting both run simultaneously causes dual audio feedback.
        viewModelScope.launch {
            val am = context.getSystemService(Context.ACCESSIBILITY_SERVICE) as AccessibilityManager
            if (am.isTouchExplorationEnabled) {
                voiceProcessor.stopListening()
                android.util.Log.i("MainNav", "TalkBack detected — custom voice commands disabled")
                // R15: Expanded message — tells user what changed AND that detection still runs.
                feedback.speakSystem(
                    "TalkBack detected. Voice commands are disabled to avoid audio conflict. " +
                    "Use standard TalkBack gestures to navigate the app. " +
                    "Obstacle detection is still active and will alert you."
                )
            }
        }
    }

    private fun logAndSpeak(command: NovaCommand, text: String) {
        fileLogger.voiceEvent(command.name, null, text)
        feedback.speakSystem(text)
    }

    /** Step the persisted speech rate slow↔normal↔fast and apply it live. */
    private suspend fun adjustSpeechRate(command: NovaCommand, faster: Boolean) {
        val current = settingsDao.get(SettingsKeys.SPEECH_RATE) ?: "normal"
        val next = when (current) {
            "slow" -> if (faster) "normal" else "slow"   // already slowest
            "fast" -> if (faster) "fast" else "normal"   // already fastest
            else   -> if (faster) "fast" else "slow"     // "normal"
        }
        val rate = when (next) { "slow" -> 0.7f; "fast" -> 1.3f; else -> 1.0f }
        feedback.setSpeechRate(rate)
        settingsDao.set(com.nova.assistant.data.local.SettingsEntity(SettingsKeys.SPEECH_RATE, next))
        // Confirmation is spoken at the new rate, giving immediate audible feedback.
        logAndSpeak(command, "Speech rate $next.")
    }

    private var lastWarningDistanceSetting = ""

    /** Apply WARNING_DISTANCE setting to NovaConstants. Callable at init and on RESUME. */
    private suspend fun applyWarningDistanceSetting() {
        val s = settingsDao.get(SettingsKeys.WARNING_DISTANCE) ?: "2"
        if (s == lastWarningDistanceSetting) return
        lastWarningDistanceSetting = s
        val (danger, warning, caution) = when (s) {
            "1"  -> Triple(1.0f, 2.0f, 3.5f)
            "3"  -> Triple(2.0f, 4.0f, 6.5f)
            else -> Triple(1.5f, 3.0f, 5.0f)  // "2" default
        }
        NovaConstants.DANGER_DISTANCE  = danger
        NovaConstants.WARNING_DISTANCE = warning
        NovaConstants.CAUTION_DISTANCE = caution
        fileLogger.i("MainNav", "WARNING_DISTANCE setting=$s → danger=${danger}m warning=${warning}m caution=${caution}m")
    }

    private suspend fun handleCommand(command: NovaCommand) {
        fileLogger.i("MainNav", "handleCommand: $command paused=${_isPaused.value} hasFrame=${frameResults.value != null}")
        val result = frameResults.value

        when (command) {
            NovaCommand.DESCRIBE_FRONT -> {
                if (_isPaused.value) {
                    logAndSpeak(command, "NOVA is paused. Say resume to get detection results.")
                    return
                }
                // H4: Return top 3 sorted by distance (was single nearest only).
                // Blind users need enough objects to orient themselves.
                val dets = result?.detections
                    ?.sortedBy { it.distanceMeters }
                    ?.take(3)
                    ?: emptyList()
                if (dets.isEmpty()) {
                    logAndSpeak(command, "Nothing detected in front.")
                } else {
                    val summary = dets.joinToString(". ") { det ->
                        "${det.className} at ${det.direction.toSpokenDirection()}, " +
                                "${"%.1f".format(det.distanceMeters)} metres"
                    }
                    logAndSpeak(command, "In front: $summary.")
                }
            }
            NovaCommand.DISTANCE_QUERY -> {
                val nearest = result?.detections?.minByOrNull { it.distanceMeters }
                logAndSpeak(
                    command,
                    nearest?.let {
                        "Nearest object is %.1f meters away.".format(it.distanceMeters)
                    } ?: "Nothing detected nearby."
                )
            }
            NovaCommand.FIND_CHAIR -> {
                fileLogger.voiceEvent(command.name, null, "[EmptyChairFinder]")
                result?.detections?.let { emptyChairFinder.findChair(it) }
            }
            NovaCommand.FIND_DOOR -> {
                val door = result?.detections?.firstOrNull { it.className == "door" }
                logAndSpeak(
                    command,
                    door?.let {
                        "Door on your ${it.direction.toSpokenDirection()}, " +
                                "%.1f meters.".format(it.distanceMeters)
                    } ?: "No door visible."
                )
            }
            NovaCommand.READ_TEXT -> {
                val bmp = frameAnalyzer.snapshotLastBitmap()
                if (bmp != null) {
                    fileLogger.i("MainNav", "READ_TEXT: bitmap=${bmp.width}x${bmp.height}")
                    // Suppress obstacle/hazard alerts (even P1/P2) while OCR runs — the user is
                    // deliberately stationary/close to the object with the text, so YOLO calling
                    // it an "unknown obstacle" at close range must not flush or repeatedly
                    // interrupt the OCR result. TextReaderEngine extends this once the actual
                    // text (and its estimated spoken duration) is known.
                    //
                    // Two changes beyond the original fixed-window approach:
                    // 1. frameAnalyzer.pause() — a close-up screen is exactly the CRITICAL-range
                    //    "unknown obstacle" case; pausing detection outright removes the race
                    //    entirely instead of just gating the alerts it produces, and frees up
                    //    CPU for ML Kit while it's running.
                    // 2. A heartbeat coroutine refreshes the quiet window every 2s while OCR is
                    //    still in flight, so a slow recognition (dense text, close-up = ML Kit
                    //    can take longer than the old fixed 5s guess) can never let the window
                    //    lapse mid-recognition and let hazard alerts resume before the result speaks.
                    feedback.beginOcrQuietPeriod(NovaConstants.OCR_INITIAL_QUIET_MS)
                    frameAnalyzer.pause()
                    logAndSpeak(command, "Reading text...")
                    val heartbeat = viewModelScope.launch {
                        while (true) {
                            delay(2000L)
                            feedback.beginOcrQuietPeriod(NovaConstants.OCR_INITIAL_QUIET_MS)
                        }
                    }
                    try {
                        textReader.readText(bmp)
                    } finally {
                        heartbeat.cancel()
                        frameAnalyzer.resume()
                    }
                } else {
                    logAndSpeak(command, "No camera frame available. Try again.")
                }
            }
            NovaCommand.PEOPLE_NEARBY -> {
                val bmp = frameAnalyzer.snapshotLastBitmap()
                if (bmp != null) {
                    fileLogger.i("MainNav", "PEOPLE_NEARBY: bitmap=${bmp.width}x${bmp.height}")
                    logAndSpeak(command, "Scanning for people.")
                    viewModelScope.launch { faceDetector.detectAndAnnounce(bmp) }
                } else {
                    fileLogger.w("MainNav", "PEOPLE_NEARBY: no camera frame")
                    logAndSpeak(command, "No camera frame available. Try again.")
                }
            }
            NovaCommand.SCAN_ROOM -> {
                // H1/R10: Full SCAN_ROOM flow — scan → pause → voice-capture name → save.
                // Audio checkpoints at 10s and 18s confirm the scan is running.
                // SR-3: rawTextFlow only emits when text doesn't match a command,
                // preventing "help" or "cancel" from being saved as a room name.
                viewModelScope.launch {
                    fileLogger.i("MainNav", "SCAN_ROOM: starting scan")
                    roomSnapshot.startRecording()
                    delay(10_000L)
                    fileLogger.i("MainNav", "SCAN_ROOM: 10s checkpoint — halfway")
                    logAndSpeak(command, "Half way.")
                    delay(8_000L)
                    fileLogger.i("MainNav", "SCAN_ROOM: 18s checkpoint — almost done")
                    logAndSpeak(command, "Almost done.")
                    delay(2_000L)
                    roomSnapshot.pauseRecording()   // freeze collection; keep frames

                    fileLogger.i("MainNav", "SCAN_ROOM: scan complete, waiting for room name")
                    logAndSpeak(command, "Scan complete. What would you like to name this room?")

                    // Wait for TTS to finish before opening mic
                    withTimeoutOrNull(5000L) {
                        feedback.isBusy.first { it }
                        feedback.isBusy.first { !it }
                    }
                    delay(500L)

                    // Suppress hazard alerts (even P1/P2) during the name-capture window. The user
                    // is deliberately stationary and about to SPEAK the room name into the mic; a
                    // "Stop! obstacle ahead" TTS firing here would both talk over their answer and
                    // corrupt ASR of the name. Same rationale + mechanism as READ_TEXT's OCR quiet
                    // period. Heartbeat refreshes the window every 2s so a full 12s wait can't lapse
                    // mid-capture; cancelled the instant the name arrives so hazards resume promptly.
                    feedback.beginOcrQuietPeriod(NovaConstants.OCR_INITIAL_QUIET_MS)
                    val nameQuietHeartbeat = viewModelScope.launch {
                        while (true) {
                            delay(2000L)
                            feedback.beginOcrQuietPeriod(NovaConstants.OCR_INITIAL_QUIET_MS)
                        }
                    }

                    // Capture room name from next spoken utterance (12s window)
                    val roomName = try {
                        withTimeoutOrNull(12_000L) {
                            voiceProcessor.rawTextFlow.first()
                        }?.trim()?.replaceFirstChar { it.uppercase() }
                    } finally {
                        nameQuietHeartbeat.cancel()
                        feedback.endQuietPeriod()   // resume hazards immediately, don't wait out the 5s tail
                    }

                    val finalName = if (roomName.isNullOrBlank()) {
                        val ts = SimpleDateFormat("HH:mm", Locale.US).format(Date())
                        fileLogger.w("MainNav", "SCAN_ROOM: no room name heard (12s timeout) — using fallback \"Room $ts\"")
                        "Room $ts"
                    } else {
                        fileLogger.i("MainNav", "SCAN_ROOM: room name captured=\"$roomName\"")
                        roomName
                    }

                    val heading = sensorManager.getCurrentSensorData().azimuthDeg
                    fileLogger.i("MainNav", "SCAN_ROOM: saving as \"$finalName\" heading=$heading°")
                    roomSnapshot.finishRecording(finalName, heading)
                    // finishRecording() speaks "Room saved as <name>" internally
                }
            }
            NovaCommand.EMERGENCY -> {
                fileLogger.voiceEvent(command.name, null, "[SOS activated]")
                _isEmergency.value = true
                val location = locationProvider.getCurrentLocation()
                emergencySOS.activate(location)
            }
            NovaCommand.CANCEL_EMERGENCY -> {
                emergencySOS.cancel()
                _isEmergency.value = false
                logAndSpeak(command, "Emergency cancelled.")
            }
            NovaCommand.BATTERY -> {
                logAndSpeak(
                    command,
                    "Battery is at ${sensorManager.getCurrentSensorData().batteryPercent} percent."
                )
            }
            NovaCommand.REPEAT -> {
                // Re-speak the last phrase. Call repeatLast() directly (not logAndSpeak)
                // so the cached lastSpokenText isn't overwritten by a log message.
                fileLogger.voiceEvent(command.name, null, "[repeat last]")
                feedback.repeatLast()
            }
            NovaCommand.TIME_QUERY -> {
                val now = SimpleDateFormat("h:mm a", Locale.US).format(Date())
                logAndSpeak(command, "The time is $now.")
            }
            NovaCommand.STATUS -> {
                val batt = sensorManager.getCurrentSensorData().batteryPercent
                val server = if (isUsingServer.value) "Connected to the inference server" else "Using on-device detection"
                val loc = locationProvider.getCurrentLocation()
                val gps = if (loc != null) "Location locked" else "Location not available"
                logAndSpeak(command, "$server. Battery $batt percent. $gps.")
            }
            NovaCommand.PAUSE -> {
                _isPaused.value = true
                frameAnalyzer.pause()
                voiceProcessor.stopListening()  // Pause mic — prevents TTS bleed-over
                logAndSpeak(command, "NOVA paused. Say resume to continue.")
                // L3: Restart mic only after TTS finishes — old fixed delay(2000L)
                // was too short on slow devices and caused mic/TTS overlap.
                viewModelScope.launch {
                    withTimeoutOrNull(6000L) {
                        feedback.isBusy.first { it }
                        feedback.isBusy.first { !it }
                    }
                    delay(500L)
                    voiceProcessor.startListening()
                }
            }
            NovaCommand.RESUME -> {
                _isPaused.value = false
                frameAnalyzer.resume()
                frameAnalyzer.resetSuppression() // Re-announce nearby objects immediately
                feedback.resetAttentionGate()     // Clear "already spoken" memory too
                voiceProcessor.startListening()
                applyWarningDistanceSetting()    // pick up any distance setting changes made while paused
                logAndSpeak(command, "NOVA resumed.")
            }
            NovaCommand.LOAD_ROOM -> {
                // H2/R11/R19: Audio menu — reads room list, waits for spoken number.
                // R11: Prepends cancel instruction so user has an exit path.
                // R19: Uses parseSpokenMenuIndex() to handle 1-20 (was 1-5 only).
                viewModelScope.launch {
                    val rooms = roomSnapshot.getAllRooms()
                    if (rooms.isEmpty()) {
                        logAndSpeak(command, "No saved rooms yet. Say 'scan room' to save one.")
                        return@launch
                    }
                    // R19: Cap menu to 20 rooms (parseSpokenMenuIndex supports up to 20)
                    val displayRooms = rooms.take(20)
                    val menu = displayRooms.mapIndexed { i, r -> "${i + 1}: ${r.roomName}" }
                        .joinToString(". ")
                    // R11: Include cancel instruction before the room list
                    logAndSpeak(command, "Say a number to load a room, or say cancel to exit. $menu")

                    // Wait for TTS to finish
                    withTimeoutOrNull(8000L) {
                        feedback.isBusy.first { it }
                        feedback.isBusy.first { !it }
                    }
                    delay(500L)

                    // Listen for spoken choice (15s window for long menus)
                    val spoken = withTimeoutOrNull(15_000L) {
                        voiceProcessor.rawTextFlow.first()
                    }?.lowercase()?.trim() ?: ""

                    // R11: Handle cancel
                    if (spoken.contains("cancel") || spoken.contains("exit") ||
                        spoken.contains("never mind") || spoken.contains("nevermind")) {
                        logAndSpeak(command, "Cancelled.")
                        return@launch
                    }

                    // R19: Use parseSpokenMenuIndex for 1-20 support
                    val choice = parseSpokenMenuIndex(spoken)
                    val selectedRoom = if (choice != null && choice in 1..displayRooms.size) {
                        displayRooms[choice - 1]
                    } else null

                    if (selectedRoom != null) {
                        val heading = sensorManager.getCurrentSensorData().azimuthDeg
                        roomSnapshot.loadRoomById(selectedRoom, heading)
                    } else {
                        logAndSpeak(command, "I didn't catch a room number. Please try again.")
                    }
                }
            }
            // SETTINGS / OPEN_HELP / GO_HOME / GO_BACK navigation is owned by the
            // app-level collector in NovaApp (MainActivity), which holds navController and
            // therefore works from ANY screen — not just while MainNavigationScreen is composed.
            // These branches are intentional no-ops so the `when` stays exhaustive without
            // double-navigating.
            NovaCommand.SETTINGS, NovaCommand.OPEN_HELP, NovaCommand.OPEN_FINDER,
            NovaCommand.GO_HOME, NovaCommand.GO_BACK -> { /* handled in NovaApp */ }

            NovaCommand.SPEAK_SLOWER -> adjustSpeechRate(command, faster = false)
            NovaCommand.SPEAK_FASTER -> adjustSpeechRate(command, faster = true)
            NovaCommand.UPDATE_CONTACT -> {
                // R3: Voice-guided emergency contact update flow.
                // Captures number then name using rawTextFlow, same pattern as SCAN_ROOM/LOAD_ROOM.
                // Fix: this used to speak an extra "Listening." announcement between the prompt
                // and the capture (unlike SCAN_ROOM/LOAD_ROOM, which go straight from prompt to
                // capture). That forced a second mic mute->unmute cycle ~500ms after the first,
                // which could leave the SpeechRecognizer in a broken state and never actually
                // capture the spoken number/name. Removed to match the working pattern.
                viewModelScope.launch {
                    logAndSpeak(
                        command,
                        "Update emergency contact. Please say the new phone number, digit by digit."
                    )
                    withTimeoutOrNull(5000L) {
                        feedback.isBusy.first { it }
                        feedback.isBusy.first { !it }
                    }
                    delay(500L)

                    // Capture phone number (15s window)
                    val spokenNumber = withTimeoutOrNull(15_000L) {
                        voiceProcessor.rawTextFlow.first()
                    }?.trim()

                    if (spokenNumber.isNullOrBlank()) {
                        logAndSpeak(command, "No number heard. Contact update cancelled.")
                        return@launch
                    }

                    // Convert spoken digits to digit string
                    val digits = com.nova.assistant.util.parseSpokenNumber(spokenNumber)
                    val numberToSave = digits.ifEmpty { spokenNumber }

                    // Read back digit by digit for confirmation
                    val readback = numberToSave.map { it }.joinToString(" ")
                    logAndSpeak(command, "I heard: $readback. Now say the name of this contact.")

                    withTimeoutOrNull(5000L) {
                        feedback.isBusy.first { it }
                        feedback.isBusy.first { !it }
                    }
                    delay(500L)

                    // Capture contact name (10s window)
                    val spokenName = withTimeoutOrNull(10_000L) {
                        voiceProcessor.rawTextFlow.first()
                    }?.trim()?.replaceFirstChar { it.uppercase() }

                    if (spokenName.isNullOrBlank()) {
                        logAndSpeak(command, "No name heard. Contact update cancelled.")
                        return@launch
                    }

                    // Save both values
                    settingsDao.set(com.nova.assistant.data.local.SettingsEntity(
                        SettingsKeys.EMERGENCY_CONTACT, numberToSave
                    ))
                    settingsDao.set(com.nova.assistant.data.local.SettingsEntity(
                        SettingsKeys.EMERGENCY_NAME, spokenName
                    ))
                    logAndSpeak(
                        command,
                        "Emergency contact updated. Number: $readback. Name: $spokenName."
                    )
                }
            }
            NovaCommand.DESCRIBE_ALL -> {
                if (_isPaused.value) {
                    logAndSpeak(command, "NOVA is paused. Say resume to get detection results.")
                    return
                }
                // H4: Sort by distance so nearest objects are named first.
                // Old code used unsorted dets.take(5) which could lead with distant objects.
                val dets = result?.detections.orEmpty()
                    .sortedBy { it.distanceMeters }
                if (dets.isEmpty()) {
                    logAndSpeak(command, "Nothing detected around you.")
                } else {
                    val summary = dets.take(5).joinToString(". ") { det ->
                        "${det.className} at ${det.direction.toSpokenDirection()}, " +
                                "${"%.1f".format(det.distanceMeters)} metres"
                    }
                    logAndSpeak(command, "Around you: $summary.")
                }
            }
            NovaCommand.HELP_COMMANDS -> {
                // H6/R20/R3: Read all available voice commands aloud.
                // "help" alone now triggers this, not SOS. Emergency requires "emergency" or "SOS".
                logAndSpeak(
                    command,
                    "Available commands: " +
                    "What's in front. " +
                    "What's around me. " +
                    "Anyone around — checks for people nearby. " +
                    "How far. " +
                    "Find a chair. " +
                    "Find the door. " +
                    "Read text. " +
                    "Scan room. " +
                    "Load room. " +
                    "Battery. " +
                    "Say again — repeats the last message. " +
                    "What time is it. " +
                    "NOVA status — battery, server, and location. " +
                    "Pause. Resume. " +
                    "Speak slower. Speak faster. " +
                    "Settings. " +
                    "Open finder — switches to the object finder for open-vocabulary search. " +
                    "Go back, or go home, to leave a screen. " +
                    "Open help — opens the help screen. " +
                    "Update contact — voice-update emergency contact. " +
                    "Say emergency or S O S for emergency alert. " +
                    "Cancel emergency to cancel it."
                )
            }
            NovaCommand.UNKNOWN -> {}
        }
    }

    override fun onCleared() {
        // Hardware lifecycle (sensors, voice, camera) is owned by NovaNavigationService —
        // nothing to clean up here. ViewModel only held flow subscriptions.
        super.onCleared()
    }
}

@Composable
fun MainNavigationScreen(
    onSettings: () -> Unit = {},
    onRooms: () -> Unit = {},
    onHelp: () -> Unit = {},
    sosEnabled: Boolean = true,
    viewModel: MainNavigationViewModel = hiltViewModel()
) {
    val frameResult by viewModel.frameResults.collectAsState(initial = null)
    val isPaused by viewModel.isPaused
    val isEmergency by viewModel.isEmergency
    val sensorData by viewModel.sensorData.collectAsState()
    val isUsingServer by viewModel.isUsingServer.collectAsState()

    val context = LocalContext.current

    // Bind to NovaNavigationService so the UI can attach/detach the camera preview surface.
    // The service was already started (and keeps running) even when we unbind.
    var novaService by remember { mutableStateOf<NovaNavigationService?>(null) }

    DisposableEffect(Unit) {
        val conn = object : ServiceConnection {
            override fun onServiceConnected(name: ComponentName?, binder: IBinder?) {
                novaService = (binder as? NovaNavigationService.LocalBinder)?.getService()
            }
            override fun onServiceDisconnected(name: ComponentName?) {
                novaService = null
            }
        }
        val intent = Intent(context, NovaNavigationService::class.java)
        context.startService(intent)          // ensure service survives unbind
        context.bindService(intent, conn, Context.BIND_AUTO_CREATE)
        onDispose {
            novaService?.detachPreviewSurface()
            context.unbindService(conn)       // service keeps running after unbind
        }
    }

    // Collect voice-commanded navigation events and route them to the appropriate screen.
    LaunchedEffect(Unit) {
        viewModel.navEvents.collect { event ->
            when (event) {
                NavEvent.Settings  -> onSettings()
                NavEvent.Rooms     -> onRooms()
                NavEvent.Help      -> onHelp()
            }
        }
    }

    Box(modifier = Modifier.fillMaxSize().background(Color.Black)) {

        if (viewModel.visionMode == "partial") {
            // Camera preview for partial vision users — wired to the service's Preview use case.
            AndroidView(
                factory = { ctx ->
                    PreviewView(ctx).also { previewView ->
                        novaService?.attachPreviewSurface(previewView.surfaceProvider)
                    }
                },
                update = { previewView ->
                    // Re-attach when service connects after the view is already created.
                    novaService?.attachPreviewSurface(previewView.surfaceProvider)
                },
                modifier = Modifier.fillMaxSize()
            )

            // Detection overlay (boxes drawn over camera feed)
            DetectionOverlay(
                frameResult = frameResult,
                modifier = Modifier.fillMaxSize()
            )
        } else {
            // Full-blind mode — no preview rendered; service camera still runs for the analyzer.
            BlindModeIndicator(modifier = Modifier.fillMaxSize())
        }

        // Status overlay (mode badge, object count, battery/thermal warnings)
        StatusOverlay(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp)
                .align(Alignment.TopCenter),
            isPaused = isPaused,
            isUsingServer = isUsingServer,
            detectionCount = frameResult?.detections?.size ?: 0,
            batteryPercent = sensorData.batteryPercent,
            temperatureCelsius = sensorData.temperatureCelsius
        )

        // Navigation icon buttons — top-right corner (always visible)
        Row(
            modifier = Modifier
                .align(Alignment.TopEnd)
                .padding(top = 16.dp, end = 16.dp),
            horizontalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            NavIconButton(icon = Icons.Default.Settings,  desc = "Settings",    onClick = onSettings)
            NavIconButton(icon = Icons.Default.Info,      desc = "Help",        onClick = onHelp)
            NavIconButton(iconRes = R.drawable.ic_bookmark,  desc = "Saved Rooms", onClick = onRooms)
        }

        // Path guidance chip — shown when path is blocked and no priority alert is active
        PathGuidanceChip(
            guidance = frameResult?.freePath?.suggestedDirection,
            isPathClear = frameResult?.freePath?.isPathClear ?: true,
            hasActiveAlert = frameResult?.topAlert != null,
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(bottom = 180.dp)
        )

        // Alert banner — slides up above SOS button when a priority alert exists
        AlertBanner(
            alert = frameResult?.topAlert,
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(bottom = 108.dp)
                .fillMaxWidth()
                .padding(horizontal = 16.dp)
        )

        // SOS button (always visible at bottom). Disabled while Finder is showing — see
        // sosEnabled's doc comment at the MainNavigationScreen call site.
        SOSButton(
            isEmergency = isEmergency,
            enabled = sosEnabled,
            onPress = { viewModel.toggleEmergency() },
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(24.dp)
        )

        // Emergency overlay
        if (isEmergency) {
            EmergencyOverlay(modifier = Modifier.fillMaxSize())
        }
    }
}

@Composable
fun BlindModeIndicator(modifier: Modifier = Modifier) {
    Box(
        modifier = modifier.background(Color.Black),
        contentAlignment = Alignment.Center
    ) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Box(
                modifier = Modifier
                    .size(96.dp)
                    .clip(CircleShape)
                    .background(MaterialTheme.colorScheme.primary.copy(alpha = 0.2f))
                    .border(2.dp, MaterialTheme.colorScheme.primary, CircleShape),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    painter = painterResource(R.drawable.ic_mic),
                    contentDescription = "Microphone — listening for voice commands",
                    modifier = Modifier.size(52.dp),
                    tint = MaterialTheme.colorScheme.primary
                )
            }
            Spacer(Modifier.height(16.dp))
            Text(
                "Listening",
                fontSize = 22.sp,
                color = MaterialTheme.colorScheme.primary,
                fontWeight = FontWeight.Bold
            )
            Text(
                "Say a command",
                fontSize = 14.sp,
                color = Color.White.copy(alpha = 0.6f)
            )
        }
    }
}

/**
 * Small circular icon button used in the nav row.
 * 52dp touch target meets minimum accessibility size.
 */
@Composable
private fun NavIconButton(
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    desc: String,
    onClick: () -> Unit
) {
    Surface(
        onClick = onClick,
        modifier = Modifier
            .size(52.dp)
            .semantics { contentDescription = desc },
        shape = CircleShape,
        color = Color.Black.copy(alpha = 0.65f),
        tonalElevation = 0.dp
    ) {
        Box(contentAlignment = Alignment.Center) {
            Icon(
                imageVector = icon,
                contentDescription = null, // set on Surface above
                tint = Color.White,
                modifier = Modifier.size(24.dp)
            )
        }
    }
}

@Composable
private fun NavIconButton(
    @DrawableRes iconRes: Int,
    desc: String,
    onClick: () -> Unit
) {
    Surface(
        onClick = onClick,
        modifier = Modifier
            .size(52.dp)
            .semantics { contentDescription = desc },
        shape = CircleShape,
        color = Color.Black.copy(alpha = 0.65f),
        tonalElevation = 0.dp
    ) {
        Box(contentAlignment = Alignment.Center) {
            Icon(
                painter = painterResource(id = iconRes),
                contentDescription = null,
                tint = Color.White,
                modifier = Modifier.size(24.dp)
            )
        }
    }
}

@Composable
fun StatusOverlay(
    modifier: Modifier = Modifier,
    isPaused: Boolean,
    isUsingServer: Boolean = false,
    detectionCount: Int = 0,
    batteryPercent: Int = 100,
    temperatureCelsius: Float = 25f
) {
    // A3: contentDescription + liveRegion so TalkBack and NOVA's live announcement
    // both reflect state changes (pause/resume, detection count). LiveRegionMode.Polite
    // prevents the status bar from interrupting a danger alert mid-sentence.
    val statusDesc = buildString {
        append(if (isPaused) "Navigation paused." else "Navigation active.")
        append(if (isUsingServer) " Using server inference." else " Using on-device inference.")
        if (detectionCount > 0) append(" $detectionCount objects detected.")
        if (batteryPercent < 15) append(" Low battery: $batteryPercent percent.")
        if (temperatureCelsius > NovaConstants.THERMAL_HOT) append(" Device overheating.")
    }
    Row(
        modifier = modifier
            .clip(RoundedCornerShape(8.dp))
            .background(Color.Black.copy(alpha = 0.6f))
            .padding(8.dp)
            .semantics {
                contentDescription = statusDesc
                liveRegion = LiveRegionMode.Polite
            },
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            Box(
                modifier = Modifier
                    .size(10.dp)
                    .clip(CircleShape)
                    .background(
                        if (isPaused) Color(0xFFFFAB00) else Color(0xFF4CAF50)
                    )
            )
            Text(
                text = if (isPaused) "Paused" else "Active",
                fontSize = 14.sp,
                color = Color.White
            )
            if (detectionCount > 0) {
                Text(
                    text = "$detectionCount objects",
                    fontSize = 12.sp,
                    color = Color.White.copy(alpha = 0.7f)
                )
            }
            // Inference source badge
            Box(
                modifier = Modifier
                    .clip(RoundedCornerShape(4.dp))
                    .background(
                        if (isUsingServer) Color(0xFF1565C0) else Color(0xFF37474F)
                    )
                    .padding(horizontal = 5.dp, vertical = 2.dp)
            ) {
                Text(
                    text = if (isUsingServer) "SERVER" else "LOCAL",
                    fontSize = 10.sp,
                    color = Color.White,
                    fontWeight = FontWeight.SemiBold,
                    letterSpacing = 0.5.sp
                )
            }
        }
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(4.dp)
        ) {
            if (batteryPercent < 15) {
                Icon(
                    imageVector = Icons.Default.Warning,
                    contentDescription = "Low battery",
                    tint = Color(0xFFFFAB00),
                    modifier = Modifier.size(18.dp)
                )
            }
            if (temperatureCelsius > NovaConstants.THERMAL_HOT) {
                Icon(
                    imageVector = Icons.Default.Warning,
                    contentDescription = "Device overheating",
                    tint = Color.Red,
                    modifier = Modifier.size(18.dp)
                )
            }
            Text(
                "NOVA",
                fontSize = 16.sp,
                color = MaterialTheme.colorScheme.primary,
                fontWeight = FontWeight.Bold
            )
        }
    }
}

@Composable
fun SOSButton(
    isEmergency: Boolean,
    onPress: () -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
) {
    Button(
        onClick = onPress,
        enabled = enabled,
        modifier = modifier
            .fillMaxWidth(0.85f)
            .height(72.dp)
            .semantics {
                contentDescription = if (isEmergency)
                    "Cancel emergency SOS alert"
                else
                    "Activate SOS — sends emergency alert to your contact"
            },
        colors = ButtonDefaults.buttonColors(
            containerColor = if (isEmergency) Color.Red else Color(0xFFB71C1C)
        ),
        shape = RoundedCornerShape(16.dp)
    ) {
        // A5: Emoji removed — contentDescription on the Button handles TalkBack.
        Text(
            "SOS",
            fontSize = 28.sp,
            fontWeight = FontWeight.Bold,
            color = Color.White
        )
    }
}

@Composable
fun EmergencyOverlay(modifier: Modifier = Modifier) {
    Box(
        modifier = modifier.background(Color.Red.copy(alpha = 0.4f)),
        contentAlignment = Alignment.Center
    ) {
        // A5: Plain text — emoji rendering is unreliable on Huawei TalkBack.
        // contentDescription ensures consistent screen-reader announcement.
        Text(
            "EMERGENCY ACTIVE",
            fontSize = 32.sp,
            fontWeight = FontWeight.Bold,
            color = Color.White,
            textAlign = TextAlign.Center,
            modifier = Modifier.semantics { contentDescription = "Emergency SOS is active" }
        )
    }
}

@Composable
fun AlertBanner(
    alert: NovaAlert?,
    modifier: Modifier = Modifier
) {
    AnimatedVisibility(
        visible = alert != null && !(alert.isPathGuidance),
        enter = slideInVertically { it } + fadeIn(),
        exit = slideOutVertically { it } + fadeOut()
    ) {
        if (alert != null) {
            val accentColor = when (alert.priority) {
                AlertPriority.P1_EMERGENCY -> Color.Red
                AlertPriority.P2_DANGER    -> Color(0xFFFF6B00)
                AlertPriority.P3_DIRECTION -> Color(0xFFFFD60A)
                else                       -> Color(0xFF64D2FF)
            }
            val arrowSymbol = when (alert.direction) {
                SpatialDirection.FAR_LEFT, SpatialDirection.LEFT   -> "←"
                SpatialDirection.FAR_RIGHT, SpatialDirection.RIGHT -> "→"
                SpatialDirection.CENTER                            -> "↑"
            }
            val arrowDesc = when (alert.direction) {
                SpatialDirection.FAR_LEFT  -> "far left"
                SpatialDirection.LEFT      -> "left"
                SpatialDirection.CENTER    -> "ahead"
                SpatialDirection.RIGHT     -> "right"
                SpatialDirection.FAR_RIGHT -> "far right"
            }
            // M2: Full contentDescription for TalkBack + liveRegion so new alerts
            // are announced automatically without user focus interaction.
            val bannerDesc = "${alert.sourceClassName.ifEmpty { "Obstacle" }} " +
                    "at $arrowDesc, ${"%.1f".format(alert.distanceMeters)} metres. " +
                    "Priority: ${alert.priority.name.replace("_", " ")}."
            Row(
                modifier = modifier
                    .background(Color.Black.copy(alpha = 0.82f), RoundedCornerShape(12.dp))
                    .height(IntrinsicSize.Min)
                    .semantics {
                        contentDescription = bannerDesc
                        liveRegion = LiveRegionMode.Assertive
                    },
                verticalAlignment = Alignment.CenterVertically
            ) {
                Box(
                    modifier = Modifier
                        .width(4.dp)
                        .fillMaxHeight()
                        .background(
                            accentColor,
                            RoundedCornerShape(topStart = 12.dp, bottomStart = 12.dp)
                        )
                )
                Column(
                    modifier = Modifier
                        .weight(1f)
                        .padding(horizontal = 12.dp, vertical = 8.dp)
                ) {
                    Text(
                        text = alert.sourceClassName.ifEmpty { "Obstacle" }
                            .replaceFirstChar { it.uppercase() },
                        fontSize = 15.sp,
                        fontWeight = FontWeight.Bold,
                        color = Color.White
                    )
                    Text(
                        text = "${alert.direction.toSpokenDirection()}  ${"%.1f".format(alert.distanceMeters)}m",
                        fontSize = 13.sp,
                        color = Color.White.copy(alpha = 0.75f)
                    )
                }
                Text(
                    text = arrowSymbol,
                    fontSize = 22.sp,
                    color = accentColor,
                    // M2: contentDescription provides direction to TalkBack (arrow glyph is not spoken)
                    modifier = Modifier
                        .padding(end = 14.dp)
                        .semantics { contentDescription = "Direction: $arrowDesc" }
                )
            }
        }
    }
}

@Composable
fun PathGuidanceChip(
    guidance: NavigationGuidance?,
    isPathClear: Boolean,
    hasActiveAlert: Boolean,
    modifier: Modifier = Modifier
) {
    val text = when (guidance) {
        NavigationGuidance.MOVE_LEFT        -> "← Move left"
        NavigationGuidance.MOVE_RIGHT       -> "Move right →"
        NavigationGuidance.STOP_ALL_BLOCKED -> "Stop — path blocked"
        NavigationGuidance.SLOW_DOWN        -> "Slow down"
        else                                -> null
    }

    AnimatedVisibility(
        visible = text != null && !isPathClear && !hasActiveAlert,
        enter = fadeIn(),
        exit = fadeOut(),
        modifier = modifier
    ) {
        if (text != null) {
            Box(
                contentAlignment = Alignment.Center,
                modifier = Modifier
                    .background(Color.Black.copy(alpha = 0.75f), RoundedCornerShape(50))
                    .padding(horizontal = 16.dp, vertical = 8.dp)
                    // M2: liveRegion.Polite so TalkBack announces direction guidance
                    // without interrupting higher-priority speech.
                    .semantics {
                        contentDescription = text
                        liveRegion = LiveRegionMode.Polite
                    }
            ) {
                Text(
                    text = text,
                    fontSize = 13.sp,
                    color = Color.White
                )
            }
        }
    }
}

@Composable
fun DetectionOverlay(
    frameResult: FrameResult?,
    modifier: Modifier = Modifier
) {
    val textMeasurer = rememberTextMeasurer()

    androidx.compose.foundation.Canvas(modifier = modifier) {
        frameResult?.detections?.forEach { det ->
            // C2: Added CAUTION tier — cyan/teal colour so partial-vision users
            // can distinguish danger (red) → warning (orange) → caution (teal) → far (dim yellow).
            val boxColor = when (det.distanceZone) {
                DistanceZone.DANGER  -> Color.Red
                DistanceZone.WARNING -> Color(0xFFFF9800)
                DistanceZone.CAUTION -> Color(0xFF00BCD4)   // teal
                DistanceZone.FAR     -> Color.Yellow
            }
            val strokeWidth = when (det.distanceZone) {
                DistanceZone.DANGER  -> 6f
                DistanceZone.WARNING -> 4f
                DistanceZone.CAUTION -> 3f
                DistanceZone.FAR     -> 2f
            }
            val bgColor = when (det.distanceZone) {
                DistanceZone.DANGER  -> Color(red = 200/255f, green = 0f,       blue = 0f,   alpha = 0.7f)
                DistanceZone.WARNING -> Color(red = 200/255f, green = 100/255f, blue = 0f,   alpha = 0.7f)
                DistanceZone.CAUTION -> Color(red = 0f,       green = 150/255f, blue = 180/255f, alpha = 0.65f)
                DistanceZone.FAR     -> Color(red = 40/255f,  green = 40/255f,  blue = 0f,   alpha = 0.55f)
            }

            val left   = det.bbox.left   * size.width
            val top    = det.bbox.top    * size.height
            val width  = det.bbox.width()  * size.width
            val height = det.bbox.height() * size.height

            drawRect(
                color = boxColor,
                topLeft = Offset(left, top),
                size = Size(width, height),
                style = androidx.compose.ui.graphics.drawscope.Stroke(width = strokeWidth)
            )

            val label = "${det.className}  ${"%.1f".format(det.distanceMeters)}m"
            val measured = textMeasurer.measure(
                text = label,
                style = TextStyle(
                    color = Color.White,
                    fontSize = 11.sp,
                    fontWeight = FontWeight.Bold
                )
            )
            val tw = measured.size.width.toFloat()
            val th = measured.size.height.toFloat()
            val lx = left.coerceAtMost(size.width - tw - 8f).coerceAtLeast(0f)
            val ly = (top - th - 4f).coerceAtLeast(0f)

            drawRoundRect(
                color = bgColor,
                topLeft = Offset(lx - 4f, ly - 2f),
                size = Size(tw + 8f, th + 4f),
                cornerRadius = CornerRadius(6f, 6f)
            )
            drawText(
                textLayoutResult = measured,
                topLeft = Offset(lx, ly)
            )
        }
    }
}
