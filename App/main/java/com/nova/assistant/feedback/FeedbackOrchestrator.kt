package com.nova.assistant.feedback

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import kotlin.math.PI
import kotlin.math.sin
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import java.io.File
import java.util.concurrent.ConcurrentHashMap
import com.nova.assistant.engine.HapticPattern
import com.nova.assistant.engine.NovaAlert
import com.nova.assistant.util.AlertPriority
import com.nova.assistant.util.FileLogger
import com.nova.assistant.util.NovaConstants
import com.nova.assistant.util.SpatialDirection
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import java.util.Locale
import java.util.concurrent.atomic.AtomicLong
import javax.inject.Inject
import javax.inject.Singleton

/**
 * NOVA Feedback Orchestrator — Combines TTS audio + haptic vibration.
 *
 * Handles:
 * - Text-to-Speech with stereo panning (left/right ear)
 * - Haptic patterns (6 distinct vibration patterns)
 * - Queue management (never overlaps speech)
 * - Speech rate adjustment (slow/normal/fast)
 */
@Singleton
class FeedbackOrchestrator @Inject constructor(
    private val context: Context,
    private val fileLogger: FileLogger,
) {
    companion object {
        private const val TAG = "FeedbackOrchestrator"
        // C1: minimum gap between successive alert deliveries.
        // Prevents chattering when the inference loop fires at 10 FPS.
        private const val MIN_ALERT_INTERVAL_MS = 1500L
        // Attention gate: if a key (class+direction) hasn't reached deliverAlert at all in this
        // long, treat its next appearance as a fresh encounter rather than "still the same object" —
        // otherwise an object that leaves and later returns would stay silently suppressed forever.
        private const val PRESENCE_TIMEOUT_MS = 8000L
        // Watchdog: forces isSpeaking/currentSpeakingPriority back to idle if no completion
        // signal (onDone/onError/marker callback) arrives within [expected duration + this
        // margin]. Without this, a completion signal that's silently lost (see the AudioTrack
        // marker-listener reliability note on playCachedPhrase) sticks isSpeaking=true forever,
        // and every subsequent alert — including P1 hazards — gets silently dropped by the
        // priority gate in deliverAlert() until an unrelated speakSystem() call happens to
        // reset the state via its own onDone. Field log 2026-07-16/17: one cached-phrase
        // playback with a lost marker callback blocked ~90s and 20+ hazard alerts (including at
        // least one "door" alert) before a Settings readout incidentally cleared it.
        private const val WATCHDOG_MARGIN_MS = 2500L
        // Backstop for the live tts.speak() path, whose real duration isn't known in advance
        // (varies with text length/rate) — generous enough to never fire during normal speech,
        // tight enough that a genuinely hung engine call recovers well within a session.
        private const val LIVE_TTS_WATCHDOG_MS = 15000L
    }

    private var tts: TextToSpeech? = null
    // Dedicated second engine for background phrase-cache rendering (synthesizeToFile).
    // MUST be a separate instance from `tts` — see cacheTts init comment below.
    private var cacheTts: TextToSpeech? = null
    private var vibrator: Vibrator? = null
    private var audioManager: AudioManager? = null
    private var isInitialized = false
    @Volatile private var isSpeaking = false

    // Watchdog plumbing — see WATCHDOG_MARGIN_MS above. Own SupervisorJob so one failed
    // watchdog coroutine can't cancel the others; tied to this singleton's lifetime.
    private val watchdogScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private val speakingGeneration = java.util.concurrent.atomic.AtomicInteger(0)

    // C1+C5: priority tracking for interrupt decisions.
    // @Volatile ensures the inference thread and main thread read the same value.
    @Volatile private var currentSpeakingPriority: Int = Int.MAX_VALUE
    private val lastAlertDeliveryMs = AtomicLong(0L)

    // Listening window: timestamp until which non-hazard (P3/P4) alerts are held back,
    // set after every TTS completion so the mic gets uncontested priority for a follow-up command.
    private val quietUntilMs = AtomicLong(0L)

    // OCR quiet period: timestamp until which ALL alerts (even P1/P2) are held back. Set by
    // TextReaderEngine while "read text" is in progress — see beginOcrQuietPeriod().
    @Volatile private var ocrQuietUntilMs = 0L

    // Finder tab active: suppresses ALL alerts (even P1/P2) for as long as the Finder tab is
    // showing — unlike ocrQuietUntilMs this isn't a bounded timer, it's tied directly to tab
    // visibility (see MainActivity's selectedTab LaunchedEffect, which sets this in lockstep
    // with suspendNavPipeline). The nav camera pipeline is already suspended while Finder is
    // open so no *new* detections should occur, but a straggling in-flight alert from the
    // frame right before the switch — or a residual queued utterance — must not speak over
    // the user's Finder voice search.
    @Volatile private var finderActive = false

    // Attention gate (per "className_direction" key) — recorded ONLY at the point an alert is
    // actually committed to speech (after listening-window/rate-limit/priority gates), so a
    // real escalation that gets dropped downstream is never mistaken for "already announced".
    private val lastSeenMs = mutableMapOf<String, Long>()
    private val lastAnnouncedLevel = mutableMapOf<String, Int>()
    private val lastAnnouncedKeyText = mutableMapOf<String, String>()

    private val _isBusy = MutableStateFlow(false)
    val isBusy: StateFlow<Boolean> = _isBusy

    // Last spoken phrase (alert or system) — replayed by the "say again" voice command.
    @Volatile private var lastSpokenText: String = ""

    // Settings
    private var speechRate = 1.0f  // 0.5 = slow, 1.0 = normal, 1.5 = fast
    private var vibrationEnabled = true
    private var vibrationIntensity = 1.0f // 0.5 = gentle, 1.0 = strong

    // Earcon AudioTracks — pre-generated at startup, replayed for each P1/P2 alert.
    // P1: ascending 440→880 Hz chirp, 60ms. P2: double 880 Hz pulse, 65ms.
    // Playing concurrently with TTS synthesis gives user ~60–80ms head-start on
    // auditory awareness before speech begins (standard screen-reader design pattern).
    private var earconP1: AudioTrack? = null
    private var earconP2: AudioTrack? = null
    private var earconListening: AudioTrack? = null

    // Phrase cache: repeated alert text (e.g. "Stop! obstacle ahead." — the same P1 message
    // fires dozens of times per session per the server logs) skips ~100-400ms of TextToSpeech
    // synthesis-engine startup latency on repeat by replaying pre-rendered PCM via AudioTrack,
    // the same zero-latency mechanism already used for earcons. Keyed by exact spokenText;
    // dynamic phrases (varying step counts, object names) simply never hit and fall back to
    // the normal tts.speak() path below — no correctness risk, pure latency optimization.
    //
    // Sample rate is captured per-phrase from the actual WAV header at render time (see
    // loadCachedPhrase) — NOT hardcoded. The old hardcoded 22050Hz playback rate didn't match
    // the TTS engine's real output rate on several devices, so cached phrases played back
    // pitch-shifted ("chipmunked") relative to live speech — reported as "another high pitched
    // voice appearing alongside NOVA's voice."
    private data class CachedPhrase(val samples: ShortArray, val sampleRate: Int)
    private val phraseCache = ConcurrentHashMap<String, CachedPhrase>()
    private val phraseCachePending = ConcurrentHashMap.newKeySet<String>()
    private val CACHE_PREFIX = "ttscache_"
    private val MAX_CACHED_PHRASES = 60  // bounds memory — dynamic phrases (varying step counts) never all fit, and don't need to

    // Tracks real-time movement magnitude (m/s²) fed from NovaSensorManager via service layer.
    // When high (running/fast walking), speech rate scales up 1.25× to deliver alerts sooner.
    @Volatile var movementMagnitude: Float = 0f

    fun initialize() {
        if (isInitialized) return
        audioManager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager

        // Initialize vibrator
        vibrator = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            val vm = context.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager
            vm.defaultVibrator
        } else {
            @Suppress("DEPRECATION")
            context.getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
        }

        // Initialize TTS — live speech engine only. Background phrase-cache rendering (see
        // cacheTts below) MUST NOT share this engine's utterance queue: TextToSpeech.speak()
        // and synthesizeToFile() both queue on the same underlying engine instance, so a burst
        // of background cache-render requests (one per unique alert phrase, e.g. every distinct
        // "person ahead. X metres." variant) could sit ahead of a real-time alert in the queue —
        // observed as long silences followed by a sudden backlog of alerts firing at once.
        tts = TextToSpeech(context) { status ->
            if (status == TextToSpeech.SUCCESS) {
                tts?.language = Locale.US
                tts?.setSpeechRate(speechRate)

                tts?.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                    override fun onStart(utteranceId: String?) {
                        // isSpeaking/currentSpeakingPriority are already set by beginSpeaking()
                        // at the point speak() was called — this is now logging-only.
                        fileLogger.d(TAG, "TTS onStart uid=$utteranceId")
                    }
                    override fun onDone(utteranceId: String?) {
                        endSpeaking()  // C5: reset so next alert can speak
                        fileLogger.d(TAG, "TTS onDone uid=$utteranceId")
                    }
                    @Suppress("OVERRIDE_DEPRECATION")
                    override fun onError(utteranceId: String?) {
                        endSpeaking()  // C5: same reset on error
                        fileLogger.w(TAG, "TTS onError uid=$utteranceId")
                    }
                })

                isInitialized = true
                initEarcons()
                fileLogger.i(TAG, "TTS initialized ✓")
            } else {
                fileLogger.e(TAG, "TTS initialization failed: status=$status")
            }
        }

        // Second, independent engine used ONLY for background synthesizeToFile() cache renders.
        // Keeping it separate from `tts` guarantees cache warm-up can never delay a live alert.
        cacheTts = TextToSpeech(context) { status ->
            if (status == TextToSpeech.SUCCESS) {
                cacheTts?.language = Locale.US
                cacheTts?.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                    override fun onStart(utteranceId: String?) {}
                    override fun onDone(utteranceId: String?) {
                        loadCachedPhrase(utteranceId.orEmpty().removePrefix(CACHE_PREFIX))
                    }
                    @Suppress("OVERRIDE_DEPRECATION")
                    override fun onError(utteranceId: String?) {
                        phraseCachePending.remove(utteranceId.orEmpty().removePrefix(CACHE_PREFIX))
                    }
                })
            } else {
                fileLogger.w(TAG, "Cache TTS engine init failed: status=$status — phrase caching disabled")
            }
        }
    }

    /**
     * Deliver a NOVA alert — speaks text and vibrates simultaneously.
     *
     * C1: Priority-aware queue mode — only P1/P2 flush the TTS queue.
     * C5: Fixed priority inversion — old code dropped P2 while P4 played.
     *     New check: incoming alert interrupts ONLY if it is strictly more
     *     urgent (lower level) than what is currently speaking.
     */
    fun deliverAlert(alert: NovaAlert) {
        if (!isInitialized) return
        val now = System.currentTimeMillis()
        val key = "${alert.sourceClassName}_${alert.direction}"

        // Record presence unconditionally, before any drop-gate below, so the attention gate's
        // "has this key gone quiet for a while" check reflects reality even on frames that get
        // dropped for other reasons (listening window, rate-limit, priority).
        val gapMs = now - (lastSeenMs[key] ?: 0L)
        if (gapMs > PRESENCE_TIMEOUT_MS) {
            // Hasn't reached us in a while — treat the next state as a fresh encounter.
            lastAnnouncedLevel.remove(key)
            lastAnnouncedKeyText.remove(key)
        }
        lastSeenMs[key] = now

        // OCR quiet period: suppresses even P1/P2 — the user asked to read text and is
        // deliberately stationary/close to the object; a hazard alert about that very object
        // would otherwise flush and drown out (or endlessly retrigger over) the OCR result.
        if (now < ocrQuietUntilMs) {
            fileLogger.d(TAG, "Alert dropped [ocr-quiet] ${alert.priority} \"${alert.spokenText.take(40)}\"")
            return
        }

        // Finder tab active: same unconditional suppression as the OCR quiet period above.
        if (finderActive) {
            fileLogger.d(TAG, "Alert dropped [finder-active] ${alert.priority} \"${alert.spokenText.take(40)}\"")
            return
        }

        // Listening window: hold back non-hazard alerts right after NOVA finishes speaking,
        // so the mic gets uncontested priority for a follow-up voice command. P1/P2 bypass this.
        if (alert.priority.level > AlertPriority.P2_DANGER.level && now < quietUntilMs.get()) {
            fileLogger.d(TAG, "Alert dropped [listening-window] ${alert.priority} \"${alert.spokenText.take(40)}\"")
            return
        }

        // C1: Rate-limit low-priority alerts.
        if (alert.priority.level >= AlertPriority.P3_DIRECTION.level &&
            now - lastAlertDeliveryMs.get() < MIN_ALERT_INTERVAL_MS) {
            fileLogger.d(TAG, "Alert dropped [rate-limit] ${alert.priority} \"${alert.spokenText.take(40)}\"")
            return
        }

        // C5: Only interrupt if strictly higher priority.
        if (isSpeaking && alert.priority.level >= currentSpeakingPriority) {
            fileLogger.d(TAG, "Alert dropped [priority] incoming=${alert.priority.level} speaking=$currentSpeakingPriority")
            return
        }

        // Attention gate: hazard tiers (P1/P2) always speak — a real danger is worth repeating
        // on its existing cooldown. Non-hazard tiers (P3/P4 objects) and path guidance only
        // speak on an actual state change, so NOVA narrates transitions ("now blocked", "got
        // closer") instead of re-describing scenery that hasn't changed since it was last said.
        // Recorded HERE (not upstream in SuppressionFilter) because this is the only place that
        // knows the alert actually got spoken — recording it earlier would let a real escalation
        // get silently marked "announced" even when this function drops it above.
        val priorLevel = lastAnnouncedLevel[key]
        val stateChanged = when {
            alert.isPathGuidance -> lastAnnouncedKeyText[key] != alert.spokenText
            alert.priority.level <= AlertPriority.P2_DANGER.level -> true
            else -> priorLevel == null || alert.priority.level < priorLevel
        }
        if (!stateChanged) {
            fileLogger.d(TAG, "Alert dropped [no-state-change] ${alert.priority} \"${alert.spokenText.take(40)}\"")
            return
        }
        lastAnnouncedLevel[key] = alert.priority.level
        lastAnnouncedKeyText[key] = alert.spokenText

        fileLogger.i(TAG, "Alert [${alert.priority}] \"${alert.spokenText.take(60)}\" dir=${alert.direction}")
        fileLogger.voiceEvent("TTS", null, alert.spokenText)

        lastAlertDeliveryMs.set(now)
        playEarcon(alert.priority)
        speakWithDirection(alert.spokenText, alert.direction, alert.priority)
        if (vibrationEnabled) vibratePattern(alert.hapticPattern)
    }

    /**
     * Speak text with stereo panning based on object direction.
     * Object on left → louder in left ear, quieter in right.
     *
     * C1: queueMode is now determined by priority:
     *   P1 → QUEUE_FLUSH (always interrupt — life-safety)
     *   P2 → QUEUE_FLUSH only if currently speaking something less urgent
     *   P3/P4 → QUEUE_ADD (append, never interrupt)
     */
    private fun speakWithDirection(text: String, direction: SpatialDirection, priority: AlertPriority) {
        val ttsEngine = tts ?: return

        val pan = when (direction) {
            SpatialDirection.FAR_LEFT  -> -0.8f
            SpatialDirection.LEFT      -> -0.5f
            SpatialDirection.CENTER    -> 0f
            SpatialDirection.RIGHT     -> 0.5f
            SpatialDirection.FAR_RIGHT -> 0.8f
        }

        currentSpeakingPriority = priority.level
        lastSpokenText = text

        // Cache hit: replay pre-rendered PCM instantly (no synthesis-engine startup latency) —
        // only at the default speech rate, since cached audio can't be re-timed on playback.
        val cached = phraseCache[text]
        if (cached != null && movementMagnitude <= 12f) {
            playCachedPhrase(cached, pan)
            return
        }

        val queueMode = when {
            priority == AlertPriority.P1_EMERGENCY -> TextToSpeech.QUEUE_FLUSH
            priority == AlertPriority.P2_DANGER &&
                currentSpeakingPriority > AlertPriority.P2_DANGER.level -> TextToSpeech.QUEUE_FLUSH
            else -> TextToSpeech.QUEUE_ADD
        }

        // Adaptive rate: scale up 1.25× when user is moving fast (running/brisk walking).
        // Reduces time-to-information at the cost of slightly reduced intelligibility.
        val adaptiveRate = if (movementMagnitude > 12f) minOf(speechRate * 1.25f, 2.0f) else speechRate
        ttsEngine.setSpeechRate(adaptiveRate)

        val params = android.os.Bundle().apply {
            putFloat(TextToSpeech.Engine.KEY_PARAM_PAN, pan)
            putFloat(TextToSpeech.Engine.KEY_PARAM_VOLUME, 1.0f)
        }
        beginSpeaking(LIVE_TTS_WATCHDOG_MS)
        ttsEngine.speak(text, queueMode, params, "nova_${System.currentTimeMillis()}")

        // Prime the cache for next time — background render, never blocks this utterance.
        maybeCachePhrase(text)
    }

    /** Play pre-rendered PCM via AudioTrack with stereo panning — same near-zero-latency
     *  mechanism as earcons. Bypasses the TTS engine entirely on a cache hit.
     *  Plays back at the phrase's own recorded sample rate (see loadCachedPhrase) — using a
     *  mismatched fixed rate here previously pitch-shifted cached phrases relative to live
     *  speech, audible as a second, higher-pitched voice. */
    private fun playCachedPhrase(phrase: CachedPhrase, pan: Float) {
        val samples = phrase.samples
        val sampleRate = phrase.sampleRate
        try {
            val minBuf = AudioTrack.getMinBufferSize(
                sampleRate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT
            )
            val track = AudioTrack(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_ASSISTANCE_ACCESSIBILITY)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build(),
                AudioFormat.Builder()
                    .setSampleRate(sampleRate)
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .build(),
                maxOf(samples.size * 2, minBuf),
                AudioTrack.MODE_STATIC,
                AudioManager.AUDIO_SESSION_ID_GENERATE
            )
            track.write(samples, 0, samples.size)
            @Suppress("DEPRECATION")
            track.setStereoVolume(
                (1f - pan.coerceIn(0f, 1f)).coerceIn(0f, 1f),
                (1f + pan.coerceIn(-1f, 0f)).coerceIn(0f, 1f)
            )
            // AudioTrack's marker-position callback is not fully reliable in practice (observed
            // in the field: onMarkerReached silently never fired for a cached "computer" alert,
            // leaving isSpeaking stuck true for ~90s and dropping every alert — including a
            // "door" alert — until an unrelated speakSystem() call happened to reset state via
            // the live engine's onDone). beginSpeaking()'s watchdog is the real safety net here;
            // the marker listener remains the fast/normal path when it does fire.
            val durationMs = (samples.size.toLong() * 1000L) / sampleRate
            beginSpeaking(durationMs + WATCHDOG_MARGIN_MS)
            track.setNotificationMarkerPosition(samples.size)
            track.setPlaybackPositionUpdateListener(object : AudioTrack.OnPlaybackPositionUpdateListener {
                override fun onMarkerReached(t: AudioTrack?) {
                    endSpeaking()
                    t?.release()
                }
                override fun onPeriodicNotification(t: AudioTrack?) {}
            })
            track.play()
        } catch (e: Exception) {
            Log.w(TAG, "Cached phrase playback failed: ${e.message}")
            isSpeaking = false
            _isBusy.value = false
        }
    }

    /** Synthesize [text] to a temp WAV once in the background on the dedicated cache engine
     *  (never the live `tts` engine — see cacheTts init); onDone loads it into phraseCache. */
    private fun maybeCachePhrase(text: String) {
        if (text.isBlank() || phraseCache.containsKey(text) || !phraseCachePending.add(text)) return
        if (phraseCache.size >= MAX_CACHED_PHRASES) { phraseCachePending.remove(text); return }
        val engine = cacheTts ?: run { phraseCachePending.remove(text); return }
        try {
            val file = File(context.cacheDir, "ttscache_${text.hashCode()}.wav")
            engine.synthesizeToFile(text, null, file, "$CACHE_PREFIX$text")
        } catch (e: Exception) {
            phraseCachePending.remove(text)
            Log.w(TAG, "Cache synthesis request failed: ${e.message}")
        }
    }

    /** Parses the WAV file synthesizeToFile wrote and stores raw PCM16 samples + the WAV's own
     *  sample rate in the cache — playback must match this rate exactly or the phrase comes out
     *  pitch-shifted (see playCachedPhrase). WAV canonical header: bytes 24-27 LE = sample rate. */
    private fun loadCachedPhrase(text: String) {
        if (text.isBlank()) return
        try {
            val file = File(context.cacheDir, "ttscache_${text.hashCode()}.wav")
            val bytes = file.readBytes()
            file.delete()
            if (bytes.size <= 44) return  // no PCM payload beyond the canonical 44-byte WAV header
            val sampleRate =
                (bytes[24].toInt() and 0xFF) or
                ((bytes[25].toInt() and 0xFF) shl 8) or
                ((bytes[26].toInt() and 0xFF) shl 16) or
                ((bytes[27].toInt() and 0xFF) shl 24)
            if (sampleRate <= 0) return  // malformed header — skip rather than cache with a bogus rate
            val pcm = bytes.copyOfRange(44, bytes.size)
            val shorts = ShortArray(pcm.size / 2) { i ->
                ((pcm[i * 2].toInt() and 0xFF) or (pcm[i * 2 + 1].toInt() shl 8)).toShort()
            }
            phraseCache[text] = CachedPhrase(shorts, sampleRate)
            fileLogger.d(TAG, "Cached TTS phrase (${shorts.size} samples @ ${sampleRate}Hz): \"${text.take(40)}\"")
        } catch (e: Exception) {
            Log.w(TAG, "Cache load failed: ${e.message}")
        } finally {
            phraseCachePending.remove(text)
        }
    }

    /**
     * Execute a haptic vibration pattern.
     */
    private fun vibratePattern(pattern: HapticPattern) {
        val vib = vibrator ?: return

        val (timings, amplitudes) = when (pattern) {
            HapticPattern.PING -> {
                longArrayOf(0, 80) to intArrayOf(0, (180 * vibrationIntensity).toInt())
            }
            HapticPattern.DOUBLE_TAP -> {
                longArrayOf(0, 80, 100, 80) to
                        intArrayOf(0, (200 * vibrationIntensity).toInt(),
                            0, (200 * vibrationIntensity).toInt())
            }
            HapticPattern.BUZZ -> {
                longArrayOf(0, 400) to intArrayOf(0, (255 * vibrationIntensity).toInt())
            }
            HapticPattern.TRIPLE_PULSE -> {
                longArrayOf(0, 60, 40, 60, 40, 60) to
                        intArrayOf(0, (220 * vibrationIntensity).toInt(),
                            0, (220 * vibrationIntensity).toInt(),
                            0, (220 * vibrationIntensity).toInt())
            }
            HapticPattern.HEARTBEAT -> {
                longArrayOf(0, 200, 100, 200) to
                        intArrayOf(0, (200 * vibrationIntensity).toInt(),
                            0, (150 * vibrationIntensity).toInt())
            }
            HapticPattern.SOS_PATTERN -> {
                // Morse SOS: ... --- ...
                longArrayOf(0, 100, 50, 100, 50, 100, 150,
                    300, 50, 300, 50, 300, 150,
                    100, 50, 100, 50, 100) to
                        IntArray(18) { if (it % 2 == 0) 0 else (255 * vibrationIntensity).toInt() }
            }
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val effect = VibrationEffect.createWaveform(timings, amplitudes, -1)
            vib.vibrate(effect)
        } else {
            @Suppress("DEPRECATION")
            vib.vibrate(timings, -1)
        }
    }

    /**
     * Speak a system message (not an alert — no vibration, no panning).
     * Guard on isInitialized to handle calls before TTS async init completes
     * or after an unexpected shutdown.
     */
    fun speakSystem(text: String) {
        if (!isInitialized) return
        val ttsEngine = tts ?: return
        lastSpokenText = text
        fileLogger.i(TAG, "speakSystem: \"${text.take(60)}\"")
        fileLogger.voiceEvent("TTS_SYS", null, text)
        beginSpeaking(LIVE_TTS_WATCHDOG_MS)
        ttsEngine.speak(text, TextToSpeech.QUEUE_ADD, null, "system_${System.currentTimeMillis()}")
    }

    /** Re-speak the last alert or system phrase. Used by the "say again" voice command. */
    fun repeatLast() {
        val text = if (lastSpokenText.isBlank()) "Nothing to repeat yet." else lastSpokenText
        // Speak directly (not via speakSystem) so lastSpokenText is never overwritten here —
        // the "nothing to repeat" fallback would otherwise become the permanent replay target.
        beginSpeaking(LIVE_TTS_WATCHDOG_MS)
        tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "repeat_${System.currentTimeMillis()}")
    }

    /**
     * Stop all current speech and vibration.
     */
    fun stop() {
        tts?.stop()
        vibrator?.cancel()
        // Bump the generation so any watchdog still pending for the interrupted utterance
        // becomes a no-op instead of firing a confusing "no completion signal" warning later.
        speakingGeneration.incrementAndGet()
        isSpeaking = false
        _isBusy.value = false
    }

    /**
     * Marks TTS as busy and arms a watchdog that force-clears the speaking state if no
     * completion signal (onDone/onError/marker callback) arrives within [watchdogDelayMs].
     * See WATCHDOG_MARGIN_MS's doc comment for why this exists — without it, a completion
     * signal that's silently lost sticks isSpeaking=true forever, and every subsequent
     * alert (including P1 hazards) is dropped indefinitely by deliverAlert()'s priority gate.
     */
    private fun beginSpeaking(watchdogDelayMs: Long) {
        val gen = speakingGeneration.incrementAndGet()
        isSpeaking = true
        _isBusy.value = true
        watchdogScope.launch {
            delay(watchdogDelayMs)
            if (speakingGeneration.get() == gen && isSpeaking) {
                fileLogger.w(TAG, "TTS watchdog: no completion signal within ${watchdogDelayMs}ms — forcing reset (gen=$gen)")
                endSpeaking()
            }
        }
    }

    /** Clears speaking state on a genuine completion signal or a watchdog timeout. Bumps the
     *  generation counter so a still-pending watchdog for an earlier, now-superseded
     *  utterance can never clobber whatever speaks next. */
    private fun endSpeaking() {
        speakingGeneration.incrementAndGet()
        isSpeaking = false
        currentSpeakingPriority = Int.MAX_VALUE
        quietUntilMs.set(System.currentTimeMillis() + NovaConstants.LISTENING_WINDOW_MS)
        _isBusy.value = false
    }

    /**
     * Barge-in: called when the user starts talking. Cuts off in-flight TTS (and its
     * queued follow-ups, since TextToSpeech.stop() flushes the queue) so the mic gets
     * priority — unless NOVA is mid-hazard (P1/P2), which must finish; cutting off
     * "Stop! stairs" mid-sentence is a worse outcome than briefly talking over a command.
     *
     * NOTE: the mic is intentionally stopped while TTS plays (see VoiceCommandProcessor.
     * notifyTtsBusy) to prevent AEC echo feedback, so onBeginningOfSpeech normally can't
     * fire mid-TTS at all — this is a defensive backstop for the known case where the
     * recognizer's stopListening() doesn't take effect immediately, not the primary
     * mechanism. True simultaneous barge-in would need a always-hot mic with echo
     * cancellation, which this single-mic architecture doesn't have.
     */
    fun interruptForUserSpeech() {
        if (isSpeaking && currentSpeakingPriority > AlertPriority.P2_DANGER.level) {
            fileLogger.d(TAG, "Barge-in: user started speaking — interrupting TTS")
            stop()
        }
    }

    /**
     * Suppress ALL alert delivery (even P1/P2) for [durationMs] from now. Call again to
     * extend/replace the deadline — TextReaderEngine calls this once on OCR start (a short
     * bound covering recognition latency) and again once the result text is known (a bound
     * sized to its estimated spoken duration). Bounded, not indefinite: a stuck OCR call can
     * never permanently silence real hazards.
     */
    fun beginOcrQuietPeriod(durationMs: Long) {
        ocrQuietUntilMs = System.currentTimeMillis() + durationMs
        fileLogger.d(TAG, "OCR quiet period: ${durationMs}ms")
    }

    /**
     * End the quiet period immediately instead of waiting out the remaining deadline. Used when
     * the deliberate stationary task finishes early (e.g. the room name was captured before the
     * heartbeat-refreshed window elapsed) so hazard alerts resume without an unnecessary tail of
     * silence. Safe to call when no quiet period is active (idempotent).
     */
    fun endQuietPeriod() {
        ocrQuietUntilMs = 0L
        fileLogger.d(TAG, "Quiet period ended early")
    }

    /** Toggles Finder-tab alert suppression — see finderActive's doc comment above. Called from
     *  MainActivity's selectedTab LaunchedEffect, in lockstep with suspendNavPipeline. */
    fun setFinderActive(active: Boolean) {
        finderActive = active
        fileLogger.d(TAG, "Finder active=$active — alert delivery ${if (active) "suppressed" else "resumed"}")
    }

    /** Clears attention-gate memory so objects near the user are re-announced immediately on resume. */
    fun resetAttentionGate() {
        lastSeenMs.clear()
        lastAnnouncedLevel.clear()
        lastAnnouncedKeyText.clear()
    }

    fun setSpeechRate(rate: Float) {
        speechRate = rate.coerceIn(0.5f, 2.0f)
        tts?.setSpeechRate(speechRate)
    }

    fun setVibrationEnabled(enabled: Boolean) {
        vibrationEnabled = enabled
    }

    fun setVibrationIntensity(intensity: Float) {
        vibrationIntensity = intensity.coerceIn(0.3f, 1.0f)
    }

    // ── Earcon helpers ────────────────────────────────────────────────────────

    private fun initEarcons() {
        val sr = 22050
        // P1: ascending chirp 440→880 Hz, 60ms, linear fade-out
        val p1Len = sr * 60 / 1000
        val p1Samples = ShortArray(p1Len) { i ->
            val hz  = 440.0 + (880.0 - 440.0) * i.toDouble() / p1Len
            val env = 1.0 - i.toDouble() / p1Len
            (Short.MAX_VALUE * 0.6 * env * sin(2.0 * PI * hz * i / sr)).toInt().toShort()
        }
        earconP1 = buildAudioTrack(p1Samples, sr)

        // P2: two 880 Hz pulses (25ms on / 15ms off / 25ms on), fade-out per pulse
        val pulse = sr * 25 / 1000
        val gap   = sr * 15 / 1000
        val p2Len = pulse + gap + pulse
        val p2Samples = ShortArray(p2Len) { i ->
            val inPulse1 = i < pulse
            val inPulse2 = i >= pulse + gap
            val env = when {
                inPulse1 -> 1.0 - i.toDouble() / pulse
                inPulse2 -> 1.0 - (i - pulse - gap).toDouble() / pulse
                else     -> 0.0
            }
            if (inPulse1 || inPulse2)
                (Short.MAX_VALUE * 0.55 * env * sin(2.0 * PI * 880.0 * i / sr)).toInt().toShort()
            else 0
        }
        earconP2 = buildAudioTrack(p2Samples, sr)

        // Listening cue: soft single 660Hz blip, 40ms — played the instant the recognizer is
        // actually armed (SpeechRecognizer.onReadyForSpeech). Blind users had no signal for when
        // the mic reopens after TTS ends, so short commands like "pause"/"resume" — which follow
        // immediately after NOVA speaks — were frequently clipped mid-word by speaking too early,
        // needing many retries to clear their (deliberately high) confidence thresholds.
        val cueLen = sr * 40 / 1000
        val cueSamples = ShortArray(cueLen) { i ->
            val env = 1.0 - i.toDouble() / cueLen
            (Short.MAX_VALUE * 0.35 * env * sin(2.0 * PI * 660.0 * i / sr)).toInt().toShort()
        }
        earconListening = buildAudioTrack(cueSamples, sr)
    }

    /** Played by VoiceCommandProcessor.onReadyForSpeech — see earconListening init comment. */
    fun playListeningCue() {
        val track = earconListening ?: return
        try {
            if (track.playState != AudioTrack.PLAYSTATE_STOPPED) track.stop()
            track.setPlaybackHeadPosition(0)
            track.play()
        } catch (e: Exception) {
            Log.w(TAG, "Listening cue play error: ${e.message}")
        }
    }

    private fun buildAudioTrack(samples: ShortArray, sampleRate: Int): AudioTrack? {
        return try {
            val minBuf = AudioTrack.getMinBufferSize(
                sampleRate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT
            )
            AudioTrack(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_ASSISTANCE_ACCESSIBILITY)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
                    .build(),
                AudioFormat.Builder()
                    .setSampleRate(sampleRate)
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .build(),
                maxOf(samples.size * 2, minBuf),
                AudioTrack.MODE_STATIC,
                AudioManager.AUDIO_SESSION_ID_GENERATE
            ).also { it.write(samples, 0, samples.size) }
        } catch (e: Exception) {
            Log.w(TAG, "Earcon build failed: ${e.message}")
            null
        }
    }

    private fun playEarcon(priority: AlertPriority) {
        val track = when {
            priority.level <= AlertPriority.P1_EMERGENCY.level -> earconP1
            priority.level <= AlertPriority.P2_DANGER.level    -> earconP2
            else -> return
        } ?: return
        try {
            // MODE_STATIC replay: stop → reset head → play.
            // setPlaybackHeadPosition(0) is only valid in PLAYSTATE_STOPPED.
            if (track.playState != AudioTrack.PLAYSTATE_STOPPED) track.stop()
            track.setPlaybackHeadPosition(0)
            track.play()
        } catch (e: Exception) {
            Log.w(TAG, "Earcon play error: ${e.message}")
        }
    }

    fun shutdown() {
        watchdogScope.cancel()
        tts?.stop()
        tts?.shutdown()
        tts = null          // Null out so speakSystem() guard catches any stale calls
        cacheTts?.stop()
        cacheTts?.shutdown()
        cacheTts = null
        vibrator?.cancel()
        earconP1?.release(); earconP1 = null
        earconP2?.release(); earconP2 = null
        earconListening?.release(); earconListening = null
        phraseCache.clear()
        phraseCachePending.clear()
        isInitialized = false
    }
}
