package com.nova.assistant.features.voicecommand

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.pm.ResolveInfo
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.speech.RecognitionListener
import android.speech.RecognitionSupport
import android.speech.RecognitionSupportCallback
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import androidx.annotation.RequiresApi
import com.nova.assistant.feedback.FeedbackOrchestrator
import com.nova.assistant.util.FileLogger
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.launch
import java.util.concurrent.ConcurrentHashMap
import javax.inject.Inject
import javax.inject.Singleton

/**
 * NOVA Voice Command Processor
 *
 * Listens continuously, recognizes speech, and matches against 27 registered commands.
 *
 * Uses Android's built-in SpeechRecognizer with EXTRA_PREFER_OFFLINE=true by default, routing
 * to the on-device language pack when installed (Google app → Settings → Voice → Offline speech
 * recognition → English). If the pack is missing (ERROR_LANGUAGE_NOT_SUPPORTED), the processor
 * silently switches to online mode (EXTRA_PREFER_OFFLINE=false) and announces it once via TTS.
 *
 * TTS echo prevention: when FeedbackOrchestrator is speaking (isBusy=true), the recognizer
 * is stopped immediately. It restarts 500 ms after TTS finishes so room reverb decays.
 * Call [notifyTtsBusy] from a coroutine observing FeedbackOrchestrator.isBusy.
 *
 * Device notes:
 *   Samsung: use com.samsung.android.bixby.agent or Google QSB
 *   Xiaomi:  uses Google QSB (MIUI ships Google apps); EXTRA_PREFER_OFFLINE works
 *   Huawei:  com.huawei.hiai or com.huawei.intelligent (no GMS on some models)
 */
@Singleton
class VoiceCommandProcessor @Inject constructor(
    private val context: Context,
    private val fileLogger: FileLogger,
    private val feedbackOrchestrator: FeedbackOrchestrator,
) {
    private var recognizer: SpeechRecognizer? = null

    @Volatile private var isListening = false
    @Volatile private var isTtsMuted = false   // true while TTS is playing

    private var retryCount = 0
    private var consecutiveErrors = 0
    private val handler = Handler(Looper.getMainLooper())

    // True once ERROR_LANGUAGE_NOT_SUPPORTED fires — switches all subsequent
    // startListening() calls to online mode (EXTRA_PREFER_OFFLINE=false).
    @Volatile private var offlinePackMissing = false
    private var announcedOnlineMode = false
    // Best available English locale — updated by checkAndTriggerPackDownload()
    // to whichever variant is installed/downloaded on this device.
    @Volatile private var activeLocale = "en-US"

    // Packages that threw SecurityException at bind time — discovered at runtime.
    // com.anthropic.claude is a known offender on Huawei EMUI.
    private val runtimeBlockedPackages = mutableSetOf<String>()

    // Watchdog: if startListening() fires but no callback arrives within WATCHDOG_MS
    // the recognizer is assumed dead (EMUI battery optimization kills it silently).
    private var watchdogRunnable: Runnable? = null

    companion object {
        private const val TAG = "VoiceCommandProcessor"
        private const val BASE_RETRY_DELAY_MS   = 500L
        private const val MAX_RETRY_DELAY_MS    = 3000L
        // How long to wait after startListening() before declaring the recognizer dead.
        // SpeechRecognizer fires onReadyForSpeech within 1-2s; 12s is generous for slow devices.
        private const val WATCHDOG_MS           = 12_000L
        // After this many consecutive errors, recreate the SpeechRecognizer entirely.
        // EMUI can get the recognizer stuck in an error loop that only destroy+create clears.
        private const val RECREATE_AFTER_ERRORS = 5
        // Google Speech Services needs ~1s to establish the connection after bind.
        // Calling startListening() immediately triggers ERROR_SERVER_DISCONNECTED (code 11).
        private const val RECREATE_SETTLE_MS    = 1500L
        // Minimum ms between two dispatches of the same command.
        // Per-command cooldown exemption (EMERGENCY/CANCEL_EMERGENCY) is data in CommandCatalog.
        private const val COMMAND_COOLDOWN_MS   = 5000L
        // Buffer after TTS ends before reopening the mic — lets room echo fully decay.
        private const val TTS_UNMUTE_DELAY_MS   = 500L
        // English locale priority for offline pack selection.
        // First installed variant wins; if none installed, first supported variant is downloaded.
        val ENGLISH_LOCALES = listOf("en-US", "en-GB", "en-AU", "en-IN", "en-CA", "en-NZ", "en-ZA")
    }

    // extraBufferCapacity=1 so tryEmit reliably reaches BOTH collectors —
    // MainNavigationViewModel (action/query commands) and the app-level NavVoiceViewModel
    // (navigation commands). With buffer=0 a momentarily-busy collector could drop the emit.
    private val _commandFlow = MutableSharedFlow<NovaCommand>(replay = 0, extraBufferCapacity = 1)
    val commandFlow: SharedFlow<NovaCommand> = _commandFlow.asSharedFlow()

    // Emits every recognized utterance that does NOT match a command.
    // Used by SCAN_ROOM (room name capture) and LOAD_ROOM (spoken number selection).
    private val _rawTextFlow = MutableSharedFlow<String>(replay = 0, extraBufferCapacity = 1)
    val rawTextFlow: SharedFlow<String> = _rawTextFlow.asSharedFlow()

    // True while the ASR has detected live voice activity (onBeginningOfSpeech..onEndOfSpeech).
    // Consumed by MainNavigationViewModel to hold back non-emergency alerts while the user talks —
    // voice commands rank second only to EMERGENCY in the speech priority hierarchy.
    private val _isUserSpeaking = MutableStateFlow(false)
    val isUserSpeaking: StateFlow<Boolean> = _isUserSpeaking

    private var isInitialized = false
    // Set true when all recognizers are blocked so we stop the retry storm.
    private var permanentlyDisabled = false

    private val lastCommandTime = ConcurrentHashMap<NovaCommand, Long>()

    // ── Lifecycle ───────────────────────────────────────────────────────────────

    fun initialize() {
        if (isInitialized) return
        if (!SpeechRecognizer.isRecognitionAvailable(context)) {
            fileLogger.e(TAG, "SpeechRecognizer.isRecognitionAvailable=false — voice commands disabled on this device")
            fileLogger.voiceEvent("ASR_INIT", null, "[disabled — SpeechRecognizer not available]", "ASR")
            return
        }
        createRecognizer()
        if (recognizer != null) {
            isInitialized = true
            fileLogger.i(TAG, "Voice processor initialized — SpeechRecognizer (on-device pack preferred via EXTRA_PREFER_OFFLINE)")
            fileLogger.voiceEvent("ASR_INIT", null, "[ready — on-device pack preferred]", "ASR")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                checkAndTriggerPackDownload()
            }
        } else if (permanentlyDisabled) {
            GlobalScope.launch(Dispatchers.Main) {
                feedbackOrchestrator.speakSystem("Voice commands unavailable on this device.")
            }
        }
    }

    private fun createRecognizer() {
        try {
            recognizer?.destroy()
            recognizer = null
            val component = getPreferredRecognizerComponent()
            recognizer = if (component != null) {
                fileLogger.i(TAG, "Using recognizer: ${component.packageName}/${component.className}")
                SpeechRecognizer.createSpeechRecognizer(context, component)
            } else {
                // All listed recognizers are blocked (e.g. only com.anthropic.claude registered on
                // Huawei EMUI). Do NOT fall through to system default — it binds the same blocked
                // package and throws SecurityException on every startListening(), causing a log
                // flood and permanent ASR_DISABLED on session start. Stay null; voice commands
                // are gracefully disabled and startListening() returns early on recognizer == null.
                fileLogger.e(TAG, "No usable recognizer — all services blocked; skipping system default")
                fileLogger.voiceEvent("ASR_INIT", null, "[disabled — all recognizers blocked]", "ASR")
                permanentlyDisabled = true
                null
            }
            recognizer?.setRecognitionListener(buildListener())
            consecutiveErrors = 0
            retryCount = 0
            fileLogger.i(TAG, "SpeechRecognizer created")
            fileLogger.voiceEvent("ASR_CREATED", null, "[recognizer ready]", "ASR")
        } catch (e: Exception) {
            fileLogger.e(TAG, "Failed to create SpeechRecognizer: ${e.message}")
            fileLogger.voiceEvent("ASR_CREATE_FAIL", null, "[${e.message}]", "ASR")
            recognizer = null
        }
    }

    /**
     * On API 33+: find the best available English offline pack and use or download it.
     *
     * Priority order: [ENGLISH_LOCALES] (en-US first, then GB, AU, IN, CA, NZ, ZA).
     * 1. If any variant is already installed → use it, update activeLocale, done.
     * 2. Else if any variant is supported (downloadable) → trigger download for the
     *    best one and announce via TTS.
     * 3. Else → log warning; online fallback remains active.
     */
    @RequiresApi(Build.VERSION_CODES.TIRAMISU)
    private fun checkAndTriggerPackDownload() {
        // checkRecognitionSupport needs a language hint; use en-US as the probe locale.
        val probeIntent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, "en-US")
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
        }
        try {
            recognizer?.checkRecognitionSupport(
                probeIntent,
                context.mainExecutor,
                object : RecognitionSupportCallback {
                    override fun onSupportResult(support: RecognitionSupport) {
                        val installed  = support.installedOnDeviceLanguages
                        val supported  = support.supportedOnDeviceLanguages
                        fileLogger.i(TAG, "Offline pack check: installed=$installed supported=$supported")

                        // 1. Already installed — pick best and use it immediately.
                        val bestInstalled = ENGLISH_LOCALES.firstOrNull { it in installed }
                        if (bestInstalled != null) {
                            activeLocale = bestInstalled
                            fileLogger.i(TAG, "Offline pack ready: $bestInstalled — activeLocale updated")
                            return
                        }

                        // 2. Not installed but downloadable — trigger download for best option.
                        val bestDownloadable = ENGLISH_LOCALES.firstOrNull { it in supported }
                        if (bestDownloadable != null) {
                            activeLocale = bestDownloadable  // optimistic: will be usable after download
                            fileLogger.i(TAG, "Triggering offline pack download: $bestDownloadable")
                            val downloadIntent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                                putExtra(RecognizerIntent.EXTRA_LANGUAGE, bestDownloadable)
                                putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                            }
                            recognizer?.triggerModelDownload(downloadIntent)
                            GlobalScope.launch(Dispatchers.Main) {
                                feedbackOrchestrator.speakSystem(
                                    "Downloading offline voice pack. Voice commands will work offline once complete."
                                )
                            }
                            return
                        }

                        // 3. Nothing available offline — online fallback remains active.
                        fileLogger.w(TAG, "No English variant available for offline download — online mode only")
                    }
                    override fun onError(error: Int) {
                        fileLogger.w(TAG, "checkRecognitionSupport error $error — skipping pack download check")
                    }
                }
            )
        } catch (e: Exception) {
            fileLogger.w(TAG, "checkAndTriggerPackDownload failed: ${e.message}")
        }
    }

    /**
     * Query installed speech recognition services and return the best available component.
     *
     * Priority:
     *   1. Google Search / Quick Search Box  (most reliable cross-device)
     *   2. Samsung Bixby                     (reliable on Samsung without GMS issues)
     *   3. Huawei HIAI                       (Huawei devices without GMS)
     *   4. Any other non-blocked recognizer
     *
     * Blocked: com.anthropic.claude — registers ClaudeRecognitionService → SecurityException on EMUI
     */
    @Suppress("DEPRECATION")
    private fun getPreferredRecognizerComponent(): ComponentName? {
        val pm = context.packageManager
        // BUG FIX 2026-07-16: was Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH) — that's the
        // ACTIVITY action for launching a recognition UI, not the SERVICE action recognizer apps
        // declare in their <service> intent-filters. queryIntentServices() against that action
        // always returned an empty list regardless of what's installed, which made
        // getPreferredRecognizerComponent() return null on every device and permanently disable
        // voice commands ("No speech recognizer services found on device"). The correct action is
        // "android.speech.RecognitionService" (== SpeechRecognizer.SERVICE_INTERFACE on SDKs where
        // that constant resolves) — the <queries> block in AndroidManifest.xml already declares
        // this exact action for package-visibility; this call just wasn't using it. Literal string
        // used directly since SERVICE_INTERFACE doesn't resolve against this project's compileSdk stub.
        val intent = Intent("android.speech.RecognitionService")
        val services: List<ResolveInfo> = pm.queryIntentServices(intent, 0)

        if (services.isEmpty()) {
            fileLogger.w(TAG, "No speech recognizer services found on device")
            return null
        }

        val packageList = services.map { it.serviceInfo.packageName }
        fileLogger.d(TAG, "Available recognizers: $packageList")
        fileLogger.voiceEvent("ASR_SERVICES", null, "[${packageList.joinToString()}]", "ASR")

        val blockedPackages = setOf("com.anthropic.claude") + runtimeBlockedPackages

        val preferredPackages = listOf(
            "com.google.android.googlequicksearchbox",  // Google Search (Pixel, most Android)
            "com.google.android.tts",                   // Google TTS (backup)
            "com.samsung.android.bixby.agent",          // Samsung Bixby
            "com.huawei.hiai",                          // Huawei AI
            "com.huawei.intelligent"                    // Huawei (alternate package)
        )

        for (pkg in preferredPackages) {
            val match = services.find { it.serviceInfo.packageName == pkg }
            if (match != null) {
                fileLogger.d(TAG, "Selected recognizer: $pkg")
                return ComponentName(match.serviceInfo.packageName, match.serviceInfo.name)
            }
        }

        val fallback = services.find { it.serviceInfo.packageName !in blockedPackages }
        return if (fallback != null) {
            fileLogger.d(TAG, "Fallback recognizer: ${fallback.serviceInfo.packageName}")
            fileLogger.voiceEvent("ASR_FALLBACK", null, "[${fallback.serviceInfo.packageName}]", "ASR")
            ComponentName(fallback.serviceInfo.packageName, fallback.serviceInfo.name)
        } else {
            fileLogger.e(TAG, "All recognizers blocked — voice commands permanently disabled")
            fileLogger.voiceEvent("ASR_BLOCKED", null, "[no usable recognizer]", "ASR")
            null
        }
    }

    // ── TTS muting ─────────────────────────────────────────────────────────────

    /**
     * Call this whenever FeedbackOrchestrator.isBusy changes.
     * busy=true  → stop the recognizer immediately (prevents echo of TTS output).
     * busy=false → restart after [TTS_UNMUTE_DELAY_MS] so room reverb fully decays.
     *
     * Wire from MainNavigationViewModel:
     *   viewModelScope.launch { feedback.isBusy.collect { voiceProcessor.notifyTtsBusy(it) } }
     */
    fun notifyTtsBusy(busy: Boolean) {
        if (busy) {
            if (!isTtsMuted) {
                isTtsMuted = true
                handler.post {
                    if (isListening) {
                        recognizer?.stopListening()
                        isListening = false
                        cancelWatchdog()
                        fileLogger.d(TAG, "Mic muted — TTS started")
                        fileLogger.voiceEvent("TTS_MUTE", null, "[mic stopped — TTS active]", "ASR")
                    }
                }
            }
        } else {
            if (isTtsMuted) {
                isTtsMuted = false
                handler.postDelayed({
                    if (!isTtsMuted && isInitialized && !permanentlyDisabled) {
                        fileLogger.d(TAG, "Mic unmuted — TTS ended, resuming after ${TTS_UNMUTE_DELAY_MS}ms buffer")
                        fileLogger.voiceEvent("TTS_UNMUTE", null, "[mic reopening after echo buffer]", "ASR")
                        startListening()
                    }
                }, TTS_UNMUTE_DELAY_MS)
            }
        }
    }

    // ── Listening ───────────────────────────────────────────────────────────────

    fun startListening() {
        if (isListening || recognizer == null || permanentlyDisabled || isTtsMuted) return
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, activeLocale)
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, false)
            // Prefer on-device pack; drops to false if pack is missing (see onError 13).
            putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, !offlinePackMissing)
        }
        try {
            recognizer?.startListening(intent)
            isListening = true
            fileLogger.d(TAG, "startListening() — watchdog armed for ${WATCHDOG_MS}ms")
            fileLogger.voiceEvent("ASR_START", null, "[mic open]", "ASR")
            armWatchdog()
        } catch (e: SecurityException) {
            val failedPkg = e.message?.let { extractPackageName(it) }
            if (failedPkg != null) {
                runtimeBlockedPackages.add(failedPkg)
                fileLogger.e(TAG, "SecurityException: $failedPkg blocked at runtime — recreating")
                fileLogger.voiceEvent("ASR_BLOCKED_PKG", null, "[$failedPkg blocked at runtime]", "ASR")
            } else {
                fileLogger.e(TAG, "SecurityException binding recognizer: ${e.message}")
                fileLogger.voiceEvent("ASR_SEC_EXCEPTION", null, "[${e.message}]", "ASR")
            }

            val nextComponent = getPreferredRecognizerComponent()
            if (nextComponent == null) {
                permanentlyDisabled = true
                fileLogger.e(TAG, "Voice commands permanently disabled — no usable recognizer")
                fileLogger.voiceEvent("ASR_DISABLED", null, "[no usable recognizer — all blocked]", "ASR")
                return
            }

            createRecognizer()
            handler.postDelayed({ startListening() }, MAX_RETRY_DELAY_MS)
        }
    }

    private fun armWatchdog() {
        cancelWatchdog()
        val r = Runnable {
            fileLogger.w(TAG, "Watchdog fired — no callback in ${WATCHDOG_MS}ms. Recognizer dead. Recreating.")
            fileLogger.voiceEvent("ASR_WATCHDOG", null, "[no callback — recreating recognizer]", "ASR")
            isListening = false
            createRecognizer()
            handler.postDelayed({ startListening() }, RECREATE_SETTLE_MS)
        }
        watchdogRunnable = r
        handler.postDelayed(r, WATCHDOG_MS)
    }

    private fun cancelWatchdog() {
        watchdogRunnable?.let { handler.removeCallbacks(it) }
        watchdogRunnable = null
    }

    fun stopListening() {
        cancelWatchdog()
        recognizer?.stopListening()
        isListening = false
        fileLogger.d(TAG, "stopListening() called")
        fileLogger.voiceEvent("ASR_STOP", null, "[mic closed]", "ASR")
    }

    // ── Recognition listener ────────────────────────────────────────────────────

    private fun buildListener() = object : RecognitionListener {
        override fun onReadyForSpeech(params: Bundle?) {
            cancelWatchdog()
            fileLogger.d(TAG, "onReadyForSpeech — mic open, ready for speech")
            fileLogger.voiceEvent("ASR_READY", null, "[mic ready — waiting for speech]", "ASR")
            // Audible cue so the user knows exactly when to speak — see earconListening comment
            // in FeedbackOrchestrator. Fixes short commands ("pause"/"resume") being clipped by
            // speaking before the recognizer is actually armed.
            feedbackOrchestrator.playListeningCue()
        }

        override fun onBeginningOfSpeech() {
            fileLogger.d(TAG, "onBeginningOfSpeech — voice activity detected")
            fileLogger.voiceEvent("ASR_VAD", null, "[voice activity detected]", "ASR")
            _isUserSpeaking.value = true
            feedbackOrchestrator.interruptForUserSpeech()
        }

        override fun onRmsChanged(rmsdB: Float) {}
        override fun onBufferReceived(buffer: ByteArray?) {}

        override fun onEndOfSpeech() {
            isListening = false
            _isUserSpeaking.value = false
            fileLogger.d(TAG, "onEndOfSpeech — speech ended, decoding")
            fileLogger.voiceEvent("ASR_SPEECH_END", null, "[speech ended — decoding]", "ASR")
        }

        override fun onPartialResults(partialResults: Bundle?) {
            val partial = partialResults
                ?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                ?.firstOrNull() ?: return
            if (partial.isNotBlank()) {
                fileLogger.d(TAG, "onPartialResults: \"$partial\"")
                fileLogger.voiceEvent("ASR_PARTIAL", partial, null, "ASR")
            }
        }

        override fun onEvent(eventType: Int, params: Bundle?) {}

        override fun onError(error: Int) {
            isListening = false
            _isUserSpeaking.value = false
            cancelWatchdog()
            consecutiveErrors++

            val errorName = errorName(error)
            fileLogger.w(TAG, "onError: $errorName ($error) consecutive=$consecutiveErrors ttsMuted=$isTtsMuted")
            fileLogger.voiceEvent("ASR_ERROR", null, "[$errorName consecutive=$consecutiveErrors]", "ASR")

            // On first ERROR_LANGUAGE_NOT_SUPPORTED: flip to online mode so all
            // subsequent startListening() calls omit EXTRA_PREFER_OFFLINE.
            // Must happen before the recreate check so the flag is set before the next startListening().
            if (error == 13 && !offlinePackMissing) {
                offlinePackMissing = true
                fileLogger.e(TAG, "On-device pack missing — switching to online recognition permanently")
                fileLogger.voiceEvent("ASR_ONLINE_FALLBACK", null, "[pack missing — online mode active]", "ASR")
                if (!announcedOnlineMode) {
                    announcedOnlineMode = true
                    GlobalScope.launch(Dispatchers.Main) {
                        feedbackOrchestrator.speakSystem("Voice using online mode.")
                    }
                }
            }

            // Recreate after too many consecutive errors — EMUI gets recognizer stuck.
            if (consecutiveErrors >= RECREATE_AFTER_ERRORS) {
                fileLogger.w(TAG, "Recreating recognizer after $consecutiveErrors consecutive errors")
                fileLogger.voiceEvent("ASR_RECREATE", null, "[${consecutiveErrors} errors — recreating]", "ASR")
                handler.postDelayed({
                    createRecognizer()
                    handler.postDelayed({ startListening() }, RECREATE_SETTLE_MS)
                }, MAX_RETRY_DELAY_MS)
                return
            }

            val shouldRetry = when (error) {
                SpeechRecognizer.ERROR_NO_MATCH,
                SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> true

                SpeechRecognizer.ERROR_AUDIO -> {
                    fileLogger.w(TAG, "Audio/mic conflict — TTS may still be active")
                    true
                }
                SpeechRecognizer.ERROR_CLIENT -> {
                    fileLogger.w(TAG, "Client error — recognizer in bad state")
                    true
                }
                SpeechRecognizer.ERROR_RECOGNIZER_BUSY -> {
                    // EMUI fires RECOGNIZER_BUSY instead of ERROR_AUDIO when TTS is active.
                    fileLogger.w(TAG, "Recognizer busy — likely TTS conflict on EMUI")
                    true
                }
                SpeechRecognizer.ERROR_SERVER -> {
                    fileLogger.e(TAG, "Server error — retrying")
                    true
                }
                SpeechRecognizer.ERROR_NETWORK,
                SpeechRecognizer.ERROR_NETWORK_TIMEOUT -> {
                    fileLogger.e(TAG, "Network error — on-device pack may not be installed")
                    true
                }
                SpeechRecognizer.ERROR_SERVER_DISCONNECTED -> {
                    fileLogger.w(TAG, "Server disconnected — device offline or server unreachable")
                    true
                }
                13 -> { // ERROR_LANGUAGE_NOT_SUPPORTED — already handled above; just retry in online mode
                    true
                }
                else -> {
                    fileLogger.w(TAG, "Unknown error $error — retrying")
                    true
                }
            }

            if (shouldRetry && !isTtsMuted) {
                retryCount++
                val delayMs = minOf(BASE_RETRY_DELAY_MS * retryCount, MAX_RETRY_DELAY_MS)
                fileLogger.d(TAG, "Retry $retryCount in ${delayMs}ms")
                handler.postDelayed({ startListening() }, delayMs)
            }
            // If isTtsMuted, notifyTtsBusy(false) will restart listening after TTS ends.
        }

        override fun onResults(results: Bundle?) {
            isListening = false
            cancelWatchdog()
            retryCount = 0
            consecutiveErrors = 0

            val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
            // Read the FULL n-best list (not just the top hypothesis) so a misheard top
            // can be recovered by a cleaner alternate. Confidence is read for logging only —
            // it is frequently absent (-1) offline, NOT used in the accept decision.
            val confidences = results?.getFloatArray(SpeechRecognizer.CONFIDENCE_SCORES)
            val nbest = matches?.mapNotNull { it?.lowercase()?.trim()?.takeIf(String::isNotBlank) }
                ?: emptyList()
            val recognized = nbest.firstOrNull() ?: ""

            fileLogger.d(TAG, "onResults: heard=\"$recognized\" nbest=${nbest.size} ttsMuted=$isTtsMuted")
            // Log every raw utterance — even unmatched ones go to voice log
            fileLogger.voiceEvent("ASR_HEARD", recognized, null, "ASR")

            processResults(nbest, confidences)

            // Restart unless TTS is speaking. If TTS is active, notifyTtsBusy(false) restarts.
            if (!isTtsMuted) {
                startListening()
            }
        }
    }

    // ── Command processing ──────────────────────────────────────────────────────

    // Scores the n-best list against the command catalog (VoiceMatcher). A command is
    // dispatched only if its score clears the command's accept threshold — this is what
    // rejects ambient speech ("I need to stop at the pharmacy" no longer fires PAUSE).
    // Below threshold → emitted to rawTextFlow so SCAN_ROOM / LOAD_ROOM can still capture
    // room names and spoken numbers.
    private fun processResults(nbest: List<String>, confidences: FloatArray?) {
        if (!isInitialized) return
        if (nbest.isEmpty()) {
            fileLogger.d(TAG, "processResults: blank utterance — ignored")
            return
        }

        val result = VoiceMatcher.match(nbest)
        val command = result.command
        val scoreStr = "%.2f".format(result.score)
        // Top-hypothesis ASR confidence, logged for offline FAR/FRR threshold tuning.
        // Often -1 (unavailable) offline — recorded, not used in the accept decision.
        val topConf = confidences?.firstOrNull()?.takeIf { it >= 0f }?.let { "%.2f".format(it) } ?: "n/a"

        if (result.accepted && command != null) {
            val exempt = result.spec?.cooldownExempt == true
            val now = System.currentTimeMillis()
            val lastTime = lastCommandTime[command] ?: 0L
            if (!exempt && now - lastTime < COMMAND_COOLDOWN_MS) {
                val suppressedMs = now - lastTime
                fileLogger.d(TAG, "Command $command suppressed — cooldown ${suppressedMs}ms < ${COMMAND_COOLDOWN_MS}ms")
                fileLogger.voiceEvent(command.name, nbest.first(), "[suppressed — cooldown ${suppressedMs}ms]", "USER")
            } else {
                lastCommandTime[command] = now
                fileLogger.i(TAG, "Command matched: $command score=$scoreStr conf=$topConf (cooldownExempt=$exempt)")
                fileLogger.voiceEvent(command.name, nbest.first(), "[dispatched score=$scoreStr conf=$topConf]", "USER")
                _commandFlow.tryEmit(command)
            }
        } else {
            // Best candidate (if any) scored below its threshold → treat as non-command.
            val top = nbest.first()
            fileLogger.d(TAG, "No command (best=${command} score=$scoreStr) — emitting to rawTextFlow: \"$top\"")
            fileLogger.voiceEvent("UNKNOWN", top, "[no command score=$scoreStr — rawTextFlow]", "AMBIENT")
            _rawTextFlow.tryEmit(top)
        }
    }

    // ── Utilities ───────────────────────────────────────────────────────────────

    private fun extractPackageName(message: String): String? {
        val cmpPattern = Regex("cmp=([^/\\s]+)/")
        return cmpPattern.find(message)?.groupValues?.getOrNull(1)
    }

    private fun errorName(error: Int): String = when (error) {
        SpeechRecognizer.ERROR_AUDIO                   -> "ERROR_AUDIO"
        SpeechRecognizer.ERROR_CLIENT                  -> "ERROR_CLIENT"
        SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> "ERROR_INSUFFICIENT_PERMISSIONS"
        SpeechRecognizer.ERROR_NETWORK                 -> "ERROR_NETWORK"
        SpeechRecognizer.ERROR_NETWORK_TIMEOUT         -> "ERROR_NETWORK_TIMEOUT"
        SpeechRecognizer.ERROR_NO_MATCH                -> "ERROR_NO_MATCH"
        SpeechRecognizer.ERROR_RECOGNIZER_BUSY         -> "ERROR_RECOGNIZER_BUSY"
        SpeechRecognizer.ERROR_SERVER                  -> "ERROR_SERVER"
        SpeechRecognizer.ERROR_SPEECH_TIMEOUT          -> "ERROR_SPEECH_TIMEOUT"
        SpeechRecognizer.ERROR_SERVER_DISCONNECTED     -> "ERROR_SERVER_DISCONNECTED"
        13                                             -> "ERROR_LANGUAGE_NOT_SUPPORTED"
        else                                           -> "ERROR_UNKNOWN_$error"
    }

    // Command matching now lives in VoiceMatcher / CommandCatalog (data-driven, scored,
    // unit-tested). The old substring `matchCommand` + `containsAny` were removed — they
    // fired on ambient speech and relied on fragile manual branch ordering.

    fun shutdown() {
        handler.removeCallbacksAndMessages(null)
        watchdogRunnable = null
        recognizer?.destroy()
        recognizer = null
        isListening = false
        isTtsMuted = false
        isInitialized = false
        retryCount = 0
        consecutiveErrors = 0
        offlinePackMissing = false
        announcedOnlineMode = false
        activeLocale = "en-US"
        // permanentlyDisabled NOT reset — singleton outlives ViewModel restarts.
        // Prevents retry storm on devices where all recognizers are blocked.
        fileLogger.i(TAG, "Voice processor shut down (permanentlyDisabled=$permanentlyDisabled)")
        fileLogger.voiceEvent("ASR_SHUTDOWN", null, "[voice processor shut down]", "ASR")
    }
}

/**
 * The registered NOVA commands (26 + UNKNOWN).
 */
enum class NovaCommand {
    DESCRIBE_FRONT,      // "What's in front of me?"
    DISTANCE_QUERY,      // "How far?"
    FIND_CHAIR,          // "Find a chair"
    FIND_DOOR,           // "Find the door"
    READ_TEXT,           // "Read text"
    SCAN_ROOM,           // "Scan room"
    LOAD_ROOM,           // "Load room"
    EMERGENCY,           // "Emergency" / "SOS" / "Call for help"
    CANCEL_EMERGENCY,    // "Cancel SOS"
    BATTERY,             // "Battery?"
    PAUSE,               // "Pause"
    RESUME,              // "Resume"
    SETTINGS,            // "Settings"
    DESCRIBE_ALL,        // "What's around me?"
    PEOPLE_NEARBY,       // "Are there people around me?"
    REPEAT,              // "Say again" — re-speak last phrase
    TIME_QUERY,          // "What time is it?"
    STATUS,              // "NOVA status" — battery + server + GPS readout
    GO_HOME,             // "Go home" — return to the navigation screen
    GO_BACK,             // "Go back" — leave the current sub-screen
    SPEAK_SLOWER,        // "Speak slower"
    SPEAK_FASTER,        // "Speak faster"
    HELP_COMMANDS,       // "Help" / "What can I say?"
    OPEN_HELP,           // "Open help" / "Help screen"
    OPEN_FINDER,         // "Open finder" / "Find object" — switches to Grounding DINO tab
    UPDATE_CONTACT,      // "Update contact"
    UNKNOWN              // Fallback
}
