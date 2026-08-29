package com.nova.assistant.ui.splash

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import android.app.ActivityManager
import android.content.Context
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.nova.assistant.data.local.SettingsDao
import com.nova.assistant.data.local.SettingsEntity
import com.nova.assistant.data.local.SettingsKeys
import com.nova.assistant.engine.InferencePipeline
import com.nova.assistant.feedback.FeedbackOrchestrator
import com.nova.assistant.util.NovaConstants
import com.nova.assistant.util.PermissionHelper
import dagger.hilt.android.lifecycle.HiltViewModel
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.FileNotFoundException
import javax.inject.Inject

// ══════════════════════════════════════════════════════════════════
// Run4 model is ACTIVE — NOVA_CLASSES_RUN4 (28 classes) is the live class list.
// NOVA_CLASSES_COCO is kept for local COCO smoke-tests (set USE_COCO_TEST_CLASSES=true).
// NOVA_CLASSES_CUSTOM (Run2, 29 classes) is DEPRECATED — see its warning above.
// ══════════════════════════════════════════════════════════════════

// DEPRECATED — DANGEROUS: Do NOT pass to YoloDetector when Run4 model is active.
// This is the Run2 29-class list. It includes "building" at index 2, which shifts
// ALL class IDs ≥ 2 by one relative to Run4. Activating this with the Run4 model
// causes every class from "bus" onward to be misidentified (e.g. bus→car, pothole→ramp).
/** ── RUN2 (inactive): 29 classes trained on Roboflow project-nova-merged-d97ey V3.
 * DO NOT reorder or remove entries — class IDs are baked into the TFLite model weights.
 * The "vegitation" typo is INTENTIONAL — matches training labels.
 */
val NOVA_CLASSES_CUSTOM = listOf(
    "bench",             // 0
    "bicycle",           // 1
    "building",          // 2  — context-only; excluded from Run4
    "bus",               // 3
    "car",               // 4
    "chair",             // 5
    "computer",          // 6
    "curb",              // 7
    "door",              // 8
    "empty-chair",       // 9
    "fence",             // 10
    "fire-extinguisher", // 11
    "occupied-chair",    // 12
    "person",            // 13
    "pole",              // 14
    "pot-plant",         // 15
    "pothole",           // 16
    "projector",         // 17
    "ramp",              // 18
    "sofa",              // 19
    "speed-breaker",     // 20
    "stairs",            // 21
    "table",             // 22
    "tree",              // 23
    "truck",             // 24
    "vegitation",        // 25
    "wall",              // 26
    "water dispenser",   // 27
    "white-board"        // 28
)

/** ── RUN4 (active): 28 classes — building removed (context-only, zero navigation value).
 * Also drops phantom classes (motorcycle, traffic-light, stop-sign) that were never
 * in any Roboflow project and never appeared in Run2 training data.
 */
val NOVA_CLASSES_RUN4 = listOf(
    "bench",             // 0
    "bicycle",           // 1
    "bus",               // 2
    "car",               // 3
    "chair",             // 4
    "computer",          // 5
    "curb",              // 6
    "door",              // 7
    "empty-chair",       // 8
    "fence",             // 9
    "fire-extinguisher", // 10
    "occupied-chair",    // 11
    "person",            // 12
    "pole",              // 13
    "pot-plant",         // 14
    "pothole",           // 15
    "projector",         // 16
    "ramp",              // 17
    "sofa",              // 18
    "speed-breaker",     // 19
    "stairs",            // 20
    "table",             // 21
    "tree",              // 22
    "truck",             // 23
    "vegitation",        // 24
    "wall",              // 25
    "water dispenser",   // 26
    "white-board"        // 27
)

/** ── RUN6-ALT (18 classes): 28→18 remap. Order is AUTHORITATIVE — it must match the model's
 * class indices exactly. Copied verbatim from data.yaml `names` in
 * kaggle-notebooks/nova_training_run6_alt.py (NOVA_CLASSES). DO NOT reorder — indices are baked
 * into the trained weights. empty-chair + occupied-chair were merged into "chair" (idx 4);
 * fence/wall/pole/tree/vegitation/sofa/projector/fire-extinguisher were dropped.
 * Activated by NovaConstants.USE_RUN6_ALT (see below).
 */
val NOVA_CLASSES_RUN6_ALT = listOf(
    "bench",            // 0
    "bicycle",          // 1
    "bus",              // 2
    "car",              // 3
    "chair",            // 4  (absorbs empty-chair + occupied-chair)
    "computer",         // 5
    "curb",             // 6
    "door",             // 7
    "person",           // 8
    "pot-plant",        // 9
    "pothole",          // 10
    "ramp",             // 11
    "speed-breaker",    // 12
    "stairs",           // 13
    "table",            // 14
    "truck",            // 15
    "water dispenser",  // 16
    "white-board"       // 17
)

/**
 * TEST MODE — COCO 80 classes for pre-built yolo26s.pt from Ultralytics.
 * Order matches official COCO class indices exactly (0–79).
 * Classes useful for NOVA navigation: person(0), bicycle(1), car(2),
 * bus(5), truck(7), bench(13), chair(56), couch(57), dining table(60).
 */
val NOVA_CLASSES_COCO = listOf(
    "person",           // 0
    "bicycle",          // 1
    "car",              // 2
    "motorcycle",       // 3
    "airplane",         // 4
    "bus",              // 5
    "train",            // 6
    "truck",            // 7
    "boat",             // 8
    "traffic light",    // 9
    "fire hydrant",     // 10
    "stop sign",        // 11
    "parking meter",    // 12
    "bench",            // 13
    "bird",             // 14
    "cat",              // 15
    "dog",              // 16
    "horse",            // 17
    "sheep",            // 18
    "cow",              // 19
    "elephant",         // 20
    "bear",             // 21
    "zebra",            // 22
    "giraffe",          // 23
    "backpack",         // 24
    "umbrella",         // 25
    "handbag",          // 26
    "tie",              // 27
    "suitcase",         // 28
    "frisbee",          // 29
    "skis",             // 30
    "snowboard",        // 31
    "sports ball",      // 32
    "kite",             // 33
    "baseball bat",     // 34
    "baseball glove",   // 35
    "skateboard",       // 36
    "surfboard",        // 37
    "tennis racket",    // 38
    "bottle",           // 39
    "wine glass",       // 40
    "cup",              // 41
    "fork",             // 42
    "knife",            // 43
    "spoon",            // 44
    "bowl",             // 45
    "banana",           // 46
    "apple",            // 47
    "sandwich",         // 48
    "orange",           // 49
    "broccoli",         // 50
    "carrot",           // 51
    "hot dog",          // 52
    "pizza",            // 53
    "donut",            // 54
    "cake",             // 55
    "chair",            // 56
    "couch",            // 57
    "potted plant",     // 58
    "bed",              // 59
    "dining table",     // 60
    "toilet",           // 61
    "tv",               // 62
    "laptop",           // 63
    "mouse",            // 64
    "remote",           // 65
    "keyboard",         // 66
    "cell phone",       // 67
    "microwave",        // 68
    "oven",             // 69
    "toaster",          // 70
    "sink",             // 71
    "refrigerator",     // 72
    "book",             // 73
    "clock",            // 74
    "vase",             // 75
    "scissors",         // 76
    "teddy bear",       // 77
    "hair drier",       // 78
    "toothbrush"        // 79
)

// ── ACTIVE: selected by NovaConstants.USE_RUN6_ALT (single source of truth, in Constants.kt) ──
// Run4 (default):  NOVA_CLASSES_RUN4     — 28 classes, mAP50=0.646
// Run6-alt:        NOVA_CLASSES_RUN6_ALT — 18 classes (flip USE_RUN6_ALT=true AND drop the .tflite)
// The E2E model emits a single class_id float (not per-class logits), so switching this list +
// the asset file is ALL the on-device change needed — there is NO YoloDetector tensor-slice edit
// (that only applies to raw-anchor [1,8400,N] models, which NOVA does not run on-device).
val NOVA_CLASSES = if (NovaConstants.USE_RUN6_ALT) NOVA_CLASSES_RUN6_ALT else NOVA_CLASSES_RUN4

@HiltViewModel
class SplashViewModel @Inject constructor(
    private val pipeline: InferencePipeline,   // Single entry point — no dual-init GPU leak
    private val feedback: FeedbackOrchestrator,
    private val settingsDao: SettingsDao,
    private val permissionHelper: PermissionHelper,
    @ApplicationContext private val appContext: Context
) : ViewModel() {

    private val _state = androidx.compose.runtime.mutableStateOf<SplashState>(SplashState.Loading)
    val state: androidx.compose.runtime.State<SplashState> = _state

    fun initialize() {
        _state.value = SplashState.Loading
        viewModelScope.launch {
            try {
                feedback.initialize()
                delay(500)
                feedback.speakSystem("NOVA is starting. Please stand still.")

                // Route all model loading through InferencePipeline — single entry point.
                // This prevents the GPU delegate from being allocated twice (once here,
                // once when MainNavigationViewModel creates NovaFrameAnalyzer).
                withContext(Dispatchers.Default) {
                    pipeline.initialize(NOVA_CLASSES)
                }

                // Check permissions
                if (!permissionHelper.allPermissionsGranted()) {
                    permissionHelper.announceMissingPermissions()
                }

                // Decision 3: One-time performance warning on marginal devices.
                // Shown once — stored in DB so it never appears again after dismissal.
                // A VIP is never hard-blocked: they may not have another device available.
                // Uses ActivityManager.MemoryInfo.totalMem (physical RAM) — NOT Runtime.maxMemory()
                // which returns the Dalvik heap growth limit (~192–512 MB on all devices).
                val perfWarned = settingsDao.get(SettingsKeys.DEVICE_PERF_WARNED) == "true"
                if (!perfWarned) {
                    val am = appContext.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
                    val memInfo = ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }
                    val totalRamMb = memInfo.totalMem / (1024L * 1024L)
                    if (totalRamMb < 3072L) {   // warn on devices with < 3 GB physical RAM
                        _state.value = SplashState.PerformanceWarning(totalRamMb)
                        return@launch
                    }
                }

                // Prompt user to disable battery optimization before proceeding.
                // On EMUI, Android kills background services aggressively — this
                // also keeps the GPU OpenCL context alive between frames.
                if (!permissionHelper.isIgnoringBatteryOptimizations()) {
                    _state.value = SplashState.BatteryOptimizationNeeded
                    return@launch
                }

                navigateToReady()

            } catch (e: FileNotFoundException) {
                // M4: Speak the error — a blind user can't read the on-screen message.
                feedback.speakSystem(
                    "NOVA could not start. AI model files are missing. Please reinstall the app."
                )
                _state.value = SplashState.Error(
                    message = "Model files not found.",
                    detail = "Copy nova_yolo_run6_alt_fp16.tflite and midas_small_fp16_true.tflite " +
                             "into app/src/main/assets/ then tap Retry.",
                    isModelMissing = true
                )
            } catch (e: Exception) {
                // M4: Generic startup failure — give actionable spoken guidance.
                feedback.speakSystem(
                    "NOVA could not start due to an error. Please restart the app."
                )
                _state.value = SplashState.Error(
                    message = e.message ?: "Initialization failed",
                    detail = e.javaClass.simpleName
                )
            }
        }
    }

    /** Called when user dismisses the performance warning. Stores flag, continues normal flow. */
    fun proceedAfterPerformanceWarning() {
        viewModelScope.launch {
            settingsDao.set(SettingsEntity(SettingsKeys.DEVICE_PERF_WARNED, "true"))
            if (!permissionHelper.isIgnoringBatteryOptimizations()) {
                _state.value = SplashState.BatteryOptimizationNeeded
            } else {
                navigateToReady()
            }
        }
    }

    /** Called when user taps "Open Battery Settings" or "Skip" on the battery optimization screen. */
    fun proceedAfterBatteryPrompt() {
        viewModelScope.launch { navigateToReady() }
    }

    /** Opens the system battery optimization dialog for NOVA's package. */
    fun openBatteryOptimizationSettings() {
        permissionHelper.openBatteryOptimizationSettings()
    }

    private suspend fun navigateToReady() {
        feedback.speakSystem("NOVA is ready.")
        val isSetupComplete = settingsDao.get(SettingsKeys.SETUP_COMPLETE) == "true"
        _state.value = SplashState.Ready(isFirstTime = !isSetupComplete)
    }

    /** Retry initialization — used by the Retry button in SplashState.Error UI. */
    fun retry() = initialize()
}

sealed class SplashState {
    data object Loading : SplashState()
    data object BatteryOptimizationNeeded : SplashState()
    data class PerformanceWarning(val ramMb: Long) : SplashState()
    data class Ready(val isFirstTime: Boolean) : SplashState()
    data class Error(
        val message: String,
        val detail: String = "",
        val isModelMissing: Boolean = false
    ) : SplashState()
}

@Composable
fun SplashScreen(
    onReady: (isFirstTime: Boolean) -> Unit,
    viewModel: SplashViewModel = hiltViewModel()
) {
    LaunchedEffect(Unit) {
        viewModel.initialize()
    }

    val state by viewModel.state

    Box(
        modifier = Modifier
            .fillMaxSize()
            .padding(32.dp),
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(24.dp)
        ) {
            Text(
                text = "NOVA",
                fontSize = 72.sp,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.primary
            )
            Text(
                text = "Navigation and Object Vision Assistant",
                fontSize = 16.sp,
                color = MaterialTheme.colorScheme.onBackground,
                textAlign = TextAlign.Center
            )

            Spacer(modifier = Modifier.height(48.dp))

            when (val s = state) {
                is SplashState.Loading -> {
                    // R14: contentDescription so TalkBack announces the progress state,
                    // not just "Progress bar" with no context.
                    CircularProgressIndicator(
                        modifier = Modifier
                            .size(64.dp)
                            .semantics { contentDescription = "Loading NOVA, please wait" },
                        color = MaterialTheme.colorScheme.primary,
                        strokeWidth = 6.dp
                    )
                    // R14: liveRegion so TalkBack announces this text when it appears.
                    Text(
                        "Loading NOVA...",
                        fontSize = 18.sp,
                        color = MaterialTheme.colorScheme.onBackground,
                        modifier = Modifier.semantics { liveRegion = LiveRegionMode.Polite }
                    )
                }
                is SplashState.PerformanceWarning -> {
                    Text(
                        "Performance Notice",
                        fontSize = 22.sp,
                        fontWeight = FontWeight.Bold,
                        textAlign = TextAlign.Center,
                        color = MaterialTheme.colorScheme.primary
                    )
                    Spacer(Modifier.height(8.dp))
                    Text(
                        "Your device has limited RAM (${s.ramMb} MB). " +
                        "NOVA will work but may respond slower than expected on this hardware.\n\n" +
                        "For best results: close other apps, keep the screen on, and ensure battery optimization is disabled.",
                        fontSize = 15.sp,
                        textAlign = TextAlign.Center,
                        color = MaterialTheme.colorScheme.onBackground
                    )
                    Spacer(Modifier.height(16.dp))
                    Button(
                        onClick = { viewModel.proceedAfterPerformanceWarning() },
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text("Understood, continue", fontSize = 18.sp)
                    }
                }
                is SplashState.BatteryOptimizationNeeded -> {
                    Text(
                        "Battery Optimization",
                        fontSize = 22.sp,
                        fontWeight = FontWeight.Bold,
                        textAlign = TextAlign.Center,
                        color = MaterialTheme.colorScheme.primary
                    )
                    Spacer(Modifier.height(8.dp))
                    Text(
                        "NOVA needs unrestricted background access to keep detecting obstacles " +
                        "when your screen turns off or you switch apps.\n\n" +
                        "Disabling battery optimization also prevents Android from killing the " +
                        "camera service and GPU acceleration mid-session.",
                        fontSize = 15.sp,
                        textAlign = TextAlign.Center,
                        color = MaterialTheme.colorScheme.onBackground
                    )
                    Spacer(Modifier.height(16.dp))
                    Button(
                        onClick = {
                            viewModel.openBatteryOptimizationSettings()
                            viewModel.proceedAfterBatteryPrompt()
                        },
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text("Open Battery Settings", fontSize = 18.sp)
                    }
                    Spacer(Modifier.height(8.dp))
                    OutlinedButton(
                        onClick = { viewModel.proceedAfterBatteryPrompt() },
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text("Skip (Not Recommended)", fontSize = 16.sp)
                    }
                }
                is SplashState.Ready -> {
                    LaunchedEffect(s) {
                        delay(500)
                        onReady(s.isFirstTime)
                    }
                    Text(
                        "Ready",
                        fontSize = 24.sp,
                        color = MaterialTheme.colorScheme.primary,
                        fontWeight = FontWeight.Bold
                    )
                }
                is SplashState.Error -> {
                    // R9: Plain text replaces ⚠️ emoji — Huawei TalkBack renders emoji
                    // inconsistently (garbled or silent). contentDescription provides
                    // the full accessible label for screen readers.
                    Text(
                        text = if (s.isModelMissing) "Error: Model files not found" else "Error: ${s.message}",
                        fontSize = 20.sp,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.error,
                        textAlign = TextAlign.Center,
                        modifier = Modifier.semantics {
                            contentDescription = if (s.isModelMissing)
                                "Startup error: Model files not found."
                            else
                                "Startup error: ${s.message}"
                        }
                    )
                    if (s.detail.isNotBlank()) {
                        Text(
                            text = s.detail,
                            fontSize = 14.sp,
                            color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.75f),
                            textAlign = TextAlign.Center
                        )
                    }
                    Spacer(Modifier.height(8.dp))
                    Button(onClick = { viewModel.retry() }) {
                        Text("Retry", fontSize = 18.sp)
                    }
                }
            }
        }
    }
}
