package com.nova.assistant.ui.settings

import android.content.Intent
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.FileProvider
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.nova.assistant.data.local.SettingsDao
import com.nova.assistant.data.local.SettingsEntity
import com.nova.assistant.data.local.SettingsKeys
import com.nova.assistant.feedback.FeedbackOrchestrator
import com.nova.assistant.util.FileLogger
import androidx.compose.foundation.layout.heightIn
import dagger.hilt.android.lifecycle.HiltViewModel
import com.nova.assistant.ml.DebugBundler
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import javax.inject.Inject

@HiltViewModel
class SettingsViewModel @Inject constructor(
    private val settingsDao: SettingsDao,
    private val feedback: FeedbackOrchestrator,
    private val fileLogger: FileLogger,
    private val bundler: DebugBundler
) : ViewModel() {

    var speechRate by mutableStateOf("normal")
    var vibration by mutableStateOf("strong")
    var warningDistance by mutableStateOf("2")
    var visionMode by mutableStateOf("partial")
    var userName by mutableStateOf("")
    var emergencyName by mutableStateOf("")
    var emergencyContact by mutableStateOf("")
    var aliveSignalInterval by mutableStateOf("5")

    // R14: Debounced jobs for text fields — prevents partial input from being persisted
    // while the user is still typing.
    private var emergencyContactSaveJob: kotlinx.coroutines.Job? = null
    private var emergencyNameSaveJob: kotlinx.coroutines.Job? = null
    private var userNameSaveJob: kotlinx.coroutines.Job? = null

    init { loadSettings() }

    fun loadSettings() {
        viewModelScope.launch {
            speechRate = settingsDao.get(SettingsKeys.SPEECH_RATE) ?: "normal"
            vibration = settingsDao.get(SettingsKeys.VIBRATION) ?: "strong"
            warningDistance = settingsDao.get(SettingsKeys.WARNING_DISTANCE) ?: "2"
            visionMode = settingsDao.get(SettingsKeys.VISION_MODE) ?: "partial"
            userName = settingsDao.get(SettingsKeys.USER_NAME) ?: ""
            emergencyName = settingsDao.get(SettingsKeys.EMERGENCY_NAME) ?: ""
            emergencyContact = settingsDao.get(SettingsKeys.EMERGENCY_CONTACT) ?: ""
            aliveSignalInterval = settingsDao.get(SettingsKeys.ALIVE_SIGNAL_INTERVAL) ?: "5"
        }
    }

    /** Debounced save for the user's own name (used in emergency SMS). */
    fun debounceSaveUserName(value: String) {
        userName = value
        userNameSaveJob?.cancel()
        userNameSaveJob = viewModelScope.launch {
            kotlinx.coroutines.delay(1500L)
            saveSetting(SettingsKeys.USER_NAME, value)
        }
    }

    /**
     * R6: Speak a brief orientation announcement when Settings is entered.
     * Blind users need spoken screen orientation — TalkBack focus doesn't announce
     * all available sections on entry.
     */
    fun announceOnEntry() {
        feedback.speakSystem(
            "Settings. " +
            "Available options: Speech rate, Vibration, Warning distance, " +
            "Vision mode, Alive signal, Your name, and Emergency contact."
        )
    }

    /**
     * R14: Debounced save for emergency contact number.
     * Waits 1.5 seconds after the last keystroke before persisting —
     * prevents partial numbers from being saved during rapid input.
     */
    fun debounceSaveContact(value: String) {
        emergencyContact = value
        emergencyContactSaveJob?.cancel()
        emergencyContactSaveJob = viewModelScope.launch {
            kotlinx.coroutines.delay(1500L)
            saveSetting(SettingsKeys.EMERGENCY_CONTACT, value)
        }
    }

    /**
     * R14: Debounced save for emergency contact name.
     */
    fun debounceSaveName(value: String) {
        emergencyName = value
        emergencyNameSaveJob?.cancel()
        emergencyNameSaveJob = viewModelScope.launch {
            kotlinx.coroutines.delay(1500L)
            saveSetting(SettingsKeys.EMERGENCY_NAME, value)
        }
    }

    fun saveSetting(key: String, value: String) {
        viewModelScope.launch {
            settingsDao.set(SettingsEntity(key, value))
            applyToFeedback(key, value)
            // R2: Speak confirmation so blind users know the change was registered.
            speakConfirmation(key, value)
        }
    }

    /**
     * R2: Spoken confirmation after every setting change.
     * R18: Warning distance includes actual threshold values + restart note.
     */
    private fun speakConfirmation(key: String, value: String) {
        val message = when (key) {
            SettingsKeys.SPEECH_RATE -> {
                val label = when (value) {
                    "slow" -> "slow"
                    "normal" -> "normal"
                    "fast" -> "fast"
                    else -> value
                }
                "Speech rate set to $label."
            }
            SettingsKeys.VIBRATION -> {
                val label = when (value) {
                    "off" -> "off"
                    "gentle" -> "gentle"
                    "strong" -> "strong"
                    else -> value
                }
                "Vibration set to $label."
            }
            SettingsKeys.WARNING_DISTANCE -> {
                // Spoken labels match the chip labels (Tight/Standard/Relaxed) from R12.
                // "restart" note removed — setting now applies within 30 seconds automatically.
                when (value) {
                    "1" -> "Warning distance set to Tight. Danger alert at 1 metre, warning at 2 metres."
                    "2" -> "Warning distance set to Standard. Danger alert at 1.5 metres, warning at 3 metres."
                    "3" -> "Warning distance set to Relaxed. Danger alert at 2 metres, warning at 4 metres."
                    else -> "Warning distance updated."
                }
            }
            SettingsKeys.USER_NAME -> {
                if (value.isBlank()) null else "Your name set to $value."
            }
            SettingsKeys.VISION_MODE -> {
                val label = if (value == "full_blind") "full blind" else "partial vision"
                "Vision mode set to $label."
            }
            SettingsKeys.ALIVE_SIGNAL_INTERVAL -> {
                when (value) {
                    "2"   -> "Alive signal set to every 2 minutes."
                    "5"   -> "Alive signal set to every 5 minutes."
                    "10"  -> "Alive signal set to every 10 minutes."
                    "off" -> "Alive signal turned off."
                    else  -> "Alive signal updated."
                }
            }
            else -> null
        }
        message?.let { feedback.speakSystem(it) }
    }

    private fun applyToFeedback(key: String, value: String) {
        when (key) {
            SettingsKeys.SPEECH_RATE -> {
                val rate = when (value) {
                    "slow" -> 0.7f
                    "normal" -> 1.0f
                    "fast" -> 1.3f
                    else -> 1.0f
                }
                feedback.setSpeechRate(rate)
            }
            SettingsKeys.VIBRATION -> {
                feedback.setVibrationEnabled(value != "off")
                val intensity = when (value) {
                    "gentle" -> 0.5f
                    "strong" -> 1.0f
                    else -> 1.0f
                }
                feedback.setVibrationIntensity(intensity)
            }
        }
    }

    fun testSpeech() {
        feedback.speakSystem("This is how NOVA will speak to you.")
    }

    /** Returns the log file path for display in the UI. */
    fun getLogFilePath(): String = fileLogger.getLogFilePath()

    /** Delete both log files and speak confirmation. */
    fun clearLogs() {
        fileLogger.clear()
        feedback.speakSystem("Log file cleared.")
    }

    var isBundling by mutableStateOf(false)
        private set

    /**
     * Zips log files into a plain bundle, then calls [onReady] with the resulting File
     * so the caller can open a share chooser. Bundling runs on IO; callbacks on Main.
     */
    fun shareLogs(onReady: (File) -> Unit, onError: () -> Unit) {
        if (isBundling) return
        isBundling = true
        viewModelScope.launch(Dispatchers.IO) {
            val file = bundler.createBundle()
            withContext(Dispatchers.Main) {
                isBundling = false
                if (file != null) onReady(file) else onError()
            }
        }
    }

    /**
     * Delete all source log files after a successful share.
     * Runs on IO. Delegates to [DebugBundler.clearSourceFiles] which verifies
     * the bundle is valid before deleting anything.
     */
    fun deleteSourceFiles(bundleFile: File) {
        viewModelScope.launch(Dispatchers.IO) {
            bundler.clearSourceFiles(bundleFile)
        }
    }
}

@Composable
fun SettingsScreen(
    onBack: () -> Unit,
    viewModel: SettingsViewModel = hiltViewModel()
) {
    // R6: Speak orientation announcement on screen entry so blind users know
    // what settings are available without navigating through all elements.
    LaunchedEffect(Unit) {
        viewModel.announceOnEntry()
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(24.dp)
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            OutlinedButton(
                onClick = onBack,
                modifier = Modifier
                    .heightIn(min = 56.dp)
                    .semantics { contentDescription = "Go back" }
            ) { Text("← Back") }
            Spacer(Modifier.width(16.dp))
            Text("Settings", fontSize = 28.sp, fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.primary)
        }
        Spacer(Modifier.height(24.dp))

        // R1: sectionName passed to ChoiceRow so each chip gets a full spoken description.
        SettingSection("Speech Rate") {
            ChoiceRow(
                sectionName = "Speech rate",
                options = listOf(
                    Triple("slow",   "Slow",   "Speech rate: Slow."),
                    Triple("normal", "Normal", "Speech rate: Normal."),
                    Triple("fast",   "Fast",   "Speech rate: Fast.")
                ),
                selected = viewModel.speechRate,
                onSelect = {
                    viewModel.speechRate = it
                    viewModel.saveSetting(SettingsKeys.SPEECH_RATE, it)
                }
            )
            // R1: contentDescription for Test Speech button
            OutlinedButton(
                onClick = { viewModel.testSpeech() },
                modifier = Modifier
                    .padding(top = 8.dp)
                    .semantics { contentDescription = "Test current speech rate" }
            ) { Text("Test Speech") }
        }

        SettingSection("Vibration") {
            ChoiceRow(
                sectionName = "Vibration",
                options = listOf(
                    Triple("off",    "Off",    "Vibration: Off."),
                    Triple("gentle", "Gentle", "Vibration: Gentle."),
                    Triple("strong", "Strong", "Vibration: Strong.")
                ),
                selected = viewModel.vibration,
                onSelect = {
                    viewModel.vibration = it
                    viewModel.saveSetting(SettingsKeys.VIBRATION, it)
                }
            )
        }

        // R12: Labels now describe the DANGER threshold (what triggers the strongest alert),
        // not a vague "near/medium/far" that misled users about which distance zone
        // the label referred to.
        SettingSection("Warning Distance") {
            ChoiceRow(
                sectionName = "Warning distance",
                options = listOf(
                    Triple("1", "Tight (danger at 1 m)",    "Warning distance: Tight. Danger alert at 1 metre, warning at 2 metres."),
                    Triple("2", "Standard (danger at 1.5 m)","Warning distance: Standard. Danger alert at 1.5 metres, warning at 3 metres."),
                    Triple("3", "Relaxed (danger at 2 m)",   "Warning distance: Relaxed. Danger alert at 2 metres, warning at 4 metres.")
                ),
                selected = viewModel.warningDistance,
                onSelect = {
                    viewModel.warningDistance = it
                    viewModel.saveSetting(SettingsKeys.WARNING_DISTANCE, it)
                }
            )
        }

        SettingSection("Vision Mode") {
            ChoiceRow(
                sectionName = "Vision mode",
                options = listOf(
                    Triple("full_blind", "Full Blind",     "Vision mode: Full Blind. Screen stays off, audio only."),
                    Triple("partial",    "Partial Vision", "Vision mode: Partial Vision. Camera preview is shown.")
                ),
                selected = viewModel.visionMode,
                onSelect = {
                    viewModel.visionMode = it
                    viewModel.saveSetting(SettingsKeys.VISION_MODE, it)
                }
            )
        }

        // R10: Alive signal interval — configurable so the user can avoid the
        // 5-minute announcement in meetings, crowded spaces, or when using NOVA
        // alongside a caregiver who already provides verbal confirmation.
        SettingSection("Alive Signal Interval") {
            ChoiceRow(
                sectionName = "Alive signal",
                options = listOf(
                    Triple("2",   "Every 2 min",  "Alive signal: Every 2 minutes."),
                    Triple("5",   "Every 5 min",  "Alive signal: Every 5 minutes. Default."),
                    Triple("10",  "Every 10 min", "Alive signal: Every 10 minutes."),
                    Triple("off", "Off",           "Alive signal: Off. NOVA will not announce itself periodically.")
                ),
                selected = viewModel.aliveSignalInterval,
                onSelect = {
                    viewModel.aliveSignalInterval = it
                    viewModel.saveSetting(SettingsKeys.ALIVE_SIGNAL_INTERVAL, it)
                }
            )
        }

        SettingSection("Your Name") {
            OutlinedTextField(
                value = viewModel.userName,
                onValueChange = { viewModel.debounceSaveUserName(it) },
                label = { Text("Your Name") },
                placeholder = { Text("Used in emergency SMS") },
                modifier = Modifier
                    .fillMaxWidth()
                    .semantics { contentDescription = "Your name, used in emergency alert messages" }
            )
        }

        SettingSection("Emergency Contact") {
            // R14: debounceSaveName waits 1.5 seconds after last keystroke before
            // persisting — prevents a half-typed name from being used in an SOS SMS.
            OutlinedTextField(
                value = viewModel.emergencyName,
                onValueChange = { viewModel.debounceSaveName(it) },
                label = { Text("Name") },
                modifier = Modifier
                    .fillMaxWidth()
                    .semantics { contentDescription = "Emergency contact name" }
            )
            Spacer(Modifier.height(8.dp))
            // R14: debounceSaveContact waits 1.5 seconds — prevents a partial phone
            // number (e.g. "+92" mid-entry) from being saved and triggering a bad SOS SMS.
            OutlinedTextField(
                value = viewModel.emergencyContact,
                onValueChange = { viewModel.debounceSaveContact(it) },
                label = { Text("Phone Number") },
                modifier = Modifier
                    .fillMaxWidth()
                    .semantics { contentDescription = "Emergency contact phone number" }
            )
        }

        // Debug Logs section — bundle log files and share them
        val context = LocalContext.current
        SettingSection("Debug Logs") {

            Row(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                modifier = Modifier.fillMaxWidth()
            ) {
                Button(
                    onClick = {
                        viewModel.shareLogs(
                            onReady = { file ->
                                try {
                                    val uri = FileProvider.getUriForFile(
                                        context,
                                        "${context.packageName}.provider",
                                        file
                                    )
                                    val intent = Intent(Intent.ACTION_SEND).apply {
                                        type = "application/octet-stream"
                                        putExtra(Intent.EXTRA_STREAM, uri)
                                        putExtra(Intent.EXTRA_SUBJECT, "NOVA Debug Logs")
                                        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                                    }
                                    // Launch chooser first — file must be accessible during share.
                                    // Source files are deleted only after the chooser is handed off.
                                    context.startActivity(Intent.createChooser(intent, "Share NOVA logs"))
                                    // Bundle is now in the chooser's hands. Wipe source files —
                                    // the bundle ZIP is the definitive copy from this point.
                                    viewModel.deleteSourceFiles(file)
                                } catch (e: Exception) {
                                    android.util.Log.e("SettingsScreen", "Share logs failed", e)
                                    // Do NOT delete sources if the chooser failed to launch.
                                }
                            },
                            onError = {
                                android.util.Log.e("SettingsScreen", "Bundle creation failed — check log")
                            }
                        )
                    },
                    enabled = !viewModel.isBundling,
                    modifier = Modifier
                        .weight(1f)
                        .heightIn(min = 48.dp)
                        .semantics { contentDescription = "Share logs" }
                ) {
                    Text(if (viewModel.isBundling) "Bundling…" else "Share Logs")
                }
                OutlinedButton(
                    onClick = { viewModel.clearLogs() },
                    modifier = Modifier
                        .heightIn(min = 48.dp)
                        .semantics { contentDescription = "Clear log file" }
                ) {
                    Text("Clear")
                }
            }
        }

        SettingSection("About NOVA") {
            Text("Version: 1.0.0-mvp", fontSize = 14.sp,
                color = MaterialTheme.colorScheme.onBackground)
            Text("Built by: Hadiya, Shikaib, Abdullah", fontSize = 14.sp,
                color = MaterialTheme.colorScheme.onBackground)
            Text("FYP — BS Data Science", fontSize = 14.sp,
                color = MaterialTheme.colorScheme.onBackground)
        }
    }
}

@Composable
private fun SettingSection(
    title: String,
    content: @Composable ColumnScope.() -> Unit
) {
    Column(modifier = Modifier
        .fillMaxWidth()
        .padding(vertical = 12.dp)) {
        Text(
            title,
            fontSize = 18.sp,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.primary
        )
        Spacer(Modifier.height(8.dp))
        content()
        Spacer(Modifier.height(8.dp))
        @Suppress("DEPRECATION")
        Divider(color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.2f))
    }
}

/**
 * R1: Accessibility-enhanced choice row.
 * Each chip receives a full spoken description (section + value + selection state)
 * plus a liveRegion so TalkBack announces changes without focus movement.
 *
 * @param sectionName  Human-readable section (e.g. "Speech rate") — used in chip descriptions.
 * @param options      Triple of (value, displayLabel, fullSpokenDescription).
 * @param selected     Currently selected value.
 * @param onSelect     Called with the value when a chip is tapped.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ChoiceRow(
    @Suppress("UNUSED_PARAMETER") sectionName: String,
    options: List<Triple<String, String, String>>,
    selected: String,
    onSelect: (String) -> Unit
) {
    Row(
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        modifier = Modifier.fillMaxWidth()
    ) {
        options.forEach { (value, label, description) ->
            val isSelected = selected == value
            // R1: full spoken description including selection state; liveRegion so
            // TalkBack announces after double-tap without the user re-focusing.
            FilterChip(
                selected = isSelected,
                onClick = { onSelect(value) },
                label = { Text(label) },
                modifier = Modifier
                    .height(48.dp)
                    .semantics {
                        contentDescription = if (isSelected)
                            "$description Selected."
                        else
                            "$description Double-tap to activate."
                        liveRegion = LiveRegionMode.Polite
                    }
            )
        }
    }
}
