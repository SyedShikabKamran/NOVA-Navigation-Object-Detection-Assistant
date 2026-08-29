package com.nova.assistant.ui.setup

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.text.KeyboardOptions
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
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.nova.assistant.data.local.SettingsDao
import com.nova.assistant.data.local.SettingsEntity
import com.nova.assistant.data.local.SettingsKeys
import com.nova.assistant.feedback.FeedbackOrchestrator
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.launch
import javax.inject.Inject

// ─────────────────────────────────────────────────────────────────────────────
// ViewModel
// ─────────────────────────────────────────────────────────────────────────────

@HiltViewModel
class SetupViewModel @Inject constructor(
    private val settingsDao: SettingsDao,
    private val feedback: FeedbackOrchestrator
) : ViewModel() {

    /**
     * Called when the visible step changes — speaks the step's prompt.
     * Setup is text-entry only (voice capture removed here; it stays available
     * everywhere else in the app, e.g. the "Update my emergency contact" voice flow).
     */
    fun beginStep(step: Int) {
        val prompt = when (step) {
            0 -> "What is the emergency phone number? Type it below."
            1 -> "Step 2 of 2. What is the name of this contact? Type it below."
            else -> return
        }
        feedback.speakSystem(prompt)
    }

    fun saveSetup(
        emergencyContact: String,
        emergencyName: String,
        onComplete: () -> Unit
    ) {
        viewModelScope.launch {
            settingsDao.set(SettingsEntity(SettingsKeys.USER_NAME, "User"))
            settingsDao.set(SettingsEntity(SettingsKeys.VISION_MODE, "partial"))
            settingsDao.set(SettingsEntity(SettingsKeys.EMERGENCY_NAME, emergencyName))
            settingsDao.set(SettingsEntity(SettingsKeys.EMERGENCY_CONTACT, emergencyContact))
            settingsDao.set(SettingsEntity(SettingsKeys.SPEECH_RATE, "normal"))
            settingsDao.set(SettingsEntity(SettingsKeys.VIBRATION, "strong"))
            settingsDao.set(SettingsEntity(SettingsKeys.WARNING_DISTANCE, "2"))
            settingsDao.set(SettingsEntity(SettingsKeys.SETUP_COMPLETE, "true"))

            feedback.speakSystem(
                "Setup complete. Emergency contact saved as $emergencyName. " +
                        "NOVA is ready. Let's start navigating safely."
            )
            onComplete()
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Composable
// ─────────────────────────────────────────────────────────────────────────────

@Composable
fun SetupScreen(
    onComplete: () -> Unit,
    viewModel: SetupViewModel = hiltViewModel()
) {
    var step by remember { mutableIntStateOf(0) }
    var emergencyContact by remember { mutableStateOf("") }
    var emergencyName by remember { mutableStateOf("") }

    // Speak each step's prompt when it becomes visible (output only — no mic capture).
    LaunchedEffect(step) {
        viewModel.beginStep(step)
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Black)
            .padding(32.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {

        // ── Header ────────────────────────────────────────────────────────
        Text(
            "NOVA Setup",
            fontSize = 28.sp,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.primary
        )
        Spacer(Modifier.height(8.dp))
        // A1: liveRegion.Assertive so TalkBack announces the step change immediately
        // when the user advances from step 1 to step 2, without requiring focus movement.
        Text(
            if (step == 0) "Step 1 of 2  •  Emergency number"
            else           "Step 2 of 2  •  Contact name",
            fontSize = 13.sp,
            color = Color.White.copy(alpha = 0.55f),
            modifier = Modifier.semantics {
                liveRegion = LiveRegionMode.Assertive
                contentDescription = if (step == 0)
                    "Step 1 of 2. Emergency number."
                else
                    "Step 2 of 2. Contact name."
            }
        )
        Spacer(Modifier.height(40.dp))

        // ── Question ──────────────────────────────────────────────────────
        val question = if (step == 0)
            "What is the emergency phone number?"
        else
            "What is the name of this contact?"

        Text(
            question,
            fontSize = 20.sp,
            fontWeight = FontWeight.Medium,
            color = Color.White,
            textAlign = TextAlign.Center
        )
        Spacer(Modifier.height(40.dp))

        // ── Text entry — the only input method during setup ──────────────
        val label = if (step == 0) "Phone number" else "Contact name"
        val value = if (step == 0) emergencyContact else emergencyName
        val keyboardType = if (step == 0) KeyboardType.Phone else KeyboardType.Text

        OutlinedTextField(
            value = value,
            onValueChange = { new ->
                if (step == 0) emergencyContact = new else emergencyName = new
            },
            label = { Text(label, color = Color.White.copy(alpha = 0.7f)) },
            keyboardOptions = KeyboardOptions(keyboardType = keyboardType),
            singleLine = true,
            colors = OutlinedTextFieldDefaults.colors(
                focusedTextColor = Color.White,
                unfocusedTextColor = Color.White,
                focusedBorderColor = MaterialTheme.colorScheme.primary,
                unfocusedBorderColor = Color.White.copy(alpha = 0.3f)
            ),
            modifier = Modifier
                .fillMaxWidth()
                .semantics { contentDescription = "$label text input" }
        )

        Spacer(Modifier.height(24.dp))

        Button(
            onClick = {
                if (step == 0) step = 1
                else viewModel.saveSetup(emergencyContact, emergencyName) { onComplete() }
            },
            enabled = value.isNotBlank(),
            modifier = Modifier
                .fillMaxWidth()
                .height(64.dp)
                .semantics { contentDescription = if (step == 0) "Next" else "Finish setup" },
            colors = ButtonDefaults.buttonColors(
                containerColor = MaterialTheme.colorScheme.primary
            )
        ) {
            Text(if (step == 0) "Next" else "Finish", fontSize = 18.sp)
        }

        // Back button on step 2
        if (step > 0) {
            Spacer(Modifier.height(8.dp))
            TextButton(
                onClick = { step = 0 },
                modifier = Modifier
                    .fillMaxWidth()
                    .semantics { contentDescription = "Go back to phone number" }
            ) {
                Text("Back", color = Color.White.copy(alpha = 0.5f), fontSize = 16.sp)
            }
        }
    }
}
