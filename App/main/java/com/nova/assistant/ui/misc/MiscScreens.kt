package com.nova.assistant.ui.misc

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.nova.assistant.data.local.RoomSnapshotDao
import com.nova.assistant.data.local.RoomSnapshotEntity
import com.nova.assistant.feedback.FeedbackOrchestrator
import com.nova.assistant.features.RoomSnapshotEngine
import com.nova.assistant.sensors.NovaSensorManager
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.launch
import org.json.JSONArray
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import javax.inject.Inject

// ════════════════════════════════════════════════════════════
// ROOMS VIEW MODEL
// ════════════════════════════════════════════════════════════
@HiltViewModel
class SavedRoomsViewModel @Inject constructor(
    private val roomDao: RoomSnapshotDao,
    private val roomEngine: RoomSnapshotEngine,
    private val sensorManager: NovaSensorManager,
    private val feedback: FeedbackOrchestrator
) : ViewModel() {

    val rooms: Flow<List<RoomSnapshotEntity>> = roomDao.getAll()

    fun loadRoom(room: RoomSnapshotEntity) {
        viewModelScope.launch {
            val heading = sensorManager.getCurrentSensorData().azimuthDeg
            roomEngine.loadRoomById(room, heading)
        }
    }

    fun deleteRoom(room: RoomSnapshotEntity) {
        viewModelScope.launch {
            roomDao.deleteById(room.id)
            feedback.speakSystem("Room ${room.roomName} deleted.")
        }
    }

    fun renameRoom(id: Long, name: String) {
        viewModelScope.launch {
            roomDao.renameById(id, name)
            feedback.speakSystem("Room renamed to $name.")
        }
    }

    fun requestScan(name: String) {
        roomEngine.pendingScanName.value = name.ifBlank { null }
    }

    fun announceEmptyState() {
        feedback.speakSystem(
            "Saved rooms. No rooms saved yet. Say Scan room to save your first room."
        )
    }
}

// ════════════════════════════════════════════════════════════
// MEMORY SCREEN (Rooms)
// ════════════════════════════════════════════════════════════
@Composable
fun SavedRoomsScreen(
    onBack: () -> Unit,
    roomsViewModel: SavedRoomsViewModel = hiltViewModel(),
) {
    Column(modifier = Modifier.fillMaxSize().padding(horizontal = 24.dp, vertical = 16.dp)) {
        // Header row
        Row(verticalAlignment = Alignment.CenterVertically) {
            OutlinedButton(
                onClick = onBack,
                modifier = Modifier.heightIn(min = 56.dp).semantics { contentDescription = "Go back" }
            ) { Text("← Back") }
            Spacer(Modifier.width(16.dp))
            Text(
                "Memory",
                fontSize = 28.sp,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.primary,
                modifier = Modifier.weight(1f)
            )
        }

        Spacer(Modifier.height(12.dp))

        RoomsTab(viewModel = roomsViewModel, onBack = onBack)
    }
}

// ────────────────────────────────────────────────────────────
// Rooms tab
// ────────────────────────────────────────────────────────────
@Composable
private fun RoomsTab(viewModel: SavedRoomsViewModel, onBack: () -> Unit) {
    val rooms by viewModel.rooms.collectAsState(initial = emptyList())
    var showScanDialog by remember { mutableStateOf(false) }
    var scanNameInput by remember { mutableStateOf("") }

    var emptyStateAnnounced by remember { mutableStateOf(false) }
    LaunchedEffect(rooms) {
        if (rooms.isEmpty() && !emptyStateAnnounced) {
            emptyStateAnnounced = true
            viewModel.announceEmptyState()
        }
    }

    if (showScanDialog) {
        AlertDialog(
            onDismissRequest = { showScanDialog = false; scanNameInput = "" },
            title = { Text("Room Name") },
            text = {
                OutlinedTextField(
                    value = scanNameInput,
                    onValueChange = { scanNameInput = it },
                    label = { Text("e.g. Living Room") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth().semantics { contentDescription = "Enter room name" }
                )
            },
            confirmButton = {
                Button(onClick = {
                    val name = scanNameInput.trim().ifBlank {
                        val ts = SimpleDateFormat("HH:mm", Locale.US).format(Date())
                        "Room $ts"
                    }
                    viewModel.requestScan(name)
                    showScanDialog = false
                    scanNameInput = ""
                    onBack()
                }) { Text("Start Scan") }
            },
            dismissButton = {
                OutlinedButton(onClick = { showScanDialog = false; scanNameInput = "" }) { Text("Cancel") }
            }
        )
    }

    Column(modifier = Modifier.fillMaxSize()) {
        Button(
            onClick = { showScanDialog = true },
            modifier = Modifier
                .fillMaxWidth()
                .heightIn(min = 56.dp)
                .semantics { contentDescription = "Scan new room" }
        ) {
            Icon(Icons.Default.Add, contentDescription = null, modifier = Modifier.size(18.dp))
            Spacer(Modifier.width(6.dp))
            Text("Scan Room")
        }

        Spacer(Modifier.height(12.dp))

        if (rooms.isEmpty()) {
            Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text("No rooms saved yet", fontSize = 18.sp,
                        color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.7f))
                    Spacer(Modifier.height(8.dp))
                    Text("Tap Scan Room or say \"Scan room\"",
                        fontSize = 14.sp,
                        color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.5f))
                }
            }
        } else {
            LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                items(rooms) { room ->
                    RoomCard(
                        room = room,
                        onTap = { viewModel.loadRoom(room) },
                        onDelete = { viewModel.deleteRoom(room) },
                        onRename = { newName -> viewModel.renameRoom(room.id, newName) }
                    )
                }
            }
        }
    }
}

@Composable
private fun RoomCard(
    room: RoomSnapshotEntity,
    onTap: () -> Unit,
    onDelete: () -> Unit,
    onRename: (String) -> Unit
) {
    val sdf = SimpleDateFormat("MMM dd, HH:mm", Locale.US)
    val dateStr = sdf.format(Date(room.createdAt))
    val objectCount = try { JSONArray(room.objectInventoryJson).length() } catch (e: Exception) { 0 }
    val cardDesc = "Room: ${room.roomName}, $objectCount object groups, saved $dateStr. Double-tap to load."

    var showRenameDialog by remember { mutableStateOf(false) }
    var renameInput by remember { mutableStateOf("") }

    if (showRenameDialog) {
        AlertDialog(
            onDismissRequest = { showRenameDialog = false; renameInput = "" },
            title = { Text("Rename Room") },
            text = {
                OutlinedTextField(
                    value = renameInput,
                    onValueChange = { renameInput = it },
                    label = { Text("New name") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth()
                )
            },
            confirmButton = {
                Button(onClick = {
                    val name = renameInput.trim().ifBlank { room.roomName }
                    onRename(name)
                    showRenameDialog = false
                    renameInput = ""
                }) { Text("Rename") }
            },
            dismissButton = {
                OutlinedButton(onClick = { showRenameDialog = false; renameInput = "" }) { Text("Cancel") }
            }
        )
    }

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onTap)
            .semantics { contentDescription = cardDesc },
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth().padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(room.roomName, fontSize = 18.sp, fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.onBackground)
                Text("$objectCount object groups · $dateStr",
                    fontSize = 13.sp,
                    color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f))
            }
            IconButton(
                onClick = { renameInput = room.roomName; showRenameDialog = true },
                modifier = Modifier.semantics { contentDescription = "Rename room ${room.roomName}" }
            ) {
                Icon(Icons.Default.Edit, contentDescription = null,
                    tint = MaterialTheme.colorScheme.primary, modifier = Modifier.size(22.dp))
            }
            IconButton(
                onClick = onDelete,
                modifier = Modifier.semantics { contentDescription = "Delete room ${room.roomName}" }
            ) {
                Icon(Icons.Default.Delete, contentDescription = null,
                    tint = MaterialTheme.colorScheme.error, modifier = Modifier.size(22.dp))
            }
        }
    }
}

// ════════════════════════════════════════════════════════════
// HELP SCREEN — Voice Command Reference
// ════════════════════════════════════════════════════════════
data class CommandHelp(val command: String, val description: String)

val COMMAND_LIST = listOf(
    CommandHelp("\"What's in front of me?\"", "Describes the closest object detected"),
    CommandHelp("\"What's around me?\"", "Describes all detected objects"),
    CommandHelp("\"Anyone around?\"", "Checks for people nearby"),
    CommandHelp("\"How far?\"", "Reports distance to nearest obstacle"),
    CommandHelp("\"Find a chair\"", "Locates the nearest chair"),
    CommandHelp("\"Find the door\"", "Locates the nearest door"),
    CommandHelp("\"Read text\"", "Reads any text visible to the camera"),
    CommandHelp("\"Scan room\"", "Records a room for 20 seconds, then save with a name"),
    CommandHelp("\"Load room\"", "Recall a previously scanned room"),
    CommandHelp("\"Help\" / \"Emergency\" / \"SOS\"", "Sends emergency SMS to your contact"),
    CommandHelp("\"Cancel SOS\"", "Cancels active emergency"),
    CommandHelp("\"Battery?\"", "Reports current battery percentage"),
    CommandHelp("\"Pause\"", "Pauses real-time navigation"),
    CommandHelp("\"Resume\"", "Resumes navigation"),
    CommandHelp("\"Settings\"", "Opens settings screen"),
    CommandHelp("\"Update my emergency contact\"", "Voice-guided update of emergency contact"),
    CommandHelp("\"Open finder\"", "Switches to the object finder (Grounding DINO search)"),
    CommandHelp("\"Open help\" / \"Help screen\"", "Navigates to this help screen"),
    CommandHelp("\"What can I say?\"", "Reads available commands aloud from navigation screen")
)

@HiltViewModel
class HelpViewModel @Inject constructor(
    private val feedback: FeedbackOrchestrator
) : ViewModel() {

    val isReadingAll = mutableStateOf(false)
    private var readAllJob: Job? = null

    fun onEnter() {
        val intro = "Help screen. Say any voice command. For example: "
        val topCommands = COMMAND_LIST.take(5).joinToString(". ") { cmd ->
            cmd.command.replace("\"", "")
        }
        feedback.speakSystem("$intro$topCommands. Tap Read All Commands to hear the full list.")
    }

    fun toggleReadAll() {
        if (isReadingAll.value) {
            stopNarration()
            return
        }
        isReadingAll.value = true
        readAllJob = viewModelScope.launch {
            feedback.speakSystem("Reading all ${COMMAND_LIST.size} voice commands.")
            delay(1500L)
            for (cmd in COMMAND_LIST) {
                if (!isReadingAll.value) break
                val phrase = cmd.command.replace("\"", "")
                feedback.speakSystem("$phrase: ${cmd.description}")
                delay(2200L)
            }
            if (isReadingAll.value) {
                feedback.speakSystem("End of voice commands.")
            }
            isReadingAll.value = false
        }
    }

    fun stopNarration() {
        readAllJob?.cancel()
        readAllJob = null
        isReadingAll.value = false
        feedback.stop()
    }

    override fun onCleared() {
        super.onCleared()
        feedback.stop()
    }
}

@Composable
fun HelpScreen(
    onBack: () -> Unit,
    viewModel: HelpViewModel = hiltViewModel()
) {
    LaunchedEffect(Unit) {
        viewModel.onEnter()
    }

    DisposableEffect(Unit) {
        onDispose { viewModel.stopNarration() }
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
                modifier = Modifier.heightIn(min = 56.dp).semantics { contentDescription = "Go back" }
            ) { Text("← Back") }
            Spacer(Modifier.width(16.dp))
            Text("Help", fontSize = 28.sp, fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.primary)
        }
        Spacer(Modifier.height(24.dp))

        Text("Voice Commands", fontSize = 20.sp, fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.onBackground)
        Spacer(Modifier.height(8.dp))
        Text("NOVA listens continuously. Say any of these:",
            fontSize = 14.sp,
            color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.7f))
        Spacer(Modifier.height(12.dp))

        val isReadingAll by viewModel.isReadingAll
        OutlinedButton(
            onClick = { viewModel.toggleReadAll() },
            modifier = Modifier
                .fillMaxWidth()
                .heightIn(min = 56.dp)
                .semantics {
                    contentDescription = if (isReadingAll)
                        "Stop reading commands"
                    else
                        "Read all ${COMMAND_LIST.size} voice commands aloud"
                }
        ) {
            Text(if (isReadingAll) "Stop Reading" else "Read All Commands", fontSize = 16.sp)
        }
        Spacer(Modifier.height(12.dp))

        COMMAND_LIST.forEach { cmd ->
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 4.dp)
                    .semantics {
                        contentDescription = "Voice command: ${cmd.command.replace("\"", "")}. ${cmd.description}"
                    },
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text(cmd.command, fontSize = 16.sp, fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.primary)
                    Spacer(Modifier.height(4.dp))
                    Text(cmd.description, fontSize = 14.sp,
                        color = MaterialTheme.colorScheme.onBackground)
                }
            }
        }

        Spacer(Modifier.height(32.dp))
        Text("Audio Feedback", fontSize = 20.sp, fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.onBackground)
        Spacer(Modifier.height(8.dp))
        Text("Audio uses stereo: objects on your left will be heard louder in your left ear.",
            fontSize = 14.sp,
            color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.7f))

        Spacer(Modifier.height(24.dp))
        Text("Vibration Patterns", fontSize = 20.sp, fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.onBackground)
        Spacer(Modifier.height(8.dp))
        listOf(
            "Single short pulse" to "Object far away",
            "Double pulse" to "Object in warning zone",
            "Long buzz" to "Danger — obstacle very close",
            "Triple pulse" to "Moving object approaching",
            "Heartbeat rhythm" to "Unknown obstacle detected"
        ).forEach { (pattern, meaning) ->
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 3.dp)
                    .semantics { contentDescription = "Vibration pattern: $pattern. Meaning: $meaning." },
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
            ) {
                Row(modifier = Modifier.padding(horizontal = 16.dp, vertical = 10.dp)) {
                    Text("$pattern: ", fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.primary, fontSize = 14.sp)
                    Text(meaning, color = MaterialTheme.colorScheme.onBackground, fontSize = 14.sp)
                }
            }
        }
    }
}
