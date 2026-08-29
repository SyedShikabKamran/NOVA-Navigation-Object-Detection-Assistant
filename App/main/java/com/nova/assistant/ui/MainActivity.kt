package com.nova.assistant.ui

import android.Manifest
import android.os.Bundle
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.*
import androidx.compose.ui.res.painterResource
import com.nova.assistant.R
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.nova.assistant.feedback.FeedbackOrchestrator
import com.nova.assistant.features.voicecommand.NovaCommand
import com.nova.assistant.features.voicecommand.VoiceCommandProcessor
import com.nova.assistant.ui.finder.FinderScreen
import com.nova.assistant.ui.navigation.MainNavigationScreen
import com.nova.assistant.ui.navigation.MainNavigationViewModel
import com.nova.assistant.ui.setup.SetupScreen
import com.nova.assistant.ui.splash.SplashScreen
import com.nova.assistant.ui.theme.NovaTheme
import com.nova.assistant.util.FileLogger
import dagger.hilt.android.AndroidEntryPoint
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import javax.inject.Inject

@AndroidEntryPoint
class MainActivity : ComponentActivity() {

    private val requiredPermissions = arrayOf(
        Manifest.permission.CAMERA,
        Manifest.permission.RECORD_AUDIO,
        Manifest.permission.ACCESS_FINE_LOCATION,
        Manifest.permission.SEND_SMS,
        Manifest.permission.VIBRATE
    )

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { /* Permissions handled by ViewModel */ }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Keep screen on during navigation (replaces android:keepScreenOn which is view-only)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        permissionLauncher.launch(requiredPermissions)

        setContent {
            NovaTheme {
                NovaApp()
            }
        }
    }
}

/**
 * Two-tab shell: Navigate (existing MainNavigationScreen) + Finder (Grounding DINO search).
 * MainNavigationScreen is always kept in composition so the navigation service stays bound
 * and alerts keep firing; FinderScreen is layered on top when the Finder tab is active.
 */
@Composable
fun MainTabScreen(
    onSettings: () -> Unit,
    onRooms: () -> Unit,
    onHelp: () -> Unit,
    selectedTab: Int = 0,
    onTabSelected: (Int) -> Unit = {},
) {

    Scaffold(
        bottomBar = {
            NavigationBar {
                NavigationBarItem(
                    selected = selectedTab == 0,
                    onClick = { onTabSelected(0) },
                    icon = { Icon(painterResource(R.drawable.ic_visibility), contentDescription = "Navigate") },
                    label = { Text("Navigate") },
                    modifier = Modifier.semantics { contentDescription = "Navigate tab" },
                )
                NavigationBarItem(
                    selected = selectedTab == 1,
                    onClick = { onTabSelected(1) },
                    icon = { Icon(Icons.Default.Search, contentDescription = "Finder") },
                    label = { Text("Finder") },
                    modifier = Modifier.semantics { contentDescription = "Finder tab" },
                )
            }
        }
    ) { innerPadding ->
        Box(Modifier.padding(innerPadding).fillMaxSize()) {
            // Navigate tab — always in composition so service stays bound and alerts fire.
            // sosEnabled=false while Finder is showing: FinderScreen is overlaid on top but
            // its background isn't a touch-consuming scrim, so a tap that misses Finder's own
            // interactive elements (e.g. a mis-tap near its mic button) falls through in
            // Compose's hit-testing to whatever sits at that same screen position underneath —
            // and SOSButton (bottom-center, 85% screen width) sits almost exactly where
            // Finder's mic button naturally lands. Disabling the button directly is the fix
            // (not a touch-consuming overlay) so it holds regardless of future layout changes
            // to either screen.
            MainNavigationScreen(
                onSettings = onSettings,
                onRooms = onRooms,
                onHelp = onHelp,
                sosEnabled = selectedTab != 1,
            )
            // Finder tab overlaid on top when selected
            AnimatedVisibility(
                visible = selectedTab == 1,
                enter = fadeIn(),
                exit = fadeOut(),
            ) {
                FinderScreen(modifier = Modifier.fillMaxSize())
            }
        }
    }
}

/**
 * App-level voice-navigation owner.
 *
 * Exposes the global command stream and a TTS hook so navigation commands can be handled
 * from ANY screen — the MainNavigationViewModel collector only drives action/query commands
 * and lives with the "main" back-stack entry, so navigation (settings/help/back/home) must be
 * owned here, where navController is in scope and survives every screen transition.
 */
@HiltViewModel
class NavVoiceViewModel @Inject constructor(
    voiceProcessor: VoiceCommandProcessor,
    private val feedback: FeedbackOrchestrator,
    val frameAnalyzer: com.nova.assistant.ml.NovaFrameAnalyzer,
    private val fileLogger: FileLogger,
) : ViewModel() {
    private companion object { private const val TAG = "NavVoice" }

    val commands: SharedFlow<NovaCommand> = voiceProcessor.commandFlow
    fun speak(text: String) = feedback.speakSystem(text)
    fun setFinderActive(active: Boolean) = feedback.setFinderActive(active)

    // Tab index: 0 = Navigate, 1 = Finder. Owned here so voice commands can switch tabs
    // from NovaApp's collector, which is the single owner of all navigation commands.
    private val _selectedTab = MutableStateFlow(0)
    val selectedTab: StateFlow<Int> = _selectedTab
    fun selectTab(index: Int) {
        val prev = _selectedTab.value
        if (prev != index) {
            val name = if (index == 1) "Finder" else "Navigate"
            fileLogger.i(TAG, "tab switch $prev→$index ($name) suspendNavPipeline will=${index == 1}")
        }
        _selectedTab.value = index
    }
}

@Composable
fun NovaApp() {
    val navController = rememberNavController()
    val navVoice: NavVoiceViewModel = hiltViewModel()
    val selectedTab by navVoice.selectedTab.collectAsState()
    val currentBackStackEntry by navController.currentBackStackEntryAsState()

    // Single owner of voice-driven navigation. Works on every screen because this collector
    // lives for the whole app session (NovaApp is always composed), unlike the per-screen
    // MainNavigationScreen collector. Action commands fall through to MainNavigationViewModel.
    LaunchedEffect(Unit) {
        navVoice.commands.collect { cmd ->
            val route = navController.currentDestination?.route
            val inApp = route != null && route != "splash" && route != "setup"
            when (cmd) {
                NovaCommand.SETTINGS -> if (inApp && route != "settings") {
                    navVoice.speak("Opening settings.")
                    navController.navigate("settings") { launchSingleTop = true }
                }
                NovaCommand.OPEN_HELP -> if (inApp && route != "help") {
                    navVoice.speak("Opening help screen.")
                    navController.navigate("help") { launchSingleTop = true }
                }
                NovaCommand.OPEN_FINDER -> if (inApp && route == "main") {
                    navVoice.speak("Opening object finder.")
                    navVoice.selectTab(1)
                }
                NovaCommand.GO_HOME, NovaCommand.GO_BACK -> if (inApp) {
                    if (route == "main" && selectedTab == 1) {
                        // On main screen but in Finder tab — go back to Navigate tab
                        navVoice.speak("Going back to navigation.")
                        navVoice.selectTab(0)
                    } else if (route != "main") {
                        navVoice.speak("Going back.")
                        navController.popBackStack("main", inclusive = false)
                    }
                }
                else -> { /* action/query commands handled by MainNavigationViewModel */ }
            }
        }
    }

    // Finder only needs a fresh preview bitmap, not full YOLO/depth navigation results —
    // running the nav pipeline while Finder is open was throttling how often the preview
    // updated (see suspendNavPipeline doc in NovaFrameAnalyzer). Suspend it whenever Finder
    // tab is actually showing, resume the instant the user leaves it.
    LaunchedEffect(selectedTab, currentBackStackEntry) {
        val route = currentBackStackEntry?.destination?.route
        val suspend = route == "main" && selectedTab == 1
        navVoice.frameAnalyzer.suspendNavPipeline = suspend
        // Same trigger as suspendNavPipeline — Finder tab suppresses alert delivery too, not
        // just new detections, so a straggling alert can't speak over the user's voice search.
        navVoice.setFinderActive(suspend)
        android.util.Log.i("NavVoice", "suspendNavPipeline=$suspend (route=$route tab=$selectedTab)")
    }

    Surface(
        modifier = Modifier.fillMaxSize(),
        color = MaterialTheme.colorScheme.background
    ) {
        NavHost(
            navController = navController,
            startDestination = "splash"
        ) {
            composable("splash") {
                SplashScreen(onReady = { isFirstTime ->
                    if (isFirstTime) {
                        navController.navigate("setup") {
                            popUpTo("splash") { inclusive = true }
                        }
                    } else {
                        navController.navigate("main") {
                            popUpTo("splash") { inclusive = true }
                        }
                    }
                })
            }
            composable("setup") {
                SetupScreen(onComplete = {
                    navController.navigate("main") {
                        popUpTo("setup") { inclusive = true }
                    }
                })
            }
            composable("main") {
                MainTabScreen(
                    onSettings = { navController.navigate("settings") },
                    onRooms = { navController.navigate("rooms") },
                    onHelp = { navController.navigate("help") },
                    selectedTab = selectedTab,
                    onTabSelected = { navVoice.selectTab(it) },
                )
            }
            composable("settings") {
                com.nova.assistant.ui.settings.SettingsScreen(
                    onBack = { navController.popBackStack() }
                )
            }
            composable("rooms") {
                com.nova.assistant.ui.misc.SavedRoomsScreen(
                    onBack = { navController.popBackStack() }
                )
            }
            composable("help") {
                com.nova.assistant.ui.misc.HelpScreen(
                    onBack = { navController.popBackStack() }
                )
            }
        }
    }
}
