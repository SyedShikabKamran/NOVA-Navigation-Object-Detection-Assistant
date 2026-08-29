package com.nova.assistant.service

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Binder
import android.os.IBinder
import android.util.Log
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import android.util.Size
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.NotificationCompat
import androidx.lifecycle.LifecycleService
import androidx.lifecycle.lifecycleScope
import com.nova.assistant.feedback.FeedbackOrchestrator
import com.nova.assistant.features.LocationProvider
import com.nova.assistant.features.voicecommand.VoiceCommandProcessor
import com.nova.assistant.ml.NovaFrameAnalyzer
import com.nova.assistant.sensors.NovaSensorManager
import com.nova.assistant.server.NovaInferenceClient
import com.nova.assistant.ui.MainActivity
import com.nova.assistant.util.FileLogger
import com.nova.assistant.util.NovaConstants
import dagger.hilt.android.AndroidEntryPoint
import java.util.concurrent.Executors
import javax.inject.Inject
import kotlinx.coroutines.launch

/**
 * NovaNavigationService — foreground LifecycleService that owns the camera,
 * sensors, and voice processor for the entire app lifetime.
 *
 * Architecture:
 *   Activity/Composable binds to this service → calls attachPreviewSurface()
 *   to wire the camera preview into the UI. When the activity is backgrounded
 *   (user locks screen, switches apps), the service keeps running so NOVA
 *   continues detecting obstacles and speaking alerts.
 *
 * Lifecycle:
 *   MainActivity.onStart()  → startService() + bindService()
 *   MainActivity.onStop()   → detachPreviewSurface(); unbindService() keeps the
 *                             service alive (started service survives unbind)
 *   User kills app          → stopSelf() called from stopNavigation(); service stops
 */
@AndroidEntryPoint
class NovaNavigationService : LifecycleService() {

    @Inject lateinit var frameAnalyzer: NovaFrameAnalyzer
    @Inject lateinit var sensorManager: NovaSensorManager
    @Inject lateinit var voiceProcessor: VoiceCommandProcessor
    @Inject lateinit var locationProvider: LocationProvider
    @Inject lateinit var feedback: FeedbackOrchestrator
    @Inject lateinit var fileLogger: FileLogger
    @Inject lateinit var inferenceClient: NovaInferenceClient

    private val binder = LocalBinder()

    // The Preview use case — UI attaches its SurfaceProvider here to show the feed.
    private val preview = Preview.Builder().build()
    private var cameraProvider: ProcessCameraProvider? = null

    companion object {
        private const val TAG = "NovaNavigationService"
        private const val CHANNEL_ID = "nova_navigation_channel"
        private const val NOTIFICATION_ID = 1001
    }

    inner class LocalBinder : Binder() {
        fun getService(): NovaNavigationService = this@NovaNavigationService
    }

    override fun onBind(intent: Intent): IBinder {
        super.onBind(intent)
        return binder
    }

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
        startForeground(NOTIFICATION_ID, buildNotification())

        feedback.initialize()
        sensorManager.start()
        voiceProcessor.initialize()
        voiceProcessor.startListening()

        // Wire sensor motion magnitude to FeedbackOrchestrator for adaptive speech rate.
        lifecycleScope.launch {
            sensorManager.sensorData.collect { data ->
                feedback.movementMagnitude = data.motionMagnitude
            }
        }

        // Only request location updates if permission has been granted.
        // On Android 10+, ACCESS_FINE_LOCATION must be granted at runtime before
        // calling LocationRequest. Some Huawei devices require this check to happen
        // before any location API call, even with permission check inside LocationProvider.
        if (ContextCompat.checkSelfPermission(
                this, android.Manifest.permission.ACCESS_FINE_LOCATION
            ) == PackageManager.PERMISSION_GRANTED
        ) {
            locationProvider.startLocationUpdates()
        } else {
            fileLogger.w(TAG, "ACCESS_FINE_LOCATION not granted — location updates skipped")
        }

        bindCamera()

        // Connect to GPU inference server (falls back to on-device if unavailable)
        lifecycleScope.launch {
            val connected = inferenceClient.connect(NovaConstants.SERVER_REGISTRY_URL)
            fileLogger.i(TAG, if (connected) "Connected to NOVA inference server"
                              else "Server unavailable — using on-device inference")
        }

        fileLogger.i(TAG, "NovaNavigationService created")
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        super.onStartCommand(intent, flags, startId)
        return START_STICKY  // Android will restart the service if killed
    }

    override fun onDestroy() {
        inferenceClient.disconnect()
        cameraProvider?.unbindAll()
        sensorManager.stop()
        voiceProcessor.shutdown()
        locationProvider.stopLocationUpdates()
        frameAnalyzer.shutdown()
        feedback.shutdown()
        fileLogger.i(TAG, "NovaNavigationService destroyed")
        super.onDestroy()
    }

    // ─── Camera ──────────────────────────────────────────────────────────────

    private fun bindCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            try {
                val provider = cameraProviderFuture.get()
                cameraProvider = provider

                // Request ~1280×720 analysis frames so the GPU server's imgsz=1280 pass has real
                // detail (CameraX default is ~640×480). On-device fallback downscales to 640²
                // anyway. FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER picks the nearest supported size.
                val analyzer = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .setResolutionSelector(
                        ResolutionSelector.Builder()
                            .setResolutionStrategy(
                                ResolutionStrategy(
                                    Size(1280, 720),
                                    ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER
                                )
                            )
                            .build()
                    )
                    .build()
                    .also { it.setAnalyzer(Executors.newSingleThreadExecutor(), frameAnalyzer) }

                provider.unbindAll()
                provider.bindToLifecycle(
                    this,   // LifecycleService IS a LifecycleOwner — camera survives activity restarts
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    analyzer
                )
                fileLogger.i(TAG, "Camera bound to service lifecycle")
            } catch (e: Exception) {
                fileLogger.e(TAG, "Camera binding failed: ${e.message}")
            }
        }, mainExecutor)
    }

    /**
     * Called by the UI composable to wire the camera preview into a PreviewView.
     * Safe to call multiple times — attaches the current surface provider.
     */
    fun attachPreviewSurface(surfaceProvider: Preview.SurfaceProvider) {
        preview.setSurfaceProvider(surfaceProvider)
    }

    /**
     * Called when the UI is backgrounded. Detaches the surface so the
     * camera continues capturing for the analyzer without rendering.
     */
    fun detachPreviewSurface() {
        preview.setSurfaceProvider(null)
    }

    /**
     * Stop navigation and release all resources. Called when user explicitly quits.
     */
    fun stopNavigation() {
        stopSelf()
    }

    // ─── Notification ─────────────────────────────────────────────────────────

    private fun createNotificationChannel() {
        val channel = NotificationChannel(
            CHANNEL_ID,
            "NOVA Navigation",
            NotificationManager.IMPORTANCE_LOW
        ).apply {
            description = "NOVA obstacle detection is running"
            setShowBadge(false)
        }
        val nm = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
        nm.createNotificationChannel(channel)
    }

    private fun buildNotification(): Notification {
        val tapIntent = Intent(this, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_SINGLE_TOP
        }
        val pendingIntent = PendingIntent.getActivity(
            this, 0, tapIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("NOVA — Navigation Active")
            .setContentText("Obstacle detection is running")
            .setSmallIcon(android.R.drawable.ic_menu_compass)
            .setContentIntent(pendingIntent)
            .setOngoing(true)
            .setSilent(true)
            .build()
    }
}
