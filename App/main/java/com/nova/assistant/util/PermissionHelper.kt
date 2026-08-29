package com.nova.assistant.util

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.PowerManager
import android.provider.Settings
import androidx.activity.ComponentActivity
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import com.nova.assistant.feedback.FeedbackOrchestrator
import dagger.hilt.android.qualifiers.ApplicationContext
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Manages runtime permissions with voice-guided prompts.
 *
 * NOVA requires these permissions to function:
 * - CAMERA: for real-time obstacle detection
 * - RECORD_AUDIO: for voice commands
 * - ACCESS_FINE_LOCATION: for hazard memory GPS tagging
 * - SEND_SMS: for emergency SOS
 * - VIBRATE: for haptic feedback (granted automatically)
 */
@Singleton
class PermissionHelper @Inject constructor(
    @ApplicationContext private val context: Context,
    private val feedback: FeedbackOrchestrator
) {
    val requiredPermissions = arrayOf(
        Manifest.permission.CAMERA,
        Manifest.permission.RECORD_AUDIO,
        Manifest.permission.ACCESS_FINE_LOCATION,
        Manifest.permission.SEND_SMS
    )

    private val permissionDescriptions = mapOf(
        Manifest.permission.CAMERA to
                "NOVA needs camera access to detect obstacles. Please tap Allow.",
        Manifest.permission.RECORD_AUDIO to
                "NOVA needs microphone access for voice commands. Please tap Allow.",
        Manifest.permission.ACCESS_FINE_LOCATION to
                "NOVA needs location access to remember hazards. Please tap Allow.",
        Manifest.permission.SEND_SMS to
                "NOVA needs SMS access to send emergency alerts. Please tap Allow."
    )

    /**
     * Check if all required permissions are granted.
     */
    fun allPermissionsGranted(): Boolean = requiredPermissions.all { permission ->
        ContextCompat.checkSelfPermission(context, permission) == PackageManager.PERMISSION_GRANTED
    }

    /**
     * Get list of permissions that are not yet granted.
     */
    fun missingPermissions(): List<String> = requiredPermissions.filter { permission ->
        ContextCompat.checkSelfPermission(context, permission) != PackageManager.PERMISSION_GRANTED
    }

    /**
     * Check a specific permission.
     */
    fun isGranted(permission: String): Boolean =
        ContextCompat.checkSelfPermission(context, permission) == PackageManager.PERMISSION_GRANTED

    /**
     * Announce what a permission is needed for.
     */
    fun announcePermission(permission: String) {
        permissionDescriptions[permission]?.let { description ->
            feedback.speakSystem(description)
        }
    }

    /**
     * Announce all missing permissions.
     */
    fun announceMissingPermissions() {
        val missing = missingPermissions()
        if (missing.isEmpty()) {
            feedback.speakSystem("All permissions granted. NOVA is ready.")
        } else {
            feedback.speakSystem("NOVA needs ${missing.size} permissions to work correctly. Please grant them now.")
        }
    }

    /**
     * Returns true if NOVA is already exempt from battery optimization.
     * False means Android/EMUI may kill the camera service when backgrounded.
     */
    fun isIgnoringBatteryOptimizations(): Boolean {
        val pm = context.getSystemService(Context.POWER_SERVICE) as PowerManager
        return pm.isIgnoringBatteryOptimizations(context.packageName)
    }

    /**
     * Opens the system dialog that lets the user whitelist NOVA from battery optimization.
     * Falls back to the battery optimization list screen on EMUI (which may block the direct
     * ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS intent).
     */
    fun openBatteryOptimizationSettings() {
        try {
            val intent = Intent(
                Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS,
                Uri.parse("package:${context.packageName}")
            ).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            context.startActivity(intent)
        } catch (e: Exception) {
            // EMUI blocks the direct per-app intent — open the battery optimization list instead
            val intent = Intent(Settings.ACTION_IGNORE_BATTERY_OPTIMIZATION_SETTINGS)
                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            context.startActivity(intent)
        }
    }

    /**
     * Open app settings if permissions were permanently denied.
     */
    fun openAppSettings() {
        val intent = Intent(
            Settings.ACTION_APPLICATION_DETAILS_SETTINGS,
            Uri.fromParts("package", context.packageName, null)
        ).apply {
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        }
        context.startActivity(intent)
    }

    companion object {
        val CRITICAL_PERMISSIONS = setOf(
            Manifest.permission.CAMERA,
            Manifest.permission.RECORD_AUDIO
        )

        val OPTIONAL_PERMISSIONS = setOf(
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.SEND_SMS
        )
    }
}
