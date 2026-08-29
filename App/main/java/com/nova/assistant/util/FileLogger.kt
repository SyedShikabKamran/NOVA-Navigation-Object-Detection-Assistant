package com.nova.assistant.util

import android.content.Context
import android.util.Log
import dagger.hilt.android.qualifiers.ApplicationContext
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executors
import javax.inject.Inject
import javax.inject.Singleton

/**
 * File-based logger for on-device debugging without ADB.
 *
 * Writes timestamped log lines to nova_debug.log in the app's external files directory:
 *   /sdcard/Android/data/com.nova.assistant/files/nova_debug.log
 *
 * Accessible via the phone's file manager (Files app) without a laptop.
 * Share the file to Claude for diagnosis using the "Share Log File" button in Settings.
 *
 * All methods ALSO call android.util.Log so LogCat still works when connected.
 *
 * Rotation: when nova_debug.log exceeds MAX_SIZE_BYTES, it is renamed to
 * nova_debug_old.log and a new nova_debug.log starts. Two files max = 6 MB total.
 *
 * Write thread: a single background executor processes all writes in order.
 * No main-thread blocking. No per-frame allocation.
 */
@Singleton
class FileLogger @Inject constructor(
    @ApplicationContext private val context: Context
) {
    companion object {
        private const val SYSTEM_TAG = "FileLogger"
        private const val LOG_FILE_NAME = "nova_debug.log"
        private const val OLD_LOG_FILE_NAME = "nova_debug_old.log"
        private const val VOICE_LOG_FILE_NAME = "nova_voice.log"
        private const val MAX_SIZE_BYTES = 3L * 1024 * 1024  // 3 MB per file, 6 MB total
        private const val MAX_VOICE_LOG_BYTES = 1L * 1024 * 1024  // 1 MB, rotate by delete
        private val DATE_FORMAT = SimpleDateFormat("MM-dd HH:mm:ss.SSS", Locale.US)
        private val VOICE_DATE_FORMAT = SimpleDateFormat("HH:mm:ss", Locale.US)
    }

    private val executor = Executors.newSingleThreadExecutor { r ->
        Thread(r, "nova-file-logger").also { it.isDaemon = true }
    }

    // Use underscore to avoid JVM signature clash with getLogFile() function below
    private val _logFile: File by lazy {
        val dir = context.getExternalFilesDir(null) ?: context.filesDir
        if (!dir.exists()) dir.mkdirs()
        File(dir, LOG_FILE_NAME)
    }

    private val _voiceLogFile: File by lazy {
        val dir = context.getExternalFilesDir(null) ?: context.filesDir
        if (!dir.exists()) dir.mkdirs()
        File(dir, VOICE_LOG_FILE_NAME)
    }

    // ── Public API ──────────────────────────────────────────────────────────────

    fun d(tag: String, message: String) {
        Log.d(tag, message)
        write("D", tag, message)
    }

    fun i(tag: String, message: String) {
        Log.i(tag, message)
        write("I", tag, message)
    }

    fun w(tag: String, message: String) {
        Log.w(tag, message)
        write("W", tag, message)
    }

    fun e(tag: String, message: String, throwable: Throwable? = null) {
        if (throwable != null) Log.e(tag, message, throwable) else Log.e(tag, message)
        val full = if (throwable != null) "$message  [${throwable.javaClass.simpleName}: ${throwable.message}]" else message
        write("E", tag, full)
    }

    /** Returns the current log file for sharing via FileProvider. */
    fun getLogFile(): File = _logFile

    /** Returns the voice command log file for sharing via FileProvider. */
    fun getVoiceLogFile(): File = _voiceLogFile

    /** Returns the log file path as a human-readable string for display in Settings. */
    fun getLogFilePath(): String = _logFile.absolutePath

    /** True if the log file exists and has content. */
    fun hasLogs(): Boolean = _logFile.exists() && _logFile.length() > 0L

    /**
     * Write a structured voice event to nova_voice.log.
     *
     * Format (one line per event):
     *   HH:mm:ss [SOURCE]    COMMAND ← "heard" → output
     *
     * [source]
     *   USER     — utterance matched a command (user spoke intentionally)
     *   AMBIENT  — utterance did NOT match any command (background voice or mumble)
     *   TTS-ECHO — recording captured during TTS playback and discarded (potential self-echo)
     *
     * [command]  NovaCommand.name, "UNKNOWN", or "TTS-ECHO"
     * [heard]    transcribed text (null when no transcript available)
     * [output]   action taken or output spoken (null = no action)
     */
    fun voiceEvent(command: String, heard: String?, output: String?, source: String = "USER") {
        executor.execute {
            try {
                if (_voiceLogFile.exists() && _voiceLogFile.length() > MAX_VOICE_LOG_BYTES) {
                    _voiceLogFile.delete()
                }
                val time = VOICE_DATE_FORMAT.format(Date())
                // Fixed-width source tag so columns align when reading the log
                val tag = "[$source]".padEnd(11)
                val line = when {
                    heard != null && output != null ->
                        "$time $tag $command ← \"$heard\" → $output\n"
                    heard != null ->
                        "$time $tag $command ← \"$heard\"\n"
                    output != null ->
                        "$time $tag $command → $output\n"
                    else ->
                        "$time $tag $command\n"
                }
                _voiceLogFile.appendText(line, Charsets.UTF_8)
            } catch (e: Exception) {
                Log.e(SYSTEM_TAG, "Voice log write failed: ${e.message}")
            }
        }
    }

    /** Delete the current log and the old-log backup. */
    fun clear() {
        executor.execute {
            try {
                _logFile.delete()
                File(_logFile.parent, OLD_LOG_FILE_NAME).delete()
                Log.i(SYSTEM_TAG, "Log files cleared")
            } catch (e: Exception) {
                Log.e(SYSTEM_TAG, "Failed to clear logs", e)
            }
        }
    }

    /** Delete all log files — debug log, old debug log, and voice log. */
    fun clearAll() {
        executor.execute {
            try {
                _logFile.delete()
                File(_logFile.parent, OLD_LOG_FILE_NAME).delete()
                _voiceLogFile.delete()
                Log.i(SYSTEM_TAG, "All log files cleared")
            } catch (e: Exception) {
                Log.e(SYSTEM_TAG, "Failed to clear all logs", e)
            }
        }
    }

    // ── Internal ─────────────────────────────────────────────────────────────

    private fun write(level: String, tag: String, message: String) {
        executor.execute {
            try {
                rotateIfNeeded()
                val line = "${DATE_FORMAT.format(Date())} $level/$tag: $message\n"
                _logFile.appendText(line, Charsets.UTF_8)
            } catch (e: Exception) {
                // Do not recurse — just drop to LogCat
                Log.e(SYSTEM_TAG, "Write failed: ${e.message}")
            }
        }
    }

    private fun rotateIfNeeded() {
        if (_logFile.exists() && _logFile.length() > MAX_SIZE_BYTES) {
            val old = File(_logFile.parent, OLD_LOG_FILE_NAME)
            old.delete()
            _logFile.renameTo(old)
            // _logFile no longer exists — next appendText() creates a new one
        }
    }
}
