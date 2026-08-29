package com.nova.assistant.ml

import android.content.Context
import android.os.Build
import com.nova.assistant.util.FileLogger
import dagger.hilt.android.qualifiers.ApplicationContext
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Bundles the current debug log files into a plain ZIP archive for off-device sharing.
 *
 * Output format:
 *   nova_bundle_<MODEL>_YYYYMMDD_HHmmss.zip
 *       ZIP contents:
 *         logs/nova_debug.log
 *         logs/nova_debug_old.log        (if present)
 *         logs/<voice_log>               (if present)
 */
@Singleton
class DebugBundler @Inject constructor(
    @ApplicationContext private val context: Context,
    private val fileLogger: FileLogger
) {
    companion object {
        private const val TAG = "DebugBundler"
        private const val BUNDLE_PREFIX = "nova_bundle_"
    }

    /**
     * Creates the debug log bundle. Blocking — call from a background coroutine.
     * Returns the output File on success, null if an unrecoverable error occurs.
     * Any previous bundle files in the output directory are deleted first.
     */
    fun createBundle(): File? {
        return try {
            val outputDir = context.getExternalFilesDir(null) ?: context.filesDir
            pruneOldBundles(outputDir)

            val model = Build.MODEL.replace(Regex("[^A-Za-z0-9_-]"), "_")
            val ts = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            val outFile = File(outputDir, "${BUNDLE_PREFIX}${model}_${ts}.zip")

            val logFile = fileLogger.getLogFile()
            val oldLogFile = File(logFile.parent ?: outputDir.absolutePath, "nova_debug_old.log")
            val voiceLogFile = fileLogger.getVoiceLogFile()

            val logBytes = if (logFile.exists()) logFile.length() else 0L
            val oldBytes = if (oldLogFile.exists()) oldLogFile.length() else 0L
            fileLogger.i(TAG, "Bundle start: log=${logBytes}B old=${oldBytes}B")

            FileOutputStream(outFile).use { fos ->
                ZipOutputStream(fos).use { zip ->
                    if (logFile.exists())      zip.addFile("logs/${logFile.name}", logFile)
                    if (oldLogFile.exists())   zip.addFile("logs/${oldLogFile.name}", oldLogFile)
                    if (voiceLogFile.exists()) zip.addFile("logs/${voiceLogFile.name}", voiceLogFile)
                }
            }

            fileLogger.i(TAG, "Bundle ready: ${outFile.name}  size=${outFile.length()}B")
            outFile
        } catch (e: Exception) {
            fileLogger.e(TAG, "Bundle creation failed", e)
            null
        }
    }

    /**
     * Delete all source log files after a successful share.
     *
     * Safety contract: only deletes if [bundleFile] exists and is non-empty (> 512 bytes).
     * Silent — errors go to LogCat only.
     * Call this AFTER the share chooser has been launched successfully.
     */
    fun clearSourceFiles(bundleFile: File) {
        if (!bundleFile.exists() || bundleFile.length() < 512) return

        runCatching { fileLogger.getLogFile().delete() }
        runCatching { File(fileLogger.getLogFile().parent ?: "", "nova_debug_old.log").delete() }
        runCatching { fileLogger.getVoiceLogFile().delete() }
    }

    private fun ZipOutputStream.addFile(entryPath: String, file: File) {
        putNextEntry(ZipEntry(entryPath))
        file.inputStream().buffered().use { it.copyTo(this) }
        closeEntry()
    }

    private fun pruneOldBundles(dir: File) {
        dir.listFiles { f -> f.name.startsWith(BUNDLE_PREFIX) }
            ?.forEach { it.delete() }
    }
}
