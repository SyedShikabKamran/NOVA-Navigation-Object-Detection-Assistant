package com.nova.assistant.server

import android.graphics.Bitmap
import com.nova.assistant.util.FileLogger
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
import kotlinx.serialization.json.Json
import kotlinx.serialization.decodeFromString
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString.Companion.toByteString
import java.io.ByteArrayOutputStream
import java.util.concurrent.TimeUnit
import javax.inject.Inject
import javax.inject.Singleton

/**
 * NovaInferenceClient — OkHttp WebSocket client for the NOVA GPU inference server.
 *
 * Hybrid mode: sendFrame() returns null on timeout/error → caller falls back to on-device.
 * Auto-reconnect: up to MAX_RETRIES with RETRY_DELAY_MS backoff.
 * Render cold-start: fetchServerUrl() retries 3× with 5s delay.
 */
@Singleton
class NovaInferenceClient @Inject constructor(
    private val fileLogger: FileLogger,
) {
    companion object {
        private const val TAG               = "NovaInferenceClient"
        private const val FRAME_TIMEOUT_MS  = 2_000L
        private const val CONNECT_TIMEOUT_MS = 10_000L
        private const val REGISTRY_RETRIES  = 3
        private const val REGISTRY_DELAY_MS = 5_000L
        private const val MAX_RETRIES       = 3
        private const val RETRY_DELAY_MS    = 2_000L
        private const val JPEG_QUALITY      = 80   // raised 70→80: server detection accuracy over a few KB
        // After MAX_RETRIES + one registry re-fetch are exhausted, the old code gave up for the
        // rest of the service's life. Colab/Kaggle cold starts can take well over a minute, so
        // that meant a slow server start permanently killed server mode (and Finder, which has
        // no fallback) until the app was restarted. This is the backstop: keep polling the
        // registry at a slow, non-spammy interval until it succeeds.
        private const val LONG_TERM_RETRY_INTERVAL_MS = 30_000L
    }

    private val httpClient = OkHttpClient.Builder()
        .connectTimeout(CONNECT_TIMEOUT_MS, TimeUnit.MILLISECONDS)
        .readTimeout(FRAME_TIMEOUT_MS, TimeUnit.MILLISECONDS)
        .pingInterval(30, TimeUnit.SECONDS)
        .build()

    private val json = Json { ignoreUnknownKeys = true; isLenient = true }

    @Volatile private var ws: WebSocket? = null
    @Volatile private var connected = false
    @Volatile private var retryCount = 0
    @Volatile private var lastServerUrl: String? = null
    @Volatile private var savedRegistryUrl: String? = null
    @Volatile private var longTermRetryRunning = false
    @Volatile private var longTermRetryJob: kotlinx.coroutines.Job? = null
    // Set by disconnect() so a long-term loop that was mid-delay() doesn't reconnect right
    // after the caller deliberately tore the connection down (e.g. service onDestroy).
    @Volatile private var intentionallyDisconnected = false

    // One pending frame at a time — server processes sequentially
    @Volatile private var pendingFrame: CompletableDeferred<String?>? = null

    // ── Public API ──────────────────────────────────────────────────────────

    /**
     * Fetch WSS URL from Render registry and open WebSocket.
     * Retries registry GET to handle Render free-tier cold starts (30-60s).
     * Falls back to SERVER_DIRECT_WSS_URL if registry is empty (e.g. Render not yet deployed).
     */
    suspend fun connect(registryUrl: String): Boolean = withContext(Dispatchers.IO) {
        intentionallyDisconnected = false
        savedRegistryUrl = registryUrl
        val serverUrl = fetchServerUrl(registryUrl)
            ?: com.nova.assistant.util.NovaConstants.SERVER_DIRECT_WSS_URL.takeIf { it.isNotBlank() }
            ?: run {
                fileLogger.e(TAG, "No server URL from registry or direct fallback — will keep polling in background")
                startLongTermReconnect(registryUrl)
                return@withContext false
            }
        fileLogger.i(TAG, "Connecting to $serverUrl")
        lastServerUrl = serverUrl
        val ok = openWebSocket(serverUrl)
        if (!ok) startLongTermReconnect(registryUrl)
        ok
    }

    // ── Round-trip latency tracking ─────────────────────────────────────────
    private var rtFrameCount  = 0
    private var rtTotalMs     = 0L
    private var rtMinMs       = Long.MAX_VALUE
    private var rtMaxMs       = 0L
    private var rtTimeoutCount = 0

    /**
     * Compress [bitmap] to JPEG and send to server.
     * Returns parsed [ServerFrameResult] or null on timeout/error.
     * Null → caller should use on-device fallback.
     * Logs JPEG→response round-trip latency per frame and a rolling summary every 20 frames.
     */
    suspend fun sendFrame(bitmap: Bitmap): ServerFrameResult? {
        if (!connected || ws == null) return null

        val jpegBytes = withContext(Dispatchers.Default) {
            ByteArrayOutputStream().also {
                bitmap.compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, it)
            }.toByteArray()
        }

        val deferred = CompletableDeferred<String?>()
        pendingFrame = deferred

        val tSend = System.currentTimeMillis()
        val sent = ws?.send(jpegBytes.toByteString()) ?: false
        if (!sent) { pendingFrame = null; connected = false; return null }

        val text = withTimeoutOrNull(FRAME_TIMEOUT_MS) { deferred.await() }
        val rtMs = System.currentTimeMillis() - tSend
        pendingFrame = null

        if (text == null) {
            rtTimeoutCount++
            fileLogger.w(TAG, "Frame timed out after ${FRAME_TIMEOUT_MS}ms (timeouts=$rtTimeoutCount)")
            return null
        }

        // Update stats
        rtFrameCount++
        rtTotalMs += rtMs
        if (rtMs < rtMinMs) rtMinMs = rtMs
        if (rtMs > rtMaxMs) rtMaxMs = rtMs

        fileLogger.d(TAG, "RTT ${rtMs}ms | jpeg=${jpegBytes.size / 1024}KB")

        // Rolling summary every 20 frames
        if (rtFrameCount % 20 == 0) {
            fileLogger.i(TAG,
                "RTT summary [${rtFrameCount}fr]: " +
                "avg=${rtTotalMs / rtFrameCount}ms " +
                "min=${rtMinMs}ms max=${rtMaxMs}ms " +
                "timeouts=$rtTimeoutCount"
            )
        }

        return try {
            json.decodeFromString<ServerFrameResult>(text)
        } catch (e: Exception) {
            fileLogger.e(TAG, "JSON parse error: ${e.message}")
            null
        }
    }

    fun isConnected(): Boolean = connected

    /**
     * Derive the server's HTTP base URL (e.g. for POST /find) from the currently connected
     * WSS URL — same host, scheme swapped, "/ws" suffix stripped. Null if not connected.
     */
    fun httpBaseUrl(): String? {
        if (!connected) return null
        val wsUrl = lastServerUrl ?: return null
        return wsUrl
            .replace("wss://", "https://")
            .replace("ws://", "http://")
            .removeSuffix("/ws")
    }

    fun disconnect() {
        intentionallyDisconnected = true
        longTermRetryJob?.cancel()
        longTermRetryJob = null
        ws?.close(1000, "Client disconnect")
        ws = null
        connected = false
        pendingFrame?.complete(null)
        pendingFrame = null
        fileLogger.i(TAG, "Disconnected from inference server.")
    }

    // ── Internal ─────────────────────────────────────────────────────────────

    private suspend fun fetchServerUrl(registryUrl: String): String? =
        withContext(Dispatchers.IO) {
            repeat(REGISTRY_RETRIES) { attempt ->
                try {
                    val response = httpClient.newCall(
                        Request.Builder().url("$registryUrl/server-url").build()
                    ).execute()
                    val body = response.body?.string() ?: return@repeat
                    val url = json.decodeFromString<Map<String, String>>(body)["url"]
                    if (!url.isNullOrBlank()) {
                        fileLogger.i(TAG, "Registry: $url")
                        return@withContext url
                    }
                    fileLogger.w(TAG, "Registry returned empty URL (attempt ${attempt + 1}/$REGISTRY_RETRIES)")
                } catch (e: Exception) {
                    fileLogger.w(TAG, "Registry fetch failed (attempt ${attempt + 1}): ${e.message}")
                }
                if (attempt < REGISTRY_RETRIES - 1) delay(REGISTRY_DELAY_MS)
            }
            null
        }

    private suspend fun openWebSocket(serverUrl: String): Boolean {
        val deferred = CompletableDeferred<Boolean>()

        ws = httpClient.newWebSocket(
            Request.Builder().url(serverUrl).build(),
            object : WebSocketListener() {
                override fun onOpen(webSocket: WebSocket, response: Response) {
                    connected = true; retryCount = 0
                    fileLogger.i(TAG, "Connected to NOVA inference server: $serverUrl")
                    deferred.complete(true)
                }

                override fun onMessage(webSocket: WebSocket, text: String) {
                    pendingFrame?.complete(text)
                }

                override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                    fileLogger.e(TAG, "WebSocket failure: ${t.message}")
                    connected = false
                    pendingFrame?.complete(null)
                    if (!deferred.isCompleted) deferred.complete(false)
                    maybeReconnect()
                }

                override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
                    webSocket.close(1000, null)
                }

                override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                    fileLogger.i(TAG, "WebSocket closed: $reason")
                    connected = false
                    pendingFrame?.complete(null)
                }
            }
        )

        return withTimeoutOrNull(CONNECT_TIMEOUT_MS) { deferred.await() } ?: false
    }

    private fun maybeReconnect() {
        lastServerUrl ?: return
        if (retryCount >= MAX_RETRIES) {
            val regUrl = savedRegistryUrl ?: run {
                fileLogger.e(TAG, "Max retries reached, no registry URL saved — staying on-device.")
                return
            }
            fileLogger.i(TAG, "Stale URL exhausted — re-fetching registry for fresh server URL...")
            GlobalScope.launch(Dispatchers.IO) {
                val freshUrl = fetchServerUrl(regUrl)
                if (freshUrl != null && freshUrl != lastServerUrl) {
                    fileLogger.i(TAG, "Fresh URL from registry: $freshUrl")
                    lastServerUrl = freshUrl
                    retryCount = 0
                    if (!openWebSocket(freshUrl)) startLongTermReconnect(regUrl)
                } else {
                    fileLogger.w(TAG, "Registry returned same/empty URL — falling back to background polling.")
                    startLongTermReconnect(regUrl)
                }
            }
            return
        }
        retryCount++
        fileLogger.i(TAG, "Reconnecting (${retryCount}/$MAX_RETRIES) in ${RETRY_DELAY_MS}ms...")
        GlobalScope.launch(Dispatchers.IO) {
            delay(RETRY_DELAY_MS * retryCount) // exponential-ish backoff
            openWebSocket(lastServerUrl ?: return@launch)
        }
    }

    /**
     * Backstop for a slow/cold-starting server: poll the registry at a slow interval until a
     * connection succeeds. Guarded by [longTermRetryRunning] so connect()'s failure path and
     * maybeReconnect()'s give-up path (which can both fire for the same outage) don't spawn
     * two competing loops. Stops itself the moment [connected] becomes true.
     */
    private fun startLongTermReconnect(registryUrl: String) {
        if (longTermRetryRunning || connected || intentionallyDisconnected) return
        longTermRetryRunning = true
        fileLogger.i(TAG, "Starting long-term reconnect polling every ${LONG_TERM_RETRY_INTERVAL_MS}ms")
        longTermRetryJob = GlobalScope.launch(Dispatchers.IO) {
            try {
                while (!connected && !intentionallyDisconnected) {
                    delay(LONG_TERM_RETRY_INTERVAL_MS)
                    if (connected || intentionallyDisconnected) break
                    fileLogger.i(TAG, "Long-term reconnect: polling registry...")
                    val url = fetchServerUrl(registryUrl)
                        ?: com.nova.assistant.util.NovaConstants.SERVER_DIRECT_WSS_URL.takeIf { it.isNotBlank() }
                    if (url != null) {
                        lastServerUrl = url
                        retryCount = 0
                        if (openWebSocket(url)) break
                    }
                }
            } finally {
                longTermRetryRunning = false
            }
        }
    }
}
