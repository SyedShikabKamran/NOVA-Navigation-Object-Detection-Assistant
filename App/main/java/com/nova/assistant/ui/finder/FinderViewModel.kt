package com.nova.assistant.ui.finder

import android.graphics.Bitmap
import android.graphics.RectF
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.nova.assistant.feedback.FeedbackOrchestrator
import com.nova.assistant.features.voicecommand.VoiceCommandProcessor
import com.nova.assistant.ml.FindResult
import com.nova.assistant.ml.GroundingDinoClient
import com.nova.assistant.ml.NovaFrameAnalyzer
import com.nova.assistant.util.FileLogger
import com.nova.assistant.util.SpatialDirection
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlinx.coroutines.withTimeoutOrNull
import javax.inject.Inject

sealed class FinderState {
    object Idle          : FinderState()
    object WaitingQuery  : FinderState()  // mic open, listening for object name
    data class Searching(val query: String, val frame: Bitmap) : FinderState()
    data class Found(val query: String, val box: RectF, val direction: String, val frame: Bitmap) : FinderState()
    data class NotFound(val query: String) : FinderState()
    data class Error(val message: String)  : FinderState()
}

@HiltViewModel
class FinderViewModel @Inject constructor(
    private val client: GroundingDinoClient,
    private val frameAnalyzer: NovaFrameAnalyzer,
    private val voiceProcessor: VoiceCommandProcessor,
    private val feedback: FeedbackOrchestrator,
    private val fileLogger: FileLogger,
) : ViewModel() {
    private companion object {
        private const val TAG = "FinderViewModel"
    }

    private val _state = MutableStateFlow<FinderState>(FinderState.Idle)
    val state: StateFlow<FinderState> = _state.asStateFlow()

    // Live camera frame updated every 500ms for the idle preview
    private val _liveBitmap = MutableStateFlow<Bitmap?>(null)
    val liveBitmap: StateFlow<Bitmap?> = _liveBitmap.asStateFlow()

    init {
        viewModelScope.launch {
            while (true) {
                val s = _state.value
                if (s is FinderState.Idle || s is FinderState.WaitingQuery) {
                    // Not recycled here deliberately: Compose's Image() composable may still be
                    // drawing the previous value on the render thread when this replaces it —
                    // recycling on this (main) thread wouldn't be safe to sequence against that.
                    // Each snapshot is a small preview-sized bitmap; the old one is simply
                    // dropped and left for GC once Compose stops referencing it.
                    _liveBitmap.value = frameAnalyzer.snapshotLastBitmap()
                }
                delay(500)
            }
        }
    }

    /** User tapped the mic button — listen for one raw utterance then search. */
    fun startListening() {
        if (_state.value !is FinderState.Idle) {
            fileLogger.d(TAG, "startListening: ignored — state=${_state.value::class.simpleName}")
            return
        }
        fileLogger.i(TAG, "startListening: mic opened, waiting for query (12s timeout)")
        _state.value = FinderState.WaitingQuery
        viewModelScope.launch {
            // rawTextFlow emits utterances that didn't match any NOVA command.
            // "water bottle", "my keys", "the fire extinguisher" all fall through here.
            val query = withTimeoutOrNull(12_000L) {
                voiceProcessor.rawTextFlow.first { it.isNotBlank() }
            }
            if (query == null) {
                fileLogger.w(TAG, "startListening: 12s timeout — no query received, returning to Idle")
                _state.value = FinderState.Idle
            } else {
                fileLogger.i(TAG, "startListening: voice query received=\"$query\"")
                doSearch(query)
            }
        }
    }

    /** User typed a query via the manual text field. */
    fun searchTyped(query: String) {
        if (query.isBlank() || _state.value !is FinderState.Idle) {
            fileLogger.d(TAG, "searchTyped: ignored blank or non-Idle state (\"$query\", state=${_state.value::class.simpleName})")
            return
        }
        fileLogger.i(TAG, "searchTyped: typed query=\"$query\"")
        viewModelScope.launch { doSearch(query) }
    }

    fun reset() {
        _state.value = FinderState.Idle
    }

    private suspend fun doSearch(query: String) {
        fileLogger.i(TAG, "doSearch: start query=\"$query\"")
        val frame = frameAnalyzer.snapshotLastBitmap()
            ?: run {
                fileLogger.w(TAG, "doSearch: no camera frame — snapshotLastBitmap() returned null")
                _state.value = FinderState.Error("Camera not ready")
                return
            }
        fileLogger.i(TAG, "doSearch: frame snapshot=${frame.width}x${frame.height}")
        _state.value = FinderState.Searching(query, frame)

        val t0 = System.currentTimeMillis()
        val result = runCatching { client.locate(frame, query) }.getOrElse { e ->
            fileLogger.e(TAG, "doSearch: locate() threw: ${e.message}")
            FindResult.ServerUnavailable
        }
        val elapsed = System.currentTimeMillis() - t0
        fileLogger.i(TAG, "doSearch: locate returned ${result::class.simpleName} in ${elapsed}ms")

        when (result) {
            is FindResult.Found -> {
                val direction = SpatialDirection.fromNormalizedX(result.box.centerX()).toSpokenDirection()
                fileLogger.i(TAG, "doSearch: FOUND query=\"$query\" direction=$direction box=[${result.box.left},${result.box.top},${result.box.right},${result.box.bottom}]")
                _state.value = FinderState.Found(query, result.box, direction, frame)
                feedback.speakSystem("$query found, $direction")
            }
            is FindResult.NotFound -> {
                fileLogger.i(TAG, "doSearch: NOT FOUND query=\"$query\"")
                _state.value = FinderState.NotFound(query)
                feedback.speakSystem("Couldn't find $query. Move the camera and try again.")
            }
            is FindResult.ServerUnavailable -> {
                // Distinct from NotFound: "move the camera" is useless advice when the real
                // problem is no server connection — tell the user that instead.
                fileLogger.w(TAG, "doSearch: SERVER UNAVAILABLE for query=\"$query\"")
                _state.value = FinderState.Error("Finder server not connected. Check your connection and try again.")
                feedback.speakSystem("Finder isn't connected to the server right now. Try again in a moment.")
            }
        }
    }
}
