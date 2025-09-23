package com.google.ai.edge.gallery.ui.videosummaryrag

import android.graphics.Bitmap
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.google.ai.edge.gallery.data.Model
import com.google.ai.edge.gallery.data.Task
import com.google.ai.edge.gallery.ui.llmchat.LlmChatModelHelper
import com.google.ai.edge.gallery.ui.llmrag.LlmRagModelHelper
import com.google.ai.edge.gallery.ui.videosummaryrag.data.VideoBatchRecord
import com.google.ai.edge.gallery.ui.videosummaryrag.data.VideoBatchRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import javax.inject.Inject
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.coroutines.suspendCoroutine
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

private const val TAG = "VideoSummaryRagVM"

@HiltViewModel
class VideoSummaryRagViewModel @Inject constructor(
  private val repository: VideoBatchRepository,
) : ViewModel() {

  data class VideoSummaryRagUiState(
    val isProcessing: Boolean = false,
    val lastBatch: VideoBatchRecord? = null,
    val history: List<VideoBatchRecord> = emptyList(),
    val errorMessage: String? = null,
  )

  private val _uiState = MutableStateFlow(VideoSummaryRagUiState())
  val uiState: StateFlow<VideoSummaryRagUiState> = _uiState.asStateFlow()

  private val _memorizationEvents = MutableSharedFlow<Unit>(extraBufferCapacity = 1)
  val memorizationEvents: SharedFlow<Unit> = _memorizationEvents.asSharedFlow()

  fun refreshHistory() {
    viewModelScope.launch(Dispatchers.IO) {
      val history = repository.listBatches()
      _uiState.update { it.copy(history = history) }
    }
  }

  fun clearError() {
    _uiState.update { it.copy(errorMessage = null) }
  }

  fun processBatch(
    task: Task,
    model: Model,
    ragTask: Task,
    ragModel: Model,
    frames: List<Bitmap>,
  ) {
    if (frames.isEmpty()) {
      _uiState.update { it.copy(errorMessage = "No frames captured. Capture frames before processing.") }
      return
    }

    if (model.instance == null) {
      _uiState.update { it.copy(errorMessage = "Vision model is still initializing. Please wait and try again.") }
      return
    }

    // Ensure the retrieval model is a properly initialized RAG instance.
    if (ragModel.instance == null || ragModel.instance !is com.google.ai.edge.gallery.ui.llmrag.RagModelInstance) {
      _uiState.update { it.copy(errorMessage = "RAG model is still initializing. Please wait and try again.") }
      return
    }

    viewModelScope.launch {
      _uiState.update { it.copy(isProcessing = true, errorMessage = null) }

      try {
        VideoSummaryRagMemoryManager.clearVisionContext(task, model)

        val prompt = buildSummaryPrompt(frames.size)
        val summaryText = withContext(Dispatchers.Default) {
          runVisionInference(model, prompt, frames)
        }.trim()

        if (summaryText.isEmpty()) {
          throw IllegalStateException("Model returned an empty summary")
        }

        val previousDocIds = withContext(Dispatchers.Default) {
          LlmRagModelHelper.getDocumentMetadataList().map { it.id }.toSet()
        }

        val title = buildBatchTitle()
        val ragError = withContext(Dispatchers.Default) {
          LlmRagModelHelper.memorizeChunks(
            model = ragModel,
            chunks = listOf("Video batch summary:\n$summaryText"),
            title = title,
            source = "video_summary",
          )
        }

        if (ragError.isNotEmpty()) {
          throw IllegalStateException(ragError)
        }

        val newMetadata = withContext(Dispatchers.Default) {
          LlmRagModelHelper.getDocumentMetadataList().firstOrNull { it.id !in previousDocIds }
        }

        val record = withContext(Dispatchers.IO) {
          repository.saveBatch(
            frameCount = frames.size,
            summary = summaryText,
            ragDocumentId = newMetadata?.id,
            ragModelName = ragModel.name,
            visionModelName = model.name,
          )
        }

        val updatedHistory = withContext(Dispatchers.IO) {
          repository.listBatches()
        }

        _uiState.update {
          it.copy(
            isProcessing = false,
            lastBatch = record,
            history = updatedHistory,
            errorMessage = null,
          )
        }

        _memorizationEvents.tryEmit(Unit)
      } catch (cancellation: CancellationException) {
        _uiState.update { it.copy(isProcessing = false) }
        throw cancellation
      } catch (throwable: Throwable) {
        Log.e(TAG, "Batch processing failed", throwable)
        _uiState.update {
          it.copy(
            isProcessing = false,
            errorMessage = throwable.message ?: "Failed to process batch",
          )
        }
        VideoSummaryRagMemoryManager.clearVisionContext(task, model)
      }
    }
  }

  private suspend fun runVisionInference(
    model: Model,
    prompt: String,
    frames: List<Bitmap>,
  ): String = suspendCoroutine { continuation ->
    val builder = StringBuilder()
    try {
      LlmChatModelHelper.runInference(
        model = model,
        input = prompt,
        images = frames,
        audioClips = emptyList(),
        resultListener = { partial, done ->
          if (partial.isNotEmpty()) {
            builder.append(partial)
          }
          if (done) {
            continuation.resume(builder.toString())
          }
        },
        cleanUpListener = {},
      )
    } catch (error: Throwable) {
      continuation.resumeWithException(error)
    }
  }

  private fun buildSummaryPrompt(frameCount: Int): String {
    return """
      You are assisting with summarising a batch of $frameCount video frames captured at 1 FPS.
      Produce a concise JSON object with the following shape:
      {
        "batch_summary": "<overall scene summary>",
        "highlights": ["<key observation 1>", "<key observation 2>"]
      }
      Focus on the most salient people, objects, and actions. Keep the batch summary under 3 sentences.
    """.trimIndent()
  }

  private fun buildBatchTitle(): String {
    val formatter = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US)
    return "Video Batch ${formatter.format(Date())}"
  }
}
