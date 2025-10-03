/*
 * Copyright 2025 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.ai.edge.gallery.ui.llmrag

import android.util.Log
import androidx.lifecycle.viewModelScope
import com.google.ai.edge.gallery.data.BuiltInTaskId
import com.google.ai.edge.gallery.data.Model
import com.google.ai.edge.gallery.ui.common.chat.ChatMessage
import com.google.ai.edge.gallery.ui.common.chat.ChatMessageBenchmarkLlmResult
import com.google.ai.edge.gallery.ui.common.chat.ChatMessageLoading
import com.google.ai.edge.gallery.ui.common.chat.ChatMessageText
import com.google.ai.edge.gallery.ui.common.chat.ChatMessageType
import com.google.ai.edge.gallery.ui.common.chat.ChatSide
import com.google.ai.edge.gallery.ui.common.chat.ChatViewModel
import com.google.ai.edge.gallery.ui.common.chat.Stat
import com.google.ai.edge.gallery.ui.llmchat.LlmModelInstance
import com.google.ai.edge.localagents.rag.models.AsyncProgressListener
import com.google.ai.edge.localagents.rag.models.LanguageModelResponse
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import javax.inject.Inject

private const val TAG = "AGLlmRagViewModel"

private val LLM_STATS =
  listOf(
    Stat(id = "time_to_first_token", label = "1st token", unit = "sec"),
    Stat(id = "prefill_speed", label = "Prefill speed", unit = "tokens/s"),
    Stat(id = "decode_speed", label = "Decode speed", unit = "tokens/s"),
    Stat(id = "latency", label = "Latency", unit = "sec"),
  )

@HiltViewModel
class LlmRagViewModel @Inject constructor() : ChatViewModel() {

  private var memorizedChunksCount = 0
  
  // Document browsing state
  private val _storedDocuments = MutableStateFlow<List<RagKnowledgeBase.DocumentMetadata>>(emptyList())
  val storedDocuments: StateFlow<List<RagKnowledgeBase.DocumentMetadata>> = _storedDocuments.asStateFlow()
  
  private val _isLoadingDocuments = MutableStateFlow(false)
  val isLoadingDocuments: StateFlow<Boolean> = _isLoadingDocuments.asStateFlow()
  
  // Track retrieved documents for the current query
  private val _retrievedDocuments = MutableStateFlow<List<String>>(emptyList())
  val retrievedDocuments: StateFlow<List<String>> = _retrievedDocuments.asStateFlow()

  fun memorizeText(model: Model, text: String, title: String = "Uploaded Document", source: String = "upload") {
    viewModelScope.launch {
      try {
        Log.d(TAG, "Starting memorization for document '$title' with model: ${model.name}")
        Log.d(TAG, "Model instance type: ${model.instance?.javaClass?.simpleName}")
        addSystemMessage(model, "Processing document '$title' for memorization...")
        
        // Validate input
        if (text.isBlank()) {
          addSystemMessage(model, "Error: Document is empty or contains no text")
          return@launch
        }
        
        // Simple chunking strategy: split by paragraphs and sentences
        val chunks = chunkText(text)
        Log.d(TAG, "Created ${chunks.size} chunks from document '$title'")
        
        val error = withContext(Dispatchers.Default) {
          RagKnowledgeBase.memorizeChunks(model, chunks, title, source)
        }
        
        if (error.isEmpty()) {
          memorizedChunksCount += chunks.size
          addSystemMessage(
            model, 
            "✅ Successfully memorized '${title}' with ${chunks.size} chunks. Total chunks in memory: $memorizedChunksCount"
          )
          // Refresh document list
          refreshStoredDocuments()
          Log.d(TAG, "Successfully memorized document '$title' with ${chunks.size} chunks")
        } else {
          addSystemMessage(model, "❌ Error memorizing '$title': $error")
          Log.e(TAG, "Failed to memorize document '$title': $error")
        }
      } catch (e: Exception) {
        val errorMessage = "Failed to memorize document '$title': ${e.message}"
        Log.e(TAG, errorMessage)
        addSystemMessage(model, "❌ $errorMessage")
      }
    }
  }

  fun memorizeImageDescriptions(model: Model, descriptions: List<String>) {
    viewModelScope.launch {
      try {
        addSystemMessage(model, "Processing ${descriptions.size} image descriptions for memorization...")
        
        // Prepare chunks with context
        val chunks = descriptions.mapIndexed { index, description ->
          "Image ${index + 1}: $description"
        }
        
        val error = withContext(Dispatchers.Default) {
          RagKnowledgeBase.memorizeChunks(model, chunks, "Image Descriptions", "image_analysis")
        }
        
        if (error.isEmpty()) {
          memorizedChunksCount += chunks.size
          addSystemMessage(
            model, 
            "Successfully memorized ${chunks.size} image descriptions. Total chunks in memory: $memorizedChunksCount"
          )
          // Refresh document list
          refreshStoredDocuments()
        } else {
          addSystemMessage(model, "Error memorizing image descriptions: $error")
        }
      } catch (e: Exception) {
        Log.e(TAG, "Failed to memorize image descriptions: ${e.message}")
        addSystemMessage(model, "Failed to memorize image descriptions: ${e.message}")
      }
    }
  }

  fun sendMessage(model: Model, content: List<Any>) {
    val textContent = content.filterIsInstance<String>().joinToString(" ")
    if (textContent.isBlank()) return

    viewModelScope.launch {
      setInProgress(true)
      val userMessage = ChatMessageText(content = textContent, side = ChatSide.USER)
      addMessage(model, userMessage)

      try {
        val assistantMessage = ChatMessageLoading()
        addMessage(model, assistantMessage)

        val accelerator =
          model.getStringConfigValue(key = com.google.ai.edge.gallery.data.ConfigKeys.ACCELERATOR, defaultValue = "")
        val llmSession =
          when (val instance = (model.instance as? RagModelInstance)?.llmInstance ?: model.instance) {
            is RagModelInstance -> instance.llmInstance.session
            is LlmModelInstance -> instance.session
            else -> null
          }

        val prefillTokens = llmSession?.sizeInTokens(textContent) ?: 0
        val start = System.currentTimeMillis()
        var firstTokenTs = 0L
        var timeToFirstToken = 0f
        var prefillSpeed = 0f
        var decodeTokens = 0

        val progressListener = object : AsyncProgressListener<LanguageModelResponse> {
          override fun run(partialResult: LanguageModelResponse, done: Boolean) {
            // Some backends may emit null or empty text in partials; guard it.
            val text = partialResult.text
            if (!text.isNullOrBlank()) {
              val now = System.currentTimeMillis()
              if (firstTokenTs == 0L) {
                firstTokenTs = now
                timeToFirstToken = ((now - start).coerceAtLeast(0L)).toFloat() / 1000f
                if (timeToFirstToken > 0f && prefillTokens > 0) {
                  prefillSpeed = prefillTokens / timeToFirstToken
                }
              }

              // Track generated tokens using total text tokens to approximate decode speed.
              val totalTokens = llmSession?.sizeInTokens(text) ?: 0
              if (totalTokens > 0) {
                decodeTokens = totalTokens
              }

              updateLastAssistantMessage(
                model = model,
                text = text,
                latencyMs = -1f,
                accelerator = accelerator,
              )
            }
          }
        }

        val response = withContext(Dispatchers.Default) {
          LlmRagModelHelper.generateResponse(model, textContent, progressListener)
        }

        // Update retrieved documents after response generation
        val retrievalResult = RagKnowledgeBase.getLastRetrievalResult()
        _retrievedDocuments.value = retrievalResult?.sourceDocuments ?: emptyList()

        // Final update with complete response
        val endTs = System.currentTimeMillis()
        val totalLatencySec = ((endTs - start).coerceAtLeast(0L)).toFloat() / 1000f
        if (timeToFirstToken == 0f) {
          // No partial updates received; treat entire latency as prefill time.
          timeToFirstToken = totalLatencySec
          if (timeToFirstToken > 0f && prefillTokens > 0) {
            prefillSpeed = prefillTokens / timeToFirstToken
          }
        }

        val decodeDurationSec =
          if (firstTokenTs == 0L) 0f else ((endTs - firstTokenTs).coerceAtLeast(0L)).toFloat() / 1000f
        val decodeSpeed = if (decodeDurationSec > 0f && decodeTokens > 0) {
          decodeTokens / decodeDurationSec
        } else {
          0f
        }

        val benchmark =
          ChatMessageBenchmarkLlmResult(
            orderedStats = LLM_STATS,
            statValues =
              mutableMapOf(
                "prefill_speed" to prefillSpeed,
                "decode_speed" to decodeSpeed,
                "time_to_first_token" to timeToFirstToken,
                "latency" to totalLatencySec,
              ),
            running = false,
            latencyMs = -1f,
            accelerator = accelerator,
          )

        updateLastAssistantMessage(
          model = model,
          text = response,
          latencyMs = (endTs - start).toFloat(),
          accelerator = accelerator,
          llmBenchmarkResult = benchmark,
        )

      } catch (e: Exception) {
        Log.e(TAG, "Failed to send message: ${e.message}")
        updateLastAssistantMessage(
          model = model,
          text = "Error: ${e.message}",
          latencyMs = -1f,
        )
      } finally {
        RagContextManager.clearChatTurn(model)
        setInProgress(false)
      }
    }
  }

  fun clearAllRagMessages(model: Model) {
    clearAllMessages(model)
    memorizedChunksCount = 0
    _retrievedDocuments.value = emptyList()
    RagContextManager.clearChatTurn(model, dropPersistentMemory = true)
    refreshStoredDocuments()
  }
  
  /**
   * Refresh the list of stored documents
   */
  fun refreshStoredDocuments() {
    viewModelScope.launch {
      _isLoadingDocuments.value = true
      try {
        val documents = withContext(Dispatchers.Default) {
          RagKnowledgeBase.getDocumentMetadataList()
        }
        _storedDocuments.value = documents
      } catch (e: Exception) {
        Log.e(TAG, "Failed to refresh stored documents: ${e.message}")
      } finally {
        _isLoadingDocuments.value = false
      }
    }
  }
  
  /**
   * Get a specific document by ID
   */
  suspend fun getDocumentById(documentId: String): RagKnowledgeBase.StoredDocument? {
    return try {
      withContext(Dispatchers.Default) {
        RagKnowledgeBase.getDocumentById(documentId)
      }
    } catch (e: Exception) {
      Log.e(TAG, "Error retrieving document $documentId: ${e.message}")
      null
    }
  }
  
  /**
   * Delete a document
   */
  fun deleteDocument(documentId: String) {
    viewModelScope.launch {
      try {
        val deleted = withContext(Dispatchers.Default) {
          RagKnowledgeBase.deleteDocument(documentId)
        }
        if (deleted) {
          refreshStoredDocuments()
        }
      } catch (e: Exception) {
        Log.e(TAG, "Failed to delete document: ${e.message}")
      }
    }
  }
  
  /**
   * Search documents
   */
  fun searchDocuments(query: String): List<RagKnowledgeBase.StoredDocument> {
    return RagKnowledgeBase.searchDocuments(query)
  }
  
  /**
   * Clear retrieved documents display
   */
  fun clearRetrievedDocuments() {
    _retrievedDocuments.value = emptyList()
  }

  private fun addSystemMessage(model: Model, text: String) {
    val systemMessage = ChatMessageText(content = text, side = ChatSide.SYSTEM)
    addMessage(model, systemMessage)
  }

  private fun updateLastAssistantMessage(
    model: Model,
    text: String,
    latencyMs: Float = -1f,
    accelerator: String = "",
    llmBenchmarkResult: ChatMessageBenchmarkLlmResult? = null,
  ) {
    val lastMessage = getLastMessage(model)
    val message =
      ChatMessageText(
        content = text,
        side = ChatSide.AGENT,
        latencyMs = latencyMs,
        accelerator = accelerator,
      ).apply { this.llmBenchmarkResult = llmBenchmarkResult }

    if (lastMessage?.type == ChatMessageType.LOADING || lastMessage is ChatMessageText && lastMessage.side == ChatSide.AGENT) {
      removeLastMessage(model)
    }
    addMessage(model, message)
  }

  /**
   * Simple text chunking strategy.
   * Splits text by paragraphs and further by sentences if paragraphs are too long.
   */
  private fun chunkText(text: String, maxChunkSize: Int = 500): List<String> {
    val paragraphs = text.split("\n\n").filter { it.trim().isNotEmpty() }
    val chunks = mutableListOf<String>()
    
    for (paragraph in paragraphs) {
      if (paragraph.length <= maxChunkSize) {
        chunks.add(paragraph.trim())
      } else {
        // Split long paragraphs by sentences
        val sentences = paragraph.split(". ").filter { it.trim().isNotEmpty() }
        var currentChunk = ""
        
        for (sentence in sentences) {
          val sentenceWithPeriod = if (sentence.endsWith(".")) sentence else "$sentence."
          
          if (currentChunk.isEmpty()) {
            currentChunk = sentenceWithPeriod
          } else if ((currentChunk + " " + sentenceWithPeriod).length <= maxChunkSize) {
            currentChunk += " $sentenceWithPeriod"
          } else {
            chunks.add(currentChunk)
            currentChunk = sentenceWithPeriod
          }
        }
        
        if (currentChunk.isNotEmpty()) {
          chunks.add(currentChunk)
        }
      }
    }
    
    return chunks
  }
}
