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
import com.google.ai.edge.gallery.ui.common.chat.ChatMessageLoading
import com.google.ai.edge.gallery.ui.common.chat.ChatMessageText
import com.google.ai.edge.gallery.ui.common.chat.ChatSide
import com.google.ai.edge.gallery.ui.common.chat.ChatViewModel
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

@HiltViewModel
class LlmRagViewModel @Inject constructor() : ChatViewModel() {

  private var memorizedChunksCount = 0
  
  // Document browsing state
  private val _storedDocuments = MutableStateFlow<List<LlmRagModelHelper.DocumentMetadata>>(emptyList())
  val storedDocuments: StateFlow<List<LlmRagModelHelper.DocumentMetadata>> = _storedDocuments.asStateFlow()
  
  private val _isLoadingDocuments = MutableStateFlow(false)
  val isLoadingDocuments: StateFlow<Boolean> = _isLoadingDocuments.asStateFlow()
  
  // Track retrieved documents for the current query
  private val _retrievedDocuments = MutableStateFlow<List<String>>(emptyList())
  val retrievedDocuments: StateFlow<List<String>> = _retrievedDocuments.asStateFlow()

  fun memorizeText(model: Model, text: String, title: String = "Uploaded Document", source: String = "upload") {
    viewModelScope.launch {
      try {
        addSystemMessage(model, "Processing text for memorization...")
        
        // Simple chunking strategy: split by paragraphs and sentences
        val chunks = chunkText(text)
        
        val error = withContext(Dispatchers.Default) {
          LlmRagModelHelper.memorizeChunks(model, chunks, title, source)
        }
        
        if (error.isEmpty()) {
          memorizedChunksCount += chunks.size
          addSystemMessage(
            model, 
            "Successfully memorized ${chunks.size} chunks. Total chunks in memory: $memorizedChunksCount"
          )
          // Refresh document list
          refreshStoredDocuments()
        } else {
          addSystemMessage(model, "Error memorizing text: $error")
        }
      } catch (e: Exception) {
        Log.e(TAG, "Failed to memorize text: ${e.message}")
        addSystemMessage(model, "Failed to memorize text: ${e.message}")
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
          LlmRagModelHelper.memorizeChunks(model, chunks, "Image Descriptions", "image_analysis")
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
      val userMessage = ChatMessageText(content = textContent, side = ChatSide.USER)
      addMessage(model, userMessage)

      try {
        val assistantMessage = ChatMessageLoading()
        addMessage(model, assistantMessage)

        val progressListener = object : AsyncProgressListener<LanguageModelResponse> {
          override fun run(partialResult: LanguageModelResponse, done: Boolean) {
            // Update the loading message with partial result
            updateLastAssistantMessage(model, partialResult.text)
          }
        }

        val response = withContext(Dispatchers.Default) {
          LlmRagModelHelper.generateResponse(model, textContent, progressListener)
        }

        // Update retrieved documents after response generation
        val retrievalResult = LlmRagModelHelper.getLastRetrievalResult()
        _retrievedDocuments.value = retrievalResult?.sourceDocuments ?: emptyList()

        // Final update with complete response
        updateLastAssistantMessage(model, response)

      } catch (e: Exception) {
        Log.e(TAG, "Failed to send message: ${e.message}")
        updateLastAssistantMessage(model, "Error: ${e.message}")
      }
    }
  }

  fun clearAllRagMessages(model: Model) {
    clearAllMessages(model)
    memorizedChunksCount = 0
    _retrievedDocuments.value = emptyList()
    LlmRagModelHelper.clearContext(model)
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
          LlmRagModelHelper.getDocumentMetadataList()
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
  suspend fun getDocumentById(documentId: String): LlmRagModelHelper.StoredDocument? {
    return withContext(Dispatchers.Default) {
      LlmRagModelHelper.getDocumentById(documentId)
    }
  }
  
  /**
   * Delete a document
   */
  fun deleteDocument(documentId: String) {
    viewModelScope.launch {
      try {
        val deleted = withContext(Dispatchers.Default) {
          LlmRagModelHelper.deleteDocument(documentId)
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
  fun searchDocuments(query: String): List<LlmRagModelHelper.StoredDocument> {
    return LlmRagModelHelper.searchDocuments(query)
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

  private fun updateLastAssistantMessage(model: Model, text: String) {
    val lastMessage = getLastMessage(model)
    if (lastMessage != null && lastMessage.side == ChatSide.AGENT) {
      removeLastMessage(model)
      val updatedMessage = ChatMessageText(content = text, side = ChatSide.AGENT)
      addMessage(model, updatedMessage)
    }
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
