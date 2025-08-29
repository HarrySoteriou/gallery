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
import com.google.ai.edge.gallery.ui.common.chat.ChatViewModel
import com.google.ai.edge.gallery.ui.common.chat.MessageBody
import com.google.ai.edge.gallery.ui.modelmanager.ModelManagerViewModel
import com.google.ai.edge.localagents.rag.llm.LanguageModelResponse
import com.google.ai.edge.localagents.rag.util.AsyncProgressListener
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import javax.inject.Inject

private const val TAG = "AGLlmRagViewModel"

@HiltViewModel
class LlmRagViewModel @Inject constructor(
  modelManagerViewModel: ModelManagerViewModel,
) : ChatViewModel(modelManagerViewModel, BuiltInTaskId.LLM_RAG) {

  private var memorizedChunksCount = 0

  fun memorizeText(model: Model, text: String) {
    viewModelScope.launch {
      try {
        addSystemMessage(model, "Processing text for memorization...")
        
        // Simple chunking strategy: split by paragraphs and sentences
        val chunks = chunkText(text)
        
        val error = withContext(Dispatchers.Default) {
          LlmRagModelHelper.memorizeChunks(model, chunks)
        }
        
        if (error.isEmpty()) {
          memorizedChunksCount += chunks.size
          addSystemMessage(
            model, 
            "Successfully memorized ${chunks.size} chunks. Total chunks in memory: $memorizedChunksCount"
          )
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
          LlmRagModelHelper.memorizeChunks(model, chunks)
        }
        
        if (error.isEmpty()) {
          memorizedChunksCount += chunks.size
          addSystemMessage(
            model, 
            "Successfully memorized ${chunks.size} image descriptions. Total chunks in memory: $memorizedChunksCount"
          )
        } else {
          addSystemMessage(model, "Error memorizing image descriptions: $error")
        }
      } catch (e: Exception) {
        Log.e(TAG, "Failed to memorize image descriptions: ${e.message}")
        addSystemMessage(model, "Failed to memorize image descriptions: ${e.message}")
      }
    }
  }

  override fun sendMessage(model: Model, content: List<Any>) {
    val textContent = content.filterIsInstance<String>().joinToString(" ")
    if (textContent.isBlank()) return

    viewModelScope.launch {
      val userMessage = ChatMessage.createUserMessage(content)
      _messages.value = _messages.value + userMessage

      try {
        val assistantMessage = ChatMessage.createAssistantMessage(MessageBody.Loading())
        _messages.value = _messages.value + assistantMessage

        val progressListener = object : AsyncProgressListener<LanguageModelResponse> {
          override fun onProgress(partialResult: LanguageModelResponse) {
            // Update the loading message with partial result
            updateLastAssistantMessage(model, MessageBody.Text(partialResult.text))
          }
        }

        val response = withContext(Dispatchers.Default) {
          LlmRagModelHelper.generateResponse(model, textContent, progressListener)
        }

        // Final update with complete response
        updateLastAssistantMessage(model, MessageBody.Text(response))

      } catch (e: Exception) {
        Log.e(TAG, "Failed to send message: ${e.message}")
        updateLastAssistantMessage(model, MessageBody.Text("Error: ${e.message}"))
      }
    }
  }

  override fun clearAllMessages(model: Model) {
    super.clearAllMessages(model)
    memorizedChunksCount = 0
    LlmRagModelHelper.clearContext(model)
  }

  private fun addSystemMessage(model: Model, text: String) {
    val systemMessage = ChatMessage.createSystemMessage(MessageBody.Text(text))
    _messages.value = _messages.value + systemMessage
  }

  private fun updateLastAssistantMessage(model: Model, messageBody: MessageBody) {
    val currentMessages = _messages.value
    if (currentMessages.isNotEmpty()) {
      val lastMessage = currentMessages.last()
      if (lastMessage.isFromAssistant()) {
        val updatedMessage = lastMessage.copy(messageBody = messageBody)
        _messages.value = currentMessages.dropLast(1) + updatedMessage
      }
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
