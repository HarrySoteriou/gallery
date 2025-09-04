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

import android.content.Context
import android.util.Log
import com.google.ai.edge.gallery.data.Model
import com.google.ai.edge.gallery.ui.llmchat.LlmChatModelHelper
import com.google.ai.edge.gallery.ui.llmchat.LlmModelInstance
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.async
import java.util.Optional
// RAG SDK imports from local source
import com.google.ai.edge.localagents.rag.chains.ChainConfig
import com.google.ai.edge.localagents.rag.chains.RetrievalAndInferenceChain
import com.google.ai.edge.localagents.rag.memory.DefaultSemanticTextMemory
import com.google.ai.edge.localagents.rag.memory.SqliteVectorStore
import com.google.ai.edge.localagents.rag.models.AsyncProgressListener
import com.google.ai.edge.localagents.rag.models.Embedder
import com.google.ai.edge.localagents.rag.models.GeckoEmbeddingModel
import com.google.ai.edge.localagents.rag.models.LanguageModelResponse
import com.google.ai.edge.localagents.rag.models.MediaPipeLlmBackend
import com.google.ai.edge.localagents.rag.prompt.PromptBuilder
import com.google.ai.edge.localagents.rag.retrieval.RetrievalConfig
import com.google.ai.edge.localagents.rag.retrieval.RetrievalRequest
import com.google.common.collect.ImmutableList
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.google.mediapipe.tasks.genai.llminference.LlmInferenceSession
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.last
import kotlinx.coroutines.guava.await

// Real RAG SDK is now available from local source

private const val TAG = "AGLlmRagModelHelper"

// Default paths for embedder models (using bundled assets)
private const val GECKO_MODEL_PATH = "file:///android_asset/Gecko_1024_quant.tflite"
private const val TOKENIZER_MODEL_PATH = "file:///android_asset/sentencepiece.model"

// RAG configuration constants
private const val USE_GPU_FOR_EMBEDDINGS = false
private const val EMBEDDING_DIMENSION = 768
private const val QA_PROMPT_TEMPLATE = """Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""

// Using real RAG SDK implementation now

data class RagModelInstance(
  val llmInstance: LlmModelInstance,
  val ragChain: RetrievalAndInferenceChain?,
  val embedder: Embedder<String>?,
  val semanticMemory: com.google.ai.edge.localagents.rag.memory.SemanticMemory<String>?
)

object LlmRagModelHelper {

  fun initialize(
    context: Context,
    model: Model,
    onDone: (String) -> Unit,
  ) {
    try {
      Log.d(TAG, "Initializing RAG model...")
      
      // First initialize the base LLM model
      LlmChatModelHelper.initialize(
        context = context,
        model = model,
        supportImage = false,
        supportAudio = false,
        onDone = { error ->
          if (error.isNotEmpty()) {
            onDone(error)
            return@initialize
          }
          
          try {
            // Get the LLM instance
            val llmInstance = model.instance as LlmModelInstance
            
            // Create MediaPipe language model wrapper for RAG
            // Note: MediaPipeLlmBackend constructor requires specific options
            // We need to create new options instead of using llmInstance properties that don't exist
            val options = LlmInference.LlmInferenceOptions.builder()
              .setModelPath(model.getPath(context))
              .setMaxTokens(128)
              .setPreferredBackend(LlmInference.Backend.GPU)
              .build()
            val sessionOptions = LlmInferenceSession.LlmInferenceSessionOptions.builder()
              .setTopK(40)
              .setTopP(0.95f)
              .setTemperature(0.8f)
              .build()
            val mediaPipeLanguageModel = MediaPipeLlmBackend(context, options, sessionOptions)
            
            // Set up embedder (Gecko model) - handle native library issues gracefully
            val embedder = try {
              GeckoEmbeddingModel(
                GECKO_MODEL_PATH,
                Optional.of(TOKENIZER_MODEL_PATH),
                USE_GPU_FOR_EMBEDDINGS,
              )
            } catch (e: UnsatisfiedLinkError) {
              Log.w(TAG, "Native embedding libraries not available, using fallback: ${e.message}")
              null
            } catch (e: Exception) {
              Log.w(TAG, "Failed to initialize Gecko embedder, using fallback: ${e.message}")
              null
            }
            
            // Create semantic memory - only if embedder is available
            val semanticMemory = if (embedder != null) {
              try {
                DefaultSemanticTextMemory(
                  SqliteVectorStore(EMBEDDING_DIMENSION),
                  embedder
                )
              } catch (e: Exception) {
                Log.w(TAG, "Failed to create semantic memory, using fallback: ${e.message}")
                null
              }
            } else {
              null
            }
            
            // Create RAG chain configuration - handle case where semantic memory is null
            val ragChain = if (semanticMemory != null) {
              try {
                val config = ChainConfig.create(
                  mediaPipeLanguageModel,
                  PromptBuilder(QA_PROMPT_TEMPLATE),
                  semanticMemory
                )
                RetrievalAndInferenceChain(config)
              } catch (e: Exception) {
                Log.w(TAG, "Failed to create RAG chain, falling back to basic LLM: ${e.message}")
                null
              }
            } else {
              Log.i(TAG, "No semantic memory available, RAG chain will be disabled")
              null
            }
            
            // Replace the model instance with our RAG instance
            model.instance = RagModelInstance(
              llmInstance = llmInstance,
              ragChain = ragChain,
              embedder = embedder,
              semanticMemory = semanticMemory
            )
            
            Log.d(TAG, "RAG model initialized successfully")
            onDone("")
            
          } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize RAG components: ${e.message}")
            onDone("Failed to initialize RAG: ${e.message}")
          }
        }
      )
      
    } catch (e: Exception) {
      Log.e(TAG, "Failed to initialize RAG model: ${e.message}")
      onDone("Failed to initialize RAG model: ${e.message}")
    }
  }

  fun cleanUp(model: Model, onDone: () -> Unit) {
    if (model.instance == null) {
      onDone()
      return
    }

    try {
      val ragInstance = model.instance as RagModelInstance
      
      // Clean up the underlying LLM instance
      model.instance = ragInstance.llmInstance
      LlmChatModelHelper.cleanUp(model, onDone)
      
      Log.d(TAG, "RAG model cleanup done.")
    } catch (e: Exception) {
      Log.e(TAG, "Failed to cleanup RAG model: ${e.message}")
      onDone()
    }
  }

  // Simple in-memory document store as fallback when native RAG is not available
  private val documentStore = mutableMapOf<String, List<String>>()
  
  suspend fun memorizeChunks(
    model: Model,
    chunks: List<String>
  ): String = coroutineScope {
    try {
      val ragInstance = model.instance as RagModelInstance
      
      Log.d(TAG, "Memorizing ${chunks.size} chunks...")
      
      // Use the stored semantic memory to record batched memory items
      val semanticMemory = ragInstance.semanticMemory
      
      if (semanticMemory != null) {
        try {
          // Use coroutines to await the ListenableFuture - following official example pattern
          semanticMemory.recordBatchedMemoryItems(ImmutableList.copyOf(chunks)).await()
          Log.d(TAG, "Successfully memorized ${chunks.size} chunks using semantic memory")
          ""
        } catch (e: Exception) {
          Log.w(TAG, "Semantic memory failed, using fallback: ${e.message}")
          // Fallback to simple in-memory storage
          val documentId = "doc_${System.currentTimeMillis()}"
          documentStore[documentId] = chunks
          Log.d(TAG, "Successfully memorized ${chunks.size} chunks using fallback storage")
          ""
        }
      } else {
        // Fallback to simple in-memory storage
        val documentId = "doc_${System.currentTimeMillis()}"
        documentStore[documentId] = chunks
        Log.d(TAG, "Successfully memorized ${chunks.size} chunks using fallback storage")
        ""
      }
    } catch (e: Exception) {
      val error = "Failed to memorize chunks: ${e.message}"
      Log.e(TAG, error)
      error
    }
  }

  suspend fun generateResponse(
    model: Model,
    prompt: String,
    callback: AsyncProgressListener<LanguageModelResponse>? = null
  ): String = coroutineScope {
    try {
      val ragInstance = model.instance as RagModelInstance
      
      if (ragInstance.ragChain != null) {
        try {
          val retrievalRequest = RetrievalRequest.create(
            prompt,
            RetrievalConfig.create(2, 0.0f, RetrievalConfig.TaskType.QUESTION_ANSWERING)
          )
          
          // Use coroutines to await the ListenableFuture - following official example pattern
          ragInstance.ragChain.invoke(retrievalRequest, callback).await().text
        } catch (e: Exception) {
          Log.w(TAG, "RAG chain failed, using fallback: ${e.message}")
          // Fall through to fallback implementation
          generateWithFallbackRAG(model, prompt, ragInstance)
        }
      } else {
        // Use fallback RAG implementation
        generateWithFallbackRAG(model, prompt, ragInstance)
      }
    } catch (e: Exception) {
      val error = "Failed to generate response: ${e.message}"
      Log.e(TAG, error)
      error
    }
  }
  
  private suspend fun generateWithFallbackRAG(
    model: Model,
    prompt: String, 
    ragInstance: RagModelInstance
  ): String {
    Log.i(TAG, "Using fallback RAG implementation")
    
    // Simple keyword-based retrieval from stored documents
    val relevantChunks = retrieveRelevantChunks(prompt, maxChunks = 3)
    
    val enhancedPrompt = if (relevantChunks.isNotEmpty()) {
      """Based on the following context information, please answer the question:

Context:
${relevantChunks.joinToString("\n\n")}

Question: $prompt

Answer:"""
    } else {
      prompt
    }
    
    // Generate response using the enhanced prompt with context
    return LlmChatModelHelper.generateResponseFlow(
      Model(name = model.name).apply { instance = ragInstance.llmInstance },
      enhancedPrompt
    ).last()
  }
  
  private fun retrieveRelevantChunks(query: String, maxChunks: Int = 3): List<String> {
    if (documentStore.isEmpty()) return emptyList()
    
    val queryWords = query.lowercase().split("\\s+".toRegex()).filter { it.length > 2 }
    val relevantChunks = mutableListOf<Pair<String, Int>>()
    
    // Simple keyword-based scoring
    documentStore.values.flatten().forEach { chunk ->
      val chunkLower = chunk.lowercase()
      val score = queryWords.count { word -> chunkLower.contains(word) }
      if (score > 0) {
        relevantChunks.add(Pair(chunk, score))
      }
    }
    
    // Return top chunks by relevance score
    return relevantChunks
      .sortedByDescending { it.second }
      .take(maxChunks)
      .map { it.first }
  }

  fun clearContext(model: Model) {
    try {
      val ragInstance = model.instance as RagModelInstance
      // Clear fallback document store
      documentStore.clear()
      Log.d(TAG, "Cleared document store")
      
      // Reset the underlying LLM session
      LlmChatModelHelper.resetSession(
        model = Model(name = model.name).apply { instance = ragInstance.llmInstance },
        supportImage = false,
        supportAudio = false
      )
    } catch (e: Exception) {
      Log.e(TAG, "Failed to clear RAG context: ${e.message}")
    }
  }
  
  fun getDocumentCount(): Int = documentStore.values.sumOf { it.size }
  
  fun clearDocuments() {
    documentStore.clear()
    Log.d(TAG, "Manually cleared all documents from store")
  }
}
