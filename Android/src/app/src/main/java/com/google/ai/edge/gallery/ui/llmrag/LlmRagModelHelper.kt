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
import com.google.ai.edge.localagents.rag.memory.DefaultVectorStore
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

// Real RAG SDK is now available from local source

private const val TAG = "AGLlmRagModelHelper"

// Default paths for embedder models (users need to push these to device)
private const val GECKO_MODEL_PATH = "/data/local/tmp/gecko.tflite"
private const val TOKENIZER_MODEL_PATH = "/data/local/tmp/sentencepiece.model"

// RAG configuration constants
private const val USE_GPU_FOR_EMBEDDINGS = true
private const val EMBEDDING_DIMENSION = 768
private const val QA_PROMPT_TEMPLATE = """Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""

// Using real RAG SDK implementation now

data class RagModelInstance(
  val llmInstance: LlmModelInstance,
  val ragChain: RetrievalAndInferenceChain,
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
            
            // Set up embedder (Gecko model)
            val embedder = GeckoEmbeddingModel(
              GECKO_MODEL_PATH,
              Optional.of(TOKENIZER_MODEL_PATH),
              USE_GPU_FOR_EMBEDDINGS,
            )
            
            // Create semantic memory
            val semanticMemory = DefaultSemanticTextMemory(
              DefaultVectorStore<String>(),
              embedder
            )
            
            // Create RAG chain configuration
            val config = ChainConfig.create(
              mediaPipeLanguageModel,
              PromptBuilder(QA_PROMPT_TEMPLATE),
              semanticMemory
            )
            
            // Create retrieval and inference chain
            val ragChain = RetrievalAndInferenceChain(config)
            
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
        // Use coroutines to await the ListenableFuture
        val future = semanticMemory.recordBatchedMemoryItems(ImmutableList.copyOf(chunks))
        val result = async(Dispatchers.IO) { future.get() }
        result.await()
      } else {
        // Fallback to individual memorization using recordMemoryItem
        for (chunk in chunks) {
          // Use coroutines to await the ListenableFuture
          val future = semanticMemory?.recordMemoryItem(chunk)
          val result = async(Dispatchers.IO) { future?.get() }
          result.await()
        }
      }
      
      Log.d(TAG, "Successfully memorized ${chunks.size} chunks")
      ""
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
      
      val retrievalRequest = RetrievalRequest.create(
        prompt,
        RetrievalConfig.create(2, 0.0f, RetrievalConfig.TaskType.QUESTION_ANSWERING)
      )
      
      // Use coroutines to await the ListenableFuture
      val future = ragInstance.ragChain.invoke(retrievalRequest, callback)
      val result = async(Dispatchers.IO) { future.get() }
      result.await().text
    } catch (e: Exception) {
      val error = "Failed to generate response: ${e.message}"
      Log.e(TAG, error)
      error
    }
  }

  fun clearContext(model: Model) {
    try {
      val ragInstance = model.instance as RagModelInstance
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
}
