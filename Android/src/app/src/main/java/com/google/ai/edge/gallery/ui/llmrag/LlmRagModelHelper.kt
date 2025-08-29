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
import com.google.ai.edge.localagents.rag.chain.ChainConfig
import com.google.ai.edge.localagents.rag.chain.RetrievalAndInferenceChain
import com.google.ai.edge.localagents.rag.chain.RetrievalConfig
import com.google.ai.edge.localagents.rag.chain.RetrievalRequest
import com.google.ai.edge.localagents.rag.chain.TaskType
import com.google.ai.edge.localagents.rag.embedding.Embedder
import com.google.ai.edge.localagents.rag.embedding.GeckoEmbeddingModel
import com.google.ai.edge.localagents.rag.llm.LanguageModel
import com.google.ai.edge.localagents.rag.llm.LanguageModelResponse
import com.google.ai.edge.localagents.rag.llm.MediaPipeLanguageModel
import com.google.ai.edge.localagents.rag.memory.DefaultSemanticTextMemory
import com.google.ai.edge.localagents.rag.memory.PromptBuilder
import com.google.ai.edge.localagents.rag.store.SqliteVectorStore
import com.google.ai.edge.localagents.rag.util.AsyncProgressListener
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.coroutineScope
import java.util.Optional

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

data class RagModelInstance(
  val llmInstance: LlmModelInstance,
  val ragChain: RetrievalAndInferenceChain,
  val embedder: Embedder<String>
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
            val mediaPipeLanguageModel = MediaPipeLanguageModel(llmInstance.engine)
            
            // Set up embedder (Gecko model)
            val embedder = GeckoEmbeddingModel(
              GECKO_MODEL_PATH,
              Optional.of(TOKENIZER_MODEL_PATH),
              USE_GPU_FOR_EMBEDDINGS,
            )
            
            // Create RAG chain configuration
            val config = ChainConfig.create(
              mediaPipeLanguageModel,
              PromptBuilder(QA_PROMPT_TEMPLATE),
              DefaultSemanticTextMemory(
                SqliteVectorStore(EMBEDDING_DIMENSION),
                embedder
              )
            )
            
            // Create retrieval and inference chain
            val ragChain = RetrievalAndInferenceChain(config)
            
            // Replace the model instance with our RAG instance
            model.instance = RagModelInstance(
              llmInstance = llmInstance,
              ragChain = ragChain,
              embedder = embedder
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
      for (chunk in chunks) {
        ragInstance.ragChain.memorize(chunk)
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
        RetrievalConfig.create(2, 0.0f, TaskType.QUESTION_ANSWERING)
      )
      
      ragInstance.ragChain.invoke(retrievalRequest, callback).await().text
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
        model = Model().apply { instance = ragInstance.llmInstance },
        supportImage = false,
        supportAudio = false
      )
    } catch (e: Exception) {
      Log.e(TAG, "Failed to clear RAG context: ${e.message}")
    }
  }
}
