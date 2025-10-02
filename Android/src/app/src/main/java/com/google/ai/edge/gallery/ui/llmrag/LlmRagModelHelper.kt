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
import com.google.ai.edge.gallery.data.Accelerator
import com.google.ai.edge.gallery.data.ConfigKeys
import com.google.ai.edge.gallery.data.DEFAULT_MAX_TOKEN
import com.google.ai.edge.gallery.data.DEFAULT_TEMPERATURE
import com.google.ai.edge.gallery.data.DEFAULT_TOPK
import com.google.ai.edge.gallery.data.DEFAULT_TOPP
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
import com.google.ai.edge.localagents.rag.models.EmbedData
import com.google.ai.edge.localagents.rag.models.Embedder
import com.google.ai.edge.localagents.rag.models.GeckoEmbeddingModel
import com.google.ai.edge.localagents.rag.models.EmbeddingRequest
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
import kotlin.coroutines.suspendCoroutine
import kotlin.coroutines.resume
import java.util.concurrent.TimeUnit

// Real RAG SDK is now available from local source

private const val TAG = "AGLlmRagModelHelper"

// Embedding model paths (separate from the main LLM model)
// These are for the Gecko embedding model used for semantic search, NOT the Gemma3-1T-IT LLM
// Note: These will be dynamically resolved to the app's external files directory
private const val GECKO_EMBEDDING_MODEL_FILENAME = "Gecko_1024_quant.tflite"
private const val GECKO_TOKENIZER_FILENAME = "sentencepiece.model"

// RAG configuration constants
private fun useGpuForEmbeddings(model: Model): Boolean {
  val accelerator = model.getStringConfigValue(
    key = ConfigKeys.ACCELERATOR,
    defaultValue = Accelerator.GPU.label,
  )
  Log.d(TAG, "Embedding backend forced to CPU (requested=$accelerator)")
  return false
}

private fun createGeckoEmbedder(context: Context, model: Model): Embedder<String>? {
  val externalFilesDir = context.getExternalFilesDir(null)
  if (externalFilesDir == null) {
    Log.w(TAG, "External files directory is null; cannot initialize Gecko embedder")
    return null
  }

  val geckoModelPath = "${externalFilesDir.absolutePath}/$GECKO_EMBEDDING_MODEL_FILENAME"
  val tokenizerPath = "${externalFilesDir.absolutePath}/$GECKO_TOKENIZER_FILENAME"

  Log.d(TAG, "Looking for Gecko model at: $geckoModelPath")
  Log.d(TAG, "Looking for tokenizer at: $tokenizerPath")

  val geckoFile = java.io.File(geckoModelPath)
  val tokenizerFile = java.io.File(tokenizerPath)

  Log.d(TAG, "Gecko model file exists: ${geckoFile.exists()}")
  Log.d(TAG, "Tokenizer file exists: ${tokenizerFile.exists()}")

  if (!geckoFile.exists()) {
    Log.w(TAG, "Gecko model file not found at: $geckoModelPath")
    return null
  }
  if (!tokenizerFile.exists()) {
    Log.w(TAG, "Tokenizer file not found at: $tokenizerPath")
    return null
  }

  val wantsGpu = useGpuForEmbeddings(model)
  Log.d(TAG, "Attempting to initialize Gecko embedder (GPU=$wantsGpu)")

  val embedder = instantiateGeckoEmbedder(geckoModelPath, tokenizerPath, wantsGpu)
  if (embedder != null) {
    return embedder
  }

  if (wantsGpu) {
    Log.w(TAG, "Retrying Gecko embedder initialization on CPU after GPU failure")
    return instantiateGeckoEmbedder(geckoModelPath, tokenizerPath, false)
  }

  return null
}

private fun instantiateGeckoEmbedder(
  geckoModelPath: String,
  tokenizerPath: String,
  enableGpu: Boolean,
): GeckoEmbeddingModel? {
  return try {
    GeckoEmbeddingModel(
      geckoModelPath,
      Optional.of(tokenizerPath),
      enableGpu,
    )
  } catch (e: UnsatisfiedLinkError) {
    Log.w(
      TAG,
      "Native embedding libraries not available (${if (enableGpu) "GPU" else "CPU"}): ${e.message}",
    )
    null
  } catch (e: Exception) {
    Log.w(
      TAG,
      "Failed to initialize Gecko embedder (${if (enableGpu) "GPU" else "CPU"}): ${e.message}",
    )
    null
  }
}
// IMPORTANT: must match the Gecko embedder model's output dimension.
// The configured filename defaults to a 1024-dim embedding; probe the runtime value when possible.
private const val DEFAULT_EMBEDDING_DIMENSION = 1024
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
  val semanticMemory: com.google.ai.edge.localagents.rag.memory.SemanticMemory<String>?,
  val embeddingDimension: Int,
)

object LlmRagModelHelper {

  fun initialize(
    context: Context,
    model: Model,
    onDone: (String) -> Unit,
  ) {
    try {
      Log.d(TAG, "Initializing RAG model...")
      
      // First initialize the base LLM model - force CPU mode for consistency
      Log.d(TAG, "Initializing base LLM model with CPU mode")
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
          
          // Get the already-initialized LLM instance (Gemma3-1T-IT)
          val llmInstance = model.instance as? LlmModelInstance
          if (llmInstance == null) {
            Log.e(TAG, "LLM instance is null, RAG initialization failed")
            onDone("Failed to initialize RAG: LLM instance is null")
            return@initialize
          }
          
          var embeddingDimension = DEFAULT_EMBEDDING_DIMENSION

          try {
           // Create MediaPipe language model wrapper for RAG
           // Note: We need to create options for the RAG backend, but we'll use the model's configuration
            val maxTokens = model.getIntConfigValue(key = ConfigKeys.MAX_TOKENS, defaultValue = DEFAULT_MAX_TOKEN)
            val topK = model.getIntConfigValue(key = ConfigKeys.TOPK, defaultValue = DEFAULT_TOPK)
            val topP = model.getFloatConfigValue(key = ConfigKeys.TOPP, defaultValue = DEFAULT_TOPP)
            val temperature = model.getFloatConfigValue(key = ConfigKeys.TEMPERATURE, defaultValue = DEFAULT_TEMPERATURE)
            val accelerator = model.getStringConfigValue(key = ConfigKeys.ACCELERATOR, defaultValue = Accelerator.GPU.label)
            
            // Force CPU mode for RAG initialization to avoid GPU-related issues
            val preferredBackend = LlmInference.Backend.CPU
            Log.d(TAG, "Forcing CPU mode for RAG initialization (original accelerator: $accelerator)")
            
            // Create options for the RAG MediaPipe backend using the model's configuration
            val modelPath = model.getPath(context)
            Log.d(TAG, "Creating RAG LLM options with model path: $modelPath")
            val ragLlmOptions = LlmInference.LlmInferenceOptions.builder()
              .setModelPath(modelPath)
              .setMaxTokens(maxTokens)
              .setPreferredBackend(preferredBackend)
              .build()
            val ragSessionOptions = LlmInferenceSession.LlmInferenceSessionOptions.builder()
              .setTopK(topK)
              .setTopP(topP)
              .setTemperature(temperature)
              .build()
            val mediaPipeLanguageModel = try {
              Log.d(TAG, "Creating MediaPipe LLM backend with CPU mode...")
              MediaPipeLlmBackend(context, ragLlmOptions, ragSessionOptions)
            } catch (e: UnsatisfiedLinkError) {
              Log.w(TAG, "MediaPipe backend native libs missing, RAG chain disabled: ${e.message}")
              null
            } catch (e: Exception) {
              Log.w(TAG, "Failed to create MediaPipe backend, RAG chain disabled: ${e.message}")
              null
            }
            // Set up embedder (Gecko embedding model - separate from Gemma3-1T-IT LLM)
            val embedder = createGeckoEmbedder(context, model)

            // Derive the semantic memory vector store dimension from the embedder when available.
            if (embedder != null) {
              embeddingDimension = determineEmbeddingDimension(embedder)
            }

            Log.d(TAG, "Using embedding dimension: $embeddingDimension")

            // Create semantic memory - only if embedder is available
            val semanticMemory = if (embedder != null) {
              try {
                DefaultSemanticTextMemory(
                  SqliteVectorStore(embeddingDimension),
                  embedder
                )
              } catch (e: UnsatisfiedLinkError) {
                Log.w(TAG, "Semantic memory native libs missing, using fallback: ${e.message}")
                null
              } catch (e: Exception) {
                Log.w(TAG, "Failed to create semantic memory, using fallback: ${e.message}")
                null
              }
            } else {
              null
            }

            // Create RAG chain configuration - handle case where semantic memory is null
            val ragChain = if (semanticMemory != null && mediaPipeLanguageModel != null) {
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
              if (semanticMemory == null) {
                Log.i(TAG, "No semantic memory available, RAG chain will be disabled")
              } else {
                Log.i(TAG, "MediaPipe backend unavailable, RAG chain will be disabled")
              }
              null
            }
            
            // Replace the model instance with our RAG instance
            Log.d(TAG, "Creating RAG instance with components - LLM: ${llmInstance != null}, RAG Chain: ${ragChain != null}, Embedder: ${embedder != null}, Semantic Memory: ${semanticMemory != null}")
            model.instance = RagModelInstance(
              llmInstance = llmInstance,
              ragChain = ragChain,
              embedder = embedder,
              semanticMemory = semanticMemory,
              embeddingDimension = embeddingDimension,
            )
            
            Log.d(TAG, "RAG model initialized successfully with CPU mode")
            onDone("")
            
          } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize RAG components: ${e.message}")
            // Even if RAG fails, we can still use basic LLM functionality
            // Create a minimal RagModelInstance with null RAG components
            try {
              model.instance = RagModelInstance(
                llmInstance = llmInstance,
                ragChain = null,
                embedder = null,
                semanticMemory = null,
                embeddingDimension = embeddingDimension,
              )
              Log.i(TAG, "RAG initialization failed, but basic LLM functionality is available")
              onDone("") // Don't report as error, just use fallback
            } catch (e2: Exception) {
              Log.e(TAG, "Failed to create fallback RAG instance: ${e2.message}")
              onDone("Failed to initialize RAG: ${e.message}")
            }
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
      // Check if model instance is properly initialized as RagModelInstance
      val ragInstance = try {
        model.instance as? RagModelInstance
      } catch (e: ClassCastException) {
        Log.w(TAG, "Model instance is not RagModelInstance during cleanup: ${e.message}")
        null
      }
      
      if (ragInstance != null) {
        // Clean up the underlying LLM instance
        model.instance = ragInstance.llmInstance
        LlmChatModelHelper.cleanUp(model, onDone)
      } else {
        // If RAG instance is not available, clean up the model directly
        LlmChatModelHelper.cleanUp(model, onDone)
      }
      
      Log.d(TAG, "RAG model cleanup done.")
    } catch (e: Exception) {
      Log.e(TAG, "Failed to cleanup RAG model: ${e.message}")
      onDone()
    }
  }

  private fun determineEmbeddingDimension(embedder: Embedder<String>): Int {
    return try {
      val request = EmbeddingRequest.create(
        ImmutableList.of(
          EmbedData.create(
            "dimension_probe",
            EmbedData.TaskType.SEMANTIC_SIMILARITY,
          )
        )
      )

      val embeddings = embedder.getEmbeddings(request).get(3, TimeUnit.SECONDS)
      val dimension = embeddings.size
      if (dimension > 0) dimension else DEFAULT_EMBEDDING_DIMENSION
    } catch (e: InterruptedException) {
      Thread.currentThread().interrupt()
      Log.w(TAG, "Embedding dimension probe interrupted, using default", e)
      DEFAULT_EMBEDDING_DIMENSION
    } catch (e: Exception) {
      Log.w(TAG, "Unable to probe embedding dimension, falling back to default: ${e.message}")
      DEFAULT_EMBEDDING_DIMENSION
    }
  }

  fun getResolvedEmbeddingDimension(model: Model): Int? {
    val ragInstance = model.instance as? RagModelInstance ?: return null
    return ragInstance.embeddingDimension.takeIf { it > 0 }
  }


  suspend fun generateResponse(
    model: Model,
    prompt: String,
    callback: AsyncProgressListener<LanguageModelResponse>? = null
  ): String = coroutineScope {
    try {
      // Check if model instance is properly initialized as RagModelInstance
      val ragInstance = try {
        model.instance as? RagModelInstance
      } catch (e: ClassCastException) {
        Log.w(TAG, "Model instance is not RagModelInstance, using basic LLM: ${e.message}")
        null
      }
      
      if (ragInstance?.ragChain != null) {
        try {
          val retrievalRequest = RetrievalRequest.create(
            prompt,
            RetrievalConfig.create(2, 0.0f, RetrievalConfig.TaskType.QUESTION_ANSWERING)
          )
          
          // Use coroutines to await the ListenableFuture - following official example pattern
          val resp = ragInstance.ragChain.invoke(retrievalRequest, callback).await()
          val text = resp.text?.trim() ?: ""
          // Guard against empty/"null" responses; fall back to local retrieval prompt building.
          if (text.isEmpty() || text.equals("null", ignoreCase = true)) {
            Log.w(TAG, "RAG chain returned empty/null text; falling back to keyword retrieval")
            generateWithFallbackRAG(model, prompt, ragInstance)
          } else {
            text
          }
        } catch (e: Exception) {
          Log.w(TAG, "RAG chain failed, using fallback: ${e.message}")
          // Fall through to fallback implementation
          generateWithFallbackRAG(model, prompt, ragInstance)
        }
      } else if (ragInstance != null) {
        // Use fallback RAG implementation
        generateWithFallbackRAG(model, prompt, ragInstance)
      } else {
        // RAG not available, use basic LLM response with fallback retrieval
        Log.i(TAG, "RAG instance not available, using basic LLM with fallback retrieval")
        generateWithBasicLLM(model, prompt)
      }
    } catch (e: Exception) {
      val error = "Failed to generate response: ${e.message}"
      Log.e(TAG, error)
      error
    }
  }
  
  private suspend fun generateWithBasicLLM(
    model: Model,
    prompt: String
  ): String {
    Log.i(TAG, "Using basic LLM without RAG")
    
    // Simple keyword-based retrieval from stored documents for context
    val retrievalResult = RagKnowledgeBase.retrieveRelevantChunks(prompt, maxChunks = 3)
    
    val enhancedPrompt = if (retrievalResult.chunks.isNotEmpty()) {
      val contextInfo = retrievalResult.chunks.joinToString("\n\n") { chunk ->
        "- $chunk"
      }
      Log.d(TAG, "Using enhanced prompt with context from documents: ${retrievalResult.sourceDocuments.joinToString(", ")}")
      """Based on the following context information, please provide a comprehensive answer to the question:

Context Information:
$contextInfo

Question: $prompt

Please provide a detailed answer based on the context provided above. If the context doesn't contain relevant information, please state that clearly.

Answer:"""
    } else {
      Log.d(TAG, "No relevant context found, using original prompt")
      prompt
    }
    
    // Generate response using the basic LLM - need to extract the underlying LlmModelInstance
    return suspendCoroutine { continuation ->
      val resultBuilder = StringBuilder()
      
      try {
        // Try to get the underlying LLM instance from RAG instance or use model directly
        val llmModel = try {
          val ragInstance = model.instance as? RagModelInstance
          if (ragInstance != null) {
            // Create a temporary model with the underlying LLM instance
            Model(name = model.name).apply { instance = ragInstance.llmInstance }
          } else {
            model
          }
        } catch (e: Exception) {
          Log.w(TAG, "Failed to extract LLM instance, using model directly: ${e.message}")
          model
        }
        
        LlmChatModelHelper.runInference(
          model = llmModel,
          input = enhancedPrompt,
          resultListener = { partialResult, done ->
            resultBuilder.append(partialResult)
            if (done) {
              continuation.resume(resultBuilder.toString())
            }
          },
          cleanUpListener = { /* No cleanup needed for this temporary call */ }
        )
      } catch (e: Exception) {
        Log.e(TAG, "Failed to run basic LLM inference: ${e.message}")
        continuation.resume("Error: Unable to generate response. ${e.message}")
      }
    }
  }
  
  private suspend fun generateWithFallbackRAG(
    model: Model,
    prompt: String, 
    ragInstance: RagModelInstance
  ): String {
    Log.i(TAG, "Using fallback RAG implementation")
    
    // Simple keyword-based retrieval from stored documents
    val retrievalResult = RagKnowledgeBase.retrieveRelevantChunks(prompt, maxChunks = 5)
    Log.d(TAG, "Retrieved ${retrievalResult.chunks.size} relevant chunks from ${retrievalResult.sourceDocuments.size} documents for query: $prompt")
    
    val enhancedPrompt = if (retrievalResult.chunks.isNotEmpty()) {
      val contextInfo = retrievalResult.chunks.joinToString("\n\n") { chunk ->
        "- $chunk"
      }
      Log.d(TAG, "Using enhanced prompt with context from documents: ${retrievalResult.sourceDocuments.joinToString(", ")}")
      """Based on the following context information, please provide a comprehensive answer to the question:

Context Information:
$contextInfo

Question: $prompt

Please provide a detailed answer based on the context provided above. If the context doesn't contain relevant information, please state that clearly.

Answer:"""
    } else {
      Log.d(TAG, "No relevant context found, using original prompt")
      """I don't have any specific context information in my knowledge base to answer this question. 

Question: $prompt

Answer: I would need more context or documents to be uploaded to provide a specific answer to this question."""
    }
    
    // Generate response using the enhanced prompt with context
    return suspendCoroutine { continuation ->
      val resultBuilder = StringBuilder()
      val tempModel = Model(name = model.name).apply { instance = ragInstance.llmInstance }
      
      LlmChatModelHelper.runInference(
        model = tempModel,
        input = enhancedPrompt,
        resultListener = { partialResult, done ->
          resultBuilder.append(partialResult)
          if (done) {
            continuation.resume(resultBuilder.toString())
          }
        },
        cleanUpListener = { /* No cleanup needed for this temporary call */ }
      )
    }
  }
  
}
