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

package com.google.ai.edge.gallery.ui.videoanalysis

import android.graphics.Bitmap
import android.util.Log
import com.google.ai.edge.gallery.data.Model
import com.google.ai.edge.gallery.ui.llmchat.LlmChatModelHelper
import com.google.ai.edge.gallery.ui.llmrag.LlmRagModelHelper
import com.google.ai.edge.gallery.ui.llmrag.RagModelInstance
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.text.SimpleDateFormat
import java.util.*

private const val TAG = "VideoAnalysisRagIntegration"

/**
 * Integration utility for using RAG with video analysis to enable querying of longer video sequences.
 * 
 * This allows the system to:
 * 1. Analyze video frames in batches using regular VLM
 * 2. Store frame descriptions in RAG memory
 * 3. Query across multiple batches of analyzed frames
 */
object VideoAnalysisRagIntegration {

  /**
   * Analyzes a batch of video frames and stores the descriptions in RAG memory.
   * 
   * @param vlmModel The vision-language model for analyzing frames
   * @param ragModel The RAG model for storing descriptions
   * @param frames The batch of frames to analyze
   * @param batchIndex The index of this batch (for tracking)
   * @param timestamp Optional timestamp for the batch
   * @return Pair of (analysis result, error message if any)
   */
  suspend fun analyzeAndMemorizeBatch(
    vlmModel: Model,
    ragModel: Model,
    frames: List<Bitmap>,
    batchIndex: Int,
    timestamp: String = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(Date())
  ): Pair<String, String> = withContext(Dispatchers.Default) {
    try {
      Log.d(TAG, "Analyzing batch $batchIndex with ${frames.size} frames")
      
      // Step 1: Analyze frames with VLM
      val analysisPrompt = buildVideoAnalysisPrompt()
      val analysisResult = analyzeFramesWithVLM(vlmModel, frames, analysisPrompt)
      
      if (analysisResult.startsWith("Error:")) {
        return@withContext Pair("", analysisResult)
      }
      
      // Step 2: Create structured description for RAG storage
      val ragDescription = buildRagDescription(batchIndex, timestamp, analysisResult, frames.size)
      
      // Step 3: Store in RAG memory
      val memorizeError = LlmRagModelHelper.memorizeChunks(ragModel, listOf(ragDescription))
      
      if (memorizeError.isNotEmpty()) {
        return@withContext Pair(analysisResult, "Failed to store in RAG: $memorizeError")
      }
      
      Log.d(TAG, "Successfully analyzed and memorized batch $batchIndex")
      return@withContext Pair(analysisResult, "")
      
    } catch (e: Exception) {
      val error = "Error analyzing batch $batchIndex: ${e.message}"
      Log.e(TAG, error)
      return@withContext Pair("", error)
    }
  }

  /**
   * Queries the RAG system about the analyzed video content.
   * 
   * @param ragModel The RAG model containing video descriptions
   * @param query The user's query about the video content
   * @return The response from RAG system
   */
  suspend fun queryVideoContent(
    ragModel: Model,
    query: String
  ): String = withContext(Dispatchers.Default) {
    try {
      Log.d(TAG, "Querying video content: $query")
      
      val response = LlmRagModelHelper.generateResponse(ragModel, query)
      Log.d(TAG, "RAG query completed successfully")
      
      return@withContext response
    } catch (e: Exception) {
      val error = "Error querying video content: ${e.message}"
      Log.e(TAG, error)
      return@withContext error
    }
  }

  /**
   * Clears all stored video analysis data from RAG memory.
   */
  fun clearVideoMemory(ragModel: Model) {
    try {
      LlmRagModelHelper.clearContext(ragModel)
      Log.d(TAG, "Cleared video analysis memory")
    } catch (e: Exception) {
      Log.e(TAG, "Error clearing video memory: ${e.message}")
    }
  }

  /**
   * Checks if a model is RAG-enabled.
   */
  fun isRagModel(model: Model): Boolean {
    return model.instance is RagModelInstance
  }

  /**
   * Gets summary of memorized video content.
   */
  suspend fun getVideoSummary(ragModel: Model): String {
    val summaryQuery = """
      Provide a comprehensive summary of all the video content that has been analyzed and stored. 
      Include:
      1. Number of video batches processed
      2. Key objects, people, and scenes detected across all batches
      3. Notable patterns or movements observed
      4. Overall timeline of events
      
      Please organize this as a structured summary.
    """.trimIndent()
    
    return queryVideoContent(ragModel, summaryQuery)
  }

  private suspend fun analyzeFramesWithVLM(
    vlmModel: Model,
    frames: List<Bitmap>,
    prompt: String
  ): String = withContext(Dispatchers.Default) {
    try {
      var result = ""
      var isComplete = false
      
      LlmChatModelHelper.runInference(
        model = vlmModel,
        input = prompt,
        resultListener = { partialResult, done ->
          result = partialResult
          isComplete = done
        },
        cleanUpListener = { },
        images = frames
      )
      
      // Wait for completion (in a real implementation, you might want to use proper async handling)
      var waitCount = 0
      while (!isComplete && waitCount < 300) { // 30 second timeout
        kotlinx.coroutines.delay(100)
        waitCount++
      }
      
      if (!isComplete) {
        return@withContext "Error: Analysis timed out"
      }
      
      return@withContext result
    } catch (e: Exception) {
      return@withContext "Error: ${e.message}"
    }
  }

  private fun buildVideoAnalysisPrompt(): String {
    return """
      Analyze this sequence of video frames and provide a detailed description.
      
      Please include:
      1. Objects and people detected in the frames
      2. Their positions and movements between frames
      3. Scene description and context
      4. Any notable activities or events
      5. Confidence levels for key detections
      
      Be thorough but concise. This analysis will be used for video understanding and querying.
    """.trimIndent()
  }

  private fun buildRagDescription(
    batchIndex: Int,
    timestamp: String,
    analysisResult: String,
    frameCount: Int
  ): String {
    return """
      Video Batch Analysis #$batchIndex
      Timestamp: $timestamp
      Frames Analyzed: $frameCount
      
      Analysis Results:
      $analysisResult
      
      ---
      This represents video content captured and analyzed as part of a continuous video sequence.
    """.trimIndent()
  }
}

/**
 * Extension function for easier integration with existing video analysis workflows.
 */
suspend fun VideoAnalysisMemoryManager.processVideoBatchWithRag(
  frames: List<Bitmap>,
  vlmModel: Model,
  ragModel: Model,
  batchIndex: Int,
  onAnalysisComplete: (String, String) -> Unit // (result, error)
) {
  val (result, error) = VideoAnalysisRagIntegration.analyzeAndMemorizeBatch(
    vlmModel = vlmModel,
    ragModel = ragModel,
    frames = frames,
    batchIndex = batchIndex
  )
  
  onAnalysisComplete(result, error)
}
