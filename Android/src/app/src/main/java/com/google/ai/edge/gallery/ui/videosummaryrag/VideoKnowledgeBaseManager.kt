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

package com.google.ai.edge.gallery.ui.videosummaryrag

import android.util.Log
import com.google.ai.edge.gallery.data.Model
import com.google.ai.edge.gallery.ui.llmrag.LlmRagModelHelper
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.text.SimpleDateFormat
import java.util.*

/**
 * Manages video batch descriptions using the existing RAG storage system.
 * Integrates seamlessly with DocumentBrowserDialog and DocumentPreviewDialog.
 */
object VideoKnowledgeBaseManager {

    private const val TAG = "VideoKnowledgeBase"
    private val dateFormatter = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault())

    /**
     * Stores a video batch description in the existing RAG system
     */
    suspend fun storeBatchDescription(
        ragModel: Model?,
        batchDescription: String
    ): String = withContext(Dispatchers.IO) {
        try {
            if (batchDescription.isEmpty()) {
                return@withContext "Empty batch description"
            }

            ragModel?.let { model ->
                val timestamp = System.currentTimeMillis()
                val dateString = dateFormatter.format(Date(timestamp))
                val title = "Video Batch - $dateString"

                val documentContent = """
                    Video Analysis Batch
                    Date: $dateString

                    $batchDescription
                """.trimIndent()

                val result = LlmRagModelHelper.memorizeChunks(
                    model = model,
                    chunks = listOf(documentContent),
                    title = title,
                    source = "video_batch_analysis"
                )

                if (result.isEmpty()) {
                    Log.d(TAG, "Successfully stored video batch: $title")
                    ""
                } else {
                    Log.w(TAG, "Failed to store video batch: $result")
                    result
                }
            } ?: "RAG model not available"

        } catch (e: Exception) {
            Log.e(TAG, "Failed to store batch description", e)
            "Failed to store batch: ${e.message}"
        }
    }

}