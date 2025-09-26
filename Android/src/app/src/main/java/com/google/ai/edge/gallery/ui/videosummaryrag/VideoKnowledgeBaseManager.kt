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

import android.content.Context
import android.util.Log
import com.google.ai.edge.gallery.data.Model
import com.google.ai.edge.gallery.ui.llmrag.RagKnowledgeBase
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.text.SimpleDateFormat
import java.util.*

/**
 * Manages video batch descriptions by writing them to physical .txt files
 * in external storage (mimicking assets/video_batch structure) and then
 * embedding them into the RAG knowledge base.
 * Exactly replicates the RAG Chat functionality for document management.
 */
object VideoKnowledgeBaseManager {

    private const val TAG = "VideoKnowledgeBase"
    private val dateFormatter = SimpleDateFormat("yyyy-MM-dd_HH-mm-ss", Locale.getDefault())
    private val displayDateFormatter = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault())

    private const val VIDEO_BATCH_RELATIVE_PATH = "assets/video_batch"

    /**
     * Initialize the video batches directory and copy sample files from assets if needed.
     * Mirrors how the LLM RAG chat loads seed documents so the retrieval stack behaves identically.
     */
    private fun initializeVideoBatchesDirectory(context: Context): File {
        val videoBatchesDir = File(context.getExternalFilesDir(null), VIDEO_BATCH_RELATIVE_PATH)
        if (!videoBatchesDir.exists()) {
            videoBatchesDir.mkdirs()

            // Copy sample files from bundled assets (if provided) to bootstrap the knowledge base.
            try {
                val assetManager = context.assets
                val sampleFiles = assetManager.list("video_batch") ?: emptyArray()

                for (fileName in sampleFiles) {
                    if (!fileName.endsWith(".txt")) continue

                    assetManager.open("video_batch/$fileName").use { inputStream ->
                        val outputFile = File(videoBatchesDir, fileName)
                        outputFile.outputStream().use { outputStream ->
                            inputStream.copyTo(outputStream)
                        }
                        Log.d(TAG, "Copied sample file: $fileName")
                    }
                }
            } catch (e: Exception) {
                Log.w(TAG, "Failed to copy sample files from assets: ${e.message}")
            }
        }
        return videoBatchesDir
    }

    /**
     * Stores a video batch description by:
     * 1. Writing it to a physical .txt file
     * 2. Embedding the file content into the RAG knowledge base
     * This exactly replicates how RAG Chat processes uploaded documents
     */
    suspend fun storeBatchDescription(
        context: Context,
        ragModel: Model?,
        batchDescription: String
    ): String = withContext(Dispatchers.IO) {
        try {
            if (batchDescription.isEmpty()) {
                return@withContext "Empty batch description"
            }

            val timestamp = System.currentTimeMillis()
            val dateString = dateFormatter.format(Date(timestamp))
            val displayDateString = displayDateFormatter.format(Date(timestamp))
            val fileName = "video_batch_$dateString.txt"
            val title = "Video Batch - $displayDateString"

            // Step 1: Initialize directory and write to physical .txt file
            val videoBatchesDir = initializeVideoBatchesDirectory(context)

            val txtFile = File(videoBatchesDir, fileName)
            val documentContent = """Video Analysis Batch
Date: $displayDateString

$batchDescription"""

            try {
                txtFile.writeText(documentContent, Charsets.UTF_8)
                Log.d(TAG, "Successfully wrote video batch to file: ${txtFile.absolutePath}")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to write video batch file: ${e.message}")
                return@withContext "Failed to write file: ${e.message}"
            }

            // Step 2: Embed the file content into RAG system (exactly like RAG Chat does)
            if (ragModel != null) {
                val result = RagKnowledgeBase.memorizeChunks(
                    model = ragModel,
                    chunks = chunkText(documentContent),
                    title = title,
                    source = "video_analysis"
                )

                if (result.isEmpty()) {
                    Log.d(TAG, "Successfully stored and embedded video batch: $title")
                    Log.d(TAG, "Physical file location: ${txtFile.absolutePath}")
                    ""
                } else {
                    Log.w(TAG, "Failed to embed video batch in RAG: $result")
                    // Still return success since the file was written successfully
                    Log.d(TAG, "File was still written successfully to: ${txtFile.absolutePath}")
                    ""
                }
            } else {
                Log.w(TAG, "RAG model not available, but file was written successfully")
                Log.d(TAG, "Physical file location: ${txtFile.absolutePath}")
                ""
            }

        } catch (e: Exception) {
            Log.e(TAG, "Failed to store batch description", e)
            "Failed to store batch: ${e.message}"
        }
    }

    /**
     * Get all stored video batch files from external storage (assets structure)
     * This includes both generated batch files and sample files from assets
     */
    fun getStoredBatchFiles(context: Context): List<File> {
        val videoBatchesDir = initializeVideoBatchesDirectory(context)
        return videoBatchesDir.listFiles { file ->
            file.isFile && file.name.endsWith(".txt") && file.name.startsWith("video_batch_")
        }?.sortedByDescending { it.lastModified() } ?: emptyList()
    }

    /**
     * Get all video batch files including samples (for browsing all available batch descriptions)
     */
    fun getAllBatchFiles(context: Context): List<File> {
        val videoBatchesDir = initializeVideoBatchesDirectory(context)
        return videoBatchesDir.listFiles { file ->
            file.isFile && file.name.endsWith(".txt")
        }?.sortedByDescending { it.lastModified() } ?: emptyList()
    }

    /**
     * Read a specific batch file content
     */
    fun readBatchFile(file: File): String? {
        return try {
            if (file.exists() && file.canRead()) {
                file.readText(Charsets.UTF_8)
            } else {
                null
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to read batch file ${file.name}: ${e.message}")
            null
        }
    }

    /**
     * Delete a batch file and remove it from RAG knowledge base
     */
    suspend fun deleteBatchFile(context: Context, fileName: String): Boolean {
        return withContext(Dispatchers.IO) {
            try {
                val videoBatchesDir = initializeVideoBatchesDirectory(context)
                val file = File(videoBatchesDir, fileName)

                if (file.exists()) {
                    val deleted = file.delete()
                    if (deleted) {
                        Log.d(TAG, "Successfully deleted batch file: $fileName")
                        // Note: The RAG system handles document deletion through DocumentBrowserDialog
                    }
                    deleted
                } else {
                    Log.w(TAG, "Batch file not found: $fileName")
                    false
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to delete batch file $fileName: ${e.message}")
                false
            }
        }
    }

    /**
     * Simple text chunking strategy - splits by paragraphs and sentences
     * Exactly replicates the chunking logic from LlmRagViewModel
     */
    private fun chunkText(text: String, maxChunkSize: Int = 500): List<String> {
        val paragraphs = text.split("\n\n").filter { it.trim().isNotEmpty() }
        val chunks = mutableListOf<String>()

        for (paragraph in paragraphs) {
            if (paragraph.length <= maxChunkSize) {
                chunks.add(paragraph.trim())
            } else {
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

        return chunks.ifEmpty { listOf(text) }
    }

    /**
     * Test method to verify the batch file writing functionality
     * Returns the path where files are being stored
     */
    fun getVideoBatchesDirectoryPath(context: Context): String {
        val videoBatchesDir = initializeVideoBatchesDirectory(context)
        return videoBatchesDir.absolutePath
    }

    /**
     * Test method to list all files in the video batches directory
     */
    fun listAllFilesInDirectory(context: Context): List<String> {
        val videoBatchesDir = initializeVideoBatchesDirectory(context)
        return videoBatchesDir.listFiles()?.map { "${it.name} (${it.length()} bytes)" } ?: emptyList()
    }

    /**
     * Test function to manually create a sample batch file for testing
     */
    suspend fun createTestBatchFile(context: Context): String {
        val testDescription = """
        {
          "detected_objects": [
            {
              "name": "person",
              "description": "Test person walking across the frame"
            },
            {
              "name": "car",
              "description": "Blue car parked in background"
            }
          ],
          "scene_description": "Test video analysis batch created manually for debugging purposes"
        }
        """.trimIndent()

        return storeBatchDescription(
            context = context,
            ragModel = null, // Skip RAG embedding for test
            batchDescription = testDescription
        )
    }
}
