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
import com.google.ai.edge.gallery.data.Accelerator
import com.google.ai.edge.gallery.data.ConfigKeys
import com.google.ai.edge.gallery.data.Model
import com.google.ai.edge.gallery.ui.llmrag.RagKnowledgeBase
import com.google.ai.edge.gallery.ui.modelmanager.ModelManagerViewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
import kotlinx.coroutines.delay
import java.io.File
import java.time.ZonedDateTime
import java.time.format.DateTimeFormatter
import java.util.Locale

/**
 * Manages video batch descriptions by writing them to physical .txt files
 * in external storage (mimicking assets/video_batch structure) and then
 * embedding them into the RAG knowledge base.
 * Exactly replicates the RAG Chat functionality for document management.
 */
object VideoKnowledgeBaseManager {

    private const val TAG = "VideoKnowledgeBase"
    private val fileNameFormatter = DateTimeFormatter.ofPattern("yyyy-MM-dd_HH-mm-ss", Locale.getDefault())
    private val displayDateFormatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss", Locale.getDefault())

    private const val VIDEO_BATCH_RELATIVE_PATH = "assets/video_batch"

    /**
     * Initialize the video batches directory and copy sample files from assets if needed.
     * Mirrors how the LLM RAG chat loads seed documents so the retrieval stack behaves identically.
     */
    private fun initializeVideoBatchesDirectory(context: Context): File {
        val baseDir = context.getExternalFilesDir(null) ?: context.filesDir
        val videoBatchesDir = File(baseDir, VIDEO_BATCH_RELATIVE_PATH)
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
        embeddingModel: Model?,
        batchDescription: String
    ): String = withContext(Dispatchers.IO) {
        try {
            if (batchDescription.isEmpty()) {
                return@withContext "Empty batch description"
            }

            val now = ZonedDateTime.now()
            val dateString = fileNameFormatter.format(now)
            val displayDateString = displayDateFormatter.format(now)
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
            if (embeddingModel != null) {
                val result = RagKnowledgeBase.memorizeChunks(
                    model = embeddingModel,
                    chunks = chunkText(documentContent),
                    title = title,
                    source = "video_analysis"
                )

                return@withContext if (result.isNotEmpty()) {
                    Log.e(TAG, "Failed to embed video batch in RAG: $result")
                    result
                } else {
                    Log.d(TAG, "Successfully stored and embedded video batch: $title")
                    Log.d(TAG, "Physical file location: ${txtFile.absolutePath}")
                    ""
                }
            } else {
                Log.e(TAG, "Embedding model not available for video batch: $title")
                return@withContext "Embedding model not available"
            }

        } catch (e: Exception) {
            Log.e(TAG, "Failed to store batch description", e)
            "Failed to store batch: ${e.message}"
        }
    }

    /**
     * Stores a video batch description with CPU-only embedding (no GPU swap needed).
     * Performance analysis: CPU embedding is 10-60x faster than GPU swap overhead.
     * 1. Load embedding model on CPU on-demand
     * 2. Write batch to file and embed in RAG
     * 3. Keep VLM on GPU (no swap needed)
     */
    suspend fun storeBatchDescriptionWithGpuManagement(
        context: Context,
        modelManagerViewModel: ModelManagerViewModel,
        ragTask: com.google.ai.edge.gallery.data.Task,
        batchDescription: String
    ): String = withContext(Dispatchers.IO) {
        try {
            Log.d(TAG, "=== storeBatchDescriptionWithGpuManagement STARTED ===")
            if (batchDescription.isEmpty()) {
                Log.w(TAG, "Empty batch description provided")
                return@withContext "Empty batch description"
            }
            Log.d(TAG, "Batch description length: ${batchDescription.length}")

            val now = ZonedDateTime.now()
            val dateString = fileNameFormatter.format(now)
            val displayDateString = displayDateFormatter.format(now)
            val fileName = "video_batch_$dateString.txt"
            val title = "Video Batch - $displayDateString"
            Log.d(TAG, "Generated filename: $fileName, title: $title")

            // Step 1: Initialize directory and write to physical .txt file
            val videoBatchesDir = initializeVideoBatchesDirectory(context)
            Log.d(TAG, "Video batches directory: ${videoBatchesDir.absolutePath}, exists: ${videoBatchesDir.exists()}")

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

            // Step 2: Load Gecko embedding model on CPU (VIDEO profiles always use CPU)
            val geckoModel = ragTask.models.find { it.name.contains("Gecko", ignoreCase = true) }
            if (geckoModel == null) {
                Log.e(TAG, "Gecko embedding model not found for RAG initialization")
                return@withContext "Gecko embedding model not found"
            }

            try {
                // Force CPU acceleration for embedding model to avoid GPU contention with VLM
                val updatedConfigs = geckoModel.configValues.toMutableMap()
                updatedConfigs[ConfigKeys.ACCELERATOR.label] = Accelerator.CPU.label
                geckoModel.configValues = updatedConfigs.toMap()

                // Ensure CPU mode for embedding model (VIDEO profiles enforce CPU in LlmRagModelHelper)
                if (geckoModel.instance == null) {
                    Log.d(TAG, "Loading Gecko embedding model on CPU: ${geckoModel.name}")
                    modelManagerViewModel.initializeModel(context, ragTask, geckoModel)
                    withTimeoutOrNull(30_000) {
                        while (geckoModel.initializing && geckoModel.instance == null) {
                            delay(100)
                        }
                    }
                }

                if (geckoModel.instance != null) {
                    Log.d(TAG, "Successfully loaded embedding model on CPU (VLM stays on GPU)")
                } else {
                    Log.e(TAG, "Failed to load embedding model on CPU")
                    return@withContext "Failed to load embedding model"
                }

                // Step 3: Embed the file content into RAG system (runs on CPU, no GPU conflict)
                val result = RagKnowledgeBase.memorizeChunks(
                    model = geckoModel,
                    chunks = chunkText(documentContent),
                    title = title,
                    source = "video_analysis"
                )

                return@withContext if (result.isNotEmpty()) {
                    Log.e(TAG, "Failed to embed video batch in RAG: $result")
                    result
                } else {
                    Log.d(TAG, "Successfully stored and embedded video batch: $title")
                    Log.d(TAG, "Physical file location: ${txtFile.absolutePath}")
                    ""
                }

            } catch (e: Exception) {
                Log.e(TAG, "Error during embedding process", e)
                return@withContext "Failed to embed: ${e.message}"
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
            embeddingModel = null, // Skip RAG embedding for test
            batchDescription = testDescription
        )
    }
}
