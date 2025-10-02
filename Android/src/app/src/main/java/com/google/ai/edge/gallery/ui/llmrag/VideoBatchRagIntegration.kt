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
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.IOException

private const val TAG = "VideoBatchRagIntegration"

/**
 * Utility class to integrate Video Summary + RAG task descriptions with the RAG Knowledge Base.
 * This class reuses existing RAG components to embed video descriptions stored in assets/video_batch.
 */
object VideoBatchRagIntegration {

    /**
     * Load and embed all video batch descriptions from assets/video_batch directory
     * into the RAG Knowledge Base using existing components.
     */
    suspend fun loadVideoBatchDescriptions(
        context: Context,
        model: Model,
        viewModel: LlmRagViewModel
    ): String = withContext(Dispatchers.IO) {
        try {
            Log.d(TAG, "Starting to load video batch descriptions from assets")
            
            val assetManager = context.assets
            val videoBatchFiles = try {
                assetManager.list("video_batch")?.filter { it.endsWith(".txt") } ?: emptyList()
            } catch (e: IOException) {
                Log.w(TAG, "Failed to list video_batch assets: ${e.message}")
                return@withContext "Failed to access video_batch directory: ${e.message}"
            }
            
            if (videoBatchFiles.isEmpty()) {
                Log.i(TAG, "No video batch files found in assets/video_batch")
                return@withContext "No video batch files found in assets/video_batch directory"
            }
            
            Log.d(TAG, "Found ${videoBatchFiles.size} video batch files: ${videoBatchFiles.joinToString(", ")}")
            
            var successCount = 0
            var errorCount = 0
            val errors = mutableListOf<String>()
            
            // Process each video batch file
            for (fileName in videoBatchFiles) {
                try {
                    Log.d(TAG, "Processing video batch file: $fileName")
                    
                    // Read file content from assets
                    val fileContent = assetManager.open("video_batch/$fileName").use { inputStream ->
                        inputStream.bufferedReader().use { it.readText() }
                    }
                    
                    // Parse and extract meaningful content from the video description
                    val parsedContent = parseVideoBatchContent(fileContent, fileName)
                    
                    if (parsedContent.isNotEmpty()) {
                        // Use the existing LlmRagViewModel memorizeText function to embed the content
                        withContext(Dispatchers.Main) {
                            viewModel.memorizeText(
                                model = model,
                                text = parsedContent,
                                title = "Video Analysis - $fileName",
                                source = "video_batch"
                            )
                        }
                        successCount++
                        Log.d(TAG, "Successfully processed and embedded: $fileName")
                    } else {
                        Log.w(TAG, "No meaningful content extracted from: $fileName")
                        errorCount++
                        errors.add("No content extracted from $fileName")
                    }
                    
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to process video batch file $fileName: ${e.message}")
                    errorCount++
                    errors.add("Failed to process $fileName: ${e.message}")
                }
            }
            
            // Return summary of the operation
            val summary = buildString {
                append("Video batch loading completed.\n")
                append("Successfully embedded: $successCount files\n")
                if (errorCount > 0) {
                    append("Failed: $errorCount files\n")
                    append("Errors:\n")
                    errors.forEach { error ->
                        append("- $error\n")
                    }
                }
            }
            
            Log.i(TAG, "Video batch loading completed: $successCount successful, $errorCount failed")
            summary
            
        } catch (e: Exception) {
            val error = "Failed to load video batch descriptions: ${e.message}"
            Log.e(TAG, error)
            error
        }
    }
    
    /**
     * Parse video batch content and extract meaningful text for embedding.
     * Handles both JSON format and plain text format.
     */
    private fun parseVideoBatchContent(content: String, fileName: String): String {
        try {
            // Extract the main content after the header
            val lines = content.lines()
            val contentStartIndex = lines.indexOfFirst { it.trim().startsWith("{") }
            
            if (contentStartIndex == -1) {
                // No JSON found, treat as plain text
                Log.d(TAG, "No JSON found in $fileName, treating as plain text")
                return content.trim()
            }
            
            // Extract JSON part
            val jsonContent = lines.drop(contentStartIndex).joinToString("\n").trim()
            
            // Parse JSON and extract meaningful content
            val jsonObject = JSONObject(jsonContent)
            val extractedContent = StringBuilder()
            
            // Add scene description
            if (jsonObject.has("scene_description")) {
                val sceneDescription = jsonObject.getString("scene_description")
                extractedContent.append("Scene Description: $sceneDescription\n\n")
            }
            
            // Add detected objects information
            if (jsonObject.has("detected_objects")) {
                val detectedObjects = jsonObject.getJSONArray("detected_objects")
                extractedContent.append("Detected Objects:\n")
                
                for (i in 0 until detectedObjects.length()) {
                    val obj = detectedObjects.getJSONObject(i)
                    val name = obj.optString("name", "Unknown")
                    val description = obj.optString("description", "No description")
                    extractedContent.append("- $name: $description\n")
                }
            }
            
            // Add any additional metadata
            val metadata = StringBuilder()
            metadata.append("File: $fileName\n")
            
            // Extract date if available from header
            lines.take(contentStartIndex).forEach { line ->
                if (line.startsWith("Date:")) {
                    metadata.append("$line\n")
                }
            }
            
            // Combine all content
            val finalContent = buildString {
                append(metadata.toString())
                append("\n")
                append(extractedContent.toString())
            }
            
            Log.d(TAG, "Successfully parsed JSON content from $fileName")
            return finalContent.trim()
            
        } catch (e: Exception) {
            Log.w(TAG, "Failed to parse JSON in $fileName, treating as plain text: ${e.message}")
            // Fallback to plain text
            return content.trim()
        }
    }
    
    /**
     * Get a summary of video batch files available in assets
     */
    suspend fun getVideoBatchFilesSummary(context: Context): String = withContext(Dispatchers.IO) {
        try {
            val assetManager = context.assets
            val videoBatchFiles = assetManager.list("video_batch")?.filter { it.endsWith(".txt") } ?: emptyList()
            
            if (videoBatchFiles.isEmpty()) {
                return@withContext "No video batch files found in assets/video_batch directory"
            }
            
            val summary = StringBuilder()
            summary.append("Available video batch files (${videoBatchFiles.size}):\n\n")
            
            videoBatchFiles.forEach { fileName ->
                try {
                    val fileContent = assetManager.open("video_batch/$fileName").use { inputStream ->
                        inputStream.bufferedReader().use { it.readText() }
                    }
                    
                    // Extract basic info
                    val lines = fileContent.lines()
                    val dateInfo = lines.find { it.startsWith("Date:") } ?: ""
                    
                    summary.append("📁 $fileName\n")
                    if (dateInfo.isNotEmpty()) {
                        summary.append("   $dateInfo\n")
                    }
                    
                    // Try to extract scene description for preview
                    try {
                        val jsonStartIndex = lines.indexOfFirst { it.trim().startsWith("{") }
                        if (jsonStartIndex != -1) {
                            val jsonContent = lines.drop(jsonStartIndex).joinToString("\n").trim()
                            val jsonObject = JSONObject(jsonContent)
                            
                            if (jsonObject.has("scene_description")) {
                                val sceneDesc = jsonObject.getString("scene_description")
                                val preview = if (sceneDesc.length > 80) {
                                    sceneDesc.take(80) + "..."
                                } else {
                                    sceneDesc
                                }
                                summary.append("   Preview: $preview\n")
                            }
                        }
                    } catch (e: Exception) {
                        // Ignore JSON parsing errors for summary
                    }
                    
                    summary.append("\n")
                    
                } catch (e: Exception) {
                    summary.append("📁 $fileName (Error reading file)\n\n")
                }
            }
            
            summary.toString()
            
        } catch (e: Exception) {
            "Failed to get video batch files summary: ${e.message}"
        }
    }
}
