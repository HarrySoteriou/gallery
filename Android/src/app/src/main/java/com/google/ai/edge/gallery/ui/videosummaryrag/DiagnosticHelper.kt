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

/**
 * Diagnostic helper to debug the VideoSummaryRAG pipeline.
 * Add calls to these functions in your code to trace execution.
 */
object DiagnosticHelper {
    private const val TAG = "VideoRAGDiagnostic"

    /**
     * Call this when the LaunchedEffect triggers
     */
    fun logLaunchedEffectTrigger(
        inProgress: Boolean,
        messageCount: Int,
        lastMessageType: String?,
        lastMessageSide: String?,
        lastMessageContent: String?
    ) {
        Log.d(TAG, "=== LaunchedEffect Triggered ===")
        Log.d(TAG, "inProgress: $inProgress")
        Log.d(TAG, "messageCount: $messageCount")
        Log.d(TAG, "lastMessageType: $lastMessageType")
        Log.d(TAG, "lastMessageSide: $lastMessageSide")
        Log.d(TAG, "lastMessageContent preview: ${lastMessageContent?.take(100)}")
    }

    /**
     * Call this before processBatchForRAG
     */
    fun logBeforeProcessBatch(
        selectedModelName: String,
        batchDescriptionLength: Int
    ) {
        Log.d(TAG, "=== About to call processBatchForRAG ===")
        Log.d(TAG, "selectedModelName: $selectedModelName")
        Log.d(TAG, "batchDescriptionLength: $batchDescriptionLength")
    }

    /**
     * Call this inside processBatchForRAG at the start
     */
    fun logProcessBatchStart(batchDescription: String) {
        Log.d(TAG, "=== processBatchForRAG STARTED ===")
        Log.d(TAG, "Thread: ${Thread.currentThread().name}")
        Log.d(TAG, "batchDescription length: ${batchDescription.length}")
        Log.d(TAG, "batchDescription preview: ${batchDescription.take(200)}")
    }

    /**
     * Call this after clearing VLM memory
     */
    fun logAfterClearMemory() {
        Log.d(TAG, "VLM memory cleared successfully")
    }

    /**
     * Call this before storeBatchDescriptionWithGpuManagement
     */
    fun logBeforeStorage(context: Context) {
        val externalFilesDir = context.getExternalFilesDir(null)
        val filesDir = context.filesDir
        Log.d(TAG, "=== Before Storage ===")
        Log.d(TAG, "externalFilesDir: ${externalFilesDir?.absolutePath} (exists: ${externalFilesDir?.exists()})")
        Log.d(TAG, "filesDir: ${filesDir?.absolutePath} (exists: ${filesDir?.exists()})")
    }

    /**
     * Call this after storeBatchDescriptionWithGpuManagement
     */
    fun logAfterStorage(result: String, context: Context) {
        Log.d(TAG, "=== After Storage ===")
        Log.d(TAG, "storeBatchDescription result: '$result'")
        Log.d(TAG, "result.isEmpty(): ${result.isEmpty()}")

        // List all files
        val files = VideoKnowledgeBaseManager.listAllFilesInDirectory(context)
        Log.d(TAG, "Files in video_batch directory: ${files.size}")
        files.forEach { file ->
            Log.d(TAG, "  - $file")
        }
    }

    /**
     * Call this to check if the LaunchedEffect conditions are met
     */
    fun logConditionCheck(
        condition1: Boolean, // !uiState.inProgress
        condition2: Boolean, // selectedModel.name.isNotEmpty()
        condition3: Boolean, // messages.isNotEmpty()
        condition4: Boolean, // lastMessage is ChatMessageText
        condition5: Boolean, // lastMessage.side == ChatSide.AGENT
        condition6: Boolean, // lastMessage.content.isNotEmpty()
        condition7: Boolean  // lastMessage.content != lastProcessedMessageContent
    ) {
        Log.d(TAG, "=== Condition Check ===")
        Log.d(TAG, "!inProgress: $condition1")
        Log.d(TAG, "model.name.isNotEmpty(): $condition2")
        Log.d(TAG, "messages.isNotEmpty(): $condition3")
        Log.d(TAG, "lastMessage is ChatMessageText: $condition4")
        Log.d(TAG, "side == AGENT: $condition5")
        Log.d(TAG, "content.isNotEmpty(): $condition6")
        Log.d(TAG, "content != lastProcessed: $condition7")
        Log.d(TAG, "ALL CONDITIONS MET: ${condition1 && condition2 && condition3 && condition4 && condition5 && condition6 && condition7}")
    }

    /**
     * Check RAG task availability
     */
    fun logRagTaskCheck(ragTask: com.google.ai.edge.gallery.data.Task?) {
        Log.d(TAG, "=== RAG Task Check ===")
        if (ragTask == null) {
            Log.e(TAG, "RAG task is NULL!")
        } else {
            Log.d(TAG, "RAG task: ${ragTask.id}")
            Log.d(TAG, "RAG task models: ${ragTask.models.map { it.name }}")
            val geckoModel = ragTask.models.find { it.name.contains("Gecko", ignoreCase = true) }
            if (geckoModel == null) {
                Log.e(TAG, "Gecko model NOT FOUND in RAG task!")
            } else {
                Log.d(TAG, "Gecko model found: ${geckoModel.name}")
                Log.d(TAG, "Gecko instance: ${geckoModel.instance?.javaClass?.simpleName}")
            }
        }
    }
}
