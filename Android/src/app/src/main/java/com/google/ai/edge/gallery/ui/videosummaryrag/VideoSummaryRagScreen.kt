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

import android.graphics.Bitmap
import android.util.Log
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.core.os.bundleOf
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.google.ai.edge.gallery.data.BuiltInTaskId
import com.google.ai.edge.gallery.firebaseAnalytics
import com.google.ai.edge.gallery.ui.common.chat.ChatView
import com.google.ai.edge.gallery.ui.common.chat.ChatMessageText
import com.google.ai.edge.gallery.ui.common.chat.ChatMessageImage
import com.google.ai.edge.gallery.ui.common.chat.ChatSide
import com.google.ai.edge.gallery.ui.llmchat.LlmAskImageViewModel
import com.google.ai.edge.gallery.ui.llmchat.LlmChatViewModelBase
import com.google.ai.edge.gallery.ui.llmrag.LlmRagModelHelper
import com.google.ai.edge.gallery.ui.llmrag.RagModelInstance
import com.google.ai.edge.gallery.ui.videoanalysis.VideoAnalysisMemoryManager
import com.google.ai.edge.gallery.ui.modelmanager.ModelManagerViewModel
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

@Composable
fun VideoSummaryRagScreen(
  modelManagerViewModel: ModelManagerViewModel,
  navigateUp: () -> Unit,
  modifier: Modifier = Modifier,
  viewModel: LlmAskImageViewModel = hiltViewModel(),
) {
  VideoSummaryRagChatViewWrapper(
    viewModel = viewModel,
    modelManagerViewModel = modelManagerViewModel,
    taskId = BuiltInTaskId.VIDEO_RAG_ANALYSIS,
    navigateUp = navigateUp,
    modifier = modifier,
  )
}

@Composable
fun VideoSummaryRagChatViewWrapper(
  viewModel: LlmChatViewModelBase,
  modelManagerViewModel: ModelManagerViewModel,
  taskId: String,
  navigateUp: () -> Unit,
  modifier: Modifier = Modifier,
) {
  val context = LocalContext.current
  val task = modelManagerViewModel.getTaskById(id = taskId)!!
  val ragTask = modelManagerViewModel.getTaskById(id = BuiltInTaskId.LLM_RAG)!!
  val uiState by viewModel.uiState.collectAsStateWithLifecycle()

  // Automatically load Gecko embedding model for RAG functionality
  LaunchedEffect(Unit) {
    modelManagerViewModel.loadModelAllowlistWhenNeeded()
    val geckoModel = ragTask.models.find { it.name == "Gecko-1024-Embedding" }
    if (geckoModel != null) {
      Log.d("VideoSummaryRagScreen", "Auto-initializing Gecko embedding model for RAG")
      modelManagerViewModel.initializeModel(context, ragTask, geckoModel)
    }
  }

  // Monitor for completed batch responses and process them for RAG storage
  LaunchedEffect(uiState.messagesByModel, uiState.inProgress) {
    if (!uiState.inProgress) {
      val selectedModel = modelManagerViewModel.uiState.value.selectedModel
      val messages = uiState.messagesByModel[selectedModel.name] ?: emptyList()

      if (messages.isNotEmpty()) {
        val lastMessage = messages.lastOrNull()
        if (lastMessage is ChatMessageText &&
            lastMessage.side == ChatSide.AGENT &&
            lastMessage.content.isNotEmpty()) {

          // Complete batch workflow: Store → Embed → Clear
          processBatchForRAG(
            ragTask = ragTask,
            batchDescription = lastMessage.content,
            visionModel = selectedModel,
            task = task,
            viewModel = viewModel
          )
        }
      }
    }
  }

  ChatView(
    task = task,
    viewModel = viewModel,
    modelManagerViewModel = modelManagerViewModel,
    onSendMessage = { model, messages ->
      for (message in messages) {
        viewModel.addMessage(model = model, message = message)
      }

      var text = ""
      val images: MutableList<Bitmap> = mutableListOf()
      var chatMessageText: ChatMessageText? = null
      for (message in messages) {
        if (message is ChatMessageText) {
          chatMessageText = message
          text = message.content
        } else if (message is ChatMessageImage) {
          images.addAll(message.bitmaps)
        }
      }

      // Auto-inject VideoAnalysis prompt when images are provided (for batch processing)
      if (images.isNotEmpty() && text.isEmpty()) {
        text = buildVideoAnalysisPrompt()
        chatMessageText = ChatMessageText(content = text, side = ChatSide.USER)
        viewModel.addMessage(model = model, message = chatMessageText)
      }

      if (text.isNotEmpty() && chatMessageText != null) {
        modelManagerViewModel.addTextInputHistory(text)
        viewModel.generateResponse(
          model = model,
          input = text,
          images = images,
          onError = {
            viewModel.handleError(
              context = context,
              task = task,
              model = model,
              modelManagerViewModel = modelManagerViewModel,
              triggeredMessage = chatMessageText,
            )
          },
        )

        firebaseAnalytics?.logEvent(
          "generate_action",
          bundleOf("capability_name" to task.id, "model_id" to model.name),
        )
      }
    },
    onRunAgainClicked = { model, message ->
      if (message is ChatMessageText) {
        viewModel.runAgain(
          model = model,
          message = message,
          onError = {
            viewModel.handleError(
              context = context,
              task = task,
              model = model,
              modelManagerViewModel = modelManagerViewModel,
              triggeredMessage = message,
            )
          },
        )
      }
    },
    onBenchmarkClicked = { _, _, _, _ -> },
    onResetSessionClicked = { model -> viewModel.resetSession(task = task, model = model) },
    showStopButtonInInputWhenInProgress = true,
    onStopButtonClicked = { model -> viewModel.stopResponse(model = model) },
    navigateUp = navigateUp,
    modifier = modifier,
  )
}

/**
 * Complete batch processing workflow for VideoSummaryRAG:
 * 1. Store description in RAG knowledge base
 * 2. Embed for searchability
 * 3. Clear VLM context for next batch
 */
private fun processBatchForRAG(
  ragTask: com.google.ai.edge.gallery.data.Task,
  batchDescription: String,
  visionModel: com.google.ai.edge.gallery.data.Model,
  task: com.google.ai.edge.gallery.data.Task,
  viewModel: LlmAskImageViewModel
) {
  CoroutineScope(Dispatchers.IO).launch {
    try {
      // Find the RAG embedding model (Gecko)
      val ragModel = ragTask.models.find {
        it.name.contains("Gecko") && it.name.contains("Embedding")
      }

      if (ragModel?.instance is RagModelInstance && batchDescription.isNotEmpty()) {
        // Step 1: Store the batch description in RAG knowledge base
        val batchId = System.currentTimeMillis()
        val title = "Video Batch Analysis - $batchId"
        val documentContent = "Video analysis batch:\n$batchDescription"

        val result = LlmRagModelHelper.memorizeChunks(
          model = ragModel,
          chunks = listOf(documentContent),
          title = title,
          source = "video_batch_analysis"
        )

        if (result.isEmpty()) {
          Log.d("VideoSummaryRagScreen", "Successfully stored batch description in RAG: $title")

          // Step 2: Clear VLM context for next batch (like VideoAnalysis does)
          VideoAnalysisMemoryManager.clearContextForNewBatch(task, visionModel)

          // Step 3: Clear chat UI for next batch
          viewModel.clearAllMessages(visionModel)

          Log.d("VideoSummaryRagScreen", "Cleared context for new batch - ready for next video analysis")
        } else {
          Log.e("VideoSummaryRagScreen", "Failed to store batch in RAG: $result")
        }
      } else {
        Log.w("VideoSummaryRagScreen", "RAG embedding model not found or not initialized")
      }
    } catch (e: Exception) {
      Log.e("VideoSummaryRagScreen", "Failed to process batch for RAG", e)
    }
  }
}

/**
 * Uses the exact same prompt as VideoAnalysis task for consistent batch descriptions
 */
private fun buildVideoAnalysisPrompt(): String {
  return """
    Analyze the following sequence of video frames and identify people. Respond in JSON format:
    {
      "detected_objects": [
        {
          "name": "object_name",
          "description": "Concise description of the object",
        }
      ],
    }
  "scene_description": "Description of the overall scene"
  """.trimIndent()
}
