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
import android.graphics.Bitmap
import android.util.Log
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.Alignment
import androidx.core.os.bundleOf
import androidx.hilt.lifecycle.viewmodel.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.google.ai.edge.gallery.data.BuiltInTaskId
import com.google.ai.edge.gallery.data.Model
import com.google.ai.edge.gallery.firebaseAnalytics
import com.google.ai.edge.gallery.ui.common.chat.ChatView
import com.google.ai.edge.gallery.ui.common.chat.ChatMessageText
import com.google.ai.edge.gallery.ui.common.chat.ChatMessageImage
import com.google.ai.edge.gallery.ui.common.chat.ChatSide
import com.google.ai.edge.gallery.ui.llmchat.LlmAskImageViewModel
import com.google.ai.edge.gallery.ui.llmchat.LlmChatViewModelBase
import com.google.ai.edge.gallery.ui.llmrag.RagContextManager
import com.google.ai.edge.gallery.ui.llmrag.RagKnowledgeBase
import com.google.ai.edge.gallery.ui.modelmanager.ModelManagerViewModel
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import com.google.ai.edge.gallery.ui.llmrag.DocumentBrowserDialog
import com.google.ai.edge.gallery.ui.llmrag.DocumentPreviewDialog
import kotlinx.coroutines.withTimeoutOrNull
import kotlinx.coroutines.withContext
import kotlinx.coroutines.delay

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
  val coroutineScope = rememberCoroutineScope()
  val task = modelManagerViewModel.getTaskById(id = taskId)
  val ragTask = modelManagerViewModel.getTaskById(id = BuiltInTaskId.LLM_RAG)

  if (task == null || ragTask == null) {
    LaunchedEffect(Unit) { modelManagerViewModel.loadModelAllowlistWhenNeeded() }
    Box(modifier = modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
      CircularProgressIndicator()
    }
    return
  }

  val uiState by viewModel.uiState.collectAsStateWithLifecycle()

  // Document browser state (reusing existing RAG document browser)
  var storedDocuments by remember {
    mutableStateOf<List<RagKnowledgeBase.DocumentMetadata>>(emptyList())
  }
  var isLoadingDocuments by remember { mutableStateOf(false) }
  var showDocumentBrowser by remember { mutableStateOf(false) }
  var showDocumentPreview by remember { mutableStateOf(false) }
  var selectedDocumentId by remember { mutableStateOf<String?>(null) }
  var previewDocument by remember {
    mutableStateOf<RagKnowledgeBase.StoredDocument?>(null)
  }
  var isLoadingPreviewDocument by remember { mutableStateOf(false) }

  // Initialize only document list, models will be loaded on-demand
  LaunchedEffect(Unit) {
    modelManagerViewModel.loadModelAllowlistWhenNeeded()
    storedDocuments = RagKnowledgeBase.getDocumentMetadataList()
  }

  // Monitor for completed batch responses and process them for RAG storage
  // Track last processed message to avoid re-processing
  var lastProcessedMessageContent by remember { mutableStateOf("") }

  LaunchedEffect(uiState.messagesByModel, uiState.inProgress) {
    Log.d("VideoSummaryRagScreen", "LaunchedEffect triggered - inProgress: ${uiState.inProgress}")
    if (!uiState.inProgress) {
      val selectedModel = modelManagerViewModel.uiState.value.selectedModel
      if (selectedModel.name.isEmpty()) {
        Log.d("VideoSummaryRagScreen", "Selected model is unset - skipping batch processing")
        return@LaunchedEffect
      }
      val messages = uiState.messagesByModel[selectedModel.name] ?: emptyList()
      Log.d("VideoSummaryRagScreen", "Messages count for ${selectedModel.name}: ${messages.size}")

      if (messages.isNotEmpty()) {
        val lastMessage = messages.lastOrNull()
        Log.d("VideoSummaryRagScreen", "Last message type: ${lastMessage?.javaClass?.simpleName}, side: ${if (lastMessage is ChatMessageText) lastMessage.side else "N/A"}")

        if (lastMessage is ChatMessageText &&
            lastMessage.side == ChatSide.AGENT &&
            lastMessage.content.isNotEmpty() &&
            lastMessage.content != lastProcessedMessageContent) {

          Log.d("VideoSummaryRagScreen", "Processing NEW batch for RAG - content length: ${lastMessage.content.length}")
          val messageContent = lastMessage.content

          // Show file path for debugging
          val dirPath = VideoKnowledgeBaseManager.getVideoBatchesDirectoryPath(context)
          Log.d("VideoSummaryRagScreen", "Video batches will be stored at: $dirPath")

          // Complete batch workflow: Clear GPU → Load Embedding → Store → Clear GPU
          processBatchForRAG(
            scope = coroutineScope,
            context = context,
            modelManagerViewModel = modelManagerViewModel,
            ragTask = ragTask,
            batchDescription = messageContent,
            visionModel = selectedModel,
            task = task,
            viewModel = viewModel,
            onStored = {
              storedDocuments = RagKnowledgeBase.getDocumentMetadataList()
              // Log files after storage
              val files = VideoKnowledgeBaseManager.listAllFilesInDirectory(context)
              Log.d("VideoSummaryRagScreen", "Files in directory after storage: ${files.joinToString(", ")}")
              lastProcessedMessageContent = messageContent
            }
          )
        } else {
          if (lastMessage is ChatMessageText && lastMessage.content == lastProcessedMessageContent) {
            Log.d("VideoSummaryRagScreen", "Skipping - already processed this message")
          } else {
            Log.d("VideoSummaryRagScreen", "Skipping batch processing - conditions not met")
          }
        }
      } else {
        Log.d("VideoSummaryRagScreen", "No messages found for processing")
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

      // CRITICAL: Limit images to prevent GPU memory overflow
      // MediaPipe keeps all images in GPU memory until session reset
      val maxImagesPerBatch = 5
      val actualImages = if (images.size > maxImagesPerBatch) {
        Log.w("VideoSummaryRagScreen", "Image batch size ${images.size} exceeds limit $maxImagesPerBatch, truncating")
        images.take(maxImagesPerBatch)
      } else {
        images
      }

      // Auto-inject VideoAnalysis prompt when images are provided (for batch processing)
      if (actualImages.isNotEmpty() && text.isEmpty()) {
        text = buildVideoAnalysisPrompt()
        chatMessageText = ChatMessageText(content = text, side = ChatSide.USER)
        viewModel.addMessage(model = model, message = chatMessageText)
      }

      if (text.isNotEmpty() && chatMessageText != null) {
        modelManagerViewModel.addTextInputHistory(text)
        viewModel.generateResponse(
          model = model,
          input = text,
          images = actualImages,
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
    // Video RAG specific - enable browse video knowledge button
    onBrowseVideoKnowledge = { showDocumentBrowser = true },
    // Manual save functionality as backup
    onClearContextClicked = { model ->
      // First save current batch if there's an agent message
      val messages = uiState.messagesByModel[model.name] ?: emptyList()
      val lastMessage = messages.lastOrNull()
      if (lastMessage is ChatMessageText &&
          lastMessage.side == ChatSide.AGENT &&
          lastMessage.content.isNotEmpty()) {

        val messageContent = lastMessage.content
        Log.d("VideoSummaryRagScreen", "Manual save triggered - processing batch")
        processBatchForRAG(
          scope = coroutineScope,
          context = context,
          modelManagerViewModel = modelManagerViewModel,
          ragTask = ragTask,
          batchDescription = messageContent,
          visionModel = model,
          task = task,
          viewModel = viewModel,
          onStored = {
            storedDocuments = RagKnowledgeBase.getDocumentMetadataList()
            lastProcessedMessageContent = messageContent
          }
        )
      } else {
        // Just clear context if no agent message to save
        RagContextManager.clearBatch(task, model)
        viewModel.clearAllMessages(model)
      }
    },
  )

  // Document browser dialog (reusing existing RAG implementation)
  if (showDocumentBrowser) {
    DocumentBrowserDialog(
      documents = storedDocuments,
      isLoading = isLoadingDocuments,
      onDismiss = { showDocumentBrowser = false },
      onRefresh = {
        // Refresh stored documents to show video batch descriptions
        isLoadingDocuments = true
        coroutineScope.launch {
          storedDocuments = RagKnowledgeBase.getDocumentMetadataList()
          isLoadingDocuments = false
        }
      },
      onViewDocument = { documentId ->
        selectedDocumentId = documentId
        showDocumentBrowser = false
        showDocumentPreview = true
        previewDocument = null
        isLoadingPreviewDocument = true

        coroutineScope.launch {
          try {
            val document = withTimeoutOrNull(10000) {
              RagKnowledgeBase.getDocumentById(documentId)
            }

            if (document != null) {
              previewDocument = document
              isLoadingPreviewDocument = false
              Log.d("VideoSummaryRagScreen", "Successfully loaded document: ${document.metadata.title}")
            } else {
              Log.w("VideoSummaryRagScreen", "Document not found or loading timed out: $documentId")
              isLoadingPreviewDocument = false
              showDocumentPreview = false
              selectedDocumentId = null
            }
          } catch (e: Exception) {
            Log.e("VideoSummaryRagScreen", "Error loading document $documentId: ${e.message}")
            isLoadingPreviewDocument = false
            showDocumentPreview = false
            selectedDocumentId = null
          }
        }
      },
      onDeleteDocument = { documentId ->
        RagKnowledgeBase.deleteDocument(documentId)
      }
    )
  }

  // Document preview dialog
  if (showDocumentPreview) {
    DocumentPreviewDialog(
      document = previewDocument,
      isLoading = isLoadingPreviewDocument,
      onDismiss = {
        showDocumentPreview = false
        previewDocument = null
        selectedDocumentId = null
        isLoadingPreviewDocument = false
      }
    )
  }
}

/**
 * Complete batch processing workflow for VideoSummaryRAG with CPU-only embedding:
 * 1. Write description to .txt file and embed in RAG (CPU-only, no GPU conflicts)
 * 2. Clear VLM context for next batch (CRITICAL: prevents memory overflow)
 * 3. Clear chat UI for next batch
 *
 * Performance: CPU embedding (0.15-0.6s) is 10-60x faster than GPU swap overhead (5-10s).
 * The VLM stays on GPU throughout the entire process.
 */
private fun processBatchForRAG(
  scope: CoroutineScope,
  context: Context,
  modelManagerViewModel: ModelManagerViewModel,
  ragTask: com.google.ai.edge.gallery.data.Task,
  batchDescription: String,
  visionModel: Model,
  task: com.google.ai.edge.gallery.data.Task,
  viewModel: LlmChatViewModelBase,
  onStored: suspend () -> Unit = {}
) {
  scope.launch(Dispatchers.IO) {
    try {
      Log.d("VideoSummaryRagScreen", "processBatchForRAG called with description length: ${batchDescription.length}")
      if (batchDescription.isNotEmpty()) {
        // CRITICAL: Clear VLM session FIRST to release image memory before embedding
        // This prevents GPU memory overflow from accumulated frames
        Log.d("VideoSummaryRagScreen", "Clearing VLM session to release image memory")
        RagContextManager.clearBatch(task, visionModel)

        // Small delay to ensure GPU memory is released before embedding
        delay(100)

        // Store batch description with CPU-only embedding (VLM stays on GPU)
        val result = VideoKnowledgeBaseManager.storeBatchDescriptionWithGpuManagement(
          context = context,
          modelManagerViewModel = modelManagerViewModel,
          ragTask = ragTask,
          batchDescription = batchDescription
        )
        Log.d("VideoSummaryRagScreen", "storeBatchDescription result: '$result'")

        if (result.isEmpty()) {
          Log.d("VideoSummaryRagScreen", "Successfully stored batch description in RAG")

          withContext(Dispatchers.Main) {
            onStored()
          }

          // Clear chat UI for next batch
          withContext(Dispatchers.Main) {
            viewModel.clearAllMessages(visionModel)
          }

          Log.d("VideoSummaryRagScreen", "Completed batch processing - ready for next video analysis")
        } else {
          Log.e("VideoSummaryRagScreen", "Failed to store batch in RAG: $result")
        }
      } else {
        Log.w("VideoSummaryRagScreen", "Empty batch description, skipping storage")
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
  "scene_description": "Generate a concise summary of the video"
  """.trimIndent()
}
