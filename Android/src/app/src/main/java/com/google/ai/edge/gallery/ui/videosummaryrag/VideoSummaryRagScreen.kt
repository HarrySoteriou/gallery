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

  var ragEmbeddingModel by remember { mutableStateOf<Model?>(null) }

  // Automatically load Gecko embedding model for RAG functionality
  LaunchedEffect(Unit) {
    modelManagerViewModel.loadModelAllowlistWhenNeeded()

    // Wait for the RAG task models to materialize after allowlist loading.
    var attempts = 0
    while (ragTask.models.isEmpty() && attempts < 50) {
      delay(100)
      attempts += 1
    }

    val geckoModel = ragTask.models.find { it.name.contains("Gecko", ignoreCase = true) }
    val ragLlmModel = ragTask.models.firstOrNull { !it.name.contains("Embedding", ignoreCase = true) }

    ragEmbeddingModel = ragLlmModel

    if (ragLlmModel != null) {
      Log.d("VideoSummaryRagScreen", "Initializing RAG LLM model: ${ragLlmModel.name}")
      modelManagerViewModel.initializeModel(context, ragTask, ragLlmModel)
    } else {
      Log.w("VideoSummaryRagScreen", "No RAG LLM model found in allowlist")
    }

    if (geckoModel != null) {
      Log.d("VideoSummaryRagScreen", "Ensuring Gecko embedding model assets are available")
      modelManagerViewModel.initializeModel(context, ragTask, geckoModel)
    } else {
      Log.w("VideoSummaryRagScreen", "Gecko embedding model not found for RAG initialization")
    }

    storedDocuments = RagKnowledgeBase.getDocumentMetadataList()
  }

  // Monitor for completed batch responses and process them for RAG storage
  LaunchedEffect(uiState.messagesByModel, uiState.inProgress, ragEmbeddingModel) {
    Log.d("VideoSummaryRagScreen", "LaunchedEffect triggered - inProgress: ${uiState.inProgress}")
    if (!uiState.inProgress) {
      if (ragEmbeddingModel == null) {
        Log.d("VideoSummaryRagScreen", "RAG embedding model not ready - deferring batch persistence")
        return@LaunchedEffect
      }
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
            lastMessage.content.isNotEmpty()) {

          Log.d("VideoSummaryRagScreen", "Processing batch for RAG - content length: ${lastMessage.content.length}")
          // Complete batch workflow: Store → Embed → Clear
          processBatchForRAG(
            scope = coroutineScope,
            context = context,
            ragModel = ragEmbeddingModel,
            batchDescription = lastMessage.content,
            visionModel = selectedModel,
            task = task,
            viewModel = viewModel,
            onStored = {
              storedDocuments = RagKnowledgeBase.getDocumentMetadataList()
            }
          )
        } else {
          Log.d("VideoSummaryRagScreen", "Skipping batch processing - conditions not met")
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
        
        Log.d("VideoSummaryRagScreen", "Manual save triggered - processing batch")
        if (ragEmbeddingModel != null) {
          processBatchForRAG(
            scope = coroutineScope,
            context = context,
            ragModel = ragEmbeddingModel,
            batchDescription = lastMessage.content,
            visionModel = model,
            task = task,
            viewModel = viewModel,
            onStored = {
          storedDocuments = RagKnowledgeBase.getDocumentMetadataList()
            }
          )
        } else {
          Log.d("VideoSummaryRagScreen", "RAG embedding model not ready during manual save")
        }
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
 * Complete batch processing workflow for VideoSummaryRAG:
 * 1. Write description to .txt file in internal storage
 * 2. Embed file content in RAG knowledge base
 * 3. Clear VLM context for next batch
 */
private fun processBatchForRAG(
  scope: CoroutineScope,
  context: Context,
  ragModel: Model?,
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
        // Step 1: Store batch description in RAG system - write .txt file and embed
        val result = VideoKnowledgeBaseManager.storeBatchDescription(
          context = context,
          ragModel = ragModel,
          batchDescription = batchDescription
        )
        Log.d("VideoSummaryRagScreen", "storeBatchDescription result: '$result'")

        if (result.isEmpty()) {
          Log.d("VideoSummaryRagScreen", "Successfully stored batch description in RAG")

          withContext(Dispatchers.Main) {
            onStored()
          }

          // Step 2: Clear VLM context for next batch
          RagContextManager.clearBatch(task, visionModel)

          // Step 3: Clear chat UI for next batch
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
