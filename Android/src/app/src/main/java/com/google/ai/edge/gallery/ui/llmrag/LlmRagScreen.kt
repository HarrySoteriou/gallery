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

import android.util.Log
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Upload
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Clear
import androidx.compose.material.icons.filled.Description
import androidx.compose.material.icons.filled.Storage
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewModelScope
import com.google.ai.edge.gallery.data.BuiltInTaskId
import com.google.ai.edge.gallery.ui.common.chat.ChatView
import com.google.ai.edge.gallery.ui.common.chat.ChatMessageText
import com.google.ai.edge.gallery.ui.common.chat.ChatInputType
import com.google.ai.edge.gallery.ui.modelmanager.ModelManagerViewModel
import kotlinx.coroutines.launch
import kotlinx.coroutines.Dispatchers  
import kotlinx.coroutines.withContext
import java.io.BufferedReader
import java.io.InputStreamReader

@Composable
fun LlmRagScreen(
  modelManagerViewModel: ModelManagerViewModel,
  navigateUp: () -> Unit,
  modifier: Modifier = Modifier,
  viewModel: LlmRagViewModel = hiltViewModel(),
) {
  val context = LocalContext.current
  val coroutineScope = rememberCoroutineScope()
  
  // Document browsing state
  val storedDocuments by viewModel.storedDocuments.collectAsStateWithLifecycle()
  val isLoadingDocuments by viewModel.isLoadingDocuments.collectAsStateWithLifecycle()
  var showDocumentBrowser by remember { mutableStateOf(false) }
  var showDocumentPreview by remember { mutableStateOf(false) }
  var selectedDocumentId by remember { mutableStateOf<String?>(null) }
  var previewDocument by remember { mutableStateOf<LlmRagModelHelper.StoredDocument?>(null) }
  
  // File processing state
  var isProcessingDocument by remember { mutableStateOf(false) }

  // Trigger model allowlist loading - this ensures models are loaded before accessing them
  LaunchedEffect(Unit) {
    modelManagerViewModel.loadModelAllowlistWhenNeeded()
    // Also refresh documents when screen loads
    viewModel.refreshStoredDocuments()
  }
  
  val uiState by modelManagerViewModel.uiState.collectAsStateWithLifecycle()
  val selectedModel = uiState.selectedModel
  var previousModel by remember { mutableStateOf<com.google.ai.edge.gallery.data.Model?>(null) }
  
  // Handle model selection changes - cleanup previous model and initialize new one
  LaunchedEffect(selectedModel) {
    if (previousModel != null && selectedModel != previousModel) {
      Log.d("LlmRagScreen", "Cleaning up previous RAG model: ${previousModel?.name}")
      val task = modelManagerViewModel.getTaskById(com.google.ai.edge.gallery.data.BuiltInTaskId.LLM_RAG)!!
      previousModel?.let { model ->
        LlmRagModelHelper.cleanUp(model) {
          Log.d("LlmRagScreen", "Previous RAG model cleaned up")
        }
      }
    }
    previousModel = selectedModel
  }
  
  // Initialize RAG model when model/download state changes - matches ChatView pattern
  val curDownloadStatus = uiState.modelDownloadStatus[selectedModel.name]
  LaunchedEffect(curDownloadStatus, selectedModel.name) {
    if (selectedModel != null && curDownloadStatus?.status == com.google.ai.edge.gallery.data.ModelDownloadStatusType.SUCCEEDED) {
      Log.d("LlmRagScreen", "Initializing RAG model '${selectedModel.name}' with Gecko and SentencePiece")
      LlmRagModelHelper.initialize(
        context = context,
        model = selectedModel,
        onDone = { error ->
          if (error.isNotEmpty()) {
            Log.e("LlmRagScreen", "Failed to initialize RAG model: $error")
          } else {
            Log.d("LlmRagScreen", "RAG model successfully initialized with Gecko embeddings")
          }
        }
      )
    }
  }
  
  // Sample context text for demonstration
  val sampleContext = """
  The Android AI Edge Gallery is a comprehensive showcase application that demonstrates the capabilities of on-device AI models using MediaPipe and TensorFlow Lite. 
  
  The gallery includes various AI tasks such as:
  - Large Language Model (LLM) inference for text generation and chat
  - Image classification and object detection
  - Audio processing and speech recognition
  - Video analysis for real-time scene understanding
  
  Key features:
  - On-device processing ensures privacy and low latency
  - Support for various model formats including TensorFlow Lite and MediaPipe
  - Configurable model parameters like temperature, top-k, and max tokens
  - GPU and CPU acceleration options
  - Model downloading and management capabilities
  
  The Retrieval Augmented Generation (RAG) feature allows models to access and utilize user-provided context documents, enabling more accurate and contextually relevant responses for specific domains or datasets.
  """.trimIndent()

  var showSampleDialog by remember { mutableStateOf(false) }

  // File picker launcher
  val documentPickerLauncher = rememberLauncherForActivityResult(
    contract = ActivityResultContracts.GetContent()
  ) { uri ->
    uri?.let {
      isProcessingDocument = true
      coroutineScope.launch {
        try {
          val content = context.contentResolver.openInputStream(uri)?.use { inputStream ->
            BufferedReader(InputStreamReader(inputStream, "UTF-8")).use { reader ->
              reader.readText()
            }
          } ?: ""
          
          if (content.isNotBlank()) {
            selectedModel?.let { model ->
              val fileName = uri.lastPathSegment ?: "Uploaded Document"
              viewModel.memorizeText(model, content, fileName, "upload")
            }
          }
        } catch (e: Exception) {
          Log.e("LlmRagScreen", "Failed to read document: ${e.message}")
        } finally {
          isProcessingDocument = false
        }
      }
    }
  }
  
  // Asset file loader
  val loadAssetDocument = { fileName: String, title: String ->
    isProcessingDocument = true
    coroutineScope.launch {
      try {
        val content = withContext(Dispatchers.IO) {
          context.assets.open("mock_documents/$fileName").use { inputStream ->
            BufferedReader(InputStreamReader(inputStream, "UTF-8")).use { reader ->
              reader.readText()
            }
          }
        }
        
        selectedModel?.let { model ->
          viewModel.memorizeText(model, content, title, "sample")
        }
      } catch (e: Exception) {
        Log.e("LlmRagScreen", "Failed to load asset document: ${e.message}")
      } finally {
        isProcessingDocument = false
      }
    }
    Unit
  }
  
  Box(modifier = modifier.fillMaxSize()) {
    // Get the task and use standard ChatView like other tasks
    val task = modelManagerViewModel.getTaskById(BuiltInTaskId.LLM_RAG)!!
    
    // Use standard ChatView with custom input
    selectedModel?.let { model ->
      ChatView(
        task = task,
        viewModel = viewModel,
        modelManagerViewModel = modelManagerViewModel,
        navigateUp = navigateUp,
        onSendMessage = { selectedModel, messages ->
          // Convert messages to content and send through viewModel
          for (message in messages) {
            viewModel.addMessage(selectedModel, message)
          }
          val textMessages = messages.filterIsInstance<ChatMessageText>()
          if (textMessages.isNotEmpty()) {
            val content = textMessages.map { it.content }
            viewModel.sendMessage(selectedModel, content)
          }
        },
        onRunAgainClicked = { _, _ -> },
        onBenchmarkClicked = { _, _, _, _ -> },
        onResetSessionClicked = { model ->
          viewModel.clearAllRagMessages(model)
        },
        chatInputType = ChatInputType.RAG,
        modifier = Modifier.fillMaxSize(),
        // RAG-specific parameters
        onUploadDocumentClicked = {
          // Trigger document upload
          documentPickerLauncher.launch("text/*")
        },
        onSelectDocumentClicked = { 
          viewModel.refreshStoredDocuments()
          showDocumentBrowser = true 
        },
        onClearContextClicked = { model ->
          viewModel.clearAllRagMessages(model)
        },
        documentPickerLauncher = documentPickerLauncher,
        loadAssetDocument = loadAssetDocument,
        isProcessingDocument = isProcessingDocument,
      )
    }
    
  }

  // Sample context dialog
  if (showSampleDialog) {
    AlertDialog(
      onDismissRequest = { showSampleDialog = false },
      title = { Text("Add Sample Context") },
      text = {
        LazyColumn {
          item {
            Text(
              text = "This will add sample context about the Android AI Edge Gallery to the RAG knowledge base:",
              style = MaterialTheme.typography.bodyMedium
            )
            Spacer(modifier = Modifier.height(8.dp))
            Card(
              modifier = Modifier.fillMaxWidth(),
              colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surfaceVariant
              )
            ) {
              Text(
                text = sampleContext,
                modifier = Modifier.padding(12.dp),
                style = MaterialTheme.typography.bodySmall
              )
            }
          }
        }
      },
      confirmButton = {
        TextButton(
          onClick = {
            selectedModel?.let { model ->
              coroutineScope.launch {
                viewModel.memorizeText(model, sampleContext, "Android AI Edge Gallery Guide", "sample")
              }
            }
            showSampleDialog = false
          }
        ) {
          Text("Add to Knowledge Base")
        }
      },
      dismissButton = {
        TextButton(onClick = { showSampleDialog = false }) {
          Text("Cancel")
        }
      }
    )
  }
  
  // Document browser dialog
  if (showDocumentBrowser) {
    DocumentBrowserDialog(
      documents = storedDocuments,
      isLoading = isLoadingDocuments,
      onDismiss = { showDocumentBrowser = false },
      onRefresh = { viewModel.refreshStoredDocuments() },
      onViewDocument = { documentId ->
        selectedDocumentId = documentId
        showDocumentBrowser = false
        showDocumentPreview = true
        coroutineScope.launch {
          previewDocument = viewModel.getDocumentById(documentId)
        }
      },
      onDeleteDocument = { documentId ->
        viewModel.deleteDocument(documentId)
      }
    )
  }
  
  // Document preview dialog
  if (showDocumentPreview) {
    DocumentPreviewDialog(
      document = previewDocument,
      isLoading = previewDocument == null && selectedDocumentId != null,
      onDismiss = { 
        showDocumentPreview = false
        previewDocument = null
        selectedDocumentId = null
      }
    )
  }
}

/**
 * Extension function to help video analysis memorize image descriptions
 */
fun LlmRagViewModel.memorizeVideoDescriptions(
  selectedModel: com.google.ai.edge.gallery.data.Model?,
  descriptions: List<String>
) {
  selectedModel?.let { model ->
    memorizeImageDescriptions(model, descriptions)
  }
}

/**
 * Extension function for video RAG analysis that sends a message and automatically 
 * memorizes the response for future queries
 */
fun LlmRagViewModel.sendVideoAnalysisMessage(
  model: com.google.ai.edge.gallery.data.Model,
  content: List<String>,
  batchNumber: Int
) {
  val textContent = content.joinToString(" ")
  if (textContent.isBlank()) return

  this.viewModelScope.launch {
    val userMessage = ChatMessageText(content = textContent, side = com.google.ai.edge.gallery.ui.common.chat.ChatSide.USER)
    addMessage(model, userMessage)

    try {
      val assistantMessage = com.google.ai.edge.gallery.ui.common.chat.ChatMessageLoading()
      addMessage(model, assistantMessage)

      val progressListener = object : com.google.ai.edge.localagents.rag.models.AsyncProgressListener<com.google.ai.edge.localagents.rag.models.LanguageModelResponse> {
        override fun run(partialResult: com.google.ai.edge.localagents.rag.models.LanguageModelResponse, done: Boolean) {
          updateLastAssistantMessage(model, partialResult.text)
          
          // When analysis is complete, automatically memorize the result
          if (done && partialResult.text.isNotBlank()) {
            val batchDescription = """
              Video Batch #$batchNumber Analysis:
              Timestamp: ${java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss", java.util.Locale.getDefault()).format(java.util.Date())}
              
              ${partialResult.text}
              
              [This is a video analysis result stored for cross-batch querying]
            """.trimIndent()
            
            // Store in RAG memory asynchronously
            this@sendVideoAnalysisMessage.viewModelScope.launch(Dispatchers.Default) {
              try {
                val error = LlmRagModelHelper.memorizeChunks(model, listOf(batchDescription), "Video Analysis Batch #$batchNumber", "video_analysis")
                if (error.isEmpty()) {
                  android.util.Log.d("VideoRAGAnalysis", "Successfully stored batch #$batchNumber in RAG memory")
                } else {
                  android.util.Log.e("VideoRAGAnalysis", "Failed to store batch in RAG: $error")
                }
              } catch (e: Exception) {
                android.util.Log.e("VideoRAGAnalysis", "Error storing batch in RAG: ${e.message}")
              }
            }
          }
        }
      }

      val response = withContext(kotlinx.coroutines.Dispatchers.Default) {
        LlmRagModelHelper.generateResponse(model, textContent, progressListener)
      }

      // Final update with complete response (already handled in progressListener)
      // The memorization also happens in the progress listener when done=true

    } catch (e: Exception) {
      android.util.Log.e("LlmRagViewModel", "Failed to send video analysis message: ${e.message}")
      updateLastAssistantMessage(model, "Error: ${e.message}")
    }
  }
}

private fun LlmRagViewModel.updateLastAssistantMessage(model: com.google.ai.edge.gallery.data.Model, text: String) {
  val lastMessage = getLastMessage(model)
  if (lastMessage != null && lastMessage.side == com.google.ai.edge.gallery.ui.common.chat.ChatSide.AGENT) {
    removeLastMessage(model)
    val updatedMessage = ChatMessageText(content = text, side = com.google.ai.edge.gallery.ui.common.chat.ChatSide.AGENT)
    addMessage(model, updatedMessage)
  }
}
