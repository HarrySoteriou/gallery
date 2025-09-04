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

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Upload
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.google.ai.edge.gallery.data.BuiltInTaskId
import com.google.ai.edge.gallery.ui.common.chat.ChatPanel
import com.google.ai.edge.gallery.ui.common.chat.ChatMessageText
import com.google.ai.edge.gallery.ui.modelmanager.ModelManagerViewModel
import kotlinx.coroutines.launch
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
  
  val uiState by modelManagerViewModel.uiState.collectAsStateWithLifecycle()
  val selectedModel = uiState.selectedModel
  
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

  Column(modifier = modifier.fillMaxSize()) {
    // Header with memorization controls
    Card(
      modifier = Modifier
        .fillMaxWidth()
        .padding(16.dp),
      elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
    ) {
      Column(
        modifier = Modifier.padding(16.dp)
      ) {
        Text(
          text = "RAG Knowledge Base",
          style = MaterialTheme.typography.titleMedium,
          fontWeight = FontWeight.Bold
        )
        
        Spacer(modifier = Modifier.height(8.dp))
        
        Text(
          text = "Add context documents to enhance the model's knowledge for more accurate responses.",
          style = MaterialTheme.typography.bodyMedium,
          color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        
        Spacer(modifier = Modifier.height(12.dp))
        
        Row(
          modifier = Modifier.fillMaxWidth(),
          horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
          Button(
            onClick = { showSampleDialog = true },
            modifier = Modifier.weight(1f)
          ) {
            Icon(Icons.Default.Upload, contentDescription = null)
            Spacer(modifier = Modifier.width(4.dp))
            Text("Add Sample Context")
          }
          
          OutlinedButton(
            onClick = {
              viewModel.clearAllRagMessages(selectedModel)
            },
            modifier = Modifier.weight(1f)
          ) {
            Text("Clear Memory")
          }
        }
      }
    }

    // Get the task and chat panel
    val task = modelManagerViewModel.getTaskById(BuiltInTaskId.LLM_RAG)!!
    
    ChatPanel(
      task = task,
      selectedModel = selectedModel,
      modelManagerViewModel = modelManagerViewModel,
      viewModel = viewModel,
      navigateUp = navigateUp,
      onSendMessage = { model, messages ->
        // Convert messages to content and send through viewModel
        for (message in messages) {
          viewModel.addMessage(model, message)
        }
        val textMessages = messages.filterIsInstance<ChatMessageText>()
        if (textMessages.isNotEmpty()) {
          val content = textMessages.map { it.content }
          viewModel.sendMessage(model, content)
        }
      },
      onRunAgainClicked = { _, _ -> },
      onBenchmarkClicked = { _, _, _, _ -> },
      modifier = Modifier.weight(1f)
    )
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
                viewModel.memorizeText(model, sampleContext)
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
