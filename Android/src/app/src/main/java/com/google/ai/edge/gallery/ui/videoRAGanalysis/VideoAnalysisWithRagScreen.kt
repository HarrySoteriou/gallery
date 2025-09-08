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

package com.google.ai.edge.gallery.ui.videoRAGanalysis

import android.graphics.Bitmap
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.google.ai.edge.gallery.R
import com.google.ai.edge.gallery.data.BuiltInTaskId
import com.google.ai.edge.gallery.data.Model
import com.google.ai.edge.gallery.ui.common.DownloadAndTryButton
import com.google.ai.edge.gallery.ui.common.chat.VideoFrameCaptureButton
import com.google.ai.edge.gallery.ui.common.chat.ChatView
import com.google.ai.edge.gallery.ui.common.chat.ChatMessageText
import com.google.ai.edge.gallery.ui.common.chat.ChatMessageImage
import com.google.ai.edge.gallery.ui.common.chat.ChatSide
import com.google.ai.edge.gallery.ui.llmrag.LlmRagViewModel
import com.google.ai.edge.gallery.ui.llmrag.sendVideoAnalysisMessage
import com.google.ai.edge.gallery.ui.modelmanager.ModelManagerViewModel
import com.google.ai.edge.gallery.ui.videoRAGanalysis.VideoAnalysisRagIntegration
import kotlinx.coroutines.launch

@Composable
fun VideoAnalysisWithRagScreen(
  modelManagerViewModel: ModelManagerViewModel,
  navigateUp: () -> Unit,
  modifier: Modifier = Modifier,
  ragViewModel: LlmRagViewModel = hiltViewModel(),
) {
  val context = LocalContext.current
  val coroutineScope = rememberCoroutineScope()
  val modelManagerUiState by modelManagerViewModel.uiState.collectAsStateWithLifecycle()
  val selectedModel = modelManagerUiState.selectedModel
  val task = modelManagerViewModel.getTaskById(BuiltInTaskId.VIDEO_RAG_ANALYSIS)!!
  
  var capturedFrames by remember { mutableStateOf<List<Bitmap>>(emptyList()) }
  var batchCount by remember { mutableStateOf(0) }
  var isAnalyzing by remember { mutableStateOf(false) }
  
  Column(modifier = modifier.fillMaxSize()) {
    // Quick Start Panel - similar to VideoAnalysisLlmScreen
    VideoRagQuickStart(
      task = task,
      model = selectedModel,
      modelManagerViewModel = modelManagerViewModel,
      ragViewModel = ragViewModel,
      onFramesCaptured = { frames ->
        capturedFrames = frames
      },
      onAnalyzeAndStore = { frames ->
        if (frames.isNotEmpty()) {
          coroutineScope.launch {
            isAnalyzing = true
            try {
              // Add frames to chat first
              val imageMessage = ChatMessageImage(
                bitmaps = frames,
                imageBitMaps = frames.map { it.asImageBitmap() },
                side = ChatSide.USER
              )
              ragViewModel.addMessage(selectedModel, imageMessage)
              
              // Add analysis prompt
              val analysisPrompt = buildVideoRagAnalysisPrompt(batchCount + 1)
              val textMessage = ChatMessageText(
                content = analysisPrompt,
                side = ChatSide.USER
              )
              ragViewModel.addMessage(selectedModel, textMessage)
              
              // Send message to get analysis and automatically store result
              ragViewModel.sendVideoAnalysisMessage(selectedModel, listOf(analysisPrompt), batchCount + 1)
              
              batchCount++
            } catch (e: Exception) {
              android.util.Log.e("VideoRAGAnalysis", "Error analyzing frames: ${e.message}")
            } finally {
              isAnalyzing = false
            }
          }
        }
      },
      capturedFrames = capturedFrames,
      isAnalyzing = isAnalyzing,
      batchCount = batchCount,
      onClearMemory = {
        ragViewModel.clearAllRagMessages(selectedModel)
        batchCount = 0
        capturedFrames = emptyList()
      },
      modifier = Modifier.padding(16.dp)
    )
    
    // Chat interface for RAG queries about video content
    ChatView(
      task = task,
      viewModel = ragViewModel,
      modelManagerViewModel = modelManagerViewModel,
      navigateUp = navigateUp,
      onSendMessage = { model, messages ->
        // Handle regular chat messages
        for (message in messages) {
          ragViewModel.addMessage(model, message)
        }
        val textMessages = messages.filterIsInstance<ChatMessageText>()
        if (textMessages.isNotEmpty()) {
          val content = textMessages.map { it.content }
          ragViewModel.sendMessage(model, content)
        }
      },
      onRunAgainClicked = { _, _ -> },
      onBenchmarkClicked = { _, _, _, _ -> },
      modifier = Modifier.weight(1f)
    )
  }
}

@Composable
fun VideoRagQuickStart(
  task: com.google.ai.edge.gallery.data.Task,
  model: Model,
  modelManagerViewModel: ModelManagerViewModel,
  ragViewModel: LlmRagViewModel,
  onFramesCaptured: (List<Bitmap>) -> Unit,
  onAnalyzeAndStore: (List<Bitmap>) -> Unit,
  capturedFrames: List<Bitmap>,
  isAnalyzing: Boolean,
  batchCount: Int,
  onClearMemory: () -> Unit,
  modifier: Modifier = Modifier,
) {
  val modelManagerUiState by modelManagerViewModel.uiState.collectAsStateWithLifecycle()
  val downloadStatus = modelManagerUiState.modelDownloadStatus[model.name]
  val isModelReady = downloadStatus?.status == com.google.ai.edge.gallery.data.ModelDownloadStatusType.SUCCEEDED ||
                     model.localFileRelativeDirPathOverride.isNotEmpty()
  
  Card(
    modifier = modifier.fillMaxWidth(),
    colors = CardDefaults.cardColors(
      containerColor = MaterialTheme.colorScheme.primaryContainer
    )
  ) {
    Column(
      modifier = Modifier.padding(16.dp),
      horizontalAlignment = Alignment.CenterHorizontally
    ) {
      Text(
        text = "Video RAG Analysis",
        style = MaterialTheme.typography.titleMedium,
        fontWeight = FontWeight.Bold,
        color = MaterialTheme.colorScheme.onPrimaryContainer
      )
      
      Spacer(modifier = Modifier.height(8.dp))
      
      Text(
        text = "Capture frames, analyze them, and store descriptions for querying across multiple video batches",
        style = MaterialTheme.typography.bodyMedium,
        color = MaterialTheme.colorScheme.onPrimaryContainer
      )
      
      if (batchCount > 0) {
        Spacer(modifier = Modifier.height(4.dp))
        Text(
          text = "$batchCount batches analyzed and stored in memory",
          style = MaterialTheme.typography.bodySmall,
          color = MaterialTheme.colorScheme.primary,
          fontWeight = FontWeight.Medium
        )
      }
      
      Spacer(modifier = Modifier.height(16.dp))
      
      if (isModelReady) {
        // Show the capture and analyze workflow
        Row(
          horizontalArrangement = Arrangement.spacedBy(8.dp),
          verticalAlignment = Alignment.CenterVertically
        ) {
          VideoFrameCaptureButton(
            onFramesCaptured = onFramesCaptured,
            enabled = !isAnalyzing
          )
          
          // "Analyze & Store" button
          if (capturedFrames.isNotEmpty()) {
            Button(
              onClick = { onAnalyzeAndStore(capturedFrames) },
              enabled = !isAnalyzing,
              colors = ButtonDefaults.buttonColors(
                containerColor = com.google.ai.edge.gallery.ui.common.getTaskBgGradientColors(task = task)[1]
              )
            ) {
              if (isAnalyzing) {
                CircularProgressIndicator(
                  modifier = Modifier.size(16.dp),
                  strokeWidth = 2.dp,
                  color = Color.White
                )
              } else {
                Icon(
                  Icons.Default.Psychology,
                  contentDescription = "",
                  tint = Color.White,
                  modifier = Modifier.size(18.dp)
                )
              }
              Spacer(modifier = Modifier.width(8.dp))
              Text(
                if (isAnalyzing) "Analyzing..." else "Analyze & Store",
                color = Color.White,
                style = MaterialTheme.typography.titleMedium,
              )
            }
          } else {
            // Show placeholder for analyze button
            OutlinedButton(
              onClick = { },
              enabled = false
            ) {
              Icon(
                Icons.Default.Psychology,
                contentDescription = "",
                modifier = Modifier.size(18.dp)
              )
              Spacer(modifier = Modifier.width(8.dp))
              Text("Capture frames first")
            }
          }
        }
        
        if (capturedFrames.isNotEmpty()) {
          Spacer(modifier = Modifier.height(8.dp))
          Text(
            text = "✓ ${capturedFrames.size} frames captured and ready for analysis",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onPrimaryContainer
          )
        }
        
        // Quick query buttons and clear memory when batches exist
        if (batchCount > 0) {
          Spacer(modifier = Modifier.height(12.dp))
          
          // Quick inquiry buttons
          Text(
            text = "Quick Queries:",
            style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.onPrimaryContainer,
            fontWeight = FontWeight.Medium
          )
          
          Spacer(modifier = Modifier.height(8.dp))
          
          Row(
            horizontalArrangement = Arrangement.spacedBy(6.dp),
            modifier = Modifier.fillMaxWidth()
          ) {
            OutlinedButton(
              onClick = {
                ragViewModel.sendMessage(model, listOf("Provide a comprehensive summary of all video batches analyzed so far"))
              },
              enabled = !isAnalyzing,
              modifier = Modifier.weight(1f)
            ) {
              Icon(Icons.Default.Summarize, contentDescription = null, modifier = Modifier.size(14.dp))
              Spacer(modifier = Modifier.width(4.dp))
              Text("Summary", style = MaterialTheme.typography.labelSmall)
            }
            
            OutlinedButton(
              onClick = {
                ragViewModel.sendMessage(model, listOf("What objects and people have been detected across all video batches?"))
              },
              enabled = !isAnalyzing,
              modifier = Modifier.weight(1f)
            ) {
              Icon(Icons.Default.Visibility, contentDescription = null, modifier = Modifier.size(14.dp))
              Spacer(modifier = Modifier.width(4.dp))
              Text("Objects", style = MaterialTheme.typography.labelSmall)
            }
            
            OutlinedButton(
              onClick = {
                ragViewModel.sendMessage(model, listOf("Describe any movements, actions, or changes observed across the video sequence"))
              },
              enabled = !isAnalyzing,
              modifier = Modifier.weight(1f)
            ) {
              Icon(Icons.Default.PlayArrow, contentDescription = null, modifier = Modifier.size(14.dp))
              Spacer(modifier = Modifier.width(4.dp))
              Text("Actions", style = MaterialTheme.typography.labelSmall)
            }
          }
          
          Spacer(modifier = Modifier.height(8.dp))
          
          OutlinedButton(
            onClick = onClearMemory,
            enabled = !isAnalyzing,
            modifier = Modifier.fillMaxWidth()
          ) {
            Icon(Icons.Default.ClearAll, contentDescription = null)
            Spacer(modifier = Modifier.width(4.dp))
            Text("Clear Memory")
          }
        }
      } else {
        // Show download button when model is not ready
        DownloadAndTryButton(
          task = task,
          model = model,
          downloadStatus = downloadStatus,
          enabled = true,
          modelManagerViewModel = modelManagerViewModel,
          onClicked = { },
          compact = false,
          canShowTryIt = false
        )
      }
    }
  }
}

private fun buildVideoRagAnalysisPrompt(batchNumber: Int): String {
  return """
    Video Batch Analysis #$batchNumber - Analyze and memorize this sequence of video frames captured at 1 FPS intervals.
    
    Please analyze these frames and provide a comprehensive description that includes:
    
    1. **Objects and People**: Identify all objects, people, and their characteristics
    2. **Movements and Actions**: Describe any movements, actions, or activities across frames
    3. **Scene Context**: Describe the overall scene, environment, and setting
    4. **Temporal Information**: Note any changes or progressions between frames
    5. **Key Details**: Include specific details that would be useful for later queries
    
    Structure your response as a detailed narrative that captures:
    - What objects/people are present and where
    - What actions or movements occurred
    - The overall context and setting
    - Any notable patterns or changes
    
    This analysis will be stored in memory to allow querying across multiple video batches. Be thorough and descriptive to enable accurate retrieval for future questions.
  """.trimIndent()
}

// Extension function to check model readiness
private fun Model.isReady(downloadStatus: Map<String, com.google.ai.edge.gallery.data.ModelDownloadStatus>): Boolean {
  val status = downloadStatus[this.name]
  return status?.status == com.google.ai.edge.gallery.data.ModelDownloadStatusType.SUCCEEDED ||
         this.localFileRelativeDirPathOverride.isNotEmpty()
}
