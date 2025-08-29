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

package com.google.ai.edge.gallery.ui.videoanalysis

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
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.google.ai.edge.gallery.R
import com.google.ai.edge.gallery.data.BuiltInTaskId
import com.google.ai.edge.gallery.data.Model
import com.google.ai.edge.gallery.ui.common.DownloadAndTryButton
import com.google.ai.edge.gallery.ui.common.chat.VideoFrameCaptureButton
import com.google.ai.edge.gallery.ui.llmrag.LlmRagViewModel
import com.google.ai.edge.gallery.ui.modelmanager.ModelManagerViewModel
import kotlinx.coroutines.launch

@Composable
fun VideoAnalysisWithRagScreen(
  modelManagerViewModel: ModelManagerViewModel,
  navigateUp: () -> Unit,
  modifier: Modifier = Modifier,
  ragViewModel: LlmRagViewModel = hiltViewModel(),
) {
  val selectedModel by modelManagerViewModel.selectedModel.collectAsStateWithLifecycle()
  val modelManagerUiState by modelManagerViewModel.uiState.collectAsStateWithLifecycle()
  
  var capturedFrames by remember { mutableStateOf<List<Bitmap>>(emptyList()) }
  var batchCount by remember { mutableStateOf(0) }
  var isAnalyzing by remember { mutableStateOf(false) }
  var lastAnalysisResult by remember { mutableStateOf("") }
  var ragModel by remember { mutableStateOf<Model?>(null) }
  
  val coroutineScope = rememberCoroutineScope()
  
  // Find RAG model
  LaunchedEffect(modelManagerUiState.tasks) {
    ragModel = modelManagerUiState.tasks
      .find { it.id == BuiltInTaskId.LLM_RAG }
      ?.models
      ?.firstOrNull { it.isReady(modelManagerUiState.modelDownloadStatus) }
  }

  Column(modifier = modifier.fillMaxSize()) {
    // Header
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
          text = "Video Analysis with RAG",
          style = MaterialTheme.typography.titleMedium,
          fontWeight = FontWeight.Bold
        )
        
        Spacer(modifier = Modifier.height(8.dp))
        
        Text(
          text = "Capture video frames, analyze them, and build a knowledge base for querying longer video sequences.",
          style = MaterialTheme.typography.bodyMedium,
          color = MaterialTheme.colorScheme.onSurfaceVariant
        )
      }
    }

    // Models status
    Card(
      modifier = Modifier
        .fillMaxWidth()
        .padding(horizontal = 16.dp),
      colors = CardDefaults.cardColors(
        containerColor = MaterialTheme.colorScheme.surfaceVariant
      )
    ) {
      Column(
        modifier = Modifier.padding(12.dp)
      ) {
        Text(
          text = "Model Status",
          style = MaterialTheme.typography.titleSmall,
          fontWeight = FontWeight.Bold
        )
        
        Spacer(modifier = Modifier.height(8.dp))
        
        Row(
          modifier = Modifier.fillMaxWidth(),
          horizontalArrangement = Arrangement.SpaceBetween
        ) {
          Column {
            Text(
              text = "VLM (Vision):",
              style = MaterialTheme.typography.bodySmall
            )
            Text(
              text = selectedModel?.name ?: "None selected",
              style = MaterialTheme.typography.bodySmall,
              color = if (selectedModel?.isReady(modelManagerUiState.modelDownloadStatus) == true) 
                Color(0xFF4CAF50) else Color(0xFFF44336)
            )
          }
          
          Column {
            Text(
              text = "RAG Model:",
              style = MaterialTheme.typography.bodySmall
            )
            Text(
              text = ragModel?.name ?: "Not available",
              style = MaterialTheme.typography.bodySmall,
              color = if (ragModel != null) Color(0xFF4CAF50) else Color(0xFFF44336)
            )
          }
        }
      }
    }

    Spacer(modifier = Modifier.height(16.dp))

    // Video capture and analysis section
    Card(
      modifier = Modifier
        .fillMaxWidth()
        .padding(horizontal = 16.dp),
      colors = CardDefaults.cardColors(
        containerColor = MaterialTheme.colorScheme.primaryContainer
      )
    ) {
      Column(
        modifier = Modifier.padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
      ) {
        Text(
          text = "Batch Analysis (${batchCount} batches processed)",
          style = MaterialTheme.typography.titleMedium,
          fontWeight = FontWeight.Bold,
          color = MaterialTheme.colorScheme.onPrimaryContainer
        )
        
        Spacer(modifier = Modifier.height(12.dp))
        
        val canAnalyze = selectedModel?.isReady(modelManagerUiState.modelDownloadStatus) == true && 
                        ragModel != null && !isAnalyzing
        
        if (canAnalyze) {
          Row(
            horizontalArrangement = Arrangement.spacedBy(12.dp),
            verticalAlignment = Alignment.CenterVertically
          ) {
            VideoFrameCaptureButton(
              onFramesCaptured = { frames ->
                capturedFrames = frames
              },
              enabled = !isAnalyzing
            )
            
            if (capturedFrames.isNotEmpty()) {
              Button(
                onClick = {
                  coroutineScope.launch {
                    isAnalyzing = true
                    selectedModel?.let { vlm ->
                      ragModel?.let { rag ->
                        VideoAnalysisRagIntegration.analyzeAndMemorizeBatch(
                          vlmModel = vlm,
                          ragModel = rag,
                          frames = capturedFrames,
                          batchIndex = batchCount
                        ).let { (result, error) ->
                          if (error.isEmpty()) {
                            lastAnalysisResult = result
                            batchCount++
                          } else {
                            lastAnalysisResult = "Error: $error"
                          }
                        }
                      }
                    }
                    isAnalyzing = false
                  }
                },
                enabled = !isAnalyzing
              ) {
                if (isAnalyzing) {
                  CircularProgressIndicator(
                    modifier = Modifier.size(16.dp),
                    strokeWidth = 2.dp
                  )
                } else {
                  Icon(Icons.Default.Psychology, contentDescription = null)
                }
                Spacer(modifier = Modifier.width(8.dp))
                Text(if (isAnalyzing) "Analyzing..." else "Analyze & Store")
              }
            }
          }
          
          if (capturedFrames.isNotEmpty()) {
            Spacer(modifier = Modifier.height(8.dp))
            Text(
              text = "✓ ${capturedFrames.size} frames ready for analysis",
              style = MaterialTheme.typography.bodySmall,
              color = MaterialTheme.colorScheme.onPrimaryContainer
            )
          }
        } else {
          Text(
            text = "Please ensure both VLM and RAG models are available",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.error
          )
        }
      }
    }

    Spacer(modifier = Modifier.height(16.dp))

    // RAG query section
    if (batchCount > 0 && ragModel != null) {
      Card(
        modifier = Modifier
          .fillMaxWidth()
          .padding(horizontal = 16.dp),
        colors = CardDefaults.cardColors(
          containerColor = MaterialTheme.colorScheme.secondaryContainer
        )
      ) {
        Column(
          modifier = Modifier.padding(16.dp)
        ) {
          Text(
            text = "Query Video Content",
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.onSecondaryContainer
          )
          
          Spacer(modifier = Modifier.height(8.dp))
          
          Text(
            text = "Ask questions about the analyzed video content across all batches.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSecondaryContainer
          )
          
          Spacer(modifier = Modifier.height(12.dp))
          
          Row(
            horizontalArrangement = Arrangement.spacedBy(8.dp)
          ) {
            Button(
              onClick = {
                coroutineScope.launch {
                  ragModel?.let { model ->
                    val summary = VideoAnalysisRagIntegration.getVideoSummary(model)
                    ragViewModel.sendMessage(model, listOf("Provide a summary of all analyzed video content"))
                  }
                }
              },
              modifier = Modifier.weight(1f)
            ) {
              Icon(Icons.Default.Summarize, contentDescription = null)
              Spacer(modifier = Modifier.width(4.dp))
              Text("Get Summary")
            }
            
            OutlinedButton(
              onClick = {
                ragModel?.let { model ->
                  VideoAnalysisRagIntegration.clearVideoMemory(model)
                  batchCount = 0
                  lastAnalysisResult = ""
                }
              },
              modifier = Modifier.weight(1f)
            ) {
              Icon(Icons.Default.ClearAll, contentDescription = null)
              Spacer(modifier = Modifier.width(4.dp))
              Text("Clear Memory")
            }
          }
        }
      }
    }

    // Show last analysis result
    if (lastAnalysisResult.isNotEmpty()) {
      Spacer(modifier = Modifier.height(16.dp))
      Card(
        modifier = Modifier
          .fillMaxWidth()
          .padding(horizontal = 16.dp)
      ) {
        Column(
          modifier = Modifier.padding(12.dp)
        ) {
          Text(
            text = "Last Analysis Result",
            style = MaterialTheme.typography.titleSmall,
            fontWeight = FontWeight.Bold
          )
          Spacer(modifier = Modifier.height(8.dp))
          Text(
            text = lastAnalysisResult,
            style = MaterialTheme.typography.bodySmall
          )
        }
      }
    }
  }
}

// Extension function to check model readiness
private fun Model.isReady(downloadStatus: Map<String, com.google.ai.edge.gallery.data.ModelDownloadStatusInfo>): Boolean {
  val status = downloadStatus[this.name]
  return status?.status == com.google.ai.edge.gallery.data.ModelDownloadStatusType.SUCCEEDED ||
         this.localFileRelativeDirPathOverride.isNotEmpty()
}
