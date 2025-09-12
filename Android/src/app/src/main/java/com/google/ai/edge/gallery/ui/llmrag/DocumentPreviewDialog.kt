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
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.material3.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Description
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import java.text.SimpleDateFormat
import java.util.*

@Composable
fun DocumentPreviewDialog(
  document: LlmRagModelHelper.StoredDocument?,
  isLoading: Boolean,
  onDismiss: () -> Unit,
  modifier: Modifier = Modifier
) {
  Dialog(
    onDismissRequest = onDismiss,
    properties = DialogProperties(
      usePlatformDefaultWidth = false
    )
  ) {
    Card(
      modifier = modifier
        .fillMaxWidth(0.95f)
        .fillMaxHeight(0.9f)
        .padding(16.dp),
      elevation = CardDefaults.cardElevation(defaultElevation = 8.dp)
    ) {
      Column(
        modifier = Modifier.fillMaxSize()
      ) {
        // Header
        Row(
          modifier = Modifier
            .fillMaxWidth()
            .padding(16.dp),
          horizontalArrangement = Arrangement.SpaceBetween,
          verticalAlignment = Alignment.CenterVertically
        ) {
          Column(
            modifier = Modifier.weight(1f)
          ) {
            Text(
              text = document?.metadata?.title ?: "Loading...",
              style = MaterialTheme.typography.headlineSmall,
              fontWeight = FontWeight.Bold
            )
            
            if (document != null) {
              Spacer(modifier = Modifier.height(4.dp))
              Text(
                text = formatTimestamp(document.metadata.timestamp),
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
              )
            }
          }
          
          IconButton(onClick = onDismiss) {
            Icon(
              Icons.Default.Close,
              contentDescription = "Close"
            )
          }
        }
        
        Divider()
        
        // Document info
        if (document != null) {
          Row(
            modifier = Modifier
              .fillMaxWidth()
              .padding(16.dp),
            horizontalArrangement = Arrangement.SpaceBetween
          ) {
            Text(
              text = "${document.chunks.size} chunks",
              style = MaterialTheme.typography.bodyMedium,
              color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            
            Text(
              text = formatSource(document.metadata.source),
              style = MaterialTheme.typography.bodyMedium,
              color = MaterialTheme.colorScheme.primary
            )
          }
          
          Divider()
        }
        
        // Content
        if (isLoading) {
          Box(
            modifier = Modifier.fillMaxSize(),
            contentAlignment = Alignment.Center
          ) {
            CircularProgressIndicator()
          }
        } else if (document == null) {
          Box(
            modifier = Modifier.fillMaxSize(),
            contentAlignment = Alignment.Center
          ) {
            Column(
              horizontalAlignment = Alignment.CenterHorizontally
            ) {
              Icon(
                Icons.Default.Description,
                contentDescription = null,
                modifier = Modifier.size(64.dp),
                tint = MaterialTheme.colorScheme.outline
              )
              Spacer(modifier = Modifier.height(16.dp))
              Text(
                text = "Document not found",
                style = MaterialTheme.typography.bodyLarge,
                color = MaterialTheme.colorScheme.onSurfaceVariant
              )
            }
          }
        } else {
          LazyColumn(
            modifier = Modifier.fillMaxSize(),
            contentPadding = PaddingValues(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
          ) {
            itemsIndexed(document.chunks) { index, chunk ->
              ChunkItem(
                chunkIndex = index + 1,
                content = chunk
              )
            }
          }
        }
      }
    }
  }
}

@Composable
private fun ChunkItem(
  chunkIndex: Int,
  content: String,
  modifier: Modifier = Modifier
) {
  Card(
    modifier = modifier.fillMaxWidth(),
    elevation = CardDefaults.cardElevation(defaultElevation = 2.dp),
    colors = CardDefaults.cardColors(
      containerColor = MaterialTheme.colorScheme.surfaceVariant
    )
  ) {
    Column(
      modifier = Modifier.padding(16.dp)
    ) {
      Text(
        text = "Chunk $chunkIndex",
        style = MaterialTheme.typography.labelMedium,
        fontWeight = FontWeight.Medium,
        color = MaterialTheme.colorScheme.primary
      )
      
      Spacer(modifier = Modifier.height(8.dp))
      
      Text(
        text = content,
        style = MaterialTheme.typography.bodyMedium,
        color = MaterialTheme.colorScheme.onSurfaceVariant
      )
    }
  }
}

private fun formatTimestamp(timestamp: Long): String {
  return if (timestamp > 0) {
    SimpleDateFormat("MMM dd, yyyy HH:mm", Locale.getDefault()).format(Date(timestamp))
  } else {
    "Unknown date"
  }
}

private fun formatSource(source: String): String {
  return when (source) {
    "upload" -> "Uploaded"
    "sample" -> "Sample"
    "image_analysis" -> "Image Analysis"
    "video_analysis" -> "Video Analysis"
    else -> source.replaceFirstChar { if (it.isLowerCase()) it.titlecase() else it.toString() }
  }
}
