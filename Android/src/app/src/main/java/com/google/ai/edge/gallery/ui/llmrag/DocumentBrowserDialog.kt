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
import androidx.compose.material3.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Description
import androidx.compose.material.icons.filled.Visibility
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import java.text.SimpleDateFormat
import java.util.*

@Composable
fun DocumentBrowserDialog(
  documents: List<LlmRagModelHelper.DocumentMetadata>,
  isLoading: Boolean,
  onDismiss: () -> Unit,
  onRefresh: () -> Unit,
  onViewDocument: (String) -> Unit,
  onDeleteDocument: (String) -> Unit,
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
        .fillMaxHeight(0.8f)
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
          Text(
            text = "Knowledge Base",
            style = MaterialTheme.typography.headlineSmall,
            fontWeight = FontWeight.Bold
          )
          
          Row(
            horizontalArrangement = Arrangement.spacedBy(8.dp)
          ) {
            IconButton(onClick = onRefresh) {
              Icon(
                Icons.Default.Refresh,
                contentDescription = "Refresh"
              )
            }
            
            IconButton(onClick = onDismiss) {
              Icon(
                Icons.Default.Close,
                contentDescription = "Close"
              )
            }
          }
        }
        
        Divider()
        
        // Content
        if (isLoading) {
          Box(
            modifier = Modifier.fillMaxSize(),
            contentAlignment = Alignment.Center
          ) {
            CircularProgressIndicator()
          }
        } else if (documents.isEmpty()) {
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
                text = "No documents in knowledge base",
                style = MaterialTheme.typography.bodyLarge,
                color = MaterialTheme.colorScheme.onSurfaceVariant
              )
              Spacer(modifier = Modifier.height(8.dp))
              Text(
                text = "Upload documents or add sample content to get started",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.outline
              )
            }
          }
        } else {
          LazyColumn(
            modifier = Modifier.fillMaxSize(),
            contentPadding = PaddingValues(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
          ) {
            items(documents) { document ->
              DocumentItem(
                document = document,
                onView = { onViewDocument(document.id) },
                onDelete = { onDeleteDocument(document.id) }
              )
            }
          }
        }
      }
    }
  }
}

@Composable
private fun DocumentItem(
  document: LlmRagModelHelper.DocumentMetadata,
  onView: () -> Unit,
  onDelete: () -> Unit,
  modifier: Modifier = Modifier
) {
  var showDeleteConfirmation by remember { mutableStateOf(false) }
  
  Card(
    modifier = modifier.fillMaxWidth(),
    elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
  ) {
    Column(
      modifier = Modifier.padding(16.dp)
    ) {
      Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
      ) {
        Column(
          modifier = Modifier.weight(1f)
        ) {
          Text(
            text = document.title,
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.Medium,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis
          )
          
          Spacer(modifier = Modifier.height(4.dp))
          
          Text(
            text = formatTimestamp(document.timestamp),
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
          )
        }
        
        Row(
          horizontalArrangement = Arrangement.spacedBy(4.dp)
        ) {
          IconButton(onClick = onView) {
            Icon(
              Icons.Default.Visibility,
              contentDescription = "View document",
              tint = MaterialTheme.colorScheme.primary
            )
          }
          
          IconButton(onClick = { showDeleteConfirmation = true }) {
            Icon(
              Icons.Default.Delete,
              contentDescription = "Delete document",
              tint = MaterialTheme.colorScheme.error
            )
          }
        }
      }
      
      Spacer(modifier = Modifier.height(8.dp))
      
      Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween
      ) {
        Text(
          text = "${document.chunkCount} chunks",
          style = MaterialTheme.typography.bodySmall,
          color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        
        Text(
          text = formatSource(document.source),
          style = MaterialTheme.typography.bodySmall,
          color = MaterialTheme.colorScheme.primary
        )
      }
    }
  }
  
  // Delete confirmation dialog
  if (showDeleteConfirmation) {
    AlertDialog(
      onDismissRequest = { showDeleteConfirmation = false },
      title = { Text("Delete Document") },
      text = { Text("Are you sure you want to delete \"${document.title}\"? This action cannot be undone.") },
      confirmButton = {
        TextButton(
          onClick = {
            onDelete()
            showDeleteConfirmation = false
          }
        ) {
          Text("Delete")
        }
      },
      dismissButton = {
        TextButton(onClick = { showDeleteConfirmation = false }) {
          Text("Cancel")
        }
      }
    )
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
