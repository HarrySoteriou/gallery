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

import androidx.activity.result.ActivityResultLauncher
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.rounded.Send
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Clear
import androidx.compose.material.icons.filled.Description
import androidx.compose.material.icons.filled.Storage
import androidx.compose.material.icons.rounded.Stop
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.google.ai.edge.gallery.ui.theme.bodyLargeNarrow

@Composable
fun RagMessageInput(
  curMessage: String,
  inProgress: Boolean,
  isResettingSession: Boolean,
  modelPreparing: Boolean,
  modelInitializing: Boolean,
  isProcessingDocument: Boolean,
  onValueChanged: (String) -> Unit,
  onSendMessage: (String) -> Unit,
  onStopButtonClicked: () -> Unit,
  onUploadDocumentClicked: () -> Unit,
  onSelectDocumentClicked: () -> Unit,
  onClearContextClicked: () -> Unit,
  documentPickerLauncher: ActivityResultLauncher<String>?,
  loadAssetDocument: (String, String) -> Unit,
  modifier: Modifier = Modifier,
) {
  var showDocumentMenu by remember { mutableStateOf(false) }

  Box(contentAlignment = Alignment.CenterStart, modifier = modifier) {
    // + button outside the input border (like other tasks)
    IconButton(
      enabled = !inProgress && !isResettingSession && !isProcessingDocument,
      onClick = { showDocumentMenu = true },
      modifier = Modifier.offset(x = 16.dp).alpha(0.8f)
    ) {
      Icon(Icons.Filled.Add, contentDescription = "Document Management", modifier = Modifier.size(28.dp))
    }
    
    Row(
      modifier = Modifier
        .fillMaxWidth()
        .padding(12.dp)
        .border(1.dp, MaterialTheme.colorScheme.outlineVariant, RoundedCornerShape(28.dp)),
      verticalAlignment = Alignment.CenterVertically,
    ) {
    // Document selection dropdown menu
    DropdownMenu(
      expanded = showDocumentMenu,
      onDismissRequest = { showDocumentMenu = false }
    ) {
        // Upload document option
        DropdownMenuItem(
          text = {
            Row(
              verticalAlignment = Alignment.CenterVertically,
              horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
              Icon(Icons.Filled.Description, contentDescription = null)
              Text("Upload Document")
            }
          },
          onClick = {
            showDocumentMenu = false
            documentPickerLauncher?.launch("text/*")
          },
          enabled = !isProcessingDocument
        )
        
        // Select from database option
        DropdownMenuItem(
          text = {
            Row(
              verticalAlignment = Alignment.CenterVertically,
              horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
              Icon(Icons.Filled.Storage, contentDescription = null)
              Text("Select from Database")
            }
          },
          onClick = {
            showDocumentMenu = false
            onSelectDocumentClicked()
          }
        )
        
        HorizontalDivider()
        
        // Sample documents
        DropdownMenuItem(
          text = {
            Row(
              verticalAlignment = Alignment.CenterVertically,
              horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
              Icon(Icons.Filled.Description, contentDescription = null)
              Text("YOLO Detectors")
            }
          },
          onClick = {
            showDocumentMenu = false
            loadAssetDocument("yolo_detectors.txt", "YOLO Object Detection Models")
          },
          enabled = !isProcessingDocument
        )
        
        DropdownMenuItem(
          text = {
            Row(
              verticalAlignment = Alignment.CenterVertically,
              horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
              Icon(Icons.Filled.Description, contentDescription = null)
              Text("LLM Models")
            }
          },
          onClick = {
            showDocumentMenu = false
            loadAssetDocument("llm_models.txt", "Large Language Models Guide")
          },
          enabled = !isProcessingDocument
        )
        
        DropdownMenuItem(
          text = {
            Row(
              verticalAlignment = Alignment.CenterVertically,
              horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
              Icon(Icons.Filled.Description, contentDescription = null)
              Text("Embedding Models")
            }
          },
          onClick = {
            showDocumentMenu = false
            loadAssetDocument("embedding_models.txt", "Embedding Models Guide")
          },
          enabled = !isProcessingDocument
        )
        
        DropdownMenuItem(
          text = {
            Row(
              verticalAlignment = Alignment.CenterVertically,
              horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
              Icon(Icons.Filled.Description, contentDescription = null)
              Text("ASR Models")
            }
          },
          onClick = {
            showDocumentMenu = false
            loadAssetDocument("asr_models.txt", "Automatic Speech Recognition Models")
          },
          enabled = !isProcessingDocument
        )
      }

      // Text field
      TextField(
        value = curMessage,
        minLines = 1,
        maxLines = 3,
        onValueChange = onValueChanged,
        colors = TextFieldDefaults.colors(
          unfocusedContainerColor = Color.Transparent,
          focusedContainerColor = Color.Transparent,
          focusedIndicatorColor = Color.Transparent,
          unfocusedIndicatorColor = Color.Transparent,
          disabledIndicatorColor = Color.Transparent,
          disabledContainerColor = Color.Transparent,
        ),
        textStyle = bodyLargeNarrow,
        modifier = Modifier.weight(1f).padding(start = 36.dp),
        placeholder = { Text("Type your message...") },
      )

      // Clear context button
      IconButton(
        enabled = !inProgress && !isResettingSession,
        onClick = onClearContextClicked,
        colors = IconButtonDefaults.iconButtonColors(
          containerColor = MaterialTheme.colorScheme.errorContainer.copy(alpha = 0.1f)
        )
      ) {
        Icon(
          Icons.Filled.Clear,
          contentDescription = "Clear Context",
          modifier = Modifier.size(20.dp),
          tint = MaterialTheme.colorScheme.error
        )
      }

      Spacer(modifier = Modifier.width(8.dp))

      if (inProgress) {
        if (!modelInitializing && !modelPreparing) {
          IconButton(
            onClick = onStopButtonClicked,
            colors = IconButtonDefaults.iconButtonColors(
              containerColor = MaterialTheme.colorScheme.secondaryContainer
            ),
          ) {
            Icon(
              Icons.Rounded.Stop,
              contentDescription = "",
              tint = MaterialTheme.colorScheme.primary,
            )
          }
        }
      }
      // Send button. Only shown when text is not empty.
      else if (curMessage.isNotEmpty()) {
        IconButton(
          enabled = !inProgress && !isResettingSession,
          onClick = { onSendMessage(curMessage.trim()) },
          colors = IconButtonDefaults.iconButtonColors(
            containerColor = MaterialTheme.colorScheme.secondaryContainer
          ),
        ) {
          Icon(
            Icons.AutoMirrored.Rounded.Send,
            contentDescription = "",
            modifier = Modifier.offset(x = 2.dp),
            tint = MaterialTheme.colorScheme.onSurface,
          )
        }
      }
      
      Spacer(modifier = Modifier.width(4.dp))
    }
  }
}
