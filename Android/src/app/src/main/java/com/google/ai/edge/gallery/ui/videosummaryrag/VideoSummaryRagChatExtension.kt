package com.google.ai.edge.gallery.ui.videosummaryrag

import android.graphics.Bitmap
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.google.ai.edge.gallery.data.Model
import com.google.ai.edge.gallery.ui.common.chat.VideoFrameCaptureButton
import com.google.ai.edge.gallery.ui.videosummaryrag.data.VideoBatchRecord

@Composable
fun VideoSummaryRagQuickStart(
  visionModel: Model?,
  ragModel: Model?,
  visionReady: Boolean,
  ragReady: Boolean,
  ragEmbeddingDimension: Int?,
  uiState: VideoSummaryRagViewModel.VideoSummaryRagUiState,
  onProcessBatch: (List<Bitmap>) -> Unit,
  modifier: Modifier = Modifier,
) {
  var capturedFrames by remember { mutableStateOf<List<Bitmap>>(emptyList()) }

  Card(
    modifier = modifier.fillMaxWidth(),
    colors = CardDefaults.cardColors(
      containerColor = MaterialTheme.colorScheme.primaryContainer,
    ),
  ) {
    Column(
      modifier = Modifier.padding(16.dp),
      verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
      Text(
        text = "Video Summary Pipeline",
        style = MaterialTheme.typography.titleMedium,
        color = MaterialTheme.colorScheme.onPrimaryContainer,
      )
      Text(
        text = "Capture five frames at 1 FPS, summarize the clip, and store the description in the RAG memory for later queries.",
        style = MaterialTheme.typography.bodyMedium,
        color = MaterialTheme.colorScheme.onPrimaryContainer,
      )

      Row(
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(12.dp),
      ) {
        VideoFrameCaptureButton(
          onFramesCaptured = { frames -> capturedFrames = frames },
          enabled = visionModel != null && !uiState.isProcessing,
        )
        Button(
          onClick = { if (capturedFrames.isNotEmpty()) onProcessBatch(capturedFrames) },
          enabled = capturedFrames.isNotEmpty() && visionModel != null && ragModel != null && visionReady && ragReady && !uiState.isProcessing,
        ) {
          Icon(imageVector = Icons.Default.PlayArrow, contentDescription = null)
          Spacer(modifier = Modifier.width(8.dp))
          Text("Summarize & Memorize")
        }
      }

      if (capturedFrames.isNotEmpty()) {
        Text(
          text = "Captured ${capturedFrames.size} frames.",
          style = MaterialTheme.typography.bodySmall,
          color = MaterialTheme.colorScheme.onPrimaryContainer,
        )
      }

      if (uiState.isProcessing) {
        Text(
          text = "Processing batch…",
          style = MaterialTheme.typography.bodyMedium,
          color = MaterialTheme.colorScheme.onPrimaryContainer,
        )
      }

      if (!visionReady || !ragReady) {
        val waitingTargets = buildList {
          if (!visionReady) add("vision")
          if (!ragReady) add("RAG")
        }
        Text(
          text = "Waiting for ${waitingTargets.joinToString(" and ")} model${if (waitingTargets.size > 1) "s" else ""} to initialize…",
          style = MaterialTheme.typography.bodySmall,
          color = MaterialTheme.colorScheme.onPrimaryContainer,
        )
      }

      ragEmbeddingDimension?.let { dimension ->
        Text(
          text = "Active RAG embedding dimension: $dimension",
          style = MaterialTheme.typography.bodySmall,
          color = MaterialTheme.colorScheme.onPrimaryContainer,
        )
      }

      uiState.lastBatch?.let { batch ->
        BatchSummaryCard(batch = batch)
      }

      uiState.errorMessage?.let { error ->
        Text(
          text = error,
          style = MaterialTheme.typography.bodySmall,
          color = MaterialTheme.colorScheme.error,
        )
      }
    }
  }
}

@Composable
fun BatchSummaryCard(batch: VideoBatchRecord, modifier: Modifier = Modifier) {
  Card(
    modifier = modifier.fillMaxWidth(),
    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
  ) {
    Column(modifier = Modifier.padding(12.dp)) {
      Text(
        text = "Stored Batch",
        style = MaterialTheme.typography.titleSmall,
        color = MaterialTheme.colorScheme.onSurfaceVariant,
      )
      Spacer(modifier = Modifier.height(6.dp))
      Text(
        text = batch.summary,
        style = MaterialTheme.typography.bodySmall,
        color = MaterialTheme.colorScheme.onSurfaceVariant,
      )
      Spacer(modifier = Modifier.height(6.dp))
      Text(
        text = "Frames: ${batch.frameCount} | Vision Model: ${batch.visionModelName} | RAG Model: ${batch.ragModelName}",
        style = MaterialTheme.typography.labelSmall,
        color = MaterialTheme.colorScheme.onSurfaceVariant,
      )
    }
  }
}

@Composable
fun VideoBatchHistory(
  history: List<VideoBatchRecord>,
  modifier: Modifier = Modifier,
) {
  if (history.isEmpty()) {
    Text(
      text = "No batches stored yet.",
      style = MaterialTheme.typography.bodySmall,
      modifier = modifier,
    )
    return
  }

  Column(modifier = modifier, verticalArrangement = Arrangement.spacedBy(8.dp)) {
    history.forEach { record ->
      Card(
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
      ) {
        Column(modifier = Modifier.padding(12.dp)) {
          Text(
            text = record.summary,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
          )
          Spacer(modifier = Modifier.height(4.dp))
          Text(
            text = "Frames: ${record.frameCount} – Stored doc: ${record.ragDocumentId ?: "N/A"}",
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
          )
        }
      }
    }
  }
}
