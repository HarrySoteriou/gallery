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

import android.content.Context
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Storage
import androidx.compose.runtime.Composable
import com.google.ai.edge.gallery.R
import com.google.ai.edge.gallery.customtasks.common.CustomTask
import com.google.ai.edge.gallery.customtasks.common.CustomTaskDataForBuiltinTask
import com.google.ai.edge.gallery.data.BuiltInTaskId
import com.google.ai.edge.gallery.data.Category
import com.google.ai.edge.gallery.data.Model
import com.google.ai.edge.gallery.data.Task
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import dagger.multibindings.IntoSet
import kotlinx.coroutines.CoroutineScope
import javax.inject.Inject
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import android.util.Log

class LlmRagTask @Inject constructor() : CustomTask {
  override val task: Task =
    Task(
      id = BuiltInTaskId.LLM_RAG,
      label = "RAG Chat",
      category = Category.LLM,
      icon = Icons.Default.Storage,
      models = mutableListOf(),
      description = "Chat with on-device large language models enhanced with Retrieval Augmented Generation (RAG) for context-aware responses using your own documents and data",
      docUrl = "https://ai.google.dev/edge/mediapipe/solutions/genai/rag/android",
      sourceCodeUrl =
        "https://github.com/google-ai-edge/gallery/blob/main/Android/src/app/src/main/java/com/google/ai/edge/gallery/ui/llmrag/LlmRagModelHelper.kt",
      textInputPlaceHolderRes = R.string.text_input_placeholder_llm_chat,
    )

  override fun initializeModelFn(
    context: Context,
    coroutineScope: CoroutineScope,
    model: Model,
    onDone: (String) -> Unit,
  ) {
    LlmRagModelHelper.initialize(
      context = context,
      model = model,
      onDone = onDone,
    )
  }

  override fun cleanUpModelFn(
    context: Context,
    coroutineScope: CoroutineScope,
    model: Model,
    onDone: () -> Unit,
  ) {
    LlmRagModelHelper.cleanUp(model = model, onDone = onDone)
  }

  @Composable
  override fun MainScreen(data: Any) {
    val myData = data as CustomTaskDataForBuiltinTask
    LlmRagScreen(
      modelManagerViewModel = myData.modelManagerViewModel,
      navigateUp = myData.onNavUp,
    )
  }
}

class RagStubTask : CustomTask {
  override val task: Task =
    Task(
      id = BuiltInTaskId.LLM_RAG,
      label = "RAG Chat (Unavailable)",
      category = Category.LLM,
      icon = Icons.Default.Storage,
      models = mutableListOf(),
      description = "RAG functionality is currently unavailable due to missing native libraries",
      textInputPlaceHolderRes = R.string.text_input_placeholder_llm_chat,
    )

  override fun initializeModelFn(
    context: Context,
    coroutineScope: CoroutineScope,
    model: Model,
    onDone: (String) -> Unit,
  ) {
    onDone("RAG functionality is not available. Native libraries are missing.")
  }

  override fun cleanUpModelFn(
    context: Context,
    coroutineScope: CoroutineScope,
    model: Model,
    onDone: () -> Unit,
  ) {
    onDone()
  }

  @Composable
  override fun MainScreen(data: Any) {
    Column(modifier = Modifier.padding(16.dp)) {
      Text(
        text = "RAG functionality is unavailable",
        style = MaterialTheme.typography.headlineSmall,
        modifier = Modifier.padding(bottom = 8.dp)
      )
      Text(
        text = "The RAG (Retrieval Augmented Generation) feature requires native libraries that are not currently available on this device. This functionality has been disabled to prevent crashes.",
        style = MaterialTheme.typography.bodyMedium
      )
    }
  }
}

@Module
@InstallIn(SingletonComponent::class)
internal object LlmRagTaskModule {
  @Provides
  @IntoSet
  fun provideTask(): CustomTask {
    // Check if RAG native libraries are available before loading the actual task class
    return if (canLoadRagLibraries()) {
      try {
        LlmRagTask()
      } catch (e: Throwable) {
        RagStubTask()
      }
    } else {
      RagStubTask()
    }
  }
  
  private fun canLoadRagLibraries(): Boolean {
    return try {
      // With the official Maven dependency, the libraries should be available automatically
      // Just check if the main RAG classes can be loaded
      Class.forName("com.google.ai.edge.localagents.rag.models.GeckoEmbeddingModel")
      Class.forName("com.google.ai.edge.localagents.rag.memory.SqliteVectorStore")
      true
    } catch (e: ClassNotFoundException) {
      Log.w("LlmRagTaskModule", "RAG classes not available: ${e.message}")
      false
    } catch (e: Exception) {
      Log.e("LlmRagTaskModule", "Error checking RAG libraries: ${e.message}")
      false
    }
  }
}
