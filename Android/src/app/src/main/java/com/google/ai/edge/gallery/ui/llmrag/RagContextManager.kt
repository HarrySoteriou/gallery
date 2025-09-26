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
import com.google.ai.edge.gallery.data.Model
import com.google.ai.edge.gallery.data.Task
import com.google.ai.edge.gallery.data.TaskCapabilities
import com.google.ai.edge.gallery.ui.llmchat.LlmChatModelHelper

private const val TAG = "AGRagContextMgr"

/**
 * Shared context management helpers for all RAG-enabled tasks.
 *
 * The functions here keep the persistent knowledge base separate from the
 * streaming model session so that chat-style tasks and video batches can reuse
 * the same cleanup behavior.
 */
object RagContextManager {

  fun clearChatTurn(model: Model, dropPersistentMemory: Boolean = false) {
    if (dropPersistentMemory) {
      RagKnowledgeBase.clearDocuments()
    }
    resetSessionInternal(model, supportImage = false, supportAudio = false)
  }

  fun clearBatch(task: Task, model: Model, dropPersistentMemory: Boolean = false) {
    if (dropPersistentMemory) {
      RagKnowledgeBase.clearDocuments()
    }
    val supportImage = TaskCapabilities.getImageSupport(task, model)
    val supportAudio = TaskCapabilities.getAudioSupport(task, model)
    resetSessionInternal(model, supportImage, supportAudio)
  }

  private fun resetSessionInternal(
    model: Model,
    supportImage: Boolean,
    supportAudio: Boolean,
  ) {
    val ragInstance = model.instance as? RagModelInstance
    val targetModel = if (ragInstance != null) {
      Model(name = model.name).apply { instance = ragInstance.llmInstance }
    } else {
      model
    }

    try {
      LlmChatModelHelper.resetSession(
        model = targetModel,
        supportImage = supportImage,
        supportAudio = supportAudio,
      )
    } catch (e: Exception) {
      Log.e(TAG, "Failed to reset session: ${e.message}")
    }
  }
}
