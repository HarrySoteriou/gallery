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
import com.google.ai.edge.gallery.data.Model
import com.google.ai.edge.gallery.data.Task
import com.google.ai.edge.gallery.data.TaskCapabilities
import com.google.ai.edge.gallery.ui.llmchat.LlmChatModelHelper
import com.google.ai.edge.gallery.ui.llmchat.LlmChatViewModelBase

/**
 * Memory management utilities for video analysis that reuse multiturn task logic.
 * 
 * This ensures consistent behavior across all tasks and proper memory cleanup
 * between batches of video frames.
 */
object VideoAnalysisMemoryManager {
  
  /**
   * Clears all accumulated context (images, text, etc.) from the model session
   * while keeping the model loaded in memory.
   * 
   * This is equivalent to the session reset used in multiturn chat tasks.
   * 
   * @param task The video analysis task
   * @param model The model instance to clear
   */
  fun clearContextForNewBatch(task: Task, model: Model) {
    val supportImage = TaskCapabilities.getImageSupport(task, model)
    val supportAudio = TaskCapabilities.getAudioSupport(task, model)
    
    LlmChatModelHelper.resetSession(
      model = model,
      supportImage = supportImage,
      supportAudio = supportAudio,
    )
  }
  
  /**
   * Complete memory clearing workflow for video analysis.
   * This combines UI message clearing with model context clearing.
   * 
   * @param viewModel The chat view model managing the UI state
   * @param task The video analysis task
   * @param model The model instance
   */
  fun clearAllMemoryForNewBatch(
    viewModel: LlmChatViewModelBase,
    task: Task,
    model: Model
  ) {
    // Clear UI messages (existing behavior)
    viewModel.clearAllMessages(model)
    
    // Clear model context using consistent multiturn logic
    clearContextForNewBatch(task, model)
  }
  
  /**
   * Example usage for video frame processing workflow.
   * 
   * Use this pattern in your video frame capture callback to ensure
   * proper memory management between batches.
   */
  fun processNewVideoBatch(
    frames: List<Bitmap>,
    viewModel: LlmChatViewModelBase,
    task: Task,
    model: Model,
    onSendMessage: (Model, List<Any>) -> Unit
  ) {
    // Step 1: Clear all previous context
    clearAllMemoryForNewBatch(viewModel, task, model)
    
    // Step 2: Process new frames (your existing logic)
    // ... your frame processing logic here ...
    
    // Step 3: Send new batch for inference
    // ... your message sending logic here ...
  }
}
