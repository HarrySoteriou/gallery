package com.google.ai.edge.gallery.ui.videosummaryrag

import com.google.ai.edge.gallery.data.Model
import com.google.ai.edge.gallery.data.Task
import com.google.ai.edge.gallery.data.TaskCapabilities
import com.google.ai.edge.gallery.ui.llmchat.LlmChatModelHelper
import com.google.ai.edge.gallery.ui.llmchat.LlmChatViewModelBase
import com.google.ai.edge.gallery.ui.llmrag.LlmRagModelHelper
import com.google.ai.edge.gallery.ui.llmrag.LlmRagViewModel

/**
 * Memory helpers shared by the Video Summary + RAG task. We keep the
 * vision-language model (referred to as [model]) and the retrieval model
 * ([ragModel]) isolated so batches never leak context into either
 * inference session.
 */
object VideoSummaryRagMemoryManager {

  /** Clears the VLM session that ingests frames. */
  fun clearVisionContext(task: Task, model: Model) {
    val supportImage = TaskCapabilities.getImageSupport(task, model)
    val supportAudio = TaskCapabilities.getAudioSupport(task, model)
    LlmChatModelHelper.resetSession(
      model = model,
      supportImage = supportImage,
      supportAudio = supportAudio,
    )
  }

  /** Clears the document store and underlying LLM session used for retrieval. */
  fun clearRagContext(ragModel: Model) {
    LlmRagModelHelper.clearContext(ragModel)
  }

  /**
   * Convenience helper that wipes both chat surfaces.
   */
  fun clearAllContexts(
    task: Task,
    model: Model,
    ragModel: Model,
    visionViewModel: LlmChatViewModelBase?,
    ragViewModel: LlmRagViewModel?,
  ) {
    visionViewModel?.clearAllMessages(model)
    ragViewModel?.clearAllRagMessages(ragModel)
    clearVisionContext(task, model)
    clearRagContext(ragModel)
  }
}
