package com.google.ai.edge.gallery.ui.videosummaryrag

import android.content.Context
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Movie
import androidx.compose.runtime.Composable
import com.google.ai.edge.gallery.R
import com.google.ai.edge.gallery.customtasks.common.CustomTask
import com.google.ai.edge.gallery.customtasks.common.CustomTaskDataForBuiltinTask
import com.google.ai.edge.gallery.data.BuiltInTaskId
import com.google.ai.edge.gallery.data.Category
import com.google.ai.edge.gallery.data.Model
import com.google.ai.edge.gallery.data.Task
import com.google.ai.edge.gallery.ui.llmchat.LlmChatModelHelper
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import dagger.multibindings.IntoSet
import kotlinx.coroutines.CoroutineScope
import javax.inject.Inject

class VideoSummaryRagTask @Inject constructor() : CustomTask {
  override val task: Task =
    Task(
      id = BuiltInTaskId.VIDEO_RAG_ANALYSIS,
      label = "Video Summary + RAG",
      category = Category.LLM,
      icon = Icons.Default.Movie,
      models = mutableListOf(),
      description = "Capture frames, summarise them with a vision-language model, and query stored batches using retrieval-augmented chat.",
      docUrl = "https://ai.google.dev/edge/mediapipe/solutions/genai/rag/android",
      sourceCodeUrl = "https://github.com/google-ai-edge/gallery",
      textInputPlaceHolderRes = R.string.text_input_placeholder_llm_chat,
    )

  override fun initializeModelFn(
    context: Context,
    coroutineScope: CoroutineScope,
    model: Model,
    onDone: (String) -> Unit,
  ) {
    LlmChatModelHelper.initialize(
      context = context,
      model = model,
      supportImage = true,
      supportAudio = false,
      onDone = onDone,
    )
  }

  override fun cleanUpModelFn(
    context: Context,
    coroutineScope: CoroutineScope,
    model: Model,
    onDone: () -> Unit,
  ) {
    LlmChatModelHelper.cleanUp(model = model, onDone = onDone)
  }

  @Composable
  override fun MainScreen(data: Any) {
    val taskData = data as CustomTaskDataForBuiltinTask
    VideoSummaryRagScreen(
      modelManagerViewModel = taskData.modelManagerViewModel,
      navigateUp = taskData.onNavUp,
    )
  }
}

@Module
@InstallIn(SingletonComponent::class)
internal object VideoSummaryRagTaskModule {
  @Provides
  @IntoSet
  fun provideTask(): CustomTask = VideoSummaryRagTask()
}
