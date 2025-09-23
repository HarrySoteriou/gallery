package com.google.ai.edge.gallery.ui.videosummaryrag

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Divider
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowDropDown
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.google.ai.edge.gallery.data.BuiltInTaskId
import com.google.ai.edge.gallery.data.Model
import com.google.ai.edge.gallery.data.ModelDownloadStatusType
import com.google.ai.edge.gallery.data.Task
import com.google.ai.edge.gallery.ui.common.chat.ChatMessageText
import com.google.ai.edge.gallery.ui.common.chat.ChatView
import com.google.ai.edge.gallery.ui.common.chat.ChatInputType
import com.google.ai.edge.gallery.ui.llmrag.LlmRagModelHelper
import com.google.ai.edge.gallery.ui.llmrag.LlmRagViewModel
import com.google.ai.edge.gallery.ui.modelmanager.ModelInitializationStatusType
import com.google.ai.edge.gallery.ui.modelmanager.ModelManagerViewModel

@Composable
fun VideoSummaryRagScreen(
  modelManagerViewModel: ModelManagerViewModel,
  navigateUp: () -> Unit,
  modifier: Modifier = Modifier,
  viewModel: VideoSummaryRagViewModel = hiltViewModel(),
  ragChatViewModel: LlmRagViewModel = hiltViewModel(),
) {
  val context = LocalContext.current
  val managerState by modelManagerViewModel.uiState.collectAsStateWithLifecycle()
  val summaryUiState by viewModel.uiState.collectAsStateWithLifecycle()

  LaunchedEffect(Unit) {
    modelManagerViewModel.loadModelAllowlistWhenNeeded()
    viewModel.refreshHistory()
  }

  val videoTask = modelManagerViewModel.getTaskById(BuiltInTaskId.VIDEO_RAG_ANALYSIS)
  val ragTask = modelManagerViewModel.getTaskById(BuiltInTaskId.LLM_RAG)

  if (videoTask == null || ragTask == null) {
    Column(modifier = modifier.padding(16.dp)) {
      Text(text = "Video Summary + RAG configuration is unavailable.")
    }
    return
  }

  val visionModels = remember(managerState.tasks) { videoTask.models.toList() }
  val ragModels = remember(managerState.tasks) { ragTask.models.toList() }

  var selectedVisionModelName by remember(visionModels) {
    mutableStateOf(visionModels.firstOrNull()?.name ?: "")
  }
  if ((selectedVisionModelName.isEmpty() || visionModels.none { it.name == selectedVisionModelName }) &&
    visionModels.isNotEmpty()
  ) {
    selectedVisionModelName = visionModels.first().name
  }

  var selectedRagModelName by remember(ragModels) {
    mutableStateOf(ragModels.firstOrNull()?.name ?: "")
  }
  if ((selectedRagModelName.isEmpty() || ragModels.none { it.name == selectedRagModelName }) &&
    ragModels.isNotEmpty()
  ) {
    selectedRagModelName = ragModels.first().name
  }

  val visionModel = modelManagerViewModel.getModelByName(selectedVisionModelName)
  val ragModel = modelManagerViewModel.getModelByName(selectedRagModelName)
  val ragEmbeddingDimension = ragModel?.let { LlmRagModelHelper.getResolvedEmbeddingDimension(it) }

  LaunchedEffect(ragModel?.name) {
    ragModel?.let { modelManagerViewModel.selectModel(it) }
  }

  val visionDownloadStatus = visionModel?.let { managerState.modelDownloadStatus[it.name] }
  val ragDownloadStatus = ragModel?.let { managerState.modelDownloadStatus[it.name] }

  fun shouldAutoInitialize(
    model: Model?,
    downloadStatus: ModelDownloadStatusType?,
    initializationStatus: ModelInitializationStatusType?
  ): Boolean {
    if (model == null) return false
    if (initializationStatus == ModelInitializationStatusType.INITIALIZED ||
      initializationStatus == ModelInitializationStatusType.INITIALIZING
    ) {
      return false
    }
    if (downloadStatus == ModelDownloadStatusType.SUCCEEDED) return true
    if (model.localFileRelativeDirPathOverride.isNotEmpty()) return true
    if (model.imported) return true
    return false
  }

  LaunchedEffect(visionDownloadStatus?.status, visionModel?.name) {
    val status = visionDownloadStatus?.status
    val initStatus = visionModel?.let { managerState.modelInitializationStatus[it.name]?.status }
    if (shouldAutoInitialize(visionModel, status, initStatus)) {
      modelManagerViewModel.initializeModel(context, videoTask, visionModel!!)
    }
  }

  LaunchedEffect(ragDownloadStatus?.status, ragModel?.name) {
    val status = ragDownloadStatus?.status
    val initStatus = ragModel?.let { managerState.modelInitializationStatus[it.name]?.status }
    if (shouldAutoInitialize(ragModel, status, initStatus)) {
      modelManagerViewModel.initializeModel(context, ragTask, ragModel!!)
    }
  }

  LaunchedEffect(Unit) {
    viewModel.memorizationEvents.collect {
      ragChatViewModel.refreshStoredDocuments()
    }
  }

  val scrollState = rememberScrollState()

  Column(
    modifier = modifier
      .fillMaxSize()
      .verticalScroll(scrollState)
      .padding(16.dp),
    verticalArrangement = Arrangement.spacedBy(24.dp),
  ) {
    Text(
      text = "Video Summary + RAG",
      style = MaterialTheme.typography.headlineSmall,
    )
    Text(
      text = "Capture short clips, summarise them with a vision-language model, and query the stored knowledge via RAG chat.",
      style = MaterialTheme.typography.bodyMedium,
    )

    ModelSelectorSection(
      title = "Vision Model",
      models = visionModels,
      selectedName = selectedVisionModelName,
      onModelSelected = { selectedVisionModelName = it },
    )
    ModelSelectorSection(
      title = "RAG Model",
      models = ragModels,
      selectedName = selectedRagModelName,
      onModelSelected = { selectedRagModelName = it },
    )

    val visionReady = visionModel?.let { model ->
      managerState.modelInitializationStatus[model.name]?.status == ModelInitializationStatusType.INITIALIZED
    } ?: false
    val ragReady = ragModel?.let { model ->
      managerState.modelInitializationStatus[model.name]?.status == ModelInitializationStatusType.INITIALIZED
    } ?: false

    VideoSummaryRagQuickStart(
      visionModel = visionModel,
      ragModel = ragModel,
      visionReady = visionReady,
      ragReady = ragReady,
      ragEmbeddingDimension = ragEmbeddingDimension,
      uiState = summaryUiState,
      onProcessBatch = { frames ->
        if (visionModel != null && ragModel != null) {
          viewModel.processBatch(
            task = videoTask,
            model = visionModel,
            ragTask = ragTask,
            ragModel = ragModel,
            frames = frames,
          )
        }
      },
    )

    Text(text = "Stored batches", style = MaterialTheme.typography.titleMedium)
    VideoBatchHistory(history = summaryUiState.history)

    Divider()

    ragModel?.let {
      VideoSummaryRagChatView(
        task = ragTask,
        modelManagerViewModel = modelManagerViewModel,
        navigateUp = navigateUp,
        viewModel = ragChatViewModel,
      )
    }
  }
}

@Composable
private fun ModelSelectorSection(
  title: String,
  models: List<Model>,
  selectedName: String,
  onModelSelected: (String) -> Unit,
) {
  Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
    Text(text = title, style = MaterialTheme.typography.labelLarge)
    ModelDropdown(models = models, selectedName = selectedName, onModelSelected = onModelSelected)
  }
}

@Composable
private fun ModelDropdown(
  models: List<Model>,
  selectedName: String,
  onModelSelected: (String) -> Unit,
) {
  var expanded by remember { mutableStateOf(false) }
  val selectedModel = models.firstOrNull { it.name == selectedName }
  val label = selectedModel?.name ?: selectedName
  val interactionSource = remember { MutableInteractionSource() }

  androidx.compose.foundation.layout.Box {
    OutlinedTextField(
      value = label,
      onValueChange = {},
      readOnly = true,
      label = { Text("Select model") },
      trailingIcon = {
        Icon(
          imageVector = Icons.Default.ArrowDropDown,
          contentDescription = null,
        )
      },
      modifier = Modifier
        .fillMaxWidth()
        .clickable(
          interactionSource = interactionSource,
          indication = null,
        ) { expanded = !expanded },
      interactionSource = interactionSource,
    )

    DropdownMenu(
      expanded = expanded,
      onDismissRequest = { expanded = false },
    ) {
      models.forEach { model ->
        DropdownMenuItem(
          text = { Text(model.name) },
          onClick = {
            onModelSelected(model.name)
            expanded = false
          },
        )
      }
    }
  }
}

@Composable
private fun VideoSummaryRagChatView(
  task: Task,
  modelManagerViewModel: ModelManagerViewModel,
  navigateUp: () -> Unit,
  viewModel: LlmRagViewModel,
) {
  Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
    Text(text = "RAG Chat", style = MaterialTheme.typography.titleMedium)
    ChatView(
      task = task,
      viewModel = viewModel,
      modelManagerViewModel = modelManagerViewModel,
      onSendMessage = { model, messages ->
        val textMessages = messages.filterIsInstance<ChatMessageText>()
        if (textMessages.isNotEmpty()) {
          val content = textMessages.map { it.content }
          viewModel.sendMessage(model, content)
        }
      },
      onRunAgainClicked = { _, _ -> },
      onBenchmarkClicked = { _, _, _, _ -> },
      onResetSessionClicked = { model -> viewModel.clearAllRagMessages(model) },
      navigateUp = navigateUp,
      chatInputType = ChatInputType.RAG,
    )
  }
}
