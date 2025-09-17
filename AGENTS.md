# Agents

  ## Why Agents Exist
  - Present curated multimodal demos (chat, RAG, vision) inside a unified gallery app.
  - Allow model authors to plug new experiences into the home grid without modifying core UI.
  - Reuse shared infrastructure for downloads, configuration, benchmarking, and chat UX.

  ## Core Building Blocks
  - `Task` (`Android/src/app/src/main/java/com/google/ai/edge/gallery/data/Task.kt`) describes the tile shown on the home screen: id, label, category, models, docs, and chat affordances.
  - `CustomTask` (`Android/src/app/src/main/java/com/google/ai/edge/gallery/customtasks/common/CustomTask.kt`) is the entry point you implement. It supplies task metadata, model init/cleanup hooks, and the
  Compose screen for the task body.
  - `ModelManagerViewModel` keeps track of the model allowlist, downloads, and currently selected model. Every agent screen receives it via `CustomTaskDataForBuiltinTask`.
  - `ChatView` (`Android/src/app/src/main/java/com/google/ai/edge/gallery/ui/common/chat`) renders the shared chat UI (messages, input bar, upload buttons, benchmarking widgets) that agents can customize via
  callbacks.
  - `LlmChatModelHelper` centralizes MediaPipe LLM session creation, inference, and reset logic used by all multiturn agents.

  ## How Agents Are Discovered
  1. Implement your `CustomTask`.
  2. Bind it in a Hilt module annotated with `@InstallIn(SingletonComponent::class)` and `@IntoSet`. (See `LlmRagTaskModule` and `VideoAnalysisRagTaskModule` for patterns.)
  3. Provide task metadata with references to applicable models. Models are sourced from `model_allowlist.json` (assets) and filtered by `taskTypes`.

  ## Agent Lifecycle
  1. Home screen loads tasks from Hilt’s `Set<CustomTask>`, groups them by category, and displays them.
  2. When the user enters a task, `initializeModelFn` is executed on a background dispatcher after the model is downloaded/selected.
  3. The task’s Compose `MainScreen` renders inside the shared scaffold (app bar, selectors, settings) supplied by the gallery.
  4. User interactions trigger `ChatView` callbacks that delegate to the task’s ViewModel/helpers.
  5. Leaving the screen (or switching models) calls `cleanUpModelFn` to release runtime resources.

  ## Adding a New Agent
  1. Define the task metadata (id, label, icon, category, description, docs/source links, and the list of `Model`s you want to expose).
  2. Hook up model initialization via `LlmChatModelHelper` or your own helper; keep long-running work off the UI thread.
  3. Reuse `ChatView` when possible for consistency. Provide lambdas (send, reset, upload, clear) that talk to your ViewModel.
  4. Expose helper APIs for other tasks (e.g., `LlmRagModelHelper`) when shared functionality is useful.
  5. Document the workflow in `README_taskname.md` alongside the agent code for future contributors.

  ## RAG-Specific Notes
  - `LlmRagModelHelper` wraps the base LLM session with Retrieval & Inference Chain support and a Gecko embedder (`Android/src/app/src/main/java/com/google/ai/edge/gallery/ui/llmrag/LlmRagModelHelper.kt`).
  - When Gecko is unavailable, it falls back to an in-memory keyword store so the UI keeps working.
  - `LlmRagViewModel` extends the chat ViewModel to support chunking, memorization, retrieval stats, and document browsing.
  - `LlmRagScreen` wires the chat surface with upload buttons, sample documents, and a document browser. Customize the `ChatInputType.RAG` callbacks to add new ingestion sources.
  - Keep the Gecko model and tokenizer in the app’s external files directory (or bundle them with the allowlist) so initialization succeeds.

  ## Video RAG Pipeline
  - `VideoAnalysisRagIntegration` coordinates VLM analysis, RAG memorization, and querying for long context video reasoning.
  - `VideoAnalysisWithRagScreen` offers a Quick Start panel to capture batches of frames, run analysis, and store the results for later queries.
  - `VideoAnalysisMemoryManager` mirrors the multiturn chat memory reset logic so each batch starts with a clean model context.

  ## Observability & Debugging
  - All helpers log to logcat with `TAG` constants (search for `AGLlmRag*` and `VideoRAGAnalysis`).
  - `LlmRagViewModel` records inference timing stats (`time_to_first_token`, `retrieval_time`, `decode_speed`, `latency`) that surface in the chat UI.
  - Persistent issues (e.g., missing Gecko files) bubble up through initialization callbacks so the UI can display meaningful errors.

  ## Extending Further
  - Wrap additional retrieval strategies (hybrid BM25 + embeddings, multimodal chunks) in `LlmRagModelHelper`.
  - Provide dual-model selection UIs for agents that need both a VLM and an LLM.
  - Add instrumentation hooks before/after memorization to report store sizes or failures.
  - Keep per-agent READMEs updated so contributors know which assets/models are required.