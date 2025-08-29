# RAG Setup Instructions for Android AI Edge Gallery

This document explains how to set up and use the Retrieval Augmented Generation (RAG) functionality in the Android AI Edge Gallery.

## Prerequisites

### 1. RAG SDK Dependencies
The RAG SDK dependency has already been added to `app/build.gradle.kts`:
```kotlin
implementation("com.google.ai.edge.localagents:localagents-rag:0.1.0")
```

### 2. Required Model Files
You need to push the following files to your Android device:

#### Gecko Embedder Model
```bash
# Download Gecko embedder model (example for 256 tokens, float precision)
# Visit: https://www.kaggle.com/models/google/gecko/frameworks/TensorFlow2/variations/text-embedding-preview-0927/versions/1

# Push the model files to device
adb push sentencepiece.model /data/local/tmp/sentencepiece.model
adb push Gecko_256_fp32.tflite /data/local/tmp/gecko.tflite
```

#### LLM Model
```bash
# Download your preferred LLM model (e.g., Gemma-3 1B)
# Visit: https://www.kaggle.com/models/google/gemma-2-2b-instruction-tune

# Push the model to device
adb shell rm -r /data/local/tmp/llm/
adb shell mkdir -p /data/local/tmp/llm/
adb push path/to/your/model /data/local/tmp/llm/model_version.task
```

## Available RAG Tasks

### 1. Basic RAG Chat
- **Task ID**: `llm_rag`
- **Description**: Chat with LLMs enhanced with RAG capabilities
- **Features**:
  - Text chunking and embedding
  - Semantic search and retrieval
  - Context-aware responses
  - Knowledge base management

### 2. Video Analysis + RAG
- **Task ID**: `video_analysis_rag`
- **Description**: Advanced video analysis with RAG memory
- **Features**:
  - Batch video frame analysis
  - Frame description storage in RAG
  - Cross-batch querying
  - Video content summarization

## Usage Workflows

### Basic RAG Usage
1. Select the "RAG Chat" task
2. Choose your LLM model
3. Add context documents using "Add Sample Context"
4. Start chatting with enhanced context awareness

### Video Analysis with RAG
1. Select the "Video Analysis + RAG" task
2. Ensure both VLM and RAG models are available
3. Capture video frames using the camera
4. Analyze and store frame descriptions
5. Query across multiple video batches
6. Get comprehensive video summaries

## Configuration

### Embedder Settings
The following constants can be modified in `LlmRagModelHelper.kt`:
- `GECKO_MODEL_PATH`: Path to Gecko embedder model
- `TOKENIZER_MODEL_PATH`: Path to tokenizer model
- `USE_GPU_FOR_EMBEDDINGS`: Enable GPU acceleration for embeddings
- `EMBEDDING_DIMENSION`: Vector dimension (768 for Gecko)

### RAG Parameters
Configure in `LlmRagModelHelper.kt`:
- `QA_PROMPT_TEMPLATE`: Template for question-answering
- Retrieval configuration (top-k, threshold, task type)

## Integration with Existing Code

### Using RAG in Custom Tasks
```kotlin
// Initialize RAG model
LlmRagModelHelper.initialize(context, model) { error ->
    if (error.isEmpty()) {
        // RAG model ready
    }
}

// Memorize content
LlmRagModelHelper.memorizeChunks(model, chunks)

// Generate response with RAG
LlmRagModelHelper.generateResponse(model, query)
```

### Video Analysis Integration
```kotlin
// Analyze video batch and store in RAG
VideoAnalysisRagIntegration.analyzeAndMemorizeBatch(
    vlmModel = visionModel,
    ragModel = ragModel,
    frames = capturedFrames,
    batchIndex = currentBatch
)

// Query video content
VideoAnalysisRagIntegration.queryVideoContent(ragModel, "What objects were detected?")
```

## Troubleshooting

### Common Issues
1. **Model not found**: Ensure model files are pushed to correct paths
2. **Embedding errors**: Check Gecko model format and tokenizer
3. **Memory issues**: Consider using quantized models for lower memory usage
4. **Performance**: Enable GPU acceleration when available

### Model Compatibility
- LLM models: Compatible with MediaPipe LLM Inference API
- Embedder: Gecko text embedding models
- Storage: SQLite vector store (persistent) or in-memory store

## Performance Considerations

### Memory Usage
- Each embedding vector uses ~3KB (768 dimensions × 4 bytes)
- Consider chunking strategy for large documents
- Clear RAG memory periodically for long sessions

### Processing Speed
- GPU acceleration recommended for embeddings
- Batch processing for multiple text chunks
- Async processing for better UI responsiveness

## Future Enhancements

### Potential Improvements
1. **Cloud Embedder**: Support for Gemini API embeddings
2. **Custom Chunking**: Advanced text segmentation strategies
3. **Multimodal RAG**: Image and audio content in RAG
4. **Streaming Responses**: Real-time response streaming
5. **Vector Search**: Advanced similarity search options
