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
import com.google.common.collect.ImmutableList
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.guava.await
import java.util.UUID

private const val TAG = "AGRagKnowledgeBase"

/**
 * Centralized knowledge base for Retrieval-Augmented Generation tasks.
 *
 * This object owns both the persisted semantic memory (when available) and the
 * in-memory fallback store so that every RAG-capable task shares the same
 * retrieval surface.
 */
object RagKnowledgeBase {

  data class DocumentMetadata(
    val id: String,
    val title: String,
    val timestamp: Long,
    val chunkCount: Int,
    val source: String,
  )

  data class StoredDocument(
    val metadata: DocumentMetadata,
    val chunks: List<String>,
  )

  data class RetrievalResult(
    val chunks: List<String>,
    val sourceDocuments: List<String>,
  )

  // In-memory fallbacks guarantee retrieval keeps working even when the native
  // semantic store or embedder cannot be initialized.
  private val documentStore = mutableMapOf<String, List<String>>()
  private val documentMetadata = mutableMapOf<String, DocumentMetadata>()

  @Volatile
  private var lastRetrievalResult: RetrievalResult? = null

  fun getLastRetrievalResult(): RetrievalResult? = lastRetrievalResult

  fun getDocumentCount(): Int = documentStore.values.sumOf { it.size }

  fun getDocumentMetadataList(): List<DocumentMetadata> {
    return documentMetadata.values.sortedByDescending { it.timestamp }
  }

  fun getStoredDocuments(): List<StoredDocument> {
    return documentStore.mapNotNull { (id, chunks) ->
      documentMetadata[id]?.let { metadata -> StoredDocument(metadata, chunks) }
    }
  }

  fun getDocumentById(documentId: String): StoredDocument? {
    val chunks = documentStore[documentId] ?: return null
    val metadata = documentMetadata[documentId] ?: return null
    return StoredDocument(metadata, chunks)
  }

  fun deleteDocument(documentId: String): Boolean {
    val removed = documentStore.remove(documentId) != null
    documentMetadata.remove(documentId)
    if (removed) {
      Log.d(TAG, "Deleted document: $documentId")
    }
    return removed
  }

  fun searchDocuments(query: String): List<StoredDocument> {
    val normalized = query.lowercase()
    return getStoredDocuments().filter { doc ->
      doc.metadata.title.lowercase().contains(normalized) ||
        doc.chunks.any { chunk -> chunk.lowercase().contains(normalized) }
    }
  }

  fun clearDocuments() {
    documentStore.clear()
    documentMetadata.clear()
    lastRetrievalResult = null
    Log.d(TAG, "Cleared knowledge base documents")
  }

  suspend fun memorizeChunks(
    model: Model,
    chunks: List<String>,
    title: String = "Document",
    source: String = "upload",
  ): String = coroutineScope {
    try {
      Log.d(TAG, "Memorizing ${chunks.size} chunks for '$title'")

      val ragInstance = try {
        model.instance as? RagModelInstance
      } catch (e: ClassCastException) {
        Log.w(TAG, "Model instance is not RagModelInstance, using fallback store: ${e.message}")
        null
      }

      if (ragInstance?.semanticMemory != null) {
        try {
          ragInstance.semanticMemory.recordBatchedMemoryItems(
            ImmutableList.copyOf(chunks),
          ).await()
          storeDocumentLocally(chunks, title, source)
          Log.d(TAG, "Stored ${chunks.size} chunks in semantic memory and fallback store")
        } catch (e: Exception) {
          Log.w(TAG, "Semantic memory failed, falling back to in-memory store: ${e.message}")
          storeDocumentLocally(chunks, title, source)
        }
      } else {
        storeDocumentLocally(chunks, title, source)
      }

      ""
    } catch (e: Exception) {
      val error = "Failed to memorize chunks: ${e.message}"
      Log.e(TAG, error)
      error
    }
  }

  fun retrieveRelevantChunks(query: String, maxChunks: Int = 3): RetrievalResult {
    if (documentStore.isEmpty()) {
      lastRetrievalResult = RetrievalResult(emptyList(), emptyList())
      return lastRetrievalResult!!
    }

    val normalizedQuery = query.lowercase()
    val queryWords = normalizedQuery.split("\\s+".toRegex()).filter { it.length > 2 }

    val scored = mutableListOf<Triple<String, Double, String>>()
    documentStore.forEach { (documentId, chunks) ->
      chunks.forEach { chunk ->
        val chunkLower = chunk.lowercase()
        var score = 0.0

        val keywordMatches = queryWords.count { chunkLower.contains(it) }
        score += keywordMatches.toDouble()

        if (chunkLower.contains(normalizedQuery)) {
          score += 3.0
        }

        queryWords.filter { it.length > 3 }.forEach { word ->
          if (chunkLower.contains(word)) {
            score += 1.5
          }
        }

        if (chunk.length > 200) {
          score += 0.5
        }

        if (score > 0) {
          scored.add(Triple(chunk, score, documentId))
        }
      }
    }

    val selected = scored.sortedByDescending { it.second }.take(maxChunks)
    val chunks = selected.map { it.first }
    val sourceDocIds = selected.map { it.third }.distinct()
    val sourceDocs = sourceDocIds.mapNotNull { documentMetadata[it]?.title }

    val result = RetrievalResult(chunks, sourceDocs)
    lastRetrievalResult = result
    return result
  }

  private fun storeDocumentLocally(
    chunks: List<String>,
    title: String,
    source: String,
  ) {
    val timestamp = System.currentTimeMillis()
    val documentId = UUID.randomUUID().toString()

    documentStore[documentId] = chunks
    documentMetadata[documentId] = DocumentMetadata(
      id = documentId,
      title = title,
      timestamp = timestamp,
      chunkCount = chunks.size,
      source = source,
    )
    Log.d(TAG, "Stored document locally: $title (${chunks.size} chunks)")
  }
}
