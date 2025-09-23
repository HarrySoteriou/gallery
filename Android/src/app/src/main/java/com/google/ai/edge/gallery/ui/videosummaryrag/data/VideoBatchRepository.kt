package com.google.ai.edge.gallery.ui.videosummaryrag.data

import android.content.Context
import dagger.hilt.android.qualifiers.ApplicationContext
import javax.inject.Inject
import javax.inject.Singleton
import kotlin.coroutines.cancellation.CancellationException
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.io.File
import java.io.IOException
import java.util.UUID

/**
 * Repository responsible for persisting video batch summaries produced by the
 * VideoSummary+RAG integration. Records are stored as JSON in the app's
 * private files directory so they survive process restarts and remain
 * accessible offline.
 */
interface VideoBatchRepository {
  suspend fun saveBatch(
    frameCount: Int,
    summary: String,
    ragDocumentId: String?,
    ragModelName: String,
    visionModelName: String
  ): VideoBatchRecord

  suspend fun listBatches(): List<VideoBatchRecord>

  suspend fun getBatch(id: String): VideoBatchRecord?

  suspend fun deleteBatch(id: String): Boolean

  suspend fun clear(): Boolean
}

@Serializable
data class VideoBatchRecord(
  val id: String,
  val timestamp: Long,
  val frameCount: Int,
  val summary: String,
  val ragDocumentId: String?,
  val ragModelName: String,
  val visionModelName: String
)

@Singleton
class JsonVideoBatchRepository @Inject constructor(
  @ApplicationContext private val context: Context,
) : VideoBatchRepository {

  private val mutex = Mutex()
  private val ioDispatcher: CoroutineDispatcher = Dispatchers.IO
  private val json = Json {
    ignoreUnknownKeys = true
    encodeDefaults = true
    prettyPrint = false
  }
  private val storageFile: File by lazy {
    File(context.filesDir, STORAGE_FILE_NAME)
  }

  override suspend fun saveBatch(
    frameCount: Int,
    summary: String,
    ragDocumentId: String?,
    ragModelName: String,
    visionModelName: String
  ): VideoBatchRecord {
    val record = VideoBatchRecord(
      id = UUID.randomUUID().toString(),
      timestamp = System.currentTimeMillis(),
      frameCount = frameCount,
      summary = summary,
      ragDocumentId = ragDocumentId,
      ragModelName = ragModelName,
      visionModelName = visionModelName,
    )
    mutex.withLock {
      val updated = readRecordsInternal().toMutableList().apply { add(0, record) }
      writeRecordsInternal(updated)
    }
    return record
  }

  override suspend fun listBatches(): List<VideoBatchRecord> {
    return mutex.withLock { readRecordsInternal() }
  }

  override suspend fun getBatch(id: String): VideoBatchRecord? {
    return mutex.withLock { readRecordsInternal().firstOrNull { it.id == id } }
  }

  override suspend fun deleteBatch(id: String): Boolean {
    var deleted = false
    mutex.withLock {
      val current = readRecordsInternal()
      val filtered = current.filterNot { it.id == id }
      if (filtered.size != current.size) {
        deleted = true
        writeRecordsInternal(filtered)
      }
    }
    return deleted
  }

  override suspend fun clear(): Boolean {
    mutex.withLock {
      if (storageFile.exists()) {
        try {
          storageFile.delete()
        } catch (ignored: SecurityException) {
          return false
        }
      }
    }
    return true
  }

  private suspend fun readRecordsInternal(): List<VideoBatchRecord> =
    withContext(ioDispatcher) {
      if (!storageFile.exists()) {
        return@withContext emptyList()
      }
      try {
        val content = storageFile.readText()
        if (content.isBlank()) {
          emptyList()
        } else {
          json.decodeFromString<List<VideoBatchRecord>>(content)
        }
      } catch (cancellation: CancellationException) {
        throw cancellation
      } catch (e: IOException) {
        emptyList()
      }
    }

  private suspend fun writeRecordsInternal(records: List<VideoBatchRecord>) =
    withContext(ioDispatcher) {
      try {
        if (!storageFile.parentFile.exists()) {
          storageFile.parentFile.mkdirs()
        }
        storageFile.writeText(json.encodeToString(records))
      } catch (cancellation: CancellationException) {
        throw cancellation
      } catch (_: IOException) {
        // Ignore write failures – caller can inspect repository history later.
      }
    }

  private companion object {
    private const val STORAGE_FILE_NAME = "video_batch_history.json"
  }
}
