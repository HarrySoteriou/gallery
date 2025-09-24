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

package com.google.ai.edge.gallery.ui.common.chat

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.util.Log
import android.view.Surface
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.wrapContentHeight
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material.icons.filled.Videocam
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.IconButtonDefaults
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.Button
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.compose.ui.window.Dialog
import androidx.core.content.ContextCompat
import androidx.exifinterface.media.ExifInterface
import java.io.File
import java.io.IOException
import kotlin.coroutines.resume
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.suspendCancellableCoroutine

private const val TAG = "VideoFrameCaptureButton"
private const val MAX_FRAMES_DEFAULT = 5
private const val CAPTURE_INTERVAL_MS = 1000L

@Composable
fun VideoFrameCaptureButton(
  onFramesCaptured: (List<Bitmap>) -> Unit,
  enabled: Boolean = true,
  modifier: Modifier = Modifier,
) {
  val context = LocalContext.current
  var showCaptureDialog by remember { mutableStateOf(false) }

  val cameraPermissionLauncher = rememberLauncherForActivityResult(
    ActivityResultContracts.RequestPermission()
  ) { granted ->
    if (granted) {
      showCaptureDialog = true
    }
  }

  IconButton(
    onClick = {
      when (PackageManager.PERMISSION_GRANTED) {
        ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) -> {
          showCaptureDialog = true
        }
        else -> {
          cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
      }
    },
    enabled = enabled,
    colors = IconButtonDefaults.iconButtonColors(
      containerColor = MaterialTheme.colorScheme.primary
    ),
    modifier = modifier
  ) {
    Icon(
      imageVector = Icons.Default.Videocam,
      contentDescription = "Capture video frames",
      tint = MaterialTheme.colorScheme.onPrimary
    )
  }

  if (showCaptureDialog) {
    VideoFrameCaptureDialog(
      onDismiss = {
        showCaptureDialog = false
      },
      onFramesCaptured = { frames ->
        onFramesCaptured(frames)
        showCaptureDialog = false
      },
      maxFrames = MAX_FRAMES_DEFAULT
    )
  }
}

@Composable
private fun VideoFrameCaptureDialog(
  onDismiss: () -> Unit,
  onFramesCaptured: (List<Bitmap>) -> Unit,
  maxFrames: Int,
) {
  val context = LocalContext.current
  val lifecycleOwner = LocalLifecycleOwner.current
  val coroutineScope = rememberCoroutineScope()

  var cameraProvider by remember { mutableStateOf<ProcessCameraProvider?>(null) }
  var imageCapture by remember { mutableStateOf<ImageCapture?>(null) }
  var previewView by remember { mutableStateOf<PreviewView?>(null) }
  var isCapturing by remember { mutableStateOf(false) }
  var captureCount by remember { mutableStateOf(0) }
  var capturedFrames by remember { mutableStateOf<List<Bitmap>>(emptyList()) }
  var captureJob by remember { mutableStateOf<Job?>(null) }

  LaunchedEffect(previewView) {
    val view = previewView ?: return@LaunchedEffect
    try {
      val provider = getCameraProvider(context)
      val preview = Preview.Builder().build().apply {
        setSurfaceProvider(view.surfaceProvider)
      }
      val rotation = view.display?.rotation ?: Surface.ROTATION_0
      val capture = ImageCapture.Builder()
        .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
        .setTargetRotation(rotation)
        .build()

      provider.unbindAll()
      provider.bindToLifecycle(
        lifecycleOwner,
        CameraSelector.DEFAULT_BACK_CAMERA,
        preview,
        capture,
      )

      cameraProvider = provider
      imageCapture = capture
    } catch (t: Throwable) {
      Log.e(TAG, "Failed to bind CameraX preview", t)
      cameraProvider?.unbindAll()
      cameraProvider = null
      imageCapture = null
    }
  }

  fun stopCapture() {
    isCapturing = false
    captureJob?.cancel()
    captureJob = null
  }

  DisposableEffect(Unit) {
    onDispose {
      stopCapture()
      cameraProvider?.unbindAll()
    }
  }

  fun startCapture() {
    val capture = imageCapture
    val view = previewView
    if (capture == null || view == null) {
      Log.w(TAG, "Camera not ready for capture")
      return
    }
    if (isCapturing) return

    isCapturing = true
    captureCount = 0
    capturedFrames = emptyList()
    capture.targetRotation = view.display?.rotation ?: Surface.ROTATION_0

    captureJob = coroutineScope.launch {
      var framesToEmit: List<Bitmap>? = null
      try {
        while (isActive && isCapturing && captureCount < maxFrames) {
          val bitmap = captureBitmap(capture, context)
          if (bitmap != null) {
            capturedFrames = capturedFrames + bitmap
            captureCount++
          }

          if (captureCount >= maxFrames) {
            framesToEmit = capturedFrames
            isCapturing = false
            break
          }

          delay(CAPTURE_INTERVAL_MS)
        }
      } catch (e: Exception) {
        Log.e(TAG, "Error capturing frame", e)
      } finally {
        captureJob = null
        if (!framesToEmit.isNullOrEmpty()) {
          onFramesCaptured(framesToEmit!!)
        }
      }
    }
  }

  Dialog(
    onDismissRequest = {
      stopCapture()
      onDismiss()
    }
  ) {
    Card(
      modifier = Modifier
        .fillMaxWidth()
        .wrapContentHeight()
    ) {
      Column(
        modifier = Modifier.padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
      ) {
        Text(
          text = "Video Frame Capture",
          style = MaterialTheme.typography.titleLarge,
          modifier = Modifier.padding(bottom = 16.dp)
        )

        Box(
          modifier = Modifier
            .fillMaxWidth()
            .height(200.dp)
        ) {
          AndroidView(
            factory = { ctx ->
              PreviewView(ctx).also { preview ->
                previewView = preview
              }
            },
            modifier = Modifier.fillMaxSize(),
            onRelease = {
              previewView = null
            }
          )

          Card(
            modifier = Modifier
              .align(Alignment.BottomCenter)
              .padding(8.dp),
            colors = CardDefaults.cardColors(
              containerColor = MaterialTheme.colorScheme.surface.copy(alpha = 0.9f)
            )
          ) {
            Text(
              text = if (isCapturing) {
                "Capturing: $captureCount/$maxFrames"
              } else {
                "Ready to capture $maxFrames frames at 1 FPS"
              },
              modifier = Modifier.padding(8.dp),
              style = MaterialTheme.typography.bodyMedium
            )
          }
        }

        if (isCapturing) {
          LinearProgressIndicator(
            progress = { captureCount.toFloat() / maxFrames.toFloat() },
            modifier = Modifier
              .fillMaxWidth()
              .padding(vertical = 8.dp)
          )
        }

        Spacer(modifier = Modifier.height(16.dp))

        Row(
          horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
          OutlinedButton(
            onClick = {
              stopCapture()
              onDismiss()
            }
          ) {
            Text("Cancel")
          }

          Button(
            onClick = {
              if (isCapturing) {
                stopCapture()
              } else {
                startCapture()
              }
            },
            enabled = imageCapture != null
          ) {
            Icon(
              imageVector = if (isCapturing) Icons.Default.Stop else Icons.Default.Videocam,
              contentDescription = null,
              modifier = Modifier.size(18.dp)
            )
            Spacer(modifier = Modifier.size(4.dp))
            Text(if (isCapturing) "Stop" else "Start")
          }
        }
      }
    }
  }
}

private suspend fun getCameraProvider(context: Context): ProcessCameraProvider =
  suspendCancellableCoroutine { continuation ->
    val future = ProcessCameraProvider.getInstance(context)
    future.addListener(
      {
        try {
          continuation.resume(future.get())
        } catch (t: Throwable) {
          if (continuation.isActive) {
            continuation.cancel(t)
          }
        }
      },
      ContextCompat.getMainExecutor(context)
    )
    continuation.invokeOnCancellation { future.cancel(false) }
  }

private suspend fun captureBitmap(
  imageCapture: ImageCapture,
  context: Context,
): Bitmap? = suspendCancellableCoroutine { continuation ->
  val tempFile = try {
    File.createTempFile("frame_", ".jpg", context.cacheDir)
  } catch (e: IOException) {
    Log.e(TAG, "Failed to create temp file for capture", e)
    continuation.resume(null)
    return@suspendCancellableCoroutine
  }

  val outputOptions = ImageCapture.OutputFileOptions.Builder(tempFile).build()
  imageCapture.takePicture(
    outputOptions,
    ContextCompat.getMainExecutor(context),
    object : ImageCapture.OnImageSavedCallback {
      override fun onError(exception: ImageCaptureException) {
        Log.e(TAG, "Image capture failed", exception)
        tempFile.delete()
        if (continuation.isActive) {
          continuation.resume(null)
        }
      }

      override fun onImageSaved(outputFileResults: ImageCapture.OutputFileResults) {
        val bitmap = decodeBitmapWithExif(tempFile)
        tempFile.delete()
        if (continuation.isActive) {
          continuation.resume(bitmap)
        }
      }
    }
  )

  continuation.invokeOnCancellation {
    tempFile.delete()
  }
}

private fun decodeBitmapWithExif(file: File): Bitmap? {
  val bitmap = BitmapFactory.decodeFile(file.absolutePath) ?: return null
  return try {
    val exif = ExifInterface(file)
    val orientation = exif.getAttributeInt(
      ExifInterface.TAG_ORIENTATION,
      ExifInterface.ORIENTATION_NORMAL,
    )
    val matrix = Matrix()
    when (orientation) {
      ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
      ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
      ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)
      ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.preScale(-1f, 1f)
      ExifInterface.ORIENTATION_FLIP_VERTICAL -> matrix.preScale(1f, -1f)
      ExifInterface.ORIENTATION_TRANSPOSE -> {
        matrix.preScale(-1f, 1f)
        matrix.postRotate(270f)
      }
      ExifInterface.ORIENTATION_TRANSVERSE -> {
        matrix.preScale(-1f, 1f)
        matrix.postRotate(90f)
      }
    }

    if (matrix.isIdentity) {
      bitmap
    } else {
      Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }
  } catch (e: IOException) {
    Log.w(TAG, "Failed to read EXIF metadata", e)
    bitmap
  }
}
