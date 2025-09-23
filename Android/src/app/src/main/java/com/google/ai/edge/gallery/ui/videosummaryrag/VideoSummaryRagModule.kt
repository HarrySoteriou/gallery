package com.google.ai.edge.gallery.ui.videosummaryrag

import com.google.ai.edge.gallery.ui.videosummaryrag.data.JsonVideoBatchRepository
import com.google.ai.edge.gallery.ui.videosummaryrag.data.VideoBatchRepository
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent

@Module
@InstallIn(SingletonComponent::class)
object VideoSummaryRagModule {

  @Provides
  fun provideVideoBatchRepository(
    repository: JsonVideoBatchRepository,
  ): VideoBatchRepository = repository
}
