/**
 * Copyright 2025 The Google AI Edge Authors.
 *
 * <p>Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
 * except in compliance with the License. You may obtain a copy of the License at
 *
 * <p>http://www.apache.org/licenses/LICENSE-2.0
 *
 * <p>Unless required by applicable law or agreed to in writing, software distributed under the
 * License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 * express or implied. See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.google.ai.edge.localagents.rag.models;

/** Interface for processing embedding data before embedding generation. */
public interface EmbeddingDataProcessor<T> {
  /**
   * Processes the input embedding data and returns the processed version.
   *
   * @param embedData The input embedding data to process
   * @return The processed embedding data
   */
  EmbedData<T> process(EmbedData<T> embedData);
}