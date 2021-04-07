/*
 * Copyright 2021 Rikai authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package ai.eto

import org.apache.spark.sql.{DataFrame, DataFrameReader, DataFrameWriter}

package object rikai {
  implicit class RikaiReader(reader: DataFrameReader) {
    def rikai(path: String): DataFrame = {
      reader.format("rikai").load(path)
    }
  }

  implicit class RikaiWriter[T](writer: DataFrameWriter[T]) {
    def rikai(path: String): Unit = {
      writer.format("rikai").save(path)
    }
  }
}
