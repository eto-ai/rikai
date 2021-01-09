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
package org.apache.spark.sql.rikai

/**
  * 3-D Bounding Box
  *
  * @param center Center point (x, y, z) of the box
  * @param orientaorientationtion the orientation of the box
  */
class Box3d(
    val center: Point,
    val orientation: Orientation
) {
  override def toString: String =
    f"Box3d(center=$center, orientation=$orientation)"
}
