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
  * @constructor Create a 3-D bounding box.
  * @param center Center point of the box.
  * @param width the width of the 3D box.
  * @param height the height of the 3D box.
  * @param length the length of the 3D box.
  * @param orientation the orientation of the 3D bounding box.
  *
  * @see [[https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/label.proto  Waymo Dataset Spec]]
  */
class Box3d(
    val center: Point,
    val length: Double,
    val width: Double,
    val height: Double,
    val orientation: Orientation
) {
  override def toString: String =
    f"Box3d(center=$center, l=$length, w=$width, h=$height, orientation=$orientation)"
}
