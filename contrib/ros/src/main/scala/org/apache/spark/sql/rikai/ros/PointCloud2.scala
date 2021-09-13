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

package org.apache.spark.sql.rikai.ros

import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.GenericInternalRow
import org.apache.spark.sql.rikai.{NDArray, NDArrayType}
import org.apache.spark.sql.types.{
  DataType,
  StructField,
  StructType,
  UserDefinedType
}

/** ROS Message - [[http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/PointCloud2.html PointCloud2]] */
class PointCloud2(val header: Header, val data: NDArray) {}

private[spark] class PointCloud2Type extends UserDefinedType[PointCloud2] {
  override def sqlType: DataType = StructType(
    Seq(
      StructField("header", HeaderType, false),
      StructField("data", NDArrayType, false)
    )
  )

  override def pyUDT: String = "rikai.contrib.ros.spark.types.PointCloud2Type"

  override def serialize(cloud: PointCloud2): Any = {
    val row = new GenericInternalRow(2)
    row.update(0, cloud.header)
    row.update(1, cloud.data)
    row
  }

  override def deserialize(datum: Any): PointCloud2 = {
    datum match {
      case row: InternalRow => {
        val headerField = row.getStruct(0, 3)
        val header = HeaderType.deserialize(headerField)
        val dataField = row.getStruct(1, 3)
        val data = NDArrayType.deserialize(dataField)
        new PointCloud2(header, data)
      }
    }
  }

  override def userClass: Class[PointCloud2] = classOf[PointCloud2]
}

case object PointCloud2Type extends PointCloud2Type
