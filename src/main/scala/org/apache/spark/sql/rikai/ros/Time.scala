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
import org.apache.spark.sql.types._

import java.time.Instant

/** ROS message Time
  *
  * @param seconds seconds since epoch.
  * @param nanoseconds nano seconds since seconds.
  */
class Time(val seconds: Int, val nanoseconds: Int) {

  /** Convert ROS [[Time]] to ``java.time.Instant`` */
  def toInstant: Instant = Instant.ofEpochSecond(seconds, nanoseconds)

  override def toString: String = s"timestamp(secs=$seconds, nsecs=$nanoseconds"
}

private[sql] class TimeType extends UserDefinedType[Time] {

  override def sqlType: DataType = StructType(
    Seq(
      StructField("seconds", IntegerType, false),
      StructField("nanoseconds", IntegerType, false)
    )
  )

  override def pyUDT: String = "rikai.spark.types.ros.TimeType"

  override def serialize(t: Time): Any = {
    val row = new GenericInternalRow(2)
    row.setInt(0, t.seconds)
    row.setInt(1, t.nanoseconds)
    row
  }

  override def deserialize(datum: Any): Time = {
    datum match {
      case row: InternalRow => {
        new Time(row.getInt(0), row.getInt(1))
      }
    }
  }

  override def userClass: Class[Time] = classOf[Time]

}

case object TimeType extends TimeType
