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

import org.apache.spark.sql.types._

/** ROS Message - [[http://docs.ros.org/en/noetic/api/std_msgs/html/msg/Header.html Header]]
  *
  * @param seq sequence ID
  * @param stamp ROS Time
  * @param frameId Frame this data is associated with.
  */
class Header(val seq: Int, val stamp: Time, val frameId: String) {

  override def toString: String = s"Header(seq=$seq, frameId=$frameId)"
}

/** User defined type for Ros Header message */
private[spark] class HeaderType extends UserDefinedType[Header] {

  override def sqlType: DataType = StructType(
    Seq(
      StructField("seq", IntegerType, false),
      StructField("stamp", TimeType, false),
      StructField("frameId", StringType, false)
    )
  )

  override def pyUDT: String = "rikai.spark.types.ros.HeaderType"

  override def serialize(obj: Header): Any = ???

  override def deserialize(datum: Any): Header = ???

  override def userClass: Class[Header] = classOf[Header]
}

case object HeaderType extends HeaderType
