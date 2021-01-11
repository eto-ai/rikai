/*
 * Copyright 2020 Rikai authors
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

import org.apache.spark.sql.types._
import org.apache.spark.unsafe.types.UTF8String

/**
  * Label as a Spark UserDefinedType
  */
private[spark] class LabelType extends UserDefinedType[Label] {

  override def sqlType: DataType = StringType

  override def pyUDT: String = "rikai.spark.types.vision.LabelType"

  override def serialize(obj: Label): Any = {
    UTF8String.fromString(obj.label)
  }

  override def deserialize(datum: Any): Label = {
    datum match {
      case row: UTF8String => new Label(row.toString())
    }
  }

  override def userClass: Class[Label] = classOf[Label]

  override def toString: String = "LabelType"

  override def hashCode: Int = classOf[LabelType].getName.hashCode

  private[spark] override def asNullable: LabelType = this
}

/**
  * A Label to identify one label instance
  */
@SQLUserDefinedType(udt = classOf[LabelType])
class Label(val label: String) {

  override def equals(x: Any): Boolean =
    x match {
      case other: Label => label == other.label
      case _            => false
    }

  override def toString: String = s"Label($label)"
}
