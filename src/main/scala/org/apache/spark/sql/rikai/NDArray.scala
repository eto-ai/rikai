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

class NDArrayType extends UserDefinedType[NDArray] {

  override def sqlType: DataType =
    StructType(
      Seq(
        StructField("type", ByteType, false),
        StructField("shape", ArrayType(IntegerType, false)),
        StructField("data", BinaryType, false)
      )
    )

  override def pyUDT: String = "rikai.spark.types.NDArrayType"

  override def serialize(obj: NDArray): Any = ???

  override def deserialize(datum: Any): NDArray = ???

  override def userClass: Class[NDArray] = classOf[NDArray]

}

@SQLUserDefinedType(udt = classOf[NDArrayType])
class NDArray {
  override def toString: String = s"ndarray()"
}
