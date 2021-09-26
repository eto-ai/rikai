#  Copyright (c) 2021 Rikai Authors
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


import std_msgs.msg
from std_msgs.msg import UInt32

from rikai.contrib.ros.convert import as_json


def test_primitive_types():
    assert {"data": 123} == as_json(UInt32(data=123))


def test_all_msgs_classes():
    """Import all known messages here to guarentee the algorithm does not break
    on any of the messages
    """
    # pylint: disable=unused-import
    import diagnostic_msgs.msg
    import geometry_msgs.msg
    import rosgraph_msgs.msg
    import rospy
    import sensor_msgs.msg
    import std_msgs
    import tf2_msgs.msg
    from genpy.message import Message

    for msg in Message.__subclasses__():
        if msg == rospy.msg.AnyMsg:
            continue
        print(msg)
        as_json(msg())


def test_simple_struct():
    msg = std_msgs.msg.ByteMultiArray(
        data=(1, 2, 3, 4),
        layout=std_msgs.msg.MultiArrayLayout(
            data_offset=123,
            dim=[
                std_msgs.msg.MultiArrayDimension(
                    label="a", size=1234, stride=18
                )
            ],
        ),
    )

    assert {
        "data": bytearray([1, 2, 3, 4]),
        "layout": {
            "data_offset": 123,
            "dim": [{"label": "a", "size": 1234, "stride": 18}],
        },
    } == as_json(msg)
