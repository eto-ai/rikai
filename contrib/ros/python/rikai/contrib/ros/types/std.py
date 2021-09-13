#  Copyright 2021 Rikai Authors
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Standard ROS Messages

http://docs.ros.org/en/noetic/api/std_msgs/html/index-msg.html
"""

from __future__ import annotations

import rospy

from rikai.contrib.ros.spark.types.ros.std import HeaderType, TimeType

__all__ = ["Time", "Header"]


class Time:
    """ROS Time"""

    __UDT__ = TimeType()

    def __init__(self, secs: int, nsecs: int):
        """Construct a time object from secs and nsecs."""
        self.seconds = secs
        self.nanoseconds = nsecs

    @staticmethod
    def from_ros(ros_time: rospy.Time) -> Time:
        """Convert from a rospy.Time."""
        return Time(ros_time.secs, ros_time.nsecs)

    def to_ros(self) -> rospy.Time:
        """Convert to rospy.Time

        Returns
        -------
        rospy.Time
            Convert this class to rospy.Time
        """
        return rospy.Time(self.seconds, self.nanoseconds)


class Header:
    """ROS Message Header

    http://docs.ros.org/en/noetic/api/std_msgs/html/msg/Header.html

    """

    __UDT__ = HeaderType()

    def __init__(self, seq: int, stamp: Time, frame_id: str):
        """ROS Standard Message - Header"""
        self.seq = seq
        self.stamp = stamp
        self.frame_id = frame_id
