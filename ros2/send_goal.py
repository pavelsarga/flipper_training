#!/usr/bin/env python3
"""
Send a goal to the flipper policy node.

Looks up the map→base_link TF to find the robot's current pose, then places
the goal <distance> metres straight ahead of the robot in map frame.

Usage:
    python3 send_goal.py [distance]
    python3 send_goal.py        # 5 m ahead (default)
    python3 send_goal.py 10     # 10 m ahead
"""

import sys
import time
import math
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import tf2_ros


class GoalSender(Node):
    def __init__(self, distance: float):
        super().__init__("goal_sender")
        self.distance = distance
        self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    def send_goal(self) -> bool:
        try:
            tf = self.tf_buffer.lookup_transform(
                "map", "base_link",
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=3.0),
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().error(f"TF lookup map→base_link failed: {e}")
            return False

        t = tf.transform
        robot_x = t.translation.x
        robot_y = t.translation.y

        # Extract yaw from quaternion
        q = t.rotation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

        goal_x = robot_x + self.distance * math.cos(yaw)
        goal_y = robot_y + self.distance * math.sin(yaw)

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.pose.position.x = goal_x
        msg.pose.position.y = goal_y
        msg.pose.position.z = 0.0
        # Face the goal the same direction the robot was facing when it was placed
        # (straight ahead), not identity -- identity made the RViz goal arrow always
        # point along map's +X axis regardless of robot heading or goal position,
        # which looked like a random/wrong rotation whenever the robot wasn't
        # already facing +X.
        msg.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.orientation.w = math.cos(yaw / 2.0)

        self.goal_pub.publish(msg)
        self.get_logger().info(
            f"Goal sent: robot=({robot_x:.2f}, {robot_y:.2f}), yaw={math.degrees(yaw):.1f}°, "
            f"goal=({goal_x:.2f}, {goal_y:.2f}) [{self.distance:.1f} m ahead]"
        )
        return True


def main():
    distance = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0

    rclpy.init()
    node = GoalSender(distance)

    # Allow TF buffer to fill before lookup
    time.sleep(0.5)
    rclpy.spin_once(node, timeout_sec=0.1)

    if node.send_goal():
        time.sleep(0.3)  # Let the message reach subscribers

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
