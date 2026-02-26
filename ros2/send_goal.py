#!/usr/bin/env python3
"""
Send a goal to the flipper policy node.

Takes x, y coordinates in world frame and publishes the goal
in base_link frame (relative to robot's current position).

Usage:
    python3 send_goal.py <x> <y>
    python3 send_goal.py 10 0      # Go to world position (10, 0)
    python3 send_goal.py -5 3      # Go to world position (-5, 3)
"""

import sys
import time
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped


class GoalSender(Node):
    def __init__(self, goal_x: float, goal_y: float):
        super().__init__("goal_sender")

        self.goal_world = np.array([goal_x, goal_y, 0.0])

        # Publisher for goal
        self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)

    def send_goal(self):
        # Create and publish goal message in WORLD frame
        goal_msg = PoseStamped()
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.header.frame_id = "world"
        goal_msg.pose.position.x = float(self.goal_world[0])
        goal_msg.pose.position.y = float(self.goal_world[1])
        goal_msg.pose.position.z = float(self.goal_world[2])
        goal_msg.pose.orientation.w = 1.0

        self.goal_pub.publish(goal_msg)

        self.get_logger().info(
            f"Goal sent: world=({self.goal_world[0]:.2f}, {self.goal_world[1]:.2f})"
        )


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 send_goal.py <x> <y>")
        print("  x, y: goal position in world frame")
        sys.exit(1)

    goal_x = float(sys.argv[1])
    goal_y = float(sys.argv[2])

    rclpy.init()
    node = GoalSender(goal_x, goal_y)

    # Give time for publisher to be discovered, then send goal
    time.sleep(0.5)
    node.send_goal()
    time.sleep(0.5)  # Give time for message to be sent

    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
