#!/usr/bin/env python3
"""
Subscribe to /policy_heightmap_debug and plot it with matplotlib.
"""
import struct
import threading

import matplotlib.pyplot as plt
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2


class HeightmapDebugPlotter(Node):
    def __init__(self):
        super().__init__("heightmap_debug_plotter")
        self._lock = threading.Lock()
        self._latest_points = None
        self._new_data = False

        self.create_subscription(PointCloud2, "/policy_heightmap_debug", self._callback, 10)
        self.get_logger().info("Subscribed to /policy_heightmap_debug")

    def _callback(self, msg: PointCloud2):
        n = msg.width * msg.height
        raw = msg.data
        points = np.frombuffer(raw, dtype=np.float32).reshape(n, 4)
        with self._lock:
            self._latest_points = points.copy()
            self._new_data = True

    def get_latest(self):
        with self._lock:
            if not self._new_data:
                return None
            self._new_data = False
            return self._latest_points


def unpack_rgb(rgb_float: np.ndarray):
    packed = np.frombuffer(rgb_float.astype(np.float32).tobytes(), dtype=np.uint32)
    r = ((packed >> 16) & 0xFF) / 255.0
    g = ((packed >> 8) & 0xFF) / 255.0
    b = (packed & 0xFF) / 255.0
    return np.stack([r, g, b], axis=-1)


def main():
    rclpy.init()
    node = HeightmapDebugPlotter()

    plt.ion()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Policy Heightmap Debug")
    ax_2d, ax_3d = axes[0], None

    spinner = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spinner.start()

    print("Waiting for /policy_heightmap_debug messages... (Ctrl+C to exit)")

    try:
        while rclpy.ok():
            points = node.get_latest()
            if points is None:
                plt.pause(0.05)
                continue

            x, y, z, rgb_f = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
            colors = unpack_rgb(rgb_f)

            # Reconstruct 2D grid by sorting into unique x/y bins
            xs = np.unique(np.round(x, 5))
            ys = np.unique(np.round(y, 5))
            rows, cols = len(xs), len(ys)

            if rows > 1 and cols > 1:
                xi = np.searchsorted(xs, np.round(x, 5))
                yi = np.searchsorted(ys, np.round(y, 5))
                grid_z = np.full((rows, cols), np.nan)
                grid_z[xi, yi] = z

                fig.clear()
                ax_2d = fig.add_subplot(1, 2, 1)
                ax_3d = fig.add_subplot(1, 2, 2, projection="3d")

                # 2D heatmap (heightmap view, front of robot at top)
                im = ax_2d.imshow(
                    grid_z,
                    origin="upper",
                    extent=[ys.min(), ys.max(), xs.min(), xs.max()],
                    cmap="viridis",
                    aspect="equal",
                )
                fig.colorbar(im, ax=ax_2d, label="Height (m)")
                ax_2d.set_title("Heightmap (top-down)")
                ax_2d.set_xlabel("Y (left-right)")
                ax_2d.set_ylabel("X (front-back)")

                # 3D scatter
                skip = max(1, len(x) // 2000)
                ax_3d.scatter(x[::skip], y[::skip], z[::skip], c=colors[::skip], s=2, marker=".")
                ax_3d.set_title("3D view")
                ax_3d.set_xlabel("X")
                ax_3d.set_ylabel("Y")
                ax_3d.set_zlabel("Z")

                fig.suptitle("Policy Heightmap Debug")
            else:
                # Fallback: plain scatter if grid reconstruction fails
                fig.clear()
                ax = fig.add_subplot(111, projection="3d")
                ax.scatter(x, y, z, c=colors, s=2, marker=".")
                ax.set_title("Policy Heightmap Debug (scatter)")

            fig.canvas.draw()
            plt.pause(0.05)

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
