#!/usr/bin/env python3
"""
Run to view debug pointcloud
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import struct
import os

print("Rendering images...")

# Plot Pointcloud    
try:
    while True:
        if os.path.exists('/tmp/pointcloud_debug.npy'):
            try:
                points = np.load('/tmp/pointcloud_debug.npy')
            except (EOFError, ValueError, OSError) as e:
                print(f"Warning: Failed to load pointcloud file: {e}")
                continue
            
            x, y, z = points[:, 0], points[:, 1], points[:, 2]
            
            colors = []
            for rgb_f in points[:, 3]:
                packed_int = struct.unpack("I", struct.pack("f", rgb_f))[0]
                r = ((packed_int >> 16) & 255) / 255.0
                g = ((packed_int >> 8) & 255) / 255.0
                b = (packed_int & 255) / 255.0
                colors.append((r, g, b))

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x, y, z, c=colors, s=2, marker='.')
            ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z))) 
            plt.savefig('/tmp/pointcloud_debug_rendered.png', dpi=100)
            print("SUCCESS: Saved /tmp/pointcloud_debug_rendered.png")
            plt.close()
        if os.path.exists('/tmp/lidar_pointcloud_debug.npy'):
            try:
                # allow_pickle=True is sometimes needed for structured numpy arrays from ROS
                points = np.load('/tmp/lidar_pointcloud_debug.npy', allow_pickle=True)
            except (EOFError, ValueError, OSError) as e:
                print(f"Warning: Failed to load lidar pointcloud file: {e}")
                continue

            # Force it into a standard 2D float array
            points = np.array(points.tolist(), dtype=float)
            x, y, z = points[:, 0], points[:, 1], points[:, 2]

            # 1. Use np.isfinite to remove NaNs AND pure Infinities
            valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            x, y, z = x[valid], y[valid], z[valid]

            # 2. FIX: Filter out the "max range" wall by ignoring points further than n meters
            r = np.sqrt(x**2 + y**2 + z**2)
            valid_range = r < 3.0
            x, y, z = x[valid_range], y[valid_range], z[valid_range]

            # 3. Downsample so Matplotlib doesn't freeze
            skip = 20
            x, y, z = x[::skip], y[::skip], z[::skip]

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Color points by their Z height using 'c=z' and a colormap
            ax.scatter(x, y, z, c=z, cmap='viridis', s=2, marker='.')
            
            # Now np.ptp() will only see the tightly cropped arena!
            ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z))) 
            
            plt.savefig('/tmp/lidar_debug_rendered.png', dpi=100)
            print("SUCCESS: Saved /tmp/lidar_debug_rendered.png")
            plt.close(fig)
except KeyboardInterrupt:
    pass
