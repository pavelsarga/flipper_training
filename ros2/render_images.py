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
            points = np.load('/tmp/pointcloud_debug.npy')
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
except KeyboardInterrupt:
    pass
