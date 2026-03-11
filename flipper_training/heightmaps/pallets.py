import math
from dataclasses import dataclass
import torch
from flipper_training.heightmaps import BaseHeightmapGenerator

@dataclass
class FixedPalletHeightmapGenerator(BaseHeightmapGenerator):
    """
    Generates a single fixed pallet at a specific (x, y) coordinate and yaw angle.
    """
    x_offset: float = 0.0     
    y_offset: float = 0.0     
    yaw: float = 0.0          
    length: float = 1.2       
    width: float = 0.8        
    height: float = 0.144     

    def _generate_heightmap(self, x, y, max_coord, rng=None):
        dx = x - self.x_offset
        dy = y - self.y_offset

        cos_t = math.cos(self.yaw)
        sin_t = math.sin(self.yaw)
        
        local_x = dx * cos_t + dy * sin_t
        local_y = -dx * sin_t + dy * cos_t

        pallet_mask = (local_x.abs() <= self.length / 2) & (local_y.abs() <= self.width / 2)

        z = torch.zeros_like(x)
        z[pallet_mask] = self.height
        if not hasattr(self, '_debug_plotted'):
            import matplotlib.pyplot as plt
            import os
            
            # x, y, and z are generated in batches (e.g., shape [B, 64, 64]).
            # We just take the first one in the batch [0] and convert to numpy.
            z_np = z[0].cpu().numpy()
            
            plt.figure(figsize=(8, 8))
            # origin='lower' aligns the image nicely with standard Cartesian coordinates
            plt.imshow(z_np, cmap='terrain', origin='lower')
            plt.colorbar(label='Height (m)')
            plt.title('Training Heightmap Debug')
            plt.xlabel('Grid Y axis')
            plt.ylabel('Grid X axis')
            
            save_path = '/tmp/training_heightmap_debug.png'
            plt.savefig(save_path, dpi=100)
            plt.close()
            print(f"Saved debug training heightmap to {save_path}")
            
            # Set flag so we only plot once per evaluation
            self._debug_plotted = True
        
        return z, {"suitable_mask": ~pallet_mask}

        
@dataclass
class StagingAreaHeightmapGenerator(BaseHeightmapGenerator):
    """
    Generates a specific sequence of four pallets to form a straight-line obstacle course.
    """
    length: float = 1.2
    width: float = 0.8
    base_height: float = 0.144

    def _generate_heightmap(self, x, y, max_coord, rng=None):
        z = torch.zeros_like(x)
        suitable_mask = torch.ones_like(x, dtype=torch.bool)
        
        # The four pallets forming a line along y = 3.0
        # Format: (x, y, yaw, number_of_stacks)
        #pallets = [
        #    (-3.0, 0.0, 0.78, 1),  # Pallet 6 (slanted)
        #    (0.0, 0.0, 0.0, 1),    # Pallet 5 (flat)
        #    (3.0, 0.0, 0.0, 2),    # Pallet 3 & 4 (stacked)
        #    (4.0, 0.0, 0.0, 2),    # Pallet 1 & 2 (stacked)
        #]
        """
            max_coord: 6.0
            heightmap_gen: ${cls:flipper_training.heightmaps.pallets.StagingAreaHeightmapGenerator}
            heightmap_gen_opts:
              length: 1.2
              width: 0.8
              base_height: 0.144
            objective: ${cls:flipper_training.rl_objectives.fixed_goal.FixedStartGoalNavigation}
            objective_opts:
              start_x_y_z: ${tensor:[-5.0, 0.0, 0.2]}  # Start 2 meters behind the first slanted pallet
              goal_x_y_z: ${tensor:[5.5, 0.0, 0.2]}    # Goal safely past the final stacked pallets
              iteration_limit: 2000                    
              max_feasible_pitch: 1.5
              max_feasible_roll: 1.5
              goal_reached_threshold: 0.35
              init_joint_angles: ${tensor:[0.0,0.0,0.0,0.0]}
              resample_random_joint_angles_on_reset: false"""

        pallets = [
            (-3.5, 0.0, 1.57, 1),   # Pallet 11 (Straight)
            (-0.5, 0.0, 0.78, 1),  # Pallet 12 (Slanted)
            (2.6, -0.6, 0.0, 1),    # Pallet 13 (Side-by-side, offset back)
            (3.4, 0.4, 0.0, 1),    # Pallet 14 (Side-by-side, offset forward)
        ]

        for px, py, pyaw, stacks in pallets:
            dx = x - px
            dy = y - py

            cos_t = math.cos(pyaw)
            sin_t = math.sin(pyaw)
            
            local_x = dx * cos_t + dy * sin_t
            local_y = -dx * sin_t + dy * cos_t

            pallet_mask = (local_x.abs() <= self.length / 2) & (local_y.abs() <= self.width / 2)

            current_height = self.base_height * stacks
            z = torch.where(pallet_mask & (z < current_height), torch.full_like(z, current_height), z)
            
            suitable_mask[pallet_mask] = False


        return z, {"suitable_mask": suitable_mask}
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torch

    # Parameters from average_ensembled_policy.yaml
    grid_res = 0.05
    max_coord = 3.2
    num_robots = 1  # We only need 1 for a quick visual test

    print("Initializing generator...")
    # Create the generator using the single pallet config opts
    generator = FixedPalletHeightmapGenerator(
        x_offset=0.0,
        y_offset=0.0,
        yaw=1.57,      # 90 degrees (as set in your single pallet config)
        length=1.2,
        width=0.8,
        height=0.144
    )

    print("Generating heightmap...")
    # The __call__ method returns x, y, z, and an extras dict (mask)
    x, y, z, extras = generator(
        grid_res=grid_res,
        max_coord=max_coord,
        num_robots=num_robots,
        rng=None
    )

    # Extract the first (and only) heightmap in the batch and move to CPU/numpy
    z_np = z[0].cpu().numpy()

    print("Plotting...")
    plt.figure(figsize=(8, 8))
    
    # Use 'extent' to map the array indices back to real-world meters on the axes
    plt.imshow(
        z_np, 
        cmap='terrain', 
        origin='lower', 
        extent=[-max_coord, max_coord, -max_coord, max_coord]
    )
    
    plt.colorbar(label='Height (m)')
    plt.title(f'Single Pallet (yaw={generator.yaw} rad)')
    plt.xlabel('Grid Y axis (meters)')
    plt.ylabel('Grid X axis (meters)')
    
    # Save the output
    save_path = '/tmp/standalone_pallet_test.png'
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"Done! Saved heightmap plot to {save_path}")