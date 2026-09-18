"""Convolutional encoder for the (1, H, W) heightmap channel of an observation."""

import torch.nn as nn

from marv_rl_training.observations import ObservationEncoder


class HeightmapEncoder(ObservationEncoder):
    def __init__(
        self,
        img_shape: tuple[int, int],
        output_dim: int,
        activate_output: bool = False,
        **kwargs,
    ):
        super(HeightmapEncoder, self).__init__(output_dim)
        self.img_shape = img_shape  # Keep for reference if needed, but not used in layer defs anymore
        # Define the sequential convolutional layers
        # Each block roughly corresponds to a downsampling stage in the original
        self.encoder = nn.Sequential(
            # Layer 1: Similar to the original stem but using 3x3 kernel
            # Input: (B, 1, H, W)
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # Output: (B, 16, H/2, W/2)
            # Layer 2: Downsample, increase channels
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # Output: (B, 32, H/4, W/4)
            # Layer 3: Downsample, increase channels
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # Output: (B, 32, H/8, W/8)
            nn.AdaptiveAvgPool2d((2, 2)),  # Pool to 2x2 spatial dimensions
            nn.Flatten(),  # Flatten features -> (B, 32 * 2 * 2)
            nn.Linear(4 * 32, output_dim),  # Linear layer -> (B, output_dim)
            nn.ReLU(inplace=True) if activate_output else nn.Identity(),
        )

    def forward(self, hm):
        # Handle potential time dimension (same as before)
        if hm.ndim > 4:
            B, T = hm.shape[:2]
            # Input shape expected: (B, T, C, H, W)
            C, H, W = hm.shape[2:]
            hm = hm.view(B * T, C, H, W)  # Use view for efficiency
            y_ter = self.encoder(hm)
            # Output shape expected: (B, T, output_dim)
            y_ter = y_ter.view(B, T, -1)
        else:
            # Input shape expected: (B, C, H, W)
            y_ter = self.encoder(hm)
            # Output shape expected: (B, output_dim)
        return y_ter
