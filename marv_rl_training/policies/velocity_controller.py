"""Forward-velocity controller for flipper-only policies: a small CNN + MLP imitating an operator.

Inputs (one control step = 0.05 s, `history` steps, newest last), the layout FtrEnv's
velocity_control.FeatureHistory builds in Isaac and misc/velocity_il/extract_velocity_il.py
builds from Gazebo bags:
    hm      [B, T, 45, 21]  robot-centric heightmap, metres relative to (base z - wheel radius),
                            row 0 = front, col 0 = robot's right
    proprio [B, T, 10]      v measured, v commanded at the previous step, flipper velocity x 4
                            (measured, rad/s), flipper angle x 4 (rad, positive = up, order fl fr rl rr)
Output: a Gaussian over the commanded forward speed, mean squashed into [0, v_max].

The T heightmaps are stacked as CNN channels (the motion of terrain through the window over T
steps is the speed signal), the proprio history goes through an MLP, and the two embeddings
meet in the head (`head_layers` hidden layers of `head_hidden`). `VelocityControllerExport`
bakes the input normalisation in and returns the mean only -- that TorchScript module is the
file FtrEnv's `velocity_controller: {type: il}` and the Gazebo velocity node load.
"""
import torch
from torch import nn

PROPRIO_DIM = 10
HM_SHAPE = (45, 21)


class VelocityNet(nn.Module):
    def __init__(self, history: int = 5, channels=(16, 32, 32), proprio_hidden: int = 64, head_hidden: int = 256,
                 head_layers: int = 2, v_max: float = 0.7, drop_proprio=()):
        super().__init__()
        self.v_max = v_max
        # proprio features zeroed after normalisation (= their training mean), e.g. index 1, the previous
        # command: on smooth operator labels copying it is a near-perfect predictor that ignores the terrain.
        # A buffer, so the TorchScript export drops them the same way wherever it is loaded.
        keep = torch.ones(PROPRIO_DIM)
        keep[list(drop_proprio)] = 0.0
        self.register_buffer("proprio_keep", keep)
        layers, c_in = [], history
        for c in channels:
            layers += [nn.Conv2d(c_in, c, 3, stride=2, padding=1), nn.GroupNorm(4, c), nn.SiLU()]
            c_in = c
        self.cnn = nn.Sequential(*layers, nn.AdaptiveAvgPool2d((3, 2)), nn.Flatten())
        self.proprio = nn.Sequential(nn.Linear(history * PROPRIO_DIM, proprio_hidden), nn.SiLU(),
                                     nn.Linear(proprio_hidden, proprio_hidden), nn.SiLU())
        head, d_in = [], c_in * 6 + proprio_hidden
        for _ in range(head_layers):
            head += [nn.Linear(d_in, head_hidden), nn.SiLU()]
            d_in = head_hidden
        self.head = nn.Sequential(*head, nn.Linear(d_in, 2))

    def forward(self, hm: torch.Tensor, proprio: torch.Tensor):
        """Normalised inputs -> (mean [B] in [0, v_max], log_std [B])."""
        z = torch.cat([self.cnn(hm), self.proprio((proprio * self.proprio_keep).flatten(1))], dim=-1)
        out = self.head(z)
        return self.v_max * torch.sigmoid(out[:, 0]), out[:, 1].clamp(-6.0, 1.0)


class VelocityControllerExport(nn.Module):
    """Raw (un-normalised) inputs -> mean commanded speed [B]; normalisation stats as buffers."""

    def __init__(self, net: VelocityNet, hm_scale: float, proprio_mean: torch.Tensor, proprio_std: torch.Tensor):
        super().__init__()
        self.net = net
        self.hm_scale = hm_scale
        self.register_buffer("proprio_mean", proprio_mean.float())
        self.register_buffer("proprio_std", proprio_std.float())

    def forward(self, hm: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        mean, _ = self.net(hm / self.hm_scale, (proprio - self.proprio_mean) / self.proprio_std)
        return mean
