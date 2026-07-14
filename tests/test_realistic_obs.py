"""Validate the 4 new from_realistic_world implementations against analytic truth."""
import math, sys
from types import SimpleNamespace
import torch
from tensordict import TensorDict

from flipper_training.observations.pan_terrain import PanTerrainState
from flipper_training.observations.previous_action import PreviousAction
from flipper_training.observations.robot_state_with_action import LocalStateVectorWithAction
from flipper_training.observations.latent_control import LatentControlParameter

def mkenv():
    robot_cfg = SimpleNamespace(
        driving_part_names=["front_left", "front_right", "rear_left", "rear_right"],
        num_driving_parts=4,
        joint_limits=torch.tensor([[-1.0472]*4, [0.7854]*4]),
        joint_max_pivot_vels=torch.tensor([[-1.5]*4, [1.5]*4]),
        joint_positions=torch.tensor([[0.256,0.2,0.0],[0.256,-0.2,0.0],[-0.256,0.2,0.0],[-0.256,-0.2,0.0]]),
    )
    return SimpleNamespace(device=torch.device("cpu"), out_dtype=torch.float32, n_robots=1,
                           robot_cfg=robot_cfg, terrain_cfg=SimpleNamespace(max_coord=4.0))

env = mkenv()
fails = []

# ---------- PanTerrainState: plane z_rel = 0.2*x  (local frame) ----------
pan = PanTerrainState(env=env)
Hs, Ws = 81, 81
xs = torch.linspace(2.0, -2.0, Hs)          # row 0 = x_max per extent convention
hm = (0.2 * xs).view(Hs, 1).expand(Hs, Ws).clone()
pitch_deg = 10.0
qy = math.sin(math.radians(pitch_deg)/2); qw = math.cos(math.radians(pitch_deg)/2)
td = TensorDict({
    "heightmap": hm.unsqueeze(0),           # batched like the real deploy path
    "heightmap_extent": torch.tensor([2.0, 2.0, -2.0, -2.0]),
    "thetas": torch.tensor([0.1, 0.3, -0.2, 0.4]),
    "quat": torch.tensor([0.0, qy, 0.0, qw]),   # ROS xyzw: +10 deg about +Y = nose DOWN
}, batch_size=[])
obs = pan.from_realistic_world(td)
assert obs.shape == (1, 18), obs.shape
H = obs[0, :15]; th_f1, th_f2, th_r = obs[0, 15].item(), obs[0, 16].item(), obs[0, 17].item()
H_expected = 0.2 * pan.bin_centers
if not torch.allclose(H, H_expected, atol=1e-4):
    fails.append(f"pan H mismatch: {H} vs {H_expected}")
if abs(th_f1 - (-0.2)) > 1e-6: fails.append(f"theta_f1 {th_f1} != -0.2")
if abs(th_f2 - 0.1) > 1e-6: fails.append(f"theta_f2 {th_f2} != 0.1")
if abs(th_r - (-math.radians(pitch_deg))) > 1e-5: fails.append(f"theta_R {th_r} != {-math.radians(pitch_deg)} (nose-down -> negative)")
# extent-too-small must raise
try:
    bad = td.clone(); bad["heightmap_extent"] = torch.tensor([0.5, 2.0, -0.5, -2.0])
    pan.from_realistic_world(bad); fails.append("pan: no error on insufficient extent")
except ValueError: pass

# ---------- PreviousAction ----------
pa = PreviousAction(env=env)
z = pa.from_realistic_world(TensorDict({}, batch_size=[]))
assert z.shape == (1, 8) and z.abs().sum() == 0, "prev_action zeros fallback broken"
v = torch.arange(8, dtype=torch.float32)
out = pa.from_realistic_world(TensorDict({"prev_action": v}, batch_size=[]))
if not torch.equal(out, v.view(1, 8)): fails.append("prev_action passthrough broken")

# ---------- LocalStateVectorWithAction ----------
rswa = LocalStateVectorWithAction(env=env)
td2 = TensorDict({
    "goal_vec_local": torch.tensor([1.0, 2.0, 0.5]),
    "xd_local": torch.tensor([0.3, 0.0, 0.0]),
    "omega_local": torch.tensor([0.0, 0.1, 0.0]),
    "thetas": torch.tensor([-1.0472, 0.7854, -0.1459, -0.1459]),
    "quat": torch.tensor([0.0, qy, 0.0, qw]),
    "prev_action": v,
}, batch_size=[])
o2 = rswa.from_realistic_world(td2)
assert o2.shape == (1, rswa.dim) == (1, 23), o2.shape
# thetas at limits -> -1 / +1; mid-range value ~0.0247
if abs(o2[0, 8].item() + 1.0) > 1e-4 or abs(o2[0, 9].item() - 1.0) > 1e-4:
    fails.append(f"with_action theta scaling wrong: {o2[0, 8:12]}")
# pitch slot (idx 1): +10deg about +Y / pi
if abs(o2[0, 1].item() - math.radians(10)/math.pi) > 1e-5:
    fails.append(f"with_action pitch wrong: {o2[0,1]}")
if not torch.equal(o2[0, 15:23], v): fails.append("with_action action tail wrong")
# goal normalization
if abs(o2[0, 12].item() - 1.0/(4.0*2**1.5)) > 1e-6: fails.append("with_action goal scaling wrong")

# ---------- LatentControlParameter ----------
lcp = LatentControlParameter(env=env)
o3 = lcp.from_realistic_world(TensorDict({"latent_control": torch.tensor([0.7])}, batch_size=[]))
if abs(o3.item() - 0.7) > 1e-6: fails.append("latent_control key path broken")
env.latent_control_params = torch.tensor([0.3])
o4 = lcp.from_realistic_world(TensorDict({}, batch_size=[]))
if abs(o4.item() - 0.3) > 1e-6: fails.append("latent_control env-attr path broken")
del env.latent_control_params
try:
    lcp.from_realistic_world(TensorDict({}, batch_size=[])); fails.append("latent_control: no error when unset")
except ValueError: pass

print("FAILURES:", fails if fails else "none — all 4 implementations validated")
sys.exit(1 if fails else 0)
