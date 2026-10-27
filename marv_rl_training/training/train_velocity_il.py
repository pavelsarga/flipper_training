"""Imitation-learn the forward-velocity controller from Gazebo shared-control rides. No Isaac.

    python -m marv_rl_training.training.train_velocity_il configs/velocity_il/velocity_il_base.yaml [key=value ...]

Data: the .npz shards of misc/velocity_il/extract_velocity_il.py, one per ride (bag), 20 Hz. A
sample is the `history` ticks ending at tick i of one ride; its label is `label_key` at tick i
(v_label = the operator's throttle; v_cmd, auto_ride's constant speed, only to test the
plumbing on rides that have no operator). Samples are kept only where auto_ride was driving,
the label exists and the SAS rescue was not holding the flippers -- and not within SETTLE_TICKS
of a reset, counting the whole history window: a teleport (policy_sweep moves the robot back
between obstacles -- pose, measured speed and flipper angles jump within one tick, which a
finite-difference velocity turns into a 10-20 rad/s spike) or a flipper-posture reset (angles
snap to 0 faster than the joints can move; Gazebo's flipper joints saturate at 2.36 rad/s). Ticks
with an /obstacle_id in `mask_obstacles` are dropped too, with the first `zone_exit_ticks` after
one: the gap strips between areas force a stop for the map reset, which is not a speed decision.

The previous command is in the proprio frame, but `drop_proprio: [1]` zeroes it inside the net:
copying it is a near-perfect predictor of a smooth label that ignores the terrain. It can still
shape the loss -- `keyframe.weight` up-weights ticks where the throttle changed (Wen et al. 2021,
keyframe-focused imitation) -- and val MAE is also reported on those change ticks, next to the
constant and copy-previous baselines.

The proprio frame is [v measured, label at the previous tick, flipper velocity x 4, flipper angle x 4]:
"the previous command" is the operator's in the data and the controller's own in closed loop,
which is what FtrEnv feeds it (last applied v). Flipper velocity is the finite difference of the
measured angles on the 20 Hz grid (velocity_il_features.flipper_velocity).

Normalisation: mean and std measured from the training data, shared within each physical
quantity (NORM_GROUPS) -- both speeds, the front velocities, the rear velocities, the front
angles, the rear angles -- so differences inside a group (v measured vs v commanded) keep their
meaning and the left/right mirror maps a normalised sample onto a normalised sample.

Rides are split train/val by ride, never by tick (neighbouring ticks are near-duplicates).
Augmentation: heightmap noise, and the left/right mirror -- columns flipped, fl<->fr and rl<->rr
swapped -- under which the right speed is unchanged.

Writes <out_dir>/<name>/: velocity_controller.pt (TorchScript VelocityControllerExport, raw
inputs -> speed, from the eval step with the lowest val MAE; what FtrEnv's velocity_controller {type: il, weights: ...} loads),
net_best.pth (that state), net_final.pth (the last step), config.yaml, metrics.json.
"""
import glob
import json
import os
import sys
import time

import numpy as np
import torch
from omegaconf import OmegaConf

from marv_rl_training.policies.velocity_controller import PROPRIO_DIM, VelocityControllerExport, VelocityNet
from marv_rl_training.training.velocity_il_features import flipper_velocity

MIRROR = [1, 0, 3, 2]
NORM_GROUPS = ([0, 1], [2, 3], [4, 5], [6, 7], [8, 9])
TELEPORT_STEP = 0.1     # m of pose change in one 0.05 s tick (2 m/s, ~3x v_max): a teleport
FLIP_RESET_VEL = 3.0    # rad/s, above the joints' 2.36 rad/s limit: a posture reset, not motion
SETTLE_TICKS = 20       # 1 s after one: the robot drops onto the ground, the flippers are reset


def grouped_stats(x):
    """Per-feature (mean, std) of [N, PROPRIO_DIM] proprio, pooled within each NORM_GROUPS group."""
    mean, std = np.zeros(x.shape[1], np.float32), np.ones(x.shape[1], np.float32)
    for g in NORM_GROUPS:
        mean[g], std[g] = x[:, g].mean(), x[:, g].std()
    return torch.tensor(mean), torch.tensor(std).clamp_min(1e-3)


def load_rides(data_dirs, label_key, history, mask_obstacles=(), zone_exit_ticks=0, keyframe_lag=10):
    """`mask_obstacles`: /obstacle_id values whose ticks are dropped (the gap strips force a stop for
    the map reset; -1 is off the course), together with the first `zone_exit_ticks` after leaving one.
    `change` is |label[t] - label[t - keyframe_lag]|: where the operator changed the throttle."""
    rides = []
    for d in data_dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
            z = np.load(f)
            label = z[label_key].astype(np.float32)
            prev = np.concatenate([label[:1], label[:-1]])
            flip_vel = flipper_velocity(z["flip_angle"])
            proprio = np.concatenate([z["v"][:, None], prev[:, None], flip_vel, z["flip_angle"]], axis=1)
            ok = z["driving"] & ~np.isnan(label) & ~z["sas"] & ~np.isnan(prev)
            step = np.linalg.norm(np.diff(z["pose"][:, :2], axis=0), axis=1)
            reset = (step > TELEPORT_STEP) | (np.abs(flip_vel[1:]).max(1) > FLIP_RESET_VEL)
            for j in np.flatnonzero(reset) + 1:
                ok[j: j + SETTLE_TICKS + history - 1] = False
            masked = np.isin(z["obstacle"], list(mask_obstacles))
            ok &= ~masked
            for j in np.flatnonzero(masked[:-1] & ~masked[1:]) + 1:
                ok[j: j + zone_exit_ticks] = False
            clean = np.nan_to_num(label)
            change = np.abs(clean - np.concatenate([np.full(keyframe_lag, clean[0]), clean[:-keyframe_lag]]))
            rides.append(dict(name=str(z["bag"]), policy=str(z["policy"]), hm=z["hm"].astype(np.float32),
                              proprio=np.nan_to_num(proprio).astype(np.float32), label=clean, ok=ok,
                              change=change.astype(np.float32)))
    return rides


def sample_index(rides, history):
    """[(ride, tick)] of usable samples: full history inside the ride, label usable at the tick."""
    idx = []
    for r, ride in enumerate(rides):
        ticks = np.flatnonzero(ride["ok"])
        idx += [(r, t) for t in ticks if t >= history - 1]
    return np.array(idx, dtype=np.int64).reshape(-1, 2)


def batch(rides, idx, history, device, augment=None):
    hm = np.stack([rides[r]["hm"][t - history + 1: t + 1] for r, t in idx])
    pr = np.stack([rides[r]["proprio"][t - history + 1: t + 1] for r, t in idx])
    y = np.array([rides[r]["label"][t] for r, t in idx], dtype=np.float32)
    hm, pr, y = (torch.from_numpy(a).to(device) for a in (hm, pr, y))
    if augment is not None:
        hm = hm + augment["hm_noise"] * torch.randn_like(hm)
        flip = torch.rand(len(idx), device=device) < augment["mirror_prob"]
        if flip.any():
            hm[flip] = hm[flip].flip(-1)
            p = pr[flip]
            p[..., 2:6] = p[..., 2:6][..., MIRROR]
            p[..., 6:10] = p[..., 6:10][..., MIRROR]
            pr[flip] = p
    return hm, pr, y


def main():
    cfg = OmegaConf.merge(OmegaConf.load(sys.argv[1]), OmegaConf.from_dotlist(sys.argv[2:]))
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    kf = cfg.get("keyframe", {}) or {}
    rides = load_rides(cfg.data_dirs, cfg.label_key, cfg.history, tuple(cfg.get("mask_obstacles", [])),
                       int(cfg.get("zone_exit_ticks", 0)), int(kf.get("lag", 10)))
    if len(rides) < 2:
        raise SystemExit(f"need >= 2 rides for a train/val split, found {len(rides)} in {list(cfg.data_dirs)}")
    order = rng.permutation(len(rides))
    n_val = max(1, int(round(cfg.val_fraction * len(rides))))
    val_rides = set(order[:n_val].tolist())
    idx = sample_index(rides, cfg.history)
    is_val = np.array([r in val_rides for r in idx[:, 0]], dtype=bool)
    tr_idx, va_idx = idx[~is_val], idx[is_val]
    print(f"{len(rides)} rides ({n_val} val: {[rides[i]['name'] for i in sorted(val_rides)]}), "
          f"{len(tr_idx)} train / {len(va_idx)} val samples", flush=True)

    all_pr = np.concatenate([rides[r]["proprio"][rides[r]["ok"]] for r in range(len(rides))])
    pr_mean, pr_std = grouped_stats(all_pr)
    # reference errors on the val rides: a constant (the train label mean) and copying the label of the
    # previous tick -- the second is what a net leaning on the previous-command input can reach blind
    y_va = np.array([rides[r]["label"][t] for r, t in va_idx])
    y_prev = np.array([rides[r]["proprio"][t, 1] for r, t in va_idx])
    y_tr_mean = float(np.mean([rides[r]["label"][t] for r, t in tr_idx]))
    # "change" ticks: the throttle moved by more than change_thresh within the keyframe lag -- where
    # copying the previous command fails, so where a net has to have learned what triggers a change
    va_change = np.array([rides[r]["change"][t] for r, t in va_idx]) > float(kf.get("change_thresh", 0.1))
    baselines = dict(const_mae=float(np.abs(y_va - y_tr_mean).mean()), copy_prev_mae=float(np.abs(y_va - y_prev).mean()),
                     const_mae_change=float(np.abs(y_va - y_tr_mean)[va_change].mean()),
                     copy_prev_mae_change=float(np.abs(y_va - y_prev)[va_change].mean()),
                     val_change_share=float(va_change.mean()))
    print(f"val baselines: constant {baselines['const_mae']:.4f} m/s, copy previous tick "
          f"{baselines['copy_prev_mae']:.4f} m/s; on change ticks ({va_change.mean():.0%}): constant "
          f"{baselines['const_mae_change']:.4f}, copy previous {baselines['copy_prev_mae_change']:.4f}", flush=True)

    net = VelocityNet(cfg.history, tuple(cfg.channels), cfg.proprio_hidden, cfg.head_hidden, cfg.head_layers,
                      cfg.v_max, tuple(cfg.get("drop_proprio", []))).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    norm = lambda hm, pr: (hm / cfg.hm_scale, (pr - pr_mean.to(device)) / pr_std.to(device))
    print(f"VelocityNet: {sum(p.numel() for p in net.parameters()):,} parameters, device {device}", flush=True)

    def evaluate():
        """(val MAE, val MAE on the change ticks)."""
        net.eval()
        errs = []
        with torch.no_grad():
            for s in range(0, len(va_idx), 2048):
                hm, pr, y = batch(rides, va_idx[s: s + 2048], cfg.history, device)
                mean, _ = net(*norm(hm, pr))
                errs.append((mean - y).abs().cpu().numpy())
        net.train()
        err = np.concatenate(errs)
        return float(err.mean()), float(err[va_change].mean())

    # keyframe weighting: the previous command is privileged here, in the LOSS only -- ticks where the
    # operator changed the throttle weigh 1 + weight * change / v_max (weights renormalised per batch)
    kf_weight = float(kf.get("weight", 0.0))

    history_log = []
    best = dict(val_mae=float('inf'))     # every pilot overfit after ~3k steps: export the best eval, not the last
    aug = dict(hm_noise=cfg.augment.hm_noise, mirror_prob=cfg.augment.mirror_prob)
    t0 = time.time()
    for step in range(1, cfg.steps + 1):
        b = tr_idx[rng.integers(0, len(tr_idx), cfg.batch_size)]
        hm, pr, y = batch(rides, b, cfg.history, device, aug)
        mean, log_std = net(*norm(hm, pr))
        nll = 0.5 * ((y - mean) / log_std.exp()) ** 2 + log_std                # Gaussian NLL
        if kf_weight > 0:
            w = 1.0 + kf_weight * torch.tensor([rides[r]["change"][t] for r, t in b], device=device) / cfg.v_max
            loss = (w * nll).sum() / w.sum()
        else:
            loss = nll.mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step % cfg.eval_every == 0 or step == cfg.steps:
            mae, mae_change = evaluate()
            history_log.append(dict(step=step, loss=float(loss), val_mae=mae, val_mae_change=mae_change))
            if mae < best['val_mae']:
                best = dict(step=step, val_mae=mae, val_mae_change=mae_change,
                            state={k: v.detach().clone() for k, v in net.state_dict().items()})
            print(f"step {step}: nll {loss.item():.4f}  val MAE {mae:.4f} m/s  on change ticks {mae_change:.4f}  "
                  f"({time.time() - t0:.0f} s)" + ("  *" if best['step'] == step else ""), flush=True)

    out = os.path.join(cfg.out_dir, cfg.name)
    os.makedirs(out, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(out, "net_final.pth"))
    net.load_state_dict(best["state"])
    torch.save(best["state"], os.path.join(out, "net_best.pth"))
    print(f"exporting the best eval: step {best['step']}, val MAE {best['val_mae']:.4f}, "
          f"on change ticks {best['val_mae_change']:.4f}", flush=True)
    export = VelocityControllerExport(net.cpu().eval(), float(cfg.hm_scale), pr_mean, pr_std)
    torch.jit.script(export).save(os.path.join(out, "velocity_controller.pt"))
    OmegaConf.save(cfg, os.path.join(out, "config.yaml"))
    json.dump(dict(rides=[r["name"] for r in rides], val_rides=[rides[i]["name"] for i in sorted(val_rides)],
                   train_samples=int(len(tr_idx)), val_samples=int(len(va_idx)), baselines=baselines,
                   best={k: v for k, v in best.items() if k != "state"},
                   history=history_log,
                   proprio_dim=PROPRIO_DIM), open(os.path.join(out, "metrics.json"), "w"), indent=1)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
