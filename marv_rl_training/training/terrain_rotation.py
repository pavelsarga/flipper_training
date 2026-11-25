"""Trainer-side terrain rotation: decide when the current course is used up, generate the next
one in the background while training continues, then swap it into the running sim
(FtrEnv.swap_terrain, ~3 s) — or, if that fails or ``in_process: false``, hand over by exiting
with ROTATE_EXIT_CODE and letting the launcher relaunch on it (~1 min).

Configured by the trainer config's ``terrain_rotation`` dict (None = off):

```yaml
terrain_rotation:
  prefix: rot            # course names <prefix>_<k:03d>; k is parsed from the current terrain
  pregen_fraction: 0.6   # start generating course k+1 at this retired fraction of course k
  min_frames: 500000     # never rotate earlier than this on one course (control-step frames)
  max_frames: 4000000    # always rotate by this, exhausted or not
  keep: 3                # rotated courses kept on disk (each ~75 MB)
  generator_args: {}     # passed to ftr_terrain_gen.rotation as --key value (n_rows, repeats, margin, ...)
  in_process: true       # swap the course inside the running sim; false = always save + exit 76
```

The exhaustion signal comes from FtrEnv's spawn sampler (``env_cfg_overrides.spawn_sampler``),
which must be enabled. Design: docs/terrain_generation_research.md §5.

Flow per course k:
1. training runs; ``step()`` is called once per iteration;
2. at ``pregen_fraction`` (or 80 % of max_frames) the sampler snapshot is written to
   ``<run_root>/terrain_rotation/stats_<course>.json`` and ``python -m ftr_terrain_gen.rotation``
   is started as a low-priority subprocess (CPU only) — course k+1 is built while k trains;
3. when the course is exhausted (and min_frames reached) or max_frames passed, ``step()``
   waits for the generator (normally finished already), writes ``next_course.txt`` and returns
   True;
4. in-process (default): the trainer calls ``FtrEnv.swap_terrain(next)``, resets its collector and
   then ``advance()`` — training continues on the new course in the same process. Fallback: the
   trainer saves its checkpoint and exits ROTATE_EXIT_CODE, and the launcher
   (slurm/lib/train_common.sh, scripts/train_rotation.sh) relaunches with ``--terrain <next>``.

``next_course.txt`` always names the course a relaunch must start on — the launchers read it before
EVERY relaunch (crash respawns too), so a respawn after in-process swaps resumes on the current
course, not the config's first one.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROTATE_EXIT_CODE = 76
WS_ROOT = Path(__file__).resolve().parents[4]
GEN_ROOT = WS_ROOT / "src" / "FTR-Bench-terrain-gen"
ASSET_DIR = WS_ROOT / "src" / "FTR-Benchmark" / "ftr_envs" / "assets" / "terrain"
_KNOWN = {"prefix", "pregen_fraction", "min_frames", "max_frames", "keep", "generator_args", "seed", "in_process"}


class TerrainRotation:
    def __init__(self, cfg: dict, run_root: Path, current_terrain: str, ftr_env, start_frames: int,
                 dr_spec: dict | None = None, log=print):
        cfg = dict(cfg)
        unknown = set(cfg) - _KNOWN
        if unknown:
            raise ValueError(f"terrain_rotation: unknown keys {sorted(unknown)}; known: {sorted(_KNOWN)}")
        if getattr(ftr_env, "_spawn_sampler", None) is None:
            raise ValueError("terrain_rotation needs env_cfg_overrides.spawn_sampler (the exhaustion signal)")
        self.prefix = str(cfg.get("prefix", "rot"))
        self.pregen_fraction = float(cfg.get("pregen_fraction", 0.6))
        self.min_frames = int(cfg.get("min_frames", 500_000))
        self.max_frames = int(cfg.get("max_frames", 4_000_000))
        self.keep = int(cfg.get("keep", 3))
        self.seed = int(cfg.get("seed", 0))
        self.in_process = bool(cfg.get("in_process", True))
        self.generator_args = dict(cfg.get("generator_args") or {})
        self.env = ftr_env
        self.dr_spec = dr_spec
        self.log = log
        self.terrain = current_terrain
        m = re.fullmatch(rf"{re.escape(self.prefix)}_(\d+)", current_terrain)
        self.index = int(m.group(1)) if m else -1  # a non-rotation start course (e.g. mixed_v3_straight) is index -1
        self.state_dir = Path(run_root) / "terrain_rotation"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.start_frames = int(start_frames)
        self.proc: subprocess.Popen | None = None
        self._gen_log = None
        self.next_course: str | None = None
        self.log(f"[terrain_rotation] course {current_terrain} (index {self.index}); pregen at "
                 f"{self.pregen_fraction:.0%} retired, rotate when exhausted after >= {self.min_frames} frames "
                 f"(at most {self.max_frames}); state in {self.state_dir}")

    # ------------------------------------------------------------------
    def _next_name(self) -> str:
        return f"{self.prefix}_{self.index + 1:03d}"

    def _start_generator(self):
        stats = self.state_dir / f"stats_{self.terrain}.json"
        self.env.write_spawn_snapshot(stats, extra={"course": self.terrain})
        cmd = [sys.executable, "-m", "ftr_terrain_gen.rotation",
               "--rotation-dir", str(self.state_dir), "--index", str(self.index + 1), "--prefix", self.prefix,
               "--output-dir", str(ASSET_DIR), "--keep", str(self.keep), "--seed", str(self.seed)]
        if self.index >= 0:
            cmd += ["--stats", str(stats)]
        if self.dr_spec:
            cmd += ["--dr-spec", json.dumps(self.dr_spec)]
        for k, v in self.generator_args.items():
            cmd += [f"--{k.replace('_', '-')}", str(v)]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(GEN_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        self._gen_log = open(self.state_dir / f"generate_{self._next_name()}.log", "w")
        self.proc = subprocess.Popen(cmd, cwd=str(GEN_ROOT), env=env, stdout=self._gen_log, stderr=subprocess.STDOUT,
                                     preexec_fn=lambda: os.nice(10))
        self._gen_t0 = time.time()
        self.log(f"[terrain_rotation] generating {self._next_name()} in the background (pid {self.proc.pid}); "
                 f"snapshot {stats.name}")

    def _wait_generator(self) -> str:
        rc = self.proc.wait()
        self._gen_log.close()
        out = (self.state_dir / f"generate_{self._next_name()}.log").read_text().strip().splitlines()
        if rc != 0 or not out:
            raise RuntimeError(f"terrain generation for {self._next_name()} failed (exit {rc}); "
                               f"see {self.state_dir}/generate_{self._next_name()}.log")
        name = out[-1].strip()
        self.log(f"[terrain_rotation] {name} ready ({time.time() - self._gen_t0:.0f} s after start)")
        return name

    # ------------------------------------------------------------------
    def step(self, collected_frames: int) -> bool:
        """Call once per training iteration. True = save and exit with ROTATE_EXIT_CODE now."""
        on_course = collected_frames - self.start_frames
        s = self.env._spawn_sampler
        if self.proc is None and (s.retired_fraction >= self.pregen_fraction or on_course >= 0.8 * self.max_frames):
            self._start_generator()
        rotate = (s.exhausted and on_course >= self.min_frames) or on_course >= self.max_frames
        if not rotate:
            return False
        if self.proc is None:
            self._start_generator()
        self.next_course = self._wait_generator()
        self.env.write_spawn_snapshot(self.state_dir / f"stats_{self.terrain}_final.json", extra={"course": self.terrain})
        (self.state_dir / "next_course.txt").write_text(self.next_course + "\n")
        with open(self.state_dir / "courses.jsonl", "a") as f:
            f.write(json.dumps({"course": self.terrain, "next": self.next_course, "frames_on_course": on_course,
                                "end_frames": collected_frames, **{k: v for k, v in s.stats().items()}}) + "\n")
        self.log(f"[terrain_rotation] rotating {self.terrain} -> {self.next_course} after {on_course} frames "
                 f"(retired {s.retired_fraction:.0%}, exhausted={s.exhausted})")
        return True

    def advance(self, collected_frames: int):
        """The trainer swapped ``next_course`` into the running sim: make it the current course."""
        self.terrain = self.next_course
        m = re.fullmatch(rf"{re.escape(self.prefix)}_(\d+)", self.terrain)
        self.index = int(m.group(1)) if m else self.index + 1
        self.start_frames = int(collected_frames)
        self.proc = None
        self._gen_log = None
        self.next_course = None
        self.swaps = getattr(self, "swaps", 0) + 1
        self.log(f"[terrain_rotation] now on {self.terrain} (index {self.index}, in-process swap #{self.swaps})")

    def stats(self) -> dict:
        return {"rotation/course_index": self.index, "rotation/generating": float(self.proc is not None and self.proc.poll() is None)}
