"""This node is flipper control only: it must not publish /cmd_vel for ANY policy.

Track velocity belongs to the operator / autodrive. A policy publishing into /cmd_vel at
the control rate would fight whatever is actually driving — and for the policies trained
behind a fixed forward speed the value would be meaningless anyway, since it is the
constant fed in during training rather than a network decision.

    python src/flipper_training/ros2/test_velocity_gating.py
"""
import ast
import glob
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
NODE = os.path.join(ROOT, "src", "flipper_training", "ros2", "flipper_policy_node.py")

fails = []
def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + ("  " + detail if detail else ""))
    if not cond:
        fails.append(name)

src = open(NODE).read()
tree = ast.parse(src)
lines = src.split("\n")

# --- every /cmd_vel publish must sit behind the flag ---------------------------------
pub_lines = [
    n.lineno for n in ast.walk(tree)
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "publish"
    and isinstance(n.func.value, ast.Attribute) and n.func.value.attr == "cmd_vel_pub"
]
gated = 0
for ln in pub_lines:
    indent = len(lines[ln - 1]) - len(lines[ln - 1].lstrip())
    for k in range(ln - 2, max(0, ln - 8), -1):
        cur = lines[k]
        if cur.strip() and (len(cur) - len(cur.lstrip())) < indent:
            if "publish_cmd_vel" in cur:
                gated += 1
            break
check("every cmd_vel publish site is gated", bool(pub_lines) and gated == len(pub_lines),
      f"{gated}/{len(pub_lines)} gated (lines {pub_lines})")

# --- the flag must default to OFF, and be declared before it is read -----------------
default_off, decl_line, read_line = False, None, None
for n in ast.walk(tree):
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "declare_parameter":
        if n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == "publish_cmd_vel":
            decl_line = n.lineno
            default_off = len(n.args) > 1 and isinstance(n.args[1], ast.Constant) and n.args[1].value is False
for i, ln in enumerate(lines, 1):
    if "self.publish_cmd_vel = self.get_parameter" in ln:
        read_line = i
check("publish_cmd_vel defaults to False (flipper control only)", default_off)
check("parameter is declared before it is read", decl_line and read_line and decl_line < read_line,
      f"declared line {decl_line}, read line {read_line}")

# --- flipper commands must NOT be gated: they are the whole point of the node ---------
flipper_pubs = [
    n.lineno for n in ast.walk(tree)
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "publish"
    and isinstance(n.func.value, ast.Subscript)
]
still_gated = 0
for ln in flipper_pubs:
    indent = len(lines[ln - 1]) - len(lines[ln - 1].lstrip())
    for k in range(ln - 2, max(0, ln - 8), -1):
        cur = lines[k]
        if cur.strip() and (len(cur) - len(cur.lstrip())) < indent:
            if "publish_cmd_vel" in cur:
                still_gated += 1
            break
check("flipper command publishes are NOT gated by publish_cmd_vel",
      still_gated == 0, f"{still_gated} of {len(flipper_pubs)} wrongly gated")

# --- the fixed-velocity warning is still informative ---------------------------------
check("keeps detecting fixed_forward_vel to warn on an explicit override",
      "_detect_fixed_forward_vel" in src and 'get("fixed_forward_vel"' in src)

n_fixed = sum(
    1 for p in glob.glob(os.path.join(ROOT, "configs/baselines/marv_config_*.yaml"))
    if (yaml.safe_load(open(p)).get("env_cfg_overrides", {}) or {}).get("fixed_forward_vel") is not None
)
check("configs with a constant training speed are still identifiable", n_fixed > 0, f"{n_fixed} configs")

print()
print("FAILED:", fails if fails else "none")
sys.exit(1 if fails else 0)
