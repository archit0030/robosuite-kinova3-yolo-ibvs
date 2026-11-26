"""Run vision-based pick and place and score it against simulator ground truth.

Ground truth is read only in this file, for scoring and for --disturb. The
controller receives images, calibration, proprioception and the goal.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vspnp.controller import PickPlaceController
from vspnp.env import CAN_HEIGHT, CONTROL_FREQ, goal_position, make_env
from vspnp.perception import CanDetector
from vspnp.robot import DT, Robot

GREEN, YELLOW, WHITE = (60, 220, 60), (255, 220, 0), (255, 255, 255)


class Recorder:
    def __init__(self, path, robot, ctrl, det, title):
        self.w = imageio.get_writer(str(path), fps=CONTROL_FREQ, codec="libx264", quality=8, macro_block_size=8)
        self.r, self.c, self.det, self.title = robot, ctrl, det, title

    @staticmethod
    def _box(img, b):
        x0, y0, x1, y1 = b[:4].astype(int)
        cv2.rectangle(img, (x0, y0), (x1, y1), GREEN, 2)
        cv2.putText(img, f"can {b[4]:.2f}", (x0, max(y0 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREEN, 2)

    def frame(self):
        panels = []
        for cam, name in ((self.r.scene_cam, "scene camera"), (self.r.wrist_cam, "wrist camera")):
            img = np.ascontiguousarray(cam.capture()[0])
            for b in self.det.detect(img):
                self._box(img, b)
            if cam is self.r.wrist_cam and self.c.overlay:
                for key, col in (("target", YELLOW), ("uv", WHITE)):
                    u = tuple(int(x) for x in self.c.overlay[key])
                    cv2.drawMarker(img, u, col, cv2.MARKER_CROSS if key == "target" else cv2.MARKER_TILTED_CROSS, 26, 2)
                cv2.arrowedLine(img, tuple(int(x) for x in self.c.overlay["uv"]),
                                tuple(int(x) for x in self.c.overlay["target"]), YELLOW, 2)
            cv2.putText(img, name, (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 2)
            panels.append(img)
        out = np.hstack(panels)
        bar = np.zeros((40, out.shape[1], 3), np.uint8)
        cv2.putText(bar, f"{self.title}   phase: {self.c.phase}   t={self.r.t * DT:.1f}s", (10, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 1, cv2.LINE_AA)
        self.w.append_data(np.vstack([bar, out]))

    def close(self):
        self.w.close()


def run_episode(env, det, seed, video=None, disturb=False):
    np.random.seed(seed)
    env.rng = np.random.default_rng(seed)
    env.placement_initializer.rng = env.rng
    for s in getattr(env.placement_initializer, "samplers", {}).values():
        s.rng = env.rng
    env.reset()

    robot = Robot(env)
    ctrl = PickPlaceController(robot, det, goal_position(env))
    body = env.obj_body_id["Can"]
    gt = lambda: np.array(env.sim.data.body_xpos[body])
    start = gt()
    rec = Recorder(video, robot, ctrl, det, f"Kinova Gen3 pick-place, seed {seed}" + (" (disturbed)" if disturb else "")) if video else None

    gt_at_grasp = {}
    moved = {"n": 0}
    joint = env.objects[env.object_id].joints[0]
    centre = np.array([0.1, -0.25])
    direction = centre - start[:2]
    direction = direction / max(np.linalg.norm(direction), 1e-6)

    def hook():
        if ctrl.phase.startswith("descend") and "p" not in gt_at_grasp:
            gt_at_grasp["p"] = gt()
        if disturb and ctrl.phase.startswith("visual servo") and moved["n"] < 24:
            q = env.sim.data.get_joint_qpos(joint).copy()
            q[:2] += direction * 0.04 / 24           # 4 cm at about 3 cm/s
            env.sim.data.set_joint_qpos(joint, q)
            moved["n"] += 1
        if rec:
            rec.frame()

    robot.on_step.append(hook)
    print(f"episode seed={seed}  (scoring only) can starts at {np.round(start, 3)}")
    res = ctrl.run()
    if rec:
        for _ in range(10):
            rec.frame()
        rec.close()

    end, half = gt(), np.array([0, 0, CAN_HEIGHT / 2])
    up = np.array(env.sim.data.body_xmat[body]).reshape(3, 3)[2, 2]
    d = lambda a, b: None if a is None else float(np.linalg.norm((a - b)[:2]) * 1000)
    m = {
        "seed": seed,
        "disturb": disturb,
        "success": bool(env._check_success()),
        "success_visual": res.success_visual,
        "failure": res.failure,
        "attempts": res.attempts,
        "upright": bool(abs(up) > 0.95),
        "coarse_err_mm": d(res.coarse_top, start + half),
        "servo_err_mm": d(res.final_top, gt_at_grasp.get("p", start) + half),
        "placement_err_mm": d(end, goal_position(env)),
        "servo_converged": res.servo_converged,
        "servo_steps": res.servo_steps,
        "final_err_px": res.err_px[-1] if res.err_px else None,
        "disturbance_mm": float(np.linalg.norm(gt_at_grasp["p"][:2] - start[:2]) * 1000) if "p" in gt_at_grasp else None,
        "sim_seconds": robot.t * DT,
        "err_px": res.err_px,
    }
    print(f"  -> robosuite success={m['success']} visual={m['success_visual']} upright={m['upright']} "
          f"coarse={m['coarse_err_mm']} servo={m['servo_err_mm']} place={m['placement_err_mm']} mm {m['failure']}", flush=True)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video", action="store_true", help="record an mp4 per episode")
    ap.add_argument("--disturb", action="store_true", help="push the can 4 cm during visual servoing")
    ap.add_argument("--weights", default=str(ROOT / "models" / "can_detector.pt"))
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    det = CanDetector(args.weights)
    env = make_env(seed=args.seed)
    tag = "disturbed" if args.disturb else "nominal"
    rows = []
    for i in range(args.episodes):
        s = args.seed + i
        video = args.out / f"{tag}_seed{s}.mp4" if args.video else None
        rows.append(run_episode(env, det, s, video, args.disturb))

    (args.out / f"{tag}_metrics.json").write_text(json.dumps(rows, indent=1))
    n = len(rows)
    print(f"\n{tag}: {sum(r['success'] for r in rows)}/{n} success, "
          f"{sum(r['upright'] for r in rows)}/{n} upright, "
          f"{sum(r['attempts'] == 1 and r['success'] for r in rows)}/{n} first attempt")
    for k in ("coarse_err_mm", "servo_err_mm", "placement_err_mm"):
        v = [r[k] for r in rows if r[k] is not None]
        if v:
            print(f"  {k:18s} mean {np.mean(v):.1f}  max {np.max(v):.1f}")


if __name__ == "__main__":
    main()
