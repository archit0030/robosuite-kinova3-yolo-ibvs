"""Render an auto-labelled YOLO dataset (single class: can) from the simulator.

Boxes come from MuJoCo segmentation masks. Ground truth is used here, for
labelling, and never by the pick-and-place controller.
"""
import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vspnp.camera import Camera
from vspnp.env import IMG_H, IMG_W, SCENE_CAM, WRIST_CAM, make_env
from vspnp.kinematics import R_DOWN, solve_ik

MIN_PIXELS = 60


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


class SceneSampler:
    def __init__(self, env, rng):
        self.env, self.rng = env, rng
        self.sim = env.sim
        self.cams = {c: Camera(self.sim, c, IMG_W, IMG_H) for c in (SCENE_CAM, WRIST_CAM)}
        body = env.obj_body_id["Can"]
        m = self.sim.model
        self.can_geoms = [i for i in range(m.ngeom) if m.geom_bodyid[i] == body]
        self.can_joint = env.objects[env.object_id].joints[0]
        self.grip_q = env.robots[0]._ref_gripper_joint_pos_indexes["right"]
        self.q_home = self.sim.data.qpos.copy()
        self.grip_open, self.grip_closed = self._gripper_extremes()
        self.light0 = m.light_diffuse.copy()
        self.sid = env.robots[0].eef_site_id["right"]

    def _gripper_extremes(self):
        out = []
        for s in (-1.0, 1.0):
            for _ in range(25):
                a = np.zeros(7)
                a[6] = s
                self.env.step(a)
            out.append(self.sim.data.qpos[self.grip_q].copy())
        return out

    def _set_can(self, pos, yaw=0.0, lying=False):
        q = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        if lying:
            c, s = np.cos(yaw / 2), np.sin(yaw / 2)
            r = np.sqrt(0.5)
            q = np.array([c * r, c * r, s * r, s * r])
        self.sim.data.set_joint_qpos(self.can_joint, np.r_[pos, q])

    def sample(self):
        rng = self.rng
        self.sim.data.qpos[:] = self.q_home
        mode = rng.choice(["pick", "held", "goal", "absent"], p=[0.66, 0.14, 0.12, 0.08])
        lying = mode == "pick" and rng.random() < 0.1
        z = 0.845 if lying else 0.86
        if mode == "goal":
            can = np.array([rng.uniform(0.13, 0.27), rng.uniform(0.32, 0.49), z])
        else:
            can = np.array([rng.uniform(-0.06, 0.26), rng.uniform(-0.45, -0.05), z])

        # Half the poses are close to the can (servoing), the rest roam.
        if rng.random() < 0.55:
            h = rng.uniform(0.0, 0.24)
            spread = 0.012 + 0.35 * h
            eef = can + np.r_[rng.normal(0, spread, 2), h]
        else:
            eef = np.array([rng.uniform(-0.05, 0.25), rng.uniform(-0.42, 0.42), rng.uniform(0.95, 1.2)])
        eef[0] = np.clip(eef[0], -0.05, 0.24)
        eef[1] = np.clip(eef[1], -0.44, 0.43)
        tilt = rng.normal(0, 0.05, 2)
        Rx = cv2.Rodrigues(np.array([tilt[0], tilt[1], 0.0]))[0]
        R = Rx @ rot_z(rng.normal(0, 0.15)) @ R_DOWN
        if not solve_ik(self.env, eef, R):
            return None
        a = rng.random() if mode != "held" else rng.uniform(0.45, 0.65)
        if rng.random() < 0.5 and mode != "held":
            a = 0.0
        self.sim.data.qpos[self.grip_q] = (1 - a) * self.grip_open + a * self.grip_closed
        if mode == "held":
            can = self.sim.data.site_xpos[self.sid] + np.r_[rng.normal(0, 0.004, 2), rng.uniform(-0.02, 0.01)]
        if mode == "absent":
            can = np.array([0.0, 0.0, -5.0])
        self._set_can(can, rng.uniform(-np.pi, np.pi), lying)
        self.sim.model.light_diffuse[:] = np.clip(self.light0 * rng.uniform(0.6, 1.3), 0, 1)
        self.sim.forward()
        return mode

    def label(self, cam):
        seg = self.sim.render(camera_name=cam.name, width=IMG_W, height=IMG_H, segmentation=True)[::-1]
        mask = np.isin(seg[..., 1], self.can_geoms) & (seg[..., 0] == 5)  # 5 = mjOBJ_GEOM
        if mask.sum() < MIN_PIXELS:
            return None
        v, u = np.nonzero(mask)
        return u.min(), v.min(), u.max() + 1, v.max() + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=1200)
    ap.add_argument("--val", type=int, default=200)
    ap.add_argument("--test", type=int, default=200)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "can")
    args = ap.parse_args()
    if args.out.exists():
        shutil.rmtree(args.out)

    for k, split in enumerate(("train", "val", "test")):
        n = getattr(args, split)
        rng = np.random.default_rng(1000 + k)
        env = make_env(seed=1000 + k)
        env.reset()
        sampler = SceneSampler(env, rng)
        (args.out / "images" / split).mkdir(parents=True)
        (args.out / "labels" / split).mkdir(parents=True)
        i = positives = 0
        while i < n:
            if sampler.sample() is None:
                continue
            cam = sampler.cams[WRIST_CAM if rng.random() < 0.65 else SCENE_CAM]
            rgb, _ = cam.capture()
            box = sampler.label(cam)
            stem = f"{split}_{i:05d}_{'wrist' if cam.name == WRIST_CAM else 'scene'}"
            cv2.imwrite(str(args.out / "images" / split / f"{stem}.jpg"), rgb[..., ::-1])
            line = ""
            if box is not None:
                x0, y0, x1, y1 = box
                line = f"0 {(x0+x1)/2/IMG_W:.6f} {(y0+y1)/2/IMG_H:.6f} {(x1-x0)/IMG_W:.6f} {(y1-y0)/IMG_H:.6f}\n"
                positives += 1
            (args.out / "labels" / split / f"{stem}.txt").write_text(line)
            i += 1
        print(f"{split}: {n} images, {positives} with a visible can", flush=True)
        env.close()

    (args.out / "data.yaml").write_text(
        f"path: {args.out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: can\n"
    )


if __name__ == "__main__":
    main()
