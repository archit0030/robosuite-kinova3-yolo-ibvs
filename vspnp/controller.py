"""Vision-based pick and place: coarse localisation, IBVS, grasp, place."""
from dataclasses import dataclass, field

import numpy as np

from .camera import inv_T, make_T, transform
from .env import BIN_FLOOR_Z, CAN_HEIGHT

HOME = np.array([-0.02, 0.0, 1.14])   # arm clear of the scene camera's view
CARRY_Z = 1.08
HOVER_ABOVE_TOP = 0.15                # start of visual servoing
GRASP_BELOW_TOP = 0.045
PLACE_CLEARANCE = 0.004
FINGER_EMPTY = 0.45                   # rad; closes to 0.51 on air, about 0.39 on the can

# Visual servoing
LAMBDA = 2.5                          # image error gain, 1/s
K_DEPTH = 1.8                         # approach gain, 1/s
Z_STANDOFF = 0.212                    # camera-to-top depth; closer, the hand occludes the can
GATE_SIGMA = 0.08                     # normalised image error at which descent is damped
TOL_PX, TOL_Z = 2.0, 0.006
V_MAX = 0.25
EMA = 0.7
MAX_LOST = 15


@dataclass
class Result:
    success_visual: bool = False
    failure: str = ""
    attempts: int = 0
    coarse_top: np.ndarray = None
    final_top: np.ndarray = None
    servo_converged: bool = False
    servo_steps: int = 0
    placed_top: np.ndarray = None
    err_px: list = field(default_factory=list)
    err_z: list = field(default_factory=list)


class PickPlaceController:
    def __init__(self, robot, detector, goal_center, log=print):
        self.r, self.det, self.goal, self.log = robot, detector, np.asarray(goal_center, float), log
        self.phase = "init"
        self.overlay = {}        # latest wrist feature/target, for the video

    def _set_phase(self, name):
        self.phase = name
        self.log(f"  [{self.r.t:4d}] {name}")

    # -- 1. coarse localisation from the fixed scene camera -----------------
    def localize_scene(self, frames=5):
        pts = []
        for _ in range(frames):
            o = self.det.observe(self.r.scene_cam, *self.r.scene_cam.capture())
            if o is not None:
                pts.append(o.p_world)
            self.r.hold(1)
        return np.median(pts, axis=0) if pts else None

    # -- 2. image-based visual servoing with the wrist camera ---------------
    def visual_servo(self, res, max_steps=200):
        cam = self.r.wrist_cam
        f = cam.K[0, 0]
        c = cam.K[:2, 2]
        # Hand-eye: the grip axis in the camera frame is x = X, y = Y (constant).
        T_cam_grip = inv_T(cam.T_world_cam) @ self.r.T_world_grip
        X, Y = T_cam_grip[:2, 3]
        s = z = uv_track = p_top = None
        lost = good = 0
        self.r.start_velocity()
        for k in range(max_steps):
            rgb, depth = cam.capture()
            o = self.det.observe(cam, rgb, depth, near_uv=uv_track, gate=120.0)
            if o is None:
                lost += 1
                self.overlay = {}
                if uv_track is None or lost > 3:
                    uv_track = s = None          # drop the track, reacquire
                if lost > MAX_LOST:
                    return None
                self.r.velocity(np.array([0, 0, 0.08]))   # back off to widen the view
                continue
            lost = 0
            m = (o.uv - c) / f
            s = m if s is None else EMA * m + (1 - EMA) * s
            z = o.z_cam if z is None else EMA * o.z_cam + (1 - EMA) * z
            p_top = o.p_world if p_top is None else EMA * o.p_world + (1 - EMA) * p_top
            uv_track = o.uv

            s_star = np.array([X, Y]) / z        # where the grip axis meets the can top
            e = s - s_star
            e_px, e_z = float(np.linalg.norm(e) * f), z - Z_STANDOFF
            res.err_px.append(e_px)
            res.err_z.append(e_z)
            self.overlay = {"box": o.box, "conf": o.conf, "uv": s * f + c, "target": s_star * f + c}

            good = good + 1 if e_px < TOL_PX and abs(e_z) < TOL_Z else 0
            if good >= 4:
                res.servo_converged = True
                res.servo_steps = k + 1
                return p_top

            # s_dot = L_xy v_xy + L_z v_z, with L_xy = -I/z and L_z = s/z.
            # s* also moves with depth: s*_dot = (s*/z) v_z. Impose e_dot = -lambda e.
            vz = K_DEPTH * e_z * np.exp(-(e @ e) / GATE_SIGMA**2)
            vxy = -z * (-LAMBDA * e - (s / z) * vz + (s_star / z) * vz)
            v_cam = np.array([vxy[0], vxy[1], vz])
            v_cam *= min(1.0, V_MAX / max(np.linalg.norm(v_cam), 1e-9))
            self.r.velocity(cam.T_world_cam[:3, :3] @ v_cam)
        res.servo_steps = max_steps
        return None

    # -- full task ----------------------------------------------------------
    def run(self, max_attempts=3):
        res = Result()
        r = self.r
        for attempt in range(1, max_attempts + 1):
            res.attempts = attempt
            r.grip = r.OPEN
            self._set_phase("home")
            up = r.T_world_grip[:3, 3].copy()
            up[2] = max(up[2], CARRY_Z)
            r.move_to(up, tol=0.02)
            r.move_to(HOME, tol=0.01)

            self._set_phase("detect (scene camera)")
            top = self.localize_scene()
            if top is None:
                res.failure = "can not detected by scene camera"
                continue
            if attempt == 1:
                res.coarse_top = top
            self.log(f"         coarse can top  {np.round(top, 4)}")

            self._set_phase("approach")
            r.move_to(np.array([top[0], top[1], CARRY_Z]), tol=0.015)
            r.move_to(top + [0, 0, HOVER_ABOVE_TOP], tol=0.008)

            self._set_phase("visual servo (wrist camera)")
            top = self.visual_servo(res)
            self.overlay = {}
            if top is None:
                res.failure = "visual servo did not converge"
                continue
            res.final_top = top
            self.log(f"         servoed can top {np.round(top, 4)}  "
                     f"err {res.err_px[-1]:.2f}px in {res.servo_steps} steps")

            self._set_phase("descend")
            r.move_to(top - [0, 0, GRASP_BELOW_TOP], speed=0.03)

            self._set_phase("grasp")
            r.grip = r.CLOSE
            r.hold(22)
            # Object pose in the gripper frame, from perception + kinematics.
            T_world_obj = make_T(np.eye(3), top - [0, 0, CAN_HEIGHT / 2])
            T_grip_obj = inv_T(r.T_world_grip) @ T_world_obj

            self._set_phase("lift")
            p = r.T_world_grip[:3, 3].copy()
            r.move_to(np.array([p[0], p[1], CARRY_Z]), tol=0.01, speed=0.03)
            if not self.holding(T_grip_obj):
                self.log("         grasp check failed, retrying")
                res.failure = "grasp failed"
                continue

            self._set_phase("transfer")
            r.move_to(np.array([*self.goal[:2], CARRY_Z]), tol=0.03, speed=0.04)
            slip = self.in_hand_slip(T_grip_obj)
            T_grip_obj = T_grip_obj @ make_T(np.eye(3), [0, 0, -slip])
            self.log(f"         in-hand slip measured {slip*1000:.1f} mm")
            T_goal_obj = make_T(T_world_obj[:3, :3], self.goal + [0, 0, PLACE_CLEARANCE])
            p_goal = (T_goal_obj @ inv_T(T_grip_obj))[:3, 3]
            r.move_to(np.array([p_goal[0], p_goal[1], CARRY_Z]), tol=0.006, speed=0.04, rot_tol=0.04, max_steps=120)

            self._set_phase("place")
            r.move_to(p_goal, speed=0.03, rot_tol=0.04, max_steps=120)
            r.hold(5)

            self._set_phase("release")
            r.grip = r.OPEN
            r.hold(15)
            self._set_phase("retreat")
            r.move_to(p_goal + [0, 0, 0.16], tol=0.015)

            self._set_phase("verify (wrist camera)")
            o = self.det.observe(r.wrist_cam, *r.wrist_cam.capture())
            if o is not None:
                res.placed_top = o.p_world
                err = np.linalg.norm(o.p_world[:2] - self.goal[:2])
                res.success_visual = bool(err < 0.04)
                res.failure = "" if res.success_visual else f"placed {err*100:.1f} cm from goal"
                self.log(f"         seen {err*1000:.1f} mm from goal")
            else:
                res.failure = "can not seen at goal"
            break
        self._set_phase("done")
        return res

    def in_hand_slip(self, T_grip_obj, frames=5):
        """How far the can slid down in the fingers, from the wrist camera.

        The hand hides part of the top face, so only its height is trusted.
        """
        zs = []
        for _ in range(frames):
            expected = transform(self.r.T_world_grip @ T_grip_obj, [0, 0, CAN_HEIGHT / 2])[2]
            o = self.det.observe(self.r.wrist_cam, *self.r.wrist_cam.capture(),
                                 z_band=(expected - 0.05, expected + 0.01))
            if o is not None:
                zs.append(expected - o.p_world[2])
            self.r.hold(1)
        return float(np.clip(np.median(zs), 0.0, 0.05)) if zs else 0.0

    def holding(self, T_grip_obj):
        """Grasp check: fingers stopped early and the can is seen in the hand."""
        o_boxes = self.det.detect(self.r.wrist_cam.capture()[0])
        p_top = transform(self.r.T_world_grip @ T_grip_obj, [0, 0, CAN_HEIGHT / 2])
        uv = self.r.wrist_cam.project(transform(inv_T(self.r.wrist_cam.T_world_cam), p_top))
        seen = any(np.linalg.norm((b[:2] + b[2:4]) / 2 - uv) < 120 for b in o_boxes)
        blocked = self.r.finger_joint < FINGER_EMPTY
        self.log(f"         grasp check: finger joint {self.r.finger_joint:.3f} "
                 f"(blocked {blocked}), can seen in hand {seen}")
        return blocked and seen
