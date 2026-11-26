"""Thin robot interface: proprioception, cameras and Cartesian commands.

Deliberately exposes no object state, so the controller cannot use it.
"""
import cv2
import numpy as np

from .camera import Camera, make_T
from .env import CONTROL_FREQ, IMG_H, IMG_W, SCENE_CAM, WRIST_CAM
from .kinematics import R_DOWN

DT = 1.0 / CONTROL_FREQ
POS_SCALE, ROT_SCALE = 0.05, 0.5   # OSC_POSE output limits per step
KI, BIAS_MAX = 0.12, 0.03          # move_to integral gain and limit (m)
REF_LAG_MAX = 0.04                 # reference may lead the arm by this much (m)


class Robot:
    OPEN, CLOSE = -1.0, 1.0

    def __init__(self, env):
        self._env = env
        self.scene_cam = Camera(env.sim, SCENE_CAM, IMG_W, IMG_H)
        self.wrist_cam = Camera(env.sim, WRIST_CAM, IMG_W, IMG_H)
        self._sid = env.robots[0].eef_site_id["right"]
        self._gq = env.robots[0]._ref_gripper_joint_pos_indexes["right"]
        self.grip = self.OPEN
        self.on_step = []       # callbacks run after every control step
        self.t = 0
        self._p_ref = None
        self._target = None
        self._bias = np.zeros(3)

    @property
    def T_world_grip(self):
        d = self._env.sim.data
        return make_T(np.array(d.site_xmat[self._sid]).reshape(3, 3), np.array(d.site_xpos[self._sid]))

    @property
    def finger_joint(self):
        """Driver joint angle: about 0 open, increasing as it closes."""
        return float(self._env.sim.data.qpos[self._gq[0]])

    def _rotvec(self):
        return cv2.Rodrigues(R_DOWN @ self.T_world_grip[:3, :3].T)[0].ravel()

    @property
    def rotation_error(self):
        return float(np.linalg.norm(self._rotvec()))

    def _rot_action(self):
        return np.clip(self._rotvec() / ROT_SCALE, -1, 1)

    def command(self, dp):
        """One control step with a world-frame translation command (metres)."""
        a = np.zeros(7)
        a[:3] = np.clip(dp / POS_SCALE, -1, 1)
        a[3:6] = self._rot_action()
        a[6] = self.grip
        self._env.step(a)
        self.t += 1
        for cb in self.on_step:
            cb()

    def start_velocity(self):
        self._target = None
        self._p_ref = self.T_world_grip[:3, 3].copy()

    def velocity(self, v):
        """Track a world-frame Cartesian velocity.

        OSC deltas are relative to the achieved pose, so small deltas are lost
        to compliance. Integrating into a reference position makes OSC's
        stiffness pull the arm along; the clamp prevents windup.
        """
        self._p_ref = self._p_ref + np.asarray(v) * DT
        p = self.T_world_grip[:3, 3]
        lag = self._p_ref - p
        n = np.linalg.norm(lag)
        if n > REF_LAG_MAX:
            self._p_ref = p + lag * REF_LAG_MAX / n
        self.command(self._p_ref - p)

    def move_to(self, p, tol=0.004, max_steps=200, speed=0.05, settle=3, rot_tol=None):
        """Closed-loop Cartesian move with the gripper pointing down."""
        self._target = np.asarray(p, dtype=float)
        if np.linalg.norm(self._target - self.T_world_grip[:3, 3]) > 0.05:
            self._bias = np.zeros(3)
        ok = 0
        for _ in range(max_steps):
            n = self._track(speed)
            level = rot_tol is None or self.rotation_error < rot_tol
            ok = ok + 1 if n < tol and level else 0
            if ok >= settle:
                return True
        return False

    def hold(self, steps):
        """Keep regulating to the last target (or here, if not near it)."""
        p = self.T_world_grip[:3, 3]
        if self._target is None or np.linalg.norm(self._target - p) > 0.02:
            self._target, self._bias = p.copy(), np.zeros(3)
        for _ in range(steps):
            self._track(0.05)

    def _track(self, speed):
        e = self._target - self.T_world_grip[:3, 3]
        n = np.linalg.norm(e)
        if n < 0.03:            # integral action against steady-state sag
            self._bias = np.clip(self._bias + KI * e, -BIAS_MAX, BIAS_MAX)
        self.command(e * min(1.0, speed / max(n, 1e-9)) + self._bias)
        return np.linalg.norm(self._target - self.T_world_grip[:3, 3])
