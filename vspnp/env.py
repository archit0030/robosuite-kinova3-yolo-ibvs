"""robosuite PickPlaceCan with a Kinova Gen3 whose base can reach both bins."""
import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.environments.manipulation.pick_place import PickPlaceCan

# robosuite mounts the arm at (-0.5, -0.1). From there the goal compartment is
# outside the Gen3's workspace with a vertical gripper, so the mount is moved
# 18 cm forward and 8 cm towards the goal bin.
BASE_XY = (-0.32, -0.02)

CONTROL_FREQ = 20
OSC_KP = 300                # stiffer than the default 150, for servo tracking
BIN_FLOOR_Z = 0.82          # known workcell geometry
CAN_HEIGHT = 0.08           # known object model
CAN_RADIUS = 0.025
SCENE_CAM = "agentview"
WRIST_CAM = "robot0_eye_in_hand"
IMG_W, IMG_H = 640, 480


class KinovaPickPlaceCan(PickPlaceCan):
    def _load_model(self):
        super()._load_model()
        robot_model = self.robots[0].robot_model
        z = float(robot_model.base_xpos_offset["bins"][2])
        robot_model.set_base_xpos([BASE_XY[0], BASE_XY[1], z])


def make_env(seed=0, horizon=4000):
    cfg = load_composite_controller_config(controller="BASIC", robot="Kinova3")
    cfg["body_parts"]["right"]["kp"] = OSC_KP
    env = KinovaPickPlaceCan(
        robots="Kinova3",
        controller_configs=cfg,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        control_freq=CONTROL_FREQ,
        horizon=horizon,
        ignore_done=True,
        seed=seed,
    )
    # robosuite JIT-compiles its controller maths with numba on first use, after
    # the GL context exists, which breaks rendering (see __init__.py). Trigger
    # the compile now, then hard-reset: that builds a fresh GL context.
    env.reset()
    env.step(np.zeros(env.action_dim))
    env.reset()
    return env


def goal_position(env):
    """Task-specified goal: can centre when standing in its compartment."""
    xy = env.target_bin_placements[env.object_id][:2]
    return np.array([xy[0], xy[1], BIN_FLOOR_Z + CAN_HEIGHT / 2])
