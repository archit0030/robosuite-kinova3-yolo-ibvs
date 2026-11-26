"""Damped-least-squares IK. Used only to pose the arm for dataset images."""
import mujoco
import numpy as np

# Gripper pointing straight down, fingers opening along world y.
R_DOWN = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])


def solve_ik(env, pos, R=R_DOWN, iters=120):
    m, d = env.sim.model._model, env.sim.data._data
    robot = env.robots[0]
    sid = robot.eef_site_id["right"]
    dofs = np.array(robot._ref_joint_vel_indexes)
    qadr = np.array(robot._ref_joint_pos_indexes)
    jids = [env.sim.model.joint_name2id(j) for j in robot.robot_joints]
    lo = np.array([m.jnt_range[j][0] if m.jnt_limited[j] else -np.inf for j in jids])
    hi = np.array([m.jnt_range[j][1] if m.jnt_limited[j] else np.inf for j in jids])
    for _ in range(iters):
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        Rc = d.site_xmat[sid].reshape(3, 3)
        ep = pos - d.site_xpos[sid]
        E = R @ Rc.T
        eo = 0.5 * np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]])
        err = np.r_[ep, eo]
        if np.linalg.norm(err) < 1e-4:
            break
        Jp, Jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, Jp, Jr, sid)
        J = np.vstack([Jp[:, dofs], Jr[:, dofs]])
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
        d.qpos[qadr] = np.clip(d.qpos[qadr] + np.clip(dq, -0.2, 0.2), lo, hi)
    mujoco.mj_forward(m, d)
    return np.linalg.norm(pos - d.site_xpos[sid]) < 0.01
