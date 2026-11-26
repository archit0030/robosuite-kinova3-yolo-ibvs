"""Pinhole camera model and rigid-body transforms for MuJoCo cameras.

Frames use the OpenCV convention: x right, y down, z forward. MuJoCo cameras
are x right, y up, z backward, hence the diag(1, -1, -1) correction.
"""
import numpy as np

_MJ_TO_CV = np.diag([1.0, -1.0, -1.0])


def make_T(R, p):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def inv_T(T):
    R, p = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ p)


def transform(T, pts):
    """Apply a 4x4 transform to an (3,) or (N, 3) array of points."""
    pts = np.asarray(pts, dtype=float)
    return pts @ T[:3, :3].T + T[:3, 3]


class Camera:
    def __init__(self, sim, name, width=640, height=480):
        self.sim, self.name, self.width, self.height = sim, name, width, height
        self.id = sim.model.camera_name2id(name)
        fovy = np.deg2rad(sim.model.cam_fovy[self.id])
        f = 0.5 * height / np.tan(0.5 * fovy)
        self.K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]])
        extent = sim.model.stat.extent
        self._near = sim.model.vis.map.znear * extent
        self._far = sim.model.vis.map.zfar * extent

    @property
    def T_world_cam(self):
        R = np.array(self.sim.data.cam_xmat[self.id]).reshape(3, 3) @ _MJ_TO_CV
        return make_T(R, np.array(self.sim.data.cam_xpos[self.id]))

    def capture(self):
        """Return (rgb uint8 HxWx3, metric depth HxW), top row first."""
        rgb, d = self.sim.render(
            camera_name=self.name, width=self.width, height=self.height, depth=True
        )
        rgb, d = rgb[::-1].copy(), d[::-1]
        depth = self._near / (1.0 - d * (1.0 - self._near / self._far))
        return rgb, depth

    def project(self, p_cam):
        """Camera-frame point(s) -> pixel(s)."""
        p = np.asarray(p_cam, dtype=float)
        uv = p @ self.K.T
        return uv[..., :2] / uv[..., 2:3]

    def deproject(self, uv, z):
        """Pixel(s) and metric depth -> camera-frame point(s)."""
        uv, z = np.asarray(uv, dtype=float), np.asarray(z, dtype=float)
        x = (uv[..., 0] - self.K[0, 2]) / self.K[0, 0] * z
        y = (uv[..., 1] - self.K[1, 2]) / self.K[1, 1] * z
        return np.stack([x, y, z], axis=-1)
