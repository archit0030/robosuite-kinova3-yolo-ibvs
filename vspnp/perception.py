"""YOLO detection plus depth refinement to a 3D point on the can's top face."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from .camera import inv_T, transform
from .env import BIN_FLOOR_Z

ROOT = Path(__file__).resolve().parents[1]

# Heights above the bin floor in which a point can belong to a standing can.
Z_BAND = (0.03, 0.13)
TOP_THICKNESS = 0.008
MIN_TOP_PIXELS = 25


@dataclass
class Observation:
    box: np.ndarray       # x0, y0, x1, y1 in pixels
    conf: float
    uv: np.ndarray        # pixel of the top-face centre
    z_cam: float          # depth of the top-face centre
    p_world: np.ndarray   # top-face centre in the world frame


class CanDetector:
    def __init__(self, weights=ROOT / "models" / "can_detector.pt", conf=0.5):
        self.model = YOLO(str(weights))
        self.conf = conf
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.detect(np.zeros((480, 640, 3), np.uint8))  # warm up

    def detect(self, rgb):
        """Return an (N, 5) array of x0, y0, x1, y1, conf, best first."""
        r = self.model.predict(rgb[..., ::-1], conf=self.conf, imgsz=640,
                               device=self.device, verbose=False)[0]
        out = r.boxes.data.cpu().numpy()[:, :5]
        return out[np.argsort(-out[:, 4])]

    def observe(self, camera, rgb, depth, near_uv=None, gate=None, z_band=None):
        """Detect the can and localise its top face. None if not found.

        near_uv/gate associate the detection with an existing track. z_band
        overrides the world heights in which the can's top may lie.
        """
        boxes = self.detect(rgb)
        obs = [o for o in (self._refine(camera, depth, b, z_band) for b in boxes) if o is not None]
        if near_uv is not None:
            obs = [o for o in obs if np.linalg.norm(o.uv - near_uv) < gate]
            obs.sort(key=lambda o: np.linalg.norm(o.uv - near_uv))
        return obs[0] if obs else None

    @staticmethod
    def _refine(camera, depth, b, z_band=None):
        x0, y0 = np.maximum(np.floor(b[:2]).astype(int), 0)
        x1, y1 = np.ceil(b[2:4]).astype(int)
        roi = depth[y0:y1, x0:x1]
        if roi.size == 0:
            return None
        v, u = np.mgrid[y0:y0 + roi.shape[0], x0:x0 + roi.shape[1]]
        uv = np.stack([u.ravel() + 0.5, v.ravel() + 0.5], axis=-1)
        T = camera.T_world_cam
        pts = transform(T, camera.deproject(uv, roi.ravel()))
        lo, hi = z_band if z_band else (BIN_FLOOR_Z + Z_BAND[0], BIN_FLOOR_Z + Z_BAND[1])
        pts = pts[(pts[:, 2] > lo) & (pts[:, 2] < hi)]
        if len(pts) < MIN_TOP_PIXELS:
            return None
        top = np.percentile(pts[:, 2], 95)
        face = pts[np.abs(pts[:, 2] - top) < TOP_THICKNESS]
        if len(face) < MIN_TOP_PIXELS:
            return None
        # Midpoint of robust extents: insensitive to uneven pixel density.
        lo, hi = np.percentile(face[:, :2], [2, 98], axis=0)
        c = np.array([*(lo + hi) / 2, np.median(face[:, 2])])
        p_cam = transform(inv_T(T), c)
        return Observation(b[:4].copy(), float(b[4]), camera.project(p_cam), float(p_cam[2]), c)
