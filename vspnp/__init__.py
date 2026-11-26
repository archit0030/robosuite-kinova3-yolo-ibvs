# Rendering gotcha (macOS, MuJoCo 3.3 + robosuite 1.5.2):
# if torch/ultralytics are loaded, numba compiles, or a YOLO model runs for the
# first time AFTER MuJoCo has created its OpenGL context, every frame rendered
# from then on is noise or black. A hard env.reset() builds a new context and
# fixes it. So: torch and ultralytics are imported here, make_env() warms up
# numba before handing the env over, and the YOLO detector is created before
# the env (03_run_pick_place.py also resets at the start of every episode).
import os
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parents[1] / ".ultralytics"))

import torch  # noqa: E402,F401
import ultralytics  # noqa: E402,F401
