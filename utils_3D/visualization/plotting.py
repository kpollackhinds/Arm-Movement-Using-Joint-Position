"""3-D pose visualisation using Rerun (https://rerun.io/).

Public API is compatible with plotting_plotly.py:
    plot_scene(cameras, point_3d, *, keypoints, title, axis_length, show, return_fig)
    plot_sequence(cameras, points_3d, *, title, axis_length, start_frame,
                  trail_keypoints, frame_duration)

Install:  pip install rerun-sdk
"""

from typing import Union

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from inference.yolov11.keypoint_mapping import KEYPOINT_MAPPING

# ── Defaults ──────────────────────────────────────────────────────────────────

_AXIS_LEN    = 20
_CAM_COLOR   = [0, 191, 255]    # #00BFFF
_POINT_COLOR = [255, 69, 0]     # #FF4500
_AXIS_COLORS = {
    "x": [255, 65, 54],         # #FF4136
    "y": [46, 204, 64],         # #2ECC40
    "z": [0, 116, 217],         # #0074D9
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_array(v) -> np.ndarray:
    return np.array(v, dtype=float)


def _rotation_to_axes(R) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    R = _to_array(R)
    if R.shape != (3, 3):
        raise ValueError(f"rotation must be a 3×3 matrix, got shape {R.shape}")
    return R[:, 0], R[:, 1], R[:, 2]


def _log_world_frame(axis_length: float) -> None:
    """Log RGB XYZ world-frame arrows at the origin as static (timeless) data."""
    for vec, name, color in [
        (np.array([1.0, 0.0, 0.0]), "x", _AXIS_COLORS["x"]),
        (np.array([0.0, 1.0, 0.0]), "y", _AXIS_COLORS["y"]),
        (np.array([0.0, 0.0, 1.0]), "z", _AXIS_COLORS["z"]),
    ]:
        rr.log(
            f"world/{name}_axis",
            rr.Arrows3D(origins=[[0.0, 0.0, 0.0]], vectors=[vec * axis_length], colors=[color]),
            static=True,
        )


def _log_cameras(cameras: list[dict], axis_length: float) -> None:
    """Log camera positions and orientation axes as static (timeless) data."""
    for cam in cameras:
        label = cam.get("label", "cam")
        pos = _to_array(cam["position"])
        x_ax, y_ax, z_ax = _rotation_to_axes(cam["rotation"])
        path = f"cameras/{label}"

        rr.log(f"{path}/position", rr.Points3D([pos], colors=[_CAM_COLOR]), static=True)

        for ax_vec, ax_name, color in [
            (x_ax, "x", _AXIS_COLORS["x"]),
            (y_ax, "y", _AXIS_COLORS["y"]),
            (z_ax, "z", _AXIS_COLORS["z"]),
        ]:
            rr.log(
                f"{path}/{ax_name}_axis",
                rr.Arrows3D(origins=[pos], vectors=[ax_vec * axis_length], colors=[color]),
                static=True,
            )


def _log_pose(
    keypoints_3d: list,
    entity_prefix: str = "pose",
    static: bool = False,
) -> None:
    """Log keypoint markers and skeleton line strips at the current time."""
    valid_pts = [_to_array(kp) for kp in keypoints_3d if kp is not None]
    if valid_pts:
        rr.log(
            f"{entity_prefix}/keypoints",
            rr.Points3D(valid_pts, colors=[_POINT_COLOR]),
            static=static,
        )

    strips = []
    for a, b in KEYPOINT_MAPPING:
        if keypoints_3d[a] is not None and keypoints_3d[b] is not None:
            strips.append([_to_array(keypoints_3d[a]), _to_array(keypoints_3d[b])])
    if strips:
        rr.log(
            f"{entity_prefix}/skeleton",
            rr.LineStrips3D(strips, colors=[_POINT_COLOR]),
            static=static,
        )


# ── Public API ────────────────────────────────────────────────────────────────

def plot_scene(
    cameras: list[dict],
    point_3d: np.ndarray | list | None = None,
    *,
    keypoints: list[np.ndarray | None] | None = None,
    title: str = "3D Triangulation Scene",
    axis_length: float = _AXIS_LEN,
    show: bool = True,
    return_fig: bool = False,
) -> None:
    """
    Plot cameras and an optional triangulated 3-D point using Rerun.

    Parameters
    ----------
    cameras : list of dict
        Each dict must have "position" (array-like (3,)) and "rotation" (3×3 matrix).
        Columns of rotation = camera local x, y, z in world coords.
        Optional key: "label" (str).
    point_3d : array-like (3,) or None
        A single triangulated point to display.
    keypoints : list of array-like (3,) or None, optional
        Pose keypoints; takes priority over point_3d.
    title : str
        Rerun application / recording name (shown in the viewer title bar).
    axis_length : float
        Length of orientation axes at each camera and the world frame.
    show : bool
        Spawn the Rerun viewer automatically.
    return_fig : bool
        Not used (Rerun has no figure object). Kept for API compatibility.
    """
    rr.init(title)
    if show:
        rr.spawn()

    _log_world_frame(axis_length * 2)
    _log_cameras(cameras, axis_length)

    if keypoints is not None:
        _log_pose(keypoints, static=True)
    elif point_3d is not None:
        pt = _to_array(point_3d)
        rr.log("pose/point", rr.Points3D([pt], colors=[_POINT_COLOR]), static=True)


def plot_sequence(
    cameras: list[dict],
    points_3d: list[list[np.ndarray | None] | None],
    *,
    raw_points_3d: list[list[np.ndarray | None] | None] | None = None,
    title: str = "3D Triangulation Sequence",
    axis_length: float = _AXIS_LEN,
    start_frame: int = 0,
    trail_keypoints: list[int] | None = None,
    frame_duration: int = 33,
) -> None:
    """
    Visualise a pose sequence frame by frame using Rerun.

    The Rerun viewer's timeline is stamped with real wall-clock timestamps
    derived from frame_duration so that 1× playback matches the original
    video speed (e.g. frame_duration=33 ms → ~30 FPS).

    Parameters
    ----------
    cameras : list of dict
        Same format as plot_scene.
    points_3d : list of (list of 17 array-like (3,)) or None
        The primary (e.g. filtered) keypoints. One entry per frame.
    raw_points_3d : list of (list of 17 array-like (3,)) or None, optional
        If provided, enables side-by-side comparison mode. These are logged
        as the baseline/raw skeleton alongside points_3d (the filtered one).
    title : str
        Rerun application / recording name.
    axis_length : float
        Length of orientation axes at each camera.
    start_frame : int
        Frame number offset for the timeline labels.
    trail_keypoints : list[int] or None
        Indices of keypoints whose accumulating paths are drawn as trails.
        Empty list / None = no trails.
    frame_duration : int
        Milliseconds between frames. Used to stamp real timestamps so Rerun
        plays back at the correct speed. Default 33 ≈ 30 FPS.
    """
    if trail_keypoints is None:
        trail_keypoints = []

    comparison_mode = raw_points_3d is not None
    fps = 1000.0 / frame_duration

    rr.init(title)

    # ── Blueprint: side-by-side when comparison mode is on ────────────────────
    static_contents = ["world/**", "cameras/**"]
    if comparison_mode:
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    name="Raw",
                    origin="/",
                    contents=static_contents + ["pose/raw/**"],
                ),
                rrb.Spatial3DView(
                    name="Filtered",
                    origin="/",
                    contents=static_contents + ["pose/filtered/**"],
                ),
            )
        )
        rr.send_blueprint(blueprint)

    rr.spawn()

    # Static geometry: world frame + cameras
    _log_world_frame(axis_length * 2)
    _log_cameras(cameras, axis_length)

    # Per-keypoint trail accumulators (one set per view in comparison mode)
    trail_acc: dict[int, list[np.ndarray]] = {k: [] for k in trail_keypoints}
    raw_trail_acc: dict[int, list[np.ndarray]] = {k: [] for k in trail_keypoints}

    for i, frame_kps in enumerate(points_3d):
        frame_num = start_frame + i
        raw_kps = raw_points_3d[i] if comparison_mode else None

        # Stamp both a sequence index and a real-time timestamp
        rr.set_time("frame", sequence=frame_num)
        rr.set_time("time", duration=frame_num / fps)

        # Choose entity prefix depending on mode
        prefix = "pose/filtered" if comparison_mode else "pose"

        # Accumulate and log trails for the primary (filtered) skeleton
        for k in trail_keypoints:
            if frame_kps is not None and frame_kps[k] is not None:
                trail_acc[k].append(_to_array(frame_kps[k]))
            if trail_acc[k]:
                rr.log(
                    f"{prefix}/trail/kp{k}",
                    rr.Points3D(trail_acc[k], colors=[[*_POINT_COLOR, 77]]),
                )

        if frame_kps is not None:
            _log_pose(frame_kps, entity_prefix=prefix)
        else:
            rr.log(f"{prefix}/keypoints", rr.Clear(recursive=False))
            rr.log(f"{prefix}/skeleton", rr.Clear(recursive=False))

        # ── Raw skeleton (comparison mode only) ───────────────────────────────
        if comparison_mode:
            for k in trail_keypoints:
                if raw_kps is not None and raw_kps[k] is not None:
                    raw_trail_acc[k].append(_to_array(raw_kps[k]))
                if raw_trail_acc[k]:
                    rr.log(
                        f"pose/raw/trail/kp{k}",
                        rr.Points3D(raw_trail_acc[k], colors=[[*_POINT_COLOR, 77]]),
                    )

            if raw_kps is not None:
                _log_pose(raw_kps, entity_prefix="pose/raw")
            else:
                rr.log("pose/raw/keypoints", rr.Clear(recursive=False))
                rr.log("pose/raw/skeleton", rr.Clear(recursive=False))