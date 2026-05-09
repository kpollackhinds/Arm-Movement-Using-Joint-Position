from typing import Union

import numpy as np
import plotly.graph_objects as go
from inference.yolov11.keypoint_mapping import KEYPOINT_MAPPING

# ── Defaults ──────────────────────────────────────────────────────────────────

_AXIS_LEN   = 20   # length of the orientation axes drawn per camera
_RAY_ALPHA  = 0.4    # opacity of rays from cameras to the triangulated point
_CAM_COLOR  = "#00BFFF"
_POINT_COLOR = "#FF4500"
_AXIS_COLORS = {"x": "#FF4136", "y": "#2ECC40", "z": "#0074D9"}  # R G B


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_array(v):
    return np.array(v, dtype=float)


def _rotation_to_axes(R):
    """
    Given a 3×3 rotation matrix R whose *columns* are the camera's local
    x, y, z axes expressed in world coordinates, return those three axes.
    """
    R = _to_array(R)
    if R.shape != (3, 3):
        raise ValueError(f"rotation must be a 3×3 matrix, got shape {R.shape}")
    # Columns: R[:,0]=x-axis, R[:,1]=y-axis, R[:,2]=z-axis (forward)
    return R[:, 0], R[:, 1], R[:, 2]


def _make_camera_traces(cam_idx, position, rotation, point):
    """Return a list of Plotly traces for a single camera."""
    pos = _to_array(position)
    traces = []

    # ── Orientation axes ──────────────────────────────────────────────────────
    x_ax, y_ax, z_ax = _rotation_to_axes(rotation)
    for ax_vec, ax_name, color in [
        (x_ax, "x", _AXIS_COLORS["x"]),
        (y_ax, "y", _AXIS_COLORS["y"]),
        (z_ax, "z", _AXIS_COLORS["z"]),
    ]:
        end = pos + ax_vec * _AXIS_LEN
        traces.append(go.Scatter3d(
            x=[pos[0], end[0]],
            y=[pos[1], end[1]],
            z=[pos[2], end[2]],
            mode="lines",
            line=dict(color=color, width=8),
            name=f"cam{cam_idx} {ax_name}-axis",
            legendgroup=f"cam{cam_idx}_axes",
            showlegend=(ax_name == "x"),   # one legend entry per camera
        ))

    # ── Ray to triangulated point ─────────────────────────────────────────────
    if point is not None:
        pt = _to_array(point)
        traces.append(go.Scatter3d(
            x=[pos[0], pt[0]],
            y=[pos[1], pt[1]],
            z=[pos[2], pt[2]],
            mode="lines",
            line=dict(color=_CAM_COLOR, width=1.5, dash="dot"),
            opacity=_RAY_ALPHA,
            name=f"cam{cam_idx} ray",
            legendgroup=f"cam{cam_idx}_ray",
            showlegend=True,
        ))

    # ── Camera marker ─────────────────────────────────────────────────────────
    traces.append(go.Scatter3d(
        x=[pos[0]], y=[pos[1]], z=[pos[2]],
        mode="markers+text",
        marker=dict(size=1, color=_CAM_COLOR, symbol="square"),
        text=[f"cam{cam_idx}"],
        textposition="top center",
        textfont=dict(size=11, color=_CAM_COLOR),
        name=f"cam{cam_idx}",
        legendgroup=f"cam{cam_idx}",
        showlegend=True,
    ))

    return traces


def _make_world_frame_traces(axis_length):
    """Return traces for a world coordinate frame at the origin (RGB = XYZ)."""
    traces = []
    origin = np.zeros(3)
    axes = [
        (np.array([1, 0, 0]), "X", _AXIS_COLORS["x"]),
        (np.array([0, 1, 0]), "Y", _AXIS_COLORS["y"]),
        (np.array([0, 0, 1]), "Z", _AXIS_COLORS["z"]),
    ]
    for direction, label, color in axes:
        end = origin + direction * axis_length
        traces.append(go.Scatter3d(
            x=[0, end[0]], y=[0, end[1]], z=[0, end[2]],
            mode="lines+text",
            line=dict(color=color, width=10),
            text=["", label],
            textposition="top center",
            textfont=dict(size=16, color=color),
            name=f"world {label}",
            legendgroup="world_frame",
            showlegend=(label == "X"),
        ))
    return traces


def _make_pose_traces(keypoints_3d: Union[list, np.ndarray], keypoint_mapping: list[tuple[int, int]]):
    """Return (markers_trace, skeleton_trace) for a pose given 3-D keypoints.

    keypoints_3d : list of items, each either array-like (3,) or None.
    keypoint_mapping : list of tuple[int, int], each tuple defines a connection between two keypoints.
    """
    xs, ys, zs = [], [], []
    for kp in keypoints_3d:
        if kp is not None:
            pt = _to_array(kp)
            xs.append(float(pt[0])); ys.append(float(pt[1])); zs.append(float(pt[2]))
        else:
            xs.append(None); ys.append(None); zs.append(None)

    markers_trace = go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="markers",
        marker=dict(size=4, color=_POINT_COLOR),
        name="pose keypoints",
        showlegend=True,
    )

    sx, sy, sz = [], [], []
    for a, b in keypoint_mapping:
        if keypoints_3d[a] is not None and keypoints_3d[b] is not None:
            pa, pb = _to_array(keypoints_3d[a]), _to_array(keypoints_3d[b])
            sx += [float(pa[0]), float(pb[0]), None]
            sy += [float(pa[1]), float(pb[1]), None]
            sz += [float(pa[2]), float(pb[2]), None]

    skeleton_trace = go.Scatter3d(
        x=sx, y=sy, z=sz,
        mode="lines",
        line=dict(color=_POINT_COLOR, width=3),
        name="skeleton",
        showlegend=True,
    )

    return markers_trace, skeleton_trace


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
) -> "go.Figure | None":
    """
    Plot cameras and an optional triangulated 3-D point.

    Parameters
    ----------
    cameras : list of dict
        Each dict must have:
            "position"  — array-like (3,)      world-space origin
            "rotation"  — array-like (3, 3)    rotation matrix
                          columns = camera local x, y, z in world coords
                          (i.e. R such that p_world = R @ p_cam + t)
        Optional keys:
            "label"     — str, overrides the default "cam{i}" label

    point_3d : array-like (3,) or None
        The triangulated 3-D point to display.

    title : str
        Plot title.

    axis_length : float
        Length of the orientation axes drawn at each camera. Tune this to
        match the scale of your scene.

    show : bool
        Call fig.show() automatically (opens browser / notebook widget).

    return_fig : bool
        Return the Plotly Figure object for further customisation.

    Returns
    -------
    plotly.graph_objects.Figure  (only if return_fig=True)

    Examples
    --------
    # Minimal — cameras only
    plot_scene([
        {"position": [0, 0, 0], "rotation": np.eye(3)},
        {"position": [1, 0, 0], "rotation": np.eye(3)},
    ])

    # With triangulated point
    plot_scene(cameras, point_3d=[0.5, 0.2, 1.8])
    """
    global _AXIS_LEN
    _AXIS_LEN = axis_length

    all_traces = []

    # ── World reference frame at origin ───────────────────────────────────────
    all_traces.extend(_make_world_frame_traces(axis_length * 2))

    for i, cam in enumerate(cameras):
        label = cam.get("label", str(i))
        traces = _make_camera_traces(
            cam_idx=label,
            position=cam["position"],
            rotation=cam["rotation"],
            point=None if keypoints is not None else point_3d,
        )
        all_traces.extend(traces)

    # ── Pose keypoints / single point ─────────────────────────────────────────
    if keypoints is not None:
        markers_trace, skeleton_trace = _make_pose_traces(keypoints, KEYPOINT_MAPPING)
        all_traces.append(markers_trace)
        all_traces.append(skeleton_trace)
    elif point_3d is not None:
        pt = _to_array(point_3d)
        all_traces.append(go.Scatter3d(
            x=[pt[0]], y=[pt[1]], z=[pt[2]],
            mode="markers+text",
            marker=dict(size=1, color=_POINT_COLOR, symbol="circle"),
            text=["3D point"],
            textposition="top center",
            textfont=dict(size=12, color=_POINT_COLOR),
            name="triangulated point",
            showlegend=True,
        ))

    # ── Layout ────────────────────────────────────────────────────────────────
    fig = go.Figure(data=all_traces)
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        scene=dict(
            camera=dict(
                up=dict(x=0, y=1, z=0),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=0.5, z=-1.5),
            ),
            xaxis=dict(title="X", backgroundcolor="#111", gridcolor="#333", showbackground=True),
            yaxis=dict(title="Y", backgroundcolor="#111", gridcolor="#333", showbackground=True),
            zaxis=dict(title="Z", backgroundcolor="#111", gridcolor="#333", showbackground=True),
            bgcolor="#111111",
            aspectmode="data",
        ),
        paper_bgcolor="#1a1a1a",
        plot_bgcolor="#1a1a1a",
        font=dict(color="#cccccc"),
        legend=dict(bgcolor="#222", bordercolor="#444", borderwidth=1),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    if show:
        fig.show()

    if return_fig:
        return fig


def plot_sequence(
    cameras: list[dict],
    points_3d: list[list[np.ndarray | None] | None],
    *,
    title: str = "3D Triangulation Sequence",
    axis_length: float = _AXIS_LEN,
    start_frame: int = 0,
    trail_keypoints: list[int] | None = None,
    frame_duration: int = 33,
) -> None:
    """
    Interactive Plotly visualization with a slider to step through
    a pose skeleton frame by frame.

    Parameters
    ----------
    cameras : list of dict
        Same format as plot_scene.

    points_3d : list of (list of 17 array-like (3,)) or None
        One entry per frame. Each entry is either None (frame skipped) or a
        list of 17 [x, y, z] keypoints.

    title : str
        Plot title.

    axis_length : float
        Length of orientation axes at each camera.

    start_frame : int
        Frame number offset for labelling (so slider shows actual frame numbers).

    trail_keypoints : list[int]
        Indices of keypoints whose paths are drawn as trails. Empty list = no trails.

    frame_duration : int
        Milliseconds per frame during playback. Lower = faster (default 33 ≈ 30 FPS).
    """
    global _AXIS_LEN
    _AXIS_LEN = axis_length

    # ── Static traces: world frame + cameras ──────────────────────────────────
    cam_traces = []
    cam_traces.extend(_make_world_frame_traces(axis_length * 2))

    for cam in cameras:
        label = cam.get("label", "cam")
        pos = _to_array(cam["position"])
        cam_traces.append(go.Scatter3d(
            x=[pos[0]], y=[pos[1]], z=[pos[2]],
            mode="markers+text",
            marker=dict(size=6, color=_CAM_COLOR, symbol="square"),
            text=[label],
            textposition="top center",
            textfont=dict(size=11, color=_CAM_COLOR),
            name=label,
            showlegend=True,
        ))
        x_ax, y_ax, z_ax = _rotation_to_axes(cam["rotation"])
        for ax_vec, ax_name, color in [
            (x_ax, "x", _AXIS_COLORS["x"]),
            (y_ax, "y", _AXIS_COLORS["y"]),
            (z_ax, "z", _AXIS_COLORS["z"]),
        ]:
            end = pos + ax_vec * _AXIS_LEN
            cam_traces.append(go.Scatter3d(
                x=[pos[0], end[0]], y=[pos[1], end[1]], z=[pos[2], end[2]],
                mode="lines",
                line=dict(color=color, width=8),
                showlegend=False,
            ))

    num_cam_traces = len(cam_traces)

    # ── Initial dynamic traces (first valid frame) ────────────────────────────
    first_valid = next((f for f in points_3d if f is not None), None)

    if trail_keypoints is None:
        trail_keypoints = []

    if first_valid is not None:
        kp_trace, skel_trace = _make_pose_traces(first_valid, KEYPOINT_MAPPING)
    else:
        kp_trace = go.Scatter3d(x=[], y=[], z=[], mode="markers",
                                marker=dict(size=4, color=_POINT_COLOR),
                                name="pose keypoints", showlegend=True)
        skel_trace = go.Scatter3d(x=[], y=[], z=[], mode="lines",
                                  line=dict(color=_POINT_COLOR, width=3),
                                  name="skeleton", showlegend=True)

    trail_traces_init = [
        go.Scatter3d(
            x=[], y=[], z=[],
            mode="markers",
            marker=dict(size=3, color=_POINT_COLOR, opacity=0.3),
            name=f"trail kp{k}",
            showlegend=True,
        )
        for k in trail_keypoints
    ]

    all_traces = cam_traces + [kp_trace, skel_trace] + trail_traces_init
    fig = go.Figure(data=all_traces)

    # ── Frames for slider ─────────────────────────────────────────────────────
    # Per-keypoint trail accumulators
    trail_acc: dict[int, tuple[list, list, list]] = {k: ([], [], []) for k in trail_keypoints}

    frames = []
    for i, frame_kps in enumerate(points_3d):
        frame_num = start_frame + i

        for k in trail_keypoints:
            if frame_kps is not None and frame_kps[k] is not None:
                tp = _to_array(frame_kps[k])
                trail_acc[k][0].append(float(tp[0]))
                trail_acc[k][1].append(float(tp[1]))
                trail_acc[k][2].append(float(tp[2]))

        if frame_kps is not None:
            xs, ys, zs = [], [], []
            for kp in frame_kps:
                if kp is not None:
                    pt = _to_array(kp)
                    xs.append(float(pt[0])); ys.append(float(pt[1])); zs.append(float(pt[2]))
                else:
                    xs.append(None); ys.append(None); zs.append(None)

            sx, sy, sz = [], [], []
            for a, b in KEYPOINT_MAPPING:
                if frame_kps[a] is not None and frame_kps[b] is not None:
                    pa, pb = _to_array(frame_kps[a]), _to_array(frame_kps[b])
                    sx += [float(pa[0]), float(pb[0]), None]
                    sy += [float(pa[1]), float(pb[1]), None]
                    sz += [float(pa[2]), float(pb[2]), None]

            frame_data = [
                go.Scatter3d(x=xs, y=ys, z=zs, mode="markers",
                             marker=dict(size=4, color=_POINT_COLOR)),
                go.Scatter3d(x=sx, y=sy, z=sz, mode="lines",
                             line=dict(color=_POINT_COLOR, width=3)),
            ]
        else:
            frame_data = [
                go.Scatter3d(x=[], y=[], z=[], mode="markers",
                             marker=dict(size=4, color=_POINT_COLOR)),
                go.Scatter3d(x=[], y=[], z=[], mode="lines",
                             line=dict(color=_POINT_COLOR, width=3)),
            ]

        for k in trail_keypoints:
            xs_t, ys_t, zs_t = trail_acc[k]
            frame_data.append(go.Scatter3d(
                x=list(xs_t), y=list(ys_t), z=list(zs_t),
                mode="markers",
                marker=dict(size=3, color=_POINT_COLOR, opacity=0.3),
            ))

        trace_indices = list(range(num_cam_traces, num_cam_traces + 2 + len(trail_keypoints)))
        frames.append(go.Frame(
            data=frame_data,
            traces=trace_indices,
            name=str(frame_num),
        ))

    fig.frames = frames

    # ── Compute global bounding box for fixed axes ────────────────────────────
    all_xs, all_ys, all_zs = [], [], []
    for cam in cameras:
        pos = _to_array(cam["position"])
        all_xs.append(float(pos[0])); all_ys.append(float(pos[1])); all_zs.append(float(pos[2]))
    for frame_kps in points_3d:
        if frame_kps is not None:
            for kp in frame_kps:
                if kp is not None:
                    pt = _to_array(kp)
                    all_xs.append(float(pt[0]))
                    all_ys.append(float(pt[1]))
                    all_zs.append(float(pt[2]))

    pad = axis_length
    x_range = [min(all_xs) - pad, max(all_xs) + pad]
    y_range = [min(all_ys) - pad, max(all_ys) + pad]
    z_range = [min(all_zs) - pad, max(all_zs) + pad]

    # ── Slider + layout ───────────────────────────────────────────────────────
    sliders = [dict(
        active=0,
        currentvalue=dict(prefix="Frame: ", font=dict(size=14)),
        pad=dict(b=10, t=40),
        steps=[
            dict(
                args=[[str(start_frame + i)], dict(frame=dict(duration=0, redraw=True), mode="immediate")],
                label=str(start_frame + i),
                method="animate",
            )
            for i in range(len(points_3d))
        ],
    )]

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        sliders=sliders,
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=1.15,
            x=0.5,
            xanchor="center",
            buttons=[
                dict(label="Play", method="animate",
                     args=[None, dict(frame=dict(duration=frame_duration, redraw=False),
                                     transition=dict(duration=0),
                                     fromcurrent=True)]),
                dict(label="Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                       transition=dict(duration=0),
                                       mode="immediate")]),
            ],
        )],
        scene=dict(
            camera=dict(
                up=dict(x=0, y=1, z=0),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=0.5, z=-1.5),
            ),
            xaxis=dict(title="X", range=x_range, backgroundcolor="#111", gridcolor="#333", showbackground=True),
            yaxis=dict(title="Y", range=y_range, backgroundcolor="#111", gridcolor="#333", showbackground=True),
            zaxis=dict(title="Z", range=z_range, backgroundcolor="#111", gridcolor="#333", showbackground=True),
            bgcolor="#111111",
            aspectmode="cube",
        ),
        paper_bgcolor="#1a1a1a",
        plot_bgcolor="#1a1a1a",
        font=dict(color="#cccccc"),
        legend=dict(bgcolor="#222", bordercolor="#444", borderwidth=1),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    fig.show()


# ── Quick sanity-check scene (run this file directly) ────────────────────────

if __name__ == "__main__":
    def _Ry(deg):
        """Rotation matrix around Y axis."""
        t = np.radians(deg)
        return np.array([
            [ np.cos(t), 0, np.sin(t)],
            [         0, 1,         0],
            [-np.sin(t), 0, np.cos(t)],
        ])

    def _Rx(deg):
        t = np.radians(deg)
        return np.array([
            [1,         0,          0],
            [0, np.cos(t), -np.sin(t)],
            [0, np.sin(t),  np.cos(t)],
        ])

    # Three cameras arranged in an arc, all loosely pointing toward the origin
    demo_cameras = [
        {
            "label": "left",
            "position": [-1.0, 0.2, 0.5],
            "rotation": _Ry(-30),
        },
        {
            "label": "center",
            "position": [0.0, 0.3, -1.2],
            "rotation": np.eye(3),
        },
        {
            "label": "right",
            "position": [1.0, 0.2, 0.5],
            "rotation": _Ry(30),
        },
    ]

    demo_point = [0.05, 0.1, 0.0]

    plot_scene(
        demo_cameras,
        point_3d=demo_point,
        title="Demo — 3 cameras + triangulated point",
        axis_length=0.2,
    )
