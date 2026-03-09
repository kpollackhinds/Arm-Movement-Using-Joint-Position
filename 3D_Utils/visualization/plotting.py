import numpy as np
import plotly.graph_objects as go

# ── Defaults ──────────────────────────────────────────────────────────────────

_AXIS_LEN   = 0.15   # length of the orientation axes drawn per camera
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
            line=dict(color=color, width=4),
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
        marker=dict(size=6, color=_CAM_COLOR, symbol="square"),
        text=[f"cam{cam_idx}"],
        textposition="top center",
        textfont=dict(size=11, color=_CAM_COLOR),
        name=f"cam{cam_idx}",
        legendgroup=f"cam{cam_idx}",
        showlegend=True,
    ))

    return traces


# ── Public API ────────────────────────────────────────────────────────────────

def plot_scene(
    cameras,
    point_3d=None,
    *,
    title="3D Triangulation Scene",
    axis_length=_AXIS_LEN,
    show=True,
    return_fig=False,
):
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

    for i, cam in enumerate(cameras):
        label = cam.get("label", str(i))
        traces = _make_camera_traces(
            cam_idx=label,
            position=cam["position"],
            rotation=cam["rotation"],
            point=point_3d,
        )
        all_traces.extend(traces)

    # ── Triangulated point ────────────────────────────────────────────────────
    if point_3d is not None:
        pt = _to_array(point_3d)
        all_traces.append(go.Scatter3d(
            x=[pt[0]], y=[pt[1]], z=[pt[2]],
            mode="markers+text",
            marker=dict(size=9, color=_POINT_COLOR, symbol="diamond"),
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
            xaxis=dict(title="X", backgroundcolor="#111", gridcolor="#333", showbackground=True),
            yaxis=dict(title="Y", backgroundcolor="#111", gridcolor="#333", showbackground=True),
            zaxis=dict(title="Z", backgroundcolor="#111", gridcolor="#333", showbackground=True),
            bgcolor="#111111",
            aspectmode="data",   # preserves true proportions — important for sanity-checking
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
