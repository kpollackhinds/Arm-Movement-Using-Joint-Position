import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CALIBRATION_DIR, LANDMARKS_DIR, CAMERA_INDICES
import numpy as np
import csv
import cv2
from utils_3D.core.triangulation import triangulate
from utils_3D.core.camera import Camera
from utils_3D.core.projection import get_camera_center_from_projection_matrix, undistort_points, decompose_projection_matrix
from utils_3D.visualization.plotting import plot_sequence
from utils_3D.core.point import Point3D
import pickle
from OneEuroFilter import OneEuroFilter

NUM_KEYPOINTS = 17
NUM_CAMS = len(CAMERA_INDICES)
USE_CONFIDENCE_WEIGHTS = True


def parse_landmarks_csv(filepath, start_frame, end_frame):
    """Parse a YOLO pose landmarks CSV between start_frame and end_frame.
    Returns one entry per frame: None if any keypoint is 'null',
    otherwise a list of NUM_KEYPOINTS [x, y, conf] triples.
    """
    points = []
    count = 0
    with open(filepath, mode="r") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for line in reader:
            if count >= end_frame:
                break
            if count >= start_frame:
                kp_cols = [line[i + 1] for i in range(NUM_KEYPOINTS)]
                if "null" in kp_cols:
                    points.append(None)
                else:
                    points.append([
                        [float(col.split(",")[0][3:]), float(col.split(",")[1][4:]), float(col.split(",")[2][7:])]
                        for col in kp_cols
                    ])
            count += 1
    return points


def filter(point: Point3D, timestamp: float, filters: list) -> Point3D:
    """
    Apply a One Euro Filter to the 3D point to smooth it over time.
    filters: [filter_x, filter_y, filter_z]
    """
    x = filters[0](point.x, timestamp)
    y = filters[1](point.y, timestamp)
    z = filters[2](point.z, timestamp)
    return Point3D(x, y, z)


def main(start_frame: int, end_frame: int):
    # Load cameras dynamically from config
    cameras = []
    for i in range(1, NUM_CAMS + 1):
        cam_dir = f"{CALIBRATION_DIR}/cam{i}"
        cameras.append(Camera(
            f"cam_{i}",
            projection_matrix=pickle.load(open(f"{cam_dir}/projection_matrix.pkl", "rb")),
            camera_matrix=pickle.load(open(f"{cam_dir}/cameraMatrix.pkl", "rb")),
            distortion_coefficients=pickle.load(open(f"{cam_dir}/dist.pkl", "rb"))
        ))

    # Load landmark CSV files dynamically
    files = [
        os.path.join(LANDMARKS_DIR, f"cam_{i}_pose_landmarks2.csv")
        for i in range(1, NUM_CAMS + 1)
    ]

    # Finding max frame count
    for f in files:
        with open(f, mode="r") as file:
            csvFile = csv.reader(file)
            next(csvFile, None)  # skip header
            frame_count = sum(1 for _ in csvFile)
    max_frames = frame_count

    if not end_frame:
        end_frame = max_frames
    else:
        end_frame = min(end_frame, max_frames)

    all_cam_points = [
        parse_landmarks_csv(f, start_frame, end_frame)
        for f in files
    ]

    projection_matrices = np.array([cam.projection_matrix for cam in cameras])
    frame_count = end_frame - start_frame
    index = 3

    triangulated_points = []
    # One filter per axis per keypoint so each signal has independent state
    euro_filters = [
        [OneEuroFilter(freq=30, mincutoff=1.0, beta=0.0) for _ in range(3)]
        for _ in range(NUM_KEYPOINTS)
    ]

    for t in range(frame_count):
        if any(all_cam_points[c][t] is None for c in range(NUM_CAMS)):
            missing = [i + 1 for i in range(NUM_CAMS) if all_cam_points[i][t] is None]
            print(f"Skipping frame {t}: missing keypoints in camera(s) {missing}")
            triangulated_points.append(None)
            continue

        frame_3d = []
        for kp_idx in range(NUM_KEYPOINTS):
            kps = [all_cam_points[c][t][kp_idx] for c in range(NUM_CAMS)]  # [x, y, conf] per camera

            weights = np.array([kp[2] for kp in kps]) if USE_CONFIDENCE_WEIGHTS else None

            kp_undistorted = np.array([
                undistort_points(np.array([[kp[0], kp[1]]]), cameras[c].camera_matrix, cameras[c].distortion_coefficients)[0]
                for c, kp in enumerate(kps)
            ])

            try:
                point_3d = triangulate(kp_undistorted, projection_matrices, weights=weights)
                filtered_point_3d = filter(point_3d, timestamp=t / 30, filters=euro_filters[kp_idx])
                frame_3d.append(filtered_point_3d)
            except Exception as e:
                print(f"Error triangulating keypoint {kp_idx} at frame {t}: {e}")
                frame_3d.append(None)

        print(f"Triangulated frame {t}: kp{index}={frame_3d[index]}")
        triangulated_points.append(frame_3d)

    # Build camera dicts for visualization
    cameras_vis = []
    for cam in cameras:
        K, R, center = decompose_projection_matrix(cam.projection_matrix)
        # R is world-to-camera; R.T columns are camera's local axes in world coords
        cameras_vis.append({"label": cam.name, "position": center, "rotation": R.T})

    if triangulated_points:
        plot_sequence(
            cameras_vis,
            triangulated_points,
            title=f"Triangulation — frames {start_frame} — {end_frame - 1}",
            start_frame=start_frame,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run offline triangulation and visualization.")
    parser.add_argument("--start_frame", type=int, default=0, help="Frame of video to start from (inclusive)")
    parser.add_argument("--end_frame", type=int, default=None, help="Frame of video to end at (exclusive)")
    args = parser.parse_args()

    main(args.start_frame, args.end_frame)
    pass

