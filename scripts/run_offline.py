import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CALIBRATION_DIR, LANDMARKS_DIR
import numpy as np
import csv
import cv2
from utils_3D.core.triangulation import triangulate
from utils_3D.core.camera import Camera
from utils_3D.core.projection import get_camera_center_from_projection_matrix, undistort_points, decompose_projection_matrix
from utils_3D.visualization.plotting import plot_sequence
import pickle
from OneEuroFilter import OneEuroFilter

# First we need to get out our array of 2d point correspondences (N,2)
# We will have M points, but that just means we triangulate M times, so we can just do it in a loop.
# First we need to assemble our array of 2d points in 2 x N format
# nose - kp1
# right shoulder - kp7
# right elbow - kp9
# right wrist - kp11
#

NUM_KEYPOINTS = 17


def parse_landmarks_csv(filepath, start_frame, end_frame):
    """Parse a YOLO pose landmarks CSV between start_frame and end_frame.
    Returns one entry per frame: None if any keypoint is 'null',
    otherwise a list of NUM_KEYPOINTS [x, y] pairs.
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
                        [float(col.split(",")[0][3:]), float(col.split(",")[1][4:])]
                        for col in kp_cols
                    ])
            count += 1
    return points

def filter(point: Point3D, timestamp: float) -> Point3D:
    """
    Apply a One Euro Filter to the 3D point to smooth it over time.
    """
    x = euro_filer(point.x, timestamp)
    y = euro_filer(point.y, timestamp)
    z = euro_filer(point.z, timestamp)
    return Point3D(x, y, z)

# Need to impot our camera matrices.
cam_1 = Camera("cam_1", 
               projection_matrix=pickle.load(open(f"{CALIBRATION_DIR}/cam1/projection_matrix.pkl", "rb")),
               camera_matrix= pickle.load(open(f"{CALIBRATION_DIR}/cam1/cameraMatrix.pkl", "rb")),
               distortion_coefficients=pickle.load(open(f"{CALIBRATION_DIR}/cam1/dist.pkl", "rb"))
               )
cam_2 = Camera("cam_2", 
               projection_matrix=pickle.load(open(f"{CALIBRATION_DIR}/cam2/projection_matrix.pkl", "rb")),
               camera_matrix= pickle.load(open(f"{CALIBRATION_DIR}/cam2/cameraMatrix.pkl", "rb")),
               distortion_coefficients=pickle.load(open(f"{CALIBRATION_DIR}/cam2/dist.pkl", "rb"))
               )
cam_3 = Camera("cam_3", 
               projection_matrix=pickle.load(open(f"{CALIBRATION_DIR}/cam3/projection_matrix.pkl", "rb")),
               camera_matrix= pickle.load(open(f"{CALIBRATION_DIR}/cam3/cameraMatrix.pkl", "rb")),
               distortion_coefficients=pickle.load(open(f"{CALIBRATION_DIR}/cam3/dist.pkl", "rb"))
               )
    
files = [
    os.path.join(LANDMARKS_DIR, "cam_1_pose_landmarks2.csv"),
    os.path.join(LANDMARKS_DIR, "cam_2_pose_landmarks2.csv"),
    os.path.join(LANDMARKS_DIR, "cam_3_pose_landmarks2.csv"),
]

start_frame = 600
end_frame = 1800

cam1_points = parse_landmarks_csv(files[0], start_frame, end_frame)
cam2_points = parse_landmarks_csv(files[1], start_frame, end_frame)
cam3_points = parse_landmarks_csv(files[2], start_frame, end_frame)

# Triangluate points
# points_array = np.array([cam1_points[t][0], cam2_points[t][0], cam3_points[t][0]])
projection_matrices = np.array([cam_1.projection_matrix, cam_2.projection_matrix, cam_3.projection_matrix])
# print(points_array.shape)
frame_count = end_frame - start_frame
index = 3

triangulated_points = []
euro_filer = OneEuroFilter(freq=30, mincutoff=1.0, beta=0.0)  # Adjust parameters as needed
for t in range(frame_count):
    if cam1_points[t] is None or cam2_points[t] is None or cam3_points[t] is None:
        print(f"Skipping frame {t}: missing keypoints in one or more cameras")
        triangulated_points.append(None)
        continue

    frame_3d = []
    for kp_idx in range(NUM_KEYPOINTS):
        kp_array = np.array([cam1_points[t][kp_idx], cam2_points[t][kp_idx], cam3_points[t][kp_idx]])
        cam1_kp_undistorted = undistort_points(kp_array[0].reshape(1, 2), cam_1.camera_matrix, cam_1.distortion_coefficients)
        cam2_kp_undistorted = undistort_points(kp_array[1].reshape(1, 2), cam_2.camera_matrix, cam_2.distortion_coefficients)
        cam3_kp_undistorted = undistort_points(kp_array[2].reshape(1, 2), cam_3.camera_matrix, cam_3.distortion_coefficients)
        kp_undistorted = np.array([cam1_kp_undistorted[0], cam2_kp_undistorted[0], cam3_kp_undistorted[0]])

        try:
            point_3d = triangulate(kp_undistorted, projection_matrices)
            filtered_point_3d = filter(point_3d, timestamp=t/30)  # Assuming 30 FPS
            frame_3d.append(point_3d)
        except Exception as e:
            print(f"Error triangulating keypoint {kp_idx} at frame {t}")
            print(f"Exception: {e}")
            frame_3d.append(None)

    print(f"Triangulated frame {t}: kp{index}={frame_3d[index]}")
    triangulated_points.append(frame_3d)

# Build camera dicts for visualization
cameras_vis = []
for cam in [cam_1, cam_2, cam_3]:
    if cam.projection_matrix:
        K, R, center = decompose_projection_matrix(cam.projection_matrix)
    # R is world-to-camera; R.T columns are camera's local axes in world coords
        cameras_vis.append({"label": cam.name, "position": center, "rotation": R.T})

if triangulated_points:
    plot_sequence(
        cameras_vis,
        triangulated_points,
        title=f"Triangulation — frames {start_frame} — {end_frame-1}",
        start_frame=start_frame,
    )

