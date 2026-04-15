import sys

import numpy as np
import csv
import cv2
from utils_3D.core.triangulation import triangulate
from utils_3D.core.camera import Camera
from utils_3D.core.projection import get_camera_center_from_projection_matrix, undistort_points, decompose_projection_matrix
from utils_3D.visualization.plotting import plot_sequence
import pickle

# First we need to get out our array of 2d point correspondences (N,2)
# We will have M points, but that just means we triangulate M times, so we can just do it in a loop.
# First we need to assemble our array of 2d points in 2 x N format
# nose - kp1
# right shoulder - kp7
# right elbow - kp9
# right wrist - kp11
#

# Need to impot our camera matrices.
cam_1 = Camera("cam_1", 
               projection_matrix=pickle.load(open("Camera_Calibration\\cam1\\projection_matrix.pkl", "rb")),
               camera_matrix= pickle.load(open("Camera_Calibration\\cam1\\cameraMatrix.pkl", "rb")),
               distortion_coefficients=pickle.load(open("Camera_Calibration\\cam1\\dist.pkl", "rb"))
               )
cam_2 = Camera("cam_2", 
               projection_matrix=pickle.load(open("Camera_Calibration\\cam2\\projection_matrix.pkl", "rb")),
               camera_matrix= pickle.load(open("Camera_Calibration\\cam2\\cameraMatrix.pkl", "rb")),
               distortion_coefficients=pickle.load(open("Camera_Calibration\\cam2\\dist.pkl", "rb"))
               )
cam_3 = Camera("cam_3", 
               projection_matrix=pickle.load(open("Camera_Calibration\\cam3\\projection_matrix.pkl", "rb")),
               camera_matrix= pickle.load(open("Camera_Calibration\\cam3\\cameraMatrix.pkl", "rb")),
               distortion_coefficients=pickle.load(open("Camera_Calibration\\cam3\\dist.pkl", "rb"))
               )
    
cam1_points = []
cam2_points = []
cam3_points = []

files = [
    "cam_1_pose_landmarks2.csv",
    "cam_2_pose_landmarks2.csv",
    "cam_3_pose_landmarks2.csv",
]


# centers = [
#     get_camera_center_from_projection_matrix(cam_1.projection_matrix),
#     get_camera_center_from_projection_matrix(cam_2.projection_matrix),
#     get_camera_center_from_projection_matrix(cam_3.projection_matrix),
# ]
# centers = np.array(centers)

# v1 = centers[1] - centers[0]
# v2 = centers[2] - centers[0]

# cross_product = np.cross(v1, v2)
# print("Cross product of v1 and v2:", cross_product)
# # spread is cross product magnitude, which gives us an idea of how well the cameras are spread out in space. A larger spread indicates better triangulation potential.
# spread = np.linalg.norm(cross_product)
# print("Spread of the cameras:", spread)

# print("Camera centers in world coordinates:")
# for i, center in enumerate(centers):
#     print(f"Camera {i+1} center: {center}")

# sys.exit(0)

count = 0
start_frame = 600
end_frame = 1800
# cam1
with open(files[0], mode="r") as file:
    csvFile = csv.reader(file)
    # read header
    next(csvFile, None)

    # Points ends up being a list of 4 points, representing the 4 keypoints at a given frame at time (t).
    for line in csvFile:
        if count >= end_frame:
            break
        if count >= start_frame:
            if "null" in (line[1], line[7], line[9], line[11]):
                cam1_points.append(None)
            else:
                points = [
                    [float(line[1].split(",")[0][3:]), float(line[1].split(",")[1][4:])],
                    [float(line[7].split(",")[0][3:]), float(line[7].split(",")[1][4:])],
                    [float(line[9].split(",")[0][3:]), float(line[9].split(",")[1][4:])],
                    [float(line[11].split(",")[0][3:]), float(line[11].split(",")[1][4:])],
                ]
                cam1_points.append(points)
        count += 1
        # print(points)

count = 0
# cam2
with open(files[1], mode="r") as file:
    csvFile = csv.reader(file)
    # read header
    next(csvFile, None)

    for line in csvFile:
        if count >= end_frame:
            break
        if count >= start_frame:
            if "null" in (line[1], line[7], line[9], line[11]):
                cam2_points.append(None)
            else:
                points = [
                    [float(line[1].split(",")[0][3:]), float(line[1].split(",")[1][4:])],
                    [float(line[7].split(",")[0][3:]), float(line[7].split(",")[1][4:])],
                    [float(line[9].split(",")[0][3:]), float(line[9].split(",")[1][4:])],
                    [float(line[11].split(",")[0][3:]), float(line[11].split(",")[1][4:])],
                ]
                cam2_points.append(points)
        count += 1
        # print(points)

count = 0
# cam3
with open(files[2], mode="r") as file:
    csvFile = csv.reader(file)
    # read header
    next(csvFile, None)

    for line in csvFile:
        if count >= end_frame:
            break
        if count >= start_frame:
            if "null" in (line[1], line[7], line[9], line[11]):
                cam3_points.append(None)
            else:
                points = [
                    [float(line[1].split(",")[0][3:]), float(line[1].split(",")[1][4:])],
                    [float(line[7].split(",")[0][3:]), float(line[7].split(",")[1][4:])],
                    [float(line[9].split(",")[0][3:]), float(line[9].split(",")[1][4:])],
                    [float(line[11].split(",")[0][3:]), float(line[11].split(",")[1][4:])],
                ]
                cam3_points.append(points)
        count += 1
        # print(points)

# Triangluate points
# points_array = np.array([cam1_points[t][0], cam2_points[t][0], cam3_points[t][0]])
projection_matrices = np.array([cam_1.projection_matrix, cam_2.projection_matrix, cam_3.projection_matrix])
# print(points_array.shape)
num_points = end_frame - start_frame
index = 3

triangulated_points = []

for t in range(num_points):
    if cam1_points[t] is None or cam2_points[t] is None or cam3_points[t] is None:
        print(f"Skipping frame {t}: missing keypoints in one or more cameras")
        continue

    points_array = np.array([cam1_points[t][index], cam2_points[t][index], cam3_points[t][index]])
    cam1_points_undistorted = undistort_points(points_array[0].reshape(1, 2), cam_1.camera_matrix, cam_1.distortion_coefficients)
    cam2_points_undistorted = undistort_points(points_array[1].reshape(1, 2), cam_2.camera_matrix, cam_2.distortion_coefficients)
    cam3_points_undistorted = undistort_points(points_array[2].reshape(1, 2), cam_3.camera_matrix, cam_3.distortion_coefficients)
    points_array_undistorted = np.array([cam1_points_undistorted[0], cam2_points_undistorted[0], cam3_points_undistorted[0]])

    try:
        point_3d = triangulate(points_array_undistorted, projection_matrices)
        print(f"Triangulated point at time {t}: {point_3d}")
        triangulated_points.append(point_3d)
    except Exception as e:
        print(f"Error triangulating point at time {t}")
        continue

# Build camera dicts for visualization
cameras_vis = []
for cam in [cam_1, cam_2, cam_3]:
    K, R, center = decompose_projection_matrix(cam.projection_matrix)
    # R is world-to-camera; R.T columns are camera's local axes in world coords
    cameras_vis.append({"label": cam.name, "position": center, "rotation": R.T})

if triangulated_points:
    plot_sequence(
        cameras_vis,
        triangulated_points,
        title=f"Triangulation — frames {start_frame}–{end_frame-1}, keypoint {index}",
        start_frame=start_frame,
    )

