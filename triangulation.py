import numpy as np
import csv
import cv2
from utils_3D.core.triangulation import triangulate
from utils_3D.core.camera import Camera
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
cam_1 = Camera("cam_1", projection_matrix=pickle.load(open("Camera_Calibration\\cam1\\projection_matrix.pkl", "rb")))
cam_2 = Camera("cam_2", projection_matrix=pickle.load(open("Camera_Calibration\\cam2\\projection_matrix.pkl", "rb")))
cam_3 = Camera("cam_3", projection_matrix=pickle.load(open("Camera_Calibration\\cam3\\projection_matrix.pkl", "rb")))

cam1_points = []
cam2_points = []
cam3_points = []

files = [
    "cam_1_pose_landmarks.csv",
    "cam_2_pose_landmarks.csv",
    "cam_3_pose_landmarks.csv",
]

# cam1
with open(files[0], mode="r") as file:
    csvFile = csv.reader(file)
    # read header
    next(csvFile, None)

    # Points ends up being a list of 4 points, representing the 4 keypoints at a given frame at time (t).
    for line in csvFile:
        points = [
            [float(line[1].split(",")[0][3:]), float(line[1].split(",")[1][4:])],
            [float(line[7].split(",")[0][3:]), float(line[7].split(",")[1][4:])],
            [float(line[9].split(",")[0][3:]), float(line[9].split(",")[1][4:])],
            [float(line[11].split(",")[0][3:]), float(line[11].split(",")[1][4:])],
        ]
        cam1_points.append(points)
        print(points)
        break

# cam2
with open(files[1], mode="r") as file:
    csvFile = csv.reader(file)
    # read header
    next(csvFile, None)

    for line in csvFile:
        points = [
            [float(line[1].split(",")[0][3:]), float(line[1].split(",")[1][4:])],
            [float(line[7].split(",")[0][3:]), float(line[7].split(",")[1][4:])],
            [float(line[9].split(",")[0][3:]), float(line[9].split(",")[1][4:])],
            [float(line[11].split(",")[0][3:]), float(line[11].split(",")[1][4:])],
        ]
        cam2_points.append(points)
        print(points)
        break

# cam3
with open(files[2], mode="r") as file:
    csvFile = csv.reader(file)
    # read header
    next(csvFile, None)

    for line in csvFile:
        points = [
            [float(line[1].split(",")[0][3:]), float(line[1].split(",")[1][4:])],
            [float(line[7].split(",")[0][3:]), float(line[7].split(",")[1][4:])],
            [float(line[9].split(",")[0][3:]), float(line[9].split(",")[1][4:])],
            [float(line[11].split(",")[0][3:]), float(line[11].split(",")[1][4:])],
        ]
        cam3_points.append(points)
        print(points)
        break

# Triangluate points
np_array = np.array([cam1_points[0], cam2_points[0], cam3_points[0]])
print(np_array.shape)

triangulated_points = cv2.triangulatePoints
