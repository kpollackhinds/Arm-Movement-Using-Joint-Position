import sys

import numpy as np
import csv
import cv2
from utils_3D.core.triangulation import triangulate, build_dlt_matrix 
from utils_3D.core.camera import Camera
from utils_3D.core.projection import get_camera_center_from_projection_matrix, undistort_points
import pickle

def full_triangulation_diagnostic(image_point_correspondences, projection_matrices, K_list, dist_list):
    """
    Run this on a single frame first, share the full output.
    """
    A = build_dlt_matrix(image_point_correspondences, projection_matrices)
    
    _, S, Vt = np.linalg.svd(A)
    X = Vt[-1]

    print("=" * 60)
    print("1. INPUT CORRESPONDENCES")
    for i, pt in enumerate(image_point_correspondences):
        print(f"   Cam {i}: {pt}")

    print("\n2. SINGULAR VALUES")
    print(f"   S = {S}")
    print(f"   S[-1]/S[-2] = {S[-1]/S[-2]:.6f}  (want << 1e-3)")
    print(f"   S[-2]/S[-3] = {S[-2]/S[-3]:.6f}")

    print("\n3. RAW HOMOGENEOUS SOLUTION")
    print(f"   X = {X}")
    print(f"   w = {X[3]:.8f}  (near zero = point at infinity)")

    print("\n4. PROJECTION MATRIX SANITY")
    for i, P in enumerate(projection_matrices):
        Q = P[:3, :3]
        cond = np.linalg.cond(Q)
        center = get_camera_center_from_projection_matrix(P)
        print(f"   Cam {i}: condition(Q)={cond:.2f}, center={center}")

    print("\n5. DLT MATRIX A")
    print(f"   Shape: {A.shape}")
    print(f"   Rank: {np.linalg.matrix_rank(A)}")
    print(f"   A =\n{A}")

    print("\n6. REPROJECTION CHECK (using raw Vt[-1])")
    errors = {}
    if abs(X[3]) > 1e-8:
        X_cart = X[:3] / X[3]
        # print(f"   Cartesian 3D point: {X_cart}")
        for i, (P, pt) in enumerate(zip(projection_matrices, image_point_correspondences)):
            x_proj_h = P @ X
            x_proj = x_proj_h[:2] / x_proj_h[2]
            err = np.linalg.norm(x_proj - pt)
            errors[i] = err
            # print(f"   Cam {i}: projected={x_proj}, actual={pt}, error={err:.4f}px")
    else:
        # print("   w ~ 0, point is at infinity — severe degeneracy")
        for i in range(len(projection_matrices)):
            errors[i] = float('inf')
    return errors


files = [
    "cam_1_pose_landmarks2.csv",
    "cam_2_pose_landmarks2.csv",
    "cam_3_pose_landmarks2.csv",
]
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

count = 0
NUMPOINTS = 400
# cam1
with open(files[0], mode="r") as file:
    csvFile = csv.reader(file)
    # read header
    next(csvFile, None)

    # Points ends up being a list of 4 points, representing the 4 keypoints at a given frame at time (t).
    for line in csvFile:
        if count >= NUMPOINTS:
            break
        points = [
            float(line[1].split(",")[0][3:]), float(line[1].split(",")[1][4:]),
            # [float(line[7].split(",")[0][3:]), float(line[7].split(",")[1][4:])],
            # [float(line[9].split(",")[0][3:]), float(line[9].split(",")[1][4:])],
            # [float(line[11].split(",")[0][3:]), float(line[11].split(",")[1][4:])],
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
        if count >= NUMPOINTS:
            break
        points = [
            float(line[1].split(",")[0][3:]), float(line[1].split(",")[1][4:]),
            # [float(line[7].split(",")[0][3:]), float(line[7].split(",")[1][4:])],
            # [float(line[9].split(",")[0][3:]), float(line[9].split(",")[1][4:])],
            # [float(line[11].split(",")[0][3:]), float(line[11].split(",")[1][4:])],
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
        if count >= NUMPOINTS:
            break
        points = [
            float(line[1].split(",")[0][3:]), float(line[1].split(",")[1][4:]),
            # [float(line[7].split(",")[0][3:]), float(line[7].split(",")[1][4:])],
            # [float(line[9].split(",")[0][3:]), float(line[9].split(",")[1][4:])],
            # [float(line[11].split(",")[0][3:]), float(line[11].split(",")[1][4:])],
        ]
        cam3_points.append(points)
        count += 1
        # print(points)

# @ frame t
projection_matrices = np.array([cam_1.projection_matrix, cam_2.projection_matrix, cam_3.projection_matrix])
intrinsic_matrices = [cam_1.camera_matrix, cam_2.camera_matrix, cam_3.camera_matrix]
distortion_coefficients = [cam_1.distortion_coefficients, cam_2.distortion_coefficients, cam_3.distortion_coefficients]

num_cams = len(projection_matrices)
cam_error_sums = {i: 0.0 for i in range(num_cams)}

for t in range(201):
    point_correspondences = np.array([cam1_points[t], cam2_points[t], cam3_points[t]])
    errors = full_triangulation_diagnostic(point_correspondences, projection_matrices, intrinsic_matrices, distortion_coefficients)
    for i in range(num_cams):
        cam_error_sums[i] += errors[i]

print("Average reprojection error (t=0..200):")
for i in range(num_cams):
    print(f"  Cam {i}: {cam_error_sums[i] / 201:.4f} px")