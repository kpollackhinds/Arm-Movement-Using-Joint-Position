import time
import numpy as np
import cv2 as cv
import glob
import pickle
import sys

# STEPS TO FIND EXTRINSICS #
# 1. Load the camera matrix and distortion coefficients
# 2. Load the image
# 3. Find the chessboard corners (determine object points)
# 4. Find the chessboard corners (determine image points)
# 5. Use cv.solvePnP to find the rotation and translation vectors
# 6. Export the rotation and translation vectors for each camera

# ###############################################################
# Displaying the camera position in 3D
# ###############################################################
# 1. Convert the rotation vector to a rotation matrix
# 2. Use matplotlib to plot the camera position

cam_num = 'cam1'
# cam_matrix_file_path = 'Camera_Calibration/cam1/cameraMatrix.pkl'
calibration_file_path = f'Camera_Calibration/{cam_num}/calibration.pkl'
# distortion_file_path = 'Camera_Calibration/cam1/dist.pkl'
image = None
cap = cv.VideoCapture(0)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        print("Failed to capture image")
        break   

    k = cv.waitKey(5)
    # 27 == escape key
    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv.imshow('img', img)
        # show the image 
        x = cv.waitKey(0)

        # confirm picture taken to be saved
        if x == ord('c'):
            image = img
            print("image saved!")
            break

    cv.imshow('img', img)

cap.release()
cv.destroyAllWindows()

chessboardSize = (10,7)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
# print(objp)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)
# print(objp)

size_of_chessboard_squares_mm = 25
objp = objp * size_of_chessboard_squares_mm

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)
print(f"corners: {corners}")
if not ret:
    print("Chessboard corners not found")
    sys.exit(1)
cv.drawChessboardCorners(img, chessboardSize, corners, ret)
cv.imshow('img', img)
cv.waitKey(0)

with open(calibration_file_path, 'rb') as f:
    cam_intrinsics = pickle.load(f)
# cam_intrinsics = pickle.load(open(calibration_file_path))

print(f"Distortion Coefficients:\n{cam_intrinsics[1]}")

# Get camera extrinsics
if ret:
    ret, rot_vec, trans_vec = cv.solvePnP(objectPoints=objp, 
                imagePoints=corners, 
                cameraMatrix=cam_intrinsics[0], 
                distCoeffs=cam_intrinsics[1],)

# Change from rotation vector to rotation matrix
rot_mat, _ = cv.Rodrigues(rot_vec)
print(rot_mat)
print(trans_vec)

Rt = np.hstack((rot_mat, trans_vec))
print(f"Camera Intrinsics Matrix:\n{cam_intrinsics[0]}")
print("Rt Matrix:\n", Rt)
print("shape of Rt:", Rt.shape)
P = cam_intrinsics[0] @ Rt
print("Camera Projection Matrix P:\n", P)

# Validate the projection matrix
# 1. Project the 3D points back to 2D
# 2. Compare the projected points with the original image points
projected_points, _ = cv.projectPoints(objectPoints=objp, 
                                        rvec=rot_vec,
                                        tvec=trans_vec,
                                        cameraMatrix=cam_intrinsics[0],
                                        distCoeffs=cam_intrinsics[1])

error = cv.norm(corners, projected_points, cv.NORM_L2) / len(projected_points)
print(f"Reprojection error: {error:.4f}")


for pt in projected_points:
    cv.circle(image, tuple(pt[0].astype(int)), 5, (0, 255, 0), -1)
cv.imshow('img', image)
cv.waitKey(0)

pickle.dump(P, open(f'Camera_Calibration/{cam_num}/projection_matrix.pkl', 'wb'))