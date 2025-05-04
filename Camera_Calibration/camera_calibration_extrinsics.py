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
        # show the image for 5 seconds
        x = cv.waitKey(5000)

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
print(objp)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)
print(objp)

size_of_chessboard_squares_mm = 25
objp = objp * size_of_chessboard_squares_mm

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

cam_intrinsics = pickle.load(calibration_file_path)

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
P = cam_intrinsics @ Rt
print("Camera Projection Matrix P:\n", P)

pickle.dump(P, open(f'Camera_Calibration/{cam_num}/projection_matrix.pkl', 'wb'))