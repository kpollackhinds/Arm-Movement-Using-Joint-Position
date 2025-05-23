import numpy as np
import cv2 as cv
import glob
import pickle
import sys


################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

chessboardSize = (10,7) # (number of inner corners per a chessboard row and column)
frameSize = (640,480) # (640,480) # size of the images used for calibration



# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
print(objp)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)
print(objp)

# sys.exit()
size_of_chessboard_squares_mm = 25
objp = objp * size_of_chessboard_squares_mm # coordinates in mm

print(objp)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


images = glob.glob('Camera_Calibration/cam1/images/*.png')

for image in images:

    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # cv.imshow('img', gray)
    # cv.waitKey(1000)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)
    
    # print(ret)
    # If found, add object points, image points (after refining them)
    if ret == True:

        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv.imshow('img', img)
        # cv.waitKey(1000)


cv.destroyAllWindows()




############## CALIBRATION #######################################################

ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

print("Camera Matrix:\n")
print(cameraMatrix)
print("\n")
print("Distortion Coefficients:\n")
print(dist)

# # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
# pickle.dump((cameraMatrix, dist), open( "Camera_Calibration/cam3/calibration.pkl", "wb" ))
# pickle.dump(cameraMatrix, open( "Camera_Calibration/cam3/cameraMatrix.pkl", "wb" ))
# pickle.dump(dist, open( "Camera_Calibration/cam3/dist.pkl", "wb" ))


############## UNDISTORTION #####################################################

img = cv.imread('Camera_Calibration\cam1\images\image0.png')
h,  w = img.shape[:2]
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))

print("New Camera Matrix:\n")
print(newCameraMatrix)
print("\n")

# Undistort
dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
# cv.imwrite('Camera_Calibration\cam1\image0_result.png', dst)



# Undistort with Remapping
mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
# cv.imwrite('caliResult2.png', dst)




# Reprojection Error
mean_error = 0

for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    print("Error for image {}: {}".format(i, error))
    mean_error += error

# error in pixels
print( "total error: {}".format(mean_error/len(objpoints)) )