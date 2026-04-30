import cv2
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CAMERA_INDICES, CALIBRATION_DIR

# cam1 =2 
# cam2 = 0

cap = cv2.VideoCapture(CAMERA_INDICES[2])

num = 6

while cap.isOpened():

    succes, img = cap.read()

    k = cv2.waitKey(5)

    # escape key
    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite(os.path.join(CALIBRATION_DIR, 'cam3', 'images', f'image{num}.png'), img)
        print("image saved!")
        num += 1

    cv2.imshow('Img',img)

# Release and destroy all windows before termination
cap.release()

cv2.destroyAllWindows()


# import cv2

# def test_camera(index):
#     cap = cv2.VideoCapture(index)
#     if cap.isOpened():
#         print(f"Camera found at index {index}")
#         cap.release()
#         return True
#     else:
#         print(f"No camera at index {index}")
#         return False

# for i in range(10): # Check indices 0 to 9
#     if test_camera(i):
#         print("\n")
#         # break