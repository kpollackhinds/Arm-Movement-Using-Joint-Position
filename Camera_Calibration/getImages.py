# import cv2

# cap = cv2.VideoCapture(1)

# num = 0

# while cap.isOpened():

#     succes, img = cap.read()

#     k = cv2.waitKey(5)

#     # escape key
#     if k == 27:
#         break
#     elif k == ord('s'): # wait for 's' key to save and exit
#         cv2.imwrite('Camera_Calibration\images\cam3\image' + str(num) + '.png', img)
#         print("image saved!")
#         num += 1

#     cv2.imshow('Img',img)

# # Release and destroy all windows before termination
# cap.release()

# cv2.destroyAllWindows()


import cv2

def test_camera(index):
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        print(f"Camera found at index {index}")
        cap.release()
        return True
    else:
        print(f"No camera at index {index}")
        return False

for i in range(10): # Check indices 0 to 9
    if test_camera(i):
        print("\n")
        # break