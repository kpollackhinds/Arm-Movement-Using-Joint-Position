import cv2 
import threading

# class camThread(threading.Thread):
#     def __init__(self, previewName, camID):
#         threading.Thread.__init__(self)
#         self.previewName = previewName
#         self.camID = camID
    
#     def run(self):
#         print (f"Starting {self.previewName}")
#         self.camPreview(self.previewName, self.camID)
    
#     def camPreview(previewName, camID):
#         cv2.namedWindow(previewName)
#         cam = cv2.VideoCapture(camID)
        
#         if cam.isOpened():
#             ret, frame = cam.read()
#         else:
#             ret = False

#         while ret:
#             ret, frame = cam.read()
#             cv2.imshow(previewName, frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
        
#         cv2.destroyWindow(previewName)
#         cam.release()

    
# import numpy as np
# import cv2

video_capture_1 = cv2.VideoCapture(1)
video_capture_2 = cv2.VideoCapture(2)
video_capture_3 = cv2.VideoCapture(3)


# fourcc = cv2.VideoWriter.fourcc(*'mp4v')
# frame_width = int(video_capture_1.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(video_capture_1.get(cv2.CAP_PROP_FRAME_HEIGHT))
# # cam1
# out1 = cv2.VideoWriter('output1.mp4', fourcc, 30.0, (frame_width, frame_height))
# # cam2
# out2 = cv2.VideoWriter('output2.mp4', fourcc, 30.0, (frame_width, frame_height))
# # cam3
# out3 = cv2.VideoWriter('output3.mp4', fourcc, 30.0, (frame_width, frame_height))

# while True:
#     ret1, frame1 = video_capture_1.read()
#     ret2, frame2 = video_capture_2.read()
#     ret3, frame3 = video_capture_3.read()

#     if ret1:
#         out1.write(frame1)
#         cv2.imshow('Cam 1', frame1)

#     if ret2:
#         out2.write(frame2)
#         cv2.imshow('Cam 2', frame2)

#     if ret3:
#         out3.write(frame3)
#         cv2.imshow('Cam 3', frame3)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# video_capture_1.release()
# video_capture_2.release()
# video_capture_3.release()
# out1.release()
# out2.release()
# out3.release()
# cv2.destroyAllWindows()

################### Test multi camera capture #########################
while True:
    ret2, frame2 = video_capture_2.read()
    ret1, frame1 = video_capture_1.read()
    ret3, frame3 = video_capture_3.read()

    if (ret1):
        cv2.imshow('Cam 0', frame1)

    if (ret2):
        cv2.imshow('Cam 1', frame2)

    if (ret3):
        cv2.imshow('Cam 2', frame3)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture_1.release()
video_capture_2.release()
video_capture_3.release()


cv2.destroyAllWindows()

