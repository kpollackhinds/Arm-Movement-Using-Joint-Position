from typing import Optional
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CAMERA_INDICES, LANDMARKS_DIR
import mediapipe as mp
from mediapipe.python.solutions import pose
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import cv2 as cv
import numpy as np
import csv


POSE_CONNECTIONS = frozenset([(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5),
                              (5, 6), (6, 8), (9, 10), (11, 12), (11, 13),
                              (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
                              (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
                              (18, 20), (11, 23), (12, 24), (23, 24), (23, 25),
                              (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
                              (29, 31), (30, 32), (27, 31), (28, 32)])


def draw_landmarks(frame, landmarks_list, connections=None):
    """
    Draw pose landmarks on a frame.
    
    Args:
        frame: OpenCV image/frame to draw on
        landmarks_list: List of landmark lists (usually contains one list of 33 NormalizedLandmarks)
        connections: Iterable of (start_idx, end_idx) tuples defining which landmarks to connect
    """
    if not landmarks_list:
        return
    
    height, width, _ = frame.shape
    
    # landmarks_list is a list containing lists of landmarks
    # Typically landmarks_list[0] contains all 33 pose landmarks
    for landmarks in landmarks_list:
        # Convert landmarks to pixel coordinates
        landmark_points = []
        for landmark in landmarks:
            x_px = int(landmark.x * width)
            y_px = int(landmark.y * height)
            landmark_points.append((x_px, y_px))
            
            # Draw each landmark as a circle
            cv.circle(frame, (x_px, y_px), 5, (0, 255, 0), -1)  # Green filled circle
        
        # Draw connections between landmarks
        if connections:
            for connection in connections:
                start_idx, end_idx = connection
                if start_idx < len(landmark_points) and end_idx < len(landmark_points):
                    start_point = landmark_points[start_idx]
                    end_point = landmark_points[end_idx]
                    cv.line(frame, start_point, end_point, (255, 255, 255), 2)  # White line


def export_landmarks(writer, landmarks, frame_number, timestamp_ms):
    if not landmarks:
        return
    
    row = [frame_number, timestamp_ms]

    # This is assuming there is only one list of landmarks being passed
    for landmark in landmarks:
        row.append(f"{landmark.x},{landmark.y},{landmark.z}")
    
    writer.writerow(row)



    
def run(model_path: str, video_path: str, export_path: str):
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options = BaseOptions(model_asset_path = model_path),
        running_mode = VisionRunningMode.VIDEO
    )

    with PoseLandmarker.create_from_options(options) as landmarker:
        # Use OpenCV’s VideoCapture to load the input video.
        cap = cv.VideoCapture(video_path)        
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            return
        # Load the frame rate of the video using OpenCV’s CV_CAP_PROP_FPS
        # You’ll need it to calculate the timestamp for each frame.
        fps = cap.get(cv.CAP_PROP_FPS)
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        print(f"FPS: {fps}")
        print(f"Total frames: {total_frames}")

        curr_frame = 0


        with open(export_path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            # Loop through each frame in the video using VideoCapture#read()
            # Convert the frame received from OpenCV to a MediaPipe’s Image object.
            while True:
                ret, frame = cap.read()
                curr_frame +=1
                timestamp_ms = int(cap.get(cv.CAP_PROP_POS_MSEC))
                if timestamp_ms >= 13000:
                    print("End of video")
                    break
                if not ret:
                    print(f"End of video or cant read frame at timestamp {timestamp_ms}")
                    break

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

                # Perform pose landmarking on the provided single image.
                # The pose landmarker must be created with the video mode.
                pose_landmarker_result = landmarker.detect_for_video(mp_image, timestamp_ms)

                if pose_landmarker_result.pose_landmarks:
                    draw_landmarks(frame, pose_landmarker_result.pose_landmarks, POSE_CONNECTIONS)


                # Implement this later
                export_landmarks(writer,pose_landmarker_result.pose_landmarks[0], curr_frame, timestamp_ms)

                cv.imshow("Landmark Frame", frame)
                
                # Wait 1ms and check if 'q' is pressed to quit
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv.destroyAllWindows()


if __name__ == "__main__":
    # cam_num = 1
    # cam_num = 2
    # cam_num = 3

    cam_nums = range(1, len(CAMERA_INDICES) + 1)

    model_path = r'C:\Users\kxfor\OneDrive\Documents\Projects\Personal_Projects\Arm-Movement-Using-Joint-Position\models\pose_landmarker_full.task'
    for num in cam_nums:
        export_path = os.path.join(LANDMARKS_DIR, f'cam_{num}_pose_landmarks.csv')
        video_path = r'C:\Users\kxfor\OneDrive\Documents\Projects\Personal_Projects\Arm-Movement-Using-Joint-Position\output{}.mp4'.format(num)
        run(model_path, video_path, export_path)