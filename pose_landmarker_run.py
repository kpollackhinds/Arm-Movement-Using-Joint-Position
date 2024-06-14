import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import time
from queue import Queue 
from typing import Union

def draw_landmarks_on_image(rgb_image, detection_result):
  world_pose_landmarks_list = detection_result.pose_world_landmarks
  try:
    right_wrist = world_pose_landmarks_list[0][15]
    print("z: ", right_wrist.z)
    print('\n')
  except:
    print("wrist not detected")
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

def print_result(result: vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
  print('pose landmarker result: {}'.format(result))

def callback_factory(queue:Queue):
  def update_queue(result:vision.PoseLandmarkerResult, rgb_frame: mp.Image, timestamp_ms: int = None):
  # annotated_frame = draw_landmarks_on_image(rgb_image=  rgb_frame, detection_result=result)
    queue.put((rgb_frame.numpy_view(), result))
  return update_queue

def display_results(queue: Queue, frame):
  if not queue.empty(): 
    frame, result = queue.get()
    frame = draw_landmarks_on_image(rgb_image=frame, detection_result=result)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
  
  cv2.imshow('Pose Detection', frame)
  return

def run_pose_detection(model_path: str, video_source: Union[vision.RunningMode] = vision.RunningMode.LIVE_STREAM):
  results_queue = Queue()
  update_results_queue = callback_factory(results_queue)

  options = vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=video_source)
  
  if video_source == 'LIVE_STREAM':
    options.result_callback=update_results_queue
  
  detector = mp.tasks.vision.PoseLandmarker.create_from_options(options)

  video_capture = cv2.VideoCapture(0)

  while True:
    ret, frame = video_capture.read()
    if not ret:
      break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_timestamp_ms = int(time.time()*1000)

    mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = rgb_frame)

    if video_source == 'LIVE_STREAM':
      detector.detect_async(mp_image, frame_timestamp_ms)
    else:
      result = detector.detect(mp_image)
      update_results_queue(result= result, rgb_frame= mp_image)
      
    display_results(queue= results_queue, frame= rgb_frame)
 

    if cv2.waitKey(1) & 0xFF == ord("q"):
      break
  
  video_capture.release()
  cv2.destroyAllWindows()

  return

if __name__ == "__main__":  
  model_path = r'C:\Users\kxfor\OneDrive\Documents\Projects\Personal_Projects\Arm-Movement-Using-Joint-Position\models\pose_landmarker_full.task'

  run_pose_detection(model_path=model_path, video_source=vision.RunningMode.IMAGE)
  

  
  