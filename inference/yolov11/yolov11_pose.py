import csv
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CAMERA_INDICES, LANDMARKS_DIR
from ultralytics import YOLO #type: ignore


def run(model_path: str, video_path: str, export_path: str, gpu: bool = True):

    # Load a pretrained YOLO11n model
    model = YOLO("yolo11n-pose.pt")

    # Run inference on the source
    if gpu:
        results = model.predict(source = video_path, imgsz= (480,640), show=True, stream=True, device=0)  # generator of Results objects

    CONF_THRESHOLD = 0.75
    # Iterate through the generator to process each frame
    with open(export_path, mode='w', newline='') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL)
        header = ['frame_number'] + [f'kp_{i}' for i in range(17)]
        writer.writerow(header)
        frame_number = 0

        for result in results:
            row: list = [frame_number]

            if result.keypoints is not None and result.boxes is not None:
                high_conf_indices = [i for i, conf in enumerate(result.boxes.conf) if conf > CONF_THRESHOLD]
                # num_persons = len(result.keypoints)
                num_persons = len(high_conf_indices)
                print(f"Frame {frame_number}: {num_persons} high-confidence person(s) detected")

                if num_persons > 1:
                    print("WARNING: more than one person in frame")
                    highest_conf_idx = max(high_conf_indices, key=lambda i: result.boxes.conf[i])
                    high_conf_indices = [highest_conf_idx]
                    num_persons = 1
                    
                if num_persons == 1:
                    i = high_conf_indices[0]
                    keypoint = result.keypoints[i]
                    
                    for kp_idx in range(keypoint.xy.shape[1]):  # 17 keypoints
                        x = float(keypoint.xy[0, kp_idx, 0])  # x coordinate of keypoint kp_idx
                        y = float(keypoint.xy[0, kp_idx, 1])  # y coordinate of keypoint kp_idx
                        conf = float(keypoint.conf[0, kp_idx]) if keypoint.conf is not None else None
                        
                        kp_data = f"x: {x:.4f}, y: {y:.4f}, conf: {conf:.4f}" if conf is not None else f"x: {x:.4f}, y: {y:.4f}"
                        row.append(kp_data)
                        print(f"Person {i}, Keypoint {kp_idx}: {kp_data}")
                
                else:
                    print(f"Frame {frame_number}: No high confidence person detected")
                    row.extend(["null"]*17)

            else:
                print("No person detected in frame")
                row.extend(["null"]*17)

            writer.writerow(row)
            frame_number += 1
        # Access keypoints: result.keypoints.xy for (x,y) coordinates
        # result.keypoints.xyn for normalized coordinates
        # result.keypoints.conf for confidence scores


   


if __name__ == "__main__":
    # cam_num = 1
    # cam_num = 2
    # cam_num = 3

    cam_nums = range(1, len(CAMERA_INDICES) + 1)


    model_path = r'C:\Users\kxfor\OneDrive\Documents\Projects\Personal_Projects\Arm-Movement-Using-Joint-Position\models\pose_landmarker_full.task'
    for num in cam_nums:
        export_path = os.path.join(LANDMARKS_DIR, f'cam_{num}_pose_landmarks2.csv')
        video_path = rf'C:\Users\Kahlil Pollack-Hinds\Documents\Projects\Arm-Movement-Using-Joint-Position\output{num}new_vid2.mp4'
        run(model_path, video_path, export_path)