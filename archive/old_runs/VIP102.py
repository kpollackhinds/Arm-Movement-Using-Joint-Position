import cv2
import mediapipe as mp
import csv
import os

mp_pose = mp.solutions.pose
rootdir = [r'C:\Users\ahalw\yoga data\DATASET\TRAIN', r'C:\Users\ahalw\yoga data\DATASET\TEST']
dataFile = r'C:\Users\ahalw\yoga data\data.csv'
# create header
header = []
for i in range(33):
    header.append('kp ' + str(i) + ' x')
    header.append('kp ' + str(i) + ' y')
    header.append('kp ' + str(i) + ' z')
header.append('pose')
print(header)
#add header to csv
with open(dataFile, 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    # write the header
    writer.writerow(header)
for x in range(2):
    for subdir, dirs, files in os.walk(rootdir[x]):
        for file in files:
            filePath = os.path.join(subdir, file)
            print(filePath)
            s = filePath.split("\\")
            poseName = s[len(s)-2]
            with mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=True,
                              min_detection_confidence=0.5) as pose:
                image = cv2.imread(filePath)
                results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                if not results.pose_landmarks:
                    continue
                row = []
                for value in results.pose_world_landmarks.landmark:
                    row.append(value.x)
                    row.append(value.y)
                    row.append(value.z)
                # append pose
                row.append(poseName)
                with open(dataFile, 'a+', newline='', encoding='UTF8') as f:
                    writer = csv.writer(f)
                    # write the data
                    writer.writerow(row)