# Multi-Camera 3D Human Pose Estimation
An implementation of a multi-camera 3D pose estimatiion pipeline using DLT triangulation. Three webcameras capture 2D joint position estimates using a pose detection model (YOLOv11). Using multiple point correspondances, and the camera projection matrix built from calibrate camera intrinsics and extrinsics, an estimate of the coordinates of the point in the world frame (3D space) can be obtained. See the [math](#math) section for details.

Basic steps:
1. Collect images of a checkerboard pattern from each camera using [getImages.py](capture/getImages.py).
2. Get camera matrix from [camera_calibration_intrinsics.py](calibration/camera_calibration_intrinsics.py).
3. Get camera extrinsics from [camera_calibration_extrinsics.py](calibration/camera_calibration_extrinsics.py).
4. Triangulate and visualize 3D pose from 2D detections using [triangulation.py](triangulation.py).


## Setup
**Requirements**
- Python
- 3 webcams (code can be easily modified to accomodate more)
- Printed Checkerboard pattern

**How I positioned my cameras**
--insert image here--

*Note you dont want to put the cameras in one line, because that creates a degenerative solution for the DLT [details](#Additional)

### Environment Setup
Clone repository:
```
git clone https://github.com/kpollackhinds/multi-camera-human-pose-estimation.git
```

Setup virtual environment (optional):
```
python -m venv venv
source venv/bin/activate  
# On Windows: venv\Scripts\activate
```

Install dependencies:
```
pip install -r requirements.txt
```

### Config
Update the [config.py](config.py) file with the correct camera index values for your setup. If you don't know the camera index, you can run `python capture/getImages.py --list --show` to see the available cameras and their indices.


### Camera Calibration (Intrinsics)
Gather at least 10-20 images of the checkerboard pattern from different angles and positions for each camera using [getImages.py](capture/getImages.py). Press 's' to save each image, and 'esc' to exit the camera feed when done.
```
python capture/getImages.py 1
python capture/getImages.py 2
python capture/getImages.py 3
```

Run the [camera_calibration_intrinsics.py](calibration/camera_calibration_intrinsics.py) script for each camera. Exports the camera matrix and distortion coefficients for each camera to a .pkl file.
``` 
python calibration/camera_calibration_intrinsics.py 1
python calibration/camera_calibration_intrinsics.py 2
python calibration/camera_calibration_intrinsics.py 3
```

### Camera Calibration (Extrinsics)
For each camera, point it at the checkerboard in its fixed position. Run [camera_calibration_extrinsics.py](calibration/camera_calibration_extrinsics.py), press 's' to capture a frame, then 'c' to confirm. Exports the projection matrix for each camera to a .pkl file.
```
python calibration/camera_calibration_extrinsics.py 1
python calibration/camera_calibration_extrinsics.py 2
python calibration/camera_calibration_extrinsics.py 3
```
### Data Collection
To record videos from each camera, run the [multi_cam_vid_capture.py](capture/multi_cam_vid_capture.py) script. Press 'esc' to stop recording.

```
python capture/multi_cam_vid_capture.py
```
To test the camera positioning/feed without recording, run the [multi_cam_vid_capture.py](capture/multi_cam_vid_capture.py) script with the `--test` flag. Press 'esc' to stop the feed.

```
python capture/multi_cam_vid_capture.py --test
```
### 3D Triangulation and Visualization
Run the [triangulation.py](triangulation.py) script to see the results of the triangulation.
```
python triangulation.py
```


## Math
Here are the details
### Additional Considerations

## Helpful Resources