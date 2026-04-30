# Multi-Camera 3D Human Pose Estimation
An implementation of a multi-camera 3D pose estimatiion pipeline using DLT triangulation. Three webcameras capture 2D joint position estimates using a pose detection model (YOLOv11). Using multiple point correspondances, and the camera projection matrix built from calibrate camera intrinsics and extrinsics, an estimate of the coordinates of the point in the world frame (3D space) can be obtained. See the [math](#math) section for details.

## Setup
**Requirements**
- Python
- 3 webcams (code can be easily modified to accomodate more)
- Printed Checkerboard pattern

**How I positioned my cameras**
--insert image here--

*Note you dont want to put the cameras in one line, because that creates a degenerative solution for the DLT [details](#Additional)

### Code Setup
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

### Camera Calibration (Intrinsics)
### Camera Calibration (Extrinsics)

### Data Collection
### 3D Triangulation and Visualization


## Math
Here are the details
### Additional Considerations

## Helpful Resources