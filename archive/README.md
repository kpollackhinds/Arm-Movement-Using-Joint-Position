# Archive

Early-stage prototype that used **MediaPipe Pose** (browser-based) to control a 2-bar servo linkage over Bluetooth in real time.

## How it worked

1. **Pose detection (`Arm controls/`)** — A p5.js web app captured webcam video and ran the MediaPipe Pose model (`@mediapipe/pose`) directly in the browser. For each frame, 2D joint angles were computed using the dot product of vectors formed by three landmarks:
   - **Lower arm angle** — elbow joint (shoulder → elbow → wrist, landmarks 12/14/16)
   - **Upper arm angle** — upper arm relative to the torso midline (elbow → shoulder midpoint → hip midpoint)

2. **Bluetooth transmission** — When either angle changed by more than 5°, the angles were formatted as `<upperAngle>&<lowerAngle>-` and sent over BLE using `p5.ble` to an **HM-10** Bluetooth module. The `-` character acts as a message terminator; a space (` `) is sent for whichever angle did not change.

3. **Arduino (`Arduino_Bluetooth_Sketch/`)** — The HM-10 module was connected to the Arduino via `SoftwareSerial`. On receiving a complete message (terminated by `-`), the sketch parsed the two angle values and wrote them to two servo motors (pins 9 and 10), driving the 2-bar linkage. The lower servo angle is inverted (`180 - angle`) to account for mounting orientation.

## Other files

| File | Description |
|------|-------------|
| `old_runs/holistic_landmarker_run.py` | Webcam demo using MediaPipe Holistic (face, pose, hands) — used for early exploration |
| `old_runs/pose_landmarker_run.py` | Webcam demo using the newer MediaPipe Tasks `PoseLandmarker` API; logs wrist z-depth |
| `old_runs/VIP102.py` | Batch-extracts MediaPipe pose landmarks from a yoga image dataset into a CSV —  from a separate classification task, unrelated to arm control |

## Notes

- `index.html` imports `ml5.js` (which includes PoseNet), but it is not used in the active code — all pose inference runs through `@mediapipe/pose`.
- Bluetooth throughput was throttled by only sending every 10th frame (`counter % 10 == 0`) and resetting the counter at 100 to avoid flooding the HM-10.
