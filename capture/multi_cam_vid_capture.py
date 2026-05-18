import cv2
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CAMERA_INDICES

NUM_CAMS = len(CAMERA_INDICES)


def capture_video():
    captures = [cv2.VideoCapture(idx) for idx in CAMERA_INDICES]

    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    writers = []
    for i, cap in enumerate(captures):
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writers.append(cv2.VideoWriter(f'output_vid_cam{i + 1}.mp4', fourcc, 30.0, (w, h)))

    while True:
        for i, (cap, writer) in enumerate(zip(captures, writers)):
            ret, frame = cap.read()
            if ret:
                writer.write(frame)
                cv2.imshow(f'Cam {i + 1}', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cap in captures:
        cap.release()
    for writer in writers:
        writer.release()
    cv2.destroyAllWindows()


def test_multi_cam_capture():
    '''
    Test function for multi-camera capture. Opens all camera feeds and displays
    them in separate windows. Press 'q' to quit.
    '''
    captures = [cv2.VideoCapture(idx) for idx in CAMERA_INDICES]

    while True:
        for i, cap in enumerate(captures):
            ret, frame = cap.read()
            if ret:
                cv2.imshow(f'Cam {i + 1}', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cap in captures:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f'Multi-camera video capture ({NUM_CAMS} cameras from config).'
    )
    parser.add_argument('--test', action='store_true',
                        help='Run in test mode: display camera feeds without recording')
    args = parser.parse_args()

    if args.test:
        test_multi_cam_capture()
    else:
        capture_video()