import cv2
import os
import sys
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CAMERA_INDICES, CALIBRATION_DIR


def getImage(cam_number):
    cap = cv2.VideoCapture(CAMERA_INDICES[cam_number - 1])
    num = 0

    while cap.isOpened():
        succes, img = cap.read()
        k = cv2.waitKey(5)

        # escape key
        if k == 27:
            break
        elif k == ord('s'): # wait for 's' key to save and exit
            cv2.imwrite(os.path.join(CALIBRATION_DIR, f'cam{cam_number}', 'images', f'image{num}.png'), img)
            print("image saved!")
            num += 1

        cv2.imshow('Img', img)

    cap.release()
    cv2.destroyAllWindows()


def test_camera(index, show=False):
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        print(f"Camera found at index {index}")
        if show:
            ret, frame = cap.read()
            if ret:
                cv2.imshow(f'Camera index {index} (any key to continue)', frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        cap.release()
        return True
    else:
        print(f"No camera at index {index}")
        return False


def list_cameras(max_index=10, show=False):
    print("Testing camera indices...")
    for i in range(max_index):
        test_camera(i, show=show)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            'Capture calibration images from a camera, or list available camera indices.\n\n'
            'Examples:\n'
            '  python getImages.py 1              # capture images for camera 1\n'
            '  python getImages.py --list         # list all detected camera indices\n'
            '  python getImages.py --list --show  # list cameras and preview a frame from each\n'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('cam_number', type=int, nargs='?',
                        help='Camera number to capture images from (e.g. 1, 2, 3)')
    parser.add_argument('--list', action='store_true',
                        help='List all detected camera indices')
    parser.add_argument('--show', action='store_true',
                        help='When listing cameras, show a preview frame for each detected camera')
    args = parser.parse_args()

    if args.list:
        list_cameras(show=args.show)
    elif args.cam_number is not None:
        getImage(args.cam_number)
    else:
        parser.print_help()