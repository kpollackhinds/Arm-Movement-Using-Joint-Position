import numpy as np
import cv2
def get_camera_center_from_extrinsics(rotation_matrix: np.ndarray, translation_vector: np.ndarray) -> np.ndarray:
    """
    Computes the camera center in world coordinates from the rotation matrix and translation vector.

    Args:
        rotation_matrix: A (3, 3) array representing the rotation of the camera.
        translation_vector: A (3,) array representing the translation of the camera.
    """
    return -rotation_matrix.T @ translation_vector

def get_camera_center_from_projection_matrix(projection_matrix: np.ndarray) -> np.ndarray:
    """
    Computes the camera center in world coordinates from the projection matrix.

    Args:
        projection_matrix: A (3, 4) array representing the camera projection matrix.
    """

    # Q  = K @ R
    # column4 = K @ t = K @ (-R @ C) = -Q @ C
    # C = -inv(Q) @ column4

    Q = projection_matrix[0:3, 0:3]
    column4 = projection_matrix[0:3, 3]
    return -np.linalg.inv(Q) @ column4

def undistort_points(points_2d: np.ndarray, camera_intrinsics: np.ndarray, distortion_coefficients: np.ndarray) -> np.array:
    """
    Args:
        points_2d: A (N, 2) array of 2D points in pixel coordinates.
        camera_intrinsics: A (3, 3) array representing the camera intrinsic matrix.
        distortion_coefficients: A (k,) array of distortion coefficients (e.g., [k1, k2, p1, p2, k3]).
    
    Returns:
        A (N, 2) array of undistorted 2D points in pixel coordinates.
    """
    points_2d = points_2d.reshape(-1, 1, 2)  # Reshape to (N, 1, 2) for OpenCV
    undistorted_points = cv2.undistortPoints(points_2d, camera_intrinsics, distortion_coefficients, P=camera_intrinsics)  # Keep points in pixel coordinates
    return undistorted_points.reshape(-1, 2)  # Reshape back to (N, 2)