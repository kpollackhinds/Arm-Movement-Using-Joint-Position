import numpy as np

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
