import numpy as np
from numpy.linalg import svd

def triangulate(image_point_correspondences: np.ndarray, projection_matrices: np.ndarray, return_homogeneous: bool = False) -> np.ndarray:
    """
    Args:
        image_point_correspondences: An (N, 2) array of 2D points in the image planes of N cameras (Non homogeneous).
        projection_matrices: An (N, 3, 4) array of projection matrices for the N cameras.
        return_homogeneous: If True, returns the homogeneous coordinates of the triangulated point. Otherwise, returns Cartesian coordinates.
    """

    # Image points should be of shape (N, 2) and projection matrices should be of shape (N, 3, 4)
    if image_point_correspondences.shape[0] != projection_matrices.shape[0]:
        raise ValueError("Number of image points must match number of projection matrices.")
    
    if image_point_correspondences.shape[1] != 2 or image_point_correspondences.ndim != 2:
        raise ValueError("Image points should be of shape (N, 2).")
    
    if projection_matrices.shape[1:] != (3, 4) or projection_matrices.ndim != 3:
        raise ValueError("Projection matrices should be of shape (N, 3, 4).")
    
    A = build_dlt_matrix(image_point_correspondences, projection_matrices)
    
    _, _, Vt = svd(A)
    X = Vt[-1]

    if not return_homogeneous:
        if X[3] < 1e-6:
            raise ValueError("Triangulated point is at infinity (homogeneous coordinate is zero).")
        
        X = X / X[3]  # Convert from homogeneous to Cartesian coordinates
        X = X[:3]

    return X

def build_dlt_matrix(image_point_correspondances: np.ndarray, projection_matrices: np.ndarray) -> np.ndarray:
    # This function will build the DLT matrix A for triangulation based on the input image points and projection matrices
    N = image_point_correspondances.shape[0]
    A = np.zeros((2 * N, 4))

    for i in range(N):
        x = image_point_correspondances[i, 0]
        y = image_point_correspondances[i, 1]
        P = projection_matrices[i]

        A[2 * i] = y * P[2] - P[1]
        A[2 * i + 1] = P[0] - x * P[2]

    return A    