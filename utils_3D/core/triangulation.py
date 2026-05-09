import numpy as np
from numpy.linalg import svd
from utils_3D.core.point import Point3D
from typing import Optional

def triangulate(image_point_correspondences: np.ndarray, projection_matrices: np.ndarray, weights: Optional[np.ndarray] = None, return_homogeneous: bool = False) -> "Point3D | np.ndarray":
    """
    Args:
        image_point_correspondences: An (N, 2) array of 2D points in the image planes of N cameras (Non homogeneous).
        projection_matrices: An (N, 3, 4) array of projection matrices for the N cameras.
        weights: An optional (N,) array of weights for each correspondence.
        return_homogeneous: If True, returns the homogeneous coordinates of the triangulated point. Otherwise, returns Cartesian coordinates.
    """

    # Image points should be of shape (N, 2) and projection matrices should be of shape (N, 3, 4)
    if image_point_correspondences.shape[0] != projection_matrices.shape[0]:
        raise ValueError("Number of image points must match number of projection matrices.")
    
    if image_point_correspondences.shape[1] != 2 or image_point_correspondences.ndim != 2:
        raise ValueError("Image points should be of shape (N, 2).")
    
    if projection_matrices.shape[1:] != (3, 4) or projection_matrices.ndim != 3:
        raise ValueError("Projection matrices should be of shape (N, 3, 4).")
    
    A = build_dlt_matrix(image_point_correspondences, projection_matrices, weights)
    
    _, S, Vt = svd(A)

    # if S[-1]/S[-2] > 1e-8:
    #     raise ValueError("Triangulation may be unstable. The smallest singular value is not sufficiently smaller than the second smallest: S[-1] = {}, S[-2] = {}, ratio = {}".format(S[-1], S[-2], S[-1]/S[-2]))
    
    X = Vt[-1]



    if not return_homogeneous:
        return Point3D.from_homogeneous(X)

    return X

def build_dlt_matrix(image_point_correspondances: np.ndarray, projection_matrices: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
    # This function will build the DLT matrix A for triangulation based on the input image points and projection matrices
    N = image_point_correspondances.shape[0]
    A = np.zeros((2 * N, 4))

    if weights is not None:
        if weights.shape[0] != N:
            raise ValueError("Weights array must have the same length as the number of correspondences.")
    else:
        weights = np.ones(N)
        
    for i in range(N):
        x = image_point_correspondances[i, 0]
        y = image_point_correspondances[i, 1]
        P = projection_matrices[i]

        A[2 * i] = weights[i] * (y * P[2] - P[1])
        A[2 * i + 1] = weights[i] * (P[0] - x * P[2])

    return A    


# TODO: Add non-linear optimization step to refine the triangulated point using the reprojection error as the cost function.