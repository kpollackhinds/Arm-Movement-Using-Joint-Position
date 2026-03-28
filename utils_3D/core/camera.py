class Camera:

    def __init__(self, name: str = "", position=None, rotation=None, camera_matrix=None, projection_matrix=None, distortion_coefficients=None):
        self.name = name
        self.position = position
        self.rotation = rotation
        self._camera_matrix = camera_matrix
        self.projection_matrix = projection_matrix
        self.distortion_coefficients = distortion_coefficients

        if self.projection_matrix is None:
            if self.position is None and self.rotation is None:
                raise ValueError("Must provide either projection_matrix or both position and rotation.")
            
    @property
    def camera_matrix(self):
        if self._camera_matrix is None:
            #TODO Try to compute the camera matrix from the projection matrix if possible
            raise ValueError("Camera matrix not set.")
        return self._camera_matrix

    