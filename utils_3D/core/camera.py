class Camera:

    def __init__(self, name: str = "", position=None, rotation=None, projection_matrix=None):
        self.name = name
        self.position = position
        self.rotation = rotation
        self.projection_matrix = projection_matrix

        if self.projection_matrix is None:
            if self.position is None and self.rotation is None:
                raise ValueError("Must provide either projection_matrix or both position and rotation.")