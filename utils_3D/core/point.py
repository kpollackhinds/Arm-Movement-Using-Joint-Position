import numpy as np


class Point3D:
    __slots__ = ("x", "y", "z", "is_homogeneous")

    def __init__(self, x: float, y: float, z: float, is_homogeneous: bool = False):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.is_homogeneous = is_homogeneous

    # ── Factories ──────────────────────────────────────────────────────────────

    @classmethod
    def from_array(cls, arr) -> "Point3D":
        arr = np.asarray(arr, dtype=float).ravel()
        if arr.size != 3:
            raise ValueError(f"Expected 3 elements, got {arr.size}.")
        return cls(arr[0], arr[1], arr[2])

    @classmethod
    def from_homogeneous(cls, arr) -> "Point3D":
        arr = np.asarray(arr, dtype=float).ravel()
        if arr.size != 4:
            raise ValueError(f"Expected 4 elements for homogeneous coords, got {arr.size}.")
        w = arr[3]
        if abs(w) < 1e-10:
            raise ValueError("Homogeneous w is zero — point is at infinity.")
        return cls(arr[0] / w, arr[1] / w, arr[2] / w)

    # ── Conversions ────────────────────────────────────────────────────────────

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)

    def to_homogeneous(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z, 1.0], dtype=float)

    def __array__(self, dtype=None):
        arr = self.to_array()
        return arr if dtype is None else arr.astype(dtype)

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z

    def __len__(self) -> int:
        return 3

    def __getitem__(self, index):
        return (self.x, self.y, self.z)[index]

    # ── Arithmetic ─────────────────────────────────────────────────────────────

    def __add__(self, other) -> "Point3D":
        o = np.asarray(other, dtype=float).ravel()
        return Point3D(self.x + o[0], self.y + o[1], self.z + o[2])

    def __radd__(self, other) -> "Point3D":
        return self.__add__(other)

    def __sub__(self, other) -> "Point3D":
        o = np.asarray(other, dtype=float).ravel()
        return Point3D(self.x - o[0], self.y - o[1], self.z - o[2])

    def __rsub__(self, other) -> "Point3D":
        o = np.asarray(other, dtype=float).ravel()
        return Point3D(o[0] - self.x, o[1] - self.y, o[2] - self.z)

    def __mul__(self, scalar) -> "Point3D":
        s = float(scalar)
        return Point3D(self.x * s, self.y * s, self.z * s)

    def __rmul__(self, scalar) -> "Point3D":
        return self.__mul__(scalar)

    def __truediv__(self, scalar) -> "Point3D":
        s = float(scalar)
        return Point3D(self.x / s, self.y / s, self.z / s)

    def __neg__(self) -> "Point3D":
        return Point3D(-self.x, -self.y, -self.z)

    # ── Comparison ─────────────────────────────────────────────────────────────

    def __eq__(self, other) -> bool:
        if isinstance(other, Point3D):
            return self.x == other.x and self.y == other.y and self.z == other.z
        return NotImplemented

    # ── Geometry ───────────────────────────────────────────────────────────────

    def distance_to(self, other: "Point3D") -> float:
        o = other if isinstance(other, Point3D) else Point3D.from_array(other)
        return float(np.sqrt((self.x - o.x) ** 2 + (self.y - o.y) ** 2 + (self.z - o.z) ** 2))

    def norm(self) -> float:
        return float(np.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2))

    # ── Display ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"Point3D(x={self.x:.4f}, y={self.y:.4f}, z={self.z:.4f})"
