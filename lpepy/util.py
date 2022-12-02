import cv2 as cv
import numpy as np


class Camera:
    """Represents the location of a camera in the world"""

    def __init__(
        self,
        intrinsic,
        distortion,
        rotation,
        translation,
    ):
        if rotation.shape == (3, 1):
            # This is a rotation vector. Make it a rotation matrix.
            rotation, _ = cv.Rodrigues(rotation)

        self.intrinsic = intrinsic
        self.distortion = distortion
        self.rotation = rotation
        self.translation = translation

    @property
    def rotation_vector(self):
        r_vec, _ = cv.Rodrigues(self.rotation)
        return r_vec

    @property
    def transformation(self):
        """Matrix that transforms world coordinates to this camera's system"""
        r = np.identity(4)
        r[:3, :3] = self.rotation
        r[:3, 3:4] = self.translation

        return r

    @property
    def projection(self):
        return self.intrinsic @ self.transformation[:3]

    @property
    def inverse(self):
        """Matrix that transforms from this camera's coordinate system to the world's

        This can be useful when trying to draw this camera in the world.
        """
        r = np.identity(4)
        inv_rot = self.rotation[:3, :3].T
        r[:3, :3] = inv_rot
        r[:3, 3] = -(inv_rot).dot(self.translation)

        return r

    def __matmul__(self, other):
        return self.transformation @ other.transformation


def initialize_camera(idx):
    """Open the camera, and attempt to read frames until successful"""
    cam = cv.VideoCapture(idx)
    print(f"Opened camera {idx} with {cam.getBackendName()}")

    retries = 5
    while retries > 0:
        ret, _ = cam.read()
        if ret:
            return cam

    return None


def add_text(img, text, line):
    """Add text to an OpenCV image"""
    if line < 0:
        y = img.shape[0] + (30 * line)
    else:
        y = 30 * line
    cv.putText(img, text, (11, y + 1), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
    cv.putText(img, text, (10, y), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
