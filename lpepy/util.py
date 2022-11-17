import cv2 as cv


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
