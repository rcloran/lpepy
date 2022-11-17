import argparse
import sys
import time

import cv2 as cv
import numpy as np

from lpepy import util

SUB_PIX_TERM_CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def make_real_world_points(columns, rows, block_size):
    """Create the real world points of the chessboard

    For example, when block_size is 1, the result would be:

    [[0, 0, 0], [1, 0, 0], [2, 0, 0], ...
     [0, 1, 0], [1, 1, 0], [2, 1, 0], ...
     ...
    ]
    """
    # The co-ordinates of grid intersections, in world co-ordinates. We orient
    # y down, and x to the right, matching the image co-ordinates.
    real_world_pts = np.zeros((columns * rows, 3), np.float32)
    real_world_pts[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    real_world_pts *= block_size

    return real_world_pts


def calibrate_camera(detected_points, shape, columns, rows, block_size):
    rwp = make_real_world_points(columns, rows, block_size)

    return cv.calibrateCamera(
        [rwp] * len(detected_points), detected_points, shape, None, None
    )


def draw_axes(img, corners, mtx, dist, columns, rows, block_size):
    axes = np.float32(
        [[0, 0, 0], [block_size, 0, 0], [0, block_size, 0], [0, 0, -block_size]]
    )
    board = make_real_world_points(columns, rows, block_size)
    ret, rvecs, tvecs = cv.solvePnP(board, corners, mtx, dist)
    imgpts, _ = cv.projectPoints(axes, rvecs, tvecs, mtx, dist)
    imgpts = np.int32(imgpts).reshape(4, 2)

    cv.line(img, imgpts[0], imgpts[1], (0, 0, 255), 5)
    cv.putText(img, "X", imgpts[1], cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
    cv.line(img, imgpts[0], imgpts[2], (0, 255, 0), 5)
    cv.putText(img, "Y", imgpts[2], cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
    cv.line(img, imgpts[0], imgpts[3], (255, 0, 0), 5)
    cv.putText(img, "Z", imgpts[3], cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))


def detection_loop(cam, columns, rows, block_size):
    scaled_width = 640

    mtx = None
    dist = None

    # Detected points, in image spaces
    detected_points = []

    last_save_time = 0

    while True:
        ret, img = cam.read()
        if not ret:
            print("Failed to get a frame")
            sys.exit(1)

        # Resize captured image for faster chessboard detection
        scale = scaled_width / img.shape[1]
        dim = (640, int(scale * img.shape[0]))
        img_small = cv.resize(img, dim)

        ret, corners = cv.findChessboardCorners(
            img_small,
            (columns, rows),
            flags=cv.CALIB_CB_ADAPTIVE_THRESH
            | cv.CALIB_CB_NORMALIZE_IMAGE
            | cv.CALIB_CB_FAST_CHECK,
        )

        if corners is not None:
            # Scale co-ordinates back up to original image size
            corners = corners / scale
            cv.drawChessboardCorners(img, (columns, rows), corners, ret)
            if mtx is not None:
                draw_axes(img, corners, mtx, dist, columns, rows, block_size)

        util.add_text(
            img, f"Saved {len(detected_points)} positions. Press any key to exit.", -1
        )

        cv.imshow("img", img)
        key = cv.waitKey(1)
        if key > 0:
            break

        if ret is True:
            # Save a picture at most once per second
            if time.time() - last_save_time > 1:
                img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                refined_corners = cv.cornerSubPix(
                    img_gray, corners, (11, 11), (-1, -1), SUB_PIX_TERM_CRITERIA
                )
                detected_points.append(refined_corners)
                _, mtx, dist, _, _ = calibrate_camera(
                    detected_points, img_gray.shape[::-1], columns, rows, block_size
                )
                last_save_time = time.time()

    return detected_points, (img.shape[1], img.shape[0])


def save_calibration(
    filename,
    columns,
    rows,
    block_size,
    camera_matrix,
    distortion_coefficients,
    rvecs,
    tvecs,
):
    extrinsic_parameters = np.array([np.concatenate(v) for v in zip(rvecs, tvecs)])

    storage = cv.FileStorage(filename, cv.FileStorage_WRITE)
    storage.write("board_width", columns)
    storage.write("board_height", rows)
    storage.write("square_size", block_size)
    storage.write("camera_matrix", camera_matrix)
    storage.write("distortion_coefficients", distortion_coefficients)
    storage.write("extrinsic_parameters", extrinsic_parameters)


def main():
    parser = argparse.ArgumentParser(
        prog="lpe-calibrate",
    )

    parser.add_argument(
        "--camera",
        default=0,
        type=int,
        help="The index of the camera to use (according to OpenCV)",
    )
    parser.add_argument(
        "--columns",
        default=9,
        type=int,
        help="Number of vertices in the grid in a rightwards direction (along the X axis)",
    )
    parser.add_argument(
        "--rows",
        default=6,
        type=int,
        help="Number of vertices in the grid in a downward direction (along the Y axis)",
    )
    parser.add_argument(
        "--block-size",
        default=20,
        type=int,
        help="The size of each block (distance between vertices along an axis), in mm",
    )
    parser.add_argument("--filename", default="calibration.yml")

    args = parser.parse_args()

    cam = util.initialize_camera(args.camera)
    detected_points, shape = detection_loop(
        cam, args.columns, args.rows, args.block_size
    )

    _, mtx, dist, rvecs, tvecs = calibrate_camera(
        detected_points, shape, args.columns, args.rows, args.block_size
    )

    save_calibration(
        args.filename, args.columns, args.rows, args.block_size, mtx, dist, rvecs, tvecs
    )


if __name__ == "__main__":
    main()
