import argparse

import cv2 as cv
import numpy as np

from lpepy import led, util, world

CAM_WIN = "camera"


def load_calibration(filename):
    fs = cv.FileStorage(cv.samples.findFile(filename), cv.FILE_STORAGE_READ)
    mtx = fs.getNode("camera_matrix").mat()
    dist = fs.getNode("distortion_coefficients").mat()

    return mtx, dist


def read(camera: cv.VideoCapture):
    ret, img = camera.read()
    if not ret:
        raise Exception("Failed to read a frame from the camera")
    return img


def add_text(img, text, line):
    if line < 0:
        y = img.shape[0] + (30 * line)
    else:
        y = 30 * line
    cv.putText(img, text, (11, y + 1), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
    cv.putText(img, text, (10, y), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))


def show_pattern(leds, pattern, idx):
    leds.fill([0, 0, 0])

    leds[idx : idx + len(pattern)] = pattern

    leds.show()


def find_highlights(img, masks):
    kernel = np.ones((3, 3), np.uint8)
    img = cv.GaussianBlur(img, (9, 9), 0)
    for mask in masks:
        mask = cv.erode(mask, kernel, iterations=1)  # Try remove some noise
        mask = cv.dilate(mask, kernel, iterations=2)
        img = cv.bitwise_and(img, img, mask=mask)

    _, max_v, _, max_p = cv.minMaxLoc(img)

    return max_v, max_p


def wait_for_capture(camera, already_captured):
    while True:
        img = read(camera)

        if already_captured != 0:
            add_text(img, f"Captured points from {already_captured} locations", 1)

        add_text(img, "Press c to begin point capture", -2)
        add_text(img, "Press ESC to quit", -1)

        cv.imshow(CAM_WIN, img)
        key = cv.waitKey(1)

        if key < 0:
            continue
        if chr(key).lower() == "c":
            return True
        if key == 27:
            return False


def capture_image_points(camera, leds):
    img = read(camera)
    backsub = cv.createBackgroundSubtractorKNN()
    backsub.apply(img)

    points = [None] * len(leds)
    rgb = [[70, 0, 0], [0, 60, 0], [0, 0, 45]]

    for i in range(0, len(leds), 3):
        show_pattern(leds, rgb, i)
        key = cv.waitKey(250)  # Wait for the world to catch up
        if key == 27:
            return
        img = read(camera)

        fgmask = backsub.apply(img)

        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        r = cv.inRange(
            hsv, np.array([165, 120, 50]), np.array([180, 255, 255])
        ) | cv.inRange(hsv, np.array([0, 120, 50]), np.array([15, 255, 255]))
        g = cv.inRange(hsv, np.array([45, 120, 50]), np.array([75, 255, 255]))
        b = cv.inRange(hsv, np.array([105, 120, 50]), np.array([135, 255, 255]))

        channels = [  # (filtered image, overlay colour)
            (r, (0, 0, 255)),
            (g, (0, 255, 0)),
            (b, (255, 0, 0)),
        ]

        for offset, (chan, annotation_colour) in enumerate(channels):
            max_v, max_p = find_highlights(gray, [fgmask, chan])

            if max_v > 127 and i + offset < len(leds):
                cv.circle(img, max_p, 20, annotation_colour, 2)
                points[i + offset] = max_p

        add_text(img, f"Identified {len(points) - points.count(None)} LEDs", -2)
        add_text(img, "Press ESC to abort", -1)
        cv.imshow(CAM_WIN, img)

    # Using just a bright white point may be more reliable
    missing = [i for i, p in enumerate(points) if p is None]
    for i in missing:
        show_pattern(leds, [[255, 255, 255]], i)
        key = cv.waitKey(250)  # Wait for the world to catch up
        if key == 27:
            return
        img = read(camera)

        fgmask = backsub.apply(img)

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        max_v, max_p = find_highlights(gray, [fgmask])

        if max_v > 160:
            cv.circle(img, max_p, 20, (255, 255, 255), 2)
            points[i] = max_p

        add_text(img, f"Identified {len(points) - points.count(None)} LEDs", -2)
        add_text(img, "Press ESC to abort", -1)
        cv.imshow(CAM_WIN, img)

    return points


def find_common(iter_1, iter_2):
    """Find elements which both lists have set"""
    assert len(iter_1) == len(iter_2)

    common_tuples = [
        (i, a, b)
        for i, (a, b) in enumerate(zip(iter_1, iter_2))
        if not (a is None or b is None)
    ]

    if len(common_tuples) == 0:
        return [], np.float64([]), np.float64([])

    valids, iter1, iter2 = zip(*common_tuples)
    return valids, np.float64(iter1), np.float64(iter2)


def initial_triangulation(pts1, pts2, mtx, dist):
    """Do the initial triangulation

    The scale that is decided upon here becomes the canonical world scale
    """
    assert len(pts1) == len(pts2)

    result_pts = [None] * len(pts1)
    valids, pts1, pts2 = find_common(pts1, pts2)

    essential_mtx, mask = cv.findEssentialMat(pts1, pts2, mtx, dist, mtx, dist)
    ret, c2_rotation, c2_translation, mask2, world_pts = cv.recoverPose(
        essential_mtx, pts1, pts2, mtx, distanceThresh=np.Inf, mask=mask
    )
    world_pts = (world_pts[:3] / world_pts[3]).T

    c2 = util.Camera(mtx, dist, c2_rotation, c2_translation)

    # c2_translation is unit vector in the right direction. This resolves scale
    # based on world co-ordinates
    ret, c2_rotation_vec, c2_translation, inliers = cv.solvePnPRansac(
        world_pts, pts2, mtx, dist, c2.rotation_vector, c2_translation, True
    )

    # Update
    c2 = util.Camera(mtx, dist, c2_rotation_vec, c2_translation)

    for i, p in enumerate(world_pts):
        result_pts[valids[i]] = p

    return result_pts, c2


def pnp_ransac(world_points, image_points, mtx, dist):
    assert len(world_points) == len(image_points)
    valids, world_pts_, img_pts_ = find_common(world_points, image_points)
    ret, c1_rotation_vec, c1_translation, inliers = cv.solvePnPRansac(
        world_pts_, img_pts_, mtx, dist, np.zeros(3), np.zeros(3), False
    )
    return util.Camera(mtx, dist, c1_rotation_vec, c1_translation)


def triangulate(world_points, pts1, pts2, mtx, dist):
    """Triangulate and calculate pose, using prior world points for reference"""
    # We received a canonical set of world points, we should estimate our pose
    # based on those to avoid scaling issues.
    # First, we need to know which points work...
    assert len(world_points) == len(pts1) == len(pts2)
    result_pts = [None] * len(world_points)

    c1 = pnp_ransac(world_points, pts1, mtx, dist)
    c2 = pnp_ransac(world_points, pts2, mtx, dist)
    valids, pts1_, pts2_ = find_common(pts1, pts2)

    # I've experimented with only considering inliers (result from pnp) in
    # triangulation by filtering them out here, but it seems that the outlier
    # filtering in our world points set gives a better result.
    #
    # If we filter out too early, the system fixates on results from the first
    # few camera positions, always discarding positions from newer camera
    # positions so that we never consider them as possibilities when trying to
    # refine world positions.

    if len(valids) == 0:
        return result_pts

    new_world = cv.triangulatePoints(c1.projection, c2.projection, pts1_.T, pts2_.T)
    new_world = (new_world[:3] / new_world[3]).T

    for i, p in enumerate(new_world):
        result_pts[valids[i]] = p

    return result_pts, c2


def main():
    parser = argparse.ArgumentParser(
        prog="lpe-capture", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--camera",
        default=0,
        type=int,
        help="The index of the camera to use (according to OpenCV)",
    )
    parser.add_argument(
        "--calibration-file",
        metavar="FILE",
        default=util.get_default_calibration_file(),
        help="Name of file containing camera calibration data (created with lpe-calibrate)",
    )
    parser.add_argument(
        "--leds", default=100, type=int, help="Number of LEDs in the LED string"
    )
    parser.add_argument(
        "--wled-address",
        default="255.255.255.255",
        help="Address of host running WLED which will display LEDs",
    )
    parser.add_argument(
        "--wled-port",
        type=int,
        default=21324,
        help="Port number that WLED is listening on",
    )
    parser.add_argument(
        "output_file",
        metavar="FILE",
        help="JSON file to which resulting coordinates will be written",
    )
    args = parser.parse_args()

    camera = util.initialize_camera(args.camera)

    mtx, dist = load_calibration(args.calibration_file)

    leds = led.WLEDControl(args.leds, args.wled_address, args.wled_port)

    image_point_collection = []
    world_points = world.PointEstimate()

    while True:
        if not wait_for_capture(camera, len(image_point_collection)):
            break

        pts = capture_image_points(camera, leds)
        if pts is None:
            break

        image_point_collection.append(pts)

        if len(image_point_collection) == 1:
            # We can't do anything with one image's points (in the future we
            # might be able to display a representation of the 2D stuff?)
            continue
        elif len(image_point_collection) == 2:
            new_world_points, camera_2 = initial_triangulation(
                image_point_collection[0], image_point_collection[1], mtx, dist
            )
            world_points.add(new_world_points)
        else:
            for prev_img_pts in image_point_collection[:-1]:
                combined_world = world_points.combined()
                new_world_points, camera_2 = triangulate(
                    combined_world,
                    prev_img_pts,
                    pts,
                    mtx,
                    dist,
                )
                world_points.add(new_world_points)

        with open(args.output_file, "w") as f:
            f.write(world_points.json())

    cv.destroyAllWindows()
    camera.release()


if __name__ == "__main__":
    main()
