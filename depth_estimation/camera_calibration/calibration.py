import glob
import os

import cv2
import numpy as np

# IMPORTANT:
# This must match your printed checkerboard INTERNAL corners
CHECKERBOARD = (10, 7)

# Real square size in meters
SQUARE_SIZE = 0.025  # 25 mm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Folder structure for calibration inputs/outputs
CALIBRATION_DIR = os.path.join(SCRIPT_DIR, "calibration_images")
IMAGE_DIR = os.path.join(CALIBRATION_DIR, "input_images")
IMAGE_GLOB = os.path.join(IMAGE_DIR, "**", "*.jpg")
OUTPUT_DIR = CALIBRATION_DIR
CORNER_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "corner_detections")

# Corner refinement criteria
CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001,
)


def compute_reprojection_error(objpoints, imgpoints, rvecs, tvecs, K, dist):
    total_error = 0.0
    total_points = 0

    for i in range(len(objpoints)):
        projected_imgpoints, _ = cv2.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], K, dist
        )
        error = cv2.norm(imgpoints[i], projected_imgpoints, cv2.NORM_L2)
        n = len(projected_imgpoints)
        total_error += error * error
        total_points += n

    rmse = np.sqrt(total_error / total_points)
    return rmse


def main():
    os.makedirs(CORNER_OUTPUT_DIR, exist_ok=True)
    image_dir_abs = os.path.abspath(IMAGE_DIR)

    # Prepare object points in checkerboard coordinates
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    objpoints = []  # 3D points in real world
    imgpoints = []  # 2D points in image plane

    image_paths = sorted(glob.glob(IMAGE_GLOB, recursive=True))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {IMAGE_GLOB}")

    image_size = None
    valid_images = []

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Skipping unreadable image: {path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if image_size is None:
            image_size = gray.shape[::-1]
        elif image_size != gray.shape[::-1]:
            print(f"Skipping {path}: resolution mismatch {gray.shape[::-1]} vs {image_size}")
            continue

        found, corners = cv2.findChessboardCorners(
            gray,
            CHECKERBOARD,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )

        if not found:
            print(f"Checkerboard NOT found: {path}")
            continue

        corners_refined = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            CRITERIA,
        )

        objpoints.append(objp.copy())
        imgpoints.append(corners_refined)
        valid_images.append(path)

        vis = img.copy()
        cv2.drawChessboardCorners(vis, CHECKERBOARD, corners_refined, found)
        relative_name = os.path.relpath(os.path.abspath(path), image_dir_abs)
        safe_name = relative_name.replace(os.sep, "__")
        if os.altsep:
            safe_name = safe_name.replace(os.altsep, "__")
        corner_filename = f"corners_{len(valid_images):03d}_{safe_name}"
        corner_output_path = os.path.join(CORNER_OUTPUT_DIR, corner_filename)
        saved = cv2.imwrite(corner_output_path, vis)
        if not saved:
            print(f"Failed to save corner visualization: {corner_output_path}")
        cv2.imshow("corners", vis)
        cv2.waitKey(150)

    cv2.destroyAllWindows()

    print(f"\nValid images used: {len(valid_images)} / {len(image_paths)}")
    if len(valid_images) < 10:
        raise RuntimeError("Too few valid calibration images. Need more variety / better detections.")

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None,
    )

    rmse = compute_reprojection_error(objpoints, imgpoints, rvecs, tvecs, K, dist)

    print("\n=== Calibration Results ===")
    print(f"Calibration success flag: {ret}")
    print("\nCamera matrix K:")
    print(K)
    print("\nDistortion coefficients:")
    print(dist.ravel())
    print(f"\nReprojection RMSE: {rmse:.4f} pixels")

    camera_matrix_path = os.path.join(OUTPUT_DIR, "camera_matrix.npy")
    dist_coeffs_path = os.path.join(OUTPUT_DIR, "dist_coeffs.npy")
    np.save(camera_matrix_path, K)
    np.save(dist_coeffs_path, dist)
    print("\nSaved:")
    print(f"  {camera_matrix_path}")
    print(f"  {dist_coeffs_path}")

    # Save one undistorted example
    sample = cv2.imread(valid_images[0])
    h, w = sample.shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    undistorted = cv2.undistort(sample, K, dist, None, new_K)

    sample_original_path = os.path.join(OUTPUT_DIR, "calibration_sample_original.jpg")
    sample_undistorted_path = os.path.join(OUTPUT_DIR, "calibration_sample_undistorted.jpg")
    cv2.imwrite(sample_original_path, sample)
    cv2.imwrite(sample_undistorted_path, undistorted)
    print(f"  {sample_original_path}")
    print(f"  {sample_undistorted_path}")
    print(f"  Corner detections: {CORNER_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
