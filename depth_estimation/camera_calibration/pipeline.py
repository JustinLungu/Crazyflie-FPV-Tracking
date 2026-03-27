from __future__ import annotations

import glob
import os

import cv2
import numpy as np

from depth_estimation.camera_calibration.constants import (
    CHECKERBOARD,
    CORNER_OUTPUT_DIR,
    CRITERIA,
    DEFAULT_IMAGE_PATH,
    IMAGE_DIR,
    IMAGE_GLOB,
    OUTPUT_DIR,
    SQUARE_SIZE,
)
from depth_estimation.pipeline_base import DepthPipeline


class CameraCalibrationPipeline(DepthPipeline):
    name = "camera_calibration"

    @staticmethod
    def compute_reprojection_error(objpoints, imgpoints, rvecs, tvecs, camera_matrix, dist):
        total_error = 0.0
        total_points = 0

        for i in range(len(objpoints)):
            projected_imgpoints, _ = cv2.projectPoints(
                objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist
            )
            error = cv2.norm(imgpoints[i], projected_imgpoints, cv2.NORM_L2)
            n = len(projected_imgpoints)
            total_error += error * error
            total_points += n

        rmse = np.sqrt(total_error / total_points)
        return rmse

    def run_calibration(self) -> dict[str, object]:
        os.makedirs(CORNER_OUTPUT_DIR, exist_ok=True)
        image_dir_abs = os.path.abspath(IMAGE_DIR)

        objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        objp *= SQUARE_SIZE

        objpoints = []
        imgpoints = []

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

        ret, camera_matrix, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints,
            imgpoints,
            image_size,
            None,
            None,
        )

        rmse = self.compute_reprojection_error(objpoints, imgpoints, rvecs, tvecs, camera_matrix, dist)

        print("\n=== Calibration Results ===")
        print(f"Calibration success flag: {ret}")
        print("\nCamera matrix K:")
        print(camera_matrix)
        print("\nDistortion coefficients:")
        print(dist.ravel())
        print(f"\nReprojection RMSE: {rmse:.4f} pixels")

        camera_matrix_path = os.path.join(OUTPUT_DIR, "camera_matrix.npy")
        dist_coeffs_path = os.path.join(OUTPUT_DIR, "dist_coeffs.npy")
        np.save(camera_matrix_path, camera_matrix)
        np.save(dist_coeffs_path, dist)
        print("\nSaved:")
        print(f"  {camera_matrix_path}")
        print(f"  {dist_coeffs_path}")

        sample = cv2.imread(valid_images[0])
        h, w = sample.shape[:2]
        new_k, _roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist, (w, h), 1, (w, h))
        undistorted = cv2.undistort(sample, camera_matrix, dist, None, new_k)

        sample_original_path = os.path.join(OUTPUT_DIR, "calibration_sample_original.jpg")
        sample_undistorted_path = os.path.join(OUTPUT_DIR, "calibration_sample_undistorted.jpg")
        cv2.imwrite(sample_original_path, sample)
        cv2.imwrite(sample_undistorted_path, undistorted)
        print(f"  {sample_original_path}")
        print(f"  {sample_undistorted_path}")
        print(f"  Corner detections: {CORNER_OUTPUT_DIR}")

        return {
            "success_flag": float(ret),
            "rmse": float(rmse),
            "camera_matrix_path": camera_matrix_path,
            "dist_coeffs_path": dist_coeffs_path,
            "sample_original_path": sample_original_path,
            "sample_undistorted_path": sample_undistorted_path,
            "used_images": valid_images,
        }

    def run_checkerboard_detection(self, image_path: str | None = None) -> bool:
        chosen_image_path = image_path or DEFAULT_IMAGE_PATH

        img = cv2.imread(chosen_image_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {chosen_image_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

        print(f"Detected: {found}")

        cv2.drawChessboardCorners(img, CHECKERBOARD, corners, found)
        cv2.imshow("corners", img)
        cv2.waitKey(0)

        return bool(found)
