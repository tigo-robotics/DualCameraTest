import cv2
import numpy as np
import os

def load_calibration_data(filename):
    if os.path.exists(filename):
        data = np.load(filename)
        return (data['mtx_left'], data['dist_left'], data['mtx_right'], 
                data['dist_right'], data['R'], data['T'])
    return None

def calibrate_cameras(cap, width, height):
    # Create lists to store the checkerboard images for left and right cameras
    images_left = []
    images_right = []

    # Define the checkerboard size
    checkerboard_size = (9, 6)

    # Prepare object points
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)

    # Termination criteria for cornerSubPix
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    while len(images_left) < 30 or len(images_right) < 30 :
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            return None

        # Split the frame into left and right views
        left_frame = frame[:, :width//2]
        right_frame = frame[:, width//2:]

        # Convert the frames to grayscale
        gray_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

        # Find the checkerboard corners in both images
        found_left, corners_left = cv2.findChessboardCorners(gray_left, checkerboard_size, None)
        found_right, corners_right = cv2.findChessboardCorners(gray_right, checkerboard_size, None)

        # If found in both images, refine the corners and add to the image lists
        if found_left and found_right:
            corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
            corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)
            
            images_left.append((gray_left, corners_left))
            images_right.append((gray_right, corners_right))
            
            # Draw the corners on the images for visual feedback
            cv2.drawChessboardCorners(left_frame, checkerboard_size, corners_left, found_left)
            cv2.drawChessboardCorners(right_frame, checkerboard_size, corners_right, found_right)
            
            print(f"Found checkerboard! Total images: {len(images_left)}")

        # Combine left and right frames
        display_frame = np.hstack((left_frame, right_frame))

        # Display the dual view with corner detection
        cv2.imshow("Dual Camera", display_frame)

        # Press 'ESC' to quit, 'c' to calibrate
        key = cv2.waitKey(1)
        if key == 27:  # ESC key
            return None
        elif key == ord('c') and len(images_left) > 10:  # 'c' key and enough images
            break

    cv2.destroyAllWindows()

    # Check if enough checkerboard images were found
    if len(images_left) > 10 and len(images_right) > 10:
        print("Calibrating cameras...")

        # Prepare object points and image points
        objpoints = [objp for _ in range(len(images_left))]
        imgpoints_left = [corners for _, corners in images_left]
        imgpoints_right = [corners for _, corners in images_right]

        # Calibrate left camera
        ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
            objpoints, imgpoints_left, gray_left.shape[::-1], None, None)

        # Calibrate right camera
        ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
            objpoints, imgpoints_right, gray_right.shape[::-1], None, None)

        # Stereo calibration
        flags = cv2.CALIB_FIX_INTRINSIC
        criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

        ret_stereo, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
            objpoints, imgpoints_left, imgpoints_right, mtx_left, dist_left, mtx_right, dist_right,
            gray_left.shape[::-1], criteria=criteria_stereo, flags=flags)

        # Save calibration results
        np.savez('stereo_calibration.npz', mtx_left=mtx_left, dist_left=dist_left,
                 mtx_right=mtx_right, dist_right=dist_right, R=R, T=T)
        print("Calibration data saved to 'stereo_calibration.npz'")

        return mtx_left, dist_left, mtx_right, dist_right, R, T
    else:
        print("Not enough checkerboards found. Cannot calibrate cameras.")
        return None

def compute_disparity(left_img, right_img):
    # Convert images to grayscale
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    # Create StereoSGBM object
    stereo = cv2.StereoSGBM_create(
        minDisparity=16,
        numDisparities=8*16,  # Increased from 16*16
        blockSize=7,
        P1=4 * 3 * 7**2,
        P2=8 * 3 * 7**2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    # Compute disparity
    #disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
    disparity = stereo.compute(left_gray, right_gray)
    # Normalize disparity
    disparity = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return disparity

def main():
    # Open the camera
    cap = cv2.VideoCapture(0)

    # Set the width and height
    width, height = 1280, 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Check for existing calibration file
    calibration_data = load_calibration_data('stereo_calibration.npz')

    if calibration_data is not None:
        print("Loaded existing calibration data.")
        mtx_left, dist_left, mtx_right, dist_right, R, T = calibration_data
    else:
        print("No existing calibration found. Starting calibration process...")
        calibration_data = calibrate_cameras(cap, width, height)
        if calibration_data is None:
            print("Calibration failed or was cancelled.")
            cap.release()
            return
        mtx_left, dist_left, mtx_right, dist_right, R, T = calibration_data

    # Compute stereo rectification
    image_size = (width // 2, height)
    R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(mtx_left, dist_left, mtx_right, dist_right, 
                                                               image_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, 
                                                               alpha=0)  # Set alpha to 0 for full rectification

    # Compute the undistortion and rectification maps
    map1_left, map2_left = cv2.initUndistortRectifyMap(mtx_left, dist_left, R1, P1, image_size, cv2.CV_32FC1)
    map1_right, map2_right = cv2.initUndistortRectifyMap(mtx_right, dist_right, R2, P2, image_size, cv2.CV_32FC1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Split the frame into left and right views
        right_frame = frame[:, :width//2]
        left_frame = frame[:, width//2:]

        # Undistort and rectify the images
        left_rectified = cv2.remap(left_frame, map1_left, map2_left, cv2.INTER_LINEAR)
        right_rectified = cv2.remap(right_frame, map1_right, map2_right, cv2.INTER_LINEAR)

        # Compute disparity
        disparity = compute_disparity(left_frame, right_frame)

        # Apply color map to disparity
        #disparity_color = cv2.applyColorMap(disparity, cv2.COLORMAP_JET)
        
        # Display the rectified images and disparity map
        cv2.imshow("Left Rectified", left_frame)
        cv2.imshow("Right Rectified", right_frame)
        cv2.imshow("Disparity Map", disparity)

        # Draw horizontal lines for epipolar visualization
        for i in range(0, left_frame.shape[0], 30):
            cv2.line(left_frame, (0, i), (left_frame.shape[1], i), (0, 255, 0), 1)
            cv2.line(right_frame, (0, i), (right_frame.shape[1], i), (0, 255, 0), 1)

        if cv2.waitKey(1) == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()