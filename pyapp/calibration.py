import cv2
import glob
import numpy as np


def calibrate_camera(images):

    # Pixel coordinates
    points2D = []

    # coordinates of the checkerboard in checkerboard world space.
    points3D = []

    # checkerboard pattern detector criteria.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    #number of rows and colums of the chessboard
    rows = 5
    cols = 8

    # square coordinates in real world space
    obj = np.zeros((rows * cols, 3), np.float32)
    obj[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)

    # frame size
    width = images[0].shape[1]
    height = images[0].shape[0]
    for frame in images:
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        success, corners = cv2.findChessboardCorners(grayFrame, (rows, cols), None)
        if success == True:
            # Convolution size for detecting corners
            conv_size = (11, 11)
            corners = cv2.cornerSubPix(grayFrame, corners, conv_size, (-1, -1), criteria)
            cv2.drawChessboardCorners(frame, (rows, cols), corners, success)
            #cv2.imshow('image', frame)
            cv2.waitKey(500)
            points2D.append(corners)
            points3D.append(obj)

    success, cameraMatrix, distCoefs, _, _ = cv2.calibrateCamera(points3D, points2D, (width, height), None, None)

    return cameraMatrix, distCoefs

    def stereo_calibrate(cameraMatrix1, distCoefs1, cameraMatrix2, distCoefs2, folderImages):

        image_names = glob.glob(folderImages)
        image_names = sorted(image_names)
        images_names1 = image_names[:len(image_names) // 2]
        images_names2 = image_names[len(image_names) // 2:]

        images_names1 = []
        images_names2 = []
        for img1, img2 in zip(images_names1, images_names2):
            _im = cv2.imread(img1, 1)
            images_names1.append(_im)

            _im = cv2.imread(img2, 1)
            images_names2.append(_im)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

        # number of rows and colums of the chessboard
        rows = 5
        columns = 8

        # coordinates of squares in the checkerboard world space
        objp = np.zeros((rows * columns, 3), np.float32)
        objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)

        # frame dimensions. Frames should be the same size.
        width = images_names1[0].shape[1]
        height = images_names1[0].shape[0]

        # Pixel coordinates of checkerboards
        imgpoints_left = []  # 2d points in image plane.
        imgpoints_right = []

        # coordinates of the checkerboard in checkerboard world space.
        objpoints = []  # 3d point in real world space

        for frame1, frame2 in zip(images_names1, images_names2):
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            c_ret1, corners1 = cv2.findChessboardCorners(gray1, (5, 8), None)
            c_ret2, corners2 = cv2.findChessboardCorners(gray2, (5, 8), None)

            if c_ret1 == True and c_ret2 == True:
                corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
                corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

                cv2.drawChessboardCorners(frame1, (5, 8), corners1, c_ret1)
                cv2.imshow('img', frame1)

                cv2.drawChessboardCorners(frame2, (5, 8), corners2, c_ret2)
                cv2.imshow('img2', frame2)
                cv2.waitKey(500)

                objpoints.append(objp)
                imgpoints_left.append(corners1)
                imgpoints_right.append(corners2)

        stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC
        success, CM1, dist1, CM2, dist2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, cameraMatrix1,
                                                                     distCoefs1,
                                                                     cameraMatrix2, distCoefs2, (width, height), criteria=criteria,
                                                                     flags=stereocalibration_flags)


        return R, T

# if __name__ == '__main__':
#
#      imageFolder = ''
    # imagesNames = sorted(glob.glob(images_folder))
    # images = []x
    # for name in images_names:
    #     image = cv2.imread(name, 1)
    #     images.append(image)
    #calibrate_camera(images)