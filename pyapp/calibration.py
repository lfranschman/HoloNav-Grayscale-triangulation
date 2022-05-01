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
        return

# if __name__ == '__main__':
#
#      imageFolder = ''
    # imagesNames = sorted(glob.glob(images_folder))
    # images = []x
    # for name in images_names:
    #     image = cv2.imread(name, 1)
    #     images.append(image)
    #calibrate_camera(images)