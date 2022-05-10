import matplotlib.pyplot as plt
import numpy as np
import cv2

from pyapp.DataAcquisition import DataAcquisition
from config import config
from pyapp.calibration_helpers import get_mat_c_to_w_series, get_lut_projection_pixel, get_lut_pixel_image,  get_lut_projection_int_pixel
from python.common.UtilImage import draw_disk


def drawPoints(image, pts, colors):
    for pt, color in zip(pts, colors):
        cv2.circle(image, tuple(pt[0]), 5, color, -1)

def drawLines(image, lines, colors):
    _, c, _ = image.shape
    for r, color in zip(lines, colors):
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        cv2.line(image, (x0, y0), (x1, y1), color, 1)

def blobDetection(frame):
    frame = (frame /256).astype('uint8')
    ret2, thresh = cv2.threshold(frame, 4.5, 256, cv2.THRESH_BINARY_INV)

    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 0;
    params.maxThreshold = 255;

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 15

    detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(thresh)
    # im_with_keypoints = cv2.drawKeypoints(thresh, keypoints, np.array([]), (0 ,0 ,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)    # left:
    # cv2.imshow("Keypoints", im_with_keypoints)
    # cv2.waitKey(0)

    #frameCopy = np.copy(data.acquisitions['ahat_depth_cam_ab_frames'][0])
    infraredPoints = []
    for keypoint in keypoints:
        coord = np.array(keypoint.pt)
        infraredPoints.append(coord)
        print(coord)
        #draw_disk(frame, coord[0], coord[1], 0, size=1)

    return infraredPoints

def projectLineToGreyscale(infraredPoints, imgIR, data):
                           #,imgLeft, imgRight):


    lutInfrared = data.acquisitions["ahat_depth_cam" + "_lut_projection"]
    lutGreyScaleLeft = data.acquisitions["vl_front_left_cam" + "_lut_projection"]
    lutGreyScaleRight = data.acquisitions["vl_front_right_cam" + "_lut_projection"]

    timestamp1 = data.acquisitions["ahat_depth_cam"].index[0]
    timestamp2 = data.acquisitions["vl_front_left_cam"].index[0]
    timestamp3 = data.acquisitions["vl_front_right_cam"].index[0]

    serie1 = data.acquisitions["ahat_depth_cam"].loc[timestamp1]
    serie2 = data.acquisitions["vl_front_left_cam"].loc[timestamp2]
    serie3 = data.acquisitions["vl_front_right_cam"].loc[timestamp3]

    extrinsic1 = get_mat_c_to_w_series(serie1)
    extrinsic2 = get_mat_c_to_w_series(serie2)
    extrinsic3 = get_mat_c_to_w_series(serie3)

    mat_w_to_c1 = np.linalg.inv(extrinsic1)
    mat_w_to_c2 = np.linalg.inv(extrinsic2)
    mat_w_to_c3 = np.linalg.inv(extrinsic3)

    color = tuple(np.random.randint(0, 255, 3).tolist())
    for sphere in infraredPoints:
        #cameraCoord3D = get_lut_projection_pixel(lutInfrared, sphere[0], sphere[1])
        #infraredPoints = np.int32(infraredPoints)

        IRcameraCoord3D = get_lut_projection_pixel(lutInfrared, sphere[0], sphere[1])
        # print(sphere)
        print(IRcameraCoord3D)

        #worldCoord = np.matmul(extrinsic1, IRcameraCoord3D)
        # leftGreyscaleCameraCoord3D = np.mul(mat_w_to_c2, worldCoord)
        # rightGreyscaleCameraCoord3D = np.mul(mat_w_to_c3, worldCoord)
        imageCoord2D = get_lut_pixel_image(lutInfrared, IRcameraCoord3D[0], IRcameraCoord3D[1], IRcameraCoord3D[2])
        print(imageCoord2D)

        # leftImageCoord2D = get_lut_pixel_image(lutGreyScaleLeft, leftGreyscaleCameraCoord3D[0], leftGreyscaleCameraCoord3D[1], leftGreyscaleCameraCoord3D[2])
        # rightImageCoord2D = get_lut_pixel_image(rightGreyscaleCameraCoord3D, rightGreyscaleCameraCoord3D[0], rightGreyscaleCameraCoord3D[1], rightGreyscaleCameraCoord3D[2])

      #  drawPoints(imgIR, imageCoord2D, color)
        # drawPoints(imgLeft, leftImageCoord2D, color)
        # drawPoints(imgRight, rightImageCoord2D, color)

    return imgIR,
           #imgLeft, imgRight

if  __name__ == '__main__':
    data = DataAcquisition()
    data.load_data(config.get_filename("optical_sphere"))

    frame1 = np.copy(data.acquisitions['vl_front_left_cam_frames'][0])
    frame2 = np.copy(data.acquisitions['vl_front_right_cam_frames'][0])

    frame3 = np.copy(data.acquisitions['lt_depth_cam_ab_frames'][0])
    frame4 = np.copy(data.acquisitions['ahat_depth_cam_ab_frames'][0])
    irpoints = blobDetection(frame4)
    #irpoints = [[238, 221], [253, 208], [216, 200], [240, 188]]
    resImage = projectLineToGreyscale(irpoints, frame4.astype(np.uint8), data)

    #plt.imshow(resImage)
    #plt.show()