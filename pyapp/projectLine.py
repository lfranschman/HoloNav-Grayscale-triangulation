import math

import matplotlib.pyplot as plt
import numpy as np
import cv2

from pyapp.DataAcquisition import DataAcquisition
from config import config
from pyapp.calibration_helpers import get_mat_c_to_w_series, get_lut_projection_pixel, get_lut_pixel_image,  get_lut_projection_int_pixel
from python.common.UtilImage import draw_disk

def houghTransform(img):
    cimg = cv2.medianBlur(img, 5)
    # cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
                              param1=50, param2=30, minRadius=0, maxRadius=0)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
    cv2.imshow('detected circles', cimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img

def drawPoints(image, points):
    lastCoord = points[len(points)-1]
    for point in points:
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.circle(image, tuple(point), 5, color, -1)
    cv2.line(image, tuple(points[0]), tuple(lastCoord), tuple(np.random.randint(0, 255, 3).tolist()), 2)
    # if points[0][0] <= lastCoord[0] and points[0][1] <= lastCoord[1]:
    #     res = image[points[0][1]:lastCoord[1], points[0][0]:lastCoord[0]]
    # elif points[0][0] <= lastCoord[0] and points[0][1] >= lastCoord[1]:
    #     res = image[lastCoord[1]:points[0][1], points[0][0]:lastCoord[0]]
    # elif points[0][0] >= lastCoord[0] and points[0][1] >= lastCoord[1]:
    #     res = image[lastCoord[1]:points[0][1], lastCoord[0]:points[0][0]]
    # else:
    #     res = image[points[0][1]:lastCoord[1], lastCoord[0]:points[0][0]]
        # cv2.imshow("png", res)
    #ski.io.imsave(f"results/cropped_image1.png", image)
    return image

def patternMatching(image, template):
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(image.shape)
    w, h = template.shape[::]

    print(str(w) + ", " + str(h))
    res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    #res = cv2.matchTemplate(image, template, cv2.TM_SQDIFF)

    threshold = 0.87
    loc = np.where(res >= threshold)
    #min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    for pt in zip(*loc[::-1]):
        cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
    cv2.imwrite('res.png', image)

    # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # top_left = min_loc
    # bottom_right = (top_left[0] + w, top_left[1] + h)
    # cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 2)
    # cv2.imwrite('resRight.png', image)
    return image


def findLine(coord1,  coord2):
    a = coord2[1] - coord1[1]
    b = coord1[0] - coord2[0]
    c = a * (coord1[0]) + b * (coord2[1])

    if (b < 0):
        print("The line passing through points 1 and 2 is:",
              a, "x - ", b, "y = ", c, "\n")
        return a, -b, c
    else:
        print("The line passing through points 1 and 2 is: ",
              a, "x + ", b, "y = ", c, "\n")
        return a, b, c


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

def projectLineToGreyscale(infraredPoints, imgLeft, imgRight, data):
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
        pointsLeft = []
        pointsRight = []
        #cameraCoord3D = get_lut_projection_pixel(lutInfrared, sphere[0], sphere[1])
        #infraredPoints = np.int32(infraredPoints)
        IRcameraCoord3D = get_lut_projection_pixel(lutInfrared, sphere[0], sphere[1])
        IRcameraCoord3D = np.array(IRcameraCoord3D)
        for i in range(1, 2000):
            IRcameraCoord3D = IRcameraCoord3D * i
            IRcameraCoord3D = np.append(IRcameraCoord3D, 1)
            # print(sphere)
            # print(IRcameraCoord3D)

            worldCoord = np.matmul(extrinsic1, IRcameraCoord3D)
            leftGreyscaleCameraCoord3D = np.matmul(mat_w_to_c2, worldCoord)
            rightGreyscaleCameraCoord3D = np.matmul(mat_w_to_c3, worldCoord)

            leftImageCoord2D = get_lut_pixel_image(lutGreyScaleLeft, leftGreyscaleCameraCoord3D[0], leftGreyscaleCameraCoord3D[1], leftGreyscaleCameraCoord3D[2])
            rightImageCoord2D = get_lut_pixel_image(rightGreyscaleCameraCoord3D, rightGreyscaleCameraCoord3D[0], rightGreyscaleCameraCoord3D[1], rightGreyscaleCameraCoord3D[2])

            if leftImageCoord2D is not None and not math.isnan(leftImageCoord2D[0]) and not math.isnan(leftImageCoord2D[1]) and not math.isnan(leftImageCoord2D[2]):
                pointsLeft.append(leftImageCoord2D)
            if rightImageCoord2D is not None and not math.isnan(rightImageCoord2D[0]) and not math.isnan(rightImageCoord2D[1]) and not math.isnan(rightImageCoord2D[2]):
                pointsRight.append(rightImageCoord2D)

        imgLeft = drawPoints(imgLeft, np.int32(pointsLeft))
        imgRight = drawPoints(imgRight, np.int32(pointsRight))

    return imgLeft, imgRight



if  __name__ == '__main__':
    data = DataAcquisition()
    data.load_data(config.get_filename("optical_sphere"))

    frame1 = np.copy(data.acquisitions['vl_front_left_cam_frames'][0])
    frame2 = np.copy(data.acquisitions['vl_front_right_cam_frames'][0])

    frame3 = np.copy(data.acquisitions['lt_depth_cam_ab_frames'][0])
    frame4 = np.copy(data.acquisitions['ahat_depth_cam_ab_frames'][0])
    irpoints = blobDetection(frame4)

    # irpoints = [[238, 221], [253, 208], [216, 200], [240, 188]]
    resImageLeft, resImageRight = projectLineToGreyscale(irpoints, frame1, frame2, data)

    resImageLeft = cv2.rotate(resImageLeft, cv2.cv2.ROTATE_90_CLOCKWISE)
    resImageRight = cv2.rotate(resImageRight, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)

    plt.subplot(121)
    plt.imshow(resImageLeft)

    plt.subplot(122)
    plt.imshow(resImageRight)
    plt.show()

