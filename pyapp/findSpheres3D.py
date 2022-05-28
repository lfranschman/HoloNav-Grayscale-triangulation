import math
import random

import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
import scipy
import skimage as ski
import skimage.io
from sphere_tracking_test import find_optical_spheres_c
from pyapp.DataAcquisition import DataAcquisition, ACQUISITIONS_HOLOLENS
from config import config
from pyapp.calibration_helpers import get_mat_c_to_w_series, get_lut_projection_pixel, get_lut_pixel_image, \
    get_lut_projection_int_pixel, get_mat_w_to_o, get_mat_divots_filename, get_mat_m_to_o_series
from python.common.File import load_pickle, save_pickle
from python.common.UtilImage import draw_disk
from pyapp.templateMatching import rotate_line_counterclockwise, rotate_line_clockwise, find_sphere
import glob
import os

from python.common.UtilMaths import get_mat_mrp, vec3_to_vec4, intersection_lines, point_based_registration, \
    squared_distance_vector_3d

left_idx = 280
right_idx = 0
depth_idx = 0


def find_markers_in_worldspace():
    data = DataAcquisition()
    newData = {}
    data.load_data(config.get_filename("optical_sphere"))
    newData["true_sphere_positions"] = pd.DataFrame([], columns=['time',
                                                                           'sphere1', 'sphere2',
                                                                           'sphere3', 'sphere4'])
    newData["true_sphere_positions"] = \
        newData["true_sphere_positions"].set_index("time")
    for acquisition in ACQUISITIONS_HOLOLENS:
        if not data.acquisitions[acquisition].empty:
            data.acquisitions[acquisition].index = data.acquisitions[
                                                       acquisition].index + pd.Timedelta(
                seconds=config.temporal_shift_hololens)
    mat_qf_to_m = load_pickle(get_mat_divots_filename(config, "mat_qf_to_m"))

    mat_w_to_o_list = []
    for qr_code_index in range(81, 840):
        timestamp = data.acquisitions["qr_code_position"].index[qr_code_index]
        qr_code_serie = data.acquisitions["qr_code_position"].loc[timestamp]
        if qr_code_serie['q1_m44'] == 0:  # front qr code not detected for this timestamp
            continue

        optical_index = data.acquisitions["probe"].index.get_loc(timestamp, method='nearest')
        optical_timestamp = data.acquisitions["probe"].index[optical_index]
        # we keep only if time difference is less than 20ms
        if abs((timestamp - optical_timestamp).total_seconds()) > 0.02:
            continue

        optical_serie = data.acquisitions["probe"].loc[optical_timestamp]
        mat_w_to_o, _ = get_mat_w_to_o(qr_code_serie, optical_serie, mat_qf_to_m)
        mat_w_to_o_list.append(mat_w_to_o)

    mat_w_to_o_list = np.array(mat_w_to_o_list)  # shape (_, 4, 4)
    translation_mean = mat_w_to_o_list[:, :3, 3]  # shape (_, 3)
    translation_mean = np.mean(translation_mean, axis=0)  # shape (3,)
    rotation = scipy.spatial.transform.Rotation.from_matrix(
        mat_w_to_o_list[:, :3, :3]).mean().as_matrix()  # shape (3,3)

    mat_w_to_o = np.identity(4)
    mat_w_to_o[:3, :3] = rotation
    mat_w_to_o[:3, 3] = translation_mean
    # print(mat_w_to_o)

    mat_o_to_w = np.linalg.inv(mat_w_to_o)
    for camera in ["vl_front_left_cam", "vl_front_right_cam"]:
        print(f"camera {camera}")
        for frame_id in range(76, len(data.acquisitions[camera])):
            timestamp = data.acquisitions[camera].index[frame_id]
            serie = data.acquisitions[camera].loc[timestamp]
            extrinsic = get_mat_c_to_w_series(serie)
            mat_w_to_c = np.linalg.inv(extrinsic)

            optical_index = data.acquisitions["probe"].index.get_loc(timestamp, method='nearest')
            optical_timestamp = data.acquisitions["probe"].index[optical_index]
            # print(f"optical_index {optical_index} timestamp {timestamp} optical_timestamp {optical_timestamp}")

            # we keep only if time difference is less than 20ms
            if abs((timestamp - optical_timestamp).total_seconds()) > 0.025:
                continue  # skip this frame

            optical_serie = data.acquisitions["probe"].loc[optical_timestamp]
            mat_m_to_o = get_mat_m_to_o_series(optical_serie)



            # NDI optical_markers_8700339, check Polaris_Spectra_Tool_Kit_Guide.pdf for the position of the spheres
            sphere_positions = [[0, 0, 0, 1], [0, 28.59, 41.02, 1], [0, 0, 88, 1],
                                [0, -44.32, 40.45, 1]]
            points = []
            for i, sphere_pos in enumerate(sphere_positions):
                pos_sphere1_c = np.matmul(mat_o_to_w, np.matmul(mat_m_to_o, sphere_pos))
                # print(f"get lut {time.time() - t1} s")
                print(f"frame{frame_id:04.0f}: coords {pos_sphere1_c}")
                points.append(pos_sphere1_c)
            dataEntry = [points[0], points[1], points[2], points[3]]
            newData["true_sphere_positions"].loc[timestamp] = dataEntry
    save_pickle(newData, config.get_filename("true_sphere_positions"))
    return points


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

def getMatOToW():

    mat_w_to_o_list = []
    mat_qf_to_m = load_pickle(get_mat_divots_filename(config, "mat_qf_to_m"))
    for qr_code_index in range(81, 840):
        timestamp = data.acquisitions["qr_code_position"].index[qr_code_index]
        qr_code_serie = data.acquisitions["qr_code_position"].loc[timestamp]
        if qr_code_serie['q1_m44'] == 0:  # front qr code not detected for this timestamp
            continue

        optical_index = data.acquisitions["probe"].index.get_loc(timestamp, method='nearest')
        optical_timestamp = data.acquisitions["probe"].index[optical_index]
        # we keep only if time difference is less than 20ms
        if abs((timestamp - optical_timestamp).total_seconds()) > 0.02:
            continue

        optical_serie = data.acquisitions["probe"].loc[optical_timestamp]
        mat_w_to_o, _ = get_mat_w_to_o(qr_code_serie, optical_serie, mat_qf_to_m)
        mat_w_to_o_list.append(mat_w_to_o)

    mat_w_to_o_list = np.array(mat_w_to_o_list)  # shape (_, 4, 4)
    translation_mean = mat_w_to_o_list[:, :3, 3]  # shape (_, 3)
    translation_mean = np.mean(translation_mean, axis=0)  # shape (3,)
    rotation = scipy.spatial.transform.Rotation.from_matrix(
        mat_w_to_o_list[:, :3, :3]).mean().as_matrix()  # shape (3,3)

    mat_w_to_o = np.identity(4)
    mat_w_to_o[:3, :3] = rotation
    mat_w_to_o[:3, 3] = translation_mean
    # print(mat_w_to_o)

    mat_o_to_w = np.linalg.inv(mat_w_to_o)
    return mat_o_to_w


def triangulate(spheres, leftID, rightID, infraredID):

    lutInfrared = data.acquisitions["ahat_depth_cam" + "_lut_projection"]
    lutGreyScaleLeft = data.acquisitions["vl_front_left_cam" + "_lut_projection"]
    lutGreyScaleRight = data.acquisitions["vl_front_right_cam" + "_lut_projection"]

    timestampIR = data.acquisitions["ahat_depth_cam"].index[infraredID]
    timestampLeft = data.acquisitions["vl_front_left_cam"].index[leftID]
    timestampRight = data.acquisitions["vl_front_right_cam"].index[rightID]

    serieIR = data.acquisitions["ahat_depth_cam"].loc[timestampIR]
    serieLeft = data.acquisitions["vl_front_left_cam"].loc[timestampLeft]
    serieRight = data.acquisitions["vl_front_right_cam"].loc[timestampRight]

    # print(timestampLeft)
    # print(timestampRight)

    extrinsicIR = get_mat_c_to_w_series(serieIR)
    extrinsicLeft = get_mat_c_to_w_series(serieLeft)
    extrinsicRight = get_mat_c_to_w_series(serieRight)

    mat_w_to_c_IR = np.linalg.inv(extrinsicIR)

    sphere_positions = []
    for (l, r) in spheres:
        cam_coord_left =  get_lut_projection_pixel(lutGreyScaleLeft, l[0], l[1])
        cam_coord_left = vec3_to_vec4(cam_coord_left)
        world_coord_left = np.matmul(extrinsicLeft, cam_coord_left)
        # print(world_coord_left)
        origin_left_world = np.matmul(extrinsicLeft, np.array([0, 0, 0, 1]))

        cam_coord_right = get_lut_projection_pixel(lutGreyScaleRight, r[0], r[1])
        cam_coord_right = vec3_to_vec4(cam_coord_right)
        world_coord_right = np.matmul(extrinsicRight, cam_coord_right)
        origin_right_world = np.matmul(extrinsicRight, np.array([0, 0, 0, 1]))

        sphere_pos = intersection_lines(origin_left_world, world_coord_left, origin_right_world, world_coord_right)
        # sphere_pos = np.matmul(mat_w_to_c_IR, vec3_to_vec4(sphere_pos)
        sphere_positions.append(sphere_pos)


    return np.array(sphere_positions)

def markersToWorld(frameID):
    # data.load_data(config.get_filename("optical_sphere"))
    mat_o_to_w = getMatOToW()
    timestamp = data.acquisitions["vl_front_left_cam"].index[frameID]

    optical_index = data.acquisitions["probe"].index.get_loc(timestamp, method='nearest')
    optical_timestamp = data.acquisitions["probe"].index[optical_index]

    # we keep only if time difference is less than 20ms
    # if abs((timestamp - optical_timestamp).total_seconds()) <= 0.02:
    #     continue  # skip this frame

    optical_serie = data.acquisitions["probe"].loc[optical_timestamp]
    mat_m_to_o = get_mat_m_to_o_series(optical_serie)

    sphere_markers = [[0, 0, 0, 1], [0, 28.59, 41.02, 1], [0, 0, 88, 1], [0, -44.32, 40.45, 1]]
    markers = []
    for sphere_pos in sphere_markers:
        wPos = np.matmul(mat_o_to_w, np.matmul(mat_m_to_o, sphere_pos))
        markers.append(wPos[0:3])
    markers = np.array(markers)
    return markers


def drawPoints(image, points):
    lastCoord = points[len(points)-1]
    # for point in points:
    #     color = tuple(np.random.randint(0, 255, 3).tolist())
    #     cv2.circle(image, tuple(point), 5, color, -1)
    cv2.line(image, tuple(points[0]), tuple(lastCoord), tuple(np.random.randint(0, 255, 3).tolist()), 2)
    # print(points[0])
    # print(lastCoord)
    return image

def cropImage(image, points):
    lastCoord = points[len(points) - 1]
    if points[0][0] <= lastCoord[0] and points[0][1] <= lastCoord[1]:
        res = image[points[0][1]:lastCoord[1], points[0][0]:lastCoord[0]]
    elif points[0][0] <= lastCoord[0] and points[0][1] >= lastCoord[1]:
        res = image[lastCoord[1]:points[0][1], points[0][0]:lastCoord[0]]
    elif points[0][0] >= lastCoord[0] and points[0][1] >= lastCoord[1]:
        res = image[lastCoord[1]:points[0][1], lastCoord[0]:points[0][0]]
    else:
        res = image[points[0][1]:lastCoord[1], lastCoord[0]:points[0][0]]
    # cv2.imshow("png", res)
    # ski.io.imsave(f"results/cropped_image1_ " + str(index) + ".png", image)
    return res

def mulitiPatternMatching(image, template):
    print(image.shape)
    w, h = template.shape[::]

    print(str(w) + ", " + str(h))
    res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.87
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
    cv2.imwrite('res.png', image)
    return image

def patternMatching(image, template):
    w, h = template.shape[::]
    res = cv2.matchTemplate(image, template, cv2.TM_SQDIFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = min_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 2)
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

def get_line(x1, y1, x2, y2):
    points = []
    issteep = abs(y2-y1) > abs(x2-x1)
    if issteep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    rev = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        rev = True
    deltax = x2 - x1
    deltay = abs(y2-y1)
    error = int(deltax / 2)
    y = y1
    ystep = None
    if y1 < y2:
        ystep = 1
    else:
        ystep = -1
    for x in range(x1, x2 + 1):
        if issteep:
            points.append((y, x))
        else:
            points.append((x, y))
        error -= deltay
        if error < 0:
            y += ystep
            error += deltax
    # Reverse the list if the coordinates were reversed
    if rev:
        points.reverse()
    return points

def blobDetection(frame):
    frame = (frame /256).astype('uint8')

    ret2, thresh = cv2.threshold(frame, 4.5, 256, cv2.THRESH_BINARY)

    _, _, stats, centroids =  cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)

    # Sort to find the biggest components
    idx = np.argsort(-stats[1:,cv2.CC_STAT_AREA])+1
    return centroids[idx[:min(4,len(idx))],:]

    # frame = (frame /256).astype('uint8')
    # ret2, thresh = cv2.threshold(frame, 4.5, 256, cv2.THRESH_BINARY_INV)
    #
    # params = cv2.SimpleBlobDetector_Params()
    #
    # # Change thresholds
    # params.minThreshold = 0;
    # params.maxThreshold = 255;
    #
    # # Filter by Area.
    # params.filterByArea = True
    # params.minArea = 15
    #
    # detector = cv2.SimpleBlobDetector_create(params)
    #
    # keypoints = detector.detect(thresh)
    # # im_with_keypoints = cv2.drawKeypoints(thresh, keypoints, np.array([]), (0 ,0 ,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)    # left:
    # # cv2.imshow("Keypoints", im_with_keypoints)
    # # cv2.waitKey(0)
    #
    # #frameCopy = np.copy(data.acquisitions['ahat_depth_cam_ab_frames'][0])
    # infraredPoints = []
    # for keypoint in keypoints:
    #     coord = np.array(keypoint.pt)
    #     infraredPoints.append(coord)
    #     print(coord)
    #     #draw_disk(frame, coord[0], coord[1], 0, size=1)
    #
    # return infraredPoints

def projectLineToGreyscale(infraredPoints, frameIDIR,  imgLeft, frameIDLeft, imgRight, frameIDRight, data):

    lutInfrared = data.acquisitions["ahat_depth_cam" + "_lut_projection"]
    lutGreyScaleLeft = data.acquisitions["vl_front_left_cam" + "_lut_projection"]
    lutGreyScaleRight = data.acquisitions["vl_front_right_cam" + "_lut_projection"]

    timestamp1 = data.acquisitions["ahat_depth_cam"].index[frameIDIR]
    timestamp2 = data.acquisitions["vl_front_left_cam"].index[frameIDLeft]
    timestamp3 = data.acquisitions["vl_front_right_cam"].index[frameIDRight]

    serie1 = data.acquisitions["ahat_depth_cam"].loc[timestamp1]
    serie2 = data.acquisitions["vl_front_left_cam"].loc[timestamp2]
    serie3 = data.acquisitions["vl_front_right_cam"].loc[timestamp3]

    extrinsic1 = get_mat_c_to_w_series(serie1)
    extrinsic2 = get_mat_c_to_w_series(serie2)
    extrinsic3 = get_mat_c_to_w_series(serie3)

    mat_w_to_c1 = np.linalg.inv(extrinsic1)
    mat_w_to_c2 = np.linalg.inv(extrinsic2)
    mat_w_to_c3 = np.linalg.inv(extrinsic3)

    lineCoordsLeft = []
    lineCoordsRight = []

    for j, sphere in enumerate(infraredPoints):
        pointsLeft = []
        pointsRight = []
        #cameraCoord3D = get_lut_projection_pixel(lutInfrared, sphere[0], sphere[1])
        #infraredPoints = np.int32(infraredPoints)
        IRcameraCoord3D = get_lut_projection_pixel(lutInfrared, sphere[0], sphere[1])
        IRcameraCoord3D = np.array(IRcameraCoord3D)
        for i in range(1, 2000):
            IRcameraCoord3D = IRcameraCoord3D * i
            if math.isinf(IRcameraCoord3D[0]) or math.isinf(IRcameraCoord3D[1]) or math.isinf(IRcameraCoord3D[1]):
                continue
            IRcameraCoord3D = np.append(IRcameraCoord3D, 1)
                # print(sphere)

            # print(IRcameraCoord3D)

            worldCoord = np.matmul(extrinsic1, IRcameraCoord3D)
            IRcameraCoord3D = IRcameraCoord3D[0:3]
            leftGreyscaleCameraCoord3D = np.matmul(mat_w_to_c2, worldCoord)
            rightGreyscaleCameraCoord3D = np.matmul(mat_w_to_c3, worldCoord)
            # print(leftGreyscaleCameraCoord3D)
            if leftGreyscaleCameraCoord3D is not None and not math.isnan(leftGreyscaleCameraCoord3D[0]) and not math.isnan(leftGreyscaleCameraCoord3D[1]) and not math.isnan(leftGreyscaleCameraCoord3D[2]):
                leftImageCoord2D = get_lut_pixel_image(lutGreyScaleLeft, leftGreyscaleCameraCoord3D[0], leftGreyscaleCameraCoord3D[1], leftGreyscaleCameraCoord3D[2])

            if rightGreyscaleCameraCoord3D is not None and not math.isnan(rightGreyscaleCameraCoord3D[0]) and not math.isnan(rightGreyscaleCameraCoord3D[1]) and not math.isnan(rightGreyscaleCameraCoord3D[2]):
                rightImageCoord2D = get_lut_pixel_image(lutGreyScaleRight, rightGreyscaleCameraCoord3D[0], rightGreyscaleCameraCoord3D[1], rightGreyscaleCameraCoord3D[2])

            if leftImageCoord2D is not None and not math.isnan(leftImageCoord2D[0]) and not math.isnan(leftImageCoord2D[1]):
                pointsLeft.append(list(leftImageCoord2D))
            if rightImageCoord2D is not None and not math.isnan(rightImageCoord2D[0]) and not math.isnan(rightImageCoord2D[1]):
                pointsRight.append(list(rightImageCoord2D))

        firstCoordLeft = pointsLeft[0]
        lastCoordLeft = pointsLeft[len(pointsLeft) - 1]

        firstCoordRight = pointsRight[0]
        lastCoordRight = pointsRight[len(pointsRight) - 1]

        leftLine = [firstCoordLeft, lastCoordLeft]
        lineCoordsLeft.append(leftLine)

        rightLine = [firstCoordRight, lastCoordRight]
        lineCoordsRight.append(rightLine)



        # print(np.unique(np.int32(pointsLeft)))
        # print(np.unique(np.int32(pointsRight)))
        imgLeft = drawPoints(imgLeft, np.int32(pointsLeft))
        imgRight = drawPoints(imgRight, np.int32(pointsRight))

    lineCoordsLeft = np.int32(lineCoordsLeft)
    lineCoordsRight = np.int32(lineCoordsRight)

    # print(lineCoordsLeft)
    # print(lineCoordsRight)

    return imgLeft, imgRight, lineCoordsLeft, lineCoordsRight



if  __name__ == '__main__':

    # find_markers_in_worldspace()
    data = DataAcquisition()
    optical_locs = load_pickle(r"C:\Users\Lesle\Documents\2022_03_30_optical_sphere\true_sphere_positions.pickle.gz")
    # # print(optical_locs["true_sphere_positions"].loc["2022-03-30 15:14:57.788923+02:00"])
    #
    data.load_data(config.get_filename("optical_sphere"))
    mean_errors = []

    for i in range(270, 271):
        leftID = i

        timestamp = data.acquisitions["vl_front_left_cam"].index[leftID]

        frameIdIR = 0
        best_ts = data.acquisitions["ahat_depth_cam"].index[0]
        for i, ts in enumerate(data.acquisitions["ahat_depth_cam"].index):
            if abs(ts - timestamp) < abs(best_ts - timestamp):
                best_ts = ts
                frameIdIR = i

        rightID = 0
        best_ts = data.acquisitions["vl_front_right_cam"].index[0]
        for i, ts in enumerate(data.acquisitions["vl_front_right_cam"].index):
            if abs(ts - timestamp) < abs(best_ts - timestamp):
                best_ts = ts
                rightID = i

        opticalID = i
        best_ts = optical_locs["true_sphere_positions"].index[0]
        for i, ts in enumerate(optical_locs["true_sphere_positions"].index):
            if abs(ts - timestamp) < abs(best_ts - timestamp):
                best_ts = ts
                opticalID = i

        print(f"left_idx : {leftID}, right_idx : {rightID}, depth_idx : {frameIdIR}, optical_idx : {opticalID}")

        # # print(len(data.acquisitions['vl_front_left_cam_lut_projection']))
        frame1 = np.copy(data.acquisitions['vl_front_left_cam_frames'][leftID])
        # ski.io.imsave("leftX.png", frame1)
        cv2.imwrite("leftX.png", frame1)
        # left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)

        # left = cv2.imread("left2.png", cv2.IMREAD_GRAYSCALE)
        # h, w = left.shape[:2]
        frame2 = np.copy(data.acquisitions['vl_front_right_cam_frames'][rightID])
        # ski.io.imsave("rightX.png", frame2)
        cv2.imwrite("rightX.png", frame2)

        # right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        # right = cv2.imread("right2.png", cv2.IMREAD_GRAYSCALE)
        #
        # # frame3 = np.copy(data.acquisitions['lt_depth_cam_ab_frames'][0])
        frame4 = np.copy(data.acquisitions['ahat_depth_cam_ab_frames'][frameIdIR])
        #
        #
        irpoints = blobDetection(frame4)
        #
        # # irpoints = [[238, 221], [253, 208], [216, 200], [240, 188]]
        resImageLeft, resImageRight, left_lines, right_lines = projectLineToGreyscale(irpoints, frameIdIR,  frame1, leftID, frame2, rightID, data)
        if len(left_lines) < 4 or len(right_lines) < 4:
            continue

        # Load input image as grayscale
        left = cv2.imread("leftX.png", cv2.IMREAD_GRAYSCALE)
        # print(type(left))
        h, w = left.shape[:2]
        left = cv2.rotate(left, cv2.ROTATE_90_CLOCKWISE)

        right = cv2.imread("rightX.png", cv2.IMREAD_GRAYSCALE)
        right = cv2.rotate(right, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Convert to RGB to draw the colored circles
        left_annotated = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
        right_annotated = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)


        rotate_line_clockwise(left_lines, h, w)
        rotate_line_counterclockwise(right_lines, h, w)

        # Convert templates points to floats
        for i in range(4):
            p1, p2 = left_lines[i]
            left_lines[i] = [np.float32(p1), np.float32(p2)]
            cv2.line(left_annotated, p1, p2, (0, 0, 0), 1)

        for i in range(4):
            p1, p2 = right_lines[i]
            right_lines[i] = [np.float32(p1), np.float32(p2)]
            cv2.line(right_annotated, p1, p2, (0, 0, 0), 1)

        # Load templates located in the "templates" folder
        # Each template can have an alpha component which will be used as a mask
        fns = glob.glob(os.path.join("templates", "*"))
        print(fns)

        templates = []
        for fn in fns:
            # Load with alpha component
            templ = cv2.imread(fn, cv2.IMREAD_UNCHANGED)

            # Split into color/alpha
            templ_gray = cv2.cvtColor(templ[:, :, :3], cv2.COLOR_BGR2GRAY)
            if templ.shape[2] == 4:
                templ_alpha = templ[:, :, 3]
            else:
                # Create an opaque plane if no alpha in image
                templ_alpha = np.ones(templ_gray.shape[:2], np.uint8)

            templates.append((templ_gray, templ_alpha))

        # Map of valid to avoid overlaps
        # It will be filled by the find_sphere function
        left_valid = np.zeros(left.shape[:2], np.uint8)
        right_valid = np.zeros(right.shape[:2], np.uint8)

        sphere_locations = []
        for i in range(4):
            posLeft = find_sphere(left, templates, left_lines[i][0], left_lines[i][1], left_valid)
            cv2.circle(left_annotated, posLeft, 8, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 1)
            posRight = find_sphere(right, templates, right_lines[i][0], right_lines[i][1], right_valid)
            cv2.circle(right_annotated, posRight, 8, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 1)
            sphere_locations.append((posLeft, posRight))
        print("2D sphere coordinates in greyscales (list of [left , right]):")
        print(np.array(sphere_locations))
        # print(sphere_locations[0])
        # print(sphere_locations[0][0])
        # print(sphere_locations[0][0][0])


        positions_spheres_3D = triangulate(sphere_locations, leftID, rightID, frameIdIR)
        positions_spheres_3D = np.array(positions_spheres_3D)
        print("triangulated points:")
        print(positions_spheres_3D)



        # left_idx: 270, right_idx: 247, depth_idx: 258, optical_idx: 1120
        sphere_markers = [[0, 0, 0, 1], [0, 28.59, 41.02, 1], [0, 0, 88, 1], [0, -44.32, 40.45, 1]]
        opticals = []
        optical_timestamp = optical_locs["true_sphere_positions"].index[opticalID]
        markers = optical_locs["true_sphere_positions"].loc[optical_timestamp]
        for marker in markers:
            opticals.append(marker)

        print("optical points in world space:")
        print(np.array(opticals))
        print("\n")
        sum = 0
        print("distances between the spheres:")
        for i in range(0, 4):
            # diff = np.linalg.norm(markers[i][0:3] - positions_spheres_3D[i])
            dist = squared_distance_vector_3d(markers[i][0:3], positions_spheres_3D[i])
            print(np.sqrt(dist))

            sum = sum + dist # taking a sum of all the differences
        MSE = sum / 4  # dividing summation by total values to obtain average

        print("mean squre error of the frame:")
        print(MSE)
        print("\n")
        mean_errors.append(MSE)

        plt.subplot(121)
        plt.imshow(left_annotated[:, :, ::-1])
        plt.title("left")
        plt.axis('off')

        # plt.subplot(122)
        # plt.imshow(left_valid)
        # plt.title("valid")
        # plt.show()
        #
        # plt.subplot(122)
        # plt.imshow(right_valid)
        # plt.title("valid")
        # plt.show()

        plt.subplot(122)
        plt.imshow(right_annotated[:, :, ::-1])
        plt.title("right")
        plt.axis('off')
        plt.show()

    print(mean_errors)
    fig = plt.figure()
    plt.plot(mean_errors)
    fig.savefig("resplotMSE_2.png", dpi=fig.dpi)
    plt.show()

