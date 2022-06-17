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
from pyapp.templateMatching import rotate_line_counterclockwise, rotate_line_clockwise, find_sphere, \
    rotate_point_counterclockwise, rotate_point_clockwise
import glob
import os
from sphere_tracking import find_optical_spheres
from python.common.UtilMaths import get_mat_mrp, vec3_to_vec4, intersection_lines, point_based_registration, \
    squared_distance_vector_3d


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

        lut_projection = data.acquisitions[camera + "_lut_projection"]

        nb_images_remaining = 40  # just want to do it for the nb_images_remaining first images
        # nb_images_remaining = 10000 # just want to do it for the nb_images_remaining first images
        # for frame_id in range(len(data.acquisitions[camera + "_frames"])):
        for frame_id in range(245, 892):
            print(f"frame_id {frame_id}")
            timestamp = data.acquisitions[camera].index[frame_id]
            serie = data.acquisitions[camera].loc[timestamp]
            extrinsic = get_mat_c_to_w_series(serie)
            mat_w_to_c = np.linalg.inv(extrinsic)

            optical_index = data.acquisitions["probe"].index.get_loc(timestamp, method='nearest')
            optical_timestamp = data.acquisitions["probe"].index[optical_index]
            # print(f"optical_index {optical_index} timestamp {timestamp} optical_timestamp {optical_timestamp}")

            # we keep only if time difference is less than 20ms
            if abs((timestamp - optical_timestamp).total_seconds()) > 0.02:
                continue  # skip this frame

            optical_serie = data.acquisitions["probe"].loc[optical_timestamp]
            mat_m_to_o = get_mat_m_to_o_series(optical_serie)

            # compute mat_w_to_o each time with the qr code in the current frame
            # if True:
            if False:
                qr_code_index = data.acquisitions["qr_code_position"].index.get_loc(timestamp, method='nearest')
                qr_code_timestamp = data.acquisitions["qr_code_position"].index[qr_code_index]
                if abs((timestamp - qr_code_timestamp).total_seconds()) > 0.1:
                    continue  # skip this frame

                qr_code_serie = data.acquisitions["qr_code_position"].loc[qr_code_timestamp]
                if qr_code_serie['q1_m44'] == 0:  # front qr code not detected for this timestamp
                    continue  # skip this frame

                mat_w_to_o, _ = get_mat_w_to_o(qr_code_serie, optical_serie, mat_qf_to_m)
                mat_o_to_w = np.linalg.inv(mat_w_to_o)

            # mat_m_to_c = np.matmul(mat_w_to_c, np.matmul(mat_o_to_w, mat_m_to_o))

            frame = np.copy(data.acquisitions[camera + '_frames'][frame_id])  # shape (h,w)
            # NDI optical_markers_8700339, check Polaris_Spectra_Tool_Kit_Guide.pdf for the position of the spheres
            sphere_positions = [[0, 0, 0, 1], [0, 28.59, 41.02, 1], [0, 0, 88, 1], [0, -44.32, 40.45, 1]]
            draw_sphere = False
            points = []
            for i, sphere_pos in enumerate(sphere_positions):
                pos_sphere1_c = np.matmul(mat_o_to_w, np.matmul(mat_m_to_o, sphere_pos))
                # t1 = time.time()
                # coord = get_lut_pixel_image(lut_projection, pos_sphere1_c[0], pos_sphere1_c[1], pos_sphere1_c[2])
                # print(f"get lut {time.time() - t1} s")
                if pos_sphere1_c is not None and not math.isnan(pos_sphere1_c[0]) and not math.isnan(
                        pos_sphere1_c[1]) and not math.isnan(pos_sphere1_c[2]):
                    print(f"center of the sphere {i}: {pos_sphere1_c}")
                    # draw_disk(frame, coord[0], coord[1], 255, size=1)
                    # draw_sphere = True
                    points.append(pos_sphere1_c)
            dataEntry = [points[0], points[1], points[2], points[3]]
            newData["true_sphere_positions"].loc[timestamp] = dataEntry
    save_pickle(newData, config.get_filename("true_sphere_positions_w"))


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
    for point in points:
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.circle(image, tuple(point), 5, color, -1)
    # cv2.line(image, tuple(points[0]), tuple(lastCoord), tuple(np.random.randint(0, 255, 3).tolist()), 2)
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
    # frame = (frame /256).astype('uint8')
    #
    # ret2, thresh = cv2.threshold(frame, 4.5, 256, cv2.THRESH_BINARY)
    #
    # _, _, stats, centroids =  cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)
    #
    # # Sort to find the biggest components
    # idx = np.argsort(-stats[1:,cv2.CC_STAT_AREA])+1
    # return centroids[idx[:min(4,len(idx))],:]

    frame = (frame /256).astype('uint8')
    ret2, thresh = cv2.threshold(frame, 4.5, 256, cv2.THRESH_BINARY_INV)

    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 0
    params.maxThreshold = 255

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 25

    # Find based on circularity
    params.filterByCircularity = True
    params.minCircularity = 0.

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

        if len(pointsLeft) > 0:
            firstCoordLeft = pointsLeft[0]
            lastCoordLeft = pointsLeft[len(pointsLeft) - 1]

        if len(pointsRight) > 0:
            firstCoordRight = pointsRight[0]
            lastCoordRight = pointsRight[len(pointsRight) - 1]

        leftLine = [firstCoordLeft, lastCoordLeft]
        lineCoordsLeft.append(leftLine)

        rightLine = [firstCoordRight, lastCoordRight]
        lineCoordsRight.append(rightLine)



        # print(np.unique(np.int32(pointsLeft)))
        # print(np.unique(np.int32(pointsRight)))
        # imgLeft = drawPoints(imgLeft, np.int32(pointsLeft))
        # imgRight = drawPoints(imgRight, np.int32(pointsRight))

    lineCoordsLeft = np.int32(lineCoordsLeft)
    lineCoordsRight = np.int32(lineCoordsRight)

    # print(lineCoordsLeft)
    # print(lineCoordsRight)

    return imgLeft, imgRight, lineCoordsLeft, lineCoordsRight



if  __name__ == '__main__':
    # find_optical_spheres_c()
    # 2022 - 03 - 30  15: 15:32.054259 + 02: 00
    data = DataAcquisition()
    optical_locs = load_pickle(r"C:\Users\Lesle\Documents\2022_03_30_optical_sphere\world_positions.pickle.gz")
    # print(optical_locs["true_sphere_positions"].loc["2022-03-30 15:14:57.788923+02:00"])
    # print(optical_locs["true_pos"].loc[100])
    data.load_data(config.get_filename("optical_sphere"))
    mean_errors = []
    all_sphere_dist = []
    minTot = 10000000
    maxTot = 0

    for i in range(100, 700):
        leftID = i

        timestamp = data.acquisitions["vl_front_left_cam"].index[leftID]
        print(timestamp)
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

        # opticalID = 0
        # best_ts = optical_locs["true_pos"].index[0]
        # for i, ts in enumerate(optical_locs["true_pos"].index):
        #     if abs(ts - timestamp) < abs(best_ts - timestamp):
        #         best_ts = ts
        #         opticalID = i

        print(f"left_idx : {leftID}, right_idx : {rightID}, depth_idx : {frameIdIR}")

        # # print(len(data.acquisitions['vl_front_left_cam_lut_projection']))
        frame1 = np.copy(data.acquisitions['vl_front_left_cam_frames'][leftID])
        # ski.io.imsave("leftX.png", frame1)
        # cv2.imwrite("leftX.png", frame1)
        # left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)

        # left = cv2.imread("left2.png", cv2.IMREAD_GRAYSCALE)
        # h, w = left.shape[:2]
        frame2 = np.copy(data.acquisitions['vl_front_right_cam_frames'][rightID])
        # ski.io.imsave("rightX.png", frame2)
        # cv2.imwrite("rightX.png", frame2)

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
        # left = cv2.imread("leftX.png", cv2.IMREAD_GRAYSCALE)
        left = frame1
        # print(type(left))
        h, w = left.shape[:2]
        left = cv2.rotate(left, cv2.ROTATE_90_CLOCKWISE)
        newH, newW = left.shape[:2]

        # right = cv2.imread("rightX.png", cv2.IMREAD_GRAYSCALE)
        right = frame2
        right = cv2.rotate(right, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Convert to RGB to draw the colored circles
        left_annotated = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
        right_annotated = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)

        rotatedBackLeft = cv2.rotate(left_annotated, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rotatedBackRight = cv2.rotate(right_annotated, cv2.ROTATE_90_CLOCKWISE)

        rotate_line_clockwise(left_lines, h, w)
        rotate_line_counterclockwise(right_lines, h, w)

        # Convert templates points to floats
        for i in range(4):
            p1, p2 = left_lines[i]
            left_lines[i] = [np.float32(p1), np.float32(p2)]
            cv2.line(left_annotated, p1, p2, (255, 0, 0), 1)

        for i in range(4):
            p1, p2 = right_lines[i]
            right_lines[i] = [np.float32(p1), np.float32(p2)]
            cv2.line(right_annotated, p1, p2, (255, 0, 0), 1)

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
            cv2.circle(rotatedBackLeft, rotate_point_counterclockwise(posLeft, newH, newW), 8, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 1)
            cv2.circle(left_annotated, posLeft, 8, (0, 128, 0), 1)
            posRight = find_sphere(right, templates, right_lines[i][0], right_lines[i][1], right_valid)
            cv2.circle(rotatedBackRight, rotate_point_clockwise(posRight, newH, newW), 8, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 1)
            sphere_locations.append((rotate_point_counterclockwise(posLeft, newH, newW),
                                     rotate_point_clockwise(posRight, newH, newW)))        # print(sphere_locations)
        # print(sphere_locations)
        # print(sphere_locations[0][0])
        # print(sphere_locations[0][0][0])
        # sphere_locations_from_opticals=[
        #     [[262, 287], [258, 76]],
        #     [[309, 239], [311, 126]],
        #     [[264, 206], [355, 161]],
        #     [[211, 236], [408, 130]]
        # ]

        positions_spheres_3D = triangulate(sphere_locations, leftID, rightID, frameIdIR)
        positions_spheres_3D = np.array(positions_spheres_3D)
        # print(positions_spheres_3D)
    #
    #     # left_idx: 270, right_idx: 247, depth_idx: 258, optical_idx: 1120
    #
        # optical_timestamp = optical_locs["true_pos"].index[opticalID]
        if leftID not in optical_locs["true_pos"].index:
            continue

        markers = optical_locs["true_pos"].loc[leftID]
        print(markers)
        # markers = np.array(markers)
        # markers = find_optical_spheres(data, leftID)

        if markers is None or len(markers) < 4:
            continue

        sum = 0
        sphere_dists = []
        n = 0
        for m in markers:
            min = 1000000
            for pos in positions_spheres_3D:
                dist = squared_distance_vector_3d(m[0:3], pos)
                if dist < min:
                    min = dist
            if np.sqrt(min) <= 11.5:
                n = n + 1
                sum = sum + np.sqrt(min)
            sphere_dists.append(np.sqrt(min))
            print(np.sqrt(min))
        if n == 0:
            continue
        RMSE = sum/n
        print(RMSE)
        mean_errors.append(RMSE)
        all_sphere_dist.append(sphere_dists)
        if RMSE < minTot:
            minTot = RMSE
        if RMSE > maxTot:
            maxTot = RMSE

        # plt.subplot(121)
        plt.imshow(left_annotated[:, :, ::-1])
        plt.title("left")
        plt.axis('off')
        cv2.imwrite("coloured.png", left_annotated[:, :, ::-1])

        # plt.subplot(122)
        # plt.imshow(left_valid)
        # plt.title("valid")
        # plt.show()
        #
        # plt.subplot(122)
        # plt.imshow(right_valid)
        # plt.title("valid")
        # plt.show()
        #
        plt.subplot(122)
        plt.imshow(rotatedBackRight)
        plt.title("right")
        plt.axis('off')
        plt.show()
    # mean_errors = [4.34857277906328, 2.945946853447162, 3.4222220630303797, 3.191348486900809, 3.4823172509001306, 3.4229454434809967, 3.4084866075200666, 3.799465246257886, 3.6169707298625675, 3.4983163538462283, 4.579579817169079, 4.651796605217604, 2.840380088300653, 3.400096544781122, 4.256996384335602, 3.0872107583133115, 3.7120756431357704, 4.6898473427611345, 3.5743659178574743, 3.2574962207668308, 3.0616897566482866, 3.2938935560280385, 3.781348579392353, 5.277135664933342, 3.5496502138698736, 7.77585980339587, 4.5697827209565505, 5.0974742526884365, 3.0866868105390446, 4.095223375188463, 4.0125960756777115, 3.6774809195945113, 3.5027570853849475, 3.689776115366807, 3.3609593144502807, 3.959127894972957, 3.212726363604178, 3.4821990379901604, 2.6223638444214568, 3.3233972388912636, 4.040105565742101, 2.1391341055997835, 2.6264847749691405, 3.389925076184357, 4.830470020033804, 4.4993551289320655, 4.218484292302167, 3.948563474516848, 3.2358716245436274, 3.459409925688549, 4.078801406112459, 3.283058951250651, 2.7012935033119536, 4.173097912952144, 4.870867400895566, 4.886693149971465, 2.1932049694275206, 5.091958295990472, 2.666308712776008, 5.707708783537761, 4.339186258511327, 2.690175231595311, 3.8470369119344356, 2.1639336807095297, 3.9849957050585307, 3.162244307440292, 2.7211555506403284, 4.379514434740701, 2.8653588983863383, 3.2185233463167666, 3.299853877405082, 3.9154079226593623, 3.717388711044623, 3.415771417592698, 4.695980515740597, 3.3517303620560903, 4.138068572237094, 3.8290158183849714, 4.9823343107993, 7.086428281657494, 2.2628598833380997, 3.5950564951838553, 3.6841659828635724, 1.834911405243826, 2.8978834128147297, 4.528191158822809, 3.9531021358713465, 4.117375945008563, 1.9470597379320562, 5.439264691354887, 1.5655878566016839, 3.305166416814083, 5.024175295100405, 8.245476599495229, 5.014344992641828, 6.347193673048732, 4.15075542702216, 2.9349294500579566, 2.866765676953377, 4.690871710973568, 3.74983078308769, 2.647905335094774, 2.774832006268536, 3.1037652272815377, 6.326248976136782, 4.497125421022292, 3.387199676486961, 5.37302533755237, 7.1561375150676465, 4.63674228171949, 3.732475044320765, 8.634447354116316, 3.8476085891292366, 3.448619117259054, 3.782688957832351, 10.150970562465517, 5.4364763972041, 3.989248055106449, 8.184081837269183, 4.616541732982609, 5.1174976136952415, 4.170217738089484, 6.734491831688603, 3.6962243834634054, 3.5663079382772267, 3.6641555750102417, 3.3316934357461734, 2.6727324551468032, 3.365051307156285, 2.566172012043256, 2.9987413082819074, 3.348604808265606, 3.335971036755129, 3.098709988226896, 2.6430123715456704, 3.3869342594819467, 2.969803523931427, 3.6062834426995583, 3.444215202695553, 2.5595138367932684, 2.51941521932443, 3.564173753057894, 3.4452708150549443, 6.249374587824506, 3.956032658040514, 2.641131031604613, 3.946921492061323, 4.7841790195942355, 3.599657974216046, 7.800277582105614, 4.038017237431269, 4.411237679040309, 4.749821397790129, 3.708602084025886, 3.752794788732597, 5.30563705984704, 4.525160971573917, 5.006429245350939, 3.1644512308521935, 4.0174104716006, 3.2838867156569154, 3.6973703860683487, 3.088737116898019, 3.2686118433612936, 3.4182889833790893, 2.639771985688331, 3.8557709880001294, 2.5638986038781515, 3.1692206931907307, 3.8532769181875515, 3.465583044355727, 3.75088798511424, 4.057443768231954, 6.437889297108959, 4.527591648951551, 4.398317984459067, 3.7435839895778056, 3.0253608142221675, 4.167114834587434, 3.457768748412153, 3.4327919153702844, 5.9844958687102405, 6.125741913666662, 3.109217158999728, 3.693667958726782, 3.588156409791107, 3.7801563432208454, 3.4008467388215715, 4.23901928016811, 3.2900171369321, 2.9803111887019633, 3.5579464354014427, 2.687274494084384, 4.005321845793136, 3.067753600963732, 3.163619591353659, 3.569704038691112, 4.4800801321167185, 3.955066696331164, 5.218606816209583, 4.619731554942148, 4.129269695168524, 3.9637156629530117, 3.970783488037086, 3.4686070198493923, 4.409091839936523, 2.8530679619087405, 3.715488566953205, 4.649772884961138, 4.2475638513484, 4.684440439418472, 4.309352318408342, 4.240242479964688, 4.515218753335394, 3.2088043847084027, 3.5306468208020947, 3.755394007662246, 4.83435483857215, 5.013454903944099, 4.568899813989224, 3.590780668740066, 4.420362537232444, 3.2730454983907116, 3.284028741414029, 5.041612633904207, 5.754724065036193, 4.945504805317297, 4.1044193706686745, 3.9052389698389267, 4.86477584295832, 4.046124728730652, 2.94550192744328, 3.4611984201967987, 3.132142010686258, 3.6714098083226863, 5.304607481008116, 3.4450064563830294, 4.30090319077138, 3.61400684694581, 3.823621924869046, 3.2369432323840455, 3.2742809006914575, 3.2651590761924223, 2.947038918113245, 3.510703340960524, 4.372255761392313, 3.232045592246152, 3.2161840314062906, 3.2663201378713347, 3.7057043805215297, 4.205565218174108, 3.6386107712342497, 4.703254311971374, 3.8650864552055233, 5.396252802775457, 3.177479841618605, 1.882227409513082, 1.9400528002145914, 2.7377807614171177, 7.850889373821774, 3.719078815590875, 2.0722978021413567, 7.877660657983842, 1.9903190366231616, 1.8547553698311026, 3.258163227061649, 2.6414274530295705, 3.4182941516950907, 0.6604772643827167, 3.1317880226868047, 2.3323061598491943, 2.874637304404038, 3.652873503773389, 4.420475840103912, 2.5868943045378288, 3.8429192914550425, 6.11500279428434, 7.017753602471205, 2.042991661884112, 7.208769915083154, 7.324261353484736, 4.74400132312446, 6.536649317856841, 4.051359318539351, 3.9018691350567263, 3.2309536414950406, 3.0242905799928423, 10.642313212115036, 3.878001997483371, 2.2530220335284543, 5.904013014870791, 9.805866048891122, 6.185921044785133, 6.271767918410009, 5.542697843994766, 2.8149823943954355, 10.334677910028635, 3.2591484668341173, 8.710041187211774, 6.504888217119547, 5.615142634281266, 10.537618258649678, 3.905395336423651, 5.693905238423314, 3.92324715280119, 3.5220819699157073, 5.314759152958165, 2.350202597308419, 4.802270921347393, 3.6925276793783115, 2.799548979989038, 2.4294616514169025, 3.64706950893475, 2.0395819018608563, 3.210992673092125, 3.0765772370945115, 4.796787403098434, 3.206779865769096, 3.665096922061963, 2.980644275499074, 3.3749023698726868, 2.752971750162203, 2.539899167348338, 3.0369977709230263, 3.2750408411985505, 2.013300061581942, 2.8107022459522355, 7.348769401602191, 3.7010806998474344, 4.073393144407596, 3.5578377064480615, 3.8659460819516376, 4.103562528158768, 6.338319023762274, 3.024215675577327, 3.3459864742986944, 6.911152465637594, 4.754637720565594, 4.93893296962179, 6.391228950133309, 6.110666781385554, 2.9243659432636573, 3.962564162636202, 3.3366787024160263, 3.635361200448351, 4.65531900828006, 3.7272431217658686, 2.9729557406326275, 3.094974881829863, 3.672090716204533, 3.6366283014201315, 3.6094449052847866, 4.900372454064727, 4.0125476759395635, 3.71626941200448, 3.786321873906446, 3.1615555446289734, 3.1891804168904114, 2.377338249498704, 2.5985409079156416, 2.386609491113936, 1.9371211174574967, 2.4875844379265923, 2.277167630396793, 4.377661873743616, 3.9809952741692403, 7.149664254638934, 9.269048750307633, 2.2107934353377403, 2.6512268369873517, 6.180800062346726, 2.4010070757385393, 2.712716507755105, 4.208305266303304, 6.187943237208393, 3.014273804677258, 3.1784045967555627, 5.99412307545371, 4.408354126582998, 3.5773407930694807, 4.906509621056362]

    print(mean_errors)
    print(np.mean(mean_errors))
    # print(minTot)
    # print(maxTot)
    fig = plt.figure()
    # plt.plot(mean_errors)
    # plt.boxplot(all_sphere_dist)
    plt.boxplot(mean_errors)
    fig.savefig("boxplotTotal.png", dpi=fig.dpi)

    plt.show()
