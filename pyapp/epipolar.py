import cv2
import numpy as np
from matplotlib import pyplot as plt
import skimage as ski
import skimage.io
from config import config
from python.common.UtilImage import draw_disk

from pyapp.DataAcquisition import DataAcquisition


def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rotation_matrx = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  res = cv2.warpAffine(image, rotation_matrx, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return res

def drawPoints(image, pts, colors):
    for pt, color in zip(pts, colors):
        cv2.circle(image, tuple(pt[0]), 5, color, -1)

def drawLines(image, lines, colors):
    _, c, _ = image.shape
    for r, color in zip(lines, colors):
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        cv2.line(image, (x0, y0), (x1, y1), color, 1)

def drawlinesAndPoints(img1, img2, lines,points1,points2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pointsLeft, pointsRight in zip(lines, points1 ,points2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        img1 = cv2.circle(img1, tuple(pointsLeft), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pointsRight), 5, color, -1)
        x0, y0 = map(int, [0, -r[2]/r[1] ])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)

    return img1, img2



def find_fund_matrix(image1, image2):
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors
    keyPointsLeft, descrLeft = sift.detectAndCompute(image1,None)
    keyPointsRight, descrRight = sift.detectAndCompute(image2,None)
    # FLANN parameters
    FLANN_INDEX_KMEANS = 1
    index_params = dict(algorithm = FLANN_INDEX_KMEANS, trees = 5)
    search_params = dict(checks=50)
    flannObject = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flannObject.knnMatch(descrLeft,descrRight,k=2)
    pointsLeft = []
    pointsRight = []
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            pointsRight.append(keyPointsRight[m.trainIdx].pt)
            pointsLeft.append(keyPointsLeft[m.queryIdx].pt)


    pointsLeft = np.int32(pointsLeft)
    pointsRight = np.int32(pointsRight)
    fundMatrix, mask = cv2.findFundamentalMat(pointsLeft,pointsRight,cv2.FM_LMEDS)
    return fundMatrix, mask, pointsLeft, pointsRight

if  __name__ == '__main__':
    data = DataAcquisition()
    data.load_data(config.get_filename("optical_sphere"))
    # # # # frame = np.copy(data.acquisitions['ahat_depth_cam_ab_frames'][84])
    # # # # plt.imshow(frame)
    # # # # plt.show()
    # # # # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # # # # ski.io.imsave(f"test_images/{'infrared'}_{84:04.0f}.png", frame)
    frame1 = np.copy(data.acquisitions['lt_depth_cam_ab_frames'][61])
    frame2 = np.copy(data.acquisitions['ahat_depth_cam_ab_frames'][84])
    frame2 = (frame2/256).astype('uint8')
    ret, thresh1 = cv2.threshold(frame1, 10000, 24000, cv2.THRESH_BINARY)
    ret2, thresh2 = cv2.threshold(frame2, 4.5, 256, cv2.THRESH_BINARY_INV)
    # # plt.imshow(thresh2, interpolation='nearest')
    # # plt.imshow(thresh2, interpolation='nearest', cmap='gray', vmin=0, vmax=255)
    # # plt.show()
    #
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 0;
    params.maxThreshold = 255;

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 15

    detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(thresh2)
    im_with_keypoints = cv2.drawKeypoints(thresh2, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)    # left:
    cv2.imshow("Keypoints", im_with_keypoints)
    #cv2.waitKey(0)

    frameCopy = np.copy(data.acquisitions['ahat_depth_cam_ab_frames'][84])
    infraredPoints = []
    for keypoint in keypoints:
      coord = np.array(keypoint.pt)
      infraredPoints.append(coord)
      print(coord)
      draw_disk(frameCopy, coord[0], coord[1], 0, size=1)

    infraredPoints = np.int32(infraredPoints)
    # print(infraredPoints)
    # plt.imshow(frameCopy, interpolation='nearest', vmin=0, vmax=255)
    # plt.show()# frame_id 92
    # center of the sphere 0: (313.3509022842888, 262.47544764121005)
    # center of the sphere 1: (272.8489279548646, 233.6973695674395)
    # center of the sphere 2: (227.06630654179853, 261.80506340088226)
    # center of the sphere 3: (272.94404080837654, 306.435626136948)

    # right:
    # frame_id 82
    # center of the sphere 0: (308.957572937851, 116.63542215668062)
    # center of the sphere 1: (349.07924505516473, 147.23053501449542)
    # center of the sphere 2: (395.24553983842543, 120.35931920462032)
    # center of the sphere 3: (349.9330255618248, 73.21466985931173)
    #
    img1 = cv2.imread(r'C:\Users\Lesle\OneDrive\Documenten\GitHub\holonav\pyapp\generated\vl_front_left_cam_0092.png', 0)
    img2 = cv2.imread(r'C:\Users\Lesle\OneDrive\Documenten\GitHub\holonav\pyapp\generated\vl_front_left_cam_0082.png', 0)

    infraredImage = cv2.imread(r'C:\Users\Lesle\OneDrive\Documenten\GitHub\holonav\pyapp\test_images\infrared_0084.png', 0)
    #img1 = rotate_image(img1, -90)
    #img2 = rotate_image(img2, 180)
    fundMatrix, mask, pointsLeft, pointsRight = find_fund_matrix(img1, infraredImage)
    fundMatrix2, mask2, pointsLeft2, pointsRight2 = find_fund_matrix(infraredImage, img2)

    # #
    # pointsLeft2 = np.array([[313.3509022842888, 262.47544764121005], [272.8489279548646, 233.6973695674395],
    #                         [227.06630654179853, 261.80506340088226], [272.94404080837654, 306.435626136948]])
    # #
    # # pointsRight = np.array([[308.957572937851, 116.63542215668062], [349.07924505516473, 147.23053501449542],
    # #                   [395.24553983842543, 120.35931920462032], [349.9330255618248, 73.21466985931173]])
    # # pointsRight = infraredPoints
    # pointsRight2 = np.array([[194.64927, 189.80580139], [166.10734558, 164.28530884],
    #                         [215.20489502, 161.12538147], [196.64306641, 131.87905884]])
    # # #
    pointsRight = infraredPoints
    # pointsRight2 = np.int32(pointsRight2)
    # #
    # pointsLeft = infraredPoints
    pointsLeft2 = infraredPoints
    # #
    # # # Inlier points only
    pointsLeft = pointsLeft[mask.ravel()==1]
    pointsRight2 = pointsRight2[mask2.ravel()==1]
    # #
    # # # draw points and epilines in the left image
    lines1 = cv2.computeCorrespondEpilines(pointsRight.reshape(-1, 1, 2), 2, fundMatrix)
    lines1 = lines1.reshape(-1, 3)
    resImageLeft, helper1 = drawlinesAndPoints(img1, infraredImage, lines1, pointsLeft, pointsRight)
    # #
    # draw points and epilines in the right image
    lines2 = cv2.computeCorrespondEpilines(pointsLeft.reshape(-1, 1, 2), 1, fundMatrix)
    lines2 = lines2.reshape(-1, 3)
    resImageRight, helper2 = drawlinesAndPoints(infraredImage, img1, lines2, pointsRight, pointsLeft)
    #
    lines3 = cv2.computeCorrespondEpilines(pointsRight2.reshape(-1, 1, 2), 2, fundMatrix2)
    lines3 = lines3.reshape(-1, 3)                                                                        #
    resImageLeft2, helper3 = drawlinesAndPoints(infraredImage, img2, lines3, pointsLeft2, pointsRight2)      # # stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
                                                                                                       # # disp = stereo.compute(img1, img2)
    # draw points and epilines in the right image                                                         # #plt.imshow(disp, 'gray')
    lines4 = cv2.computeCorrespondEpilines(pointsLeft2.reshape(-1, 1, 2), 1, fundMatrix2)                   #
    lines4 = lines4.reshape(-1, 3)                                                                        # resImageLeft = rotate_image(resImageLeft, 90)
    resImageRight2, helper4 = drawlinesAndPoints(img2, infraredImage, lines4, pointsRight2, pointsLeft2)     # resImageRight = rotate_image(resImageLeft, -90)

    #resImageRight = rotate_image(resImageRight, 90)
    plt.subplot(121)
    plt.imshow(resImageRight)
    # # # #ski.io.imsave(f"results/{'common_points_epilines_left'}_{4:04.0f}.png", resImageLeft)
    plt.subplot(122)
    plt.imshow(resImageLeft2)
    # # plt.imshow(helper1)
    # plt.subplot(121)
    # plt.imshow(resImageRight2)
    # plt.subplot(122)
    # plt.imshow(resImageLeft2)
    plt.savefig("used_infrared.png")
     #ski.io.imsave(f"results/{'common_points_epilines_right'}_{4:04.0f}.png", resImageRight)
    plt.show()

    # gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    # frame2 = frame2.astype(np.uint8)
    # plt.imshow(frame2, interpolation='nearest', vmin=0, vmax=255)
    #
    # ski.io.imsave(f"test_images/{'infrared'}_{84:04.0f}.png", frame2)
    # plt.show()