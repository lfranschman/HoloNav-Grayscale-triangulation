import cv2
import numpy as np
from matplotlib import pyplot as plt



def drawPoints(img, pts, colors):
    for pt, color in zip(pts, colors):
        cv2.circle(img, tuple(pt[0]), 5, color, -1)

def drawEpilines(img, lines, colors):
    _, c, _ = img.shape
    for r, color in zip(lines, colors):
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        cv2.line(img, (x0, y0), (x1, y1), color, 1)

def find_fund_matrix(image1, image2):
    # Load the greyscale images
    leftImage = cv2.imread(image1, 0)
    rightImage = cv2.imread(image2,0)

    # initiate pointvectors to store the matched points of both images
    pointsLeft = []
    pointsRight = []

    # Find as many possible matches between images as possible
    # Find SIFT key points and descriptors for both images
    # passing a mask to only search a part of the image can be donewith sift.detect()
    sift = cv2.SIFT_create()

    keyPointsLeft, descriptorsLeft = sift.detectAndCompute(leftImage, None)
    keyPointsRight, descriptorsRight = sift.detectAndCompute(rightImage, None)

    # create flann object for image points matching
    FLANN_INDEX_KDTREE = 1
    index_params = dict(FLANN_INDEX_KDTREE, 5)
    search_params = dict(40)

    flannObject = cv2.FlannBasedMatcher(index_params, search_params)
    matchedPoints = flannObject.knnMatch(descriptorsLeft, descriptorsRight, 2)

    # ratio test
    for i, (m, n) in enumerate(matchedPoints):
        if m.distance < 0.8 * n.distance:
            pointsLeft.append(keyPointsLeft[m.trainIdx].pt)
            pointsRight.append(keyPointsRight[n.trainIdx].pt)

    pointsLeft = np.int32(pointsLeft)
    pointsRight = np.int32(pointsRight)
    FundMatrix, mask = cv2.findFundamentalMat(pointsLeft, pointsRight, cv2.FM_LMEDS)
    # Inlier points only
    # pointsLeft = pointsLeft[mask.ravel() == 1]
    # pointsRight = pointsRight[mask.ravel() == 1]

    return FundMatrix, mask, pointsLeft, pointsRight

if __name__ == '__main__':


    FundMatrix, mask, pointsLeft, pointsRight = find_fund_matrix("image1", "image2")

    #sphere_location_left= ?
    #sphere_location_right =?

    # just use inliers
    pointsLeft = pointsLeft[mask.ravel() == 1]
    pointsRight = pointsRight[mask.ravel() == 1]

    colors = tuple(np.random.randint(0, 255, 6).tolist())
    drawPoints("imageLeft", pointsLeft, colors[3:6])
    drawPoints("imageRight", pointsRight, colors[0:3])

    # find epilines corresponding to points in right image and draw them on the left image
    epilinesRight = cv2.computeCorrespondEpilines(pointsRight.reshape(-1, 1, 2), 2, FundMatrix)
    epilinesRight = epilinesRight.reshape(-1, 3)
    drawEpilines("imageLeft", epilinesRight, colors[0:3])

    # find epilines corresponding to points in left image and draw them on the right image
    epilinesL = cv2.computeCorrespondEpilines(pointsLeft.reshape(-1, 1, 2), 1, FundMatrix)
    epilinesL = epilinesL.reshape(-1, 3)
    drawEpilines("imageRight", epilinesL, colors[3:6])

    plt.subplot(121), plt.imshow("imageLeft")
    plt.subplot(122), plt.imshow("imageRight")
    plt.show()