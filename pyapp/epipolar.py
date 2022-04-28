import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def drawlinesAndPoints(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img1.shape
    img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for r, pointsLeft, pointsRight in zip(lines, pts1 ,pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        img1 = cv.circle(img1, tuple(pointsLeft), 5, color, -1)
        img2 = cv.circle(img2, tuple(pointsRight), 5, color, -1)
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)

    return img1,img2

def find_fund_matrix(image1, image2):
    sift = cv.SIFT_create()

    # find the keypoints and descriptors
    keyPointsLeft, descrLeft = sift.detectAndCompute(img1,None)
    keyPointsRight, descrRight = sift.detectAndCompute(img2,None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flannObject = cv.FlannBasedMatcher(index_params,search_params)
    matches = flannObject.knnMatch(descrLeft,descrRight,k=2)
    pointsleft = []
    pointsRight = []
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            pointsleft.append(keyPointsLeft[m.queryIdx].pt)
            pointsRight.append(keyPointsRight[m.trainIdx].pt)


    pts1 = np.int32(pointsleft)
    pts2 = np.int32(pointsRight)
    fundMatrix, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_LMEDS)
    return fundMatrix, mask, pts1, pts2

if  __name__ == '__main__':
    img1 = cv.imread(r'C:\Users\Lesle\OneDrive\Documenten\GitHub\holonav\pyapp\test_images\vl_front_left_cam_0088.png',0)  #queryimage # left image
    img2 = cv.imread(r'C:\Users\Lesle\OneDrive\Documenten\GitHub\holonav\pyapp\test_images\vl_front_right_cam_0088.png',0) #trainimage # right image

    fundMatrix, mask, pointsleft, pointsright = find_fund_matrix(img1, img2)

    # Inlier points only
    pointsleft = pointsleft[mask.ravel()==1]
    pointsright = pointsright[mask.ravel()==1]

    # draw points and epilines in the left image
    lines1 = cv.computeCorrespondEpilines(pointsright.reshape(-1, 1, 2), 2, fundMatrix)
    lines1 = lines1.reshape(-1, 3)
    resImageLeft, helper1 = drawlinesAndPoints(img1, img2, lines1, pointsleft, pointsright)

    # draw points and epilines in the right image
    lines2 = cv.computeCorrespondEpilines(pointsleft.reshape(-1, 1, 2), 1, fundMatrix)
    lines2 = lines2.reshape(-1, 3)
    resImageRight, helper2 = drawlinesAndPoints(img2, img1, lines2, pointsright, pointsleft)

    plt.subplot(121)
    plt.imshow(resImageLeft)
    plt.subplot(122)
    plt.imshow(resImageRight)
    plt.show()