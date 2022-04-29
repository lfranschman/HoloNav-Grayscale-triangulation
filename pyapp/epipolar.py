import numpy as np
import cv2
from matplotlib import pyplot as plt
from config import config
import skimage as ski
import skimage.io
from pyapp.DataAcquisition import DataAcquisition
def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rotation_matrx = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  res = cv2.warpAffine(image, rotation_matrx, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return res

def drawlinesAndPoints(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r, pointsLeft, pointsRight in zip(lines, pts1 ,pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        img1 = cv2.circle(img1, tuple(pointsLeft), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pointsRight), 5, color, -1)
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)

    return img1,img2

def find_fund_matrix(image1, image2):
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors
    keyPointsLeft, descrLeft = sift.detectAndCompute(img1,None)
    keyPointsRight, descrRight = sift.detectAndCompute(img2,None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flannObject = cv2.FlannBasedMatcher(index_params,search_params)
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
    fundMatrix, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
    return fundMatrix, mask, pts1, pts2

if  __name__ == '__main__':
    # data = DataAcquisition()
    # data.load_data(config.get_filename("optical_sphere"))
    # frame = np.copy(data.acquisitions["vl_front_right_cam" + '_frames'][78])
    # ski.io.imsave(f"test_images/{'vl_front_right_cam'}_{78:04.0f}.png", frame)
    # left:
    # frame_id 97
    # center of the sphere 0: (319.71857546531635, 260.6429657947044)
    # center of the sphere 1: (277.2733399114904, 229.57999440448162)
    # center of the sphere 2: (226.79491166784646, 258.66479608478386)
    # center of the sphere 3: (276.01048072657915, 306.5909886519823)
    #
    # right:
    # frame_id 97
    # center of the sphere 0: (320.74284086414787, 101.27987236026208)
    # center of the sphere 1: (365.8998185654118, 132.0215938672441)
    # center of the sphere 2: (419.43092116436685, 97.89421957313039)
    # center of the sphere 3: (365.8397009965465, 48.99749282748726)
    img1 = cv2.imread(r'C:\Users\Lesle\OneDrive\Documenten\GitHub\holonav\pyapp\test_images\vl_front_left_cam_0088.png', 0)  #queryimage # left image
    img2 = cv2.imread(r'C:\Users\Lesle\OneDrive\Documenten\GitHub\holonav\pyapp\test_images\vl_front_right_cam_0078.png', 0) #trainimage # right image
    img1 = rotate_image(img1, -90)
    img2 = rotate_image(img2, 90)
    fundMatrix, mask, pointsleft, pointsright = find_fund_matrix(img1, img2)
    # pointsLeft = np.array(np.ndarray(319.71857546531635, 260.6429657947044), np.ndarray(277.2733399114904, 229.57999440448162),
    #                        np.ndarray(319.71857546531635, 260.6429657947044), np.ndarray(319.71857546531635, 260.6429657947044), dtype=float)
    # print(type(pointsleft[0]))

    # Inlier points only
    pointsleft = pointsleft[mask.ravel()==1]
    pointsright = pointsright[mask.ravel()==1]

    # draw points and epilines in the left image
    lines1 = cv2.computeCorrespondEpilines(pointsright.reshape(-1, 1, 2), 2, fundMatrix)
    lines1 = lines1.reshape(-1, 3)
    resImageLeft, helper1 = drawlinesAndPoints(img1, img2, lines1, pointsleft, pointsright)

    # draw points and epilines in the right image
    lines2 = cv2.computeCorrespondEpilines(pointsleft.reshape(-1, 1, 2), 1, fundMatrix)
    lines2 = lines2.reshape(-1, 3)
    resImageRight, helper2 = drawlinesAndPoints(img2, img1, lines2, pointsright, pointsleft)



    plt.subplot(121)
    plt.imshow(resImageLeft)
    plt.subplot(122)
    plt.imshow(resImageRight)
    plt.show()