import cv2
from matplotlib import pyplot as plt

#Perfroming the following steps:
#calibrate individual cameras
#stereo calibrate the cameras
#calculate and draw epipolar lines
#rectify the images such that the epipolar lines in both images are parrallel
#call the function below to create the depthmap based on the disparity

def create_disparity_map(img1, img2):

    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disp = stereo.compute(img1,img2)
    plt.imshow(disp,'gray')
    plt.show()