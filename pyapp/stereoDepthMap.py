import cv2
from matplotlib import pyplot as plt

def create_disparity_map(img1, img2):

    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disp = stereo.compute(img1,img2)
    plt.imshow(disp,'gray')
    plt.show()