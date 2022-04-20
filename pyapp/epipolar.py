import cv2
from matplotlib import pyplot as plt

def epilines():
    return

def fund_matrix(image1, image2):
    # Load the greyscale images
    leftImage = cv2.imread(image1, 0)
    rightImage = cv2.imread(image2,0)

    # Find SIFT key points and descriptors for both images
    sift = cv2.xfeatures2d.SIFT_create()
    return
