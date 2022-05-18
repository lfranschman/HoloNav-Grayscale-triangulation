import cv2
# Read the original image
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

import numpy as np
from config import config
from pyapp.DataAcquisition import DataAcquisition


def canny():
    img = cv2.imread(r"C:\Users\Lesle\OneDrive\Documenten\GitHub\holonav\pyapp\test_images\vl_front_left_cam_0000.png")
    # Display original image

    # cv2.imshow('Original', img)
    # cv2.waitKey(0)
    # Convert to graycsale
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img, (3, 3), 0)
    # Sobel Edge Detection
    # sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=1)  # Sobel Edge Detection on the X axis
    # sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=1)  # Sobel Edge Detection on the Y axis
    # sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=1)  # Combined X and Y Sobel Edge Detection
    # Display Sobel Edge Detection Images
    # cv2.imshow('Sobel X', sobelx)
    # cv2.waitKey(0)
    # cv2.imshow('Sobel Y', sobely)
    # cv2.waitKey(0)
    # cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
    # cv2.waitKey(0)
    # Canny Edge Detection
    edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=300)  # Canny Edge Detection
    # # Display Canny Edge Detection Image
    cv2.imshow('Canny Edge Detection', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def thresholding():
    data = DataAcquisition()
    data.load_data(config.get_filename("optical_sphere"))
    # read image
    frame1 = np.copy(data.acquisitions['vl_front_left_cam_frames'][0])

    # img = cv2.imread(r"C:\Users\Lesle\OneDrive\Documenten\GitHub\holonav\pyapp\test_images\vl_front_left_cam_0000.png")

    # convert to gray
    # gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    # threshold
    thresh = cv2.threshold(frame1, 128, 255, cv2.THRESH_BINARY)[1]

    # morphology edgeout = dilated_mask - mask
    # morphology dilate
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilate = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)

    # get absolute difference between dilate and thresh
    diff = cv2.absdiff(dilate, thresh)

    # invert
    edges = 255 - diff

    # write result to disk
    cv2.imwrite("results/dthresh.jpg", thresh)
    cv2.imwrite("results/dilate.jpg", dilate)
    cv2.imwrite("results/diff.jpg", diff)
    cv2.imwrite("results/edges.jpg", edges)

    # display it
    # cv2.imshow("thresh", thresh)
    # cv2.imshow("dilate", dilate)
    # cv2.imshow("diff", diff)
    # cv2.imshow("edges", edges)
    #cv2.waitKey(0)

if  __name__ == '__main__':
    # canny()
    # thresholding()
    img = mpimg.imread(r"C:\Users\Lesle\OneDrive\Documenten\GitHub\holonav\pyapp\test_images\vl_front_left_cam_0000.png")
    plt.imshow(img[200:300, 100:300])
    plt.show()
