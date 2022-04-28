import cv2
from matplotlib import pyplot as plt

imgL = cv2.imread('imageLeft',0)
imgR = cv2.imread('imageRight',0)

stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disp = stereo.compute(imgL,imgR)
plt.imshow(disp,'gray')
plt.show()