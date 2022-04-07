import colorsys
import numpy as np
import skimage as ski
import skimage.color

def hls_to_rgb(color):
    return colorsys.hls_to_rgb(color[0], color[1], color[2])

# image.shape (h,w) dtype=whatever
# output.shape (h,w,3) min 0 max 255 dtype=uint8
def convert_gray_to_hsv_mapping(image, max_value, max_hue=0.84):
    image = image.copy()
    image[image > max_value] = max_value
    image_hsv = np.zeros((image.shape + (3,)), dtype=np.float64) # shape (h,w,3)
    image_hsv[:,:,0] = image[:,:]/max_value*max_hue
    image_hsv[:,:,1] = 1
    image_hsv[:,:,2] = 1
    image = ski.color.hsv2rgb(image_hsv) # shape (h,w,3) min 0 max 1 dtype=float64
    image = (image*255).astype(np.uint8) # shape (h,w,3) min 0 max 255 dtype=uint8
    return image
