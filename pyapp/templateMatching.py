import cv2 as cv
import matplotlib.pyplot as plt

import numpy as np

import glob
import os

import random


def get_line(x1, y1, x2, y2):
  points = []
  issteep = abs(y2 - y1) > abs(x2 - x1)
  if issteep:
    x1, y1 = y1, x1
    x2, y2 = y2, x2
  rev = False
  if x1 > x2:
    x1, x2 = x2, x1
    y1, y2 = y2, y1
    rev = True
  deltax = x2 - x1
  deltay = abs(y2 - y1)
  error = int(deltax / 2)
  y = y1
  ystep = None
  if y1 < y2:
    ystep = 1
  else:
    ystep = -1
  for x in range(x1, x2 + 1):
    if issteep:
      points.append((y, x))
    else:
      points.append((x, y))
    error -= deltay
    if error < 0:
      y += ystep
      error += deltax
  # Reverse the list if the coordinates were reversed
  if rev:
    points.reverse()
  return points


def rotate_line_counterclockwise(lines, h, w):
  for i in range(4):
    p1, p2 = lines[i]
    p1 = [p1[1], w-p1[0]]
    p2 = [p2[1], w-p2[0]]
    lines[i] = [p1, p2]

def rotate_line_clockwise(lines, h, w):
  for i in range(4):
    p1, p2 = lines[i]
    p1 = [h-p1[1], p1[0]]
    p2 = [h-p2[1], p2[0]]
    lines[i] = [p1, p2]

def rotate_point_counterclockwise(point, h, w):
  for i in range(4):
    p = point
    p = [p[1], w-p[0]]
    return p

def rotate_point_clockwise(point, h, w):
  for i in range(4):
    p = point
    p = [h-p[1], p[0]]
    return p


def find_sphere(img, templates, pt1, pt2, valid, line_thickness=10):
  scores = []
  pos_list = []
  x = []
  y = []
  for (templ_gray, templ_alpha) in templates:
    h, w = templ_gray.shape

    # Make a mask with the search space
    # It initially is all zeros then
    # we use the cv.line to draw a line which will add valid spaces
    mask = np.zeros_like(img)
    cv.line(mask, np.int0(pt1) + [w//2, h//2], np.int0(pt2) + [w//2, h//2], 255, line_thickness)
    _, mask = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)

    pointsOnLine = get_line( np.int0(pt1[0] + (w//2)), np.int0(pt1[1]) + (h//2),
              np.int0(pt2[0]) + (w//2) , np.int0(pt2[1]) + (h//2))

    # print("line:\n")
    # print(pointsOnLine)


    mask[valid == 255] = 0

    # Crop the mask because matchTemplate will return an image map smaller than the image itself.
    mask = mask[h-1:,w-1:]

    # Template matching with a mask
    res = cv.matchTemplate(img, templ_gray, cv.TM_CCOEFF_NORMED, mask=templ_alpha)
    # print(res)
    # plt.plot(res[0])
    # resValues = []
    # for i, (l, r) in enumerate(pointsOnLine):
    #   rval = res[l, r]
    #   resValues.append(rval)

    # fig = plt.figure()
    # plt.plot(resValues)
    # fig.savefig("template_metrics_" + str(i) + ".png", dpi=fig.dpi)
    # plt.show()


    # Remove any match which is invalid (not on the line AND not in valid)
    res[mask != 255] = 0

    # In some cases, it can output NaN/Inf, not really why
    # and this could be picked up by the max function
    res[np.isnan(res)] = 0
    res[np.isinf(res)] = 0

    # Get the 2D position of the match
    pos = np.int0(np.unravel_index(np.argmax(res), res.shape))
    # x.append(pos[0])
    # y.append(pos[1])
    scores.append(res[pos[0], pos[1]])
    # print(scores)
    pos_list.append(pos)

  # Find the best template match among ALL the templates
  best_idx = np.argmax(scores)

  # plt.savefig(str(best_idx) + '_foo.png')

  pos = pos_list[best_idx][::-1]

  # Mark the match in valid
  h, w = templates[best_idx][0].shape
  cv.rectangle(valid, pos, pos+[2*w,2*h], 255, -1)

  # Shift by half to get the center
  return pos + [w//2, h//2]

if  __name__ == '__main__':

  # Load input image as grayscale
  left = cv.imread("left2.png", cv.IMREAD_GRAYSCALE)
  h, w = left.shape[:2]
  left = cv.rotate(left, cv.ROTATE_90_CLOCKWISE)
  newH, newW = left.shape[:2]
  right = cv.imread("right2.png", cv.IMREAD_GRAYSCALE)
  right = cv.rotate(right, cv.ROTATE_90_COUNTERCLOCKWISE)

  # Convert to RGB to draw the colored circles
  left_annotated = cv.cvtColor(left, cv.COLOR_GRAY2BGR)
  right_annotated = cv.cvtColor(right, cv.COLOR_GRAY2BGR)

  leftRotatedBack = cv.rotate(left_annotated, cv.ROTATE_90_COUNTERCLOCKWISE)
  rightRotatedBack = cv.rotate(right_annotated, cv.ROTATE_90_CLOCKWISE)


  # Coordinates of line got from projectLine.py
  # left_lines = [[[308, 192], [253, 356]], [[345, 150], [291, 314]], [[309, 126], [253, 285]], [[268, 155], [210, 311]]]
  # right_lines = [[[316, 28], [368, 122]], [[279, 64], [329, 164]], [[314, 94], [367, 194]], [[355, 73], [410, 168]]]

  left_lines = [
    [[391, 97], [341, 262]],
    [[374, 80], [322, 241]],
    [[361, 128],[309, 293]],
    [[347, 99], [293, 259]]
  ]
  right_lines = [
    [[233, 108], [278, 216]],
    [[249, 129], [296, 238]],
    [[263,  83], [311, 186]],
    [[277, 114], [326, 220]]
  ]

  # left_lines = [[[345, 150], [291, 314]], [[309, 192], [253, 356]], [[309, 126], [253, 285]], [[268, 155], [210, 311]]]
  # right_lines = [[[279,  64], [329, 164]], [[315,  28], [368, 122]], [[314,  94], [367, 194]], [[355,  73], [410, 168]]]

  rotate_line_clockwise(left_lines, h, w)
  rotate_line_counterclockwise(right_lines, h, w)

  # Convert templates points to floats
  for i in range(4):
    p1, p2 = left_lines[i]
    left_lines[i] = [np.float32(p1), np.float32(p2)]
    cv.line(left_annotated, p1, p2, (255,0,0), 1)

  for i in range(4):
    p1, p2 = right_lines[i]
    right_lines[i] = [np.float32(p1), np.float32(p2)]
    cv.line(right_annotated, p1, p2, (255,0,0), 1)

  # Load templates located in the "templates" folder
  # Each template can have an alpha component which will be used as a mask
  fns = glob.glob(os.path.join("templates", "*"))
  print(fns)

  templates = []
  for fn in fns:
    # Load with alpha component
    templ = cv.imread(fn, cv.IMREAD_UNCHANGED)

    # Split into color/alpha
    templ_gray = cv.cvtColor(templ[:,:,:3], cv.COLOR_BGR2GRAY)
    if templ.shape[2] == 4:
      templ_alpha = templ[:,:,3]
    else:
      # Create an opaque plane if no alpha in image
      templ_alpha = np.ones(templ_gray.shape[:2], np.uint8)

    templates.append((templ_gray, templ_alpha))


  # Map of valid to avoid overlaps
  # It will be filled by the find_sphere function
  left_valid = np.zeros(left.shape[:2], np.uint8)

  for i in range(4):
    pos = find_sphere(left, templates, left_lines[i][0], left_lines[i][1], left_valid)
    # print(rotate_point_clockwise(pos, newH, newW))
    #cv.circle(leftRotatedBack, rotate_point_counterclockwise(pos, newH, newW), 8, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 1)
    cv.circle(left_annotated, pos, 8, (0,128,0), 1)
    # cv.putText(left_annotated, f' {i}', pos, cv.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), lineType=cv.LINE_AA)

  right_valid = np.zeros(right.shape[:2], np.uint8)

  for i in range(4):
    pos = find_sphere(right, templates, right_lines[i][0], right_lines[i][1], right_valid)
    cv.circle(rightRotatedBack, rotate_point_clockwise(pos, newH, newW), 8, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 1)

    # cv.circle(right_annotated, pos, 8, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 1)
    # cv.putText(right_annotated, f' {i}', pos, cv.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), lineType=cv.LINE_AA)



  # Show results
  plt.subplot(121)
  plt.imshow(left_annotated[:, :, ::-1])
  plt.title("left")
  plt.axis('off')

  # plt.subplot(122)
  # plt.imshow(left_valid)
  # plt.title("valid")
  # plt.show()
  #
  # plt.subplot(122)
  # plt.imshow(right_valid)
  # plt.title("valid")
  # plt.show()

  plt.subplot(122)
  plt.imshow(rightRotatedBack[:,:,::-1])
  plt.title("right")
  plt.axis('off')
  plt.show()


