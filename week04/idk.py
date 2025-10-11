import cv2

# print( cv2.__version__)

import numpy as np
# การคำนวน math

paper = np.ones((600, 800, 3), dtype=np.uint8) * 255

cv2.rectangle(paper, (100,250), (300, 150), (255, 157, 0), 5)
cv2.circle(paper, (400, 150), 70, (0, 255, 0), -1)

pts = np.array([[550, 220], [700, 60], [750, 120]], np.int32)

# cv2.fillPoly(paper, [pts], (0, 0, 255))
cv2.fillPoly(paper, [pts], (0, 0, 255))

# for x in range():
#     cv2.circle(paper, (x, 350), 30, (0, 0, 0), -1)

cv2.imwrite("shape_test.png", paper)

print("Create a new file shape_test.png")
