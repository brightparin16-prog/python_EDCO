# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# print(np.__version__)

# img = cv2.imshow("C:/Users/Asus/Desktop/python-couse2/WRG-championship-logo-new.png")

# degree = [15, 45, , 90, 180]
# rotation_image = []

# center = (width // 2 height // 2)
# # for loop use to know about circle round of data
# for x in degree:
    rotation_matrix = cv2.getRotationMatrix2D()

# rotation = cv2.warpAffine(img, rotation)
# # degree = 90
# # rotation_image = []
# # cv2.imshow("Rotation", img)

# cv2.waitKey(0)
# cv2.destroyAllWindows()