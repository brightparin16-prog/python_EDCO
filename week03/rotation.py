import cv2
import numpy as np


print(np.__version__)

img = cv2.imshow("C:/Users/Asus/Desktop/python-couse2/WRG-championship-logo-new.png")


# degree = 90
# rotation_image = []
# cv2.imshow("Rotation", img)

cv2.waitKey(0)
cv2.destroyAllWindows()