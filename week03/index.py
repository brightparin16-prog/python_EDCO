import cv2

img = cv2.imread("istockphoto-140472118-612x612.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# cv2.imshow("WRG",rgb)

# resize of img
new_width = 408
new_height = 612

resized_image = cv2.resize(img, (new_height, new_width))


cv2.show("Result of image, resized_image")

cv2.waitKey(0)
cv2.destroyAllWindow()