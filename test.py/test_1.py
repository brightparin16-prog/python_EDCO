import cv2

img = cv2.imread("rainbow.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("Show rainbow image", gray)
cv2.waitKey(0)
cv2.destroyallwindow()