import cv2

img = cv2.imread("new.png")
#  .cvtCollor is's mean convert color | include with 2 argumant (1.image, function to color[gray])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("Show an image", rgb)
cv2.waitKey(0)
cv2.destroyallwindow()