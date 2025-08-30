import cv2
 

img = cv2.imread("istockphoto-140472118-612x612.jpg")


height, width = img.shape[:2]

print(f"Standrad of size an image: {width}x{height}")

cv2.imshow("Result", img)

cv2.waitKey(0)
cv2.destroyAllWindows