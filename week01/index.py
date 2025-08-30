# import คือการเรียกใช้งานตัว Library หรือ Module ตัว OpenCV โดย ใช้ชื่อว่า cv2
import cv2
#The name of libraly[cv2] |,.The named fo a function()
# .imread() ใช้ในการอ่านรูปภา | .imread() Used to read an images
 

img = cv2.imread("week01/meme.png")

# .imwrite() Used to new an images | incli=ude 2 argument (1. The named of new file  ["new"], 2.image)
cv2.imwrite("new.png", img)
# .imshow() Show an images
cv2.imshow("show an image", img)

# .waitKey(0) Press any button for...
# .destroyallWindow() Close a window
cv2.waitKey(0)
cv2.destroyAllWindow()
