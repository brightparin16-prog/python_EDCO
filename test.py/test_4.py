import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

img = cv2.imread("people-taking-selfie-together-registration-day_23-2149096795.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(
    gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60)
)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)

cv2.imshow("family", img)
cv2.waitKey(0)
cv2.destroyALLWindow()