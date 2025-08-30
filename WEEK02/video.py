import cv2

print(cv2.__version__)

cap = cv2.VideoCapture("C:/Users/Asus/Videos/360p-watermark.mp4")

while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Video Playblack", frame)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyWindow()