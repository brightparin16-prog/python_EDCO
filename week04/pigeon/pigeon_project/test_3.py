import cv2
from ultralytics import YOLO

# --- 1. Load a Lighter Model ---
# yolov8n is the smallest model, but you can use even smaller, 
# specialized versions if available (e.g., a heavily pruned custom model).
# For general speed, stick with yolov8n or try yolov8n-lite if available.
model = YOLO('yolov8n.pt') 

# --- 2. Initialize Webcam Capture ---
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# --- 3. Main Loop for Real-Time Processing ---
while True:
    ret, frame = cap.read() 
    
    if not ret:
        break

    # --- 4. Perform Faster Detection (YOLOv8) ---
    # Key optimizations for speed:
    # 1. 'imgsz=320': Reduce the input image size for faster processing (default is 640).
    # 2. 'conf=0.6': Increase the minimum confidence threshold to filter out weak detections earlier.
    # 3. 'device=0' (or 'cuda'): Specify GPU usage if available and configured.
    results = model(
        frame, 
        stream=True, 
        imgsz=320,  # <-- Optimization: Lower image resolution
        conf=0.6,   # <-- Optimization: Higher confidence threshold
        device=0    # <-- Optimization: Use GPU (if available)
    )

    # --- 5. Process Results and Draw Bounding Boxes ---
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # The 'conf' check is mostly handled in the 'model()' call now, 
            # but we keep the class check.

            # Get the confidence score and class label
            confidence = round(float(box.conf[0]), 2)
            cls = int(box.cls[0])
            class_name = model.names[cls] 

            # Only process if it's a bird or pigeon (already filtered by conf=0.6)
            if class_name in ['bird', 'pigeon']:
                # Get bounding box coordinates and convert to integers
                x1, y1, x2, y2 = map(int, box.xyxy[0]) 

                # Draw the bounding box and label on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f'{class_name}: {confidence}'
                cv2.putText(frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # --- 6. Display the Frame ---
    cv2.imshow('Pigeon Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 7. Cleanup ---
cap.release()
cv2.destroyAllWindows()