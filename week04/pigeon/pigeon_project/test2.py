import cv2
from ultralytics import YOLO # Library used in the project for YOLOv8

# --- 1. Load the Model ---
# This loads the trained YOLO model (replace with the actual path to your .pt file)
model = YOLO('yolov8n.pt') 
# You need to confirm the model is trained to recognize 'pigeon' or 'bird'

# --- 2. Initialize Webcam Capture ---
# '0' typically refers to the default webcam.
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# --- 3. Main Loop for Real-Time Processing ---
while True:
    # Read a single frame from the camera
    ret, frame = cap.read() 
    
    if not ret:
        break # Exit if the frame is not captured successfully

    # --- 4. Perform Detection (YOLOv8) ---
    # Run the model on the current frame. 'stream=True' is for efficiency.
    # The 'results' will contain the location and class of detected objects.
    results = model(frame, stream=True)

    # --- 5. Process Results and Draw Bounding Boxes ---
    for r in results:
        boxes = r.boxes # Get the bounding boxes and detection data
        for box in boxes:
            # Get bounding box coordinates and convert to integers
            x1, y1, x2, y2 = map(int, box.xyxy[0]) 
            
            # Get the confidence score and class label
            confidence = round(float(box.conf[0]), 2)
            cls = int(box.cls[0])
            class_name = model.names[cls] # Get human-readable name

            # Only process if it's a bird or pigeon (based on your model)
            if class_name in ['bird', 'pigeon'] and confidence > 0.5:
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