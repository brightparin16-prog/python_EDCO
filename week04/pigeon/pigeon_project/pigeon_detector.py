import cv2
import numpy as np

# --- Configuration ---
# WEBCAM_INDEX 0 usually refers to the default camera. Change if you have multiple cameras.
WEBCAM_INDEX = 0 
DETECTION_LABEL = "Pigeon Body (Simulated)"

def detect_pigeon_body_webcam():
    """
    Initializes the webcam, reads frames continuously, simulates the detection 
    of a pigeon's body using a mock bounding box in the center, and draws the result.

    In a real-world application, a trained machine learning model would be applied 
    to each frame here.
    """
    # Initialize video capture
    cap = cv2.VideoCapture(WEBCAM_INDEX)

    if not cap.isOpened():
        print(f"ERROR: Cannot open webcam with index {WEBCAM_INDEX}. Check if the camera is connected and available.")
        return
    
    window_name = 'Pigeon Detection Result (Webcam)'
    
    print("Webcam started. Press 'q' to quit.")

    try:
        while True:
            # 1. Read frame from webcam
            ret, frame = cap.read()
            
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # Get frame dimensions
            height, width, _ = frame.shape
            
            # Check if the frame is large enough for processing
            if height < 100 or width < 100:
                print("Frame too small. Stopping processing.")
                break

            # 2. Simulate the Object Detection Result
            # We simulate a bounding box covering the center 50% of the frame.
            
            x_min = int(width * 0.25)
            y_min = int(height * 0.25)
            x_max = int(width * 0.75)
            y_max = int(height * 0.75)
            
            mock_detections = [
                {'box': [x_min, y_min, x_max, y_max], 'label': DETECTION_LABEL, 'confidence': 0.95}
            ]

            # 3. Draw Bounding Boxes and Labels
            for detection in mock_detections:
                # Extract box coordinates (x_min, y_min, x_max, y_max)
                x_min, y_min, x_max, y_max = detection['box']
                label = detection['label']
                confidence = detection['confidence']
                
                # Define colors and thickness
                color = (0, 255, 0)  # Green BBox
                thickness = 2
                font_scale = 0.7
                
                # Draw the rectangle (Bounding Box)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, thickness)
                
                # Prepare the label text
                text = f"{label}: {confidence:.2f}"
                
                # Get text size for background box
                (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                
                # Calculate text position (place it just above the top edge of the box)
                text_y = y_min - 10 # 10 pixels above the box
                
                # Ensure the text doesn't go off the top of the screen
                if text_y < text_h + 5: 
                    text_y = y_min + text_h + 5 # If it does, place it just inside the box
                    
                
                # Draw background box for text
                # We use (x_min, text_y - text_h) as the top-left corner
                cv2.rectangle(frame, (x_min, text_y - text_h), (x_min + text_w, text_y + baseline), color, -1)
                
                # Put the text on the image
                # The text baseline is placed at text_y
                cv2.putText(frame, text, (x_min, text_y + baseline), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)


            # 4. Display the result
            cv2.imshow(window_name, frame)
            
            # Check for 'q' key press to quit (waitKey(1) is necessary for continuous video)
            if cv2.waitKey(1) == ord('q'):
                break
                
    except Exception as e:
        print(f"An error occurred during processing: {e}")
        
    finally:
        # 5. Release the capture and destroy all windows
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam closed and windows destroyed.")

if __name__ == "__main__":
    # --- Instructions for the user ---
    # This script now uses your default webcam (index 0).
    # 1. Ensure you have a webcam connected.
    # 2. Run the script and press 'q' to quit the live feed.
    # 3. The green box will appear in the center, simulating pigeon detection.
    # 4. To use real detection, you would replace the 'mock_detections' logic with
    #    a call to a trained model (like a TensorFlow, PyTorch, or Ultralytics YOLO model).
    
    detect_pigeon_body_webcam()
