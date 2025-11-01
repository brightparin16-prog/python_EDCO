import cv2
import numpy as np
import os

# --- Configuration ---
# You need two files for the YOLO model:
# 1. Weights file (yolov3.weights)
# 2. Configuration file (yolov3.cfg)
# These files are typically downloaded separately. For this example to run,
# please ensure you have downloaded these files and placed them in the same directory
# as this script, or update the paths below.

# Placeholder paths (replace with actual downloaded paths if needed)
weights_path = "yolov3.weights"
config_path = "yolov3.cfg"
names_path = "coco.names" # Class names file

# Check if required files exist (mocking the check for the environment)
if not os.path.exists(weights_path) or not os.path.exists(config_path) or not os.path.exists(names_path):
    print("WARNING: YOLO files are not found.")
    print("To run this script, you must download the YOLO weights, config, and names files:")
    print("Weights: https://pjreddie.com/media/files/yolov3.weights")
    print("Config: https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg?raw=true")
    print("Names: https://github.com/pjreddie/darknet/blob/master/data/coco.names?raw=true")
    print("Please place these files in the same directory as this script.")
    # Exit gracefully if files are missing in a real environment
    # raise FileNotFoundError("Required YOLO files missing.")

# Load class names from the coco.names file
# If the file is missing, use a generic list
try:
    with open(names_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    print(f"Could not find {names_path}. Using generic class names.")
    classes = [f"Class {i}" for i in range(80)]

# Generate random colors for each class
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# Load the DNN model
try:
    net = cv2.dnn.readNet(weights_path, config_path)
except Exception as e:
    print(f"Error loading YOLO model. Check file paths and integrity. Details: {e}")
    # Placeholder for the network object to prevent further crashes if loading fails
    net = None

# --- Detection Parameters ---
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
SCALE_FACTOR = 0.00392 # Scaling factor for the image's pixel values
INPUT_SIZE = (416, 416) # YOLOv3 standard input size

# Get the names of the output layers
def get_output_layers(net):
    """Returns the names of the output layers of the network."""
    layer_names = net.getLayerNames()
    # Get the names of the output layers
    # In OpenCV 4+, this requires a check for the sequence
    try:
        unconnected_layers = net.getUnconnectedOutLayers()
        return [layer_names[i - 1] for i in unconnected_layers]
    except AttributeError:
        # Handling older OpenCV versions or specific error cases
        return [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    """Draws a bounding box around the detected object."""
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def process_frame(frame, net):
    """Performs object detection on a single frame."""
    if net is None:
        return frame # Return original frame if model failed to load

    (H, W) = frame.shape[:2]

    # Create a 4D blob from the frame for the network input
    blob = cv2.dnn.blobFromImage(frame, SCALE_FACTOR, INPUT_SIZE, (0, 0, 0), True, crop=False)
    net.setInput(blob)

    # Run forward pass to get output layer results
    outs = net.forward(get_output_layers(net))

    # Initialization
    class_ids = []
    confidences = []
    boxes = []

    # Process each detection output
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > CONFIDENCE_THRESHOLD:
                # Object detected
                center_x = int(detection[0] * W)
                center_y = int(detection[1] * H)
                width = int(detection[2] * W)
                height = int(detection[3] * H)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Apply Non-Max Suppression to suppress weak, overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    # Draw bounding boxes
    for i in indices:
        try:
            box_index = i[0]
        except TypeError:
            # Handle cases where i is a single integer (OpenCV 4.x)
            box_index = i

        box = boxes[box_index]
        left, top, width, height = box
        right = left + width
        bottom = top + height

        draw_bounding_box(frame, class_ids[box_index], confidences[box_index], left, top, right, bottom)

    return frame


def main():
    """Main function to start webcam capture and detection."""
    # Open the default webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream (webcam).")
        return

    print("--- Starting Real-Time Object Detection ---")
    print("Press 'q' or 'ESC' to exit the video stream.")

    while cv2.waitKey(1) < 0:
        # Read a new frame
        hasFrame, frame = cap.read()

        if not hasFrame:
            print("Finished processing video/live stream.")
            break

        # Process the frame
        processed_frame = process_frame(frame, net)

        # Display the resulting frame
        cv2.imshow("Real-Time Object Detection", processed_frame)

        # Break the loop if 'q' or 'ESC' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
