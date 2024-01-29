import cv2
import numpy as np

# Load YOLO
yolov3_weights_path = "path/to/yolov3.weights"
yolov3_cfg_path = "path/to/yolov3.cfg"
net = cv2.dnn.readNet(yolov3_weights_path, yolov3_cfg_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Load classes
coco_names_path = "path/to/coco.names"
classes = []
with open(coco_names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize webcam or video source
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam, or specify a file path for a video.

while True:
    # Read and display the frame
    ret, frame = cap.read()
    if not ret:
        break

    # Detecting objects
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Information to display on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # You can adjust the confidence threshold
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and labels
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i], 2))
        color = colors[i]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label + " " + confidence, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the resulting frame
    cv2.imshow("Real-Time Object Detection", frame)

    # Exit the loop if the 'esc' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("esc"):
        break

# Release the video capture object
cap.release()

# Close all windows
cv2.destroyAllWindows()