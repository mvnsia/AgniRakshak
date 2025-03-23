import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(1)  # Change to 0 if using internal webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run fire detection on the frame
    results = model(frame)

    # Draw the detections on the frame
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            cls = int(box.cls[0])

            if conf > 0.5:  # If confidence is above threshold
                label = f"Fire: {conf:.2f}"
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("Fire Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
