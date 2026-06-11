from ultralytics import YOLO
import cv2
from datetime import datetime

print("🎥 YOLO Human & Vehicle Detection Starting...")

# Load YOLO model (downloads automatically first time)
model = YOLO('yolov8n.pt')

# Objects we want to detect
TARGET_OBJECTS = ['person', 'car', 'truck', 'motorcycle', 'bus']

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

last_logged = {}
print("✅ YOLO Detection started. Press Q to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Run YOLO detection
    results = model(frame, verbose=False)

    for result in results:
        for box in result.boxes:
            label = result.names[int(box.cls)]

            if label not in TARGET_OBJECTS:
                continue

            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])

            # Choose color based on object
            if label == 'person':
                color = (0, 255, 0)   # Green for person
            else:
                color = (255, 165, 0) # Orange for vehicles

            # Draw box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {confidence:.0%}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)

            # Log once every 5 seconds per object type
            now = datetime.now()
            if label not in last_logged or (now - last_logged[label]).seconds >= 5:
                last_logged[label] = now
                timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
                print(f"✅ {label} detected ({confidence:.0%} confidence) at {timestamp}")

    cv2.imshow('YOLO Detection - Security AI', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("YOLO Detection stopped.")
