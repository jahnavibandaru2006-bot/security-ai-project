import cv2
import numpy as np
from datetime import datetime

print("🎥 Motion Detection System Starting...")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Read first frame as background
ret, background = cap.read()
background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
background = cv2.GaussianBlur(background, (21, 21), 0)

last_logged = None
print("✅ Motion detection started. Press Q to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Find difference between background and current frame
    diff = cv2.absdiff(background, gray)
    thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours (moving objects)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False

    for contour in contours:
        if cv2.contourArea(contour) < 2000:
            continue  # Ignore small movements

        motion_detected = True
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Motion Detected!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Log motion once every 3 seconds
    if motion_detected:
        now = datetime.now()
        if last_logged is None or (now - last_logged).seconds >= 3:
            last_logged = now
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
            print(f"🚨 Motion detected at {timestamp}")

    # Update background slowly
    background = cv2.addWeighted(background, 0.95, gray, 0.05, 0)

    cv2.imshow('Motion Detection - Security AI', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Motion detection stopped.")
