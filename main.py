import face_recognition
import cv2
import numpy as np
import os
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from ultralytics import YOLO
from datetime import datetime

print("🔐 Security AI System Starting...")
print("=" * 50)

# ---- EMAIL SETTINGS ----
SENDER_EMAIL = "jahnavibandaru2006@gmail.com"
APP_PASSWORD = "ihzw zfpc phot aypk"
RECEIVER_EMAIL = "jahnavibandaru2006@gmail.com"

def send_email_alert(image_path, timestamp, reason):
    try:
        msg = MIMEMultipart()
        msg['Subject'] = f"⚠️ Security Alert - {reason}"
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECEIVER_EMAIL
        body = f"""
⚠️ SECURITY ALERT!
Reason: {reason}
Time: {timestamp}
Please check the attached photo.
        """
        msg.attach(MIMEText(body, 'plain'))
        with open(image_path, 'rb') as f:
            img = MIMEImage(f.read())
            img.add_header('Content-Disposition', 'attachment', filename='alert.jpg')
            msg.attach(img)
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as server:
            server.login(SENDER_EMAIL, APP_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        print(f"📧 Email sent: {reason}")
    except Exception as e:
        print(f"Email error: {e}")

# ---- MODULE 1: Load Known Faces ----
print("\n📦 Loading Module 1: Face Recognition...")
known_encodings = []
known_names = []
dataset_path = "dataset"
for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    if os.path.isdir(person_folder):
        for image_file in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_file)
            try:
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image, model="small")
                if encodings:
                    known_encodings.append(encodings[0])
                    known_names.append(person_name)
            except:
                pass
print(f"✅ Module 1 Ready: {len(known_names)} known faces loaded")

# ---- MODULE 3: Load YOLO Model ----
print("\n📦 Loading Module 3: YOLO Detection...")
yolo_model = YOLO('yolov8n.pt')
TARGET_OBJECTS = ['person', 'car', 'truck', 'motorcycle', 'bus']
print("✅ Module 3 Ready: YOLO model loaded")

# ---- MODULE 2: Motion Detection Setup ----
print("\n📦 Loading Module 2: Motion Detection...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
ret, background = cap.read()
background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
background_gray = cv2.GaussianBlur(background_gray, (21, 21), 0)
print("✅ Module 2 Ready: Motion detection initialized")

os.makedirs("reports/unknown_persons", exist_ok=True)

print("\n" + "=" * 50)
print("🚀 ALL MODULES ACTIVE! Press Q to quit.")
print("=" * 50 + "\n")

last_alerts = {}

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    display = frame.copy()
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

    # ---- MODULE 2: Motion Detection ----
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    diff = cv2.absdiff(background_gray, gray)
    thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    motion = any(cv2.contourArea(c) > 2000 for c in contours)
    if motion:
        cv2.putText(display, "🚨 Motion Detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    background_gray = cv2.addWeighted(background_gray, 0.95, gray, 0.05, 0)

    # ---- MODULE 3: YOLO Detection ----
    yolo_results = yolo_model(frame, verbose=False)
    for result in yolo_results:
        for box in result.boxes:
            label = result.names[int(box.cls)]
            if label not in TARGET_OBJECTS:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 165, 0), 2)
            cv2.putText(display, f"{label} {conf:.0%}",
                        (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

    # ---- MODULE 1: Face Recognition ----
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = np.ascontiguousarray(small_frame[:, :, ::-1])
    face_locations = face_recognition.face_locations(rgb_small)
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations, model="small")

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"
        color = (0, 0, 255)
        if known_encodings:
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match = np.argmin(face_distances)
            if matches[best_match]:
                name = known_names[best_match]
                color = (0, 255, 0)

        top *= 4; right *= 4; bottom *= 4; left *= 4
        cv2.rectangle(display, (left, top), (right, bottom), color, 2)
        cv2.rectangle(display, (left, bottom-35), (right, bottom), color, cv2.FILLED)
        cv2.putText(display, name, (left+6, bottom-6),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        # ---- MODULE 4: Security Automation ----
        if name == "Unknown":
            alert_key = "unknown_face"
            if alert_key not in last_alerts or (now - last_alerts[alert_key]).seconds >= 30:
                last_alerts[alert_key] = now
                print(f"⚠️  Unknown person detected at {timestamp}")
                photo_path = f"reports/unknown_persons/unknown_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(photo_path, frame)
                print(f"📸 Photo saved!")
                send_email_alert(photo_path, timestamp, "Unknown Person Detected")
        else:
            if name not in last_alerts or (now - last_alerts[name]).seconds >= 10:
                last_alerts[name] = now
                print(f"✅ {name} recognized at {timestamp}")

    # Status bar
    cv2.putText(display, f"Security AI - All Modules Active | {timestamp}",
                (10, display.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('🔐 Security AI System - All Modules', display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\n✅ Security AI System stopped.")