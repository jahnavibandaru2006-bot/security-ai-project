import face_recognition
import cv2
import numpy as np
import os
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from datetime import datetime

SENDER_EMAIL = "jahnavibandaru2006@gmail.com"
APP_PASSWORD = "ihzw zfpc phot aypk"
RECEIVER_EMAIL = "jahnavibandaru2006@gmail.com"

def send_email_alert(image_path, timestamp):
    try:
        msg = MIMEMultipart()
        msg['Subject'] = f"⚠️ Security Alert - Unknown Person Detected!"
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECEIVER_EMAIL
        body = f"""
⚠️ SECURITY ALERT!
An unknown person was detected.
Time: {timestamp}
Please check the attached photo.
        """
        msg.attach(MIMEText(body, 'plain'))
        with open(image_path, 'rb') as f:
            img = MIMEImage(f.read())
            img.add_header('Content-Disposition', 'attachment', filename='unknown_person.jpg')
            msg.attach(img)
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as server:
            server.login(SENDER_EMAIL, APP_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        print(f"📧 Email alert sent at {timestamp}")
    except Exception as e:
        print(f"Email error: {e}")

known_encodings = []
known_names = []

dataset_path = "../dataset"
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

print(f"✅ Loaded {len(known_names)} known faces")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

last_alert = None
os.makedirs("../reports/unknown_persons", exist_ok=True)
print("📷 Security Automation started. Press Q to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

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
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom-35), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, name, (left+6, bottom-6),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        if name == "Unknown":
            now = datetime.now()
            if last_alert is None or (now - last_alert).seconds >= 30:
                last_alert = now
                timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
                print(f"⚠️ Unknown person detected at {timestamp}")
                photo_path = f"../reports/unknown_persons/unknown_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(photo_path, frame)
                print(f"📸 Photo saved: {photo_path}")
                send_email_alert(photo_path, timestamp)

    cv2.imshow('Security Automation - Security AI', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Security Automation stopped.")