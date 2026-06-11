import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime

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
                    print(f"Loaded: {person_name} - {image_file}")
            except Exception as e:
                print(f"Skipping {image_file}: {e}")

print(f"\n✅ Loaded {len(known_names)} known faces")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("📷 Camera started. Press Q to quit.\n")

last_logged = {}  # Track last log time per name

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
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        # Only log once every 5 seconds per person
        now = datetime.now()
        if name not in last_logged or (now - last_logged[name]).seconds >= 5:
            last_logged[name] = now
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
            if name == "Unknown":
                print(f"⚠️  Unknown person detected at {timestamp}")
            else:
                print(f"✅ {name} recognized at {timestamp}")

    cv2.imshow('Face Recognition - Security AI', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Camera stopped.")
