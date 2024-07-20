import cv2
import numpy as np
import os
from cv2 import face


def load_patient_data(data_path):
    patient_images = []
    patient_labels = []
    patient_names = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

    for patient_name in os.listdir(data_path):
        patient_folder = os.path.join(data_path, patient_name)
        if os.path.isdir(patient_folder):
            for image_file in os.listdir(patient_folder):
                if image_file.lower().endswith(valid_extensions):
                    image_path = os.path.join(patient_folder, image_file)
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    if image is not None:
                        patient_images.append(image)
                        patient_labels.append(len(patient_names))
                    else:
                        print(f"Failed to load image: {image_path}")
            if patient_images:  # Only add patient name if images were loaded
                patient_names.append(patient_name)

    print(f"Loaded {len(patient_images)} images")
    print(f"Loaded {len(patient_labels)} labels")
    print(f"Loaded {len(patient_names)} patient names")

    return patient_images, patient_labels, patient_names



face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_recognizer = face.LBPHFaceRecognizer_create()

data_path = './patient_data'
patient_images, patient_labels, patient_names = load_patient_data(data_path)

if len(patient_images) == 0 or len(patient_labels) == 0:
    print("No data loaded. Please check your data_path and image files.")
    exit()


patient_labels = np.array(patient_labels)

face_recognizer.train(patient_images, np.array(patient_labels))

print("Face detector:", face_detector)
print("Face recognizer:", face_recognizer)

def recognize_patients():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                label, confidence = face_recognizer.predict(face_roi)

                if confidence < 100:
                    patient_name = patient_names[label]
                    cv2.putText(frame, f"{patient_name} ({confidence:.2f})", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Unknown", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            cv2.imshow('Patient Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

recognize_patients()