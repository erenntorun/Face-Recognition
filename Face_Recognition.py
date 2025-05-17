import cv2
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QInputDialog, QMessageBox
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
import sys
import csv

CONFIDENCE_THRESHOLD = 40  # stricter threshold value

# Get username, create folder, and collect face data
def collect_user_data():
    username, ok = QInputDialog.getText(None, "New User", "Enter a new username (letters/numbers only):")
    if not ok or not username:
        print("Username could not be retrieved.")
        return False

    username = ''.join(c for c in username if c.isalnum())
    user_dir = f"./faces/{username}"
    os.makedirs(user_dir, exist_ok=True)

    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def face_extractor(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            return None
        for (x, y, w, h) in faces:
            return img[y:y+h, x:x+w]

    cap = cv2.VideoCapture(0)
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera frame could not be read.")
            break

        face_img = face_extractor(frame)
        if face_img is not None:
            count += 1
            face = cv2.resize(face_img, (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            file_name_path = f"{user_dir}/{count}.jpg"
            success = cv2.imwrite(file_name_path, face)
            if success:
                print(f"[✓] Saved: {file_name_path}")
            else:
                print(f"[✗] FAILED TO SAVE: {file_name_path}")

            cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Face Capture', face)
        else:
            print("No face detected. Are you looking at the camera?")

        if cv2.waitKey(1) == 13 or count == 100:
            break

    cap.release()
    cv2.destroyAllWindows()

    if count == 0:
        print("No images were saved.")
        return False

    print("100 face images successfully saved.")
    return True

# Scan user directories
def load_training_data(base_path='./faces/'):
    Training_Data, Labels = [], []
    label_map = {}
    label_id = 0

    for user_folder in os.listdir(base_path):
        user_path = join(base_path, user_folder)
        if not os.path.isdir(user_path):
            continue

        onlyfiles = [f for f in listdir(user_path) if isfile(join(user_path, f))]

        for i, file in enumerate(onlyfiles):
            img_path = join(user_path, file)
            images = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if images is None:
                continue
            Training_Data.append(np.asarray(images, dtype=np.uint8))
            Labels.append(label_id)

        label_map[label_id] = user_folder
        label_id += 1

    return Training_Data, Labels, label_map

# Train and save the model
def train_and_save_model():
    data, labels, label_map = load_training_data()
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(np.asarray(data), np.asarray(labels))
    model.save('trained_model.xml')

    with open('label_map.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for label_id, user in label_map.items():
            writer.writerow([label_id, user])

    print("Model trained and saved.")

# Load label map
def load_label_map():
    label_map = {}
    if os.path.exists('label_map.csv'):
        with open('label_map.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                label_map[int(row[0])] = row[1]
    return label_map

# Log recognition results
def log_result(username, success, confidence):
    with open('login_log.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([username, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Success' if success else 'Fail', confidence])

# PyQt5 interface
class FaceRecognitionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Unlock System")
        self.image_label = QLabel()
        self.status_label = QLabel("Status: Waiting...")

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.status_label)
        self.setLayout(layout)

        self.face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.model = cv2.face.LBPHFaceRecognizer_create()
        self.model.read('trained_model.xml')
        self.label_map = load_label_map()

        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_classifier.detectMultiScale(gray, 1.3, 5)
        label_text = "Unknown user - Locked"
        color = (0, 0, 255)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (200, 200))

            label, confidence = self.model.predict(roi_gray)
            conf_percent = int(100 * (1 - confidence / 400))

            if label in self.label_map and confidence < CONFIDENCE_THRESHOLD:
                user = self.label_map[label]
                label_text = f"{user} - Unlocked ({conf_percent}%)"
                color = (0, 255, 0)
                log_result(user, True, conf_percent)
            else:
                label_text = "Unknown user - Locked"
                color = (0, 0, 255)
                log_result("Unknown", False, conf_percent)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            break

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_image))
        self.status_label.setText(f"Status: {label_text}")

    def closeEvent(self, event):
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    app = QApplication(sys.argv)

    mode, ok = QInputDialog.getItem(None, "Select Action", "Do you want to register a new user?", ["Yes (register new user)", "No (enter system)"]) 

    if ok and mode.startswith("Yes"):
        if collect_user_data():
            train_and_save_model()
    elif ok and mode.startswith("No"):
        train_and_save_model()

    window = FaceRecognitionApp()
    window.show()
    sys.exit(app.exec_())
