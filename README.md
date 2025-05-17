# Face Recognition Unlock System

A camera-based desktop application for real-time face recognition and authentication. Built with Python, OpenCV, and PyQt5, this system allows user registration, model training, and secure login using only facial biometrics.

---

## 🚀 Features

* Real-time face detection and recognition
* Multi-user support with per-user data folders
* Secure login with confidence threshold
* PyQt5 graphical interface with camera feed
* Automatic model retraining on startup
* Access logging to `login_log.csv`

---

## 📁 Project Structure

```
face_unlock_system/
├── faces/                   # User-specific face image folders
├── haarcascade_frontalface_default.xml  # Haar cascade for face detection
├── trained_model.xml        # Saved LBPH model
├── label_map.csv            # User ID ↔ name mapping
├── login_log.csv            # Login attempt logs
├── main.py                  # Main application code
├── README.md                # Project description
└── requirements.txt         # Python dependencies
```

---

## 🛠️ Requirements

* Python 3.8+
* opencv-contrib-python
* PyQt5
* numpy

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🖥️ How to Use

1. Clone this repository:

```bash
git clone https://github.com/erenntorun/Face-Eecognition-Unlock.git
cd Face-Eecognition-Unlock
```

2. Ensure your webcam is connected.
3. Run the application:

```bash
python main.py
```

4. Choose "Yes" to register a new user or "No" to use face recognition.

---

## 📌 Notes

* System saves 100 grayscale face images per user.
* Model uses LBPH algorithm for fast and efficient recognition.
* Confidence threshold (default: 40) prevents misrecognition.

---

## 🧠 Future Improvements

* PIN/password fallback authentication
* FaceNet / Dlib embedding upgrade
* Admin panel for user management
* GUI-based user deletion & visualization

---

## 🙋‍♂️ Author

Developed by \[Eren Torun], 2025
