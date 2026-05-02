#  Driver Drowsiness Detection System

A real-time driver monitoring system that detects fatigue based on eye behavior and triggers a **beep alert** to prevent potential accidents.

---

## 🔍 Overview

This project uses computer vision techniques to monitor eye movement through a webcam feed. It calculates the **Eye Aspect Ratio (EAR)** to determine whether the driver's eyes are closed for a prolonged duration.

If drowsiness is detected:

*  "DROWSY!" is displayed on screen
*  A continuous beep sound is triggered until the driver becomes alert

---

## ⚙️ Features

* Real-time webcam-based monitoring
* Eye Aspect Ratio (EAR) based detection
* Facial landmark tracking using MediaPipe Face Mesh
* Continuous beep alert system
* Visual feedback on screen
* Low latency processing

---

##  Tech Stack

* Python
* OpenCV
* MediaPipe Face Mesh
* NumPy
* SciPy
* Pygame

---

##  How It Works

1. Capture video using webcam
2. Detect face and extract facial landmarks
3. Identify eye landmark points
4. Compute Eye Aspect Ratio (EAR)
5. Compare EAR with threshold
6. If EAR < threshold for consecutive frames:

   * Mark as drowsy
   * Trigger alarm sound

---

##  How to Run

```bash
pip install opencv-python mediapipe numpy scipy pygame
python main.py
```

---

##  Project Structure

```bash
project/
│
├── main.py
├── utils.py
├── alarm.wav
├── README.md
```

---

##  Use Cases

* Driver safety systems
* Smart vehicle monitoring
* Fatigue detection systems
* Real-time alert systems

---

##  Team

* Akshiti 
* Himanshu
* Subir
* Dipanshu

---

##  Future Improvements

* Yawn detection
* Head pose detection
* Attention score system
* Web dashboard (Streamlit)
* Mobile integration

---

## 📄 License

This project is open-source and available for learning and experimentation.
