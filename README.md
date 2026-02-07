# Hand-Gesture-Drone-Control
Real-time hand gesture recognition system for drone/robot control using MediaPipe &amp; ANN.

# Drone-Gesture-Control

Real-time hand gesture recognition system for drone/robot control using **MediaPipe** and **Artificial Neural Network (ANN)**.  

This project allows controlling a drone or robot using hand gestures captured via a webcam. Gestures are recognized in real-time, and corresponding actions like forward, backward, rotate, slow/fast speed, up/down, and emergency stop are performed.

---

## Features

- Real-time hand tracking using **MediaPipe**
- Gesture dataset collection and labeling
- ANN model training for gesture recognition
- Real-time gesture-based drone/robot control
- Drone features include:  
  - Forward / Backward  
  - Left / Right  
  - Up / Down  
  - Slow / Fast speed  
  - Rotate left / right  
  - Stop / Emergency stop  
- Live action overlay on camera window  

---

## Dataset

- CSV format dataset containing **hand landmarks (x, y)** and gesture labels
- Balanced dataset for all gestures to ensure accurate predictions

---

## Requirements

- Python 3.11  
- Libraries: `opencv-python`, `mediapipe`, `pandas`, `scikit-learn`, `joblib`

Install with:

```bash
pip install opencv-python mediapipe pandas scikit-learn joblib

