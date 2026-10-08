# 🖐️ Hand Gesture Controller

A touchless laptop control system that uses real-time hand tracking to control PDF navigation with hand movements.

## 🚀 Features

- Real-time hand tracking using webcam
- Hand movement detection
- Move hand up → Previous PDF page
- Move hand down → Next PDF page
- Works with Chrome PDF viewer
- Touchless computer interaction

## 🛠️ Tech Stack

- Python
- OpenCV
- MediaPipe
- PyAutoGUI

## ⚙️ How It Works

```text
Webcam
   ↓
OpenCV
   ↓
MediaPipe Hand Landmarker
   ↓
Hand Landmark Detection
   ↓
Movement Detection
   ↓
PyAutoGUI
   ↓
Chrome PDF