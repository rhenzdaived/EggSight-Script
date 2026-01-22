# EggSight – IoT-Based Rotten Egg Sorting Script

## Overview

This repository contains the main Python script (eggsorter.py) used for the EggSight IoT-Based Rotten Egg Sorting Device. The script runs on a Raspberry Pi and integrates computer vision, a lightweight TensorFlow Lite CNN model (MobileNetV2-based), hardware control, and Firebase Firestore logging to automatically detect, classify, and sort eggs as fresh or rotten.

The system operates in real time using an IR sensor for egg detection, a Pi Camera for image acquisition, a DC motor–driven conveyor belt, a servo motor–based sorting gate, and a cloud-connected dashboard for monitoring daily egg statistics.

## Key Features
- Real-time egg detection using an IR sensor

- Image capture via Raspberry Pi Camera (Picamera2)

- Lightweight CNN inference using TensorFlow Lite

- Automatic sorting using a servo motor gate

- Conveyor belt control using a VNH2SP30 motor driver

- Cloud logging of egg data and daily summaries via Firebase Firestore

- Timezone-aware daily summaries (Asia/Manila)

- Live preview window with classification overlay
