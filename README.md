# Skin Disease Classifier — Android

An Android application for skin condition image classification using deep learning and TensorFlow Lite.

The project demonstrates the integration of a MobileNetV2-based image classification model into a mobile application using Google ML Kit and Jetpack Compose.

## Overview

The application allows users to select a skin image from their device and obtain a predicted category with its corresponding confidence score.

The model was trained using a skin condition image dataset and converted to TensorFlow Lite for on-device inference.

## Features

- Image selection from the device gallery
- On-device image classification
- Prediction results with confidence scores
- Simple interface developed with Jetpack Compose
- Offline inference using TensorFlow Lite and Google ML Kit

## Technologies

- **Language:** Kotlin
- **Development environment:** Android Studio
- **UI:** Jetpack Compose
- **Deep learning:** TensorFlow / Keras
- **Architecture:** MobileNetV2
- **Mobile inference:** TensorFlow Lite
- **Integration:** Google ML Kit
- **Training environment:** Google Colab

## Model

The classifier was trained using transfer learning with MobileNetV2.

- Input: 224 × 224 RGB images
- Output: Skin condition classification
- Model format: `.tflite`
- Deployment: Local inference on Android

The TensorFlow Lite model includes metadata and class labels for integration with ML Kit.

## How to Run

1. Clone or download this repository.
2. Open the project in Android Studio.
3. Sync the Gradle dependencies.
4. Verify that `model.tflite` is located in `app/src/main/assets/`.
5. Connect an Android device or start an emulator.
6. Build and run the application.
7. Select an image and press **Classify Image**.

## Intended Use

This application is an educational prototype demonstrating the deployment of deep learning models on mobile devices.

**It is not intended for clinical diagnosis, medical decision-making, or replacement of professional dermatological evaluation.**

## Acknowledgments

- Google TensorFlow Lite
- Google ML Kit
- MobileNetV2
- Skin Assessment image dataset

## Project Status

Educational prototype developed for exploring biomedical image classification on Android.
