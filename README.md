# Multi-modal Psychopathology Assessment for State-of-Mind Prediction

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-red)
![License](https://img.shields.io/badge/license-MIT-green)

## 📌 Project Overview
[cite_start]This project presents a Fusion Multi-modal ‘State-of-Mind’ Prediction application designed for real-time psychopathology assessment[cite: 8]. [cite_start]Unlike traditional single-modality systems, this application fuses computer vision and signal processing to interpret emotional states during live or recorded conversations[cite: 9]. [cite_start]By cross-referencing facial cues with involuntary vocal intonations (prosody), the system calculates emotional congruence, significantly reducing false positives (e.g., a smiling face paired with a stressed voice)[cite: 88]. 

[cite_start]This tool serves as an objective "Clinical Decision Support System" for therapists and psychiatrists, providing timeline-based insights into a patient's emotional volatility[cite: 95].

## ✨ Key Features
* [cite_start]**Dual-Channel Emotion Classification:** Analyzes facial expressions into 7 categories [cite: 102] and vocal tones into 8 categories simultaneously.
* **Advanced Speaker Diarization:** Utilizes K-Means clustering on acoustic features (MFCCs, spectral bandwidth/centroid) to separate speakers in the audio track (e.g., isolating the patient's voice from the clinician's).
* [cite_start]**Robust Data Augmentation:** The acoustic model utilizes noise injection, pitch shifting, and time-stretching to improve real-world generalization[cite: 59].
* **Algorithmic Decision-Level Fusion:** Actively compares visual and acoustic predictions to output "Fused" (congruent) or "Mixed" (incongruent) emotional states.
* [cite_start]**Interactive Multimodal Dashboard:** A Streamlit-based web app featuring a custom HTML/JS synchronized media player and interactive Plotly scatter plots mapping emotional trends over time[cite: 89].

## 🧠 System Architecture
[cite_start]The system operates two parallel deep learning pipelines[cite: 22]:
1. [cite_start]**Visual Model (FER Engine):** A custom 2D Convolutional Neural Network (CNN) [cite: 26] [cite_start]utilizing OpenCV Haar Cascades for face detection[cite: 27]. [cite_start]Trained on the **FER-2013** dataset.
2. **Audio Model (SER Engine):** An advanced hybrid 1D CNN-LSTM network that captures local acoustic features and long-term temporal dependencies from Mel-spectrograms. [cite_start]Trained on the **RAVDESS** dataset.

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/multi-modal-psychopathology-assessment.git](https://github.com/yourusername/multi-modal-psychopathology-assessment.git)
cd multi-modal-psychopathology-assessment