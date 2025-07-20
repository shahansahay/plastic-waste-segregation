
# ♻️ Plastic Waste Segregation System

A real-time plastic classification system using **YOLOv8** for object detection and **Gemini/GPT APIs** for label validation. Built to support automated data collection, verification, and continual model retraining for a plastic waste segregation conveyor system.

---

## 📸 System Overview

- **Hardware**: Logitech C920 webcam (mounted 70 cm above circular conveyor belt)
- **Model**: YOLOv8m (`best.pt`)
- **Objects Detected**: PET, HDPE, LDPE, PP, PS, CAN
- **Verification**: Gemini (multi-label) + fallback to GPT
- **Dataset**: ~4900 manually labeled images, augmented via Roboflow

---

## 🛠️ Features

- Real-time detection from live webcam
- Auto frame extraction from videos
- Auto crop and label save to `queue/`
- Gemini + GPT fallback label verification
- Verified dataset auto-sorted to `dataset/images/verified/`
- Ready for retraining & expansion

---

## 📂 Folder Structure

```
plastic-waste-segregation/
├── scripts/
│   ├── camera_check.py
│   ├── capture_dataset.py
│   ├── detect_live_cam.py
│   ├── detect_and_queue.py
│   ├── verify_with_gemini.py
│   ├── verify_single_label.py
│   ├── verify_confidence_fallback.py
│   ├── checker_valid.py
│   └── test_all_scripts.py
├── captured_videos/
├── VIDEOS/
├── queue/
├── dataset/
│   ├── images/verified/
│   └── labels/verified/
├── best.pt
├── PlasticSeg_Report_PFS.docx
├── PlasticWasteScripts_README.docx
└── .env
```

---

## 🚀 Setup Instructions

### 🔧 Step 1: Install Python & Git
- [Install Python 3.10+](https://www.python.org/downloads)
- [Install Git](https://git-scm.com/downloads)

### 🔧 Step 2: Clone & Setup Virtual Environment
```bash
git clone https://github.com/shahansahay/plastic-waste-segregation.git
cd plastic-waste-segregation
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 🔧 Step 3: Install Python Packages
```bash
pip install --upgrade pip
pip install ultralytics opencv-python torch rich requests
```

### 🔧 Step 4: Add API Keys
Create a `.env` file based on the `.env.example`:

```
GEMINI_API_KEY=your_gemini_api_key
GPT_API_KEY=your_gpt_api_key
```

---

## 📜 Script Descriptions

| Script                          | Purpose |
|----------------------------------|---------|
| `camera_check.py`              | Lists available webcam indices |
| `capture_dataset.py`          | Captures webcam videos to `captured_videos/` |
| `detect_live_cam.py`          | Real-time YOLOv8 detection from webcam |
| `detect_and_queue.py`         | Detects from videos, crops & saves to `queue/` |
| `verify_with_gemini.py`       | Verifies multi-labels via Gemini API |
| `verify_single_label.py`      | Verifies one label per image with Gemini |
| `verify_confidence_fallback.py` | Gemini → fallback to GPT if unsure |
| `checker_valid.py`            | Utility for testing logic |
| `test_all_scripts.py`         | Auto-runs all `.py` scripts for crash testing |

---

## 🔑 API Services Used

- [Gemini API](https://aistudio.google.com/)
- [OpenAI GPT-4o](https://platform.openai.com/)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)

---

## 🔄 Retraining Workflow

1. Run detection scripts on real belt input
2. Queue low-confidence images to `queue/`
3. Run Gemini/GPT verification
4. Verified images → `dataset/`
5. Retrain YOLO model via `ultralytics` CLI or script

---

## 🧪 Testing

Run all scripts to validate they work:
```bash
python scripts/test_all_scripts.py
```

---

## 📈 Future Improvements

- Add robotic arm/air jet sorting mechanism
- Brand classification via OCR (e.g. Bisleri)
- Deploy to Jetson Nano or Orin edge devices
- Streamlit dashboard for real-time monitoring
- Use Redis/SQL for database logging of detections

---

## 📄 License

This project is for research and educational use. License details can be added in a future release.

---

## 👨‍🔧 Maintainer

**Shahan Sahay**  
Intern @ Plastic Segregation AI Systems  
July 2025  
[shahansahay.com](https://shahansahay.com)
