# 🔁 Plastic Waste Segregation Pipeline

## 🎥 1. Dataset Capture Script (Video to Images)
Captures video from webcam or `.mp4` and extracts every Nth frame:

```python
import cv2
import os

video_path = 'input.mp4'  # or use camera with 0
output_dir = 'captured_frames'
frame_interval = 15

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_count = 0
saved_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        filename = os.path.join(output_dir, f'frame_{saved_count:04d}.jpg')
        cv2.imwrite(filename, frame)
        saved_count += 1

    frame_count += 1

cap.release()
```

---

## 🎯 2. YOLOv8 Detection Script
Live object detection using your trained `best.pt`:

```python
from ultralytics import YOLO
import cv2

model = YOLO('best.pt')
cap = cv2.VideoCapture(1)  # or 0 based on camera index

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, show=True, conf=0.5)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## ✅ 3. Low Confidence Detection → Gemini Verification
This saves images with low confidence to `low_conf/`:

```python
from ultralytics import YOLO
import cv2
import os

model = YOLO('best.pt')
cap = cv2.VideoCapture(1)
output_dir = "low_conf"

os.makedirs(output_dir, exist_ok=True)

threshold = 0.5
counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, conf=threshold, save=False, show=True)

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < threshold:
                cv2.imwrite(f"{output_dir}/low_{counter}.jpg", frame)
                counter += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 🧠 4. Gemini Verification & Auto Dataset Update
Verifies low-confidence image → moves to dataset if valid:

```python
import requests
import shutil

def verify_image_with_gemini(img_path):
    # pseudo-code – your actual Gemini key/method
    response = requests.post("https://gemini-api/verify", files={"file": open(img_path, "rb")})
    result = response.json()
    return result['valid'], result['label']

src_dir = "low_conf"
dst_dir = "verified_dataset"

os.makedirs(dst_dir, exist_ok=True)

for filename in os.listdir(src_dir):
    img_path = os.path.join(src_dir, filename)
    valid, label = verify_image_with_gemini(img_path)

    if valid:
        shutil.move(img_path, os.path.join(dst_dir, label + "_" + filename))
    else:
        print(f"Rejected: {filename}")
```

---

## 💽 5. Cleaned requirements.txt

```
ultralytics==8.3.163
opencv-python
torch==2.7.1
torchvision
numpy
pillow
python-dotenv
```