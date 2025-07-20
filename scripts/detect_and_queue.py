from ultralytics import YOLO
import cv2, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "best.pt"
VIDEO_DIR = ROOT / "VIDEOS"
QUEUE_DIR = ROOT / "queue"
QUEUE_DIR.mkdir(exist_ok=True)

videos = sorted(VIDEO_DIR.glob("*.mp4"))
print("\nAvailable videos:")
for idx, v in enumerate(videos):
    print(f"[{idx}] {v.name}")
video_idx = int(input("\nEnter video index: ").strip())
video_path = str(videos[video_idx])

model = YOLO(str(MODEL_PATH)).to("mps")  # Use Metal GPU on Mac

cap = cv2.VideoCapture(video_path)
print("\nRunning detection... Press ESC to quit early.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    ts = int(time.time() * 1000)
    results = model.predict(frame, conf=0.1, verbose=False)[0]

    if len(results.boxes):
        img_path = QUEUE_DIR / f"{ts}.jpg"
        txt_path = QUEUE_DIR / f"{ts}.txt"
        annotated = results.plot()

        cv2.imwrite(str(img_path), frame)
        with open(txt_path, "w") as f:
            for box in results.boxes:
                cls, conf, xywh = int(box.cls[0]), float(box.conf[0]), box.xywh[0]
                x, y, w, h = xywh.tolist()
                f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

        ann_path = QUEUE_DIR / f"{ts}_annotated.jpg"
        cv2.imwrite(str(ann_path), annotated)

    cv2.imshow("Live Detection", results.plot())
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

