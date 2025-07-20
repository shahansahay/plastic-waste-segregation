import base64
import json
import shutil
import requests
from pathlib import Path
from rich import print

GEMINI_API_KEY = "YOUR_GEMINI_KEY"
GPT_API_KEY = "YOUR_OPENAI_KEY"

queue_dir = Path("queue")
verified_img_dir = Path("dataset/images/verified")
annotated_img_dir = Path("dataset/images/annotated")
verified_lbl_dir = Path("dataset/labels/verified")
for d in [verified_img_dir, annotated_img_dir, verified_lbl_dir]:
    d.mkdir(parents=True, exist_ok=True)

LABEL_MAP = {
    "0": "PET",
    "1": "HDPE",
    "2": "LDPE",
    "3": "PP",
    "4": "CAN"
}

def verify_with_gemini(image_b64, label_list):
    prompt = f"The image contains plastic waste labeled as: {', '.join(label_list)}. Are all these labels correct? Reply only 'Yes' or 'No'."
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inlineData": {
                    "mimeType": "image/jpeg",
                    "data": image_b64
                }}
            ]
        }]
    }
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": GEMINI_API_KEY
    }
    try:
        res = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
            headers=headers,
            json=payload,
            timeout=10
        )
        text = res.json()['candidates'][0]['content']['parts'][0]['text'].strip().lower()
        return "yes" in text
    except Exception as e:
        print(f"[red]⚠ Gemini failed: {e}[/red]")
        return None

def verify_with_gpt(image_b64, label_list):
    headers = {
        "Authorization": f"Bearer {GPT_API_KEY}",
        "Content-Type": "application/json"
    }
    prompt = f"The image contains plastic waste labeled as: {', '.join(label_list)}. Are ALL these labels correct? Respond with only 'Yes' or 'No'."
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    }
                ]
            }
        ]
    }
    try:
        res = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=15)
        text = res.json()['choices'][0]['message']['content'].strip().lower()
        return "yes" in text
    except Exception as e:
        print(f"[red]⚠ GPT fallback failed: {e}[/red]")
        return False

for img_path in queue_dir.glob("*.jpg"):
    if "_annotated" in img_path.name:
        continue

    txt_path = queue_dir / f"{img_path.stem}.txt"
    ann_path = queue_dir / f"{img_path.stem}_annotated.jpg"

    if not txt_path.exists():
        print(f"[yellow]⚠️ No label for {img_path.name}[/yellow]")
        continue

    with open(txt_path) as f:
        lines = f.readlines()
    labels = set()
    for line in lines:
        parts = line.strip().split()
        if not parts: continue
        class_id = parts[0]
        if class_id in LABEL_MAP:
            labels.add(LABEL_MAP[class_id])
    if not labels:
        print(f"[red]❌ No known labels for {img_path.name}[/red]")
        continue

    with open(img_path, "rb") as img_file:
        image_b64 = base64.b64encode(img_file.read()).decode("utf-8")

    print(f"\n🧠 Verifying {img_path.name} with labels: [bold]{', '.join(labels)}[/bold]")

    result = verify_with_gemini(image_b64, labels)
    if result is None:
        print("[yellow]🔁 Falling back to GPT...[/yellow]")
        result = verify_with_gpt(image_b64, labels)

    if result:
        shutil.copy(img_path, verified_img_dir / img_path.name)
        shutil.copy(txt_path, verified_lbl_dir / txt_path.name)
        if ann_path.exists():
            shutil.copy(ann_path, annotated_img_dir / ann_path.name)
        print(f"[green]✅ Accepted by AI and moved: {img_path.name}[/green]")
    else:
        print(f"[red]❌ Rejected: {img_path.name}[/red]")
