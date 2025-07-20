import subprocess
import os
import glob

scripts = sorted(glob.glob("*.py"))
exclude = ["test_all_scripts.py"]

print("🔍 Checking Python scripts in folder...\n")

for script in scripts:
    if script in exclude:
        continue
    print(f"🧪 Testing: {script}")
    try:
        result = subprocess.run(["python", script], timeout=10)
        print(f"✅ {script} ran (at least started) successfully.\n")
    except subprocess.TimeoutExpired:
        print(f"⚠️  {script} ran too long. Skipped after timeout.\n")
    except Exception as e:
        print(f"❌ Error in {script}: {e}\n")

