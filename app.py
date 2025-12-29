from fastapi import FastAPI, File, UploadFile
import shutil
import os
import cv2
import numpy as np

# ✅ 关键修复：不要用 mp.solutions
from mediapipe.python.solutions import pose as mp_pose

app = FastAPI()

# ======================
# 基础配置
# ======================
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

IDEAL_KNEE = {
    "recovery": 45,
    "outsweep": 120,
    "propulsion": 170
}

# ======================
# 工具函数
# ======================
def calc_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

def classify_phase(knee):
    if knee < 70:
        return "recovery"
    elif knee < 140:
        return "outsweep"
    else:
        return "propulsion"

def score(angle, ideal):
    return max(0, 100 - abs(angle - ideal))

# ======================
# API 接口
# ======================
@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    video_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    pose = mp_pose.Pose(static_image_mode=False)
    cap = cv2.VideoCapture(video_path)

    data = {
        "recovery": {"knee": [], "frames": 0},
        "outsweep": {"knee": [], "frames": 0},
        "propulsion": {"knee": [], "frames": 0}
    }

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark

            hip = [lm[mp_pose.PoseLandmark.RIGHT_HIP].x,
                   lm[mp_pose.PoseLandmark.RIGHT_HIP].y]
            knee = [lm[mp_pose.PoseLandmark.RIGHT_KNEE].x,
                    lm[mp_pose.PoseLandmark.RIGHT_KNEE].y]
            ankle = [lm[mp_pose.PoseLandmark.RIGHT_ANKLE].x,
                     lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y]

            knee_angle = calc_angle(hip, knee, ankle)
            phase = classify_phase(knee_angle)

            data[phase]["knee"].append(knee_angle)
            data[phase]["frames"] += 1

    cap.release()

    phase_scores = {}
    total_scores = []

    for phase, d in data.items():
        if d["frames"] == 0:
            continue
        avg_knee = sum(d["knee"]) / len(d["knee"])
        s = score(avg_knee, IDEAL_KNEE[phase])
        phase_scores[phase] = round(s, 2)
        total_scores.append(s)

    result = {
        "total_score": round(sum(total_scores) / len(total_scores), 2) if total_scores else 0,
        "phase_scores": phase_scores
    }

    return result

# ======================
# Render / 本地 启动入口
# ======================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))
    )