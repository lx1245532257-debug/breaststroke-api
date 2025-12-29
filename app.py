from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import shutil
import os
import cv2
import numpy as np
import mediapipe as mp

app = FastAPI()

# 获取当前文件所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 必须使用 static 目录，Bubble 才能访问到处理后的视频
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# 蛙泳腿标准角度
IDEAL_KNEE = {"recovery": 45, "outsweep": 120, "propulsion": 170}

def calc_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

def classify_phase(knee):
    if knee < 70: return "recovery"
    elif knee < 140: return "outsweep"
    else: return "propulsion"

def get_feedback(phase_scores):
    feedback_list = []
    for phase, score in phase_scores.items():
        if score < 85:
            if phase == "recovery": feedback_list.append("收腿阶段膝盖折叠不够。")
            if phase == "propulsion": feedback_list.append("蹬腿要完全蹬直并拢。")
    return " ".join(feedback_list) if feedback_list else "动作非常标准！"

@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    raw_path = os.path.join(STATIC_DIR, file.filename)
    processed_filename = f"out_{file.filename}"
    processed_path = os.path.join(STATIC_DIR, processed_filename)
    
    with open(raw_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False)
    cap = cv2.VideoCapture(raw_path)
    
    # 设置视频写入
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(processed_path, fourcc, fps, (w, h))

    data = {"recovery": {"knee": []}, "outsweep": {"knee": []}, "propulsion": {"knee": []}}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            # 获取右腿关键点
            hip = [lm[mp_pose.PoseLandmark.RIGHT_HIP].x, lm[mp_pose.PoseLandmark.RIGHT_HIP].y]
            knee = [lm[mp_pose.PoseLandmark.RIGHT_KNEE].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE].y]
            ankle = [lm[mp_pose.PoseLandmark.RIGHT_ANKLE].x, lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y]

            angle = calc_angle(hip, knee, ankle)
            phase = classify_phase(angle)
            data[phase]["knee"].append(angle)

            # 在视频上画线
            p1 = (int(hip[0]*w), int(hip[1]*h))
            p2 = (int(knee[0]*w), int(knee[1]*h))
            p3 = (int(ankle[0]*w), int(ankle[1]*h))
            cv2.line(frame, p1, p2, (0, 255, 0), 5)
            cv2.line(frame, p2, p3, (0, 255, 0), 5)
            cv2.putText(frame, str(int(angle)), p2, cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)

        out.write(frame)

    cap.release()
    out.release()

    phase_scores = {p: round(max(0, 100 - abs(sum(d["knee"])/len(d["knee"]) - IDEAL_KNEE[p])), 2) 
                    for p, d in data.items() if d["knee"]}
    
    total_score = round(sum(phase_scores.values()) / len(phase_scores), 2) if phase_scores else 0

    # 动态生成 Render 上的视频地址
    render_url = f"https://{os.environ.get('RENDER_EXTERNAL_HOSTNAME', 'localhost:8000')}/static/{processed_filename}"

    return {
        "total_score": total_score,
        "text_feedback": get_feedback(phase_scores),
        "processed_video_url": render_url
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))