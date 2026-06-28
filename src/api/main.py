from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile, json, shutil, uuid, cv2
from pathlib import Path
from ultralytics import YOLO

app = FastAPI(
    title="VolleyVision API",
    description="API REST para análisis táctico automático de voleibol",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# carga modelos al arrancar
detection_model = YOLO("runs/detect/models/volleyvision_v1-2/weights/best.pt")
events_model    = YOLO("runs/detect/models/volleyvision_events_v1-2-3/weights/volleyvision_events_best_final.pt")

CONF_POR_CLASE = {
    "service":      0.75,
    "block":        0.45,
    "attack":       0.45,
    "reception":    0.45,
    "setting":      0.45,
    "Defense-Move": 0.50,
}

jobs = {}  # en producción usarías Redis


def analyze_video(job_id: str, video_path: str, every_n: int = 10):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    detections_log = []
    events_log     = []
    last_event     = None
    last_frame     = -15
    prev_gray      = None
    frame_id       = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % every_n == 0:
            h, w = frame.shape[:2]

            det = detection_model(frame, conf=0.4, verbose=False)
            for box in det[0].boxes:
                cls = detection_model.names[int(box.cls)]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections_log.append({
                    "frame": frame_id,
                    "time":  round(frame_id / fps, 2),
                    "class": cls,
                    "conf":  round(float(box.conf), 3),
                    "cx":    round((x1 + x2) / 2 / w, 3),
                    "cy":    round((y1 + y2) / 2 / h, 3),
                })

            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            motion = cv2.absdiff(prev_gray, curr_gray).mean() if prev_gray is not None else 0.0
            prev_gray = curr_gray

            if motion > 2.0:
                evt = events_model(frame, conf=0.25, verbose=False)
                for box in evt[0].boxes:
                    cls      = events_model.names[int(box.cls)]
                    conf_val = float(box.conf)
                    umbral   = CONF_POR_CLASE.get(cls, 0.45)
                    if (cls != "stand" and conf_val >= umbral and
                        (cls != last_event or frame_id - last_frame > 15)):
                        events_log.append({
                            "frame": frame_id,
                            "time":  round(frame_id / fps, 2),
                            "event": cls,
                            "conf":  round(conf_val, 3),
                        })
                        last_event = cls
                        last_frame = frame_id

        frame_id += 1

    cap.release()

    # resumen
    event_counts = {}
    for e in events_log:
        event_counts[e["event"]] = event_counts.get(e["event"], 0) + 1

    jobs[job_id] = {
        "status":       "done",
        "frames":       frame_id // every_n,
        "detections":   len(detections_log),
        "events":       len(events_log),
        "event_counts": event_counts,
        "events_log":   events_log,
    }


@app.get("/")
def root():
    return {
        "name":    "VolleyVision API",
        "version": "1.0.0",
        "status":  "running"
    }


@app.get("/health")
def health():
    return {
        "status":        "ok",
        "models_loaded": True,
        "detection":     "volleyvision_v1",
        "events":        "volleyvision_events_final"
    }


@app.post("/analyze")
async def analyze(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    every_n: int = 10
):
    job_id = str(uuid.uuid4())

    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    shutil.copyfileobj(file.file, tfile)
    tfile.flush()

    jobs[job_id] = {"status": "processing"}
    background_tasks.add_task(analyze_video, job_id, tfile.name, every_n)

    return {"job_id": job_id, "status": "processing"}


@app.get("/results/{job_id}")
def get_results(job_id: str):
    if job_id not in jobs:
        return JSONResponse(status_code=404, content={"error": "job no encontrado"})
    return jobs[job_id]


@app.get("/jobs")
def list_jobs():
    return {
        job_id: {"status": data["status"]}
        for job_id, data in jobs.items()
    }