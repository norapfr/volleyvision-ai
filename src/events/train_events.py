
from ultralytics import YOLO
from dotenv import load_dotenv
import os

load_dotenv()

RESUME = True  # ← cambia a True para continuar desde donde se dejó

LAST_CHECKPOINT = "run/detect/models/volleyvision_events_v1-2/weights/last.pt"

if RESUME and os.path.exists(LAST_CHECKPOINT):
    print("🔄 Reanudando entrenamiento desde último checkpoint...")
    model = YOLO(LAST_CHECKPOINT)
    results = model.train(resume=True)
else:
    model = YOLO("yolov8n.pt")  # nano: más rápido que small en CPU
    results = model.train(
        data="data/annotations/events/data.yaml",
        epochs=30,        # suficiente para convergir en detección de eventos de voleibol
        imgsz=416,        # reducido de 640 (~2.3x más rápido)
        batch=16,         # más batch = mejor uso de CPU
        workers=4,        # hilos de carga de datos
        cache=True,       # cachea imágenes en RAM
        name="volleyvision_events_v1-2",
        project="models",
        device="cpu",
        patience=8,       # early stopping: para si no mejora en 8 épocas
        dropout=0.2,
        cls=1.5,          # penaliza más los errores en clases minoritarias
    )

print("✅ Entrenamiento de eventos completado")
print("Modelo en: run/detect/models/volleyvision_events_v1-2/weights/best.pt")