from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

model = YOLO("yolov8n.pt")  # descarga automáticamente los pesos base

results = model.train(
    data="data/annotations/volleyballyolo/data.yaml",
    epochs=30,
    imgsz=640,
    batch=8,          # baja a 4 si te da error de memoria
    name="volleyvision_v1",
    project="models",
    device="cpu"      # si tienes GPU nvidia cambia a device=0
)

print("✅ Entrenamiento completado")
print("Mejor modelo en: models/volleyvision_v1/weights/best.pt")