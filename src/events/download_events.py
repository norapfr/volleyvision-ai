import os
from roboflow import Roboflow
from dotenv import load_dotenv

load_dotenv()

rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])

print("Descargando dataset de eventos...")
rf.workspace("activity-graz-uni").project("volleyball-activity-dataset").version(3).download(
    "yolov8", 
    location="data/annotations/events"
)

print("✅ Dataset de eventos descargado")