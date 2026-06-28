import os
from roboflow import Roboflow
from dotenv import load_dotenv

load_dotenv()
rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])

print("Descargando dataset 1/4: Volleyball Tracker...")
rf.workspace("volleyball-analyser").project("volleyball-tracker").version(1).download("yolov8", location="data/annotations/tracker")

print("Descargando dataset 2/4: VolleyBallYolo...")
rf.workspace("volleyballyolo").project("volleyballyolo").version(1).download("yolov8", location="data/annotations/volleyballyolo")

print("Descargando dataset 3/4: Court Keypoints...")
rf.workspace("primaryws").project("volleyball_court_key_points_regression_dataset").version(7).download("yolov8", location="data/annotations/court")

print("Descargando dataset 4/4: Ball Detection...")
rf.workspace("primaryws").project("volleyball_ball_object_detection_dataset").version(1).download("yolov8", location="data/annotations/ball")

print("✅ Todos los datasets descargados")