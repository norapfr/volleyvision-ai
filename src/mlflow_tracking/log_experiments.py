import mlflow
import pandas as pd
from pathlib import Path

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("VolleyVision")

# ── Modelo 1: Detección — desde CSV real ──────────────────────────
results_path = Path("runs/detect/models/volleyvision_v1-2/results.csv")

with mlflow.start_run(run_name="detection_yolov8n_volleyballyolo"):
    mlflow.log_params({
        "model":       "yolov8n",
        "dataset":     "volleyballyolo + volleyball-tracker",
        "epochs":      30,
        "imgsz":       640,
        "batch":       8,
        "device":      "CPU Intel i7-1165G7",
        "classes":     2,
        "class_names": "ball, person",
        "optimizer":   "AdamW",
        "dropout":     0.0,
    })

    if results_path.exists():
        df = pd.read_csv(results_path)
        df.columns = df.columns.str.strip()
        print(f"Columnas disponibles: {list(df.columns)}")

        for _, row in df.iterrows():
            step = int(row["epoch"])
            mlflow.log_metrics({
                "train_box_loss": float(row["train/box_loss"]),
                "train_cls_loss": float(row["train/cls_loss"]),
                "train_dfl_loss": float(row["train/dfl_loss"]),
                "mAP50":          float(row["metrics/mAP50(B)"]),
                "mAP50_95":       float(row["metrics/mAP50-95(B)"]),
                "precision":      float(row["metrics/precision(B)"]),
                "recall":         float(row["metrics/recall(B)"]),
            }, step=step)

        print(f"Metricas de deteccion cargadas: {len(df)} epochs")
    else:
        print(f"No se encontro: {results_path}")

    mlflow.set_tags({
        "phase":    "detection",
        "task":     "object_detection",
        "sport":    "volleyball",
        "hardware": "CPU Intel i7-1165G7",
        "dataset":  "Roboflow Universe"
    })

    print("Experimento de deteccion registrado")


# ── Modelo 2: Eventos — métricas reales de logs ───────────────────
with mlflow.start_run(run_name="events_yolov8n_graz_colab"):
    mlflow.log_params({
        "model":         "yolov8n",
        "dataset":       "volleyball-activity-dataset-graz-v3",
        "dataset_size":  25000,
        "epochs":        33,
        "imgsz":         416,
        "batch":         16,
        "device":        "GPU Tesla T4 (Google Colab)",
        "classes":       7,
        "class_names":   "Defense-Move, attack, block, reception, service, setting, stand",
        "dropout":       0.2,
        "cls_weight":    1.5,
        "amp":           False,
        "optimizer":     "AdamW",
        "training":      "resumed from CPU checkpoint epoch 12"
    })

    # métricas reales del entrenamiento en Colab
    # fuente: logs del entrenamiento (epochs 1-20 en Colab + epochs 1-12 en CPU)
    mlflow.log_metrics({
        "mAP50":               0.991,
        "mAP50_95":            0.898,
        "precision":           0.966,
        "recall":              0.976,
        "mAP50_attack":        0.992,
        "mAP50_block":         0.992,
        "mAP50_reception":     0.993,
        "mAP50_service":       0.993,
        "mAP50_setting":       0.986,
        "mAP50_Defense_Move":  0.984,
        "mAP50_stand":         0.990,
        "test_mAP50":          0.971,
        "test_mAP50_95":       0.881,
        "test_precision":      0.972,
        "test_recall":         0.968,
    })

    # curva de aprendizaje parcial — epochs conocidos de los logs
    epochs_conocidos = [
        {"epoch": 3,  "mAP50": 0.746, "mAP50_95": 0.587, "precision": 0.675, "recall": 0.734},
        {"epoch": 5,  "mAP50": 0.851, "mAP50_95": 0.701, "precision": 0.943, "recall": 0.389},
        {"epoch": 6,  "mAP50": 0.888, "mAP50_95": 0.741, "precision": 0.646, "recall": 0.646},
        {"epoch": 7,  "mAP50": 0.888, "mAP50_95": 0.741, "precision": 0.817, "recall": 0.830},
        {"epoch": 9,  "mAP50": 0.916, "mAP50_95": 0.773, "precision": 0.842, "recall": 0.646},
        {"epoch": 12, "mAP50": 0.954, "mAP50_95": 0.823, "precision": 0.907, "recall": 0.893},
        {"epoch": 20, "mAP50": 0.975, "mAP50_95": 0.861, "precision": 0.935, "recall": 0.941},
        {"epoch": 33, "mAP50": 0.991, "mAP50_95": 0.898, "precision": 0.966, "recall": 0.976},
    ]

    for e in epochs_conocidos:
        mlflow.log_metrics({
            "mAP50":     e["mAP50"],
            "mAP50_95":  e["mAP50_95"],
            "precision": e["precision"],
            "recall":    e["recall"],
        }, step=e["epoch"])

    mlflow.set_tags({
        "phase":    "event_classification",
        "task":     "object_detection",
        "sport":    "volleyball",
        "hardware": "GPU Tesla T4 (Google Colab)",
        "dataset":  "Graz University of Technology"
    })

    print("Experimento de eventos registrado")


print("\nTodos los experimentos registrados")
print("Lanza la UI con: mlflow ui --host 127.0.0.1")