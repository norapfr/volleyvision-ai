from ultralytics import YOLO

model = YOLO("runs/detect/models/volleyvision_events_v1-2-3/weights/volleyvision_multiangle_best2.pt")

metrics = model.val(
    data="data/annotations/events/data.yaml",
    split="test",        # ← usa el test set, no el val
    conf=0.4,
    iou=0.5
)

print(f"\n📊 Resultados vs modelo anterior:")
print(f"{'Métrica':<12} {'Nuevo':>8} {'Anterior':>10}")
print(f"{'─'*32}")
print(f"{'mAP50':<12} {metrics.box.map50:>8.3f} {'0.972':>10}")
print(f"{'mAP50-95':<12} {metrics.box.map:>8.3f} {'0.889':>10}")
print(f"{'Precision':<12} {metrics.box.mp:>8.3f} {'0.969':>10}")
print(f"{'Recall':<12} {metrics.box.mr:>8.3f} {'0.977':>10}")