import yaml
from pathlib import Path
from collections import Counter

# lee el data.yaml para ver las clases
with open("data/annotations/events/data.yaml") as f:
    config = yaml.safe_load(f)

print("Clases del dataset:")
for i, name in enumerate(config['names']):
    print(f"  {i}: {name}")

print(f"\nTotal clases: {config['nc']}")

# cuenta imágenes por split
for split in ['train', 'valid', 'test']:
    split_path = Path(f"data/annotations/events/{split}/images")
    if split_path.exists():
        count = len(list(split_path.glob("*.jpg")))
        print(f"{split}: {count} imágenes")

# cuenta instancias por clase en train
labels_path = Path("data/annotations/events/train/labels")
class_counts = Counter()

for label_file in labels_path.glob("*.txt"):
    with open(label_file) as f:
        for line in f:
            class_id = int(line.split()[0])
            class_counts[class_id] += 1

print("\nInstancias por clase en train:")
for class_id, count in sorted(class_counts.items()):
    print(f"  {config['names'][class_id]}: {count}")