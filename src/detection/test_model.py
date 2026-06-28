from ultralytics import YOLO

model = YOLO("runs/detect/models/volleyvision_v1-2/weights/best.pt")

# descarga una imagen de prueba de internet o usa una tuya
results = model("IMG_7710.webp")  # detecta en la imagen

results[0].show()  # abre una ventana con las detecciones
results[0].save("output_test.jpg")  # o guárdala