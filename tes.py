from ultralytics import YOLO
from PIL import Image

model = YOLO("exp-9.pt")

# Test dengan 1 gambar dulu
results = model("download.jpg", conf=0.25)  # conf rendah dulu

for r in results:
    print(f"Jumlah deteksi: {len(r.boxes)}")
    for box in r.boxes:
        print(f"Class: {model.names[int(box.cls)]}")
        print(f"Confidence: {float(box.conf):.2f}")
    
    # Simpan hasil visual
    r.save("hasil_deteksi.jpg")