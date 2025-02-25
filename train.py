from ultralytics import YOLO

# Load a model
model = YOLO("ultralytics/cfg/models/v8/TLFINet.yaml")  # build a new model from scratch

# Use the model
model.train(data="NEU-DET.yaml",
            epochs=300,
            batch=32,
            imgsz=224,
            patience=100,
            workers=8,
            close_mosaic=15)
