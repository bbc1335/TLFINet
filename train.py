from ultralytics import YOLO

# Load a model
model = YOLO("ultralytics/cfg/models/v8/TLFINet-Ablation-Focus.yaml")  # build a new model from scratch
# model = YOLO("yolov8m.yaml")  # build a new model from scratch
# model = YOLO("/home/bbc1335/Documents/Detection/ultralytics/runs/detect/YOLOv8改3/weights/best.pt")  # Load a pretrained model (recommended for training)

# Use the model
model.train(data="NEU-DET.yaml",
            epochs=300,
            batch=32,
            imgsz=224,
            patience=100,
            workers=8,
            # optimizer="SGD",
            # conf=0.001,
            # iou=0.5,
            name="AblationTLFINetFocuse300sz224",
            close_mosaic=15)
# metrics = model.val()  # evaluate model performance on the validation set
# results = model("")  # predict on an image
# path = model.export(format="onnx")  # export the model to ONNX format
