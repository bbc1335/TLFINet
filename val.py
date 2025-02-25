from ultralytics import YOLO


if __name__ == '__main__':
    # Load a model
    model = YOLO('runs\\detect\\GC-DETTLFIS512B32E300\\weights\\best.pt')

    # Customize validation settings
    validation_results = model.val(data='ultralytics\\cfg\\datasets\\NEU-DET.yaml',
                                imgsz=224,
                                batch=32,
                                conf=0.001,
                                iou=0.5,
                                device='0')