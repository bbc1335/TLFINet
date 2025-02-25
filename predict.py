from ultralytics import YOLO

# Load a model
# model = YOLO('runs\\detect\\终改1P45Depth1SZ224\\weights\\best.pt')  # pretrained YOLOv8n model
model = YOLO('runs\\detect\\终改1P45Depth1SZ224\\weights\\best.pt')  # pretrained YOLOv8n model

# Run batched inference on a list of images
img_paths = ['H:\\ultralytics\\save\\NEU_DETTestImg\\crazing_22.jpg', 
             'H:\\ultralytics\\save\\NEU_DETTestImg\\inclusion_93.jpg', 
             'H:\\ultralytics\\save\\NEU_DETTestImg\\patches_1.jpg',
             'H:\\ultralytics\\save\\NEU_DETTestImg\\pitted_surface_175.jpg', 
             'H:\\ultralytics\\save\\NEU_DETTestImg\\scratches_51.jpg', 
             'H:\\ultralytics\\save\\NEU_DETTestImg\\rolled-in_scale_78.jpg']

# results = model(['crazing_278.jpg', 'inclusion_143.jpg', 'patches_1.jpg', 'pitted_surface_175.jpg', 'rolled-in_scale_100.jpg', 'scratches_94.jpg'])  # return a list of Results objects

# Process results list
# for i, result in enumerate(results):
    # boxes = result.boxes  # Boxes object for bounding box outputs
    # masks = result.masks  # Masks object for segmentation masks outputs
    # keypoints = result.keypoints  # Keypoints object for pose outputs
    # probs = result.probs  # Probs object for classification outputs
    # result.show()  # display to screen
    # result.save(filename=f'TLFI{i}.jpg')  # save to disk
for i in range(len(img_paths)):
    model.predict(img_paths[i], save=True, conf=0.25, iou=0.5, imgsz=224, line_width=1)  # save to disk with default settings