from ultralytics import YOLO

# Load a model
model = YOLO('best.pt')  # pretrained TLFINet model

# Run batched inference on a list of images
img_paths = ['crazing_22.jpg', 
             'inclusion_93.jpg', 
             'patches_1.jpg',
             'pitted_surface_175.jpg', 
             'scratches_51.jpg', 
             'rolled-in_scale_78.jpg']

# Process results list
for i in range(len(img_paths)):
    model.predict(img_paths[i], save=True, conf=0.25, iou=0.5, imgsz=224, line_width=1)  # save to disk with default settings
