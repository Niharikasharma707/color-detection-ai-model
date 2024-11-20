import torch
import cv2

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load and preprocess image
image_path = "C:/Users/Signity_Laptop/Pictures/fashion/one.jpg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform inference
results = model(image_rgb)

# Extract bounding boxes, labels, and confidence scores
detected_objects = results.xyxy[0].cpu().numpy()
for x1, y1, x2, y2, conf, cls in detected_objects:
    # Draw bounding box
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    # Add label
    label = f'{model.names[int(cls)]} {conf:.2f}'
    cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

# Show image with detections
cv2.imshow('Object Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
