import os
import tempfile
import shutil
import logging
from pathlib import Path
from io import BytesIO
from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image
import torch
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("object-detection", model="valentinafeve/yolos-fashionpedia")
# Configure the logger
logging.basicConfig(filename="app.log", level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")

# Create a FastAPI application
app = FastAPI(debug=True, title='Fashion AI', summary='This API Provides Access to Object Detection in Fashion Images')

# Load model and processor
processor = AutoImageProcessor.from_pretrained("valentinafeve/yolos-fashionpedia")
model = AutoModelForObjectDetection.from_pretrained("valentinafeve/yolos-fashionpedia")
# Load model directly

# Endpoint to upload an image for object detection
@app.post("/detect_features/")
async def detect_features(file: UploadFile):
    try:
        # Create a temporary directory
        temp_dir = Path("temp")
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Save the uploaded image
        image_path = temp_dir / file.filename
        with open(image_path, "wb") as image_file:
            shutil.copyfileobj(file.file, image_file)

        # Load image
        image = Image.open(image_path).convert("RGB")  # Ensure image is in RGB format

        # Process the image and perform object detection
        inputs = processor(images=image, return_tensors="pt", size={"longest_edge": 512})  # Updated here
        outputs = model(**inputs)

        # Process outputs
        results = []
        for score, label, box in zip(outputs.logits.softmax(-1)[0, :-1], outputs.logits[0, :-1], outputs.pred_boxes[0]):
            if score.item() > 0.5:  # Threshold to filter weak detections
                results.append({
                    "label": model.config.id2label[label.item()],
                    "score": score.item(),
                    "box": box.detach().numpy().tolist()  # Convert tensor to list
                })

        # Cleanup temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)

        return JSONResponse(content={"detected_features": results})

    except Exception as e:
        logging.error("An error occurred: %s", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Run the application with: uvicorn app:app --reload

