from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from io import BytesIO
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from datetime import datetime
import numpy as np
import cv2
import os
from components.models import detect_objects, generate_caption, draw_detections, translate_caption

# Dossier pour sauvegarder les résultats
SAVE_DIR = "static/results"
os.makedirs(SAVE_DIR, exist_ok=True)

app = FastAPI()

# Montre le dossier static
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/process")
async def process(file: UploadFile):
    try:
        image = Image.open(BytesIO(await file.read()))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    # Détection et génération
    objects = detect_objects(image)
    caption = generate_caption(image)
    translated_caption = translate_caption(caption)
    
    # Sauvegarder les résultats
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    annotated_image_path = os.path.join(SAVE_DIR, f"{timestamp}_result.jpg")
    frame_with_detections = draw_detections(np.array(image), objects)
    cv2.imwrite(annotated_image_path, frame_with_detections)
    with open(os.path.join(SAVE_DIR, f"{timestamp}_result.txt"), "w") as f:
        f.write(translated_caption)
    
    return {"objects": objects, "caption": translated_caption}

@app.get("/results/{filename}")
async def get_result_file(filename: str):
    file_path = os.path.join(SAVE_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)
