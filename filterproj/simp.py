import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from .seqmodel import processimage
import json

app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Image Filter API is running!"}

@app.post("/filter")
async def apply_filter(
    file: UploadFile = File(...),
    values: str = Form(...),

    cartoon_name: str = Form(""),
    background_name: str = Form(""),

    cartoon_file: UploadFile | None = File(None),
    background_file: UploadFile | None = File(None),
):
    """
    Receives:
    - main input image
    - operations JSON
    - optional preset names
    - optional uploaded cartoon / background images
    """

    # ---------- Parse operations ----------
    try:
        operations = json.loads(values)
    except Exception as e:
        return {"error": f"Invalid JSON in 'values': {str(e)}"}

    # ---------- Read main image ----------
    image_bytes = await file.read()

    # ---------- Read optional uploaded images ----------
    cartoon_bytes = None
    background_bytes = None

    if cartoon_file:
        cartoon_bytes = await cartoon_file.read()

    if background_file:
        background_bytes = await background_file.read()

    # ---------- Process ----------
    processed_bytes = processimage(
        image_bytes=image_bytes,
        operations=operations,
        cartoon_name=cartoon_name,
        background_name=background_name,
        cartoon_bytes=cartoon_bytes,
        background_bytes=background_bytes
    )

    return Response(content=processed_bytes, media_type="image/jpeg")
