from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import pytesseract
from typing import List

app = FastAPI()

# Thresholds for object detection
thres = 0.45  # Confidence threshold

# Load class names for object detection
classFile = "models/coco.names"
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').rsplit('\n')

# Load pre-trained model for object detection
configPath = 'models/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'models/frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

@app.post("/analyze-image")
async def analyze_image(
    image: UploadFile = File(...),
    mode: str = Form(...)
) -> JSONResponse:
    """
    Analyze the uploaded image based on the mode specified ("OD" for Object Detection, "OCR" for Text Recognition).
    """

    # Load the image from the uploaded file
    image_data = await image.read()
    np_img = np.fromstring(image_data, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if mode.upper() == "OD":
        # Perform Object Detection
        return detect_objects(img)
    elif mode.upper() == "OCR":
        # Perform OCR
        return perform_ocr(img)
    else:
        return JSONResponse(content={"error": "Invalid mode. Choose 'OD' or 'OCR'."}, status_code=400)

def detect_objects(img) -> JSONResponse:
    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    detected_objects = []

    for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
        obj_name = classNames[classId - 1]  # Get the class name
        detected_objects.append({"object": obj_name, "confidence": round(confidence * 100, 2)})

    return JSONResponse(content={"mode": "OD", "detected_objects": detected_objects})

def perform_ocr(img) -> JSONResponse:
    # Convert the image to grayscale for better OCR results
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return JSONResponse(content={"mode": "OCR", "text": text})

