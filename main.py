from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import pytesseract
from typing import List

app = FastAPI()

# Threshold for object detection
thres = 0.45  # Confidence threshold

# Load class names for object detection
classFile = "models/coco.names"
try:
    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')
except FileNotFoundError:
    print("Error: 'coco.names' file not found. Ensure it is placed in the 'models' directory.")

# Load pre-trained model for object detection
configPath = 'models/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'models/frozen_inference_graph.pb'
try:
    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)
except cv2.error as e:
    print(f"Error loading model: {e}. Ensure model files are placed in the 'models' directory.")

@app.post("/analyze-image")
async def analyze_image(
    image: UploadFile = File(...),
    mode: str = Form(...)
) -> JSONResponse:
    """
    Analyze the uploaded image based on the mode specified ("OD" for Object Detection, "OCR" for Text Recognition).
    """
    try:
        # Load the image from the uploaded file
        image_data = await image.read()
        np_img = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image.")
        
        print(f"Received mode: {mode}")
        
        if mode.upper() == "OD":
            # Perform Object Detection
            print("Performing Object Detection")
            return detect_objects(img)
        elif mode.upper() == "OCR":
            # Perform OCR
            print("Performing OCR")
            return perform_ocr(img)
        else:
            raise ValueError("Invalid mode. Choose 'OD' or 'OCR'.")
    except ValueError as e:
        print(f"ValueError: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=400)
    except Exception as e:
        print(f"Exception: {e}")
        return JSONResponse(content={"error": "Internal Server Error"}, status_code=500)

def detect_objects(img) -> JSONResponse:
    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    detected_objects = []

    if classIds is not None:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            obj_name = classNames[classId - 1]  # Get the class name
            detected_objects.append({"object": obj_name, "confidence": round(confidence * 100, 2)})
            print(f"Detected {obj_name} with confidence {round(confidence * 100, 2)}%")

    return JSONResponse(content={"mode": "OD", "detected_objects": detected_objects})

def perform_ocr(img) -> JSONResponse:
    # Convert the image to grayscale for better OCR results
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    print(f"OCR extracted text: {text}")
    return JSONResponse(content={"mode": "OCR", "text": text})
