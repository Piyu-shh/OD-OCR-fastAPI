from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import cv2

app = FastAPI()

# Threshold for object detection
thres = 0.45  # Confidence threshold

# Load class names for object detection
classFile = "models/coco.names"
try:
    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')
except FileNotFoundError:
    raise RuntimeError("Error: 'coco.names' file not found. Ensure it is placed in the 'models' directory.")

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
    raise RuntimeError(f"Error loading model: {e}. Ensure model files are placed in the 'models' directory.")


@app.post("/analyze-image")
async def analyze_image(
    image: UploadFile = File(...),
    mode: str = Form(...),
) -> JSONResponse:
    """
    Analyze the uploaded image based on the mode specified ("OD" for Object Detection).
    """
    try:
        # Load the image from the uploaded file
        image_data = await image.read()
        np_img = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

        if mode.upper() == "OD":
            # Perform Object Detection
            return detect_objects(img)
        else:
            raise HTTPException(status_code=400, detail="Invalid mode. Choose 'OD'.")
    except HTTPException as he:
        # Re-raise HTTP exceptions to be handled by FastAPI
        raise he
    except Exception as e:
        return JSONResponse(content={"error": "Internal Server Error", "details": str(e)}, status_code=500)


def detect_objects(img) -> JSONResponse:
    """
    Perform object detection on the input image and return unique detected object names.
    """
    try:
        # Perform object detection
        classIds, confs, bbox = net.detect(img, confThreshold=thres)

        # Check if no objects were detected
        if classIds is None or len(classIds) == 0:
            return JSONResponse(content={"mode": "OD", "detected_objects": [], "message": "No objects detected."})

        detected_objects = set()  # Use a set to ensure unique object names

        # Process detections
        for classId in classIds.flatten():
            # Safeguard against classId out of range
            obj_name = classNames[classId - 1] if classId - 1 < len(classNames) else "Unknown"
            detected_objects.add(obj_name)

        return JSONResponse(content={"mode": "OD", "detected_objects": list(detected_objects)})
    except Exception as e:
        return JSONResponse(content={"error": "Object detection failed", "details": str(e)}, status_code=500)



def perform_ocr(img) -> JSONResponse:
    """
    Perform OCR on the input image.
    """
    try:
        # Convert the image to grayscale for better OCR results
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray)
        print(f"OCR extracted text: {text}")
        return JSONResponse(content={"mode": "OCR", "text": text})
    except Exception as e:
        print(f"Error during OCR: {e}")
        return JSONResponse(content={"error": "OCR failed", "details": str(e)}, status_code=500)
