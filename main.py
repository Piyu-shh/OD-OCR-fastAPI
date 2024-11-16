from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import pytesseract

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
    Analyze the uploaded image based on the mode specified ("OD" for Object Detection, "OCR" for Text Recognition).
    """
    try:
        # Load the image from the uploaded file
        image_data = await image.read()
        np_img = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

        print(f"Received mode: {mode}")

        if mode.upper() == "OD":
            # Perform Object Detection
            print("Performing Object Detection...")
            return detect_objects(img)
        elif mode.upper() == "OCR":
            # Perform OCR
            print("Performing OCR...")
            return perform_ocr(img)
        else:
            raise HTTPException(status_code=400, detail="Invalid mode. Choose 'OD' or 'OCR'.")
    except HTTPException as he:
        # Re-raise HTTP exceptions to be handled by FastAPI
        raise he
    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse(content={"error": "Internal Server Error", "details": str(e)}, status_code=500)

def detect_objects(img) -> JSONResponse:
    """
    Perform object detection on the input image.
    """
    try:
        # Perform object detection
        classIds, confs, bbox = net.detect(img, confThreshold=thres)

        # Check if no objects were detected
        if classIds is None or len(classIds) == 0:
            return JSONResponse(content={"mode": "OD", "detected_objects": [], "message": "No objects detected."})

        detected_objects = []

        # Ensure classIds and confs are iterable
        if isinstance(classIds, (list, np.ndarray)):
            # Handle both single and multiple detections
            if classIds.ndim == 1:
                # Single detection, wrap in a list
                classIds = [classIds]
                confs = [confs]
                bbox = [bbox]
            for idx in range(len(classIds)):
                classId = classIds[idx][0] if isinstance(classIds[idx], (list, tuple, np.ndarray)) else classIds[idx]
                confidence = confs[idx][0] if isinstance(confs[idx], (list, tuple, np.ndarray)) else confs[idx]
                box = bbox[idx].tolist() if isinstance(bbox[idx], (list, tuple, np.ndarray)) else bbox[idx]

                # Safeguard against classId out of range
                if classId - 1 < len(classNames):
                    obj_name = classNames[classId - 1]
                else:
                    obj_name = "Unknown"

                detected_objects.append({
                    "object": obj_name,
                    "confidence": round(float(confidence) * 100, 2),
                    "bounding_box": box
                })

                print(f"Detected {obj_name} with confidence {round(float(confidence) * 100, 2)}% at {box}")
        else:
            raise ValueError("Invalid format for detection results.")

        return JSONResponse(content={"mode": "OD", "detected_objects": detected_objects})
    except Exception as e:
        print(f"Error during object detection: {e}")
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
