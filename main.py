from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
import numpy as np
import cv2
from io import BytesIO

model = YOLO("yolov8n.pt").to("cuda")
app = FastAPI()

@app.post("/process-image/")
async def process_image(file: UploadFile = File(...)):
	# Read the uploaded file's bytes
	contents = await file.read()

	# Convert bytes to a NumPy array and decode it as an image
	image_array = np.frombuffer(contents, np.uint8)
	image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

	# Process image using YOLO
	results = model(image, verbose=False)

	# Draw detections on image
	annotated_frame = results[0].plot()

	# Get image dimensions as a sample output
	height, width, _ = annotated_frame.shape

	# Encode image back as original format
	original_format = '.' + file.content_type.split('/')[1]
	_, encoded_image = cv2.imencode(original_format, annotated_frame)

	# Return result image as a BytesIO buffer
	return StreamingResponse(
		BytesIO(encoded_image.tobytes()), 
		media_type=file.content_type
	)
