from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import random
import base64
from io import BytesIO
import uvicorn

app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/process-image/")
async def process_image(file: UploadFile = File(...)):
    data = await file.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        return {"image": "", "isPassed": False, "isContinue": False}

    # Texto centrado "PRUEBA"
    h, w = img.shape[:2]
    text = "PRUEBA"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(1.0, w / 800.0)
    thickness = max(2, int(2 * scale))
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (w - tw) // 2
    y = (h + th) // 2
    cv2.putText(img, text, (x, y), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # Codificar imagen en base64
    ext = "." + (file.content_type.split("/")[1] if "/" in file.content_type else "jpg")
    success, buf = cv2.imencode(ext, img)
    if not success:
        return {"image": "", "isPassed": False, "isContinue": False}

    img_base64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    # TODO: Integrate this kind of return into main API
    return {
        "image": img_base64,
        "isPassed": bool(random.getrandbits(1)),
        "isContinue": bool(random.getrandbits(1))
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8141, workers=1, log_level="info")

