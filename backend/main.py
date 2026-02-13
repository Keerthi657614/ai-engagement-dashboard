from fastapi import FastAPI, WebSocket
import base64
import numpy as np
import cv2
import json

from engagement import analyze_frame

app = FastAPI()

@app.get("/")
def home():
    return {"message":"Backend running"}

# ... (imports) ...

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            img_bytes = base64.b64decode(data.split(",")[1])
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            state, engagement, coords = analyze_frame(frame)

            await websocket.send_text(json.dumps({
                "state": state,
                "engagement": engagement,
                "coords": coords
            }))
    except Exception as e:
        print(f"Connection closed: {e}")