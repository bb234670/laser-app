import cv2
import numpy as np
import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, HTMLResponse

app = FastAPI()
@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <html>
        <head>
            <title>Laser Hit Analyzer</title>
            <meta name="viewport" content="width=device-width, initial-scale=1">
        </head>
        <body style="font-family: sans-serif; padding: 20px;">
            <h2>Upload Laser Video</h2>
            <form action="/analyze" enctype="multipart/form-data" method="post">
                <label><strong>1. Select Video:</strong></label><br>
                <input name="file" type="file" accept="video/mp4" style="margin-top: 8px; margin-bottom: 20px;"><br>
                
                <label><strong>2. Target Coordinates (src_pts):</strong></label><br>
                <input name="pts" type="text" value="[[100,100], [400,100], [100,400], [400,400]]" style="width: 100%; margin-top: 8px; margin-bottom: 20px; padding: 8px;"><br>
                
                <input type="submit" value="Analyze Video" style="padding: 10px 20px; background-color: #007bff; color: white; border: none; border-radius: 5px;">
            </form>
        </body>
    </html>
    """

def detect_red(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 120, 200])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([170, 120, 200])
    upper2 = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower1, upper1) + cv2.inRange(hsv, lower2, upper2)
    return mask

def cluster_hits(points, radius=20):
    clusters = []
    for p in points:
        if not any(np.linalg.norm(np.array(p)-np.array(c)) < radius for c in clusters):
            clusters.append(p)
    return clusters

def process_video(path, src_pts):
    cap = cv2.VideoCapture(path)
    hits = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 2 != 0:
            continue

        mask = detect_red(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            area = cv2.contourArea(c)
            if 5 < area < 200:
                x, y, w, h = cv2.boundingRect(c)
                cx, cy = x + w//2, y + h//2
                hits.append([cx, cy])

    cap.release()
    hits = cluster_hits(hits)

    dst_pts = np.float32([[0,0],[500,0],[0,500],[500,500]])
    M = cv2.getPerspectiveTransform(np.float32(src_pts), dst_pts)

    mapped_hits = []
    for h in hits:
        pt = np.array([[h]], dtype="float32")
        mapped = cv2.perspectiveTransform(pt, M)
        mapped_hits.append((int(mapped[0][0][0]), int(mapped[0][0][1])))

    return mapped_hits

def draw_map(hits):
    img = np.ones((500,500,3), dtype=np.uint8) * 255
    for (x,y) in hits:
        cv2.circle(img, (x,y), 6, (0,0,255), -1)
    cv2.imwrite("output.png", img)

@app.post("/analyze", response_class=HTMLResponse)
async def analyze(file: UploadFile = File(...), pts: str = Form(...)):
    # 1. Save the video
    contents = await file.read()
    with open("input.mp4", "wb") as f:
        f.write(contents)

    # 2. Process the video
    src_pts = np.array(eval(pts), dtype="float32")
    hits = process_video("input.mp4", src_pts)
    
    # 3. Draw and save the output.png map
    draw_map(hits)

    # 4. Calculate score
    score = f"{min(len(hits),40)}/40"

    # 5. Return a visual webpage showing the result
    return f"""
    <html>
        <head>
            <meta name="viewport" content="width=device-width, initial-scale=1">
        </head>
        <body style="font-family: sans-serif; padding: 20px; text-align: center;">
            <h2>Analysis Complete!</h2>
            <h3>Score: {score}</h3>
            <img src="/download" style="max-width: 100%; border: 2px solid black; margin-top: 10px;">
            <br><br>
            <a href="/" style="padding: 10px 20px; background-color: #007bff; color: white; text-decoration: none; border-radius: 5px;">Analyze Another Video</a>
        </body>
    </html>
    """
@app.get("/download")
def download():
    return FileResponse("output.png")
