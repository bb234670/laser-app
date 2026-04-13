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
            <style>
                body { font-family: sans-serif; padding: 20px; max-width: 600px; margin: auto; }
                #videoFrame { max-width: 100%; border: 2px solid black; margin-top: 10px; display: none; touch-action: manipulation; }
                .btn { padding: 10px 20px; background-color: #007bff; color: white; border: none; border-radius: 5px; font-size: 16px; margin-top: 15px;}
                .btn:disabled { background-color: #cccccc; }
                .instructions { color: #555; font-size: 14px; margin-bottom: 5px; }
            </style>
        </head>
        <body>
            <h2>Virtual Range Setup</h2>
            <form action="/analyze" enctype="multipart/form-data" method="post" id="uploadForm">
                
                <p class="instructions"><strong>1. Select your video:</strong></p>
                <input id="videoFile" name="file" type="file" accept="video/mp4" style="margin-bottom: 10px;"><br>
                
                <p class="instructions" id="step2" style="display:none;"><strong>2. Tap the 4 corners of the target paper (Top-Left, Top-Right, Bottom-Left, Bottom-Right):</strong></p>
                <canvas id="videoFrame"></canvas>
                
                <input id="ptsInput" name="pts" type="hidden">
                
                <br>
                <input id="submitBtn" type="submit" value="Analyze Video" class="btn" disabled>
            </form>

            <script>
                const fileInput = document.getElementById('videoFile');
                const canvas = document.getElementById('videoFrame');
                const ctx = canvas.getContext('2d');
                const ptsInput = document.getElementById('ptsInput');
                const submitBtn = document.getElementById('submitBtn');
                const step2 = document.getElementById('step2');
                
                let points = [];

                // 1. When a video is selected, extract the first frame
                fileInput.addEventListener('change', function(e) {
                    const file = e.target.files[0];
                    if (!file) return;

                    const video = document.createElement('video');
                    video.src = URL.createObjectURL(file);
                    
                    // Tell the video to load its metadata and first frame
                    video.addEventListener('loadeddata', function() {
                        video.currentTime = 0; 
                    });

                    video.addEventListener('seeked', function() {
                        // Match canvas internal resolution to video's true resolution
                        canvas.width = video.videoWidth;
                        canvas.height = video.videoHeight;
                        
                        // Draw the frame onto the canvas
                        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                        
                        // Reset everything
                        points = [];
                        ptsInput.value = "";
                        submitBtn.disabled = true;
                        canvas.style.display = 'block';
                        step2.style.display = 'block';
                    });
                });

                // 2. Listen for taps on the canvas
                canvas.addEventListener('click', function(e) {
                    if (points.length >= 4) return; // Stop after 4 taps

                    // Calculate correct coordinates even if the canvas is scaled down on mobile
                    const rect = canvas.getBoundingClientRect();
                    const scaleX = canvas.width / rect.width;
                    const scaleY = canvas.height / rect.height;

                    const x = Math.round((e.clientX - rect.left) * scaleX);
                    const y = Math.round((e.clientY - rect.top) * scaleY);

                    points.push([x, y]);

                    // Draw a red dot where the user tapped
                    ctx.beginPath();
                    ctx.arc(x, y, 15, 0, 2 * Math.PI);
                    ctx.fillStyle = 'red';
                    ctx.fill();

                    // If 4 points are tapped, auto-fill the hidden input and enable the button
                    if (points.length === 4) {
                        ptsInput.value = JSON.stringify(points);
                        submitBtn.disabled = false;
                        submitBtn.value = "Target Set! Analyze Video";
                        submitBtn.style.backgroundColor = "#28a745"; // turn green
                    }
                });
            </script>
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

def process_video(path, src_pts, w, h):
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
                x, y, w_box, h_box = cv2.boundingRect(c)
                cx, cy = x + w_box//2, y + h_box//2
                hits.append([cx, cy])

    cap.release()
    hits = cluster_hits(hits)

    # UPDATED: Match the transform to the exact dimensions of your diagram
    dst_pts = np.float32([[0,0], [w,0], [0,h], [w,h]])
    M = cv2.getPerspectiveTransform(np.float32(src_pts), dst_pts)

    mapped_hits = []
    for h_pt in hits:
        pt = np.array([[h_pt]], dtype="float32")
        mapped = cv2.perspectiveTransform(pt, M)
        mapped_hits.append((int(mapped[0][0][0]), int(mapped[0][0][1])))

    return mapped_hits

def draw_map(hits, bg_path):
    # UPDATED: Load your custom diagram
    img = cv2.imread(bg_path)
    for (x,y) in hits:
        # Draw red circles (adjust the '6' to make dots bigger/smaller)
        cv2.circle(img, (x,y), 6, (0,0,255), -1) 
    cv2.imwrite("output.png", img)

@app.post("/analyze", response_class=HTMLResponse)
async def analyze(file: UploadFile = File(...), pts: str = Form(...)):
    # 1. Save the uploaded video
    contents = await file.read()
    with open("input.mp4", "wb") as f:
        f.write(contents)

    # 2. Load your custom diagram to get its exact dimensions
    bg_path = "target.png"
    bg_img = cv2.imread(bg_path)
    if bg_img is None:
        return "<h3>Error: target.png not found! Did you upload it to Cloud Shell?</h3>"
    h, w, _ = bg_img.shape

    # 3. Process the video using those dimensions
    src_pts = np.array(eval(pts), dtype="float32")
    hits = process_video("input.mp4", src_pts, w, h)
    
    # 4. Draw on the background image
    draw_map(hits, bg_path)

    # 5. Calculate score
    score = f"{min(len(hits),40)}/40"

    # 6. Return a visual webpage showing the result
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
