from ultralytics import YOLO
import cv2

# Load YOLOv8 model (change path if needed)
model = YOLO("runs/detect/greenball_yolov83/weights/best.pt")
# Load video
cap = cv2.VideoCapture("test.mp4")
if not cap.isOpened():
    raise Exception("⚠️ Couldn't open the video file.")````````````````````````````````````````````````                                     
# Video properties
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_tracked_ball.mp4", fourcc, fps, (w, h))

frame_num = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 inference
    results = model.predict(source=frame, conf=0.05, verbose=False)

    if results and len(results[0].boxes) > 0:
        print(f"[DETECTED] Frame {frame_num}: {len(results[0].boxes)} box(es)")
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw center point
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
    else:
        print(f"[NO DETECTION] Frame {frame_num}")
    # Write frame to output
    out.write(frame)

    # Show progress every 10 frames
    if frame_num % 10 == 0:
        print(f"[INFO] Processed frame: {frame_num}")
    frame_num += 1

# Release resources
cap.release()
out.release()
print("Output saved as 'output_tracked_ball.mp4    '")


