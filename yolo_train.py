from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO(r"runs\detect\greenball_yolov83\weights\best.pt")

# Open the input video
video_path = r"test.mp4"
cap = cv2.VideoCapture(video_path)

# Video writer setup
out = cv2.VideoWriter("output_tracked_ball.mp4", 
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      cap.get(cv2.CAP_PROP_FPS),
                      (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference
    results = model.predict(source=frame, conf=0.3, verbose=False)

    # Draw results on frame
    annotated_frame = results[0].plot()

    # Save to output
    out.write(annotated_frame)

cap.release()
out.release()

