import cv2
import torch

# Load trained model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/greenball_yolov5_cpu387/weights/best.pt', force_reload=True)

# Load input video
cap = cv2.VideoCapture('test.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_tracked_ball.mp4', fourcc, int(cap.get(5)), (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Inference
    results = model(frame)
    results.render()  # Draws boxes in-place

    out.write(results.imgs[0])  # results.imgs[0] is np.uint8 image with boxes

cap.release()
out.release()
