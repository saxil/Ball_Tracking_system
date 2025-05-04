# main.py (Modified for YOLOv5 Tracking)

import cv2
import torch
# import yolov5 # Optional: can also use the yolov5 library directly if installed
import numpy as np
import math
import time # To measure performance
import os # To check file paths

# --- Configuration ---
VIDEO_PATH = r'C:\Users\sahil\Desktop\everything\Desktop\ml\Projects\Ball_Tracking_system\test.mp4' # <<< MAKE SURE THIS IS YOUR VIDEO FILE
OUTPUT_VIDEO_PATH = 'output_yolov5_tracked_ball.mp4' # <<< VIDEO WILL BE SAVED HERE

# --- YOLOv5 Configuration ---
# Option 1: Use torch.hub (requires internet on first run for 'yolov5s', etc.)
# Or specify path to local yolov5 repo checkout: YOLOV5_REPO_PATH = 'path/to/yolov5'
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True) # Example: yolov5s
# model = torch.hub.load(YOLOV5_REPO_PATH, 'custom', path='path/to/your/yolov5_weights.pt', source='local') # Example: Custom local model

# Option 2: Specify path directly to your trained .pt file
YOLOV5_WEIGHTS_PATH = 'yolov5s.pt' # <<< PATH TO YOUR YOLOv5 .pt WEIGHTS FILE (e.g., yolov5s.pt, best.pt)

BALL_CLASS_ID = 32 # Class ID for 'sports ball' in COCO dataset (verify if custom trained)
CONFIDENCE_THRESHOLD = 0.40 # Your detection confidence threshold
IOU_THRESHOLD = 0.45 # NMS IOU threshold for YOLOv5

# --- Tracking Configuration ---
RE_DETECT_INTERVAL = 15 # How often to run YOLO detection even if tracking is okay (frames)
TRACKER_TYPE = "CSRT" # Options: "CSRT", "KCF", "MIL", "GOTURN", "DaSiamRPN", "Nano" (install opencv-contrib-python for most)

# --- Model Loading (YOLOv5) ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

try:
    # Using torch.hub to load the model from the specified weights path
    # Assumes the standard Ultralytics YOLOv5 structure/repo is accessible
    # or loading a local custom model.
    if not os.path.exists(YOLOV5_WEIGHTS_PATH):
         print(f"Error: YOLOv5 weights file not found at {YOLOV5_WEIGHTS_PATH}")
         print("Please download weights (e.g., yolov5s.pt) or specify the correct path.")
         # Attempt to load standard yolov5s if path invalid, requires internet
         try:
             print("Attempting to load 'yolov5s' from torch.hub...")
             model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
             print("Loaded 'yolov5s' successfully.")
         except Exception as hub_e:
             print(f"Failed to load 'yolov5s' from torch.hub: {hub_e}")
             print("Exiting.")
             exit()
    else:
        # Load from local .pt file. This assumes the yolov5 repo structure is implicitly known by torch.hub
        # or you might need to specify the repo path if loading custom models fails.
        # model = torch.hub.load('ultralytics/yolov5', 'custom', path=YOLOV5_WEIGHTS_PATH, force_reload=False) # force_reload=True if cache issues
        # Safer alternative if just loading weights:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=YOLOV5_WEIGHTS_PATH) # force_reload=False
        print(f"Loaded YOLOv5 model from: {YOLOV5_WEIGHTS_PATH}")

    model.to(device)
    model.conf = CONFIDENCE_THRESHOLD # Set confidence threshold
    model.iou = IOU_THRESHOLD      # Set IoU threshold for NMS

    # Optional: Specify classes to detect (e.g., only ball) - can improve speed slightly
    # model.classes = [BALL_CLASS_ID]

    model.eval()

except Exception as e:
    print(f"Error loading YOLOv5 model: {e}")
    print("Ensure torch, yolov5 dependencies are installed, and the weights path is correct.")
    print("You might need to clone the yolov5 repo: git clone https://github.com/ultralytics/yolov5")
    exit()

# --- Video I/O Setup ---
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Could not open video file: {VIDEO_PATH}")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) else 30

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))
print(f"Input video: {VIDEO_PATH} ({frame_width}x{frame_height} @ {fps} FPS)")
print(f"Output video will be saved to: {OUTPUT_VIDEO_PATH}")

# --- Tracker Initialization ---
def create_tracker(tracker_type):
    """Creates an OpenCV tracker based on the specified type."""
    if tracker_type == 'CSRT':
        return cv2.TrackerCSRT_create()
    elif tracker_type == 'KCF':
        return cv2.TrackerKCF_create()
    elif tracker_type == 'MIL':
        return cv2.TrackerMIL_create()
    # Add other trackers from opencv-contrib-python if needed
    # elif tracker_type == 'GOTURN':
    #     return cv2.TrackerGOTURN_create() # Requires model files
    # elif tracker_type == 'DaSiamRPN':
    #     return cv2.TrackerDaSiamRPN_create() # Requires model files
    # elif tracker_type == 'Nano':
    #     return cv2.TrackerNano_create() # Requires model files
    else:
        print(f"Warning: Unknown tracker type '{tracker_type}'. Using CSRT.")
        return cv2.TrackerCSRT_create()

tracker = None
tracker_initialized = False
bbox_tracked = None # Stores the current tracked bounding box (x, y, w, h)
frame_count = 0
detection_count = 0
tracking_count = 0
start_time = time.time()

# --- Processing Loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break

    frame_count += 1
    display_frame = frame.copy()
    current_loop_start_time = time.time()

    # --- Step 1: Try TRACKING ---
    tracking_successful_this_frame = False
    if tracker_initialized:
        ok, bbox_tracked = tracker.update(frame)
        if ok:
            tracking_successful_this_frame = True
            tracking_count += 1
            p1 = (int(bbox_tracked[0]), int(bbox_tracked[1]))
            p2 = (int(bbox_tracked[0] + bbox_tracked[2]), int(bbox_tracked[1] + bbox_tracked[3]))
            cv2.rectangle(display_frame, p1, p2, (0, 255, 0), 2, 1) # GREEN for tracked
            cv2.putText(display_frame, f"Tracking ({TRACKER_TYPE})", (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            print(f"Frame {frame_count}: Tracking failed - Attempting re-detection")
            tracker_initialized = False
            tracker = None
            bbox_tracked = None

    # --- Step 2: Run DETECTION if needed ---
    run_detection = not tracker_initialized or (frame_count % RE_DETECT_INTERVAL == 0)

    if run_detection:
        detection_count += 1
        print(f"Frame {frame_count}: Running YOLOv5 Detection...")
        best_ball_bbox_detected = None # Reset detected box for this frame
        try:
            # Perform inference with YOLOv5
            # YOLOv5 expects RGB format, cv2 reads BGR
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame_rgb) # Perform detection

            # Parse results
            # results.xyxy[0] contains detections for image 0 as [x1, y1, x2, y2, conf, class]
            detections = results.xyxy[0].cpu().numpy() # Move to CPU and convert to numpy

            ball_detections = []
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                if int(cls) == BALL_CLASS_ID and conf >= CONFIDENCE_THRESHOLD: # Check class and confidence
                    w = x2 - x1
                    h = y2 - y1
                    # Store as (x, y, w, h), score
                    ball_detections.append(((int(x1), int(y1), int(w), int(h)), conf))

            # Select the best ball detection (highest confidence)
            if ball_detections:
                ball_detections.sort(key=lambda x: x[1], reverse=True)
                best_ball_bbox_detected = ball_detections[0][0] # (x, y, w, h)
                best_score = ball_detections[0][1]

                # Draw BLUE box for newly detected object (optional)
                p1_det = (best_ball_bbox_detected[0], best_ball_bbox_detected[1])
                p2_det = (best_ball_bbox_detected[0] + best_ball_bbox_detected[2], best_ball_bbox_detected[1] + best_ball_bbox_detected[3])
                cv2.rectangle(display_frame, p1_det, p2_det, (255, 0, 0), 2, 1) # BLUE for detected
                cv2.putText(display_frame, f"Detected {best_score:.2f}", (p1_det[0], p1_det[1] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # --- Step 3: Initialize or Re-initialize TRACKER ---
            if best_ball_bbox_detected:
                if best_ball_bbox_detected[2] > 0 and best_ball_bbox_detected[3] > 0:
                    # Create and initialize a NEW tracker
                    print(f"Frame {frame_count}: Initializing tracker with bbox: {best_ball_bbox_detected}")
                    tracker = create_tracker(TRACKER_TYPE)
                    try:
                        ok_init = tracker.init(frame, best_ball_bbox_detected) # Use original BGR frame for tracker init
                        if ok_init:
                            tracker_initialized = True
                            bbox_tracked = best_ball_bbox_detected
                            # Overwrite blue detection box with green if tracking also marked this frame
                            if tracking_successful_this_frame:
                                p1 = (int(bbox_tracked[0]), int(bbox_tracked[1]))
                                p2 = (int(bbox_tracked[0] + bbox_tracked[2]), int(bbox_tracked[1] + bbox_tracked[3]))
                                cv2.rectangle(display_frame, p1, p2, (0, 255, 0), 2, 1) # Ensure green
                                cv2.putText(display_frame, f"Tracking ({TRACKER_TYPE}) (Re-Init)", (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        else:
                            print(f"Frame {frame_count}: Tracker initialization failed!")
                            tracker_initialized = False
                            tracker = None
                    except Exception as e:
                        print(f"Frame {frame_count}: Exception during tracker.init: {e}")
                        tracker_initialized = False
                        tracker = None
                else:
                    print(f"Frame {frame_count}: Invalid detection bbox w={best_ball_bbox_detected[2]}, h={best_ball_bbox_detected[3]}. Cannot init tracker.")
                    if tracker_initialized: # If tracking failed this frame, mark as not initialized
                         tracker_initialized = False
                         tracker = None

            elif run_detection: # Only print if detection ran and found nothing
                 print(f"Frame {frame_count}: Detection ran, but no ball found.")
                 if not tracking_successful_this_frame:
                      tracker_initialized = False
                      tracker = None

        except Exception as e:
            print(f"Error during YOLOv5 prediction or processing: {e}")
            # Continue processing next frame, maintaining current tracking state

    # --- Calculate and Display FPS ---
    current_loop_end_time = time.time()
    processing_time = current_loop_end_time - current_loop_start_time
    fps_current = 1.0 / processing_time if processing_time > 0 else 0
    cv2.putText(display_frame, f"FPS: {fps_current:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(display_frame, f"Frame: {frame_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # --- Write Output Frame ---
    out.write(display_frame)

    # --- Display (Optional) ---
    cv2.imshow("YOLOv5 Ball Tracking (Green=Tracked, Blue=Detected)", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Interrupted by user.")
        break

# --- Cleanup ---
total_time = time.time() - start_time
print(f"\n--- Processing Summary ---")
print(f"Total Frames Processed: {frame_count}")
print(f"Total Time: {total_time:.2f} seconds")
if frame_count > 0 and total_time > 0:
    print(f"Average FPS: {frame_count / total_time:.2f}")
print(f"Detection ran on: {detection_count} frames")
print(f"Tracking successful on: {tracking_count} frames")

cap.release()
out.release()
cv2.destroyAllWindows()
print("Processing finished. Output video saved to:", OUTPUT_VIDEO_PATH)