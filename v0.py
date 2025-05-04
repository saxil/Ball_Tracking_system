import cv2

# Load reference image (the ball you want to detect)
reference_image = cv2.imread(r'C:\Users\sahil\Desktop\everything\Desktop\ml\Projects\Ball_Tracking_system\greenball_dataset\images\train\ball_reference.jpg', cv2.IMREAD_GRAYSCALE)
if reference_image is None:
    print("Reference image not found!")
    exit()

# Create ORB detector
orb = cv2.ORB_create()

# Compute keypoints and descriptors for reference image
kp_ref, des_ref = orb.detectAndCompute(reference_image, None)

# Initialize matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and descriptors in current frame
    kp_frame, des_frame = orb.detectAndCompute(gray_frame, None)

    if des_frame is not None:
        # Match descriptors
        matches = bf.match(des_ref, des_frame)
        matches = sorted(matches, key=lambda x: x.distance)

        # Draw matches if enough good ones found
        if len(matches) > 10:
            result_img = cv2.drawMatches(reference_image, kp_ref,
                                         frame, kp_frame,
                                         matches[:20], None,
                                         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imshow("Ball Matching", result_img)
        else:
            cv2.imshow("Ball Matching", frame)
    else:
        cv2.imshow("Ball Matching", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
