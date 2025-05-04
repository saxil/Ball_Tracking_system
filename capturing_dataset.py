import cv2
import os

# Create a directory to save images
save_dir = 'green_ball_images'
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

print("Press 'c' to capture an image. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Capture Green Ball Images', frame)
    key = cv2.waitKey(1)

    if key & 0xFF == ord('c'):
        img_name = f"{save_dir}/image_{count:04d}.jpg"
        cv2.imwrite(img_name, frame)
        print(f"Captured {img_name}")
        count += 1
    elif key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
