import os
import cv2
import albumentations as A
import shutil

# Paths
input_img_dir = 'greenball_dataset/images/train'
input_lbl_dir = 'greenball_dataset/labels/train'
output_img_dir = 'greenball_dataset/images/train_aug'
output_lbl_dir = 'greenball_dataset/labels/train_aug'

os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_lbl_dir, exist_ok=True)

# Define augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomBrightnessContrast(p=0.3),
    A.Rotate(limit=30, p=0.5),
    A.RandomScale(scale_limit=0.1, p=0.4),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def read_labels(label_path):
    bboxes = []
    class_labels = []
    with open(label_path, 'r') as f:
        for line in f.readlines():
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            bboxes.append([x_center, y_center, width, height])
            class_labels.append(int(class_id))
    return bboxes, class_labels

def save_labels(label_path, bboxes, class_labels):
    with open(label_path, 'w') as f:
        for bbox, cls in zip(bboxes, class_labels):
            f.write(f"{cls} {' '.join([str(round(x, 6)) for x in bbox])}\n")

# Augment each image 3 times
for file_name in os.listdir(input_img_dir):
    if not file_name.endswith(('.jpg', '.png', '.jpeg')):
        continue

    img_path = os.path.join(input_img_dir, file_name)
    lbl_path = os.path.join(input_lbl_dir, os.path.splitext(file_name)[0] + '.txt')

    image = cv2.imread(img_path)
    height, width = image.shape[:2]

    if not os.path.exists(lbl_path):
        continue

    bboxes, class_labels = read_labels(lbl_path)

    for i in range(3):  # Create 3 augmented versions
        augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        aug_img = augmented['image']
        aug_bboxes = augmented['bboxes']
        aug_classes = augmented['class_labels']

        if not aug_bboxes:
            continue  # Skip if no boxes

        # Save augmented image and label
        new_img_name = f"{os.path.splitext(file_name)[0]}_aug{i}.jpg"
        new_lbl_name = f"{os.path.splitext(file_name)[0]}_aug{i}.txt"

        cv2.imwrite(os.path.join(output_img_dir, new_img_name), aug_img)
        save_labels(os.path.join(output_lbl_dir, new_lbl_name), aug_bboxes, aug_classes)

print("✅ Augmentation complete. Files saved in:")
print(f"   - Images: {output_img_dir}")
print(f"   - Labels: {output_lbl_dir}")
