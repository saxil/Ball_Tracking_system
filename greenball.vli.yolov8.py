import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

# Paths to original datasets
roboflow_images = Path("roboflow_data/valid/images")
roboflow_labels = Path("roboflow_data/valid/labels")

kaggle_images = Path("kaggle_data/Images")
kaggle_labels = Path("kaggle_data/Labels")

# Combined dataset path
combined_dir = Path("combined_greenball")
images_dir = combined_dir / "images"
labels_dir = combined_dir / "labels"

# Ensure output dirs are clean
for d in [images_dir, labels_dir]:
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)

# Copy all images and labels
def copy_dataset(src_images, src_labels):
    for image_file in src_images.glob("*.jpg"):
        label_file = src_labels / (image_file.stem + ".txt")
        if label_file.exists():
            shutil.copy(image_file, images_dir / image_file.name)
            shutil.copy(label_file, labels_dir / label_file.name)

copy_dataset(roboflow_images, roboflow_labels)
copy_dataset(kaggle_images, kaggle_labels)

# Split into train/val
all_images = list(images_dir.glob("*.jpg"))
train_imgs, val_imgs = train_test_split(all_images, test_size=0.2, random_state=42)

def move_images_to_split(split_name, image_list):
    split_image_dir = combined_dir / split_name / "images"
    split_label_dir = combined_dir / split_name / "labels"
    split_image_dir.mkdir(parents=True, exist_ok=True)
    split_label_dir.mkdir(parents=True, exist_ok=True)

    for img_path in image_list:
        lbl_path = labels_dir / (img_path.stem + ".txt")
        shutil.move(str(img_path), split_image_dir / img_path.name)
        if lbl_path.exists():
            shutil.move(str(lbl_path), split_label_dir / lbl_path.name)

move_images_to_split("train", train_imgs)
move_images_to_split("val", val_imgs)

# Remove leftover combined root-level image/label folders
shutil.rmtree(images_dir)
shutil.rmtree(labels_dir)

# Write the data.yaml file
with open(combined_dir / "data.yaml", "w") as f:
    f.write(
        f"path: {combined_dir.resolve()}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"nc: 1\n"
        f"names: ['greenball']\n"
    )

print(f"✅ Dataset combined and split. Check: {combined_dir.resolve()}")
