import os
import random
import shutil

# Set paths (using raw strings or forward slashes to avoid escape characters)
base_dir = r"C:\Users\sahil\Desktop\everything\Desktop\ml\Projects\Ball_Tracking_system\greenball_dataset"
img_dir = os.path.join(base_dir, "images/train")
lbl_dir = os.path.join(base_dir, "labels/train")

val_img_dir = os.path.join(base_dir, "images/val")
val_lbl_dir = os.path.join(base_dir, "labels/val")

# Create val folders
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(val_lbl_dir, exist_ok=True)

# Split
files = os.listdir(img_dir)
val_split = 0.2
val_count = int(len(files) * val_split)
val_files = random.sample(files, val_count)

for f in val_files:
    # Move image
    shutil.move(os.path.join(img_dir, f), os.path.join(val_img_dir, f))
    
    # Move label
    label_name = f.replace(".jpg", ".txt").replace(".png", ".txt")
    shutil.move(os.path.join(lbl_dir, label_name), os.path.join(val_lbl_dir, label_name))

print("✅ Validation split done.")
