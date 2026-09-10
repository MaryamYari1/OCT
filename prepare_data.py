import os
import shutil
import random

source_base = "/Users/mary/Downloads/CellData/OCT"
dest_base = "data"

n_images_per_class = 400
n_val_per_class = 80

def copy_random_images(source_split, class_folder_name, dest_class_name, dest_split, n_images):
    class_folder = os.path.join(source_base, source_split, class_folder_name)
    files = os.listdir(class_folder)
    selected_files = random.sample(files, n_images)

    dest_folder = os.path.join(dest_base, dest_split, dest_class_name)

    for filename in selected_files:
        src_path = os.path.join(class_folder, filename)
        dst_path = os.path.join(dest_folder, filename)
        shutil.copy(src_path, dst_path)

    print(f"{source_split}/{class_folder_name} -> {dest_split}/{dest_class_name}: {len(selected_files)} images copied")


copy_random_images("train", "CNV", "abnormal", "train", n_images_per_class)
copy_random_images("train", "DME", "abnormal", "train", n_images_per_class)
copy_random_images("train", "DRUSEN", "abnormal", "train", n_images_per_class)
copy_random_images("train", "NORMAL", "normal", "train", n_images_per_class)

copy_random_images("test", "CNV", "abnormal", "val", n_val_per_class)
copy_random_images("test", "DME", "abnormal", "val", n_val_per_class)
copy_random_images("test", "DRUSEN", "abnormal", "val", n_val_per_class)
copy_random_images("test", "NORMAL", "normal", "val", n_val_per_class)


for split in ["train", "val"]:
    for cls in ["normal", "abnormal"]:
        folder = os.path.join(dest_base, split, cls)
        count = len(os.listdir(folder))
        print(f"{split}/{cls}: {count} images")