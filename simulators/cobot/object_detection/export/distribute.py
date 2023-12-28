import os
import json
import shutil


def distribute_images_and_labels(base_image_folder, train_json, val_json, output_root):
    with open(train_json, 'r') as file:
        train_data = json.load(file)
    with open(val_json, 'r') as file:
        val_data = json.load(file)

    train_images = {os.path.basename(item['file_name']) for item in train_data['images']}
    val_images = {os.path.basename(item['file_name']) for item in val_data['images']}

    train_dir = os.path.join(output_root, 'train/images')
    val_dir = os.path.join(output_root, 'val/images')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for image_name in os.listdir(base_image_folder):
        source_path = os.path.join(base_image_folder, image_name)
        if image_name in train_images:
            shutil.move(source_path, os.path.join(train_dir, image_name))
        elif image_name in val_images:
            shutil.move(source_path, os.path.join(val_dir, image_name))

    shutil.copy(train_json, os.path.join(output_root, 'train/label.json'))
    shutil.copy(val_json, os.path.join(output_root, 'val/label.json'))


base_image_folder = 'images'
train_json = 'train.json'
val_json = 'val.json'
output_root = 'mediapipe_dataset'

distribute_images_and_labels(base_image_folder, train_json, val_json, output_root)
