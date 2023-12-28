import os
import json
import shutil
import zipfile


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
            shutil.copy(source_path, os.path.join(train_dir, image_name))
        elif image_name in val_images:
            shutil.copy(source_path, os.path.join(val_dir, image_name))

    shutil.copy(train_json, os.path.join(output_root, 'train/labels.json'))
    shutil.copy(val_json, os.path.join(output_root, 'val/labels.json'))

    def create_zip(directory, zip_name):
        with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if not file.startswith('__MACOSX'):
                        zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), directory))

    create_zip(output_root, f"{output_root}.zip")


base_image_folder = 'images'
train_json = 'train.json'
val_json = 'val.json'
output_root = 'mediapipe_dataset'

distribute_images_and_labels(base_image_folder, train_json, val_json, output_root)
