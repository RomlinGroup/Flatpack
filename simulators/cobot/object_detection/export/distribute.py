import os
import json
import shutil
import zipfile
import random


def split_dataset(json_path, train_ratio=0.8):
    with open(json_path, 'r') as file:
        data = json.load(file)

    random.shuffle(data['images'])
    num_train_images = int(len(data['images']) * train_ratio)
    train_images = data['images'][:num_train_images]
    val_images = data['images'][num_train_images:]

    train_annotations = [anno for anno in data['annotations'] if
                         anno['image_id'] in [img['id'] for img in train_images]]
    val_annotations = [anno for anno in data['annotations'] if anno['image_id'] in [img['id'] for img in val_images]]

    train_data = {'images': train_images, 'annotations': train_annotations, 'categories': data['categories']}
    val_data = {'images': val_images, 'annotations': val_annotations, 'categories': data['categories']}

    return train_data, val_data


def distribute_images_and_labels(base_image_folder, train_data, val_data, output_root):
    train_dir = os.path.join(output_root, 'train/images')
    val_dir = os.path.join(output_root, 'val/images')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    train_images = {os.path.basename(item['file_name']) for item in train_data['images']}
    val_images = {os.path.basename(item['file_name']) for item in val_data['images']}

    for image_name in os.listdir(base_image_folder):
        source_path = os.path.join(base_image_folder, image_name)
        if image_name in train_images:
            shutil.copy(source_path, os.path.join(train_dir, image_name))
        elif image_name in val_images:
            shutil.copy(source_path, os.path.join(val_dir, image_name))

    train_json = os.path.join(output_root, 'train/labels.json')
    val_json = os.path.join(output_root, 'val/labels.json')
    with open(train_json, 'w') as file:
        json.dump(train_data, file, indent=4)
    with open(val_json, 'w') as file:
        json.dump(val_data, file, indent=4)

    def create_zip(directory, zip_name):
        with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if not file.startswith('__MACOSX'):
                        zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), directory))

    create_zip(output_root, f"{output_root}.zip")


json_path = 'result.json'
base_image_folder = 'images'
output_root = 'mediapipe_dataset'

train_data, val_data = split_dataset(json_path)
distribute_images_and_labels(base_image_folder, train_data, val_data, output_root)
