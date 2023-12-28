import json
import random


def split_coco_data(json_file, train_file, val_file, split_ratio=0.8):
    with open(json_file, 'r') as file:
        data = json.load(file)

    images = data['images']

    random.shuffle(images)

    split_index = int(len(images) * split_ratio)

    train_images = images[:split_index]
    val_images = images[split_index:]

    train_data = {'images': train_images}
    val_data = {'images': val_images}

    with open(train_file, 'w') as file:
        json.dump(train_data, file, indent=4)

    with open(val_file, 'w') as file:
        json.dump(val_data, file, indent=4)


split_coco_data('result.json', 'train.json', 'val.json')
