import os 
from glob import glob
import shutil
import json
from tqdm import tqdm


class_mappping = {
    3: 0,
    6: 1,
    8: 2,
    4: 3,
    2: 4
}

def make_yolo_data(json_path):
    base_folder  = os.path.dirname(json_path)
    with open(json_path, 'r') as f:
        data = json.load(f)
    images = data['images']
    annotations = data['annotations']
    labels_folder = os.path.join(base_folder, 'labels')
    if not os.path.exists(labels_folder):
        os.makedirs(labels_folder, exist_ok=True)

    for image in tqdm(images):
        with open(os.path.join(labels_folder, image['file_name'].split('.')[0] + '.txt'), 'w') as txt_file:
            image_id = image['id']
            h ,w = image['height'], image['width']
            for annotation in annotations:
                if annotation['image_id'] == image_id and annotation['category_id'] in class_mappping.keys():
                    bbox = annotation['bbox']
                    x, y, width, height = bbox
                    x_center = x + width / 2
                    y_center = y + height / 2
                    x_center /= w
                    y_center /= h
                    width /= w
                    height /= h
                    txt_file.write(f'{class_mappping[annotation["category_id"]]} {x_center} {y_center} {width} {height}\n')

if __name__ == "__main__":
    json_paths = ['/home/tanpv/fiftyone/coco-2017/train/labels.json',
                  '/home/tanpv/fiftyone/coco-2017/validation/labels.json',]
    for json_path in json_paths:
        make_yolo_data(json_path)
 