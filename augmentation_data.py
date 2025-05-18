import os
import cv2
import random
import json
import shutil
import xml.etree.ElementTree as ET
from tqdm import tqdm
from glob import glob
from sklearn.model_selection import train_test_split

# ----------- CONFIG -----------
INPUT_DIR = "./epillID/classification_data/segmented_nih_pills_224/"
TEMP_DIR = "./temp_augmented"
OUTPUT_DIR = "./augmented_data"
SPLITS = ["train", "val", "test"]
SPLIT_RATIO = (0.7, 0.15, 0.15)
RANDOM_SEED = 42

# ----------- UTILS -----------
def parse_filename(fname):
    name = os.path.basename(fname).split(".")[0]
    label = name.split("_")[0]  # Lấy phần trước dấu _ đầu tiên
    x, y, w, h = 0, 0, 224, 224  # bbox giả định toàn ảnh
    return label, x, y, w, h

def convert_bbox_to_yolo(bbox, img_w, img_h):
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    w /= img_w
    h /= img_h
    return cx, cy, w, h

def make_dirs():
    shutil.rmtree(TEMP_DIR, ignore_errors=True)
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    for fmt in ["yolo", "coco", "voc"]:
        for split in SPLITS:
            os.makedirs(f"{OUTPUT_DIR}/{fmt}/{split}/images", exist_ok=True)
            os.makedirs(f"{OUTPUT_DIR}/{fmt}/{split}/labels", exist_ok=True)

# ----------- AUGMENTATION METHODS -----------
def rotate_image(img, bbox):
    h, w = img.shape[:2]
    angle = random.choice([90, 180, 270])
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    rotated = cv2.warpAffine(img, M, (w, h))

    x, y, bw, bh = bbox
    box_pts = [(x, y), (x + bw, y), (x, y + bh), (x + bw, y + bh)]
    new_pts = [M @ [px, py, 1] for px, py in box_pts]
    xs, ys = zip(*new_pts)
    new_x = max(0, int(min(xs)))
    new_y = max(0, int(min(ys)))
    new_w = int(max(xs)) - new_x
    new_h = int(max(ys)) - new_y
    return rotated, (new_x, new_y, new_w, new_h)

def mosaic_images(items):
    if len(items) < 4: return []
    mosaic_data = []
    size = 224
    for i in range(0, len(items) - 3, 4):
        mosaic_img = 255 * np.ones((size * 2, size * 2, 3), dtype=np.uint8)
        new_bboxes = []
        labels = []

        for j in range(4):
            item = items[i + j]
            img = cv2.imread(item['image_path'])
            bbox = item['bbox']
            x_offset = (j % 2) * size
            y_offset = (j // 2) * size
            mosaic_img[y_offset:y_offset + size, x_offset:x_offset + size] = img

            bx, by, bw, bh = bbox
            new_bbox = (bx + x_offset, by + y_offset, bw, bh)
            new_bboxes.append(new_bbox)
            labels.append(item['label'])

        fname = f"mosaic_{i}.jpg"
        out_path = f"{TEMP_DIR}/{fname}"
        cv2.imwrite(out_path, mosaic_img)
        # Lưu chỉ một bbox đầu tiên đại diện (nếu bạn muốn nhiều bbox, cần chỉnh lại export)
        mosaic_data.append({
            'file_name': fname,
            'image_path': out_path,
            'bbox': new_bboxes[0],
            'label': labels[0]
        })
    return mosaic_data

def cutmix_images(items):
    if len(items) < 2: return []
    cutmix_data = []
    for i in range(0, len(items) - 1, 2):
        img1 = cv2.imread(items[i]['image_path'])
        img2 = cv2.imread(items[i + 1]['image_path'])
        h, w = img1.shape[:2]
        lam = 0.5
        x1 = int(w * lam)
        new_img = img1.copy()
        new_img[:, x1:] = img2[:, x1:]
        b1 = items[i]['bbox']
        b2 = items[i + 1]['bbox']
        bboxes = [b1, b2]
        fname = f"cutmix_{i}.jpg"
        cv2.imwrite(f"{TEMP_DIR}/{fname}", new_img)
        cutmix_data.append({'file_name': fname, 'image_path': f"{TEMP_DIR}/{fname}", 'bbox': bboxes[0], 'label': items[i]['label']})
    return cutmix_data

# ----------- EXPORT FORMATS -----------
def save_yolo(item, split):
    img = cv2.imread(item['image_path'])
    h, w = img.shape[:2]
    cx, cy, bw, bh = convert_bbox_to_yolo(item['bbox'], w, h)
    with open(f"{OUTPUT_DIR}/yolo/{split}/labels/{item['file_name'].replace('.jpg','.txt')}", "w") as f:
        f.write(f"0 {cx} {cy} {bw} {bh}\n")
    shutil.copy(item['image_path'], f"{OUTPUT_DIR}/yolo/{split}/images/{item['file_name']}")

def save_voc(item, split):
    img = cv2.imread(item['image_path'])
    h, w = img.shape[:2]
    x, y, bw, bh = item['bbox']
    ann = ET.Element("annotation")
    ET.SubElement(ann, "filename").text = item['file_name']
    size = ET.SubElement(ann, "size")
    ET.SubElement(size, "width").text = str(w)
    ET.SubElement(size, "height").text = str(h)
    ET.SubElement(size, "depth").text = "3"
    obj = ET.SubElement(ann, "object")
    ET.SubElement(obj, "name").text = item['label']
    bbox = ET.SubElement(obj, "bndbox")
    ET.SubElement(bbox, "xmin").text = str(x)
    ET.SubElement(bbox, "ymin").text = str(y)
    ET.SubElement(bbox, "xmax").text = str(x + bw)
    ET.SubElement(bbox, "ymax").text = str(y + bh)
    tree = ET.ElementTree(ann)
    tree.write(f"{OUTPUT_DIR}/voc/{split}/labels/{item['file_name'].replace('.jpg','.xml')}")
    shutil.copy(item['image_path'], f"{OUTPUT_DIR}/voc/{split}/images/{item['file_name']}")

def save_coco(data, split):
    coco = {"images": [], "annotations": [], "categories": [{"id": 0, "name": "pill"}]}
    for i, item in enumerate(data):
        img = cv2.imread(item['image_path'])
        h, w = img.shape[:2]
        coco["images"].append({"id": i, "file_name": item['file_name'], "width": w, "height": h})
        x, y, bw, bh = item['bbox']
        coco["annotations"].append({
            "id": i,
            "image_id": i,
            "category_id": 0,
            "bbox": [x, y, bw, bh],
            "area": bw * bh,
            "iscrowd": 0
        })
        shutil.copy(item['image_path'], f"{OUTPUT_DIR}/coco/{split}/images/{item['file_name']}")
    with open(f"{OUTPUT_DIR}/coco/{split}/labels/annotations.json", "w") as f:
        json.dump(coco, f)

# ----------- MAIN PIPELINE -----------
import numpy as np
def main():
    make_dirs()

    # Load original data
    data = []
    for img_path in glob(f"{INPUT_DIR}/*.jpg"):
        label, x, y, w, h = parse_filename(img_path)
        data.append({
            "file_name": os.path.basename(img_path),
            "image_path": img_path,
            "label": label,
            "bbox": (x, y, w, h)
        })

    rotated_data = []
    for item in tqdm(data, desc="Rotate"):
        img = cv2.imread(item['image_path'])
        new_img, new_bbox = rotate_image(img, item['bbox'])
        new_name = f"rot_{item['file_name']}"
        out_path = f"{TEMP_DIR}/{new_name}"
        cv2.imwrite(out_path, new_img)
        rotated_data.append({
            "file_name": new_name,
            "image_path": out_path,
            "label": item['label'],
            "bbox": new_bbox
        })

    mosaic_data = mosaic_images(data)
    cutmix_data = cutmix_images(data)

    all_data = rotated_data + mosaic_data + cutmix_data
    random.shuffle(all_data)

    trainval, test = train_test_split(all_data, test_size=SPLIT_RATIO[2], random_state=RANDOM_SEED)
    train, val = train_test_split(trainval, test_size=SPLIT_RATIO[1]/(SPLIT_RATIO[0]+SPLIT_RATIO[1]), random_state=RANDOM_SEED)

    for split_name, items in zip(SPLITS, [train, val, test]):
        for item in tqdm(items, desc=f"Saving {split_name}"):
            save_yolo(item, split_name)
            save_voc(item, split_name)
        save_coco(items, split_name)

    print("DONE! All augmented data saved in ./output")

if __name__ == "__main__":
    main()
