import json
import argparse
import pandas as pd
from tqdm import tqdm

import cv2

from utils import parse_txt, xywh_to_xyxy, denornmalized_vertex

from typing import Any

# https://cocodataset.org/#format-data
# https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch


CLASSES = {
    "Bag": 0,
    "Shoe": 1,
    "Outerwear": 2,
    "Hat": 3,
    "Dress": 4,
    "Pants": 5,
    "Skirt": 6,
    "Top": 7,
    "Shorts": 8,
    "Glasses": 9,
}


def create_image_section(image_id: int, width: int, height: int, file_name: str) -> dict[str:Any]:
    image = {
        "id": image_id,
        "width": width,
        "height": height,
        "file_name": file_name,
        # "license": int,
        # "flickr_url": str,
        # "coco_url": str,
        # "date_captured": datetime,
    }

    return image


def create_annotation_section(annotation_id: int, image_id: int, category_id: int,
                              min_x: int, min_y: int, width: int, height: int):
    bbox = (min_x, min_y, width, height)
    area = width * height

    annotation = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": [],
        "area": area,
        "bbox": bbox,
        "iscrowd": 0,
    }

    return annotation


def images_annotations_info(csv_file: str) -> tuple[list[dict[str:Any]]]:
    df = pd.read_csv(csv_file)

    images = []
    annotations = []

    annotations_id = 1   # In COCO dataset format, you must start annotation id with '1'

    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        image = cv2.imread(row["images"])
        h, w, _ = image.shape

        tmp = create_image_section(row["images"], w, h, int(row["id"]))
        images.append(tmp)

        label = parse_txt(row["labels"])

        for line in label:
            category_id, xywh = line
            xyxy = xywh_to_xyxy(xywh)
            xyxy = denornmalized_vertex(xyxy, image.shape[:2])
            xywh = denornmalized_vertex(xywh, image.shape[:2])

            tmp = create_annotation_section(annotations_id, int(row["id"]), category_id,
                                            xyxy[0], xyxy[1], xywh[2], xywh[3])
            annotations.append(tmp)
            annotations_id += 1

    return images, annotations


def categories_info():
    categories = []

    for k, v in CLASSES.items():
        categories.append({
            "id": v,
            "name": k,
            "supercategory": k
        })

    return categories


def get_args():
    parser = argparse.ArgumentParser(
        "Yolo format annotations to COCO dataset format")
    parser.add_argument("--csv_file",
                        type=str,
                        help="CSV file stored images and label paths.",
                        default="./data.csv")
    parser.add_argument('-o', '--output',
                        type=str,
                        help="Name the output json file",
                        # nargs="?",
                        default="./annotations.json")
    # const="")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()

    coco_format = {}
    coco_format["images"], coco_format["annotations"] = images_annotations_info(
        args.csv_file)

    coco_format["categories"] = categories_info()

    with open(args.output, "w") as outfile:
        json.dump(coco_format, outfile)
