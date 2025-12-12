import os
import xml.etree.ElementTree as ET
from ultralytics import YOLO


DATASETS_DIR = "data"
ANNOTATIONS_PATH = os.path.join(DATASETS_DIR, "annotations.xml")
LABELS_DIR = os.path.join(DATASETS_DIR, "labels")


def ensure_labels():
    os.makedirs(LABELS_DIR, exist_ok=True)

    tree = ET.parse(ANNOTATIONS_PATH)
    root = tree.getroot()

    for image in root.findall("image"):
        image_id = image.get("id")
        width = float(image.get("width"))
        height = float(image.get("height"))

        label_file_path = os.path.join(LABELS_DIR, f"{image_id}.txt")
        with open(label_file_path, "w") as label_file:
            for box in image.findall("box"):
                xtl = float(box.get("xtl"))
                ytl = float(box.get("ytl"))
                xbr = float(box.get("xbr"))
                ybr = float(box.get("ybr"))

                x_center = (xtl + xbr) / 2
                y_center = (ytl + ybr) / 2
                bbox_width = xbr - xtl
                bbox_height = ybr - ytl

                x_center /= width
                y_center /= height
                bbox_width /= width
                bbox_height /= height
                box = [
                    "0",
                    f"{x_center:.6f}",
                    f"{y_center:.6f}",
                    f"{bbox_width:.6f}",
                    f"{bbox_height:.6f}",
                ]
                label_file.write(" ".join(box) + "\n")


def main():
    ensure_labels()

    model = YOLO("yolo11n.pt")

    model.train(
        data="strawberries.yaml",
        device="mps",
        save=True,
        plots=False,
        project="./runs",
    )


if __name__ == "__main__":
    main()
