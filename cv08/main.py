import cv2
from matplotlib import pyplot as plt
from skimage.feature import blob_log
from scipy.stats import entropy

FILES = [f"data/pvi_cv08_sunflowers{i}" for i in range(1, 5)]


def iou(boxA, boxB):
    # box = (x1, y1, x2, y2)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea)


def evaluate_detections(detected_boxes, ground_truth_boxes, iou_threshold=0.5):
    TP = 0
    FP = 0
    FN = 0

    matched_gt = set()

    for det in detected_boxes:
        match_found = False
        for i, gt in enumerate(ground_truth_boxes):
            if i in matched_gt:
                continue
            if iou(det, gt) >= iou_threshold:
                TP += 1
                matched_gt.add(i)
                match_found = True
                break
        if not match_found:
            FP += 1

    FN = len(ground_truth_boxes) - len(matched_gt)
    return TP, FP, FN


def load():
    results = []

    for file_prefix in FILES:
        img = cv2.imread(f"{file_prefix}.jpg")
        boxes = []

        with open(f"{file_prefix}.txt", "r") as f:
            for line in f:
                x1, y1, x2, y2 = map(int, line.strip().split())
                boxes.append((x1, y1, x2, y2))
        results.append((img, boxes))

    return results


def main():
    data = load()

    ref_img = cv2.imread("data/pvi_cv08_sunflower_template.jpg")
    ref_img_hsv = cv2.cvtColor(ref_img, cv2.COLOR_BGR2HSV)
    ref_h, _, _ = cv2.split(ref_img_hsv)
    ref_hist = cv2.calcHist([ref_h], [0], None, [180], [0, 180])
    ref_hist = ref_hist / ref_hist.sum()

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB))
    plt.title("Reference Image")

    plt.subplot(1, 2, 2)
    plt.plot(ref_hist)
    plt.title("Reference Hue Histogram")
    plt.tight_layout()

    for img, boxes in data:
        manually_annotated = img.copy()
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(manually_annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(manually_annotated, cv2.COLOR_BGR2RGB))
        plt.title("Image with Annotations")
        plt.tight_layout()

        res = blob_log(
            255 - cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
            # eliminovalo hodne overlapping blobu, necham to tak
            overlap=0.1,
        )

        detected = img.copy()
        detected_boxes = []
        for y, x, sigma_sigma_boy in res:
            # obcas tam byly bloby se sigmou 1? nevim proc zkousel jsem si hrat s parametrama ale akorat mi to davalo vic mensich blobu,
            # anyways, filtruju je tady
            if sigma_sigma_boy < 5:
                continue
            
            h, w = img.shape[:2]
            x1 = max(0, int(x - 2 * sigma_sigma_boy))
            y1 = max(0, int(y - 2 * sigma_sigma_boy))
            x2 = min(w, int(x + 2 * sigma_sigma_boy))
            y2 = min(h, int(y + 2 * sigma_sigma_boy))
            blob = img[y1:y2, x1:x2]

            blob_hsv = cv2.cvtColor(blob, cv2.COLOR_BGR2HSV)
            blob_h, _, _ = cv2.split(blob_hsv)
            blob_hist = cv2.calcHist([blob_h], [0], None, [180], [0, 180])
            blob_hist = blob_hist / blob_hist.sum()

            diff = entropy(ref_hist + 0.001, blob_hist + 0.001)

            print(
                f"Blob at ({x1}, {y1}, {x2}, {y2}) - area: {(x2 - x1) * (y2 - y1)}, sigma: {sigma_sigma_boy}, entropy diff: {diff}"
            )
            if diff < 0.7:
                detected_boxes.append((x1, y1, x2, y2))
                cv2.rectangle(detected, (x1, y1), (x2, y2), (255, 0, 0), 2)

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(detected, cv2.COLOR_BGR2RGB))
        plt.title("Detected Blobs")
        plt.tight_layout()

        TP, FP, FN = evaluate_detections(detected_boxes, boxes)

        print("True Positives (TP):", TP)
        print("False Positives (FP):", FP)
        print("False Negatives (FN):", FN)

        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0

        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")

        plt.show()


if __name__ == "__main__":
    main()
