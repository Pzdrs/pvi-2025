import os

import cv2
import numpy as np


def match_features(char_features, features):
    best_match = None
    best_distance = float("inf")
    for char, feat in features.items():
        distance = np.linalg.norm(np.array(char_features) - np.array(feat))
        if distance < best_distance:
            best_distance = distance
            best_match = char
    return best_match, best_distance


def main():
    features = {}

    for path in os.listdir("data/dir_znaky"):
        img = cv2.imread(os.path.join("data/dir_znaky", path))
        binary = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        horizontal = np.sum(binary < 255, axis=1)
        vertical = np.sum(binary < 255, axis=0)

        features[path.split(".")[0].lower()] = (*horizontal, *vertical)

    text_img = cv2.imread("data/pvi_cv07_text.bmp", cv2.IMREAD_GRAYSCALE)

    _, binary = cv2.threshold(text_img, 127, 255, cv2.THRESH_BINARY_INV)

    cv2.imshow("Binary Image", binary)
    cv2.waitKey(0)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    result_text = ""
    for i in range(1, num_labels):
        x, y, w, h, _ = stats[i]
        char_img = binary[y : y + h, x : x + w]

        horizontal = np.sum(char_img > 0, axis=1)
        vertical = np.sum(char_img > 0, axis=0)
        char_features = (*horizontal, *vertical)

        best_match, _ = match_features(char_features, features)

        result_text += best_match

    print(f"Resulting text: {result_text}")


if __name__ == "__main__":
    main()
