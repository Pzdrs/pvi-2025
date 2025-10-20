import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import label, center_of_mass


def load_image(path: str):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Soubor '{path}' nebyl nalezen.")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    return img_rgb, gray, hue


def segment_images(gray, hue, th_gray=150, th_hue=75):
    BWgray = np.where(gray >= th_gray, 1, 0).astype(np.uint8)
    BWhue = np.where(hue < th_hue, 1, 0).astype(np.uint8)
    return BWgray, BWhue


def clean_binary(binary, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return opened


def label_regions(binary):
    labeled, num = label(binary)
    return labeled, num


def compute_region_data(binary, labeled, num_labels):
    centers = center_of_mass(binary, labeled, range(1, num_labels + 1))
    sizes = [(labeled == i).sum() for i in range(1, num_labels + 1)]
    return centers, sizes


def classify_by_size(sizes, centers):
    thresholds = {
        5400: 10,
        4900: 5,
        4000: 2,
        0: 1,
    }

    results = []
    for s, (y, x) in zip(sizes, centers):
        val = next(value for threshold, value in thresholds.items() if s > threshold)
        results.append(((int(x), int(y)), s, val))

    return results


def show_segmentation(gray, hue, BWgray, BWhue):
    plt.figure("Segmentation", figsize=(12, 8))

    plt.subplot(2, 3, 1)
    plt.title("Gray Image")
    plt.imshow(gray, cmap="gray")
    plt.colorbar()

    plt.subplot(2, 3, 2)
    plt.title("Gray Histogram")
    plt.plot(cv2.calcHist([gray], [0], None, [256], [0, 256]))

    plt.subplot(2, 3, 3)
    plt.title("Binary Image from Gray")
    plt.imshow(BWgray, cmap="jet")
    plt.colorbar()

    plt.subplot(2, 3, 4)
    plt.title("Hue Image")
    plt.imshow(hue, cmap="jet")
    plt.colorbar()

    plt.subplot(2, 3, 5)
    plt.title("Hue Histogram")
    plt.plot(cv2.calcHist([hue], [0], None, [180], [0, 180]))

    plt.subplot(2, 3, 6)
    plt.title("Binary Image from Hue")
    plt.imshow(BWhue, cmap="jet")
    plt.colorbar()

    plt.tight_layout()


def show_binary_processing(BWhue, opened, labeled):
    plt.figure("Binary Processing Steps", figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plt.title("Binary Image from Hue")
    plt.imshow(BWhue, cmap="jet")
    plt.colorbar()

    plt.subplot(2, 2, 2)
    plt.title("Binary Image from Hue - Opening")
    plt.imshow(opened, cmap="jet")
    plt.colorbar()

    plt.subplot(2, 2, 3)
    plt.title("Binary Image from Hue - Label")
    plt.imshow(labeled, cmap="jet")
    plt.colorbar()

    plt.subplot(2, 2, 4)
    plt.title("Binary Image from Hue - Centroids")
    plt.imshow(labeled, cmap="jet")
    plt.colorbar()

    
    centroids = center_of_mass(opened, labeled, range(1, labeled.max() + 1))
    if centroids:
        x_coords = [c[1] for c in centroids]
        y_coords = [c[0] for c in centroids]
        plt.scatter(x_coords, y_coords, color="white", s=40, marker="x", label="Centroids")
        plt.legend() 

    plt.tight_layout()
    plt.show()


def show_final(img_rgb, results, labeled):
    plt.figure("Final Output", figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Počet pixelů v objektech")
    plt.imshow(img_rgb)
    for (x, y), size, _ in results:
        plt.text(x, y, str(size), color="blue", fontsize=12)

    plt.subplot(1, 2, 2)
    plt.title("Klasifikace mincí podle velikosti")
    plt.imshow(img_rgb)
    for (x, y), _, val in results:
        plt.text(x, y, f"{val} Kč", color="blue", fontsize=12)
    

    plt.tight_layout()
    plt.show()


def main():
    path = Path("data/pvi_cv05_mince_noise.png")
    img_rgb, gray, hue = load_image(path.as_posix())

    BWgray, BWhue = segment_images(gray, hue)

    show_segmentation(gray, hue, BWgray, BWhue)

    opened = clean_binary(BWhue)

    labeled, num = label_regions(opened)

    centers, sizes = compute_region_data(opened, labeled, num)

    results = classify_by_size(sizes, centers)

    show_binary_processing(BWhue, opened, labeled)

    show_final(img_rgb, results, labeled)


if __name__ == "__main__":
    main()