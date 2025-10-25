import cv2
from matplotlib import pyplot as plt
import numpy as np


def segment(image, threshold=40):
    return np.where(image > threshold, 0, 1)


def granulometric_spectrum(
    binary_image, min_size=5, max_size=50, step=2, return_heatmap=False
):
    """Granulometrické spektrum + volitelná heatmapa."""
    binary = np.uint8(binary_image > 0) * 255
    sizes = list(range(min_size, max_size + 1, step))
    survived_pixels = []

    if return_heatmap:
        heatmap = np.zeros_like(binary, dtype=np.float32)

    for size in sizes:
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, se)
        survived_pixels.append(np.sum(opened > 0))
        if return_heatmap:
            heatmap += opened / 255.0

    survived_pixels = np.array(survived_pixels)
    diff = survived_pixels[:-1] - survived_pixels[1:]
    sizes = np.array(sizes[:-1])

    if return_heatmap:
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        return sizes, diff, heatmap
    else:
        return sizes, diff, None


def main():
    original = cv2.imread("data/pvi_cv06_mince.jpg")
    original_hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
    h, _, _ = cv2.split(original_hsv)

    hue_hist = cv2.calcHist([h], [0], None, [180], [0, 180])
    segmented_image = segment(h, threshold=40)

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")

    plt.subplot(2, 2, 2)
    plt.imshow(h, cmap="jet")
    plt.title("Hue Image")
    plt.colorbar()

    plt.subplot(2, 2, 3)
    plt.plot(hue_hist)
    plt.title("Hue Histogram")
    plt.xlabel("Hue Value")

    plt.subplot(2, 2, 4)
    plt.imshow(segmented_image, cmap="jet")
    plt.title("Segmented Image")
    plt.colorbar()
    plt.tight_layout()

    plt.figure(figsize=(12, 6))

    opening = cv2.morphologyEx(
        (segmented_image * 255).astype(np.uint8),
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8),
        iterations=5,
    )
    distance_transformed = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

    plt.subplot(2, 3, 1)
    plt.title("Distance Transform")
    plt.imshow(distance_transformed, cmap="jet")

    _, sure_fg = cv2.threshold(
        distance_transformed, 0.7 * distance_transformed.max(), 255, cv2.THRESH_BINARY
    )
    sure_fg = np.uint8(sure_fg)

    plt.subplot(2, 3, 2)
    plt.title("Sure Foreground")
    plt.imshow(sure_fg, cmap="gray")

    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    plt.subplot(2, 3, 3)
    plt.title("Unknown")
    plt.imshow(unknown, cmap="gray")
    plt.colorbar()

    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    plt.subplot(2, 3, 4)
    plt.title("Markers")
    plt.imshow(markers, cmap="jet")
    plt.colorbar()

    segmented_watershed = cv2.watershed(original, markers)
    watershed_border = np.uint8(segmented_watershed == -1)
    watershed_border = cv2.dilate(watershed_border, kernel, iterations=3)

    outlined = original.copy()
    outlined[watershed_border == 1] = [0, 255, 0]

    plt.subplot(2, 3, 5)
    plt.title("Watershed Border")
    plt.imshow(watershed_border, cmap="jet")
    plt.colorbar()

    plt.subplot(2, 3, 6)
    plt.title("Binary Image with Watershed")
    segmented_removed = cv2.subtract(segmented_image.astype(np.uint8), watershed_border)
    plt.imshow(segmented_removed, cmap="jet")
    plt.colorbar()
    plt.tight_layout()

    plt.figure(figsize=(12, 4))
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(
        segmented_removed.astype(np.uint8)
    )

    labels_norm = cv2.normalize(labels_im, None, 0, 255, cv2.NORM_MINMAX)
    labels_colored = cv2.applyColorMap(labels_norm.astype(np.uint8), cv2.COLORMAP_TURBO)

    plt.subplot(1, 3, 1)
    plt.title("Binary Image with Watershed")
    plt.imshow(segmented_removed, cmap="jet")
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.title("Region Ident.")
    plt.imshow(labels_colored)
    plt.colorbar()

    filtered = np.zeros_like(segmented_removed, dtype=np.uint8)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= 1000:
            filtered[labels_im == i] = 255

    plt.subplot(1, 3, 3)
    plt.title("Result - Binary Image")
    plt.imshow(filtered, cmap="jet")
    plt.colorbar()
    plt.tight_layout()

    plt.figure(figsize=(10, 5))
    sizes, diff, heatmap = granulometric_spectrum(
        filtered, min_size=40, max_size=65, step=1, return_heatmap=True
    )

    plt.subplot(1, 2, 1)
    plt.title("Result - Granulometry")
    plt.imshow(heatmap, cmap="jet")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title("Granul. Image Histogram")
    plt.plot(sizes, diff)
    plt.xlabel("Value")
    plt.ylabel("#")

    plt.tight_layout()

    
    plt.show()

    estimated_objects = diff / (sizes**2)
    estimated_objects[estimated_objects < 0.9] = 0
    estimated_objects = np.floor(estimated_objects)
    indices = np.nonzero(estimated_objects)[0]

    for index in indices:
        print(
            f"No. objects: {int(estimated_objects[index])} size: {sizes[index]} x {sizes[index]}"
        )


if __name__ == "__main__":
    main()
