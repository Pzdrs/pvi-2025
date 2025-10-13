import cv2
import numpy as np
import matplotlib.pyplot as plt

filenames = [f"data/pvi_cv04_im0{i}.png" for i in range(1, 7)]
plt.figure(figsize=(12, 6))

for i, file in enumerate(filenames, 1):
    im = cv2.imread(file)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    
    edges = cv2.Canny(gray, 100, 256)
    _, bin_original = cv2.threshold(gray, 128, 1, cv2.THRESH_BINARY)
    bin_edges = (edges > 0).astype(np.uint8)

    sum_orig = np.sum(bin_original)
    sum_edges = np.sum(bin_edges)

    plt.subplot(2, 6, i)
    plt.imshow(bin_original, cmap='jet')
    plt.title(f"No {i}: {sum_orig}")
    plt.axis('off')

    plt.subplot(2, 6, i + 6)
    plt.imshow(bin_edges, cmap='jet')
    plt.title(f"No {i}: {sum_edges}")
    plt.axis('off')

plt.tight_layout()
plt.show()