import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

plt.close('all')

def crunch(filepath):
    image_data = cv2.imread(os.path.join(DATA_DIR, filepath))
    histogram = cv2.calcHist([image_data], [0], None, [256], [0, 256])

    return {
        "hist": histogram,
        "image": image_data
    }

DATA_DIR = "data"

references = ("cv01_auto.jpg","cv01_jablko.jpg","cv01_mesic.jpg")
samples = ("cv01_u01.jpg", "cv01_u02.jpg", "cv01_u03.jpg")
processed_samples = [crunch(sample) for sample in samples]

fig, axes = plt.subplots(2, 3)


for i, ref in enumerate(references):
    ref_data = crunch(ref)

    distances = np.abs([cv2.compareHist(ref_data["hist"], sample["hist"], cv2.HISTCMP_CORREL) for sample in processed_samples])
    axes[0, i].imshow(cv2.cvtColor(ref_data["image"], cv2.COLOR_BGR2RGB))
    axes[1, i].imshow(cv2.cvtColor(processed_samples[np.argmax(distances)]["image"], cv2.COLOR_BGR2RGB))

plt.show()