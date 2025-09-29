import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dctn


def feature_gray_hist(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def feature_hue_hist(img):
    hue = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 0]
    hist = cv2.calcHist([hue], [0], None, [180], [0, 180])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def feature_dct(img, R=5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dctM = dctn(np.float32(gray))
    dctRvec = dctM[0:R, 0:R].flatten()
    return dctRvec / np.linalg.norm(dctRvec)


def compare_features(feature_func, title):
    features = [feature_func(img) for img in images]
    n = len(images)
    fig, axes = plt.subplots(n, n, figsize=(12, 12))
    fig.suptitle(f"Features - {title}")

    for i in range(n):
        dists = [np.linalg.norm(features[i] - features[j]) for j in range(n)]
        order = np.argsort(dists)

        for k, j in enumerate(order):
            ax = axes[i, k]
            ax.imshow(cv2.cvtColor(images[j], cv2.COLOR_BGR2RGB))

    plt.tight_layout()
    plt.show()


images = [cv2.imread(f"data/pvi_cv03_im{i:02d}.jpg") for i in range(1, 10)]

compare_features(feature_gray_hist, "Hist. Gray")
compare_features(feature_hue_hist, "Hist. Hue")
compare_features(feature_dct, "DCT 5x5")