import cv2
import numpy as np
import matplotlib.pyplot as plt

def amplitude_spectrum(image):
    fft2 = np.fft.fft2(image)
    fft2_shift = np.fft.fftshift(fft2)
    spectrum = np.log(1 + np.abs(fft2_shift))
    return spectrum

def my_median_filter(image, mask_size=5):
    pad = mask_size // 2
    padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    result = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+mask_size, j:j+mask_size]
            result[i, j] = np.median(region)
    return result

img = cv2.imread('data/pvi_cv04.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- Filtrování šumu ---
averIm = cv2.blur(gray, (3, 3))
medIm = cv2.medianBlur(gray, 5)
scuffedIm = my_median_filter(gray)


def show_results(title, original, new):
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))

    # --- Original ---
    axs[0, 0].imshow(original, cmap='gray')
    axs[0, 0].set_title("Originál")
    axs[0, 0].axis('off')

    axs[0, 1].hist(original.ravel(), bins=256, range=(0, 256))
    axs[0, 1].set_title("Histogram (originál)")

    axs[0, 2].imshow(amplitude_spectrum(original), cmap='jet')
    axs[0, 2].set_title("Amplitudové spektrum (originál)")
    axs[0, 2].axis('off')

    # --- New (filtrovaný) ---
    axs[1, 0].imshow(new, cmap='gray')
    axs[1, 0].set_title(title)
    axs[1, 0].axis('off')

    axs[1, 1].hist(new.ravel(), bins=256, range=(0, 256))
    axs[1, 1].set_title("Histogram (filtrovaný)")

    axs[1, 2].imshow(amplitude_spectrum(new), cmap='jet')
    axs[1, 2].set_title("Amplitudové spektrum (filtrovaný)")
    axs[1, 2].axis('off')

    plt.tight_layout()
    plt.show()

show_results('Průměrovací filtr (3x3)', gray, averIm)
show_results('Mediánový filtr (5x5 - OpenCV)', gray, medIm)
show_results('Mediánový filtr (5x5 - Scuffed)', gray, my_median_filter(gray))