import cv2
import numpy as np
from matplotlib import pyplot as plt


def show_fft2(img):
    fft2 = np.fft.fft2(img)
    fft2_shift = np.fft.fftshift(fft2)
    spectrum = np.log(1 + np.abs(fft2_shift))

    plt.figure(figsize=(6, 6))
    plt.imshow(spectrum, cmap="jet")
    plt.title("Spektrum")
    plt.colorbar()
    plt.show()


def main():
    img = cv2.imread("data/pvi_cv03_im09.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    show_fft2(gray)


if __name__ == '__main__':
    main()