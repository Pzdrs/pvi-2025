import os

import cv2
import easyocr
import numpy as np
from matplotlib import pyplot as plt


def extract(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT.create()
    kp, des = sift.detectAndCompute(gray, None)

    return kp, des, gray


def show_matches_side_by_side(temp_gr, temp_kp, test_gr, test_kp, matches_mask, good):
    draw_params = dict(
        matchColor=(150, 255, 0),
        singlePointColor=None,
        matchesMask=matches_mask,
        flags=2,
    )
    img3 = cv2.drawMatches(
        temp_gr, temp_kp, test_gr, test_kp, good, None, **draw_params
    )

    plt.title("Detected matches")
    plt.imshow(img3, "gray")


def show_regions(photo, name, surname):
    plt.subplot(1, 3, 1)
    plt.imshow(photo, cmap="gray")

    plt.subplot(1, 3, 2)
    plt.imshow(name, cmap="gray")

    plt.subplot(1, 3, 3)
    plt.imshow(surname, cmap="gray")


def show_ocr(aligned, fname, lname):
    text1 = f"first_name: '{fname}'"
    text2 = f"last_name: '{lname}'"
    aligned = aligned.copy()

    plt.text(20, 20, text1, color="red")
    plt.text(20, 40, text2, color="red")

    plt.imshow(aligned, cmap="gray")


def main(plot=True):
    ref_img = cv2.imread("data/obcansky_prukaz_cr_sablona_2012_2014.png")
    ref_keypoints, ref_descriptors, ref_gr = extract(ref_img)

    for test in os.listdir("data/tests"):
        test_img = cv2.imread(os.path.join("data/tests", test))
        test_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        kp, des, gr = extract(test_img)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(ref_descriptors, des, k=2)

        hits = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                hits.append(m)

        assert len(hits) >= 5, "Not enough matches found!"

        src_pts = np.float32(
            [ref_keypoints[m.queryIdx].pt for m in hits]
        ).reshape(-1, 1, 2)
        dst_pts = np.float32([kp[m.trainIdx].pt for m in hits]).reshape(
            -1, 1, 2
        )

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()

        if plot:
            plt.figure()
            plt.subplot(1, 3, 1)
            show_matches_side_by_side(
                ref_gr, ref_keypoints, gr, kp, matches_mask, hits
            )

        h, w = ref_gr.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(
            -1, 1, 2
        )
        dst = cv2.perspectiveTransform(pts, M)

        detected = cv2.polylines(
            test_rgb.copy(), [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA
        )

        if plot:
            plt.subplot(1, 3, 2)
            plt.imshow(detected)

        aligned = cv2.warpPerspective(gr, np.linalg.inv(M), (w, h))

        if plot:
            plt.subplot(1, 3, 3)
            plt.imshow(aligned, cmap="gray")
       
        # equalize histogram to improve contrast 
        clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(4,4))
        aligned = clahe.apply(aligned)

        # ish regions nebudu to hrotit vic 
        photo_gr = aligned[66:202, 5:114]
        name_gr = aligned[50:70, 70:150]
        surname_gr = aligned[35:60, 70:150]

        if plot:
            plt.figure()
            plt.subplot(1, 3, 1)
            show_regions(photo_gr, name_gr, surname_gr)

        # OCR
        reader = easyocr.Reader(["en", "cs"])
        name_results = reader.readtext(name_gr)
        surname_results = reader.readtext(surname_gr)

        fname = name_results[0][1]
        lname = surname_results[0][1]
        
        print(f"{test} - {fname} {lname}")

        if plot:
            plt.figure()
            show_ocr(aligned, fname, lname)

            plt.show()


if __name__ == "__main__":
    main(plot=False)
