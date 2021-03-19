import argparse
import numpy as np
import cv2 as cv

threshold = 0.5770975056689343


def find_contour_proportions(filename: str) -> float:
    image = cv.imread(filename)
    raw_image = image.copy()

    bw_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred_image = cv.GaussianBlur(bw_image, (3, 3), 0)
    _, otsu_image = cv.threshold(blurred_image, 0, 255, cv.THRESH_OTSU)

    kernel = np.ones((8, 8), np.uint8)
    opened_image = cv.morphologyEx(otsu_image, cv.MORPH_OPEN, kernel)
    closed_image = cv.morphologyEx(opened_image, cv.MORPH_CLOSE, kernel)

    contours, _ = cv.findContours(closed_image, mode=cv.RETR_LIST, method=cv.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)
    contour = sorted_contours[2]

    contours_image = raw_image.copy()
    x, y, w, h = cv.boundingRect(contour)
    cv.rectangle(contours_image, (x, y), (x + w, y + h), (0, 255, 0), thickness=15)
    
    return w / h


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Big Lab', add_help=False)

    parser.add_argument('-i', action="store", dest="image", help="absolute path to the query image")

    parser.add_argument('-h', '--help', action="help", help="show this help message")

    args = parser.parse_args()

    ratio = find_contour_proportions(args.image)
    if ratio <= threshold:
        print("Yes, I think it will fit")
    else:
        print("No, I don't think it will fit")

