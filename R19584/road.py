import cv2 as cv
import numpy as np


if __name__ == "__main__":
    img = cv.imread('./data/input.jpg')
    for line in cv.HoughLines(cv.Canny(img, 150, 250), 1, np.pi / 180, 190):
        cv.line(
            img,
            (
                int(np.cos(line[0][1]) * line[0][0] - 2000 * np.sin(line[0][1])),
                int(np.sin(line[0][1]) * line[0][0] + 2000 * np.cos(line[0][1]))
            ),
            (
                int(np.cos(line[0][1]) * line[0][0] + 2000 * np.sin(line[0][1])),
                int(np.sin(line[0][1]) * line[0][0] - 2000 * np.cos(line[0][1]))
            ),
            (0, 0, 255),
            thickness=2
        )

    cv.imwrite('./data/output.jpg', img)
