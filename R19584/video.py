import cv2 as cv
import numpy as np


def main():
    map_size = 5

    kernel = np.ones((map_size, map_size))
    kernel /= np.sum(kernel)

    handler = cv.VideoCapture('data/input.mp4')

    output_file = cv.VideoWriter(
        'data/output.mp4',
        int(handler.get(cv.CAP_PROP_FOURCC)),
        int(handler.get(cv.CAP_PROP_FPS)),
        (
            int(handler.get(cv.CAP_PROP_FRAME_WIDTH)),
            int(handler.get(cv.CAP_PROP_FRAME_HEIGHT))
        )
    )

    frame_count = int(handler.get(cv.CAP_PROP_FRAME_COUNT))

    for i in range(frame_count):
        print("Processing frame: {:3d}/{} {:7.2%}".format(i + 1, frame_count, (i + 1) / frame_count))

        _, frame = handler.read()

        output_file.write(cv.fastNlMeansDenoisingColored(frame, map_size))

    handler.release()
    output_file.release()


if __name__ == "__main__":
    main()
