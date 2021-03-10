import cv2 as cv
from tqdm import tqdm


def main():
    front = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    profile = cv.CascadeClassifier('haarcascade_profileface.xml')

    handler = cv.VideoCapture('face.mp4')
    frame_count = int(handler.get(cv.CAP_PROP_FRAME_COUNT))
    frames = [handler.read()[1] for _ in tqdm(range(frame_count), desc="Loading frames")]

    face_detected_counter = 0

    for img in tqdm(frames, desc="Processing"):

        faces = front.detectMultiScale(img, scaleFactor=1.15, minNeighbors=6)

        if len(faces) == 0:
            faces = profile.detectMultiScale(img, scaleFactor=1.15, minNeighbors=6)

        if len(faces) > 0:

            face_detected_counter += 1

            for (x, y, w, h) in faces:
                cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

            cv.imshow('live', img)
            cv.waitKey(1)

    for _ in tqdm(range(frame_count), initial=face_detected_counter, bar_format='Face detected on {n_fmt} frames: {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt}'):
        break

    out_file = cv.VideoWriter('out.mp4', int(handler.get(6)), int(handler.get(5)), (int(handler.get(3)), int(handler.get(4))))

    cv.destroyAllWindows()

    for frame in tqdm(frames, desc="Saving"):
        out_file.write(frame)

    out_file.release()

    print("DONE!")


if __name__ == "__main__":
    main()
