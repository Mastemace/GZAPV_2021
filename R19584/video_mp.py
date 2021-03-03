import cv2 as cv
import multiprocessing as mp
from time import time


def main(threads: int, size: int) -> None:

    mp_frames: list[tuple[int, list]] = []

    for i in range(frame_count):
        frames_segment: list = []  # 5: i-2, i-1, i, i+1, i+2

        last_ok: int = i

        for j in range(int(size / 2)):  # 5: 0, 1 -> i-1, i-2
            index = i - j - 1 if i - j - 1 in range(frame_count) else last_ok
            frames_segment.append(frames[index])
            last_ok = index

        last_ok = i

        for j in range(int((size + 1) / 2)):  # 5: 0, 1, 2 -> i, i+1, i+2
            index = i + j if i + j in range(frame_count) else last_ok
            frames_segment.append(frames[index])
            last_ok = index

        mp_frames.append((i, frames_segment))

    start_time_s = time()

    with mp.Pool(threads) as pool:
        mp_return = pool.starmap(mp_process, mp_frames, threads)

    t_delta = time() - start_time_s
    t_frame = t_delta / frame_count

    out_file = cv.VideoWriter('data/out_s_{}_t_{}.mp4'.format(size, threads),
                              int(handler.get(cv.CAP_PROP_FOURCC)), int(handler.get(cv.CAP_PROP_FPS)),
                              (int(handler.get(cv.CAP_PROP_FRAME_WIDTH)), int(handler.get(cv.CAP_PROP_FRAME_HEIGHT))))

    mp_return.sort(key=get_frame_id)

    for mp_task in mp_return:
        out_file.write(mp_task[1])

    out_file.release()

    print("s {} t {} t_delta: {} t_frame {}".format(size, threads, t_delta, t_frame))


def mp_process(index: int, data: list) -> tuple[int, list]:
    return index, cv.fastNlMeansDenoisingColoredMulti(data, int(len(data) / 2), len(data))


def get_frame_id(e: tuple[int, list]) -> int:
    return e[0]


if __name__ == "__main__":
    handler = cv.VideoCapture('data/input.mp4')
    frame_count = int(handler.get(cv.CAP_PROP_FRAME_COUNT))
    frames = []

    for _ in range(frame_count):
        frames.append(handler.read()[1])

    for s in [1, 3, 5, 7, 9, 11]:
        for t in range(12):
            main(t + 1, s)

    handler.release()
