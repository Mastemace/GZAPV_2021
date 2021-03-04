import cv2 as cv
import numpy as np
from glob import glob
import pandas
from tqdm import tqdm
from scipy.signal.windows import triang
import psutil
import math


def main():
    res = 1024
    factor = 3
    train_size = 200
    test_size = 150

    svm = train_folder(sorted(glob('./data/train/*.*'))[:train_size], res, factor)
    predict_folder(svm, sorted(glob('./data/test/*.*'))[:test_size], res, factor)


def separate(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 10, 10, minRadius=900)

    if circles is not None:
        circle = circles[0][0]
        circle[2] *= 1.05

        min_x = np.clip(int(circle[0] - circle[2]), 0, img.shape[1] - 1)
        max_x = np.clip(int(circle[0] + circle[2]), 0, img.shape[1] - 1)
        min_y = np.clip(int(circle[1] - circle[2]), 0, img.shape[0] - 1)
        max_y = np.clip(int(circle[1] + circle[2]), 0, img.shape[0] - 1)

        img = img[min_y:max_y, min_x:max_x, ...]

    return img


def resize(img, out_res):
    if img.shape[0] > img.shape[1]:
        dx = int(out_res / img.shape[0] * img.shape[1])
        img = cv.resize(img, (dx, out_res), interpolation=cv.INTER_LANCZOS4)
    else:
        dy = int(out_res / img.shape[1] * img.shape[0])
        img = cv.resize(img, (out_res, dy), interpolation=cv.INTER_LANCZOS4)

    sy = (out_res - img.shape[0]) // 2
    sx = (out_res - img.shape[1]) // 2

    out_img = np.zeros((out_res, out_res, 3), dtype=np.uint8)
    out_img[sy:img.shape[0] + sy, sx:img.shape[1] + sx, ...] = img

    return out_img


def equalize(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img[..., 0] = clahe.apply(img[..., 0])
    img = cv.cvtColor(img, cv.COLOR_LAB2BGR)

    return img


def ram(paths, res, train, ncols=10):
    tot = psutil.virtual_memory().total / 1024 / 1024
    exp = (psutil.virtual_memory().used + len(paths) * res * res * 4 * (2 if train else 1)) / 1024 / 1024
    diff = math.ceil(exp * ncols / tot)
    col = 2 if exp < tot / 2 else 3 if exp < tot else 1
    full = "▉" * min(diff, ncols)
    empty = " " * min(ncols - diff, ncols)

    print(f"\33[9{col}mExpected RAM usage: {exp / tot : 4.0%} |{full}{empty}| {exp : ,.0f}/{tot : ,.0f}MB\33[0m")

    if exp > tot:
        exit(-1)


def prepare(paths, res, factor, train):

    ram(paths, res, train)

    labels = pandas.read_csv('D:\\_Retina_Data\\trainLabels.csv')

    train_x = np.zeros((len(paths), res * res), dtype=np.float32)
    train_y = np.zeros(len(paths), dtype=np.int32)

    tr = triang(factor * 2 + 1).reshape(factor * 2 + 1, 1)
    kernel = np.dot(tr, tr.T)
    kernel /= np.sum(kernel)

    for i in tqdm(range(len(paths)), desc="Preparing images"):
        img = cv.imread(paths[i])
        img = separate(img)
        img = resize(img, res)
        img = equalize(img)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = cv.filter2D(img, -1, kernel)
        train_x[i, ...] = img.flatten() / 255.0
        train_y[i] = labels.loc[labels['image'] == paths[i].split("\\")[-1].split(".")[0]].iloc[0]['level']

    train_y[train_y != 0] = -1
    train_y[train_y == 0] = 1

    return train_x, train_y


def train_folder(paths, res, factor):
    train_x, train_y = prepare(paths, res, factor, True)

    svm = cv.ml.SVM_create()
    svm.setType(cv.ml.SVM_C_SVC)
    svm.setKernel(cv.ml.SVM_LINEAR)
    svm.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6))

    print("Training", end="\t")
    svm.train(train_x, cv.ml.ROW_SAMPLE, train_y)
    print("DONE")

    return svm


def predict_folder(svm, paths, res, factor):
    predict_x, predict_y = prepare(paths, res, factor, False)

    predictions = np.zeros(len(paths), dtype=np.int32)

    for i in tqdm(range(predict_x.shape[0]), desc="Predicting"):
        predictions[i] = svm.predict(predict_x[i:i + 1, ...])[1]

    diff = predictions == predict_y
    print("Prediction success: {:d} ({:8.3%})".format(np.sum(diff), np.sum(diff) / predict_y.shape[0]))


if __name__ == "__main__":
    main()
