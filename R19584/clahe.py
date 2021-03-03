import cv2 as cv

def main():
    img = cv.imread("./data/test_clahe.jpg")

    img = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img[..., 0] = clahe.apply(img[..., 0])
    img = cv.cvtColor(img, cv.COLOR_LAB2BGR)

    cv.imwrite("./data/test_clahe_out.jpg", img)

if __name__ == "__main__":
    main()
