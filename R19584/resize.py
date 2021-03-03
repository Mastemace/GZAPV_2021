import cv2 as cv


def main(scale):
    img = cv.imread("./data/test_img_downscale.png")
    size_down = (img.shape[1] // scale, img.shape[0] // scale)
    size_up = (img.shape[1], img.shape[0])

    cv.imwrite("./data/test_img_downscale_out_linear.png", cv.resize(cv.resize(img, size_down, interpolation=cv.INTER_LINEAR), size_up, interpolation=cv.INTER_LINEAR))
    cv.imwrite("./data/test_img_downscale_out_nearest.png", cv.resize(cv.resize(img, size_down, interpolation=cv.INTER_NEAREST), size_up, interpolation=cv.INTER_NEAREST))
    cv.imwrite("./data/test_img_downscale_out_cubic.png", cv.resize(cv.resize(img, size_down, interpolation=cv.INTER_CUBIC), size_up, interpolation=cv.INTER_CUBIC))
    cv.imwrite("./data/test_img_downscale_out_lanczos4.png", cv.resize(cv.resize(img, size_down, interpolation=cv.INTER_LANCZOS4), size_up, interpolation=cv.INTER_LANCZOS4))


if __name__ == "__main__":
    main(5)
