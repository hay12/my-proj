import numpy as np
import cv2
from ex1_utils import MAX_INT_8, LOAD_GRAY_SCALE, LOAD_RGB

gamma_slider_max = 100
gamma_range = 2
title_window = 'Gamma Correction'


def on_trackbar(val):
    gamma_val = gamma_range * (val / gamma_slider_max)
    print("\rGamma {:.2f}".format(gamma_val), end='')
    gamma_img = np.power(img/MAX_INT_8 ,gamma_val) * MAX_INT_8
    gamma_img = gamma_img.astype(np.uint8)
    cv2.imshow(title_window, gamma_img)


def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    global img
    img = cv2.imread(img_path)
    if rep == LOAD_GRAY_SCALE:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.namedWindow(title_window)
    trackbar_name = 'gamma'
    cv2.createTrackbar(trackbar_name, title_window, gamma_slider_max // gamma_range, gamma_slider_max, on_trackbar)
    on_trackbar(gamma_slider_max // gamma_range)
    cv2.waitKey()


if __name__ == '__main__':
    gammaDisplay('beach.jpg', LOAD_RGB)
