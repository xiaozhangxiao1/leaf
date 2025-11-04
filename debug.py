import cv2
import numpy as np
from utils import show_images


def debug_white_background(image_path):
    """调试白色背景检测"""
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 尝试不同的白色范围
    ranges = [
        ([0, 0, 200], [180, 55, 255]),
        ([0, 0, 180], [180, 65, 255]),
        ([0, 0, 220], [180, 45, 255])
    ]

    masks = []
    titles = []

    for i, (lower, upper) in enumerate(ranges):
        lower_white = np.array(lower)
        upper_white = np.array(upper)
        mask = cv2.inRange(hsv, lower_white, upper_white)
        masks.append(mask)
        titles.append(f'Range {i + 1}: {lower}-{upper}')

    # 显示原图和所有掩码
    images = [img] + masks
    titles = ['Original'] + titles
    show_images(images, titles)


# 测试一张图片
debug_white_background('images/leaf1.jpg')