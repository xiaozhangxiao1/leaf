import math
import os

import cv2

try:  # matplotlib 在无图形环境下可能不可用
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - 在缺失 matplotlib 时禁用预览
    plt = None


def show_images(images, titles, figsize=(15, 10), cols=3):
    """显示多张图片"""
    if not images:
        return

    if plt is None:
        print("当前环境未安装 matplotlib，已跳过图像预览。")
        return

    cols = max(1, cols)
    rows = math.ceil(len(images) / cols)
    plt.figure(figsize=figsize)

    for i, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, cols, i + 1)
        if len(image.shape) == 2:  # 灰度图
            plt.imshow(image, cmap="gray")
        else:  # 彩色图
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def save_result(image, filename, folder='results'):
    """保存结果图片"""
    if not os.path.exists(folder):
        os.makedirs(folder)
    cv2.imwrite(os.path.join(folder, filename), image)
    print(f'已保存: {folder}/{filename}')
